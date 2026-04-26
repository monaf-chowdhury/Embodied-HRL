"""
agent.py — SMGW (Semantic Manager, Grounded Worker) agent.

Binds together:
  * TaskSpec          — grounding layer (indices, goals, epsilons, text embs)
  * VisualEncoder     — frozen image features (context only)
  * SemanticManager   — discrete Q over task IDs with completion masking
  * GroundedWorker    — SAC actor/critic on task-space targets; optional chunks
  * ManagerBuffer     — option-level transitions
  * WorkerBuffer      — low-level or chunk-level transitions

Public API used by train.py / warmup.py:

  * agent.select_task(z, proprio, task_state, completion, deterministic)
        -> chosen_task_id
  * agent.execute_option(env, task_id, state, z)
        -> OptionResult (see dataclass)
  * agent.update_worker() -> dict of losses
  * agent.update_manager() -> dict of losses
  * agent.save(path) / agent.load(path)

  (Note: BC pretraining steps live in warmup.py, not on the agent.)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config
from utils import TaskSpec, build_frozen_text_embeddings
from encoder import VisualEncoder
from networks import SemanticManager, GroundedWorkerActor, GroundedWorkerCritic
from buffers import ManagerBuffer, WorkerBuffer


# =============================================================================
# Option execution result
# =============================================================================

@dataclass
class OptionResult:
    """Everything the training loop needs to log after one option completes."""
    # State at option start / end
    z_start: np.ndarray
    proprio_start: np.ndarray
    task_state_start: np.ndarray       # flattened per-task slices at start
    completion_start: np.ndarray
    z_end: np.ndarray
    proprio_end: np.ndarray
    task_state_end: np.ndarray
    completion_end: np.ndarray

    # Option-level outcomes
    chosen_task: int
    chosen_task_completed: bool        # did THE chosen task's bit flip?
    any_task_completed: bool           # did ANY new task's bit flip?
    new_completions: int               # count of bits that flipped during option
    steps_taken: int
    env_done: bool
    termination_reason: str            # "completed" | "close_enough" | "budget" | "env_done"

    # Aggregate
    option_return: float               # sum of dense shaping + completion bonuses
    env_reward_sum: float              # raw env reward summed over option
    task_error_reduction: float        # Δ error on chosen task
    frames: List[np.ndarray] = field(default_factory=list)  # optional for video
    last_worker_losses: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Utility: build the manager's flattened task_state input
# =============================================================================

def build_task_state_flat(spec: TaskSpec, full_state: np.ndarray) -> np.ndarray:
    """Concat padded task-slices for ALL tasks -> (n_tasks * max_goal_dim,)"""
    parts = [spec.padded_state_slice_for(full_state, k) for k in range(spec.n_tasks)]
    return np.concatenate(parts, axis=0).astype(np.float32)


# =============================================================================
# SMGW Agent
# =============================================================================

class SMGWAgent:

    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        self.tasks: List[str] = list(config.training.tasks_to_complete)
        self.n_tasks = len(self.tasks)

        # ---- Task spec (grounding) ----
        self.spec = TaskSpec(self.tasks, device=self.device)
        text_embs, text_src = build_frozen_text_embeddings(self.tasks, device=self.device)
        self.spec.attach_text_embeddings(text_embs, text_src)
        print(f"  [TaskSpec] text embeddings: {self.spec.text_source}  "
              f"(dim={self.spec.text_embedding_dim})")

        # ---- Visual encoder (frozen) ----
        self.encoder = VisualEncoder(config.encoder, device=self.device)

        # ---- Dimensions ----
        z_dim = config.encoder.raw_dim
        proprio_dim = config.worker.proprio_dim
        max_goal_dim = self.spec.max_goal_dim
        text_dim = self.spec.text_embedding_dim
        action_dim = 9                  # Franka Kitchen action = 9-D
        self.action_dim = action_dim
        self.H_chunk = max(1, config.worker.action_chunk_len)

        # ---- Semantic manager ----
        self.manager = SemanticManager(
            z_dim=z_dim, proprio_dim=proprio_dim, n_tasks=self.n_tasks,
            max_goal_dim=max_goal_dim, text_embed_dim=text_dim,
            hidden_dim=config.manager.hidden_dim,
            n_layers=config.manager.n_layers,
        ).to(self.device)
        self.manager_target = SemanticManager(
            z_dim=z_dim, proprio_dim=proprio_dim, n_tasks=self.n_tasks,
            max_goal_dim=max_goal_dim, text_embed_dim=text_dim,
            hidden_dim=config.manager.hidden_dim,
            n_layers=config.manager.n_layers,
        ).to(self.device)
        self.manager_target.load_state_dict(self.manager.state_dict())
        self.manager_opt = torch.optim.Adam(self.manager.parameters(),
                                            lr=config.manager.lr)

        # ---- Grounded worker ----
        self.worker_actor = GroundedWorkerActor(
            z_dim=z_dim, proprio_dim=proprio_dim, action_dim=action_dim,
            max_goal_dim=max_goal_dim, text_embed_dim=text_dim,
            hidden_dim=config.worker.hidden_dim,
            n_layers=config.worker.n_layers,
            action_chunk_len=self.H_chunk,
        ).to(self.device)

        self.worker_critic = GroundedWorkerCritic(
            z_dim=z_dim, proprio_dim=proprio_dim, action_dim=action_dim,
            max_goal_dim=max_goal_dim, text_embed_dim=text_dim,
            hidden_dim=config.worker.hidden_dim,
            n_layers=config.worker.n_layers,
            action_chunk_len=self.H_chunk,
        ).to(self.device)
        self.worker_critic_target = GroundedWorkerCritic(
            z_dim=z_dim, proprio_dim=proprio_dim, action_dim=action_dim,
            max_goal_dim=max_goal_dim, text_embed_dim=text_dim,
            hidden_dim=config.worker.hidden_dim,
            n_layers=config.worker.n_layers,
            action_chunk_len=self.H_chunk,
        ).to(self.device)
        self.worker_critic_target.load_state_dict(self.worker_critic.state_dict())

        self.worker_actor_opt = torch.optim.Adam(self.worker_actor.parameters(),
                                                 lr=config.worker.actor_lr)
        self.worker_critic_opt = torch.optim.Adam(self.worker_critic.parameters(),
                                                  lr=config.worker.critic_lr)

        # SAC entropy temperature
        if config.worker.auto_alpha:
            self.log_alpha = torch.tensor(np.log(config.worker.init_alpha),
                                          requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha],
                                              lr=config.worker.alpha_lr)
            # Target entropy scales with number of squashed-Gaussian dims
            self.target_entropy = -float(action_dim * self.H_chunk)
        else:
            self.log_alpha = torch.tensor(np.log(config.worker.init_alpha),
                                          device=self.device)

        # ---- Buffers ----
        z_dtype = np.float16 if config.buffer.z_storage_dtype == 'float16' else np.float32
        self.manager_buf = ManagerBuffer(
            capacity=config.buffer.manager_capacity,
            z_dim=z_dim, proprio_dim=proprio_dim,
            n_tasks=self.n_tasks, max_goal_dim=max_goal_dim, z_dtype=z_dtype,
        )
        self.worker_buf = WorkerBuffer(
            capacity=config.buffer.worker_capacity,
            z_dim=z_dim, proprio_dim=proprio_dim,
            action_dim=action_dim, action_chunk_len=self.H_chunk,
            max_goal_dim=max_goal_dim, n_tasks=self.n_tasks, z_dtype=z_dtype,
        )

        # Counters
        self.total_env_steps = 0
        self.total_options = 0
        self.total_episodes = 0
        self.epsilon = config.manager.epsilon_start

    # =========================================================================
    # Epsilon schedule (manager ε-greedy)
    # =========================================================================

    def _update_epsilon(self):
        cfg = self.config.manager
        frac = self.total_env_steps / max(cfg.epsilon_decay_steps, 1)
        self.epsilon = max(cfg.epsilon_end,
                           cfg.epsilon_start
                           - (cfg.epsilon_start - cfg.epsilon_end) * frac)

    # =========================================================================
    # Manager task selection
    # =========================================================================

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def select_task(self,
                    z: np.ndarray,
                    proprio: np.ndarray,
                    full_state: np.ndarray,
                    completion: np.ndarray,
                    deterministic: bool = False) -> int:
        """
        Pick the next task given current observation and completion mask.
        completion[k] == 1 ⇒ task k is DISALLOWED (not in the argmax pool).
        If all tasks already completed, returns 0 (caller should check).

        Exploration: ε-greedy over the UNMASKED tasks only.
        """
        remaining = np.where(completion < 0.5)[0]
        if remaining.size == 0:
            return 0                                    # nothing left
        if (not deterministic) and np.random.random() < self.epsilon:
            return int(np.random.choice(remaining))

        task_state_flat = build_task_state_flat(self.spec, full_state)
        z_t = torch.from_numpy(z).float().unsqueeze(0).to(self.device)
        # We feed raw (unnormalised) proprio to the manager; it has a
        # relatively low-d input and a LayerNorm inside the torso.
        p_t = torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(self.device)
        ts_t = torch.from_numpy(task_state_flat).unsqueeze(0).to(self.device)
        c_t = torch.from_numpy(completion.astype(np.float32)).unsqueeze(0).to(self.device)

        q = self.manager.q_masked(z_t, p_t, ts_t, c_t, self.spec.text_embeddings)
        return int(q.argmax(dim=1).item())

    # =========================================================================
    # Worker action query
    # =========================================================================

    @torch.no_grad()
    def get_worker_chunk(self,
                         z: np.ndarray,
                         proprio: np.ndarray,
                         full_state: np.ndarray,
                         task_id: int,
                         deterministic: bool = False) -> np.ndarray:
        """
        Returns a (H_chunk, action_dim) chunk of actions to execute.
        For single-step mode H_chunk == 1 and this is a (1, A) array.
        """
        target = self.spec.padded_goal_for(task_id)
        cur = self.spec.padded_state_slice_for(full_state, task_id)
        mask = self.spec.padded_mask_for(task_id)
        embed = self.spec.text_embeddings[task_id].unsqueeze(0)

        z_t = torch.from_numpy(z).float().unsqueeze(0).to(self.device)
        p = self.worker_buf.normalize_proprio(proprio)
        p_t = torch.from_numpy(p.astype(np.float32)).unsqueeze(0).to(self.device)
        tt = torch.from_numpy(target).unsqueeze(0).to(self.device)
        tc = torch.from_numpy(cur).unsqueeze(0).to(self.device)
        tm = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        if deterministic:
            a = self.worker_actor.get_action_deterministic(z_t, p_t, tt, tc, tm, embed)
        else:
            a, _ = self.worker_actor(z_t, p_t, tt, tc, tm, embed)

        chunk = self.worker_actor.action_to_chunk(a).cpu().numpy().squeeze(0)
        # Clip for numerical safety
        return np.clip(chunk, -1.0, 1.0)

    # =========================================================================
    # Reward computation — ALL grounded in task-state and completion bits
    # =========================================================================

    def _worker_step_reward(self,
                            spec_err_before: float,
                            spec_err_after: float,
                            action_flat: np.ndarray,
                            completion_bit_flipped: bool) -> float:
        cfg = self.config.worker
        progress = spec_err_before - spec_err_after       # positive = closer
        r = cfg.progress_weight * progress
        r += cfg.completion_bonus * (1.0 if completion_bit_flipped else 0.0)
        r -= cfg.action_cost * float(np.sum(action_flat ** 2))
        return float(r)

    # =========================================================================
    # Option execution — runs a single option to completion
    # =========================================================================

    def execute_option(self,
                       env,
                       task_id: int,
                       start_img: np.ndarray,
                       start_state: np.ndarray,
                       start_z: np.ndarray,
                       completion: np.ndarray,
                       deterministic_worker: bool = False,
                       collect_frames: bool = False,
                       train_worker_online: bool = True,
                       update_every_n_env_steps: int = 2) -> OptionResult:
        """
        Run the worker under the chosen `task_id` until one of:
          (a) the env flips the chosen task's completion bit,
          (b) the chosen task's task-space error drops below ε_k,
          (c) K = subgoal_horizon env steps have elapsed,
          (d) the env returns done.

        Transitions written: ONE worker-buffer transition per queried chunk.
        The transition stores (obs_before_chunk, chunk_actions, summed_reward,
        obs_after_executed_portion, done). If the chunk terminates early
        (env done or task success mid-chunk), we still write the transition
        — the reward is just the sum over whatever was actually executed,
        and done is set.

        Hindsight: if a DIFFERENT task's completion bit flipped during this
        option, we also write a relabelled chunk transition with that task
        as the target (task-space HER).
        """
        cfg = self.config
        task_state_start_flat = build_task_state_flat(self.spec, start_state)
        chosen_name = self.spec.name(task_id)
        err_start_option = self.spec.task_error(start_state, task_id)

        state = start_state.copy()
        z = start_z.copy()
        proprio = state.copy()

        steps_taken = 0
        env_done = False
        chosen_completed = False
        new_completion_names: List[str] = []

        frames = [start_img.copy()] if collect_frames else []
        option_env_reward = 0.0

        K = cfg.manager.subgoal_horizon
        H = self.H_chunk
        eps_k = self.spec.epsilon(task_id)
        termination_reason = "budget"
        last_worker_losses: Dict[str, float] = {}

        # Track which tasks were already done BEFORE this option started,
        # so we can detect NEW completions during the option.
        already_completed = set(
            n for k_, n in enumerate(self.tasks) if completion[k_] > 0.5
        )

        while steps_taken < K and not env_done and not chosen_completed:
            # --- Query a fresh chunk ---
            chunk = self.get_worker_chunk(z, proprio, state, task_id,
                                          deterministic=deterministic_worker)
            action_flat = chunk.reshape(-1)           # (H*A,)
            z_chunk_start = z.copy()
            state_chunk_start = state.copy()
            proprio_chunk_start = proprio.copy()

            chunk_reward = 0.0
            chunk_newly_completed_other = []
            chunk_done = False

            for h in range(H):
                if steps_taken >= K:
                    break
                a_step = chunk[h]
                next_img, env_r, done_env, info = env.step(a_step)
                next_state = np.asarray(info['state'], dtype=np.float64)
                next_z = self.encoder.encode_numpy(next_img).squeeze()
                completion_names_now = info.get('tasks_completed_names', [])

                # Newly completed (for this chunk / option)
                just_completed = [n for n in completion_names_now
                                  if n not in already_completed]
                if just_completed:
                    for n in just_completed:
                        if n == chosen_name:
                            chosen_completed = True
                        if n not in new_completion_names:
                            new_completion_names.append(n)
                        if n != chosen_name and n not in chunk_newly_completed_other:
                            chunk_newly_completed_other.append(n)
                    already_completed.update(just_completed)

                err_before = self.spec.task_error(state, task_id)
                err_after = self.spec.task_error(next_state, task_id)
                chunk_reward += self._worker_step_reward(
                    err_before, err_after, a_step,
                    completion_bit_flipped=(chosen_name in just_completed),
                )
                option_env_reward += env_r

                if collect_frames:
                    frames.append(next_img)

                # Advance
                state, proprio, z = next_state, next_state, next_z
                steps_taken += 1
                self.total_env_steps += 1

                # Online SAC update, spaced out
                if (train_worker_online
                        and len(self.worker_buf) > cfg.buffer.batch_size
                        and self.total_env_steps % update_every_n_env_steps == 0):
                    loss_dict = self.update_worker()
                    if loss_dict:
                        last_worker_losses = loss_dict

                if done_env:
                    env_done = True
                    chunk_done = True
                    termination_reason = "env_done"
                    break
                if chosen_completed:
                    chunk_done = True
                    termination_reason = "completed"
                    break
                if self.spec.task_error(state, task_id) < eps_k:
                    chunk_done = True
                    termination_reason = "close_enough"
                    break

            # --- Write chunk transition ---
            target_pad = self.spec.padded_goal_for(task_id)
            mask_pad = self.spec.padded_mask_for(task_id)
            cur_pad = self.spec.padded_state_slice_for(state_chunk_start, task_id)
            cur_pad_next = self.spec.padded_state_slice_for(state, task_id)
            self.worker_buf.add(
                z=z_chunk_start,
                proprio=proprio_chunk_start,
                task_target=target_pad, task_cur=cur_pad, task_mask=mask_pad,
                task_id=task_id, action_flat=action_flat,
                reward=chunk_reward,
                z_next=z, proprio_next=proprio,
                task_cur_next=cur_pad_next,
                done=float(chunk_done),
            )

            # --- Task-space HER: relabel for any OTHER task that flipped ---
            for other_name in chunk_newly_completed_other:
                other_k = self.tasks.index(other_name)
                other_target = self.spec.padded_goal_for(other_k)
                other_mask = self.spec.padded_mask_for(other_k)
                other_cur = self.spec.padded_state_slice_for(state_chunk_start, other_k)
                other_cur_next = self.spec.padded_state_slice_for(state, other_k)
                other_r = (self.config.worker.progress_weight
                           * (self.spec.task_error(state_chunk_start, other_k)
                              - self.spec.task_error(state, other_k))
                           + self.config.worker.completion_bonus
                           - self.config.worker.action_cost
                           * float(np.sum(action_flat ** 2)))
                self.worker_buf.add(
                    z=z_chunk_start,
                    proprio=proprio_chunk_start,
                    task_target=other_target, task_cur=other_cur,
                    task_mask=other_mask, task_id=other_k,
                    action_flat=action_flat, reward=float(other_r),
                    z_next=z, proprio_next=proprio,
                    task_cur_next=other_cur_next,
                    done=1.0,                  # the relabelled task finished
                )

            if chunk_done:
                break

        # --- End of option bookkeeping ---
        err_end_option = self.spec.task_error(state, task_id)
        task_state_end_flat = build_task_state_flat(self.spec, state)

        completion_end = completion.copy()
        for k_, n in enumerate(self.tasks):
            if n in already_completed:
                completion_end[k_] = 1.0
        new_complete_count = int(round(completion_end.sum() - completion.sum()))

        # Option-level return for the manager — grounded in completion bits
        option_return = (self.config.manager.completion_bonus * new_complete_count
                         + self.config.manager.dense_shaping_weight
                         * (err_start_option - err_end_option)
                         - self.config.manager.option_cost)
        if not chosen_completed and new_complete_count == 0:
            option_return -= self.config.worker.failure_penalty
        if completion_end.sum() >= self.n_tasks:
            option_return += self.config.manager.all_done_bonus

        return OptionResult(
            z_start=start_z, proprio_start=start_state,
            task_state_start=task_state_start_flat, completion_start=completion,
            z_end=z, proprio_end=state,
            task_state_end=task_state_end_flat, completion_end=completion_end,
            chosen_task=task_id,
            chosen_task_completed=chosen_completed,
            any_task_completed=(new_complete_count > 0),
            new_completions=new_complete_count,
            steps_taken=steps_taken, env_done=env_done,
            termination_reason=termination_reason,
            option_return=float(option_return),
            env_reward_sum=float(option_env_reward),
            task_error_reduction=float(err_start_option - err_end_option),
            frames=frames,
            last_worker_losses=last_worker_losses,
        )

    # =========================================================================
    # Updates — worker SAC
    # =========================================================================

    def update_worker(self) -> Dict[str, float]:
        if len(self.worker_buf) < self.config.buffer.batch_size:
            return {}
        b = self.worker_buf.sample(self.config.buffer.batch_size)
        z = torch.from_numpy(b['z']).to(self.device)
        p = torch.from_numpy(b['proprio']).to(self.device)
        tt = torch.from_numpy(b['task_target']).to(self.device)
        tc = torch.from_numpy(b['task_cur']).to(self.device)
        tm = torch.from_numpy(b['task_mask']).to(self.device)
        tid = torch.from_numpy(b['task_id']).long().to(self.device)
        a = torch.from_numpy(b['action']).to(self.device)
        r = torch.from_numpy(b['reward']).unsqueeze(1).to(self.device)
        zn = torch.from_numpy(b['z_next']).to(self.device)
        pn = torch.from_numpy(b['proprio_next']).to(self.device)
        tcn = torch.from_numpy(b['task_cur_next']).to(self.device)
        d = torch.from_numpy(b['done']).unsqueeze(1).to(self.device)

        te = self.spec.text_embeddings[tid]       # (B, d_text)

        # ---- Critic update ----
        with torch.no_grad():
            next_a, next_logp = self.worker_actor(zn, pn, tt, tcn, tm, te)
            q1n, q2n = self.worker_critic_target(zn, pn, tt, tcn, tm, te, next_a)
            q_next = torch.min(q1n, q2n) - self.alpha * next_logp
            # For chunked worker the TD is effectively n-step already because
            # the reward is chunk-summed and gamma^H discounts next-state value.
            gamma_eff = self.config.worker.gamma ** self.H_chunk
            target_q = r + gamma_eff * (1.0 - d) * q_next

        q1, q2 = self.worker_critic(z, p, tt, tc, tm, te, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.worker_critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(), 1.0)
        self.worker_critic_opt.step()

        # ---- Actor update ----
        new_a, logp = self.worker_actor(z, p, tt, tc, tm, te)
        q1a, q2a = self.worker_critic(z, p, tt, tc, tm, te, new_a)
        actor_loss = (self.alpha * logp - torch.min(q1a, q2a)).mean()
        self.worker_actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_actor.parameters(), 1.0)
        self.worker_actor_opt.step()

        # ---- Alpha update ----
        if self.config.worker.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.log_alpha.data.clamp_(min=np.log(self.config.worker.min_alpha))

        # ---- Soft update critic target ----
        self._soft_update(self.worker_critic, self.worker_critic_target,
                          self.config.worker.tau)

        return {
            'worker_critic_loss': float(critic_loss.item()),
            'worker_actor_loss': float(actor_loss.item()),
            'worker_alpha': float(self.alpha.item()),
            'worker_mean_logp': float(logp.mean().item()),
        }

    # =========================================================================
    # Updates — manager DQN
    # =========================================================================

    def update_manager(self) -> Dict[str, float]:
        if len(self.manager_buf) < self.config.buffer.batch_size:
            return {}
        b = self.manager_buf.sample(self.config.buffer.batch_size)
        z = torch.from_numpy(b['z']).to(self.device)
        p = torch.from_numpy(b['proprio']).float().to(self.device)
        ts = torch.from_numpy(b['task_state']).to(self.device)
        c = torch.from_numpy(b['completion']).to(self.device)
        act = torch.from_numpy(b['action']).long().to(self.device)
        r = torch.from_numpy(b['reward']).unsqueeze(1).to(self.device)
        zn = torch.from_numpy(b['z_next']).to(self.device)
        pn = torch.from_numpy(b['proprio_next']).float().to(self.device)
        tsn = torch.from_numpy(b['task_state_next']).to(self.device)
        cn = torch.from_numpy(b['completion_next']).to(self.device)
        d = torch.from_numpy(b['done']).unsqueeze(1).to(self.device)

        # Q(s, a) where a is the task_id we actually took
        q_all = self.manager(z, p, ts, c, self.spec.text_embeddings)
        q_sa = q_all.gather(1, act.unsqueeze(1))

        # Bootstrap: max over NOT-YET-COMPLETED tasks at the next state
        with torch.no_grad():
            q_next = self.manager_target.q_masked(zn, pn, tsn, cn,
                                                   self.spec.text_embeddings)
            # If all tasks complete at next state, mask is -inf; treat value as 0.
            all_done = (cn.sum(dim=-1, keepdim=True) >= self.n_tasks).float()
            q_next_max = q_next.max(dim=1, keepdim=True).values
            q_next_max = torch.where(all_done > 0.5,
                                     torch.zeros_like(q_next_max), q_next_max)
            target = r + self.config.manager.gamma * (1 - d) * q_next_max

        loss = F.mse_loss(q_sa, target)
        self.manager_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 1.0)
        self.manager_opt.step()

        self._soft_update(self.manager, self.manager_target,
                          self.config.manager.tau)

        return {
            'manager_loss': float(loss.item()),
            'manager_q_mean': float(q_all.mean().item()),
            'manager_q_sa_mean': float(q_sa.mean().item()),
        }

    # =========================================================================
    # Soft update
    # =========================================================================

    def _soft_update(self, source, target, tau):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str):
        torch.save({
            'manager': self.manager.state_dict(),
            'manager_target': self.manager_target.state_dict(),
            'worker_actor': self.worker_actor.state_dict(),
            'worker_critic': self.worker_critic.state_dict(),
            'worker_critic_target': self.worker_critic_target.state_dict(),
            'log_alpha': float(self.log_alpha.detach().cpu().item()),
            'total_env_steps': self.total_env_steps,
            'total_options': self.total_options,
            'total_episodes': self.total_episodes,
            'epsilon': self.epsilon,
            'proprio_mean': self.worker_buf.proprio_stats.mean,
            'proprio_M2': self.worker_buf.proprio_stats.M2,
            'proprio_n': self.worker_buf.proprio_stats.n,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.manager.load_state_dict(ckpt['manager'])
        self.manager_target.load_state_dict(ckpt['manager_target'])
        self.worker_actor.load_state_dict(ckpt['worker_actor'])
        self.worker_critic.load_state_dict(ckpt['worker_critic'])
        self.worker_critic_target.load_state_dict(ckpt['worker_critic_target'])
        if self.config.worker.auto_alpha:
            self.log_alpha.data = torch.tensor(ckpt['log_alpha'],
                                               device=self.device)
        self.total_env_steps = ckpt.get('total_env_steps', 0)
        self.total_options = ckpt.get('total_options', 0)
        self.total_episodes = ckpt.get('total_episodes', 0)
        self.epsilon = ckpt.get('epsilon', self.config.manager.epsilon_end)
        self.worker_buf.proprio_stats.mean = ckpt['proprio_mean']
        self.worker_buf.proprio_stats.M2 = ckpt['proprio_M2']
        self.worker_buf.proprio_stats.n = ckpt['proprio_n']