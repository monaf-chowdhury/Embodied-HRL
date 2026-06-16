"""
specialist.py -- Lean per-skill QC-FQL learner.

Current research path:
  1. Load replay-labelled FrankaKitchen demonstrations.
  2. Train one visual low-level policy per task (flow BC + one-step Q actor).
  3. Evaluate skills alone and under a scripted task order.

No hierarchy, no teacher/student, no residual stack. Keep this file boring.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from demo_dataset import build_or_load_demo_dataset
from encoder import VisualEncoder
from env_wrapper import FrankaKitchenImageWrapper
from networks import FlowActor, TwinQ
from offline_algorithms import make_offline_algorithm
from utils import TaskSpec, build_frozen_text_embeddings


@dataclass
class OptionResult:
    """Outcome of rolling one skill (option) forward; consumed by the evaluators."""
    z_end: np.ndarray
    proprio_end: np.ndarray
    completion_end: np.ndarray
    chosen_task_completed: bool
    env_done: bool
    termination_reason: str
    env_reward_sum: float
    frames: List[np.ndarray]


class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.mean = np.zeros(dim, dtype=np.float32)
        self.std = np.ones(dim, dtype=np.float32)

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = (x.std(axis=0) + self.eps).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


@dataclass
class Skill:
    """Per-skill QC-FQL networks: flow BC + one-step actor (FlowActor) and a
    twin chunk critic with a target copy."""
    flow: FlowActor
    critic: TwinQ
    critic_target: TwinQ
    actor_opt: torch.optim.Optimizer
    critic_opt: torch.optim.Optimizer


class SkillAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        self.tasks = list(config.training.tasks_to_complete)
        self.n_tasks = len(self.tasks)
        self.spec = TaskSpec(self.tasks, device=self.device)
        text_embs, text_src = build_frozen_text_embeddings(self.tasks, device=self.device)
        self.spec.attach_text_embeddings(text_embs, text_src)
        print(f"  [TaskSpec] text embeddings: {self.spec.text_source}  (dim={self.spec.text_embedding_dim})")

        self.encoder = VisualEncoder(config.encoder, device=self.device)
        self.z_dim = config.encoder.raw_dim
        self.proprio_dim = config.worker.proprio_dim
        self.max_goal_dim = self.spec.max_goal_dim
        self.H_chunk = max(1, config.worker.action_chunk_len)
        self.env_action_dim = 9
        self.action_dim = self.env_action_dim * self.H_chunk
        self.policy_input_dim = self.z_dim + self.proprio_dim + 4 * self.max_goal_dim

        self.proprio_norm = RunningNorm(self.proprio_dim)
        self.demo_dataset = None

        hidden = config.specialist.hidden_dim
        layers = config.specialist.n_layers
        use_ln = bool(config.specialist.use_layernorm)
        dropout = float(config.specialist.dropout)
        self.skills: List[Skill] = []
        for _ in self.tasks:
            flow = FlowActor(
                self.policy_input_dim, self.action_dim, hidden, layers,
                flow_steps=config.specialist.flow_steps,
                use_layernorm=use_ln, dropout=dropout,
            ).to(self.device)
            critic = TwinQ(self.policy_input_dim, hidden, layers, self.action_dim,
                           use_layernorm=use_ln, dropout=dropout).to(self.device)
            critic_target = copy.deepcopy(critic).to(self.device).eval()
            for p in critic_target.parameters():
                p.requires_grad_(False)
            self.skills.append(
                Skill(
                    flow=flow,
                    critic=critic,
                    critic_target=critic_target,
                    actor_opt=torch.optim.Adam(flow.parameters(), lr=config.worker.actor_lr),
                    critic_opt=torch.optim.Adam(critic.parameters(), lr=config.worker.critic_lr),
                )
            )

        self.stage_a_task_success = np.zeros(self.n_tasks, dtype=np.float32)
        self.curriculum_task_order = list(range(self.n_tasks))
        self.total_env_steps = 0
        self.total_options = 0

    def normalize_proprio(self, p: np.ndarray) -> np.ndarray:
        return self.proprio_norm(p)

    def reset_optimizers(self, task_ids: Optional[List[int]] = None):
        """Re-create per-skill optimizers (fresh Adam moments).

        Checkpoints store network weights only, so after agent.load() the
        optimizers still hold first/second-moment estimates accumulated for
        the *discarded* weights. Call this after any load that is followed by
        further training (e.g. online rollback) so the first post-restore
        updates are not driven by stale momenta.
        """
        ids = list(range(self.n_tasks)) if task_ids is None else [int(k) for k in task_ids]
        for k in ids:
            skill = self.skills[k]
            skill.actor_opt = torch.optim.Adam(
                skill.flow.parameters(), lr=self.config.worker.actor_lr)
            skill.critic_opt = torch.optim.Adam(
                skill.critic.parameters(), lr=self.config.worker.critic_lr)

    def _input_arrays(self,
                      z: np.ndarray,
                      proprio: np.ndarray,
                      task_target: np.ndarray,
                      task_cur: np.ndarray,
                      task_mask: np.ndarray) -> np.ndarray:
        delta = (task_target - task_cur) * task_mask
        return np.concatenate([z, proprio, task_target, task_cur, delta, task_mask], axis=-1).astype(np.float32)

    def _input_from_batch(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        x = self._input_arrays(
            batch["z"].astype(np.float32),
            batch["proprio"].astype(np.float32),
            batch["task_target"].astype(np.float32),
            batch["task_cur"].astype(np.float32),
            batch["task_mask"].astype(np.float32),
        )
        return torch.from_numpy(x).to(self.device)

    def _input_next_from_batch(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        x = self._input_arrays(
            batch["z_next"].astype(np.float32),
            batch["proprio_next"].astype(np.float32),
            batch["task_target"].astype(np.float32),
            batch["task_cur_next"].astype(np.float32),
            batch["task_mask"].astype(np.float32),
        )
        return torch.from_numpy(x).to(self.device)

    def _input_from_state(self, z: np.ndarray, state: np.ndarray, task_id: int) -> torch.Tensor:
        p = self.normalize_proprio(state)
        tt = self.spec.padded_goal_for(task_id)
        tc = self.spec.padded_state_slice_for(state, task_id)
        tm = self.spec.padded_mask_for(task_id)
        x = self._input_arrays(
            z.astype(np.float32)[None, :],
            p.astype(np.float32)[None, :],
            tt[None, :],
            tc[None, :],
            tm[None, :],
        )
        return torch.from_numpy(x).to(self.device)

    def flow_bc_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Flow-matching BC update (trains only the velocity field)."""
        skill = self.skills[task_id]
        x = self._input_from_batch(batch)
        a = torch.from_numpy(batch["action"]).to(self.device).clamp(-0.999, 0.999)
        loss = skill.flow.bc_flow_loss(x, a)
        skill.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.flow.parameters(), 1.0)
        skill.actor_opt.step()
        return {"flow_bc_loss": float(loss.item())}

    @staticmethod
    def _soft_update(src: nn.Module, dst: nn.Module, tau: float):
        with torch.no_grad():
            for p, p_targ in zip(src.parameters(), dst.parameters()):
                p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

    def qc_fql_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """One QC-FQL update for skill `task_id`.

        Critic (Q-chunking, unbiased h-step backup):
            target = R_disc + gamma^nstep * (1 - done) * min_i Q_target,i(s', a')
            a' = onestep(s', noise),   R_disc = sum_t' gamma^t' r_{t+t'}
        Actor (Flow Q-Learning, joint flow BC + distillation + Q):
            L = bc_flow_loss + alpha * ||onestep(s,z) - flow_ode(s,z)||^2 - Q(s, onestep(s,z))
        """
        skill = self.skills[task_id]
        cfg = self.config.specialist
        gamma = float(self.config.worker.gamma)

        x = self._input_from_batch(batch)
        xn = self._input_next_from_batch(batch)
        a = torch.from_numpy(batch["action"]).to(self.device).clamp(-0.999, 0.999)
        r = torch.from_numpy(batch["reward"]).unsqueeze(1).to(self.device)
        d = torch.from_numpy(batch["done"]).unsqueeze(1).to(self.device)
        nstep = torch.from_numpy(batch["nstep"].astype(np.float32)).unsqueeze(1).to(self.device)

        # --- Critic: chunked h-step TD backup -------------------------------
        with torch.no_grad():
            noise = torch.randn(xn.shape[0], self.action_dim, device=self.device)
            next_a = skill.flow.onestep_action(xn, noise).clamp(-0.999, 0.999)
            tq1, tq2 = skill.critic_target(xn, next_a)
            next_q = torch.min(tq1, tq2)
            disc = torch.pow(torch.full_like(nstep, gamma), nstep)
            target_q = r + disc * (1.0 - d) * next_q
        q1, q2 = skill.critic(x, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        skill.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.critic.parameters(), 1.0)
        skill.critic_opt.step()

        # --- Actor: flow BC + one-step distillation + Q-maximization --------
        bc_flow_loss = skill.flow.bc_flow_loss(x, a)
        noise = torch.randn(x.shape[0], self.action_dim, device=self.device)
        with torch.no_grad():
            flow_target = skill.flow.flow_action(x, noise).clamp(-0.999, 0.999)
        onestep_a = skill.flow.onestep_action(x, noise)
        distill_loss = F.mse_loss(onestep_a, flow_target)
        q1_pi, q2_pi = skill.critic(x, onestep_a)
        q_pi = torch.min(q1_pi, q2_pi)
        if bool(cfg.fql_normalize_q):
            q_loss = -(q_pi / q_pi.abs().mean().detach().clamp(min=1e-6)).mean()
        else:
            q_loss = -q_pi.mean()
        actor_loss = bc_flow_loss + float(cfg.fql_alpha) * distill_loss + q_loss
        skill.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.flow.parameters(), 1.0)
        skill.actor_opt.step()

        self._soft_update(skill.critic, skill.critic_target, float(cfg.target_tau))

        return {
            "qc_critic_loss": float(critic_loss.item()),
            "qc_actor_loss": float(actor_loss.item()),
            "qc_bc_flow_loss": float(bc_flow_loss.item()),
            "qc_distill_loss": float(distill_loss.item()),
            "qc_q_loss": float(q_loss.item()),
            "qc_q_mean": float(q_pi.mean().item()),
            "qc_target_q_mean": float(target_q.mean().item()),
        }

    @torch.no_grad()
    def get_worker_chunk(self,
                         z: np.ndarray,
                         proprio: np.ndarray,
                         full_state: np.ndarray,
                         task_id: int,
                         deterministic: bool = True) -> np.ndarray:
        # A flow/one-step policy IS a map from a latent z ~ N(0, I) to an action;
        # there is no closed-form "mean", so deployment samples a latent like
        # FQL's sample_actions (zeroing the latent would evaluate one arbitrary,
        # never-targeted slice of the policy). Eval reproducibility is handled by
        # seeding the RNG around each evaluation, not by zeroing the latent.
        del full_state, deterministic
        x = self._input_from_state(z, proprio, task_id)
        skill = self.skills[task_id]
        cfg = self.config.specialist
        if cfg.offline_algo == "flow_bc":
            # No critic in flow-BC mode: integrate the BC flow ODE from a latent.
            noise = torch.randn(1, self.action_dim, device=self.device)
            action = skill.flow.flow_action(x, noise)
        else:
            n = max(1, int(cfg.best_of_n))
            noise = torch.randn(n, self.action_dim, device=self.device)
            if n > 1:
                # Best-of-N: sample N one-step candidates, pick the highest-Q one.
                xr = x.expand(n, -1)
                cand = skill.flow.onestep_action(xr, noise).clamp(-1.0, 1.0)
                q1, q2 = skill.critic(xr, cand)
                action = cand[torch.min(q1, q2).squeeze(-1).argmax()].unsqueeze(0)
            else:
                action = skill.flow.onestep_action(x, noise)
        return action.cpu().numpy().reshape(self.H_chunk, self.env_action_dim).clip(-1.0, 1.0)

    def _worker_step_reward(self,
                            spec_err_before: float,
                            spec_err_after: float,
                            action_step: np.ndarray,
                            completion_bit_flipped: bool,
                            task_id: int) -> float:
        cfg = self.config.worker
        eps = float(self.spec.epsilon(task_id))
        sigma = float(cfg.sigma)
        # Potential-based shaping: phi(s) = exp(-(e/eps)/sigma) in (0, 1].
        # Normalising by per-task eps makes sigma dimensionless and uniform
        # across tasks. The shaping term is gamma*phi(s') - phi(s), which
        # preserves the optimal policy (Ng et al. 1999).
        phi_before = float(np.exp(-(spec_err_before / eps) / sigma))
        phi_after  = float(np.exp(-(spec_err_after  / eps) / sigma))
        reward = cfg.progress_weight * (cfg.gamma * phi_after - phi_before)
        reward += cfg.completion_bonus * float(completion_bit_flipped)
        reward -= cfg.action_cost * float(np.sum(action_step ** 2))
        return float(reward)

    def execute_option(self,
                       env,
                       task_id: int,
                       start_img: np.ndarray,
                       start_state: np.ndarray,
                       start_z: np.ndarray,
                       completion: np.ndarray,
                       deterministic_worker: bool = True,
                       collect_frames: bool = False) -> OptionResult:
        chosen_name = self.spec.name(task_id)
        state = start_state.copy()
        z = start_z.copy()
        completion_mask = completion.copy()
        frames = [start_img.copy()] if collect_frames else []
        option_env_reward = 0.0
        env_done = False
        chosen_completed = False
        termination_reason = "budget"
        steps_taken = 0

        while steps_taken < self.config.manager.subgoal_horizon and not env_done and not chosen_completed:
            chunk = self.get_worker_chunk(z, state, state, task_id, deterministic=deterministic_worker)
            for h in range(self.H_chunk):
                if steps_taken >= self.config.manager.subgoal_horizon:
                    break
                action_step = chunk[h]
                next_img, env_reward, done_env, info = env.step(action_step)
                next_state = np.asarray(info["state"], dtype=np.float64)
                next_z = self.encoder.encode_numpy(next_img).squeeze()
                raw_completion = self.spec.completion_mask_from_names(info.get("tasks_completed_names", []))
                completion_mask = np.maximum(completion_mask, raw_completion)
                if chosen_name in info.get("tasks_completed_names", []):
                    chosen_completed = True
                    termination_reason = "completed"

                option_env_reward += float(env_reward)
                state = next_state
                z = next_z
                env_done = bool(done_env)
                steps_taken += 1
                self.total_env_steps += 1
                if collect_frames:
                    frames.append(next_img.copy())
                if env_done:
                    termination_reason = "env_done"
                    break
                if not chosen_completed and self.spec.is_close(state, task_id):
                    chosen_completed = True
                    termination_reason = "close_enough"
                    break

        self.total_options += 1
        return OptionResult(
            z_end=z.copy(),
            proprio_end=state.copy(),
            completion_end=completion_mask.copy(),
            chosen_task_completed=bool(chosen_completed),
            env_done=env_done,
            termination_reason=termination_reason,
            env_reward_sum=float(option_env_reward),
            frames=frames,
        )

    def save(self, path: str):
        torch.save(
            {
                "tasks": self.tasks,
                "proprio_mean": self.proprio_norm.mean,
                "proprio_std": self.proprio_norm.std,
                "stage_a_task_success": self.stage_a_task_success,
                "curriculum_task_order": self.curriculum_task_order,
                "skills": [
                    {
                        "flow": s.flow.state_dict(),
                        "critic": s.critic.state_dict(),
                        "critic_target": s.critic_target.state_dict(),
                    }
                    for s in self.skills
                ],
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        saved_tasks = list(ckpt.get("tasks", []))
        if saved_tasks and saved_tasks != self.tasks:
            raise ValueError(f"Checkpoint tasks {saved_tasks} do not match current tasks {self.tasks}.")
        self.proprio_norm.mean = np.asarray(ckpt["proprio_mean"], dtype=np.float32)
        self.proprio_norm.std = np.asarray(ckpt["proprio_std"], dtype=np.float32)
        self.stage_a_task_success = np.asarray(
            ckpt.get("stage_a_task_success", np.zeros(self.n_tasks, dtype=np.float32)),
            dtype=np.float32,
        )
        self.curriculum_task_order = list(ckpt.get("curriculum_task_order", list(range(self.n_tasks))))
        for skill, saved in zip(self.skills, ckpt["skills"]):
            skill.flow.load_state_dict(saved["flow"])
            skill.critic.load_state_dict(saved["critic"])
            if "critic_target" in saved:
                skill.critic_target.load_state_dict(saved["critic_target"])
            else:
                skill.critic_target.load_state_dict(saved["critic"])
            skill.actor_opt = torch.optim.Adam(skill.flow.parameters(), lr=self.config.worker.actor_lr)
            skill.critic_opt = torch.optim.Adam(skill.critic.parameters(), lr=self.config.worker.critic_lr)


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def train_offline_skills(agent: SkillAgent,
                         config: Config,
                         verbose: bool = True,
                         writer=None) -> Dict[str, float]:
    if verbose:
        print(f"  [Stage A] Building demo dataset from: {', '.join(config.warmup.dataset_ids)}")
    ds, stats = build_or_load_demo_dataset(agent, config, verbose=verbose)
    agent.demo_dataset = ds
    agent.proprio_norm.fit(ds.w_p)

    results: Dict[str, float] = dict(stats)
    for task_id, task_name in enumerate(agent.tasks):
        safe = _safe_name(task_name)
        n = int(ds.worker_task_counts[task_id])
        if verbose:
            print(f"  [Skill] {task_name}: {n:,} demo transitions")
        if n == 0:
            continue

        algo = make_offline_algorithm(
            config.specialist.offline_algo,
            agent,
            config,
            writer=writer,
            verbose=verbose,
        )
        algo_result = algo.train_task(ds, task_id, task_name)
        results.update(algo_result.metrics)
        if verbose:
            _print_skill_metrics(task_name, safe, config.specialist.offline_algo, algo_result.metrics)

    if verbose:
        print("  [Stage A] Worker labels by task:")
        for k, name in enumerate(agent.tasks):
            print(f"    {name:<14} {int(ds.worker_task_counts[k]):,}")
    return results


def _print_skill_metrics(task_name: str, safe: str, algo: str, metrics: Dict[str, float]):
    del task_name
    algo = algo.lower().replace("-", "_")
    if f"flow_bc_loss/{safe}_final" in metrics:
        print(f"    flow BC loss={metrics[f'flow_bc_loss/{safe}_final']:.4f}")
    if algo == "qc_fql" and f"qc_critic_loss/{safe}_final" in metrics:
        print(f"    QC-FQL critic={metrics[f'qc_critic_loss/{safe}_final']:.4f}  "
              f"distill={metrics[f'qc_distill_loss/{safe}_final']:.4f}  "
              f"q_mean={metrics[f'qc_q_mean/{safe}_final']:.3f}  "
              f"target_q={metrics[f'qc_target_q_mean/{safe}_final']:.3f}")
    if f"prefix_success/{safe}_best" in metrics:
        print(f"    prefix-val: success={metrics[f'prefix_success/{safe}_best']*100:.1f}%  "
              f"err={metrics[f'prefix_error/{safe}_best']:.4f}  "
              f"best_step={int(metrics[f'prefix_best_step/{safe}'])}")


__all__ = ["SkillAgent", "train_offline_skills", "OptionResult"]
