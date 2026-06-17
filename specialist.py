"""
specialist.py -- shared skill-conditioned QC-FQL learner.

Current research path:
  1. Load replay-labelled FrankaKitchen demonstrations.
  2. Train one shared visual, task-conditioned action-chunk policy.
  3. Compose requested tasks with a fixed next-incomplete predicate planner.

There are no isolated per-task policies in this branch. Task identity enters
through learned task-ID embeddings plus object-goal/current/mask conditioning.
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
from networks import FlowActor, TwinQ
from offline_algorithms import make_offline_algorithm
from utils import TaskSpec


@dataclass
class OptionResult:
    """Outcome of rolling one task-conditioned option forward."""
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
        if x.size == 0:
            return
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = (x.std(axis=0) + self.eps).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


class SkillAgent:
    """One shared task-conditioned actor/critic used for every skill."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        self.tasks = list(config.training.tasks_to_complete)
        self.n_tasks = len(self.tasks)
        self.spec = TaskSpec(self.tasks, device=self.device)
        print(
            "  [TaskSpec] conditioning: learned task-id embeddings "
            f"(dim={config.specialist.task_embedding_dim})"
        )

        self.encoder = VisualEncoder(config.encoder, device=self.device)
        self.z_dim = config.encoder.raw_dim
        self.proprio_dim = config.worker.proprio_dim
        self.max_goal_dim = self.spec.max_goal_dim
        self.H_chunk = max(1, config.worker.action_chunk_len)
        self.env_action_dim = 9
        self.action_dim = self.env_action_dim * self.H_chunk
        self.task_embedding_dim = int(config.specialist.task_embedding_dim)
        self.policy_input_dim = (
            self.z_dim
            + self.proprio_dim
            + self.task_embedding_dim
            + 4 * self.max_goal_dim
        )

        self.proprio_norm = RunningNorm(self.proprio_dim)
        self.demo_dataset = None

        hidden = config.specialist.hidden_dim
        layers = config.specialist.n_layers
        use_ln = bool(config.specialist.use_layernorm)
        dropout = float(config.specialist.dropout)

        self.actor_task_embed = nn.Embedding(self.n_tasks, self.task_embedding_dim).to(self.device)
        self.critic_task_embed = nn.Embedding(self.n_tasks, self.task_embedding_dim).to(self.device)
        self.critic_task_embed_target = copy.deepcopy(self.critic_task_embed).to(self.device).eval()
        for p in self.critic_task_embed_target.parameters():
            p.requires_grad_(False)

        self.flow = FlowActor(
            self.policy_input_dim,
            self.action_dim,
            hidden,
            layers,
            flow_steps=config.specialist.flow_steps,
            use_layernorm=use_ln,
            dropout=dropout,
        ).to(self.device)
        self.critic = TwinQ(
            self.policy_input_dim,
            hidden,
            layers,
            self.action_dim,
            use_layernorm=use_ln,
            dropout=dropout,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device).eval()
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(
            list(self.flow.parameters()) + list(self.actor_task_embed.parameters()),
            lr=config.worker.actor_lr,
        )
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.critic_task_embed.parameters()),
            lr=config.worker.critic_lr,
        )

        self.stage_a_task_success = np.zeros(self.n_tasks, dtype=np.float32)
        self.curriculum_task_order = list(range(self.n_tasks))
        self.total_env_steps = 0
        self.total_options = 0

    # ------------------------------------------------------------------
    # Conditioning helpers
    # ------------------------------------------------------------------

    def normalize_proprio(self, p: np.ndarray) -> np.ndarray:
        return self.proprio_norm(p)

    def reset_optimizers(self, task_ids: Optional[List[int]] = None):
        """Re-create optimizers after checkpoint rollback.

        `task_ids` is accepted for backward compatibility with the old per-skill
        online code; the shared model always resets both optimizers.
        """
        del task_ids
        self.actor_opt = torch.optim.Adam(
            list(self.flow.parameters()) + list(self.actor_task_embed.parameters()),
            lr=self.config.worker.actor_lr,
        )
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.critic_task_embed.parameters()),
            lr=self.config.worker.critic_lr,
        )

    def _embed_table(self, stream: str) -> nn.Embedding:
        if stream == "actor":
            return self.actor_task_embed
        if stream == "critic":
            return self.critic_task_embed
        if stream == "critic_target":
            return self.critic_task_embed_target
        raise ValueError(f"Unknown conditioning stream '{stream}'.")

    def _condition_tensor(
        self,
        z: np.ndarray,
        proprio: np.ndarray,
        task_target: np.ndarray,
        task_cur: np.ndarray,
        task_mask: np.ndarray,
        task_id: np.ndarray,
        stream: str,
    ) -> torch.Tensor:
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        p_t = torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
        tt_t = torch.as_tensor(task_target, dtype=torch.float32, device=self.device)
        tc_t = torch.as_tensor(task_cur, dtype=torch.float32, device=self.device)
        tm_t = torch.as_tensor(task_mask, dtype=torch.float32, device=self.device)
        tid_t = torch.as_tensor(task_id, dtype=torch.long, device=self.device).view(-1)
        emb = self._embed_table(stream)(tid_t)
        delta = (tt_t - tc_t) * tm_t
        return torch.cat([z_t, p_t, emb, tt_t, tc_t, delta, tm_t], dim=-1)

    def _input_from_batch(self, batch: Dict[str, np.ndarray], stream: str = "actor") -> torch.Tensor:
        return self._condition_tensor(
            batch["z"].astype(np.float32),
            batch["proprio"].astype(np.float32),
            batch["task_target"].astype(np.float32),
            batch["task_cur"].astype(np.float32),
            batch["task_mask"].astype(np.float32),
            batch["task_id"],
            stream=stream,
        )

    def _input_next_from_batch(self, batch: Dict[str, np.ndarray], stream: str = "actor") -> torch.Tensor:
        return self._condition_tensor(
            batch["z_next"].astype(np.float32),
            batch["proprio_next"].astype(np.float32),
            batch["task_target"].astype(np.float32),
            batch["task_cur_next"].astype(np.float32),
            batch["task_mask"].astype(np.float32),
            batch["task_id"],
            stream=stream,
        )

    def _input_from_state(
        self,
        z: np.ndarray,
        state: np.ndarray,
        task_id: int,
        stream: str = "actor",
    ) -> torch.Tensor:
        p = self.normalize_proprio(state)
        tt = self.spec.padded_goal_for(task_id)
        tc = self.spec.padded_state_slice_for(state, task_id)
        tm = self.spec.padded_mask_for(task_id)
        return self._condition_tensor(
            z.astype(np.float32)[None, :],
            p.astype(np.float32)[None, :],
            tt[None, :],
            tc[None, :],
            tm[None, :],
            np.asarray([task_id], dtype=np.int64),
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Offline/online optimization
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_update(src: nn.Module, dst: nn.Module, tau: float):
        with torch.no_grad():
            for p, p_targ in zip(src.parameters(), dst.parameters()):
                p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

    def flow_bc_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Shared flow-matching BC update."""
        x = self._input_from_batch(batch, stream="actor")
        a = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device).clamp(-0.999, 0.999)
        loss = self.flow.bc_flow_loss(x, a)
        self.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.flow.parameters()) + list(self.actor_task_embed.parameters()), 1.0)
        self.actor_opt.step()
        return {"flow_bc_loss": float(loss.item())}

    def qc_fql_step(
        self,
        batch: Dict[str, np.ndarray],
        critic_batch: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """One shared QC-FQL update on task-conditioned relabeled batches."""
        cfg = self.config.specialist
        gamma = float(self.config.worker.gamma)
        if critic_batch is None:
            critic_batch = batch

        # --- Critic: full-corpus task-relabeled h-step TD backup ------------
        xc = self._input_from_batch(critic_batch, stream="critic")
        xcn_actor = self._input_next_from_batch(critic_batch, stream="actor")
        xcn_critic_target = self._input_next_from_batch(critic_batch, stream="critic_target")
        ac = torch.as_tensor(critic_batch["action"], dtype=torch.float32, device=self.device).clamp(-0.999, 0.999)
        r = torch.as_tensor(critic_batch["reward"], dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.as_tensor(critic_batch["done"], dtype=torch.float32, device=self.device).unsqueeze(1)
        nstep = torch.as_tensor(critic_batch["nstep"], dtype=torch.float32, device=self.device).unsqueeze(1)
        with torch.no_grad():
            noise = torch.randn(xcn_actor.shape[0], self.action_dim, device=self.device)
            next_a = self.flow.onestep_action(xcn_actor, noise).clamp(-0.999, 0.999)
            tq1, tq2 = self.critic_target(xcn_critic_target, next_a)
            next_q = torch.min(tq1, tq2)
            disc = torch.pow(torch.full_like(nstep, gamma), nstep)
            target_q = r + disc * (1.0 - d) * next_q
        q1, q2 = self.critic(xc, ac)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.critic_task_embed.parameters()), 1.0)
        self.critic_opt.step()

        # --- Actor: flow BC + one-step distillation + Q maximization --------
        x_actor = self._input_from_batch(batch, stream="actor")
        a = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device).clamp(-0.999, 0.999)
        bc_flow_loss = self.flow.bc_flow_loss(x_actor, a)
        noise = torch.randn(x_actor.shape[0], self.action_dim, device=self.device)
        with torch.no_grad():
            flow_target = self.flow.flow_action(x_actor, noise).clamp(-0.999, 0.999)
        onestep_a = self.flow.onestep_action(x_actor, noise)
        distill_loss = F.mse_loss(onestep_a, flow_target)

        critic_params = list(self.critic.parameters()) + list(self.critic_task_embed.parameters())
        prev_requires_grad = [p.requires_grad for p in critic_params]
        for p in critic_params:
            p.requires_grad_(False)
        try:
            x_q = self._input_from_batch(batch, stream="critic")
            q1_pi, q2_pi = self.critic(x_q, onestep_a)
            q_pi = torch.min(q1_pi, q2_pi)
            if bool(cfg.fql_normalize_q):
                q_loss = -(q_pi / q_pi.abs().mean().detach().clamp(min=1e-6)).mean()
            else:
                q_loss = -q_pi.mean()
            actor_loss = bc_flow_loss + float(cfg.fql_alpha) * distill_loss + q_loss
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.flow.parameters()) + list(self.actor_task_embed.parameters()), 1.0)
            self.actor_opt.step()
        finally:
            for p, req in zip(critic_params, prev_requires_grad):
                p.requires_grad_(req)

        tau = float(cfg.target_tau)
        self._soft_update(self.critic, self.critic_target, tau)
        self._soft_update(self.critic_task_embed, self.critic_task_embed_target, tau)

        return {
            "qc_critic_loss": float(critic_loss.item()),
            "qc_actor_loss": float(actor_loss.item()),
            "qc_bc_flow_loss": float(bc_flow_loss.item()),
            "qc_distill_loss": float(distill_loss.item()),
            "qc_q_loss": float(q_loss.item()),
            "qc_q_mean": float(q_pi.mean().item()),
            "qc_target_q_mean": float(target_q.mean().item()),
            "qc_reward_mean": float(r.mean().item()),
            "qc_done_mean": float(d.mean().item()),
        }

    # ------------------------------------------------------------------
    # Deployment/evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_worker_chunk(
        self,
        z: np.ndarray,
        proprio: np.ndarray,
        full_state: np.ndarray,
        task_id: int,
        deterministic: bool = True,
    ) -> np.ndarray:
        # Flow/FQL policies are maps from latent noise to actions. There is no
        # closed-form deterministic mean, so eval samples noise under a seeded
        # RNG. `deterministic` is kept for evaluator API compatibility.
        del full_state, deterministic
        x_actor = self._input_from_state(z, proprio, task_id, stream="actor")
        algo = self.config.specialist.offline_algo.lower().replace("-", "_")
        if algo in {"flow_bc", "shared_flow_bc_positive"}:
            noise = torch.randn(1, self.action_dim, device=self.device)
            action = self.flow.flow_action(x_actor, noise)
        else:
            n = max(1, int(self.config.specialist.best_of_n))
            noise = torch.randn(n, self.action_dim, device=self.device)
            x_actor_n = x_actor.expand(n, -1)
            cand = self.flow.onestep_action(x_actor_n, noise).clamp(-1.0, 1.0)
            if n > 1:
                x_critic = self._input_from_state(z, proprio, task_id, stream="critic").expand(n, -1)
                q1, q2 = self.critic(x_critic, cand)
                action = cand[torch.min(q1, q2).squeeze(-1).argmax()].unsqueeze(0)
            else:
                action = cand[:1]
        return action.cpu().numpy().reshape(self.H_chunk, self.env_action_dim).clip(-1.0, 1.0)

    def _worker_step_reward(
        self,
        spec_err_before: float,
        spec_err_after: float,
        action_step: np.ndarray,
        completion_bit_flipped: bool,
        task_id: int,
    ) -> float:
        cfg = self.config.worker
        eps = float(self.spec.epsilon(task_id))
        sigma = float(cfg.sigma)
        phi_before = float(np.exp(-(spec_err_before / eps) / sigma))
        phi_after = float(np.exp(-(spec_err_after / eps) / sigma))
        reward = cfg.progress_weight * (cfg.gamma * phi_after - phi_before)
        reward += cfg.completion_bonus * float(completion_bit_flipped)
        reward -= cfg.action_cost * float(np.sum(action_step ** 2))
        return float(reward)

    def execute_option(
        self,
        env,
        task_id: int,
        start_img: np.ndarray,
        start_state: np.ndarray,
        start_z: np.ndarray,
        completion: np.ndarray,
        deterministic_worker: bool = True,
        collect_frames: bool = False,
    ) -> OptionResult:
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

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "flow": {k: v.detach().cpu().clone() for k, v in self.flow.state_dict().items()},
            "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
            "critic_target": {k: v.detach().cpu().clone() for k, v in self.critic_target.state_dict().items()},
            "actor_task_embed": {
                k: v.detach().cpu().clone() for k, v in self.actor_task_embed.state_dict().items()
            },
            "critic_task_embed": {
                k: v.detach().cpu().clone() for k, v in self.critic_task_embed.state_dict().items()
            },
            "critic_task_embed_target": {
                k: v.detach().cpu().clone() for k, v in self.critic_task_embed_target.state_dict().items()
            },
        }

    def restore_snapshot(self, snap: Dict[str, Dict[str, torch.Tensor]]):
        self.flow.load_state_dict({k: v.to(self.device) for k, v in snap["flow"].items()})
        self.critic.load_state_dict({k: v.to(self.device) for k, v in snap["critic"].items()})
        self.critic_target.load_state_dict({k: v.to(self.device) for k, v in snap["critic_target"].items()})
        self.actor_task_embed.load_state_dict(
            {k: v.to(self.device) for k, v in snap["actor_task_embed"].items()})
        self.critic_task_embed.load_state_dict(
            {k: v.to(self.device) for k, v in snap["critic_task_embed"].items()})
        self.critic_task_embed_target.load_state_dict(
            {k: v.to(self.device) for k, v in snap["critic_task_embed_target"].items()})

    def save(self, path: str):
        torch.save(
            {
                "format": "shared_qc_fql_v1",
                "tasks": self.tasks,
                "proprio_mean": self.proprio_norm.mean,
                "proprio_std": self.proprio_norm.std,
                "stage_a_task_success": self.stage_a_task_success,
                "curriculum_task_order": self.curriculum_task_order,
                **self.snapshot(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("format") != "shared_qc_fql_v1":
            raise ValueError(
                "Checkpoint is not a shared_qc_fql_v1 checkpoint. "
                "Old per-skill checkpoints are intentionally incompatible with the shared-policy pivot."
            )
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
        self.restore_snapshot(ckpt)
        self.reset_optimizers()


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def train_offline_skills(
    agent: SkillAgent,
    config: Config,
    verbose: bool = True,
    writer=None,
) -> Dict[str, float]:
    if verbose:
        print(f"  [Stage A] Building demo dataset from: {', '.join(config.warmup.dataset_ids)}")
    ds, stats = build_or_load_demo_dataset(agent, config, verbose=verbose)
    agent.demo_dataset = ds
    norm_source = ds.c_p if getattr(ds, "c_p", np.zeros((0, agent.proprio_dim))).shape[0] > 0 else ds.w_p
    agent.proprio_norm.fit(norm_source)

    results: Dict[str, float] = dict(stats)
    if verbose:
        print(f"  [Stage A] Shared actor rows : {ds.n_worker():,}")
        print(f"  [Stage A] Shared critic rows: {ds.n_critic():,}")
        print("  [Stage A] Positive labels by task:")
        for k, name in enumerate(agent.tasks):
            print(f"    {name:<14} {int(ds.worker_task_counts[k]):,}")

    algo = make_offline_algorithm(
        config.specialist.offline_algo,
        agent,
        config,
        writer=writer,
        verbose=verbose,
    )
    algo_result = algo.train(ds)
    results.update(algo_result.metrics)
    if verbose:
        _print_shared_metrics(config.specialist.offline_algo, algo_result.metrics)
    return results


def _print_shared_metrics(algo: str, metrics: Dict[str, float]):
    algo = algo.lower().replace("-", "_")
    if "flow_bc_loss/final" in metrics:
        print(f"  [Shared] flow BC loss={metrics['flow_bc_loss/final']:.4f}")
    if algo in {"qc_fql", "shared_qc_fql"} and "qc_critic_loss/final" in metrics:
        print(
            "  [Shared] QC-FQL "
            f"critic={metrics['qc_critic_loss/final']:.4f}  "
            f"distill={metrics['qc_distill_loss/final']:.4f}  "
            f"q_mean={metrics['qc_q_mean/final']:.3f}  "
            f"target_q={metrics['qc_target_q_mean/final']:.3f}"
        )
    if "prefix_success/mean_best" in metrics:
        print(
            f"  [Shared] prefix-val mean={metrics['prefix_success/mean_best']*100:.1f}%  "
            f"best_step={int(metrics['prefix_best_step'])}"
        )


__all__ = ["SkillAgent", "train_offline_skills", "OptionResult"]
