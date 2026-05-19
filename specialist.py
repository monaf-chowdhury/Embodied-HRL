"""
specialist.py -- Lean per-skill BC/IQL learner.

Current research path:
  1. Load replay-labelled FrankaKitchen demonstrations.
  2. Train one visual low-level policy per task with the same protocol.
  3. Evaluate skills alone and under a scripted task order.

No hierarchy, no teacher/student, no residual stack. Keep this file boring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from config import Config
from demo_dataset import build_or_load_demo_dataset
from encoder import VisualEncoder
from env_wrapper import FrankaKitchenImageWrapper
from networks import build_mlp
from offline_algorithms import make_offline_algorithm
from utils import TaskSpec, build_frozen_text_embeddings, build_task_state_flat


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class OptionResult:
    z_start: np.ndarray
    proprio_start: np.ndarray
    task_state_start: np.ndarray
    completion_start: np.ndarray
    z_end: np.ndarray
    proprio_end: np.ndarray
    task_state_end: np.ndarray
    completion_end: np.ndarray
    chosen_task: int
    chosen_task_completed: bool
    any_task_completed: bool
    new_completions: int
    offtask_completions: int
    steps_taken: int
    env_done: bool
    termination_reason: str
    option_return: float
    env_reward_sum: float
    task_error_reduction: float
    frames: List[np.ndarray]
    last_worker_losses: Dict[str, float]


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


class SkillActor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, action_dim: int):
        super().__init__()
        self.trunk = build_mlp(input_dim, hidden_dim, hidden_dim, n_layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _dist(self, x: torch.Tensor) -> Normal:
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def forward(self, x: torch.Tensor):
        dist = self._dist(x)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logp.sum(-1, keepdim=True)

    def deterministic(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.tanh(self.mean_head(h))

    def log_prob_from_action(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self._dist(x)
        clipped = action.clamp(-0.999, 0.999)
        pre_tanh = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        logp = dist.log_prob(pre_tanh) - torch.log(1 - clipped.pow(2) + 1e-6)
        return logp.sum(-1, keepdim=True)


class TwinQ(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, action_dim: int):
        super().__init__()
        self.q1 = build_mlp(input_dim + action_dim, hidden_dim, 1, n_layers)
        self.q2 = build_mlp(input_dim + action_dim, hidden_dim, 1, n_layers)

    def forward(self, x: torch.Tensor, action: torch.Tensor):
        xa = torch.cat([x, action], dim=-1)
        return self.q1(xa), self.q2(xa)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.v = build_mlp(input_dim, hidden_dim, 1, n_layers)

    def forward(self, x: torch.Tensor):
        return self.v(x)


@dataclass
class Skill:
    actor: SkillActor
    critic: TwinQ
    value: ValueNet
    actor_opt: torch.optim.Optimizer
    critic_opt: torch.optim.Optimizer
    value_opt: torch.optim.Optimizer


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
        self.skills: List[Skill] = []
        for _ in self.tasks:
            actor = SkillActor(self.policy_input_dim, hidden, layers, self.action_dim).to(self.device)
            critic = TwinQ(self.policy_input_dim, hidden, layers, self.action_dim).to(self.device)
            value = ValueNet(self.policy_input_dim, hidden, layers).to(self.device)
            self.skills.append(
                Skill(
                    actor=actor,
                    critic=critic,
                    value=value,
                    actor_opt=torch.optim.Adam(actor.parameters(), lr=config.worker.actor_lr),
                    critic_opt=torch.optim.Adam(critic.parameters(), lr=config.worker.critic_lr),
                    value_opt=torch.optim.Adam(value.parameters(), lr=config.worker.critic_lr),
                )
            )

        self.stage_a_task_success = np.zeros(self.n_tasks, dtype=np.float32)
        self.curriculum_task_order = list(range(self.n_tasks))
        self.total_env_steps = 0
        self.total_options = 0
        self.total_episodes = 0
        self.epsilon = 0.0

    def normalize_proprio(self, p: np.ndarray) -> np.ndarray:
        return self.proprio_norm(p)

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

    def bc_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> float:
        skill = self.skills[task_id]
        x = self._input_from_batch(batch)
        a = torch.from_numpy(batch["action"]).to(self.device).clamp(-0.999, 0.999)
        pred = skill.actor.deterministic(x)
        loss = F.mse_loss(pred, a)
        skill.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.actor.parameters(), 1.0)
        skill.actor_opt.step()
        return float(loss.item())

    def iql_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        skill = self.skills[task_id]
        x = self._input_from_batch(batch)
        xn = self._input_next_from_batch(batch)
        a = torch.from_numpy(batch["action"]).to(self.device).clamp(-0.999, 0.999)
        r = torch.from_numpy(batch["reward"]).unsqueeze(1).to(self.device)
        d = torch.from_numpy(batch["done"]).unsqueeze(1).to(self.device)

        with torch.no_grad():
            q1_det, q2_det = skill.critic(x, a)
            q_det = torch.min(q1_det, q2_det)
        v = skill.value(x)
        adv = q_det - v
        expectile = self.config.specialist.iql_expectile
        weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        value_loss = (weight * adv.pow(2)).mean()
        skill.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.value.parameters(), 1.0)
        skill.value_opt.step()

        with torch.no_grad():
            target_q = r + self.config.worker.gamma * (1.0 - d) * skill.value(xn)
        q1, q2 = skill.critic(x, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        skill.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.critic.parameters(), 1.0)
        skill.critic_opt.step()

        with torch.no_grad():
            q1_pi, q2_pi = skill.critic(x, a)
            adv_pi = torch.min(q1_pi, q2_pi) - skill.value(x)
            exp_adv = torch.exp(self.config.specialist.iql_adv_beta * adv_pi).clamp(
                max=self.config.specialist.iql_max_weight
            )
        logp = skill.actor.log_prob_from_action(x, a)
        actor_loss = -(exp_adv * logp).mean()
        skill.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.actor.parameters(), 1.0)
        skill.actor_opt.step()

        return {
            "iql_value_loss": float(value_loss.item()),
            "iql_critic_loss": float(critic_loss.item()),
            "iql_actor_loss": float(actor_loss.item()),
            "iql_adv_mean": float(adv_pi.mean().item()),
            "iql_weight_mean": float(exp_adv.mean().item()),
        }

    @torch.no_grad()
    def get_worker_chunk(self,
                         z: np.ndarray,
                         proprio: np.ndarray,
                         full_state: np.ndarray,
                         task_id: int,
                         deterministic: bool = True) -> np.ndarray:
        del full_state
        x = self._input_from_state(z, proprio, task_id)
        skill = self.skills[task_id]
        action = skill.actor.deterministic(x) if deterministic else skill.actor(x)[0]
        return action.cpu().numpy().reshape(self.H_chunk, self.env_action_dim).clip(-1.0, 1.0)

    def _worker_step_reward(self,
                            spec_err_before: float,
                            spec_err_after: float,
                            action_step: np.ndarray,
                            completion_bit_flipped: bool) -> float:
        cfg = self.config.worker
        reward = cfg.progress_weight * (spec_err_before - spec_err_after)
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
                       collect_frames: bool = False,
                       train_worker_online: bool = False,
                       update_every_n_env_steps: int = 1) -> OptionResult:
        del train_worker_online, update_every_n_env_steps
        chosen_name = self.spec.name(task_id)
        state = start_state.copy()
        z = start_z.copy()
        completion_mask = completion.copy()
        already_completed = {self.tasks[i] for i, v in enumerate(completion_mask) if v > 0.5}
        new_completion_names: List[str] = []
        start_error = self.spec.task_error(state, task_id)
        frames = [start_img.copy()] if collect_frames else []
        option_return = 0.0
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
                completion_next = np.maximum(completion_mask, raw_completion)

                just_completed = [
                    name for name in info.get("tasks_completed_names", [])
                    if name not in already_completed
                ]
                for name in just_completed:
                    already_completed.add(name)
                    if name not in new_completion_names:
                        new_completion_names.append(name)
                if chosen_name in just_completed:
                    chosen_completed = True
                    termination_reason = "completed"

                err_before = self.spec.task_error(state, task_id)
                err_after = self.spec.task_error(next_state, task_id)
                option_return += self._worker_step_reward(
                    err_before, err_after, action_step, chosen_name in just_completed
                )
                option_env_reward += float(env_reward)
                state = next_state
                z = next_z
                completion_mask = completion_next
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
            z_start=start_z.copy(),
            proprio_start=start_state.copy(),
            task_state_start=build_task_state_flat(self.spec, start_state),
            completion_start=completion.copy(),
            z_end=z.copy(),
            proprio_end=state.copy(),
            task_state_end=build_task_state_flat(self.spec, state),
            completion_end=completion_mask.copy(),
            chosen_task=task_id,
            chosen_task_completed=bool(chosen_completed),
            any_task_completed=bool(new_completion_names),
            new_completions=int(np.sum(completion_mask > completion)),
            offtask_completions=int(sum(1 for n in new_completion_names if n != chosen_name)),
            steps_taken=steps_taken,
            env_done=env_done,
            termination_reason=termination_reason,
            option_return=float(option_return),
            env_reward_sum=float(option_env_reward),
            task_error_reduction=float(start_error - self.spec.task_error(state, task_id)),
            frames=frames,
            last_worker_losses={},
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
                        "actor": s.actor.state_dict(),
                        "critic": s.critic.state_dict(),
                        "value": s.value.state_dict(),
                    }
                    for s in self.skills
                ],
            },
            path,
        )


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
    if f"bc/{safe}_loss_final" in metrics:
        print(f"    BC final={metrics[f'bc/{safe}_loss_final']:.4f}  "
              f"best={metrics[f'bc/{safe}_loss_best']:.4f} @ step {int(metrics[f'bc/{safe}_best_step'])}")
    algo = algo.lower().replace("-", "_")
    if algo in ("bc_iql", "iql") and f"iql_value_loss/{safe}_final" in metrics:
        print(f"    IQL value={metrics[f'iql_value_loss/{safe}_final']:.4f}  "
              f"critic={metrics[f'iql_critic_loss/{safe}_final']:.4f}  "
              f"actor={metrics[f'iql_actor_loss/{safe}_final']:.4f}")
    elif algo in ("td3bc", "td3_bc") and f"td3bc_critic_loss/{safe}_final" in metrics:
        print(f"    TD3+BC critic={metrics[f'td3bc_critic_loss/{safe}_final']:.4f}  "
              f"actor={metrics[f'td3bc_actor_loss/{safe}_final']:.4f}  "
              f"bc={metrics[f'td3bc_bc_loss/{safe}_final']:.4f}")
    elif algo == "awr" and f"awr_critic_loss/{safe}_final" in metrics:
        print(f"    AWR value={metrics[f'awr_value_loss/{safe}_final']:.4f}  "
              f"critic={metrics[f'awr_critic_loss/{safe}_final']:.4f}  "
              f"actor={metrics[f'awr_actor_loss/{safe}_final']:.4f}")
    elif algo in ("bet", "behavior_transformer", "sequence_bc") and f"bet/{safe}_loss_final" in metrics:
        print(f"    BeT loss={metrics[f'bet/{safe}_loss_final']:.4f}  "
              f"best={metrics[f'bet/{safe}_loss_best']:.4f} @ step {int(metrics[f'bet/{safe}_best_step'])}")


__all__ = ["SkillAgent", "train_offline_skills", "OptionResult"]
