"""Offline skill-training algorithms for the lean FrankaKitchen branch.

The SkillAgent owns observations, policies, critics, and evaluation. This file
owns how a per-task policy is trained from the demo dataset.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from demo_dataset import sample_oracle_prefix_states
from env_wrapper import FrankaKitchenImageWrapper

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover - tqdm is optional at runtime.
    def trange(*args, **kwargs):
        del kwargs
        return range(*args)


def clone_state_dict_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def restore_state_dict(module: nn.Module, state: Dict[str, torch.Tensor], device: str):
    module.load_state_dict({k: v.to(device) for k, v in state.items()})


def soft_update(src: nn.Module, dst: nn.Module, tau: float):
    with torch.no_grad():
        for p, p_targ in zip(src.parameters(), dst.parameters()):
            p_targ.data.mul_(1.0 - tau).add_(tau * p.data)


def set_postfix(iterator, **kwargs):
    if hasattr(iterator, "set_postfix"):
        iterator.set_postfix(**kwargs)


def _prefix_tasks_for(agent, task_id: int) -> List[str]:
    """Use the scripted chain prefix for task-conditioned validation."""
    task_id = int(task_id)
    return [agent.tasks[k] for k in range(task_id)]


def _prefix_actor_eval(agent,
                       config,
                       samples: Sequence[Dict[str, np.ndarray]],
                       target_id: int) -> Dict[str, float]:
    """Evaluate a skill from fixed oracle prefix states.

    Success is measured with the environment completion bit, not the auxiliary
    close-enough condition, so this validation matches the chain metric.
    """
    if not samples:
        return {
            "success_rate": 0.0,
            "mean_final_error": float("inf"),
            "mean_options": 0.0,
        }

    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=False,
    )
    prev_env_steps = int(getattr(agent, "total_env_steps", 0))
    prev_options = int(getattr(agent, "total_options", 0))
    successes: List[float] = []
    final_errors: List[float] = []
    options: List[float] = []
    try:
        for i, sample in enumerate(samples):
            env.reset(seed=int(config.eval.prefix_sample_seed) + 1_000 + i)
            qpos, qvel = env.observation_to_qpos_qvel(sample["state"])
            env.set_mujoco_state(qpos, qvel)
            env._current_obs = {"observation": np.asarray(sample["state"], dtype=np.float64).copy()}
            env._step_count = 0

            state = np.asarray(sample["state"], dtype=np.float64).copy()
            img = env.render_image()
            z = agent.encoder.encode_numpy(img).squeeze()
            completion = np.asarray(sample["completion"], dtype=np.float32).copy()
            done = False
            n_opts = 0

            while (
                not done
                and completion[int(target_id)] < 0.5
                and n_opts < config.manager.max_high_level_steps
            ):
                result = agent.execute_option(
                    env=env,
                    task_id=int(target_id),
                    start_img=img,
                    start_state=state,
                    start_z=z,
                    completion=completion,
                    deterministic_worker=True,
                    collect_frames=False,
                )
                state = result.proprio_end
                z = result.z_end
                completion = result.completion_end
                done = result.env_done
                n_opts += 1
                if not done:
                    img = env.render_image()

            successes.append(float(completion[int(target_id)] > 0.5))
            final_errors.append(float(agent.spec.task_error(state, int(target_id))))
            options.append(float(n_opts))
    finally:
        env.close()
        agent.total_env_steps = prev_env_steps
        agent.total_options = prev_options

    return {
        "success_rate": float(np.mean(successes)),
        "mean_final_error": float(np.mean(final_errors)),
        "mean_options": float(np.mean(options)),
    }


class BeTActor(nn.Module):
    """A lightweight BeT-style actor: classify an action chunk code + residual.

    This is not a full Decision Transformer. It is a practical behavior
    transformer baseline for action-chunk imitation: cluster demo action chunks,
    predict the nearest action mode, then predict a residual for precision.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 action_dim: int, codebook: np.ndarray):
        super().__init__()
        from networks import build_mlp

        codebook_t = torch.as_tensor(codebook, dtype=torch.float32)
        self.register_buffer("codebook", codebook_t)
        self.action_dim = int(action_dim)
        self.n_codes = int(codebook_t.shape[0])
        self.trunk = build_mlp(input_dim, hidden_dim, hidden_dim, n_layers)
        self.logits_head = nn.Linear(hidden_dim, self.n_codes)
        self.offset_head = nn.Linear(hidden_dim, self.n_codes * self.action_dim)

    def forward_parts(self, x: torch.Tensor):
        h = self.trunk(x)
        logits = self.logits_head(h)
        offsets = self.offset_head(h).view(-1, self.n_codes, self.action_dim)
        return logits, offsets

    def deterministic(self, x: torch.Tensor) -> torch.Tensor:
        logits, offsets = self.forward_parts(x)
        code_id = torch.argmax(logits, dim=-1)
        row = torch.arange(x.shape[0], device=x.device)
        action = self.codebook[code_id] + offsets[row, code_id]
        return torch.tanh(action)


@dataclass
class AlgorithmResult:
    metrics: Dict[str, float]
    best_actor_state: Optional[Dict[str, torch.Tensor]] = None


class OfflineAlgorithm:
    def __init__(self, agent, config, writer=None, verbose: bool = True):
        self.agent = agent
        self.config = config
        self.writer = writer
        self.verbose = verbose
        self.log_interval = max(1, int(config.specialist.log_interval))

    def sample(self, ds, task_id: int):
        return ds.sample_worker_task_batch(
            task_id,
            self.config.specialist.batch_size,
            proprio_normalizer=self.agent.normalize_proprio,
        )

    def scalar(self, tag: str, value: float, step: int):
        if self.writer is not None and np.isfinite(float(value)):
            self.writer.add_scalar(tag, float(value), int(step))

    def should_log(self, step: int, total_steps: int) -> bool:
        return step == 1 or step % self.log_interval == 0 or step == total_steps

    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        raise NotImplementedError


class BCAlgorithm(OfflineAlgorithm):
    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        safe = task_name.replace(" ", "_")
        n_steps = int(self.config.specialist.n_teacher_bc_steps)
        skill = self.agent.skills[task_id]
        best_loss = float("inf")
        best_step = 0
        best_actor_state = clone_state_dict_cpu(skill.actor)
        losses = []

        iterator = trange(n_steps, desc=f"BC/{task_name}", leave=False, disable=not self.verbose)
        for i in iterator:
            step = i + 1
            loss = self.agent.bc_step(task_id, self.sample(ds, task_id))
            losses.append(loss)
            if loss < best_loss:
                best_loss = float(loss)
                best_step = step
                best_actor_state = clone_state_dict_cpu(skill.actor)
            if self.should_log(step, n_steps):
                self.scalar(f"skill/{safe}/bc_loss", loss, step)
            if self.verbose:
                set_postfix(iterator, loss=f"{loss:.4f}", best=f"{best_loss:.4f}")

        restore_state_dict(skill.actor, best_actor_state, self.agent.device)
        skill.actor_opt = torch.optim.Adam(skill.actor.parameters(), lr=self.config.worker.actor_lr)
        return AlgorithmResult(
            metrics={
                f"bc/{safe}_loss_final": float(np.mean(losses[-100:])),
                f"bc/{safe}_loss_best": float(best_loss),
                f"bc/{safe}_best_step": float(best_step),
            },
            best_actor_state=best_actor_state,
        )


class IQLAlgorithm(OfflineAlgorithm):
    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        safe = task_name.replace(" ", "_")
        n_steps = int(self.config.specialist.n_offline_rl_steps)
        if n_steps <= 0:
            return AlgorithmResult(metrics={})
        skill = self.agent.skills[task_id]
        best_actor_loss = float("inf")
        loss_best_actor_state = clone_state_dict_cpu(skill.actor)
        best_actor_state: Optional[Dict[str, torch.Tensor]] = None
        best_prefix_success = -1.0
        best_prefix_error = float("inf")
        best_prefix_step = 0
        prefix_samples: List[Dict[str, np.ndarray]] = []
        eval_interval = int(getattr(self.config.specialist, "iql_eval_interval", 0))
        n_prefix_states = int(getattr(self.config.specialist, "iql_eval_prefix_states", 0))
        if eval_interval > 0 and n_prefix_states > 0:
            try:
                prefix_samples, prefix_stats = sample_oracle_prefix_states(
                    agent=self.agent,
                    config=self.config,
                    prefix_tasks=_prefix_tasks_for(self.agent, task_id),
                    target_task=task_name,
                    max_states=n_prefix_states,
                    verbose=False,
                )
                if self.verbose:
                    print(
                        f"    IQL prefix-val states={len(prefix_samples)} "
                        f"matches={int(prefix_stats.get('matching_segments', 0.0))}"
                    )
            except Exception as exc:
                prefix_samples = []
                if self.verbose:
                    print(f"    IQL prefix-val disabled for {task_name}: {exc}")
        metrics_list = []
        iterator = trange(n_steps, desc=f"IQL/{task_name}", leave=False, disable=not self.verbose)
        for i in iterator:
            step = i + 1
            metrics = self.agent.iql_step(task_id, self.sample(ds, task_id))
            metrics_list.append(metrics)
            if float(metrics["iql_actor_loss"]) < best_actor_loss:
                best_actor_loss = float(metrics["iql_actor_loss"])
                loss_best_actor_state = clone_state_dict_cpu(skill.actor)
            if prefix_samples and (step % eval_interval == 0 or step == n_steps):
                eval_stats = _prefix_actor_eval(self.agent, self.config, prefix_samples, task_id)
                success = float(eval_stats["success_rate"])
                error = float(eval_stats["mean_final_error"])
                if self.writer is not None:
                    tb_step = self.config.specialist.n_teacher_bc_steps + step
                    self.scalar(f"skill/{safe}/iql_prefix_success_rate", success, tb_step)
                    self.scalar(f"skill/{safe}/iql_prefix_mean_final_error", error, tb_step)
                    self.scalar(f"skill/{safe}/iql_prefix_mean_options", eval_stats["mean_options"], tb_step)
                if (
                    success > best_prefix_success
                    or (success == best_prefix_success and error < best_prefix_error)
                ):
                    best_prefix_success = success
                    best_prefix_error = error
                    best_prefix_step = step
                    best_actor_state = clone_state_dict_cpu(skill.actor)
            if self.should_log(step, n_steps):
                tb_step = self.config.specialist.n_teacher_bc_steps + step
                for key, value in metrics.items():
                    self.scalar(f"skill/{safe}/{key}", value, tb_step)
            if self.verbose:
                set_postfix(
                    iterator,
                    v=f"{metrics['iql_value_loss']:.3f}",
                    q=f"{metrics['iql_critic_loss']:.3f}",
                )
        if best_actor_state is None:
            # Fallback only when prefix validation is unavailable. This keeps
            # runs from crashing on unusual task orders or missing prefix data.
            best_actor_state = loss_best_actor_state
        restore_state_dict(skill.actor, best_actor_state, self.agent.device)
        skill.actor_opt = torch.optim.Adam(skill.actor.parameters(), lr=self.config.worker.actor_lr)
        out = {}
        for key in metrics_list[-1].keys():
            out[f"{key}/{safe}_final"] = float(np.mean([m[key] for m in metrics_list[-100:]]))
        out[f"iql_actor_loss/{safe}_best"] = float(best_actor_loss)
        if best_prefix_step > 0:
            out[f"iql_prefix_success/{safe}_best"] = float(best_prefix_success)
            out[f"iql_prefix_error/{safe}_best"] = float(best_prefix_error)
            out[f"iql_prefix_best_step/{safe}"] = float(best_prefix_step)
        return AlgorithmResult(metrics=out, best_actor_state=best_actor_state)


class BCIQLAlgorithm(OfflineAlgorithm):
    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        bc = BCAlgorithm(self.agent, self.config, self.writer, self.verbose).train_task(ds, task_id, task_name)
        iql = IQLAlgorithm(self.agent, self.config, self.writer, self.verbose).train_task(ds, task_id, task_name)
        # BCAlgorithm restores best-BC actor before returning.
        # IQLAlgorithm now restores best-IQL actor before returning.
        # The in-place state is already correct; propagate iql.best_actor_state
        # so callers know which state is loaded.
        return AlgorithmResult(metrics={**bc.metrics, **iql.metrics}, best_actor_state=iql.best_actor_state)


class TD3BCAlgorithm(OfflineAlgorithm):
    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        safe = task_name.replace(" ", "_")
        bc = BCAlgorithm(self.agent, self.config, self.writer, self.verbose).train_task(ds, task_id, task_name)
        n_steps = int(self.config.specialist.n_offline_rl_steps)
        if n_steps <= 0:
            return bc

        skill = self.agent.skills[task_id]
        actor_target = copy.deepcopy(skill.actor).to(self.agent.device).eval()
        critic_target = copy.deepcopy(skill.critic).to(self.agent.device).eval()
        alpha = float(self.config.specialist.td3bc_alpha)
        tau = float(self.config.specialist.td3bc_tau)
        policy_noise = float(self.config.specialist.td3bc_policy_noise)
        noise_clip = float(self.config.specialist.td3bc_noise_clip)
        policy_freq = max(1, int(self.config.specialist.td3bc_policy_freq))
        metrics_list = []

        iterator = trange(n_steps, desc=f"TD3BC/{task_name}", leave=False, disable=not self.verbose)
        for i in iterator:
            step = i + 1
            batch = self.sample(ds, task_id)
            x = self.agent._input_from_batch(batch)
            xn = self.agent._input_next_from_batch(batch)
            a = torch.from_numpy(batch["action"]).to(self.agent.device).clamp(-0.999, 0.999)
            r = torch.from_numpy(batch["reward"]).unsqueeze(1).to(self.agent.device)
            d = torch.from_numpy(batch["done"]).unsqueeze(1).to(self.agent.device)

            with torch.no_grad():
                noise = (torch.randn_like(a) * policy_noise).clamp(-noise_clip, noise_clip)
                next_a = (actor_target.deterministic(xn) + noise).clamp(-0.999, 0.999)
                q1_t, q2_t = critic_target(xn, next_a)
                target_q = r + self.config.worker.gamma * (1.0 - d) * torch.min(q1_t, q2_t)

            q1, q2 = skill.critic(x, a)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            skill.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(skill.critic.parameters(), 1.0)
            skill.critic_opt.step()

            actor_loss = torch.zeros((), device=self.agent.device)
            bc_loss = torch.zeros((), device=self.agent.device)
            lam = torch.zeros((), device=self.agent.device)
            if step % policy_freq == 0:
                pi = skill.actor.deterministic(x)
                q_pi = skill.critic.q1(torch.cat([x, pi], dim=-1))
                lam = alpha / q_pi.abs().mean().detach().clamp(min=1e-6)
                bc_loss = F.mse_loss(pi, a)
                actor_loss = bc_loss - lam * q_pi.mean()
                skill.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(skill.actor.parameters(), 1.0)
                skill.actor_opt.step()
                soft_update(skill.actor, actor_target, tau)
                soft_update(skill.critic, critic_target, tau)

            metrics = {
                "td3bc_critic_loss": float(critic_loss.item()),
                "td3bc_actor_loss": float(actor_loss.item()),
                "td3bc_bc_loss": float(bc_loss.item()),
                "td3bc_lambda": float(lam.item()),
            }
            metrics_list.append(metrics)
            if self.should_log(step, n_steps):
                tb_step = self.config.specialist.n_teacher_bc_steps + step
                for key, value in metrics.items():
                    self.scalar(f"skill/{safe}/{key}", value, tb_step)
            if self.verbose:
                set_postfix(iterator, q=f"{metrics['td3bc_critic_loss']:.3f}", bc=f"{metrics['td3bc_bc_loss']:.4f}")

        out = dict(bc.metrics)
        for key in metrics_list[-1].keys():
            out[f"{key}/{safe}_final"] = float(np.mean([m[key] for m in metrics_list[-100:]]))
        return AlgorithmResult(metrics=out, best_actor_state=bc.best_actor_state)


class AWRAlgorithm(OfflineAlgorithm):
    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        safe = task_name.replace(" ", "_")
        bc = BCAlgorithm(self.agent, self.config, self.writer, self.verbose).train_task(ds, task_id, task_name)
        n_steps = int(self.config.specialist.n_offline_rl_steps)
        if n_steps <= 0:
            return bc

        skill = self.agent.skills[task_id]
        temperature = float(self.config.specialist.awr_temperature)
        max_weight = float(self.config.specialist.awr_max_weight)
        metrics_list = []
        iterator = trange(n_steps, desc=f"AWR/{task_name}", leave=False, disable=not self.verbose)
        for i in iterator:
            step = i + 1
            batch = self.sample(ds, task_id)
            x = self.agent._input_from_batch(batch)
            xn = self.agent._input_next_from_batch(batch)
            a = torch.from_numpy(batch["action"]).to(self.agent.device).clamp(-0.999, 0.999)
            r = torch.from_numpy(batch["reward"]).unsqueeze(1).to(self.agent.device)
            d = torch.from_numpy(batch["done"]).unsqueeze(1).to(self.agent.device)

            with torch.no_grad():
                q1_det, q2_det = skill.critic(x, a)
                q_det = torch.min(q1_det, q2_det)
            v = skill.value(x)
            adv = q_det - v
            expectile = self.config.specialist.iql_expectile
            value_weight = torch.where(adv > 0, expectile, 1.0 - expectile)
            value_loss = (value_weight * adv.pow(2)).mean()
            skill.value_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(skill.value.parameters(), 1.0)
            skill.value_opt.step()

            with torch.no_grad():
                next_v = skill.value_target(xn) if self.config.specialist.iql_use_value_target else skill.value(xn)
                target_q = r + self.config.worker.gamma * (1.0 - d) * next_v
            q1, q2 = skill.critic(x, a)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            skill.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(skill.critic.parameters(), 1.0)
            skill.critic_opt.step()

            with torch.no_grad():
                q1_a, q2_a = skill.critic(x, a)
                adv_actor = torch.min(q1_a, q2_a) - skill.value(x)
                weights = torch.exp(adv_actor / max(temperature, 1e-6)).clamp(max=max_weight)
            pred = skill.actor.deterministic(x)
            actor_loss = (weights * (pred - a).pow(2).mean(dim=-1, keepdim=True)).mean()
            skill.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(skill.actor.parameters(), 1.0)
            skill.actor_opt.step()
            if self.config.specialist.iql_use_value_target:
                self.agent._soft_update(
                    skill.value,
                    skill.value_target,
                    float(self.config.specialist.iql_value_target_tau),
                )

            metrics = {
                "awr_value_loss": float(value_loss.item()),
                "awr_critic_loss": float(critic_loss.item()),
                "awr_actor_loss": float(actor_loss.item()),
                "awr_weight_mean": float(weights.mean().item()),
                "awr_weight_max": float(weights.max().item()),
                "awr_target_q_mean": float(target_q.mean().item()),
            }
            metrics_list.append(metrics)
            if self.should_log(step, n_steps):
                tb_step = self.config.specialist.n_teacher_bc_steps + step
                for key, value in metrics.items():
                    self.scalar(f"skill/{safe}/{key}", value, tb_step)
            if self.verbose:
                set_postfix(iterator, q=f"{metrics['awr_critic_loss']:.3f}", w=f"{metrics['awr_weight_mean']:.2f}")

        out = dict(bc.metrics)
        for key in metrics_list[-1].keys():
            out[f"{key}/{safe}_final"] = float(np.mean([m[key] for m in metrics_list[-100:]]))
        return AlgorithmResult(metrics=out, best_actor_state=bc.best_actor_state)


def _init_codebook(actions: np.ndarray, n_codes: int, n_iter: int = 15) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    n_codes = int(min(max(2, n_codes), len(actions)))
    rng_idx = np.random.choice(len(actions), size=n_codes, replace=False)
    centers = actions[rng_idx].copy()
    for _ in range(n_iter):
        d2 = ((actions[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = np.argmin(d2, axis=1)
        for k in range(n_codes):
            mask = labels == k
            if np.any(mask):
                centers[k] = actions[mask].mean(axis=0)
    return centers.astype(np.float32)


class BeTAlgorithm(OfflineAlgorithm):
    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        safe = task_name.replace(" ", "_")
        indices = ds.worker_indices_by_task[int(task_id)]
        codebook = _init_codebook(ds.w_a[indices], self.config.specialist.bet_num_bins)
        skill = self.agent.skills[task_id]
        skill.actor = BeTActor(
            self.agent.policy_input_dim,
            self.config.specialist.hidden_dim,
            self.config.specialist.n_layers,
            self.agent.action_dim,
            codebook,
        ).to(self.agent.device)
        skill.actor_opt = torch.optim.Adam(skill.actor.parameters(), lr=self.config.worker.actor_lr)

        n_steps = int(self.config.specialist.bet_steps)
        best_loss = float("inf")
        best_step = 0
        best_actor_state = clone_state_dict_cpu(skill.actor)
        losses = []
        offset_weight = float(self.config.specialist.bet_offset_weight)

        iterator = trange(n_steps, desc=f"BeT/{task_name}", leave=False, disable=not self.verbose)
        for i in iterator:
            step = i + 1
            batch = self.sample(ds, task_id)
            x = self.agent._input_from_batch(batch)
            a = torch.from_numpy(batch["action"]).to(self.agent.device).clamp(-0.999, 0.999)
            with torch.no_grad():
                d2 = ((a[:, None, :] - skill.actor.codebook[None, :, :]) ** 2).sum(dim=-1)
                labels = torch.argmin(d2, dim=-1)
                target_residual = a - skill.actor.codebook[labels]
            logits, offsets = skill.actor.forward_parts(x)
            row = torch.arange(x.shape[0], device=x.device)
            pred_residual = offsets[row, labels]
            cls_loss = F.cross_entropy(logits, labels)
            residual_loss = F.mse_loss(pred_residual, target_residual)
            loss = cls_loss + offset_weight * residual_loss
            skill.actor_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(skill.actor.parameters(), 1.0)
            skill.actor_opt.step()
            losses.append(float(loss.item()))
            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_step = step
                best_actor_state = clone_state_dict_cpu(skill.actor)
            if self.should_log(step, n_steps):
                self.scalar(f"skill/{safe}/bet_loss", float(loss.item()), step)
                self.scalar(f"skill/{safe}/bet_cls_loss", float(cls_loss.item()), step)
                self.scalar(f"skill/{safe}/bet_residual_loss", float(residual_loss.item()), step)
            if self.verbose:
                set_postfix(iterator, loss=f"{loss.item():.3f}", cls=f"{cls_loss.item():.3f}")

        restore_state_dict(skill.actor, best_actor_state, self.agent.device)
        return AlgorithmResult(
            metrics={
                f"bet/{safe}_loss_final": float(np.mean(losses[-100:])),
                f"bet/{safe}_loss_best": float(best_loss),
                f"bet/{safe}_best_step": float(best_step),
            },
            best_actor_state=best_actor_state,
        )


def make_offline_algorithm(name: str, agent, config, writer=None, verbose: bool = True) -> OfflineAlgorithm:
    name = name.lower().replace("-", "_")
    if name == "bc":
        return BCAlgorithm(agent, config, writer, verbose)
    if name in ("bc_iql", "iql"):
        return BCIQLAlgorithm(agent, config, writer, verbose)
    if name in ("td3bc", "td3_bc"):
        return TD3BCAlgorithm(agent, config, writer, verbose)
    if name == "awr":
        return AWRAlgorithm(agent, config, writer, verbose)
    if name in ("bet", "behavior_transformer", "sequence_bc"):
        return BeTAlgorithm(agent, config, writer, verbose)
    raise ValueError(f"Unknown offline algorithm '{name}'")
