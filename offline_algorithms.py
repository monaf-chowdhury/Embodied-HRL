"""Offline per-skill training algorithms (QC-FQL branch).

Two algorithms only:
  * FlowBCAlgorithm  — flow-matching behavior cloning (staged experiment 1:
                       "does an expressive policy class alone beat the Gaussian?")
  * QCFQLAlgorithm   — full Q-chunking + Flow Q-Learning (chunked twin critic,
                       flow BC, one-step Q-maximizing actor).

Both use the same prefix-state validation for model selection: every
`eval_interval` steps the current policy is rolled out from oracle prefix
states and the best-by-success checkpoint (flow + critic) is kept.

References:
  Flow Q-Learning — Park, Li, Levine, ICML 2025 (arXiv:2502.02538)
  RL with Action Chunking — Li, Zhou, Levine, NeurIPS 2025 (arXiv:2507.07969)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from demo_dataset import sample_oracle_prefix_states
from env_wrapper import FrankaKitchenImageWrapper
from utils import rng_isolated

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    def trange(*args, **kwargs):
        del kwargs
        return range(*args)


def clone_state_dict_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def restore_state_dict(module: nn.Module, state: Dict[str, torch.Tensor], device: str):
    module.load_state_dict({k: v.to(device) for k, v in state.items()})


def set_postfix(iterator, **kwargs):
    if hasattr(iterator, "set_postfix"):
        iterator.set_postfix(**kwargs)


def _prefix_tasks_for(agent, task_id: int) -> List[str]:
    """Scripted chain prefix (tasks before `task_id`) for task-conditioned eval."""
    return [agent.tasks[k] for k in range(int(task_id))]


@rng_isolated
def _prefix_actor_eval(agent,
                       config,
                       samples: Sequence[Dict[str, np.ndarray]],
                       target_id: int) -> Dict[str, float]:
    """Roll out the current skill from fixed oracle prefix states.

    Success uses the environment completion bit (matches the chain metric).
    Seeded + RNG-isolated so model-selection rollouts (including the flow
    policy's latent samples) are reproducible across steps and do not perturb
    the offline training noise stream.
    """
    if not samples:
        return {"success_rate": 0.0, "mean_final_error": float("inf"), "mean_options": 0.0}

    np.random.seed(12345)
    torch.manual_seed(12345)
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=False,
    )
    prev_env_steps = int(getattr(agent, "total_env_steps", 0))
    prev_options = int(getattr(agent, "total_options", 0))
    successes, final_errors, options = [], [], []
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
            while (not done
                   and completion[int(target_id)] < 0.5
                   and n_opts < config.manager.max_high_level_steps):
                result = agent.execute_option(
                    env=env, task_id=int(target_id), start_img=img, start_state=state,
                    start_z=z, completion=completion, deterministic_worker=True,
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


@dataclass
class AlgorithmResult:
    metrics: Dict[str, float]


class OfflineAlgorithm:
    def __init__(self, agent, config, writer=None, verbose: bool = True):
        self.agent = agent
        self.config = config
        self.writer = writer
        self.verbose = verbose
        self.log_interval = max(1, int(config.specialist.log_interval))

    def sample(self, ds, task_id: int) -> Dict[str, np.ndarray]:
        return ds.sample_worker_task_batch(
            task_id, self.config.specialist.batch_size,
            proprio_normalizer=self.agent.normalize_proprio,
        )

    def scalar(self, tag: str, value: float, step: int):
        if self.writer is not None and np.isfinite(float(value)):
            self.writer.add_scalar(tag, float(value), int(step))

    def should_log(self, step: int, total_steps: int) -> bool:
        return step == 1 or step % self.log_interval == 0 or step == total_steps

    # -- shared prefix-validation model selection ------------------------------

    def _load_prefix_samples(self, task_id: int, task_name: str):
        eval_interval = int(self.config.specialist.eval_interval)
        n_states = int(self.config.specialist.eval_prefix_states)
        if eval_interval <= 0 or n_states <= 0:
            return [], eval_interval
        try:
            samples, stats = sample_oracle_prefix_states(
                agent=self.agent, config=self.config,
                prefix_tasks=_prefix_tasks_for(self.agent, task_id),
                target_task=task_name, max_states=n_states, verbose=False,
            )
            if self.verbose:
                print(f"    prefix-val states={len(samples)} "
                      f"matches={int(stats.get('matching_segments', 0.0))}")
            return samples, eval_interval
        except Exception as exc:
            if self.verbose:
                print(f"    prefix-val disabled for {task_name}: {exc}")
            return [], eval_interval

    def _snapshot(self, skill):
        return {
            "flow": clone_state_dict_cpu(skill.flow),
            "critic": clone_state_dict_cpu(skill.critic),
            "critic_target": clone_state_dict_cpu(skill.critic_target),
        }

    def _restore(self, skill, snap):
        restore_state_dict(skill.flow, snap["flow"], self.agent.device)
        restore_state_dict(skill.critic, snap["critic"], self.agent.device)
        restore_state_dict(skill.critic_target, snap["critic_target"], self.agent.device)

    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        raise NotImplementedError


class FlowBCAlgorithm(OfflineAlgorithm):
    """Flow-matching BC only (no critic)."""

    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        safe = task_name.replace(" ", "_")
        n_steps = int(self.config.specialist.n_flow_bc_steps)
        if n_steps <= 0:
            return AlgorithmResult(metrics={})
        skill = self.agent.skills[task_id]
        prefix_samples, eval_interval = self._load_prefix_samples(task_id, task_name)
        best_success, best_error, best_step, best_snap = -1.0, float("inf"), 0, None
        metrics_list = []

        iterator = trange(n_steps, desc=f"FlowBC/{task_name}", leave=False, disable=not self.verbose)
        for i in iterator:
            step = i + 1
            metrics = self.agent.flow_bc_step(task_id, self.sample(ds, task_id))
            metrics_list.append(metrics)
            if prefix_samples and (step % eval_interval == 0 or step == n_steps):
                ev = _prefix_actor_eval(self.agent, self.config, prefix_samples, task_id)
                self.scalar(f"skill/{safe}/prefix_success_rate", ev["success_rate"], step)
                if ev["success_rate"] > best_success or (
                        ev["success_rate"] == best_success and ev["mean_final_error"] < best_error):
                    best_success, best_error, best_step = ev["success_rate"], ev["mean_final_error"], step
                    best_snap = self._snapshot(skill)
            if self.should_log(step, n_steps):
                for key, value in metrics.items():
                    self.scalar(f"skill/{safe}/{key}", value, step)
            if self.verbose:
                set_postfix(iterator, bc=f"{metrics['flow_bc_loss']:.4f}")

        if best_snap is not None:
            self._restore(skill, best_snap)
        out = {f"{k}/{safe}_final": float(np.mean([m[k] for m in metrics_list[-100:]]))
               for k in metrics_list[-1].keys()}
        if best_step > 0:
            out[f"prefix_success/{safe}_best"] = float(best_success)
            out[f"prefix_error/{safe}_best"] = float(best_error)
            out[f"prefix_best_step/{safe}"] = float(best_step)
        return AlgorithmResult(metrics=out)


class QCFQLAlgorithm(OfflineAlgorithm):
    """Q-chunking + Flow Q-Learning (chunked critic + flow BC + one-step actor)."""

    def train_task(self, ds, task_id: int, task_name: str) -> AlgorithmResult:
        safe = task_name.replace(" ", "_")
        n_steps = int(self.config.specialist.n_offline_rl_steps)
        if n_steps <= 0:
            return AlgorithmResult(metrics={})
        skill = self.agent.skills[task_id]
        prefix_samples, eval_interval = self._load_prefix_samples(task_id, task_name)
        best_success, best_error, best_step, best_snap = -1.0, float("inf"), 0, None
        metrics_list = []

        iterator = trange(n_steps, desc=f"QC-FQL/{task_name}", leave=False, disable=not self.verbose)
        for i in iterator:
            step = i + 1
            metrics = self.agent.qc_fql_step(task_id, self.sample(ds, task_id))
            metrics_list.append(metrics)
            if prefix_samples and (step % eval_interval == 0 or step == n_steps):
                ev = _prefix_actor_eval(self.agent, self.config, prefix_samples, task_id)
                self.scalar(f"skill/{safe}/prefix_success_rate", ev["success_rate"], step)
                self.scalar(f"skill/{safe}/prefix_mean_final_error", ev["mean_final_error"], step)
                if ev["success_rate"] > best_success or (
                        ev["success_rate"] == best_success and ev["mean_final_error"] < best_error):
                    best_success, best_error, best_step = ev["success_rate"], ev["mean_final_error"], step
                    best_snap = self._snapshot(skill)
            if self.should_log(step, n_steps):
                for key, value in metrics.items():
                    self.scalar(f"skill/{safe}/{key}", value, step)
            if self.verbose:
                set_postfix(iterator, q=f"{metrics['qc_critic_loss']:.3f}",
                            distill=f"{metrics['qc_distill_loss']:.4f}")

        # Fall back to the final policy if prefix validation was unavailable.
        if best_snap is not None:
            self._restore(skill, best_snap)
        out = {f"{k}/{safe}_final": float(np.mean([m[k] for m in metrics_list[-100:]]))
               for k in metrics_list[-1].keys()}
        if best_step > 0:
            out[f"prefix_success/{safe}_best"] = float(best_success)
            out[f"prefix_error/{safe}_best"] = float(best_error)
            out[f"prefix_best_step/{safe}"] = float(best_step)
        return AlgorithmResult(metrics=out)


def make_offline_algorithm(name: str, agent, config, writer=None, verbose: bool = True) -> OfflineAlgorithm:
    name = name.lower().replace("-", "_")
    if name == "flow_bc":
        return FlowBCAlgorithm(agent, config, writer, verbose)
    if name == "qc_fql":
        return QCFQLAlgorithm(agent, config, writer, verbose)
    raise ValueError(f"Unknown offline algorithm '{name}' (use 'flow_bc' or 'qc_fql').")
