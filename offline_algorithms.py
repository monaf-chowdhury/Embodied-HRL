"""Offline shared skill-conditioned training algorithms.

Two algorithms are intentionally kept:
  * shared_flow_bc_positive -- diagnostic flow BC on positive skill rows only.
  * shared_qc_fql           -- shared task-conditioned QC-FQL on relabeled rows.

Both train one shared model and use chain-context prefix validation for model
selection. Legacy aliases `flow_bc` and `qc_fql` map to the shared variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from demo_dataset import sample_oracle_prefix_states
from env_wrapper import FrankaKitchenImageWrapper
from utils import rng_isolated

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    def trange(*args, **kwargs):
        del kwargs
        return range(*args)


def set_postfix(iterator, **kwargs):
    if hasattr(iterator, "set_postfix"):
        iterator.set_postfix(**kwargs)


def _prefix_tasks_for(agent, task_id: int) -> List[str]:
    return [agent.tasks[k] for k in range(int(task_id))]


@rng_isolated
def _prefix_actor_eval(agent, config, samples: Sequence[Dict[str, np.ndarray]], target_id: int) -> Dict[str, float]:
    """Roll the shared policy from fixed oracle prefix states for one task."""
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

    def scalar(self, tag: str, value: float, step: int):
        if self.writer is not None and np.isfinite(float(value)):
            self.writer.add_scalar(tag, float(value), int(step))

    def should_log(self, step: int, total_steps: int) -> bool:
        return step == 1 or step % self.log_interval == 0 or step == total_steps

    def sample_positive(self, ds) -> Dict[str, np.ndarray]:
        return ds.sample_positive_shared_batch(
            self.config.specialist.batch_size,
            spec=self.agent.spec,
            proprio_normalizer=self.agent.normalize_proprio,
        )

    def sample_relabel(self, ds) -> Dict[str, np.ndarray]:
        return ds.sample_shared_relabel_batch(
            self.config.specialist.batch_size,
            spec=self.agent.spec,
            proprio_normalizer=self.agent.normalize_proprio,
        )

    def _load_prefix_samples(self) -> Dict[int, Tuple[Sequence[Dict[str, np.ndarray]], Dict[str, float]]]:
        n_states = int(self.config.specialist.eval_prefix_states)
        if int(self.config.specialist.eval_interval) <= 0 or n_states <= 0:
            return {}
        out = {}
        for task_id, task_name in enumerate(self.agent.tasks):
            try:
                samples, stats = sample_oracle_prefix_states(
                    agent=self.agent,
                    config=self.config,
                    prefix_tasks=_prefix_tasks_for(self.agent, task_id),
                    target_task=task_name,
                    max_states=n_states,
                    verbose=False,
                )
                out[task_id] = (samples, stats)
                if self.verbose:
                    print(
                        f"    prefix-val {task_name:<14} states={len(samples)} "
                        f"matches={int(stats.get('matching_segments', 0.0))}"
                    )
            except Exception as exc:
                if self.verbose:
                    print(f"    prefix-val disabled for {task_name}: {exc}")
        return out

    def _eval_prefixes(
        self,
        prefix_samples: Dict[int, Tuple[Sequence[Dict[str, np.ndarray]], Dict[str, float]]],
        step: int,
    ) -> Dict[str, float]:
        successes, errors = [], []
        out: Dict[str, float] = {}
        for task_id, task_name in enumerate(self.agent.tasks):
            if task_id not in prefix_samples:
                continue
            samples, _stats = prefix_samples[task_id]
            ev = _prefix_actor_eval(self.agent, self.config, samples, task_id)
            safe = task_name.replace(" ", "_")
            self.scalar(f"skill/{safe}/prefix_success_rate", ev["success_rate"], step)
            self.scalar(f"skill/{safe}/prefix_mean_final_error", ev["mean_final_error"], step)
            out[f"prefix_success/{safe}"] = ev["success_rate"]
            out[f"prefix_error/{safe}"] = ev["mean_final_error"]
            successes.append(ev["success_rate"])
            errors.append(ev["mean_final_error"])
        out["prefix_success/mean"] = float(np.mean(successes)) if successes else 0.0
        out["prefix_error/mean"] = float(np.mean(errors)) if errors else float("inf")
        self.scalar("shared/prefix_success_mean", out["prefix_success/mean"], step)
        self.scalar("shared/prefix_error_mean", out["prefix_error/mean"], step)
        return out

    def _finalize_metrics(
        self,
        metrics_list: List[Dict[str, float]],
        best_step: int,
        best_eval: Dict[str, float],
    ) -> AlgorithmResult:
        out: Dict[str, float] = {}
        if metrics_list:
            tail = metrics_list[-min(100, len(metrics_list)):]
            for key in metrics_list[-1].keys():
                out[f"{key}/final"] = float(np.mean([m[key] for m in tail]))
        if best_step > 0:
            out["prefix_success/mean_best"] = float(best_eval.get("prefix_success/mean", 0.0))
            out["prefix_error/mean_best"] = float(best_eval.get("prefix_error/mean", float("inf")))
            out["prefix_best_step"] = float(best_step)
            for task_name in self.agent.tasks:
                safe = task_name.replace(" ", "_")
                if f"prefix_success/{safe}" in best_eval:
                    out[f"prefix_success/{safe}_best"] = float(best_eval[f"prefix_success/{safe}"])
                    out[f"prefix_error/{safe}_best"] = float(best_eval[f"prefix_error/{safe}"])
        return AlgorithmResult(metrics=out)

    def train(self, ds) -> AlgorithmResult:
        raise NotImplementedError


class SharedFlowBCPositiveAlgorithm(OfflineAlgorithm):
    """Diagnostic shared flow BC on positive skill rows only."""

    def train(self, ds) -> AlgorithmResult:
        n_steps = int(self.config.specialist.n_flow_bc_steps)
        if n_steps <= 0:
            return AlgorithmResult(metrics={})
        prefix_samples = self._load_prefix_samples()
        eval_interval = int(self.config.specialist.eval_interval)
        best_score, best_error, best_step = -1.0, float("inf"), 0
        best_snap, best_eval = None, {}
        metrics_list: List[Dict[str, float]] = []

        iterator = trange(n_steps, desc="SharedFlowBC", leave=False, disable=not self.verbose)
        for i in iterator:
            step = i + 1
            metrics = self.agent.flow_bc_step(self.sample_positive(ds))
            metrics_list.append(metrics)
            if prefix_samples and eval_interval > 0 and (step % eval_interval == 0 or step == n_steps):
                ev = self._eval_prefixes(prefix_samples, step)
                score, err = ev["prefix_success/mean"], ev["prefix_error/mean"]
                if score > best_score or (score == best_score and err < best_error):
                    best_score, best_error, best_step = score, err, step
                    best_snap, best_eval = self.agent.snapshot(), ev
            if self.should_log(step, n_steps):
                for key, value in metrics.items():
                    self.scalar(f"shared/{key}", value, step)
            if self.verbose:
                set_postfix(iterator, bc=f"{metrics['flow_bc_loss']:.4f}")

        if best_snap is not None:
            self.agent.restore_snapshot(best_snap)
        return self._finalize_metrics(metrics_list, best_step, best_eval)


class SharedQCFQLAlgorithm(OfflineAlgorithm):
    """Shared task-conditioned Q-chunking + Flow Q-Learning."""

    def train(self, ds) -> AlgorithmResult:
        n_steps = int(self.config.specialist.n_offline_rl_steps)
        if n_steps <= 0:
            return AlgorithmResult(metrics={})
        prefix_samples = self._load_prefix_samples()
        eval_interval = int(self.config.specialist.eval_interval)
        best_score, best_error, best_step = -1.0, float("inf"), 0
        best_snap, best_eval = None, {}
        metrics_list: List[Dict[str, float]] = []

        iterator = trange(n_steps, desc="SharedQC-FQL", leave=False, disable=not self.verbose)
        for i in iterator:
            step = i + 1
            metrics = self.agent.qc_fql_step(
                batch=self.sample_relabel(ds),
                critic_batch=self.sample_relabel(ds),
            )
            metrics_list.append(metrics)
            if prefix_samples and eval_interval > 0 and (step % eval_interval == 0 or step == n_steps):
                ev = self._eval_prefixes(prefix_samples, step)
                score, err = ev["prefix_success/mean"], ev["prefix_error/mean"]
                if score > best_score or (score == best_score and err < best_error):
                    best_score, best_error, best_step = score, err, step
                    best_snap, best_eval = self.agent.snapshot(), ev
            if self.should_log(step, n_steps):
                for key, value in metrics.items():
                    self.scalar(f"shared/{key}", value, step)
            if self.verbose:
                set_postfix(
                    iterator,
                    q=f"{metrics['qc_critic_loss']:.3f}",
                    distill=f"{metrics['qc_distill_loss']:.4f}",
                )

        if best_snap is not None:
            self.agent.restore_snapshot(best_snap)
        return self._finalize_metrics(metrics_list, best_step, best_eval)


def make_offline_algorithm(name: str, agent, config, writer=None, verbose: bool = True) -> OfflineAlgorithm:
    key = name.lower().replace("-", "_")
    if key in {"flow_bc", "shared_flow_bc_positive"}:
        return SharedFlowBCPositiveAlgorithm(agent, config, writer, verbose)
    if key in {"qc_fql", "shared_qc_fql"}:
        return SharedQCFQLAlgorithm(agent, config, writer, verbose)
    raise ValueError(
        f"Unknown offline algorithm '{name}' "
        "(use 'shared_qc_fql' or 'shared_flow_bc_positive')."
    )
