"""
train.py — SMGW training: Stage A (task-grounded warmup) -> Stage B (joint HRL).

Usage:
    python train.py [--seed 42] [--device cuda] [--log_dir logs/run1]
    python train.py --bc_only --demo_datasets franka-complete franka-mixed franka-partial
    python train.py --total_steps 1500000 --action_chunk 4

Design notes:
  - Stage A: warmup.run_stage_a_warmup loads offline demos, replays them
    through MuJoCo for rendering, BC-trains the worker actor, and optionally
    CE-trains the manager with demo-derived labels.
  - Stage B: for every env step, (a) manager picks a task from REMAINING
    tasks, (b) worker runs an option (chunks of H actions) with dense
    task-grounded rewards, (c) worker SAC updates online, (d) when the
    option ends, the manager observes its option-level return and
    does a DQN-style update.
  - Checkpoints: evenly spaced across total_env_steps (configurable),
    plus a "best" checkpoint on full_task_success_rate. Each checkpoint
    saves model weights AND records N evaluation videos so you can
    visually inspect what the policy does at that point.
"""
from __future__ import annotations

import os
import sys
import time
import random
import datetime
import argparse
import numpy as np
import torch
from typing import Dict, List, Optional, Sequence
from torch.utils.tensorboard import SummaryWriter

from config import Config
from env_wrapper import FrankaKitchenImageWrapper
from agent import SMGWAgent
from demo_dataset import sample_oracle_prefix_states
from specialist import SpecialistSkillAgent, run_specialist_stage_a_warmup
from warmup import run_stage_a_warmup
from utils import save_video, format_time, format_steps


# =============================================================================
# Stdout tee logger
# =============================================================================

class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"train_log_{ts}.txt")
        self.terminal = sys.stdout
        self.file = open(path, "w", buffering=1)
        self.log_path = path
        sys.stdout = self

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.terminal
        self.file.close()


# =============================================================================
# Checkpoint scheduler
# =============================================================================

class CheckpointScheduler:
    def __init__(self, total_steps: int, n_checkpoints: int):
        self.thresholds = [
            int((i + 1) * total_steps / n_checkpoints)
            for i in range(n_checkpoints)
        ]
        self.next_idx = 0

    def should_save(self, steps: int) -> bool:
        if self.next_idx >= len(self.thresholds):
            return False
        if steps >= self.thresholds[self.next_idx]:
            self.next_idx += 1
            return True
        return False

    def label(self, steps: int) -> str:
        return f"periodic_{self.next_idx:02d}_step{steps}"

# =============================================================================
# Evaluator
# =============================================================================

def evaluate(agent: SMGWAgent,
             config: Config,
             n_episodes: Optional[int] = None,
             deterministic: bool = True,
             record_dir: Optional[str] = None,
             n_videos: int = 0) -> Dict[str, float]:
    """
    Run `n_episodes` deterministic evaluation episodes.

    If `record_dir` is set, save the first `n_videos` episodes as MP4s.
    Metrics produced (per evaluation run):
    * any_task_success_rate     — fraction of eps where ≥1 task completed
    * full_task_success_rate    — fraction of eps where ALL tasks completed  (MAIN metric)
    * mean_tasks_completed      — average number of tasks completed per ep
    * mean_options_used         — average number of task attempts / options
    * mean_chosen_task_success  — how often the chosen task actually got its bit flipped
    * mean_env_reward           — raw env reward per ep (for comparison with baselines)

    Chained success is what we really care about: full_task_success_rate.

    """
    n_eps = n_episodes or config.eval.n_eval_episodes
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
    )

    any_success = []
    full_success = []
    tasks_completed = []
    options_used = []
    chosen_task_successes = []
    env_rewards = []
    termination_reasons: Dict[str, int] = {}

    for ep_i in range(n_eps):
        collect_frames = (record_dir is not None and ep_i < n_videos)
        frames_accum = []

        img, state = env.reset(seed=config.training.seed + 10_000 + ep_i)
        z = agent.encoder.encode_numpy(img).squeeze()
        proprio = state.copy()
        completion = np.zeros(agent.n_tasks, dtype=np.float32)

        if collect_frames:
            frames_accum.append(img.copy())

        done = False
        n_options = 0
        chosen_success_count = 0
        ep_env_reward = 0.0

        while (not done
               and n_options < config.manager.max_high_level_steps
               and completion.sum() < agent.n_tasks):

            if config.training.mode == "hierarchical":
                task_id = agent.select_task(z, proprio, state, completion,
                                            deterministic=deterministic)
            else:
                task_id = select_flat_task(agent, config, completion)

            result = agent.execute_option(
                env=env, task_id=task_id,
                start_img=img, start_state=state, start_z=z,
                completion=completion,
                deterministic_worker=deterministic,
                collect_frames=collect_frames,
                train_worker_online=False,
            )

            # Advance episode state from option result
            state = result.proprio_end
            proprio = state
            z = result.z_end
            completion = result.completion_end
            ep_env_reward += result.env_reward_sum
            done = result.env_done
            n_options += 1
            if result.chosen_task_completed:
                chosen_success_count += 1

            reason = result.termination_reason
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

            if collect_frames:
                # append frames from this option (skip the first frame to
                # avoid duplicates — it's the same as the previous last frame)
                if result.frames:
                    frames_accum.extend(result.frames[1:])

            # The env doesn't give us an updated "img" field back from
            # execute_option because we pushed frames out separately. To
            # keep the outer loop in sync, re-render:
            img = env.render_image()

        n_done = int(completion.sum())
        any_success.append(n_done >= 1)
        full_success.append(n_done >= agent.n_tasks)
        tasks_completed.append(n_done)
        options_used.append(n_options)
        chosen_task_successes.append(chosen_success_count / max(n_options, 1))
        env_rewards.append(ep_env_reward)

        if collect_frames and frames_accum:
            out_path = os.path.join(record_dir, f"ep_{ep_i:03d}.mp4")
            save_video(frames_accum, out_path, fps=config.training.video_fps)

    env.close()

    return {
        'any_task_success_rate': float(np.mean(any_success)),
        'full_task_success_rate': float(np.mean(full_success)),
        'mean_tasks_completed': float(np.mean(tasks_completed)),
        'mean_options_used': float(np.mean(options_used)),
        'mean_chosen_task_success': float(np.mean(chosen_task_successes)),
        'mean_env_reward': float(np.mean(env_rewards)),
        'std_env_reward': float(np.std(env_rewards)),
        'n_episodes': n_eps,
        'termination_reasons': termination_reasons,
    }


def evaluate_single_task_worker(agent: SMGWAgent,
                                config: Config,
                                n_episodes: Optional[int] = None,
                                deterministic: bool = True,
                                record_dir: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate the worker alone by repeatedly commanding a single task until
    success or episode termination. This is the first real post-BC signal.

    If `record_dir` is set, save the best rollout for each task:
      - best successful rollout if any succeeded
      - otherwise the closest / best-reward attempt
    """
    n_eps = n_episodes or config.eval.n_single_task_episodes
    results: Dict[str, float] = {}
    report_lines = [
        "Stage A Single-Task Evaluation",
        f"episodes_per_task={n_eps}",
        "",
    ]

    print(SEP2)
    print("  STAGE A EVAL  —  Deterministic worker, one commanded task at a time")
    print(SEP2)

    for task_id, task_name in enumerate(agent.tasks):
        env = FrankaKitchenImageWrapper(
            tasks_to_complete=[task_name],
            img_size=config.encoder.img_size,
            terminate_on_tasks_completed=True,
        )
        success = []
        option_counts = []
        env_rewards = []
        best_rollout = None

        for ep_idx in range(n_eps):
            img, state = env.reset(
                seed=config.training.seed + 20_000 + 1000 * task_id + ep_idx
            )
            z = agent.encoder.encode_numpy(img).squeeze()
            completion = np.zeros(agent.n_tasks, dtype=np.float32)
            collect_frames = record_dir is not None
            frames_accum = [img.copy()] if collect_frames else []

            done = False
            ep_success = False
            ep_options = 0
            ep_env_reward = 0.0

            while not done and not ep_success and ep_options < config.manager.max_high_level_steps:
                result = agent.execute_option(
                    env=env,
                    task_id=task_id,
                    start_img=img,
                    start_state=state,
                    start_z=z,
                    completion=completion,
                    deterministic_worker=deterministic,
                    collect_frames=collect_frames,
                    train_worker_online=False,
                )
                state = result.proprio_end
                z = result.z_end
                completion = result.completion_end
                done = result.env_done
                ep_options += 1
                ep_env_reward += result.env_reward_sum
                ep_success = bool(
                    result.chosen_task_completed
                    or completion[task_id] > 0.5
                    or agent.spec.is_close(state, task_id)
                )
                if collect_frames and result.frames:
                    frames_accum.extend(result.frames[1:])
                if not done:
                    img = env.render_image()

            success.append(float(ep_success))
            option_counts.append(float(ep_options))
            env_rewards.append(float(ep_env_reward))
            final_error = agent.spec.task_error(state, task_id)

            if collect_frames and frames_accum:
                score = (
                    1 if ep_success else 0,
                    -float(final_error),
                    float(ep_env_reward),
                    -float(ep_options),
                )
                if best_rollout is None or score > best_rollout["score"]:
                    best_rollout = {
                        "score": score,
                        "frames": list(frames_accum),
                        "success": bool(ep_success),
                        "reward": float(ep_env_reward),
                        "final_error": float(final_error),
                        "options": int(ep_options),
                        "episode_index": int(ep_idx),
                    }

        env.close()

        sr = float(np.mean(success))
        mean_opts = float(np.mean(option_counts))
        mean_reward = float(np.mean(env_rewards))
        safe_task = task_name.lower().replace(" ", "_")
        results[f"single_task/{safe_task}_success_rate"] = sr
        results[f"single_task/{safe_task}_mean_options"] = mean_opts
        results[f"single_task/{safe_task}_mean_env_reward"] = mean_reward

        print(f"  {task_name:<14} success={sr*100:5.1f}%  "
              f"mean_options={mean_opts:4.1f}  env_reward={mean_reward:7.3f}")

        if record_dir is not None and best_rollout is not None:
            task_dir = os.path.join(record_dir, safe_task)
            os.makedirs(task_dir, exist_ok=True)
            video_name = 'best_success.mp4' if best_rollout["success"] else 'best_attempt.mp4'
            video_path = os.path.join(task_dir, video_name)
            save_video(best_rollout["frames"], video_path, fps=config.training.video_fps)
            report_lines.append(
                f"{task_name}: success_rate={sr:.4f} mean_options={mean_opts:.3f} "
                f"mean_env_reward={mean_reward:.4f} saved={video_path} "
                f"best_episode={best_rollout['episode_index']} "
                f"best_success={int(best_rollout['success'])} "
                f"best_final_error={best_rollout['final_error']:.6f} "
                f"best_options={best_rollout['options']} "
                f"best_reward={best_rollout['reward']:.4f}"
            )

    success_keys = [k for k in results if k.endswith("_success_rate")]
    results["single_task/mean_success_rate"] = float(
        np.mean([results[k] for k in success_keys])
    ) if success_keys else 0.0
    print(SEP3)
    print(f"  Mean single-task success: {results['single_task/mean_success_rate']*100:5.1f}%")
    print(SEP2 + "\n")

    if record_dir is not None:
        os.makedirs(record_dir, exist_ok=True)
        report_path = os.path.join(record_dir, 'summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"  [Stage A Video] Saved best single-task rollouts -> {record_dir}")
        print(f"  [Stage A Video] Summary -> {report_path}")

    return results


def evaluate_oracle_prefix_states(agent: SMGWAgent,
                                  config: Config,
                                  prefix_tasks: Sequence[str],
                                  target_task: str,
                                  n_states: int = 20,
                                  deterministic: bool = True,
                                  record_dir: Optional[str] = None,
                                  n_videos: int = 0) -> Dict[str, float]:
    """
    Oracle prefix-state evaluation:
      - sample demo states where exactly `prefix_tasks` are complete
      - the replay-labelled active target is `target_task`
      - run the frozen policy on `target_task` only from those states
    """
    samples, sample_stats = sample_oracle_prefix_states(
        agent=agent,
        config=config,
        prefix_tasks=prefix_tasks,
        target_task=target_task,
        max_states=n_states,
        verbose=True,
    )

    target_id = agent.tasks.index(target_task)
    prefix_ids = [agent.tasks.index(name) for name in prefix_tasks]
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
    )

    success = []
    option_counts = []
    env_rewards = []
    final_errors = []
    best_errors = []
    prefix_preserved = []
    offtask_completion = []
    termination_reasons: Dict[str, int] = {}
    report_lines = [
        "Stage A Oracle Prefix-State Evaluation",
        f"target_task={target_task}",
        f"prefix_tasks={list(prefix_tasks)}",
        f"n_states={len(samples)}",
        "",
    ]

    for sample_idx, sample in enumerate(samples):
        collect_frames = record_dir is not None and sample_idx < n_videos
        env.reset(seed=config.training.seed + 30_000 + sample_idx)
        qpos, qvel = env.observation_to_qpos_qvel(sample["state"])
        env.set_mujoco_state(qpos, qvel)
        env._current_obs = {"observation": np.asarray(sample["state"], dtype=np.float64).copy()}
        env._step_count = 0

        state = np.asarray(sample["state"], dtype=np.float64).copy()
        img = env.render_image()
        z = agent.encoder.encode_numpy(img).squeeze()
        completion = np.asarray(sample["completion"], dtype=np.float32).copy()

        frames_accum = [img.copy()] if collect_frames else []
        done = False
        ep_success = False
        ep_options = 0
        ep_env_reward = 0.0
        best_err = float(agent.spec.task_error(state, target_id))

        while not done and not ep_success and ep_options < config.manager.max_high_level_steps:
            result = agent.execute_option(
                env=env,
                task_id=target_id,
                start_img=img,
                start_state=state,
                start_z=z,
                completion=completion,
                deterministic_worker=deterministic,
                collect_frames=collect_frames,
                train_worker_online=False,
            )
            state = result.proprio_end
            z = result.z_end
            completion = result.completion_end
            done = result.env_done
            ep_options += 1
            ep_env_reward += result.env_reward_sum
            cur_err = float(agent.spec.task_error(state, target_id))
            best_err = min(best_err, cur_err)
            ep_success = bool(
                result.chosen_task_completed
                or completion[target_id] > 0.5
                or agent.spec.is_close(state, target_id)
            )
            termination_reasons[result.termination_reason] = (
                termination_reasons.get(result.termination_reason, 0) + 1
            )
            if collect_frames and result.frames:
                frames_accum.extend(result.frames[1:])
            if not done:
                img = env.render_image()

        final_err = float(agent.spec.task_error(state, target_id))
        prefix_ok = bool(np.all(completion[prefix_ids] > 0.5)) if prefix_ids else True
        extra_complete = [
            agent.tasks[k] for k in range(agent.n_tasks)
            if k not in prefix_ids and k != target_id and completion[k] > 0.5
        ]

        success.append(float(ep_success))
        option_counts.append(float(ep_options))
        env_rewards.append(float(ep_env_reward))
        final_errors.append(final_err)
        best_errors.append(best_err)
        prefix_preserved.append(float(prefix_ok))
        offtask_completion.append(float(len(extra_complete) > 0))

        if collect_frames and frames_accum:
            os.makedirs(record_dir, exist_ok=True)
            sample_name = f"sample_{sample_idx:03d}"
            suffix = "success" if ep_success else "fail"
            video_path = os.path.join(record_dir, f"{sample_name}_{suffix}.mp4")
            save_video(frames_accum, video_path, fps=config.training.video_fps)

        report_lines.append(
            f"sample={sample_idx:03d} success={int(ep_success)} options={ep_options} "
            f"final_error={final_err:.6f} best_error={best_err:.6f} "
            f"prefix_preserved={int(prefix_ok)} offtask_completed={int(len(extra_complete) > 0)} "
            f"dataset={str(sample['dataset_id'])} episode={str(sample['episode_id'])} "
            f"step={int(sample['step_index'])}"
        )

    env.close()

    result = {
        "prefix_eval/success_rate": float(np.mean(success)) if success else 0.0,
        "prefix_eval/mean_options": float(np.mean(option_counts)) if option_counts else 0.0,
        "prefix_eval/mean_env_reward": float(np.mean(env_rewards)) if env_rewards else 0.0,
        "prefix_eval/mean_final_error": float(np.mean(final_errors)) if final_errors else 0.0,
        "prefix_eval/mean_best_error": float(np.mean(best_errors)) if best_errors else 0.0,
        "prefix_eval/prefix_preservation_rate": float(np.mean(prefix_preserved)) if prefix_preserved else 0.0,
        "prefix_eval/offtask_completion_rate": float(np.mean(offtask_completion)) if offtask_completion else 0.0,
        "prefix_eval/n_states": float(len(samples)),
        "prefix_eval/termination_reasons": termination_reasons,
    }
    for k, v in sample_stats.items():
        result[f"prefix_eval_sampler/{k}"] = float(v)

    if record_dir is not None:
        os.makedirs(record_dir, exist_ok=True)
        report_path = os.path.join(record_dir, "summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        result["prefix_eval/report_path"] = report_path

    return result

# =============================================================================
# Save checkpoint and record eval videos
# =============================================================================

def save_checkpoint_and_videos(agent: SMGWAgent, config: Config, label: str):
    ckpt_dir = os.path.join(config.training.log_dir, 'checkpoints')
    video_dir = os.path.join(config.training.log_dir, 'videos', label)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{label}.pt')
    agent.save(ckpt_path)
    print(f"  [Checkpoint] Saved -> {os.path.basename(ckpt_path)}")

    if config.training.record_video:
        os.makedirs(video_dir, exist_ok=True)
        n = config.training.video_n_episodes
        print(f"  [Video] Recording {n} eval episodes at '{label}'...")
        result = evaluate(agent, config,
                          n_episodes=n, deterministic=True,
                          record_dir=video_dir, n_videos=n)
        print(f"  [Video] Saved to {video_dir}/")
        print(f"  [Video] Any-task: {result['any_task_success_rate']*100:.1f}%  "
              f"Full-task: {result['full_task_success_rate']*100:.1f}%  "
              f"Mean tasks done: {result['mean_tasks_completed']:.2f}/{agent.n_tasks}")


def derive_stage_a_task_success(agent: SMGWAgent,
                                single_task_eval: Dict[str, float]) -> np.ndarray:
    rates = []
    for task_name in agent.tasks:
        safe_task = task_name.lower().replace(" ", "_")
        rates.append(single_task_eval.get(f"single_task/{safe_task}_success_rate", 0.0))
    return np.asarray(rates, dtype=np.float32)


def scripted_manager_prob(config: Config, env_steps: int) -> float:
    horizon = max(int(config.training.scripted_manager_steps), 0)
    if horizon <= 0 or env_steps >= horizon:
        return 0.0
    frac = env_steps / max(horizon, 1)
    start = float(config.training.scripted_manager_prob_start)
    end = float(config.training.scripted_manager_prob_end)
    return max(0.0, start + frac * (end - start))


def task_order_for_controller(agent: SMGWAgent, config: Config) -> list[int]:
    if config.training.controller_order_mode == "stage_a_rank":
        return list(agent.curriculum_task_order)
    return list(range(agent.n_tasks))


def unlocked_task_ids(agent: SMGWAgent,
                      config: Config,
                      completion: np.ndarray) -> list[int]:
    remaining = [k for k in range(agent.n_tasks) if completion[k] < 0.5]
    if agent.total_env_steps >= config.training.unlock_remaining_tasks_steps:
        return remaining

    threshold = float(config.training.min_stage_a_task_success)
    supported = [
        k for k in remaining
        if float(agent.stage_a_task_success[k]) >= threshold
    ]
    return supported if supported else remaining


def select_stage_b_task(agent: SMGWAgent,
                        config: Config,
                        z: np.ndarray,
                        proprio: np.ndarray,
                        state: np.ndarray,
                        completion: np.ndarray) -> tuple[int, bool, float]:
    prob = scripted_manager_prob(config, agent.total_env_steps)
    unlocked = unlocked_task_ids(agent, config, completion)
    if config.training.scripted_manager_mode == "stage_a_rank":
        candidate_order = agent.curriculum_task_order
    else:
        candidate_order = list(range(agent.n_tasks))
    remaining = [k for k in candidate_order if k in unlocked]

    if prob > 0.0 and remaining and np.random.random() < prob:
        return int(remaining[0]), True, prob

    gated_completion = completion.copy()
    for k in range(agent.n_tasks):
        if k not in unlocked:
            gated_completion[k] = 1.0

    return (
        int(agent.select_task(z, proprio, state, gated_completion, deterministic=False)),
        False,
        prob,
    )


def select_flat_task(agent: SMGWAgent,
                     config: Config,
                     completion: np.ndarray) -> int:
    unlocked = unlocked_task_ids(agent, config, completion)
    controller_order = task_order_for_controller(agent, config)
    remaining = [k for k in controller_order if k in unlocked and completion[k] < 0.5]
    if remaining:
        return int(remaining[0])
    fallback = [k for k in controller_order if completion[k] < 0.5]
    if fallback:
        return int(fallback[0])
    return 0


# =============================================================================
# Pretty-printed banners
# =============================================================================

SEP = "=" * 76
SEP2 = "-" * 76
SEP3 = "." * 76


def print_start_banner(config: Config, log_path: str):
    print(f"\n{SEP}")
    print(f"  SMGW — Semantic Manager, Grounded Worker")
    print(f"  FrankaKitchen-v1")
    print(f"  Started : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(SEP)
    print(f"  Encoder        : {config.encoder.name.upper()}  "
          f"({config.encoder.raw_dim}-d features, FROZEN)")
    print(f"  Tasks          : {config.training.tasks_to_complete}")
    if config.training.mode == "hierarchical":
        print(f"  High-Level     : learned manager, completion-mask gated")
    else:
        print(f"  High-Level     : scripted controller ({config.training.controller_order_mode})")
    if config.training.mode == "specialist_skills":
        print(f"  Worker         : per-skill visual students  "
              f"(chunk_len = {config.worker.action_chunk_len})")
    else:
        print(f"  Worker         : SAC, FiLM-conditioned, "
              f"chunk_len = {config.worker.action_chunk_len}")
    print(f"  Subgoal K      : {config.manager.subgoal_horizon} env steps / option")
    print(f"  Max HL steps   : {config.manager.max_high_level_steps} per episode")
    print(SEP2)
    print(f"  Training mode  : {config.training.mode}")
    if config.training.mode == "hierarchical":
        print(f"  Manager reward : completion={config.manager.completion_bonus}  "
              f"offtask={config.manager.offtask_completion_bonus}  "
              f"dense={config.manager.dense_shaping_weight}  "
              f"option_cost={config.manager.option_cost}  "
              f"demo_ce_weight={config.manager.online_demo_ce_weight}  "
              f"all_done={config.manager.all_done_bonus}")
    print(f"  Worker reward  : progress={config.worker.progress_weight}  "
          f"completion={config.worker.completion_bonus}  "
          f"action_cost={config.worker.action_cost}  "
          f"failure={config.worker.failure_penalty}")
    if config.training.mode == "specialist_skills":
        print(f"  Chaining stage  : scripted controller over remaining tasks  "
              f"(order={config.training.controller_order_mode})")
        print(f"  Online stage    : conservative per-skill fine-tuning scaffold  "
              f"(currently frozen eval by default)")
    else:
        print(f"  Stage B stabil.: deterministic_rollout_steps={config.training.deterministic_worker_rollout_steps:,}  "
              f"demo_bc_weight={config.worker.online_demo_bc_weight}  "
              f"demo_bc_steps={config.worker.online_demo_bc_steps:,}  "
              f"demo_mix={config.worker.online_demo_mix_ratio_start:.2f}->{config.worker.online_demo_mix_ratio_end:.2f}  "
              f"worker_update_start={config.training.worker_update_start_steps:,}")
        print(f"  Stage B curric.: freeze_manager_steps={config.training.manager_freeze_steps:,}  "
              f"scripted_manager_steps={config.training.scripted_manager_steps:,}  "
              f"scripted_prob={config.training.scripted_manager_prob_start:.2f}->{config.training.scripted_manager_prob_end:.2f}  "
              f"mode={config.training.scripted_manager_mode}  "
              f"min_stage_a_success={config.training.min_stage_a_task_success:.2f}  "
              f"unlock_all={config.training.unlock_remaining_tasks_steps:,}  "
              f"controller={config.training.controller_order_mode}")
    print(f"  NOTE: Zero latent-distance terms anywhere. All rewards are "
          f"grounded in benchmark completion bits and task-space errors.")
    print(SEP2)
    print(f"  Stage A demos  : {config.warmup.dataset_ids}")
    print(f"  Stage A cache  : {config.warmup.cache_dir}  "
          f"(rebuild={config.warmup.rebuild_cache})")
    if config.training.mode == "specialist_skills":
        print(f"  Stage A skill  : teacher_bc={config.specialist.n_teacher_bc_steps}  "
              f"teacher_iql={config.specialist.n_teacher_iql_steps}  "
              f"teacher_online={config.specialist.n_teacher_online_steps}  "
              f"student_distill={config.specialist.n_student_distill_steps}  "
              f"rollout_distill={config.specialist.n_student_rollout_distill_steps}  "
              f"dagger={config.specialist.n_student_dagger_steps}  "
              f"(batch={config.specialist.batch_size})")
        print(f"  Teacher reward : task={config.specialist.teacher_online_reward_task_weight:.2f}  "
              f"approach={config.specialist.teacher_online_reward_approach_weight:.2f}  "
              f"completion={config.specialist.teacher_online_reward_completion:.2f}  "
              f"action_cost={config.specialist.teacher_online_reward_action_cost:.4f}")
        print(f"  Teacher policy : affordance-aware privileged input  "
              f"residual_online={config.specialist.teacher_residual_online}  "
              f"residual_scale={config.specialist.teacher_residual_scale:.3f}  "
              f"online_noise={config.specialist.teacher_online_exploration_std:.3f}")
    else:
        print(f"  Stage A BC/IQL : worker_bc={config.warmup.n_worker_sl_steps}  "
              f"worker_iql={config.warmup.n_worker_iql_steps}  "
              f"manager_ce={config.warmup.n_manager_sl_steps}  "
              f"(batch={config.warmup.sl_batch_size})")
    if config.warmup.focus_task:
        print(f"  Stage A focus  : task='{config.warmup.focus_task}'  "
              f"weight={config.warmup.focus_task_weight:.2f}  "
              f"tail_start={config.warmup.focus_task_tail_start:.2f}  "
              f"tail_weight={config.warmup.focus_task_tail_weight:.2f}")
        if config.warmup.n_focus_refine_steps > 0:
            print(f"  Stage A refine : steps={config.warmup.n_focus_refine_steps}  "
                  f"min_seg_prog={config.warmup.focus_refine_min_seg_prog:.2f}")
    print(f"  Stage B steps  : {config.training.total_env_steps:,}")
    print(f"  Batch size     : {config.buffer.batch_size}")
    print(f"  Buffer cap     : worker={config.buffer.worker_capacity:,}  "
          f"manager={config.buffer.manager_capacity:,}")
    print(f"  Seed           : {config.training.seed}")
    print(f"  Device         : {config.training.device}")
    if torch.cuda.is_available():
        print(f"  GPU            : {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM)")
    print(f"  Log dir        : {config.training.log_dir}")
    print(f"  Train log      : {log_path}")
    print(f"{SEP}\n")


# =============================================================================
# Main training
# =============================================================================

def train(config: Config):
    logger = Logger(config.training.log_dir)

    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)
        torch.cuda.manual_seed_all(config.training.seed)
    if config.training.deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    writer = SummaryWriter(config.training.log_dir)
    ckpt_sched = CheckpointScheduler(
        config.training.total_env_steps,
        config.training.n_periodic_checkpoints,
    )

    # -- Build env and agent --
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        seed=config.training.seed,
        terminate_on_tasks_completed=False,
    )
    if config.training.mode == "specialist_skills":
        agent = SpecialistSkillAgent(config)
    else:
        agent = SMGWAgent(config)

    if config.training.mode != "hierarchical":
        config.warmup.n_manager_sl_steps = 0

    print_start_banner(config, logger.log_path)

    # =========================================================================
    # Stage A: task-grounded warmup
    # =========================================================================
    print(SEP2)
    print(f"  STAGE A  —  Task-Grounded Warmup")
    print(SEP2 + "\n")

    t0 = time.time()
    if config.training.mode == "specialist_skills":
        warmup_stats = run_specialist_stage_a_warmup(agent, config, verbose=True)
    else:
        warmup_stats = run_stage_a_warmup(agent, config, verbose=True)
    warmup_elapsed = time.time() - t0
    print(f"\n  Stage A complete in {format_time(warmup_elapsed)}.")
    for k, v in warmup_stats.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(f'warmup/{k}', v, 0)

    stage_a_video_dir = None
    if config.training.record_video:
        stage_a_video_dir = os.path.join(
            config.training.log_dir, 'videos', 'stage_a_single_task'
        )
    single_task_eval = evaluate_single_task_worker(
        agent,
        config,
        deterministic=config.eval.deterministic_worker,
        record_dir=stage_a_video_dir,
    )
    # Stage A updates the online networks directly. Before Stage B starts, the
    # target networks must be synchronized to the warmed-up weights; otherwise
    # online TD updates bootstrap against stale pre-warmup targets.
    if config.training.mode != "specialist_skills":
        agent.manager_target.load_state_dict(agent.manager.state_dict())
        agent.worker_critic_target.load_state_dict(agent.worker_critic.state_dict())
    agent.stage_a_task_success = derive_stage_a_task_success(agent, single_task_eval)
    agent.curriculum_task_order = list(np.argsort(-agent.stage_a_task_success))
    for k, v in single_task_eval.items():
        writer.add_scalar(f'stage_a_eval/{k}', v, 0)
    controller_order = task_order_for_controller(agent, config)
    order_names = [agent.tasks[k] for k in controller_order]
    order_scores = [float(agent.stage_a_task_success[k]) for k in controller_order]
    print(f"  [Controller] order = {order_names}")
    print(f"  [Controller] stage-A success = {[round(s, 3) for s in order_scores]}")

    if config.training.prefix_eval_only:
        target_task = config.training.prefix_target_task.strip()
        prefix_tasks = list(config.training.prefix_condition_tasks)
        if not target_task:
            raise ValueError(
                "prefix_eval_only requires config.training.prefix_target_task "
                "(or --prefix_target_task on the CLI)."
            )
        prefix_video_dir = None
        if config.training.record_video:
            safe_target = target_task.lower().replace(" ", "_")
            safe_prefix = "__".join(t.lower().replace(" ", "_") for t in prefix_tasks) or "reset"
            prefix_video_dir = os.path.join(
                config.training.log_dir,
                "videos",
                f"prefix_eval_{safe_target}__from__{safe_prefix}",
            )
        print(SEP2)
        print("  STAGE A PREFIX EVAL  —  Oracle prefix states, frozen worker")
        print(SEP2)
        print(f"  Target task         : {target_task}")
        print(f"  Prefix tasks        : {prefix_tasks}")
        prefix_eval = evaluate_oracle_prefix_states(
            agent=agent,
            config=config,
            prefix_tasks=prefix_tasks,
            target_task=target_task,
            n_states=config.training.prefix_eval_n_states,
            deterministic=True,
            record_dir=prefix_video_dir,
            n_videos=config.training.video_n_episodes if config.training.record_video else 0,
        )
        for k, v in prefix_eval.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(k, v, 0)
        print(f"  Success rate        : {prefix_eval['prefix_eval/success_rate']*100:5.1f}%")
        print(f"  Mean options        : {prefix_eval['prefix_eval/mean_options']:.2f}")
        print(f"  Mean final error    : {prefix_eval['prefix_eval/mean_final_error']:.6f}")
        print(f"  Mean best error     : {prefix_eval['prefix_eval/mean_best_error']:.6f}")
        print(f"  Prefix preserved    : {prefix_eval['prefix_eval/prefix_preservation_rate']*100:5.1f}%")
        print(f"  Off-task completion : {prefix_eval['prefix_eval/offtask_completion_rate']*100:5.1f}%")
        print(f"  Sampled states      : {int(prefix_eval['prefix_eval/n_states'])}")
        print(f"  Termination mix     : {prefix_eval['prefix_eval/termination_reasons']}")
        if "prefix_eval/report_path" in prefix_eval:
            print(f"  Report              : {prefix_eval['prefix_eval/report_path']}")
        print(SEP2)
        print(f"{SEP}\n")
        writer.close()
        env.close()
        print(f"\n{SEP}")
        print("  STAGE A PREFIX EVAL COMPLETE")
        print(SEP2)
        print(f"  Mean single-task SR  {single_task_eval.get('single_task/mean_success_rate', 0.0)*100:.1f}%")
        print(f"  Prefix success rate  {prefix_eval.get('prefix_eval/success_rate', 0.0)*100:.1f}%")
        print(f"  Log file             {logger.log_path}")
        print(f"{SEP}\n")
        logger.close()
        return

    if config.training.scripted_eval_only:
        scripted_video_dir = None
        if config.training.record_video:
            scripted_video_dir = os.path.join(
                config.training.log_dir, 'videos', 'stage_a_scripted_chain'
            )
        print(SEP2)
        print("  STAGE A CHAIN EVAL  —  Frozen worker, scripted chaining")
        print(SEP2)
        chain_eval = evaluate(
            agent,
            config,
            deterministic=True,
            record_dir=scripted_video_dir,
            n_videos=config.training.video_n_episodes if config.training.record_video else 0,
        )
        for k, v in chain_eval.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'stage_a_chain_eval/{k}', v, 0)
        print(f"  Any-task success    {chain_eval['any_task_success_rate']*100:5.1f}%")
        print(f"  Full-task success   {chain_eval['full_task_success_rate']*100:5.1f}%")
        print(f"  Mean tasks done     {chain_eval['mean_tasks_completed']:.2f} / {agent.n_tasks}")
        print(f"  Mean options used   {chain_eval['mean_options_used']:.1f}")
        print(f"  Chosen-task SR      {chain_eval['mean_chosen_task_success']*100:5.1f}%")
        print(f"  Mean env reward     {chain_eval['mean_env_reward']:.4f}  "
              f"+-  {chain_eval['std_env_reward']:.4f}")
        print(f"  Termination mix     {chain_eval['termination_reasons']}")
        print(SEP2)
        print(f"{SEP}\n")
        writer.close()
        env.close()
        print(f"\n{SEP}")
        print("  STAGE A SCRIPTED EVAL COMPLETE")
        print(SEP2)
        print(f"  Mean single-task SR   {single_task_eval.get('single_task/mean_success_rate', 0.0)*100:.1f}%")
        print(f"  Scripted full-task SR {chain_eval.get('full_task_success_rate', 0.0)*100:.1f}%")
        print(f"  Log file              {logger.log_path}")
        print(f"{SEP}\n")
        logger.close()
        return

    print(f"{SEP}\n")

    if config.training.stage_a_only:
        writer.close()
        env.close()
        print(f"\n{SEP}")
        print("  STAGE A ONLY RUN COMPLETE")
        print(SEP2)
        if config.training.mode == "specialist_skills":
            print("  Specialist skills  offline teacher/student training complete")
        else:
            print(f"  Worker BC loss     {warmup_stats.get('worker_bc_loss_final', float('nan')):.4f}")
        print(f"  Mean single-task SR {single_task_eval.get('single_task/mean_success_rate', 0.0)*100:.1f}%")
        print(f"  Log file           {logger.log_path}")
        print(f"{SEP}\n")
        logger.close()
        return

    if config.training.mode == "specialist_skills":
        scripted_video_dir = None
        if config.training.record_video:
            scripted_video_dir = os.path.join(
                config.training.log_dir, 'videos', 'stage_a_scripted_chain'
            )
        print(SEP2)
        print("  STAGE A CHAIN EVAL  —  Frozen specialist students, scripted chaining")
        print(SEP2)
        chain_eval = evaluate(
            agent,
            config,
            deterministic=True,
            record_dir=scripted_video_dir,
            n_videos=config.training.video_n_episodes if config.training.record_video else 0,
        )
        for k, v in chain_eval.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'stage_a_chain_eval/{k}', v, 0)
        print(f"  Any-task success    {chain_eval['any_task_success_rate']*100:5.1f}%")
        print(f"  Full-task success   {chain_eval['full_task_success_rate']*100:5.1f}%")
        print(f"  Mean tasks done     {chain_eval['mean_tasks_completed']:.2f} / {agent.n_tasks}")
        print(f"  Mean options used   {chain_eval['mean_options_used']:.1f}")
        print(f"  Chosen-task SR      {chain_eval['mean_chosen_task_success']*100:5.1f}%")
        print(f"  Mean env reward     {chain_eval['mean_env_reward']:.4f}  "
              f"+-  {chain_eval['std_env_reward']:.4f}")
        print(f"  Termination mix     {chain_eval['termination_reasons']}")
        writer.close()
        env.close()
        print(f"\n{SEP}")
        print("  SPECIALIST STAGE A COMPLETE")
        print(SEP2)
        print(f"  Mean single-task SR   {single_task_eval.get('single_task/mean_success_rate', 0.0)*100:.1f}%")
        print(f"  Scripted full-task SR {chain_eval.get('full_task_success_rate', 0.0)*100:.1f}%")
        print(f"  Log file              {logger.log_path}")
        print(f"{SEP}\n")
        logger.close()
        return

    # =========================================================================
    # Stage B: joint HRL training
    # =========================================================================
    print(SEP2)
    if config.training.mode == "hierarchical":
        print(f"  STAGE B  —  Joint HRL (manager DQN + worker SAC)")
    else:
        print(f"  STAGE B  —  Flat Scripted Chaining + Online Worker Fine-Tuning")
    print(SEP2 + "\n")

    best_full_task_sr = -1.0
    train_start = time.time()
    last_worker_losses = {}
    last_manager_losses = {}

    # Rolling windows for terminal logging
    run_env_rewards = []
    run_any_success = []
    run_full_success = []
    run_tasks_completed = []
    run_options_per_ep = []
    run_scripted_choices = []
    run_unlocked_task_counts = []

    # Explicit eval trigger — fire when total_env_steps crosses next_eval_at
    next_eval_at = config.eval.eval_every_env_steps

    # Worker online update cadence: update every N env steps
    updates_per_step = config.training.worker_updates_per_env_step
    update_every_n = max(1, int(round(1.0 / max(updates_per_step, 1e-6))))

    while agent.total_env_steps < config.training.total_env_steps:

        # ---- Episode reset ----
        img, state = env.reset(seed=config.training.seed + agent.total_episodes)
        z = agent.encoder.encode_numpy(img).squeeze()
        proprio = state.copy()
        completion = np.zeros(agent.n_tasks, dtype=np.float32)

        ep_env_reward = 0.0
        ep_tasks_completed = 0
        ep_options = 0
        env_done = False
        scripted_prob = scripted_manager_prob(config, agent.total_env_steps)
        unlocked_now = unlocked_task_ids(agent, config, completion)

        # ---- Option loop ----
        while (not env_done
               and ep_options < config.manager.max_high_level_steps
               and completion.sum() < agent.n_tasks):

            if config.training.mode == "hierarchical":
                task_id, used_scripted_manager, scripted_prob = select_stage_b_task(
                    agent, config, z, proprio, state, completion
                )
                run_scripted_choices.append(float(used_scripted_manager))
            else:
                task_id = select_flat_task(agent, config, completion)
                used_scripted_manager = True
                run_scripted_choices.append(1.0)
            unlocked_now = unlocked_task_ids(agent, config, completion)

            # -- Worker executes the option --
            result = agent.execute_option(
                env=env,
                task_id=task_id,
                start_img=img, start_state=state, start_z=z,
                completion=completion,
                deterministic_worker=(
                    agent.total_env_steps < config.training.deterministic_worker_rollout_steps
                ),
                collect_frames=False,
                train_worker_online=True,
                update_every_n_env_steps=update_every_n,
            )

            if config.training.mode == "hierarchical":
                # -- Write option transition to manager buffer --
                agent.manager_buf.add(
                    z=result.z_start, proprio=result.proprio_start,
                    task_state=result.task_state_start,
                    completion=result.completion_start,
                    action=result.chosen_task,
                    reward=result.option_return,
                    z_next=result.z_end, proprio_next=result.proprio_end,
                    task_state_next=result.task_state_end,
                    completion_next=result.completion_end,
                    done=float(result.env_done
                               or (result.completion_end.sum() >= agent.n_tasks)),
                )
            if result.last_worker_losses:
                last_worker_losses = result.last_worker_losses

            if config.training.mode == "hierarchical":
                # -- Manager online update --
                if agent.total_env_steps >= config.training.manager_freeze_steps:
                    for _ in range(config.training.manager_updates_per_option):
                        loss = agent.update_manager()
                        if loss:
                            last_manager_losses = loss

            if config.training.mode == "hierarchical":
                # -- Update epsilon schedule --
                agent._update_epsilon()

            # -- Advance outer state --
            state = result.proprio_end
            proprio = state
            z = result.z_end
            completion = result.completion_end
            env_done = result.env_done
            ep_env_reward += result.env_reward_sum
            ep_tasks_completed = int(completion.sum())
            ep_options += 1
            agent.total_options += 1

            # We need a fresh image for the next option. env.render_image()
            # re-renders from the current MuJoCo state (no step is taken).
            img = env.render_image()

        # ---- End of episode ----
        agent.total_episodes += 1

        run_env_rewards.append(ep_env_reward)
        run_any_success.append(ep_tasks_completed >= 1)
        run_full_success.append(ep_tasks_completed >= agent.n_tasks)
        run_tasks_completed.append(ep_tasks_completed)
        run_options_per_ep.append(ep_options)
        run_unlocked_task_counts.append(len(unlocked_now))

        # ---- Per-episode TB scalars ----
        if agent.total_episodes % config.training.tb_every_episodes == 0:
            s = agent.total_env_steps
            writer.add_scalar('train/ep_env_reward', ep_env_reward, s)
            writer.add_scalar('train/ep_any_success', float(ep_tasks_completed >= 1), s)
            writer.add_scalar('train/ep_full_success',
                              float(ep_tasks_completed >= agent.n_tasks), s)
            writer.add_scalar('train/ep_tasks_completed', ep_tasks_completed, s)
            writer.add_scalar('train/ep_options', ep_options, s)
            if config.training.mode == "hierarchical":
                writer.add_scalar('train/epsilon', agent.epsilon, s)
                writer.add_scalar('train/scripted_manager_prob', scripted_prob, s)
            writer.add_scalar('train/controller_fraction',
                              float(np.mean(run_scripted_choices)) if run_scripted_choices else 0.0,
                              s)
            writer.add_scalar(
                'train/scripted_manager_fraction',
                float(np.mean(run_scripted_choices)) if run_scripted_choices else 0.0,
                s,
            )
            writer.add_scalar(
                'train/unlocked_task_count',
                float(np.mean(run_unlocked_task_counts)) if run_unlocked_task_counts else 0.0,
                s,
            )
            writer.add_scalar('train/worker_buffer_size', len(agent.worker_buf), s)
            writer.add_scalar('train/manager_buffer_size', len(agent.manager_buf), s)

            for k, v in last_worker_losses.items():
                writer.add_scalar(f'worker/{k}', v, s)
            for k, v in last_manager_losses.items():
                writer.add_scalar(f'manager/{k}', v, s)

        # ---- Terminal logging ----
        if agent.total_episodes % config.training.log_every_episodes == 0 and run_env_rewards:
            elapsed = time.time() - train_start
            sps = agent.total_env_steps / max(elapsed, 1)
            remaining_steps = config.training.total_env_steps - agent.total_env_steps
            eta = remaining_steps / max(sps, 1)
            pct = 100.0 * agent.total_env_steps / config.training.total_env_steps

            print(f"\n{SEP2}")
            print(f"  Step {agent.total_env_steps:>10,} / "
                  f"{config.training.total_env_steps:,} ({pct:5.1f}%)  "
                  f"Ep {agent.total_episodes:,}")
            print(f"  Elapsed {format_time(elapsed)}  ETA {format_time(eta)}  "
                  f"Speed {sps:.0f} steps/s")
            print(SEP3)
            n_win = len(run_env_rewards)
            print(f"  Last {n_win} episodes:")
            print(f"    Env reward        mean={np.mean(run_env_rewards):.3f}  "
                  f"max={np.max(run_env_rewards):.3f}")
            print(f"    Any-task success  {np.mean(run_any_success)*100:5.1f}%")
            print(f"    Full-task success {np.mean(run_full_success)*100:5.1f}%")
            print(f"    Mean tasks done   {np.mean(run_tasks_completed):.2f} / {agent.n_tasks}")
            print(f"    Mean options/ep   {np.mean(run_options_per_ep):.1f}")
            print(SEP3)
            if config.training.mode == "hierarchical":
                print(f"  Exploration  epsilon={agent.epsilon:.3f}")
                print(f"  Curriculum   scripted_prob={scripted_prob:.3f}  "
                      f"scripted_frac={np.mean(run_scripted_choices)*100:5.1f}%  "
                      f"manager_updates={'on' if agent.total_env_steps >= config.training.manager_freeze_steps else 'frozen'}  "
                      f"unlocked={np.mean(run_unlocked_task_counts):.1f}")
                print(f"  Buffers      worker={len(agent.worker_buf):>7,}  "
                      f"manager={len(agent.manager_buf):>6,}")
            else:
                print(f"  Controller   scripted_frac={np.mean(run_scripted_choices)*100:5.1f}%  "
                      f"unlocked={np.mean(run_unlocked_task_counts):.1f}")
                print(f"  Buffers      worker={len(agent.worker_buf):>7,}  "
                      f"manager=     0")
            unlocked_names = [agent.tasks[k] for k in unlocked_task_ids(agent, config, completion)]
            print(f"  Unlocked     {unlocked_names}")
            if last_worker_losses:
                print(f"  Worker (SAC)  critic={last_worker_losses.get('worker_critic_loss', 0):.5f}  "
                      f"actor={last_worker_losses.get('worker_actor_loss', 0):.4f}  "
                      f"alpha={last_worker_losses.get('worker_alpha', 0):.4f}")
            if config.training.mode == "hierarchical" and last_manager_losses:
                print(f"  Manager (DQN) loss={last_manager_losses.get('manager_loss', 0):.5f}  "
                      f"q_mean={last_manager_losses.get('manager_q_mean', 0):.3f}")

            run_env_rewards.clear()
            run_any_success.clear()
            run_full_success.clear()
            run_tasks_completed.clear()
            run_options_per_ep.clear()
            run_scripted_choices.clear()
            run_unlocked_task_counts.clear()

        # ---- Periodic evaluation ----
        if agent.total_env_steps >= next_eval_at:
            next_eval_at = agent.total_env_steps + config.eval.eval_every_env_steps
            t0 = time.time()
            eval_result = evaluate(agent, config,
                                   deterministic=config.eval.deterministic_worker)
            eval_t = time.time() - t0

            is_best = (eval_result['full_task_success_rate'] > best_full_task_sr)
            if is_best and config.training.save_best:
                best_full_task_sr = eval_result['full_task_success_rate']
                save_checkpoint_and_videos(agent, config, 'best')

            for k, v in eval_result.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'eval/{k}', v, agent.total_env_steps)

            badge = "  <- NEW BEST" if is_best else f"  (best so far: {best_full_task_sr*100:.1f}%)"
            print(f"\n{SEP}")
            print(f"  EVALUATION  Step {format_steps(agent.total_env_steps)}  "
                  f"({eval_result['n_episodes']} eps, {eval_t:.1f}s)")
            print(SEP2)
            print(f"  Any-task success    {eval_result['any_task_success_rate']*100:5.1f}%")
            print(f"  Full-task success   {eval_result['full_task_success_rate']*100:5.1f}%{badge}")
            print(f"  Mean tasks done     {eval_result['mean_tasks_completed']:.2f}  /  {agent.n_tasks}")
            print(f"  Mean options used   {eval_result['mean_options_used']:.1f}")
            print(f"  Chosen-task SR      {eval_result['mean_chosen_task_success']*100:5.1f}%")
            print(f"  Mean env reward     {eval_result['mean_env_reward']:.4f}  "
                  f"+-  {eval_result['std_env_reward']:.4f}")
            print(f"  Termination mix     {eval_result['termination_reasons']}")
            print(f"{SEP}\n")

        # ---- Periodic checkpoint ----
        if ckpt_sched.should_save(agent.total_env_steps):
            lbl = ckpt_sched.label(agent.total_env_steps)
            save_checkpoint_and_videos(agent, config, lbl)

    # =========================================================================
    # Training complete
    # =========================================================================
    writer.close()
    total_time = time.time() - train_start

    print(f"\n{SEP}")
    print(f"  TRAINING COMPLETE")
    print(SEP2)
    print(f"  Total time        {format_time(total_time)}")
    print(f"  Total env steps   {agent.total_env_steps:,}")
    print(f"  Total options     {agent.total_options:,}")
    print(f"  Total episodes    {agent.total_episodes:,}")
    print(f"  Best full-task SR {best_full_task_sr*100:.1f}%")
    if config.training.mode == "hierarchical":
        print(f"  Final epsilon     {agent.epsilon:.4f}")
    print(f"  Log file          {logger.log_path}")
    print(f"{SEP}\n")

    # Final checkpoint + video
    save_checkpoint_and_videos(agent, config, 'final')
    logger.close()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMGW — FrankaKitchen-v1")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['specialist_skills', 'flat_scripted', 'hierarchical'])
    parser.add_argument('--total_steps', type=int, default=None,
                        help='Override config.training.total_env_steps')
    parser.add_argument('--encoder', type=str, default='r3m',
                        choices=['r3m', 'dinov2'])
    parser.add_argument('--subgoal_horizon', type=int, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--tasks', type=str, nargs='+', default=None)
    parser.add_argument('--action_chunk', type=int, default=None,
                        help='Worker action chunk length (1 = single-step SAC). '
                             'Use this flag to run the chunk-ablation experiment.')
    parser.add_argument('--bc_only', action='store_true',
                        help='Run Stage A demo BC + single-task worker eval, then exit.')
    parser.add_argument('--scripted_eval_only', action='store_true',
                        help='Run Stage A warmup, then evaluate the frozen worker under scripted chaining and exit.')
    parser.add_argument('--prefix_eval_only', action='store_true',
                        help='Run Stage A warmup, then evaluate the frozen worker from oracle demo prefix states and exit.')
    parser.add_argument('--prefix_target_task', type=str, default=None,
                        help='Target task for oracle prefix-state evaluation, e.g. \"light switch\".')
    parser.add_argument('--prefix_condition_tasks', type=str, nargs='*', default=None,
                        help='Tasks that must already be complete in oracle prefix-state evaluation.')
    parser.add_argument('--prefix_eval_states', type=int, default=None,
                        help='Maximum number of oracle prefix states to evaluate.')
    parser.add_argument('--demo_source', type=str, default=None,
                        choices=['auto', 'minari', 'd4rl'])
    parser.add_argument('--demo_datasets', type=str, nargs='+', default=None,
                        help='Demo datasets or aliases, e.g. franka-complete '
                             'franka-mixed franka-partial.')
    parser.add_argument('--demo_cache_dir', type=str, default=None)
    parser.add_argument('--rebuild_demo_cache', action='store_true')
    parser.add_argument('--worker_bc_steps', type=int, default=None)
    parser.add_argument('--worker_iql_steps', type=int, default=None)
    parser.add_argument('--manager_ce_steps', type=int, default=None)
    parser.add_argument('--focus_task', type=str, default=None,
                        help='Optional Stage-A focus task to oversample during offline worker training.')
    parser.add_argument('--focus_task_weight', type=float, default=None,
                        help='Relative oversampling weight for focus_task during offline worker training.')
    parser.add_argument('--focus_task_tail_start', type=float, default=None,
                        help='Normalized segment progress threshold after which focus-task samples receive extra weight.')
    parser.add_argument('--focus_task_tail_weight', type=float, default=None,
                        help='Extra weight multiplier for later-timestep focus-task samples.')
    parser.add_argument('--focus_refine_steps', type=int, default=None,
                        help='Extra BC-only refinement steps on the focus-task tail after IQL.')
    parser.add_argument('--focus_refine_min_seg_prog', type=float, default=None,
                        help='Minimum normalized segment progress for focus-task refinement samples.')
    parser.add_argument('--online_demo_bc_weight', type=float, default=None)
    parser.add_argument('--online_demo_bc_steps', type=int, default=None)
    parser.add_argument('--online_demo_mix_ratio_start', type=float, default=None)
    parser.add_argument('--online_demo_mix_ratio_end', type=float, default=None)
    parser.add_argument('--online_demo_mix_steps', type=int, default=None)
    parser.add_argument('--manager_online_demo_ce_weight', type=float, default=None)
    parser.add_argument('--manager_online_demo_ce_steps', type=int, default=None)
    parser.add_argument('--deterministic_worker_rollout_steps', type=int, default=None)
    parser.add_argument('--manager_freeze_steps', type=int, default=None)
    parser.add_argument('--scripted_manager_steps', type=int, default=None)
    parser.add_argument('--scripted_manager_prob_start', type=float, default=None)
    parser.add_argument('--scripted_manager_prob_end', type=float, default=None)
    parser.add_argument('--worker_update_start_steps', type=int, default=None)
    parser.add_argument('--min_stage_a_task_success', type=float, default=None)
    parser.add_argument('--unlock_remaining_tasks_steps', type=int, default=None)
    parser.add_argument('--controller_order_mode', type=str, default=None,
                        choices=['given_order', 'stage_a_rank'])
    parser.add_argument('--single_task_eval_episodes', type=int, default=None)
    parser.add_argument('--teacher_bc_steps', type=int, default=None)
    parser.add_argument('--teacher_iql_steps', type=int, default=None)
    parser.add_argument('--teacher_online_steps', type=int, default=None)
    parser.add_argument('--teacher_online_rollout_episodes', type=int, default=None)
    parser.add_argument('--teacher_online_prefix_states', type=int, default=None)
    parser.add_argument('--teacher_reward_task_weight', type=float, default=None)
    parser.add_argument('--teacher_reward_approach_weight', type=float, default=None)
    parser.add_argument('--teacher_reward_completion', type=float, default=None)
    parser.add_argument('--teacher_reward_action_cost', type=float, default=None)
    parser.add_argument('--teacher_online_exploration_std', type=float, default=None)
    parser.add_argument('--teacher_residual_scale', type=float, default=None)
    parser.add_argument('--disable_teacher_residual_online', action='store_true',
                        help='Let online teacher IQL update the whole teacher actor instead of a bounded residual.')
    parser.add_argument('--student_distill_steps', type=int, default=None)
    parser.add_argument('--student_rollout_distill_steps', type=int, default=None)
    parser.add_argument('--student_dagger_steps', type=int, default=None)
    parser.add_argument('--student_demo_bc_weight', type=float, default=None)
    parser.add_argument('--specialist_batch_size', type=int, default=None)
    parser.add_argument('--teacher_eval_episodes', type=int, default=None)
    parser.add_argument('--teacher_rollout_episodes', type=int, default=None)
    parser.add_argument('--teacher_prefix_states', type=int, default=None)
    parser.add_argument('--dagger_rollout_episodes', type=int, default=None)
    parser.add_argument('--dagger_prefix_states', type=int, default=None)
    parser.add_argument('--no_video', action='store_true')
    args = parser.parse_args()

    config = Config()
    if args.mode is not None:
        config.training.mode = args.mode
    config.training.seed = args.seed
    config.training.device = args.device
    if args.total_steps is not None:
        config.training.total_env_steps = args.total_steps
    if args.log_dir is not None:
        config.training.log_dir = args.log_dir
    if args.tasks is not None:
        config.training.tasks_to_complete = args.tasks
    config.encoder.name = args.encoder
    if args.subgoal_horizon is not None:
        config.manager.subgoal_horizon = args.subgoal_horizon
    if args.action_chunk is not None:
        assert args.action_chunk >= 1, "action_chunk must be >= 1"
        config.worker.action_chunk_len = args.action_chunk
    if args.bc_only:
        config.training.stage_a_only = True
    if args.scripted_eval_only:
        config.training.scripted_eval_only = True
    if args.prefix_eval_only:
        config.training.prefix_eval_only = True
    if args.prefix_target_task is not None:
        config.training.prefix_target_task = args.prefix_target_task
    if args.prefix_condition_tasks is not None:
        config.training.prefix_condition_tasks = args.prefix_condition_tasks
    if args.prefix_eval_states is not None:
        config.training.prefix_eval_n_states = args.prefix_eval_states
    if args.demo_source is not None:
        config.warmup.dataset_source = args.demo_source
    if args.demo_datasets is not None:
        config.warmup.dataset_ids = args.demo_datasets
    if args.demo_cache_dir is not None:
        config.warmup.cache_dir = args.demo_cache_dir
    if args.rebuild_demo_cache:
        config.warmup.rebuild_cache = True
    if args.worker_bc_steps is not None:
        config.warmup.n_worker_sl_steps = args.worker_bc_steps
    if args.worker_iql_steps is not None:
        config.warmup.n_worker_iql_steps = args.worker_iql_steps
    if args.manager_ce_steps is not None:
        config.warmup.n_manager_sl_steps = args.manager_ce_steps
    if args.focus_task is not None:
        config.warmup.focus_task = args.focus_task
    if args.focus_task_weight is not None:
        config.warmup.focus_task_weight = args.focus_task_weight
    if args.focus_task_tail_start is not None:
        config.warmup.focus_task_tail_start = args.focus_task_tail_start
    if args.focus_task_tail_weight is not None:
        config.warmup.focus_task_tail_weight = args.focus_task_tail_weight
    if args.focus_refine_steps is not None:
        config.warmup.n_focus_refine_steps = args.focus_refine_steps
    if args.focus_refine_min_seg_prog is not None:
        config.warmup.focus_refine_min_seg_prog = args.focus_refine_min_seg_prog
    if args.online_demo_bc_weight is not None:
        config.worker.online_demo_bc_weight = args.online_demo_bc_weight
    if args.online_demo_bc_steps is not None:
        config.worker.online_demo_bc_steps = args.online_demo_bc_steps
    if args.online_demo_mix_ratio_start is not None:
        config.worker.online_demo_mix_ratio_start = args.online_demo_mix_ratio_start
    if args.online_demo_mix_ratio_end is not None:
        config.worker.online_demo_mix_ratio_end = args.online_demo_mix_ratio_end
    if args.online_demo_mix_steps is not None:
        config.worker.online_demo_mix_steps = args.online_demo_mix_steps
    if args.manager_online_demo_ce_weight is not None:
        config.manager.online_demo_ce_weight = args.manager_online_demo_ce_weight
    if args.manager_online_demo_ce_steps is not None:
        config.manager.online_demo_ce_steps = args.manager_online_demo_ce_steps
    if args.deterministic_worker_rollout_steps is not None:
        config.training.deterministic_worker_rollout_steps = args.deterministic_worker_rollout_steps
    if args.manager_freeze_steps is not None:
        config.training.manager_freeze_steps = args.manager_freeze_steps
    if args.scripted_manager_steps is not None:
        config.training.scripted_manager_steps = args.scripted_manager_steps
    if args.scripted_manager_prob_start is not None:
        config.training.scripted_manager_prob_start = args.scripted_manager_prob_start
    if args.scripted_manager_prob_end is not None:
        config.training.scripted_manager_prob_end = args.scripted_manager_prob_end
    if args.worker_update_start_steps is not None:
        config.training.worker_update_start_steps = args.worker_update_start_steps
    if args.min_stage_a_task_success is not None:
        config.training.min_stage_a_task_success = args.min_stage_a_task_success
    if args.unlock_remaining_tasks_steps is not None:
        config.training.unlock_remaining_tasks_steps = args.unlock_remaining_tasks_steps
    if args.controller_order_mode is not None:
        config.training.controller_order_mode = args.controller_order_mode
    if args.single_task_eval_episodes is not None:
        config.eval.n_single_task_episodes = args.single_task_eval_episodes
    if args.teacher_bc_steps is not None:
        config.specialist.n_teacher_bc_steps = args.teacher_bc_steps
    if args.teacher_iql_steps is not None:
        config.specialist.n_teacher_iql_steps = args.teacher_iql_steps
    if args.teacher_online_steps is not None:
        config.specialist.n_teacher_online_steps = args.teacher_online_steps
    if args.teacher_online_rollout_episodes is not None:
        config.specialist.teacher_online_rollout_episodes = args.teacher_online_rollout_episodes
    if args.teacher_online_prefix_states is not None:
        config.specialist.teacher_online_prefix_states = args.teacher_online_prefix_states
    if args.teacher_reward_task_weight is not None:
        config.specialist.teacher_online_reward_task_weight = args.teacher_reward_task_weight
    if args.teacher_reward_approach_weight is not None:
        config.specialist.teacher_online_reward_approach_weight = args.teacher_reward_approach_weight
    if args.teacher_reward_completion is not None:
        config.specialist.teacher_online_reward_completion = args.teacher_reward_completion
    if args.teacher_reward_action_cost is not None:
        config.specialist.teacher_online_reward_action_cost = args.teacher_reward_action_cost
    if args.teacher_online_exploration_std is not None:
        config.specialist.teacher_online_exploration_std = args.teacher_online_exploration_std
    if args.teacher_residual_scale is not None:
        config.specialist.teacher_residual_scale = args.teacher_residual_scale
    if args.disable_teacher_residual_online:
        config.specialist.teacher_residual_online = False
    if args.student_distill_steps is not None:
        config.specialist.n_student_distill_steps = args.student_distill_steps
    if args.student_rollout_distill_steps is not None:
        config.specialist.n_student_rollout_distill_steps = args.student_rollout_distill_steps
    if args.student_dagger_steps is not None:
        config.specialist.n_student_dagger_steps = args.student_dagger_steps
    if args.student_demo_bc_weight is not None:
        config.specialist.student_demo_bc_weight = args.student_demo_bc_weight
    if args.specialist_batch_size is not None:
        config.specialist.batch_size = args.specialist_batch_size
    if args.teacher_eval_episodes is not None:
        config.specialist.teacher_eval_episodes = args.teacher_eval_episodes
    if args.teacher_rollout_episodes is not None:
        config.specialist.teacher_rollout_episodes = args.teacher_rollout_episodes
    if args.teacher_prefix_states is not None:
        config.specialist.teacher_prefix_states = args.teacher_prefix_states
    if args.dagger_rollout_episodes is not None:
        config.specialist.dagger_rollout_episodes = args.dagger_rollout_episodes
    if args.dagger_prefix_states is not None:
        config.specialist.dagger_prefix_states = args.dagger_prefix_states
    if args.no_video:
        config.training.record_video = False

    config.__post_init__()  # re-run to update encoder raw_dim if needed

    if config.training.device == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available — use --device cpu"

    train(config)
