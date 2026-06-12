"""
Lean offline skill training for FrankaKitchen.

Pipeline:
  demos -> replay labels/rendered images -> per-task BC/IQL -> evaluation.

This branch intentionally has no learned hierarchy and no teacher/student
curriculum. The controller is a scripted "next incomplete task" evaluator.
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from demo_dataset import build_or_load_demo_dataset, sample_oracle_prefix_states
from env_wrapper import FrankaKitchenImageWrapper
from online_finetune import run_online_finetuning
from specialist import SkillAgent, train_offline_skills
from utils import format_time, save_video


SEP = "=" * 76
SEP2 = "-" * 76
SEP3 = "." * 76


class TeeLogger:
    def __init__(self, path: str, console_print=None):
        self.path = path
        self.console_print = console_print or print
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def write(self, text: str):
        self.console_print(text, end="")
        self.f.write(text)
        self.f.flush()

    def line(self, text: str = ""):
        self.write(text + "\n")

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


def set_seed(seed: int, deterministic_torch: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def task_order(agent: SkillAgent, mode: str) -> List[int]:
    if mode == "stage_a_rank":
        return list(agent.curriculum_task_order)
    return list(range(agent.n_tasks))


def evaluate_single_task(agent: SkillAgent,
                         config: Config,
                         n_episodes: int,
                         record_dir: Optional[str] = None) -> Dict[str, float]:
    np.random.seed(12345)
    torch.manual_seed(12345)
    results: Dict[str, float] = {}
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
    report = ["Single-task deterministic evaluation", f"episodes_per_task={n_episodes}", ""]

    for task_id, task_name in enumerate(agent.tasks):
        success, options, rewards, final_errors = [], [], [], []
        best_rollout = None
        for ep in range(n_episodes):
            env = FrankaKitchenImageWrapper(
                tasks_to_complete=[task_name],
                img_size=config.encoder.img_size,
                terminate_on_tasks_completed=True,
            )
            try:
                img, state = env.reset(seed=10_000 + 1000 * task_id + ep)
                z = agent.encoder.encode_numpy(img).squeeze()
                completion = np.zeros(agent.n_tasks, dtype=np.float32)
                done = False
                n_opts = 0
                ep_reward = 0.0
                frames = [img.copy()] if record_dir else []

                while not done and completion[task_id] < 0.5 and n_opts < config.manager.max_high_level_steps:
                    result = agent.execute_option(
                        env=env,
                        task_id=task_id,
                        start_img=img,
                        start_state=state,
                        start_z=z,
                        completion=completion,
                        deterministic_worker=True,
                        collect_frames=record_dir is not None,
                    )
                    state = result.proprio_end
                    z = result.z_end
                    completion = result.completion_end
                    done = result.env_done
                    n_opts += 1
                    ep_reward += result.env_reward_sum
                    if record_dir and result.frames:
                        frames.extend(result.frames[1:])
                    if not done:
                        img = env.render_image()

                ok = bool(completion[task_id] > 0.5 or agent.spec.is_close(state, task_id))
                err = float(agent.spec.task_error(state, task_id))
                success.append(float(ok))
                options.append(float(n_opts))
                rewards.append(float(ep_reward))
                final_errors.append(err)

                score = (int(ok), -err, ep_reward, -n_opts)
                if record_dir and (best_rollout is None or score > best_rollout["score"]):
                    best_rollout = {
                        "score": score,
                        "frames": frames,
                        "success": ok,
                        "reward": ep_reward,
                        "error": err,
                        "options": n_opts,
                        "episode": ep,
                    }
            finally:
                env.close()

        safe = task_name.replace(" ", "_")
        results[f"single_task/{safe}_success_rate"] = float(np.mean(success))
        results[f"single_task/{safe}_mean_options"] = float(np.mean(options))
        results[f"single_task/{safe}_mean_env_reward"] = float(np.mean(rewards))
        results[f"single_task/{safe}_mean_final_error"] = float(np.mean(final_errors))
        report.append(
            f"{task_name:<14} success={np.mean(success)*100:5.1f}%  "
            f"options={np.mean(options):4.1f}  reward={np.mean(rewards):6.3f}  "
            f"final_error={np.mean(final_errors):.4f}"
        )
        if record_dir and best_rollout and best_rollout["frames"]:
            suffix = "success" if best_rollout["success"] else "best_fail"
            save_video(
                best_rollout["frames"],
                os.path.join(record_dir, f"{safe}_{suffix}.mp4"),
                fps=config.training.video_fps,
            )

    results["single_task/mean_success_rate"] = float(np.mean([
        results[f"single_task/{name.replace(' ', '_')}_success_rate"]
        for name in agent.tasks
    ]))
    if record_dir:
        with open(os.path.join(record_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")
    return results


def evaluate_scripted_chain(agent: SkillAgent,
                            config: Config,
                            n_episodes: int,
                            record_dir: Optional[str] = None,
                            seed_base: int = 99_999) -> Dict[str, float]:
    np.random.seed(12345)
    torch.manual_seed(12345)
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
    order = task_order(agent, config.training.controller_order_mode)
    any_success, full_success, tasks_done, options_used, chosen_sr, rewards = [], [], [], [], [], []
    terminations: Dict[str, int] = {}
    task_episode_completed = np.zeros(agent.n_tasks, dtype=np.float64)
    task_first_option_sum = np.zeros(agent.n_tasks, dtype=np.float64)
    task_first_option_count = np.zeros(agent.n_tasks, dtype=np.float64)
    final_budget_episodes = 0
    final_env_done_episodes = 0

    for ep in range(n_episodes):
        env = FrankaKitchenImageWrapper(
            tasks_to_complete=config.training.tasks_to_complete,
            img_size=config.encoder.img_size,
            terminate_on_tasks_completed=True,
        )
        try:
            img, state = env.reset(seed=int(seed_base) + ep)
            z = agent.encoder.encode_numpy(img).squeeze()
            completion = np.zeros(agent.n_tasks, dtype=np.float32)
            done = False
            n_opts = 0
            ep_reward = 0.0
            chosen_successes = 0
            first_completion_option = np.full(agent.n_tasks, np.nan, dtype=np.float32)
            frames = [img.copy()] if record_dir and ep < config.training.video_n_episodes else []

            while not done and completion.sum() < agent.n_tasks and n_opts < config.manager.max_high_level_steps:
                remaining = [k for k in order if completion[k] < 0.5]
                task_id = int(remaining[0]) if remaining else int(order[0])
                completion_before = completion.copy()
                result = agent.execute_option(
                    env=env,
                    task_id=task_id,
                    start_img=img,
                    start_state=state,
                    start_z=z,
                    completion=completion,
                    deterministic_worker=True,
                    collect_frames=bool(frames),
                )
                state = result.proprio_end
                z = result.z_end
                completion = result.completion_end
                done = result.env_done
                n_opts += 1
                ep_reward += result.env_reward_sum
                chosen_successes += int(result.chosen_task_completed)
                terminations[result.termination_reason] = terminations.get(result.termination_reason, 0) + 1
                newly_completed = np.flatnonzero((completion > 0.5) & (completion_before < 0.5))
                for completed_task in newly_completed:
                    if np.isnan(first_completion_option[completed_task]):
                        first_completion_option[completed_task] = float(n_opts)
                if frames and result.frames:
                    frames.extend(result.frames[1:])
                if not done:
                    img = env.render_image()

            done_count = int(completion.sum())
            final_budget_episodes += int((not done) and done_count < agent.n_tasks and n_opts >= config.manager.max_high_level_steps)
            final_env_done_episodes += int(done and done_count < agent.n_tasks)
            any_success.append(float(done_count >= 1))
            full_success.append(float(done_count >= agent.n_tasks))
            tasks_done.append(float(done_count))
            options_used.append(float(n_opts))
            chosen_sr.append(float(chosen_successes / max(n_opts, 1)))
            rewards.append(float(ep_reward))
            task_episode_completed += (completion > 0.5).astype(np.float64)
            seen = ~np.isnan(first_completion_option)
            task_first_option_sum[seen] += first_completion_option[seen]
            task_first_option_count[seen] += 1.0
            if frames:
                save_video(frames, os.path.join(record_dir, f"ep_{ep:03d}.mp4"), fps=config.training.video_fps)
        finally:
            env.close()

    out = {
        "eval/any_task_success_rate": float(np.mean(any_success)),
        "eval/full_task_success_rate": float(np.mean(full_success)),
        "eval/mean_tasks_completed": float(np.mean(tasks_done)),
        "eval/mean_options_used": float(np.mean(options_used)),
        "eval/mean_chosen_task_success": float(np.mean(chosen_sr)),
        "eval/mean_env_reward": float(np.mean(rewards)),
        "eval/std_env_reward": float(np.std(rewards)),
        "eval/final_budget_episode_rate": float(final_budget_episodes / max(n_episodes, 1)),
        "eval/final_env_done_failure_rate": float(final_env_done_episodes / max(n_episodes, 1)),
        "eval/termination_reasons": terminations,
    }
    for task_id, task_name in enumerate(agent.tasks):
        safe = task_name.replace(" ", "_")
        out[f"eval/task/{safe}_completion_rate"] = float(task_episode_completed[task_id] / max(n_episodes, 1))
        if task_first_option_count[task_id] > 0:
            out[f"eval/task/{safe}_mean_first_option"] = float(
                task_first_option_sum[task_id] / task_first_option_count[task_id]
            )
        else:
            out[f"eval/task/{safe}_mean_first_option"] = float("nan")
    return out


def evaluate_prefix(agent: SkillAgent,
                    config: Config,
                    prefix_tasks: Sequence[str],
                    target_task: str,
                    n_states: int,
                    record_dir: Optional[str] = None) -> Dict[str, float]:
    np.random.seed(12345)
    torch.manual_seed(12345)
    samples, sampler_stats = sample_oracle_prefix_states(
        agent=agent,
        config=config,
        prefix_tasks=prefix_tasks,
        target_task=target_task,
        max_states=n_states,
        verbose=True,
    )
    target_id = agent.tasks.index(target_task)
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=False,
    )
    successes, best_errors, final_errors = [], [], []
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
    try:
        for i, sample in enumerate(samples):
            env.reset(seed=30_000 + i)
            qpos, qvel = env.observation_to_qpos_qvel(sample["state"])
            env.set_mujoco_state(qpos, qvel)
            env._current_obs = {"observation": np.asarray(sample["state"], dtype=np.float64).copy()}
            env._step_count = 0
            state = np.asarray(sample["state"], dtype=np.float64).copy()
            img = env.render_image()
            z = agent.encoder.encode_numpy(img).squeeze()
            completion = np.asarray(sample["completion"], dtype=np.float32).copy()
            best_err = float(agent.spec.task_error(state, target_id))
            frames = [img.copy()] if record_dir and i < config.training.video_n_episodes else []
            done = False
            n_opts = 0
            while not done and completion[target_id] < 0.5 and n_opts < config.manager.max_high_level_steps:
                result = agent.execute_option(
                    env=env,
                    task_id=target_id,
                    start_img=img,
                    start_state=state,
                    start_z=z,
                    completion=completion,
                    deterministic_worker=True,
                    collect_frames=bool(frames),
                )
                state = result.proprio_end
                z = result.z_end
                completion = result.completion_end
                done = result.env_done
                n_opts += 1
                best_err = min(best_err, float(agent.spec.task_error(state, target_id)))
                if frames and result.frames:
                    frames.extend(result.frames[1:])
                if not done:
                    img = env.render_image()
            ok = bool(completion[target_id] > 0.5 or agent.spec.is_close(state, target_id))
            successes.append(float(ok))
            best_errors.append(best_err)
            final_errors.append(float(agent.spec.task_error(state, target_id)))
            if frames:
                suffix = "success" if ok else "fail"
                save_video(frames, os.path.join(record_dir, f"sample_{i:03d}_{suffix}.mp4"), fps=config.training.video_fps)
    finally:
        env.close()

    out = {
        "prefix/success_rate": float(np.mean(successes)) if successes else 0.0,
        "prefix/best_error": float(np.mean(best_errors)) if best_errors else 0.0,
        "prefix/final_error": float(np.mean(final_errors)) if final_errors else 0.0,
        "prefix/n_states": float(len(samples)),
    }
    for k, v in sampler_stats.items():
        out[f"prefix_sampler/{k}"] = float(v)
    return out


def evaluate_chain_context_skills(agent: SkillAgent,
                                  config: Config,
                                  n_states: int) -> Dict[str, float]:
    """Evaluate each skill from oracle states after its scripted prefix."""
    np.random.seed(12345)
    torch.manual_seed(12345)
    out: Dict[str, float] = {}
    prev_env_steps = int(getattr(agent, "total_env_steps", 0))
    prev_options = int(getattr(agent, "total_options", 0))

    for target_id, target_name in enumerate(agent.tasks):
        safe = target_name.replace(" ", "_")
        try:
            samples, sampler_stats = sample_oracle_prefix_states(
                agent=agent,
                config=config,
                prefix_tasks=agent.tasks[:target_id],
                target_task=target_name,
                max_states=n_states,
                verbose=False,
            )
        except Exception as exc:
            print(f"  [Chain-context] {target_name}: skipped ({exc})")
            out[f"chain_context/{safe}_success_rate"] = 0.0
            out[f"chain_context/{safe}_mean_final_error"] = float("inf")
            out[f"chain_context/{safe}_mean_options"] = 0.0
            out[f"chain_context/{safe}_n_states"] = 0.0
            continue

        env = FrankaKitchenImageWrapper(
            tasks_to_complete=config.training.tasks_to_complete,
            img_size=config.encoder.img_size,
            terminate_on_tasks_completed=False,
        )
        successes: List[float] = []
        final_errors: List[float] = []
        options: List[float] = []
        try:
            for i, sample in enumerate(samples):
                env.reset(seed=int(config.eval.prefix_sample_seed) + 10_000 + target_id * 1_000 + i)
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
                    and completion[target_id] < 0.5
                    and n_opts < config.manager.max_high_level_steps
                ):
                    result = agent.execute_option(
                        env=env,
                        task_id=target_id,
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

                successes.append(float(completion[target_id] > 0.5))
                final_errors.append(float(agent.spec.task_error(state, target_id)))
                options.append(float(n_opts))
        finally:
            env.close()

        out[f"chain_context/{safe}_success_rate"] = float(np.mean(successes)) if successes else 0.0
        out[f"chain_context/{safe}_mean_final_error"] = float(np.mean(final_errors)) if final_errors else 0.0
        out[f"chain_context/{safe}_mean_options"] = float(np.mean(options)) if options else 0.0
        out[f"chain_context/{safe}_n_states"] = float(len(samples))
        out[f"chain_context/{safe}_matching_segments"] = float(sampler_stats.get("matching_segments", 0.0))

    agent.total_env_steps = prev_env_steps
    agent.total_options = prev_options
    return out


def print_banner(config: Config, log_path: str):
    print(f"\n{SEP}")
    print("  Lean Skill Learning — FrankaKitchen-v1")
    print(f"  Started        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Seed           : {config.training.seed}  deterministic_torch={config.training.deterministic_torch}")
    print(f"  Encoder        : {config.encoder.name.upper()} ({config.encoder.raw_dim}-d, frozen)")
    print(f"  Image size     : {config.encoder.img_size}")
    print(f"  Tasks          : {config.training.tasks_to_complete}")
    print(f"  Policy         : one visual policy per task")
    print(f"  Controller     : scripted next-incomplete task")
    print(SEP2)
    print(f"  Demos          : {config.warmup.dataset_ids}")
    print(f"  Demo source    : {config.warmup.dataset_source}")
    print(f"  Cache          : {config.warmup.cache_dir}  rebuild={config.warmup.rebuild_cache}")
    print(f"  Render batch   : {config.warmup.render_batch_size}  max_eps_per_dataset={config.warmup.max_episodes_per_dataset}")
    print(f"  Offline algo   : {config.specialist.offline_algo}")
    print(f"  Skill net      : hidden={config.specialist.hidden_dim}  layers={config.specialist.n_layers}  "
          f"chunk={config.worker.action_chunk_len}")
    print(f"  Optimizer      : actor_lr={config.worker.actor_lr}  critic_lr={config.worker.critic_lr}  "
          f"gamma={config.worker.gamma}")
    print(f"  Train steps    : bc={config.specialist.n_teacher_bc_steps}  "
          f"offline_rl={config.specialist.n_offline_rl_steps}  bet={config.specialist.bet_steps}  "
          f"batch={config.specialist.batch_size}")
    print(f"  Option budget  : subgoal_horizon={config.manager.subgoal_horizon}  "
          f"max_high_level_steps={config.manager.max_high_level_steps}")
    print(f"  TB log interval: every {config.specialist.log_interval} optimizer steps")
    print(f"  IQL params     : expectile={config.specialist.iql_expectile}  "
          f"adv_beta={config.specialist.iql_adv_beta}  max_weight={config.specialist.iql_max_weight}  "
          f"adv_norm={config.specialist.iql_normalize_advantage}  "
          f"value_target={config.specialist.iql_use_value_target}  "
          f"value_target_tau={config.specialist.iql_value_target_tau}  "
          f"prefix_eval_interval={config.specialist.iql_eval_interval}  "
          f"prefix_eval_states={config.specialist.iql_eval_prefix_states}")
    print(f"  LQL params     : enabled={config.specialist.lql_enabled}  "
          f"lambda_lb={config.specialist.lql_lambda_lb}  "
          f"n_transitions={config.specialist.lql_n_transitions}  "
          f"min_gap={config.specialist.lql_min_gap}")
    print(f"  TD3+BC params  : alpha={config.specialist.td3bc_alpha}  tau={config.specialist.td3bc_tau}  "
          f"policy_noise={config.specialist.td3bc_policy_noise}  noise_clip={config.specialist.td3bc_noise_clip}  "
          f"policy_freq={config.specialist.td3bc_policy_freq}")
    print(f"  AWR params     : temperature={config.specialist.awr_temperature}  "
          f"max_weight={config.specialist.awr_max_weight}")
    print(f"  BeT params     : steps={config.specialist.bet_steps}  bins={config.specialist.bet_num_bins}  "
          f"offset_weight={config.specialist.bet_offset_weight}")
    print(f"  Dense reward   : progress={config.worker.progress_weight}  "
          f"completion={config.worker.completion_bonus}  action_cost={config.worker.action_cost}  "
          f"sigma={config.worker.sigma}")
    print(f"  Reward eqn     : r = {config.worker.progress_weight}*(gamma*phi(s')-phi(s)) "
          f"+ {config.worker.completion_bonus}*done "
          f"- {config.worker.action_cost}*||a||^2   "
          f"phi(s)=exp(-(e/eps)/sigma)")
    print(f"  Eval episodes  : single={config.eval.n_single_task_episodes}  chain={config.eval.n_eval_episodes}  "
          f"prefix_states={config.training.prefix_eval_n_states}")
    print(f"  Chain-context  : enabled={config.eval.chain_context_eval}  "
          f"states={config.eval.chain_context_eval_states}")
    print(f"  Video          : record={config.training.record_video}  n={config.training.video_n_episodes}  "
          f"fps={config.training.video_fps}")
    print(f"  Online AWAC    : enabled={config.online.enabled}  steps={config.online.total_env_steps}  "
          f"mode={config.online.mode}  updates_per_env_step={config.online.updates_per_env_step}  "
          f"batch={config.online.batch_size}")
    print(f"  Online replay  : demo_fraction={config.online.demo_fraction_start}->{config.online.demo_fraction_end}  "
          f"demo_decay={config.online.demo_fraction_decay_steps}  "
          f"bc_anchor={config.online.bc_anchor_weight}->{config.online.bc_anchor_weight_end}  "
          f"anchor_decay={config.online.bc_anchor_decay_steps}  "
          f"awac_temp={config.online.awac_temperature}  max_weight={config.online.awac_max_weight}")
    print(f"  Online collect : exploration_noise={config.online.exploration_noise}  "
          f"failure_priority={config.online.failure_priority}  eval_interval={config.online.eval_interval_steps}")
    print(f"  Online repair  : freeze_threshold={config.online.freeze_success_threshold}  "
          f"next_skill_threshold={config.online.next_skill_collection_threshold}  "
          f"collect_next_on_success={config.online.collect_next_on_success}  "
          f"actor_success_only={config.online.actor_success_only}  "
          f"actor_high_return_fallback={config.online.actor_include_high_return_failures}  "
          f"min_actor_success={config.online.min_actor_success_samples}  "
          f"critic_huber={config.online.critic_huber_loss}  "
          f"huber_delta={config.online.critic_huber_delta}  "
          f"rollback_drop={config.online.rollback_drop_tolerance}")
    if config.online.load_checkpoint:
        print(f"  Load checkpoint: {config.online.load_checkpoint}  skip_offline={config.online.skip_offline_training}")
    print(f"  Device         : {config.training.device}")
    print(f"  Log dir        : {config.training.log_dir}")
    print(f"  Train log      : {log_path}")
    print(f"{SEP}\n")


def train(config: Config):
    set_seed(config.training.seed, config.training.deterministic_torch)
    os.makedirs(config.training.log_dir, exist_ok=True)
    log_path = os.path.join(
        config.training.log_dir,
        f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    import builtins
    orig_print = builtins.print
    logger = TeeLogger(log_path, console_print=orig_print)

    def tee_print(*args, sep=" ", end="\n", file=None, flush=False):
        # Preserve normal print semantics while teeing stdout into the log file.
        # If a library prints to an explicit file handle, leave it alone.
        if file is not None:
            orig_print(*args, sep=sep, end=end, file=file, flush=flush)
            return
        logger.write(sep.join(str(a) for a in args) + end)
        if flush:
            logger.flush()

    builtins.print = tee_print
    writer = SummaryWriter(config.training.log_dir)
    try:
        print_banner(config, log_path)
        agent = SkillAgent(config)
        t0 = time.time()
        if config.online.skip_offline_training:
            print("  [Stage A] Skipping offline optimizer steps; preparing demo replay only.")
            ds, stats = build_or_load_demo_dataset(agent, config, verbose=True)
            agent.demo_dataset = ds
            agent.proprio_norm.fit(ds.w_p)
            if not config.online.load_checkpoint:
                raise ValueError("--skip_offline_training requires --load_checkpoint.")
        else:
            stats = train_offline_skills(agent, config, verbose=True, writer=writer)
        if config.online.load_checkpoint:
            print(f"  [Checkpoint] Loading policy state -> {config.online.load_checkpoint}")
            agent.load(config.online.load_checkpoint)
        print(f"\n  Stage A complete in {format_time(time.time() - t0)}.")
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"warmup/{k}", float(v), 0)

        if config.eval.chain_context_eval:
            print(SEP2)
            print("  CHAIN-CONTEXT SKILL EVAL")
            print(SEP2)
            chain_context = evaluate_chain_context_skills(
                agent,
                config,
                config.eval.chain_context_eval_states,
            )
            for k, v in chain_context.items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    writer.add_scalar(k, float(v), 0)
            print(f"  Oracle prefix states per task: {config.eval.chain_context_eval_states}")
            for name in agent.tasks:
                safe = name.replace(" ", "_")
                print(
                    f"  {name:<14} prefix_success="
                    f"{chain_context[f'chain_context/{safe}_success_rate']*100:5.1f}%  "
                    f"options={chain_context[f'chain_context/{safe}_mean_options']:.1f}  "
                    f"err={chain_context[f'chain_context/{safe}_mean_final_error']:.4f}"
                )

        print(SEP2)
        print("  SINGLE-TASK EVAL")
        print(SEP2)
        single_dir = os.path.join(config.training.log_dir, "videos", "single_task") if config.training.record_video else None
        single = evaluate_single_task(agent, config, config.eval.n_single_task_episodes, record_dir=single_dir)
        for k, v in single.items():
            writer.add_scalar(k, float(v), 0)
        for name in agent.tasks:
            safe = name.replace(" ", "_")
            print(f"  {name:<14} success={single[f'single_task/{safe}_success_rate']*100:5.1f}%  "
                  f"options={single[f'single_task/{safe}_mean_options']:.1f}  "
                  f"err={single[f'single_task/{safe}_mean_final_error']:.4f}")
        print(f"  Mean single-task SR: {single['single_task/mean_success_rate']*100:5.1f}%")
        agent.stage_a_task_success = np.asarray([
            single[f"single_task/{name.replace(' ', '_')}_success_rate"] for name in agent.tasks
        ], dtype=np.float32)
        agent.curriculum_task_order = list(np.argsort(-agent.stage_a_task_success))

        if config.training.prefix_eval_only:
            print(SEP2)
            print("  PREFIX-STATE EVAL")
            print(SEP2)
            prefix_dir = os.path.join(config.training.log_dir, "videos", "prefix_eval") if config.training.record_video else None
            prefix = evaluate_prefix(
                agent,
                config,
                config.training.prefix_condition_tasks,
                config.training.prefix_target_task,
                config.training.prefix_eval_n_states,
                record_dir=prefix_dir,
            )
            for k, v in prefix.items():
                writer.add_scalar(k, float(v), 0)
            print(f"  Target       : {config.training.prefix_target_task}")
            print(f"  Prefix       : {config.training.prefix_condition_tasks}")
            print(f"  Success      : {prefix['prefix/success_rate']*100:5.1f}%")
            print(f"  Best error   : {prefix['prefix/best_error']:.4f}")
            print(f"  Final error  : {prefix['prefix/final_error']:.4f}")
        else:
            print(SEP2)
            print("  SCRIPTED CHAIN EVAL")
            print(SEP2)
            chain_dir = os.path.join(config.training.log_dir, "videos", "scripted_chain") if config.training.record_video else None
            chain = evaluate_scripted_chain(agent, config, config.eval.n_eval_episodes, record_dir=chain_dir)
            for k, v in chain.items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    writer.add_scalar(k, float(v), 0)
            print(f"  Any-task success   : {chain['eval/any_task_success_rate']*100:5.1f}%")
            print(f"  Full-task success  : {chain['eval/full_task_success_rate']*100:5.1f}%")
            print(f"  Mean tasks done    : {chain['eval/mean_tasks_completed']:.2f}/{agent.n_tasks}")
            print(f"  Chosen-task SR     : {chain['eval/mean_chosen_task_success']*100:5.1f}%")
            print(f"  Budget fail eps    : {chain['eval/final_budget_episode_rate']*100:5.1f}%")
            print(f"  Env-horizon fail   : {chain['eval/final_env_done_failure_rate']*100:5.1f}%")
            print("  Per-task chain completion:")
            for name in agent.tasks:
                safe = name.replace(" ", "_")
                rate = chain[f"eval/task/{safe}_completion_rate"] * 100.0
                first_opt = chain[f"eval/task/{safe}_mean_first_option"]
                first_str = f"{first_opt:.1f}" if np.isfinite(first_opt) else "n/a"
                print(f"    {name:<14} completion={rate:5.1f}%  first_option={first_str}")
            print(f"  Termination mix    : {chain['eval/termination_reasons']}")

            if config.online.enabled:
                online_stats = run_online_finetuning(
                    agent,
                    config,
                    writer=writer,
                    evaluate_fn=evaluate_scripted_chain,
                    initial_eval=chain,
                    verbose=True,
                )
                for k, v in online_stats.items():
                    writer.add_scalar(k, float(v), int(online_stats["online/env_steps"]))

                print(SEP2)
                print("  FINAL SCRIPTED CHAIN EVAL AFTER ONLINE")
                print(SEP2)
                final_chain_dir = (
                    os.path.join(config.training.log_dir, "videos", "scripted_chain_online_final")
                    if config.training.record_video else None
                )
                # Report on episode seeds DISJOINT from the model-selection
                # stream, repeated to average out eval nondeterminism.
                repeats = max(1, int(config.eval.final_eval_repeats))
                repeat_evals: List[Dict[str, float]] = []
                for rep in range(repeats):
                    rep_eval = evaluate_scripted_chain(
                        agent,
                        config,
                        config.eval.n_eval_episodes,
                        record_dir=final_chain_dir if rep == 0 else None,
                        seed_base=int(config.eval.final_eval_seed_base) + 1_000 * rep,
                    )
                    repeat_evals.append(rep_eval)
                    print(f"  [Final eval repeat {rep + 1}/{repeats}] "
                          f"full={rep_eval['eval/full_task_success_rate']*100:5.1f}%  "
                          f"tasks={rep_eval['eval/mean_tasks_completed']:.2f}/{agent.n_tasks}")
                final_chain = {
                    key: float(np.mean([e[key] for e in repeat_evals]))
                    for key, value in repeat_evals[0].items()
                    if isinstance(value, (int, float))
                }
                final_chain["eval/termination_reasons"] = repeat_evals[0]["eval/termination_reasons"]
                full_std = float(np.std([e["eval/full_task_success_rate"] for e in repeat_evals]))
                for k, v in final_chain.items():
                    if isinstance(v, (int, float)) and np.isfinite(float(v)):
                        writer.add_scalar(f"final_after_online/{k}", float(v), int(online_stats["online/env_steps"]))
                writer.add_scalar("final_after_online/eval/full_task_success_std", full_std,
                                  int(online_stats["online/env_steps"]))
                print(f"  Any-task success   : {final_chain['eval/any_task_success_rate']*100:5.1f}%")
                print(f"  Full-task success  : {final_chain['eval/full_task_success_rate']*100:5.1f}%"
                      f"  (+/- {full_std*100:.1f}pp over {repeats} repeats, report seeds "
                      f"{config.eval.final_eval_seed_base}+)")
                print(f"  Mean tasks done    : {final_chain['eval/mean_tasks_completed']:.2f}/{agent.n_tasks}")
                print(f"  Chosen-task SR     : {final_chain['eval/mean_chosen_task_success']*100:5.1f}%")
                print(f"  Env-horizon fail   : {final_chain['eval/final_env_done_failure_rate']*100:5.1f}%")
                print("  Per-task chain completion (mean over repeats):")
                for name in agent.tasks:
                    safe = name.replace(" ", "_")
                    rate = final_chain[f"eval/task/{safe}_completion_rate"] * 100.0
                    first_opt = final_chain[f"eval/task/{safe}_mean_first_option"]
                    first_str = f"{first_opt:.1f}" if np.isfinite(first_opt) else "n/a"
                    print(f"    {name:<14} completion={rate:5.1f}%  first_option={first_str}")
                print(f"  Termination mix    : {final_chain['eval/termination_reasons']}")

        ckpt_dir = os.path.join(config.training.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        agent.save(os.path.join(ckpt_dir, "checkpoint_final.pt"))
        print(f"\n{SEP}")
        print("  RUN COMPLETE")
        print(SEP2)
        print(f"  Log file       : {log_path}")
        print(f"  Checkpoint     : {os.path.join(ckpt_dir, 'checkpoint_final.pt')}")
        print(f"{SEP}\n")
    finally:
        writer.close()
        builtins.print = orig_print
        logger.close()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Lean per-skill BC/IQL for FrankaKitchen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder", type=str, default="r3m", choices=["r3m", "dinov2", "dinov3"])
    parser.add_argument("--dinov3_model", type=str, default=None)
    parser.add_argument("--dinov3_weights", type=str, default=None)
    parser.add_argument("--dinov3_repo_or_dir", type=str, default=None)
    parser.add_argument("--dinov3_source", type=str, default=None, choices=["github", "local"])
    parser.add_argument("--log_dir", type=str, default="logs/lean_skills")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--demo_datasets", nargs="+", default=None)
    parser.add_argument("--demo_source", type=str, default=None, choices=["auto", "minari", "d4rl"])
    parser.add_argument("--demo_cache_dir", type=str, default=None)
    parser.add_argument("--rebuild_demo_cache", action="store_true")
    parser.add_argument("--bc_steps", "--teacher_bc_steps", dest="bc_steps", type=int, default=None)
    parser.add_argument("--iql_steps", "--teacher_iql_steps", "--offline_rl_steps", dest="offline_rl_steps", type=int, default=None)
    parser.add_argument("--offline_algo", type=str, default=None,
                        choices=["bc", "bc_iql", "iql", "td3bc", "td3_bc", "awr", "bet", "behavior_transformer", "sequence_bc"])
    parser.add_argument("--batch_size", "--specialist_batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--action_chunk", "--action_chunk_len", dest="action_chunk", type=int, default=None)
    parser.add_argument("--subgoal_horizon", type=int, default=None)
    parser.add_argument("--max_high_level_steps", type=int, default=None)
    parser.add_argument("--iql_expectile", type=float, default=None)
    parser.add_argument("--iql_adv_beta", type=float, default=None)
    parser.add_argument("--iql_max_weight", type=float, default=None)
    parser.add_argument("--iql_use_value_target", action="store_true")
    parser.add_argument("--iql_value_target_tau", type=float, default=None)
    parser.add_argument("--iql_normalize_advantage", "--iql_adv_normalize", action="store_true")
    parser.add_argument("--iql_eval_interval", type=int, default=None)
    parser.add_argument("--iql_eval_prefix_states", type=int, default=None)
    parser.add_argument("--no_lql", action="store_true",
                        help="Disable the LQL lower-bound critic penalty (A/B baseline).")
    parser.add_argument("--lql_lambda_lb", type=float, default=None)
    parser.add_argument("--lql_n_transitions", type=int, default=None)
    parser.add_argument("--lql_min_gap", type=int, default=None)
    parser.add_argument("--td3bc_alpha", type=float, default=None)
    parser.add_argument("--awr_temperature", type=float, default=None)
    parser.add_argument("--awr_max_weight", type=float, default=None)
    parser.add_argument("--bet_steps", type=int, default=None)
    parser.add_argument("--bet_num_bins", type=int, default=None)
    parser.add_argument("--bet_offset_weight", type=float, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--single_task_eval_episodes", type=int, default=None)
    parser.add_argument("--chain_eval_episodes", type=int, default=None)
    parser.add_argument("--final_eval_repeats", type=int, default=None)
    parser.add_argument("--controller_order_mode", type=str, default=None, choices=["given_order", "stage_a_rank"])
    parser.add_argument("--prefix_eval_only", action="store_true")
    parser.add_argument("--prefix_target_task", type=str, default="")
    parser.add_argument("--prefix_condition_tasks", nargs="*", default=None)
    parser.add_argument("--prefix_eval_states", type=int, default=None)
    parser.add_argument("--chain_context_eval_states", type=int, default=None)
    parser.add_argument("--no_chain_context_eval", action="store_true")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--online_finetune", action="store_true")
    parser.add_argument("--online_mode", type=str, default=None, choices=["skill_repair", "chain"])
    parser.add_argument("--online_steps", type=int, default=None)
    parser.add_argument("--online_eval_interval", type=int, default=None)
    parser.add_argument("--online_log_interval_episodes", type=int, default=None)
    parser.add_argument("--online_updates_per_env_step", type=float, default=None)
    parser.add_argument("--online_batch_size", type=int, default=None)
    parser.add_argument("--online_buffer_capacity_per_skill", type=int, default=None)
    parser.add_argument("--online_demo_fraction_start", type=float, default=None)
    parser.add_argument("--online_demo_fraction_end", type=float, default=None)
    parser.add_argument("--online_demo_fraction_decay_steps", type=int, default=None)
    parser.add_argument("--online_awac_temperature", type=float, default=None)
    parser.add_argument("--online_awac_max_weight", type=float, default=None)
    parser.add_argument("--online_bc_anchor_weight", type=float, default=None)
    parser.add_argument("--online_bc_anchor_weight_end", type=float, default=None)
    parser.add_argument("--online_bc_anchor_decay_steps", type=int, default=None)
    parser.add_argument("--online_critic_target_tau", type=float, default=None)
    parser.add_argument("--online_no_critic_huber", action="store_true")
    parser.add_argument("--online_critic_huber_delta", type=float, default=None)
    parser.add_argument("--online_normalize_advantage", action="store_true")
    parser.add_argument("--online_actor_all_attempts", action="store_true")
    parser.add_argument("--online_actor_high_return_fallback", action="store_true")
    parser.add_argument("--online_no_collect_next_on_success", action="store_true")
    parser.add_argument("--online_min_actor_success_samples", type=int, default=None)
    parser.add_argument("--online_freeze_success_threshold", type=float, default=None)
    parser.add_argument("--online_next_skill_collection_threshold", type=float, default=None)
    parser.add_argument("--online_rollback_drop_tolerance", type=float, default=None)
    parser.add_argument("--online_exploration_noise", type=float, default=None)
    parser.add_argument("--online_failure_priority", type=float, default=None)
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--skip_offline_training", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.training.mode = "lean_skills"
    cfg.training.seed = args.seed
    cfg.training.device = args.device
    cfg.training.log_dir = args.log_dir
    cfg.encoder.name = args.encoder
    if args.dinov3_model is not None:
        cfg.encoder.dinov3_model = args.dinov3_model
    if args.dinov3_weights is not None:
        cfg.encoder.dinov3_weights = args.dinov3_weights
    if args.dinov3_repo_or_dir is not None:
        cfg.encoder.dinov3_repo_or_dir = args.dinov3_repo_or_dir
    if args.dinov3_source is not None:
        cfg.encoder.dinov3_source = args.dinov3_source
    cfg.refresh_encoder_dim()
    if args.tasks is not None:
        cfg.training.tasks_to_complete = args.tasks
    if args.demo_datasets is not None:
        cfg.warmup.dataset_ids = args.demo_datasets
    if args.demo_source is not None:
        cfg.warmup.dataset_source = args.demo_source
    if args.demo_cache_dir is not None:
        cfg.warmup.cache_dir = args.demo_cache_dir
    if args.rebuild_demo_cache:
        cfg.warmup.rebuild_cache = True
    if args.bc_steps is not None:
        cfg.specialist.n_teacher_bc_steps = args.bc_steps
    if args.offline_rl_steps is not None:
        cfg.specialist.n_offline_rl_steps = args.offline_rl_steps
        cfg.specialist.n_teacher_iql_steps = args.offline_rl_steps
    if args.offline_algo is not None:
        cfg.specialist.offline_algo = args.offline_algo
    if args.batch_size is not None:
        cfg.specialist.batch_size = args.batch_size
    if args.hidden_dim is not None:
        cfg.specialist.hidden_dim = args.hidden_dim
    if args.n_layers is not None:
        cfg.specialist.n_layers = args.n_layers
    if args.action_chunk is not None:
        cfg.worker.action_chunk_len = args.action_chunk
    if args.subgoal_horizon is not None:
        cfg.manager.subgoal_horizon = args.subgoal_horizon
    if args.max_high_level_steps is not None:
        cfg.manager.max_high_level_steps = args.max_high_level_steps
    if args.iql_expectile is not None:
        cfg.specialist.iql_expectile = args.iql_expectile
    if args.iql_adv_beta is not None:
        cfg.specialist.iql_adv_beta = args.iql_adv_beta
    if args.iql_max_weight is not None:
        cfg.specialist.iql_max_weight = args.iql_max_weight
    if args.iql_use_value_target:
        cfg.specialist.iql_use_value_target = True
    if args.iql_value_target_tau is not None:
        cfg.specialist.iql_value_target_tau = args.iql_value_target_tau
    if args.iql_normalize_advantage:
        cfg.specialist.iql_normalize_advantage = True
    if args.iql_eval_interval is not None:
        cfg.specialist.iql_eval_interval = args.iql_eval_interval
    if args.iql_eval_prefix_states is not None:
        cfg.specialist.iql_eval_prefix_states = args.iql_eval_prefix_states
    if args.no_lql:
        cfg.specialist.lql_enabled = False
    if args.lql_lambda_lb is not None:
        cfg.specialist.lql_lambda_lb = args.lql_lambda_lb
    if args.lql_n_transitions is not None:
        cfg.specialist.lql_n_transitions = args.lql_n_transitions
    if args.lql_min_gap is not None:
        cfg.specialist.lql_min_gap = args.lql_min_gap
    if args.td3bc_alpha is not None:
        cfg.specialist.td3bc_alpha = args.td3bc_alpha
    if args.awr_temperature is not None:
        cfg.specialist.awr_temperature = args.awr_temperature
    if args.awr_max_weight is not None:
        cfg.specialist.awr_max_weight = args.awr_max_weight
    if args.bet_steps is not None:
        cfg.specialist.bet_steps = args.bet_steps
    if args.bet_num_bins is not None:
        cfg.specialist.bet_num_bins = args.bet_num_bins
    if args.bet_offset_weight is not None:
        cfg.specialist.bet_offset_weight = args.bet_offset_weight
    if args.log_interval is not None:
        cfg.specialist.log_interval = args.log_interval
    if args.single_task_eval_episodes is not None:
        cfg.eval.n_single_task_episodes = args.single_task_eval_episodes
    if args.chain_eval_episodes is not None:
        cfg.eval.n_eval_episodes = args.chain_eval_episodes
    if args.final_eval_repeats is not None:
        cfg.eval.final_eval_repeats = args.final_eval_repeats
    if args.controller_order_mode is not None:
        cfg.training.controller_order_mode = args.controller_order_mode
    if args.prefix_eval_only:
        cfg.training.prefix_eval_only = True
        cfg.training.prefix_target_task = args.prefix_target_task
        cfg.training.prefix_condition_tasks = args.prefix_condition_tasks or []
    if args.prefix_eval_states is not None:
        cfg.training.prefix_eval_n_states = args.prefix_eval_states
    if args.chain_context_eval_states is not None:
        cfg.eval.chain_context_eval_states = args.chain_context_eval_states
    if args.no_chain_context_eval:
        cfg.eval.chain_context_eval = False
    if args.no_video:
        cfg.training.record_video = False
    if args.online_finetune:
        cfg.online.enabled = True
    if args.online_mode is not None:
        cfg.online.mode = args.online_mode
    if args.online_steps is not None:
        cfg.online.total_env_steps = args.online_steps
    if args.online_eval_interval is not None:
        cfg.online.eval_interval_steps = args.online_eval_interval
    if args.online_log_interval_episodes is not None:
        cfg.online.log_interval_episodes = args.online_log_interval_episodes
    if args.online_updates_per_env_step is not None:
        cfg.online.updates_per_env_step = args.online_updates_per_env_step
    if args.online_batch_size is not None:
        cfg.online.batch_size = args.online_batch_size
    if args.online_buffer_capacity_per_skill is not None:
        cfg.online.online_buffer_capacity_per_skill = args.online_buffer_capacity_per_skill
    if args.online_demo_fraction_start is not None:
        cfg.online.demo_fraction_start = args.online_demo_fraction_start
    if args.online_demo_fraction_end is not None:
        cfg.online.demo_fraction_end = args.online_demo_fraction_end
    if args.online_demo_fraction_decay_steps is not None:
        cfg.online.demo_fraction_decay_steps = args.online_demo_fraction_decay_steps
    if args.online_awac_temperature is not None:
        cfg.online.awac_temperature = args.online_awac_temperature
    if args.online_awac_max_weight is not None:
        cfg.online.awac_max_weight = args.online_awac_max_weight
    if args.online_bc_anchor_weight is not None:
        cfg.online.bc_anchor_weight = args.online_bc_anchor_weight
    if args.online_bc_anchor_weight_end is not None:
        cfg.online.bc_anchor_weight_end = args.online_bc_anchor_weight_end
    if args.online_bc_anchor_decay_steps is not None:
        cfg.online.bc_anchor_decay_steps = args.online_bc_anchor_decay_steps
    if args.online_critic_target_tau is not None:
        cfg.online.critic_target_tau = args.online_critic_target_tau
    if args.online_no_critic_huber:
        cfg.online.critic_huber_loss = False
    if args.online_critic_huber_delta is not None:
        cfg.online.critic_huber_delta = args.online_critic_huber_delta
    if args.online_normalize_advantage:
        cfg.online.normalize_advantage = True
    if args.online_actor_all_attempts:
        cfg.online.actor_success_only = False
    if args.online_actor_high_return_fallback:
        cfg.online.actor_include_high_return_failures = True
    if args.online_no_collect_next_on_success:
        cfg.online.collect_next_on_success = False
    if args.online_min_actor_success_samples is not None:
        cfg.online.min_actor_success_samples = args.online_min_actor_success_samples
    if args.online_freeze_success_threshold is not None:
        cfg.online.freeze_success_threshold = args.online_freeze_success_threshold
    if args.online_next_skill_collection_threshold is not None:
        cfg.online.next_skill_collection_threshold = args.online_next_skill_collection_threshold
    if args.online_rollback_drop_tolerance is not None:
        cfg.online.rollback_drop_tolerance = args.online_rollback_drop_tolerance
    if args.online_exploration_noise is not None:
        cfg.online.exploration_noise = args.online_exploration_noise
    if args.online_failure_priority is not None:
        cfg.online.failure_priority = args.online_failure_priority
    if args.load_checkpoint:
        cfg.online.load_checkpoint = args.load_checkpoint
    if args.skip_offline_training:
        cfg.online.skip_offline_training = True
    cfg.__post_init__()
    return cfg


if __name__ == "__main__":
    config = parse_args()
    if config.training.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available; use --device cpu"
    train(config)
