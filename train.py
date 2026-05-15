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
from demo_dataset import sample_oracle_prefix_states
from env_wrapper import FrankaKitchenImageWrapper
from specialist import SkillAgent, train_offline_skills
from utils import format_time, save_video


SEP = "=" * 76
SEP2 = "-" * 76
SEP3 = "." * 76


class TeeLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def write(self, text: str):
        print(text, end="")
        self.f.write(text)
        self.f.flush()

    def line(self, text: str = ""):
        self.write(text + "\n")

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
    results: Dict[str, float] = {}
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
    report = ["Single-task deterministic evaluation", f"episodes_per_task={n_episodes}", ""]

    for task_id, task_name in enumerate(agent.tasks):
        env = FrankaKitchenImageWrapper(
            tasks_to_complete=[task_name],
            img_size=config.encoder.img_size,
            terminate_on_tasks_completed=True,
        )
        success, options, rewards, final_errors = [], [], [], []
        best_rollout = None
        try:
            for ep in range(n_episodes):
                img, state = env.reset(seed=config.training.seed + 10_000 + 1000 * task_id + ep)
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
                            record_dir: Optional[str] = None) -> Dict[str, float]:
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
    order = task_order(agent, config.training.controller_order_mode)
    any_success, full_success, tasks_done, options_used, chosen_sr, rewards = [], [], [], [], [], []
    terminations: Dict[str, int] = {}

    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=True,
    )
    try:
        for ep in range(n_episodes):
            img, state = env.reset(seed=config.training.seed + 20_000 + ep)
            z = agent.encoder.encode_numpy(img).squeeze()
            completion = np.zeros(agent.n_tasks, dtype=np.float32)
            done = False
            n_opts = 0
            ep_reward = 0.0
            chosen_successes = 0
            frames = [img.copy()] if record_dir and ep < config.training.video_n_episodes else []

            while not done and completion.sum() < agent.n_tasks and n_opts < config.manager.max_high_level_steps:
                remaining = [k for k in order if completion[k] < 0.5]
                task_id = int(remaining[0]) if remaining else int(order[0])
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
                if frames and result.frames:
                    frames.extend(result.frames[1:])
                if not done:
                    img = env.render_image()

            done_count = int(completion.sum())
            any_success.append(float(done_count >= 1))
            full_success.append(float(done_count >= agent.n_tasks))
            tasks_done.append(float(done_count))
            options_used.append(float(n_opts))
            chosen_sr.append(float(chosen_successes / max(n_opts, 1)))
            rewards.append(float(ep_reward))
            if frames:
                save_video(frames, os.path.join(record_dir, f"ep_{ep:03d}.mp4"), fps=config.training.video_fps)
    finally:
        env.close()

    return {
        "eval/any_task_success_rate": float(np.mean(any_success)),
        "eval/full_task_success_rate": float(np.mean(full_success)),
        "eval/mean_tasks_completed": float(np.mean(tasks_done)),
        "eval/mean_options_used": float(np.mean(options_used)),
        "eval/mean_chosen_task_success": float(np.mean(chosen_sr)),
        "eval/mean_env_reward": float(np.mean(rewards)),
        "eval/std_env_reward": float(np.std(rewards)),
        "eval/termination_reasons": terminations,
    }


def evaluate_prefix(agent: SkillAgent,
                    config: Config,
                    prefix_tasks: Sequence[str],
                    target_task: str,
                    n_states: int,
                    record_dir: Optional[str] = None) -> Dict[str, float]:
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
            env.reset(seed=config.training.seed + 30_000 + i)
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


def print_banner(config: Config, log_path: str):
    print(f"\n{SEP}")
    print("  Lean Skill Learning — FrankaKitchen-v1")
    print(f"  Started        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Encoder        : {config.encoder.name.upper()} ({config.encoder.raw_dim}-d, frozen)")
    print(f"  Tasks          : {config.training.tasks_to_complete}")
    print(f"  Policy         : one visual BC/IQL policy per task")
    print(f"  Controller     : scripted next-incomplete task")
    print(SEP2)
    print(f"  Demos          : {config.warmup.dataset_ids}")
    print(f"  Cache          : {config.warmup.cache_dir}  rebuild={config.warmup.rebuild_cache}")
    print(f"  BC/IQL steps   : bc={config.specialist.n_teacher_bc_steps}  "
          f"iql={config.specialist.n_teacher_iql_steps}  batch={config.specialist.batch_size}")
    print(f"  Dense reward   : progress={config.worker.progress_weight}  "
          f"completion={config.worker.completion_bonus}  action_cost={config.worker.action_cost}")
    print(f"  Eval episodes  : single={config.eval.n_single_task_episodes}  chain={config.eval.n_eval_episodes}")
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
    logger = TeeLogger(log_path)
    import builtins
    orig_print = builtins.print
    builtins.print = lambda *args, **kwargs: logger.line(" ".join(str(a) for a in args))
    writer = SummaryWriter(config.training.log_dir)
    try:
        print_banner(config, log_path)
        agent = SkillAgent(config)
        t0 = time.time()
        stats = train_offline_skills(agent, config, verbose=True)
        print(f"\n  Stage A complete in {format_time(time.time() - t0)}.")
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"warmup/{k}", float(v), 0)

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
                if isinstance(v, (int, float)):
                    writer.add_scalar(k, float(v), 0)
            print(f"  Any-task success   : {chain['eval/any_task_success_rate']*100:5.1f}%")
            print(f"  Full-task success  : {chain['eval/full_task_success_rate']*100:5.1f}%")
            print(f"  Mean tasks done    : {chain['eval/mean_tasks_completed']:.2f}/{agent.n_tasks}")
            print(f"  Chosen-task SR     : {chain['eval/mean_chosen_task_success']*100:5.1f}%")
            print(f"  Termination mix    : {chain['eval/termination_reasons']}")

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
    parser.add_argument("--encoder", type=str, default="r3m", choices=["r3m", "dinov2"])
    parser.add_argument("--log_dir", type=str, default="logs/lean_skills")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--demo_datasets", nargs="+", default=None)
    parser.add_argument("--demo_source", type=str, default=None, choices=["auto", "minari", "d4rl"])
    parser.add_argument("--demo_cache_dir", type=str, default=None)
    parser.add_argument("--rebuild_demo_cache", action="store_true")
    parser.add_argument("--bc_steps", "--teacher_bc_steps", dest="bc_steps", type=int, default=None)
    parser.add_argument("--iql_steps", "--teacher_iql_steps", dest="iql_steps", type=int, default=None)
    parser.add_argument("--batch_size", "--specialist_batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--single_task_eval_episodes", type=int, default=None)
    parser.add_argument("--chain_eval_episodes", type=int, default=None)
    parser.add_argument("--controller_order_mode", type=str, default=None, choices=["given_order", "stage_a_rank"])
    parser.add_argument("--prefix_eval_only", action="store_true")
    parser.add_argument("--prefix_target_task", type=str, default="")
    parser.add_argument("--prefix_condition_tasks", nargs="*", default=None)
    parser.add_argument("--prefix_eval_states", type=int, default=None)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.training.mode = "lean_skills"
    cfg.training.seed = args.seed
    cfg.training.device = args.device
    cfg.training.log_dir = args.log_dir
    cfg.encoder.name = args.encoder
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
    if args.iql_steps is not None:
        cfg.specialist.n_teacher_iql_steps = args.iql_steps
    if args.batch_size is not None:
        cfg.specialist.batch_size = args.batch_size
    if args.single_task_eval_episodes is not None:
        cfg.eval.n_single_task_episodes = args.single_task_eval_episodes
    if args.chain_eval_episodes is not None:
        cfg.eval.n_eval_episodes = args.chain_eval_episodes
    if args.controller_order_mode is not None:
        cfg.training.controller_order_mode = args.controller_order_mode
    if args.prefix_eval_only:
        cfg.training.prefix_eval_only = True
        cfg.training.prefix_target_task = args.prefix_target_task
        cfg.training.prefix_condition_tasks = args.prefix_condition_tasks or []
    if args.prefix_eval_states is not None:
        cfg.training.prefix_eval_n_states = args.prefix_eval_states
    if args.no_video:
        cfg.training.record_video = False
    cfg.__post_init__()
    return cfg


if __name__ == "__main__":
    config = parse_args()
    if config.training.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available; use --device cpu"
    train(config)
