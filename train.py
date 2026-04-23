"""
train.py — SMGW training: Stage A (task-grounded warmup) -> Stage B (joint HRL).

Usage:
    python train.py [--seed 42] [--device cuda] [--log_dir logs/run1]
    python train.py --total_steps 1500000 --action_chunk 4
    python train.py --no_warmup_demo --no_video

Design notes:
  - Stage A: warmup.run_stage_a_warmup fills the worker buffer, BC-trains
    the worker actor, and CE-trains the manager with KNOWN labels.
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
import os
import sys
import time
import datetime
import argparse
import numpy as np
import torch
from __future__ import annotations
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter

from config import Config
from env_wrapper import FrankaKitchenImageWrapper
from agent import SMGWAgent
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
    * mean_options_used         — average number of options (manager decisions)
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

        img, state = env.reset()
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

            task_id = agent.select_task(z, proprio, state, completion,
                                        deterministic=deterministic)

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
    print(f"  Manager        : 4-way discrete, completion-mask gated")
    print(f"  Worker         : SAC, FiLM-conditioned, "
          f"chunk_len = {config.worker.action_chunk_len}")
    print(f"  Subgoal K      : {config.manager.subgoal_horizon} env steps / option")
    print(f"  Max HL steps   : {config.manager.max_high_level_steps} per episode")
    print(SEP2)
    print(f"  Manager reward : completion={config.manager.completion_bonus}  "
          f"dense={config.manager.dense_shaping_weight}  "
          f"option_cost={config.manager.option_cost}  "
          f"all_done={config.manager.all_done_bonus}")
    print(f"  Worker reward  : progress={config.worker.progress_weight}  "
          f"completion={config.worker.completion_bonus}  "
          f"action_cost={config.worker.action_cost}  "
          f"failure={config.worker.failure_penalty}")
    print(f"  NOTE: Zero latent-distance terms anywhere. All rewards are "
          f"grounded in benchmark completion bits and task-space errors.")
    print(SEP2)
    print(f"  Stage A probes : {config.warmup.n_probe_episodes_per_task} ep/task "
          f"* {config.warmup.probe_steps_per_episode} steps")
    print(f"  Stage A BC     : worker={config.warmup.n_worker_sl_steps} steps, "
          f"manager={config.warmup.n_manager_sl_steps} steps")
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

    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)

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
    agent = SMGWAgent(config)

    print_start_banner(config, logger.log_path)

    # =========================================================================
    # Stage A: task-grounded warmup
    # =========================================================================
    print(SEP2)
    print(f"  STAGE A  —  Task-Grounded Warmup")
    print(SEP2 + "\n")

    t0 = time.time()
    warmup_stats = run_stage_a_warmup(agent, config, verbose=True)
    warmup_elapsed = time.time() - t0
    print(f"\n  Stage A complete in {format_time(warmup_elapsed)}.")
    for k, v in warmup_stats.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(f'warmup/{k}', v, 0)
    print(f"{SEP}\n")

    # =========================================================================
    # Stage B: joint HRL training
    # =========================================================================
    print(SEP2)
    print(f"  STAGE B  —  Joint HRL (manager DQN + worker SAC)")
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

    # Explicit eval trigger — fire when total_env_steps crosses next_eval_at
    next_eval_at = config.eval.eval_every_env_steps

    # Worker online update cadence: update every N env steps
    updates_per_step = config.training.worker_updates_per_env_step
    update_every_n = max(1, int(round(1.0 / max(updates_per_step, 1e-6))))

    while agent.total_env_steps < config.training.total_env_steps:

        # ---- Episode reset ----
        img, state = env.reset()
        z = agent.encoder.encode_numpy(img).squeeze()
        proprio = state.copy()
        completion = np.zeros(agent.n_tasks, dtype=np.float32)

        ep_env_reward = 0.0
        ep_tasks_completed = 0
        ep_options = 0
        env_done = False

        # ---- Option loop ----
        while (not env_done
               and ep_options < config.manager.max_high_level_steps
               and completion.sum() < agent.n_tasks):

            # -- Manager picks a task --
            task_id = agent.select_task(z, proprio, state, completion,
                                        deterministic=False)

            # -- Worker executes the option --
            result = agent.execute_option(
                env=env,
                task_id=task_id,
                start_img=img, start_state=state, start_z=z,
                completion=completion,
                deterministic_worker=False,
                collect_frames=False,
                train_worker_online=True,
                update_every_n_env_steps=update_every_n,
            )

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

            # -- Manager online update --
            for _ in range(config.training.manager_updates_per_option):
                loss = agent.update_manager()
                if loss:
                    last_manager_losses = loss

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

        # ---- Per-episode TB scalars ----
        if agent.total_episodes % config.training.tb_every_episodes == 0:
            s = agent.total_env_steps
            writer.add_scalar('train/ep_env_reward', ep_env_reward, s)
            writer.add_scalar('train/ep_any_success', float(ep_tasks_completed >= 1), s)
            writer.add_scalar('train/ep_full_success',
                              float(ep_tasks_completed >= agent.n_tasks), s)
            writer.add_scalar('train/ep_tasks_completed', ep_tasks_completed, s)
            writer.add_scalar('train/ep_options', ep_options, s)
            writer.add_scalar('train/epsilon', agent.epsilon, s)
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
            print(f"  Exploration  epsilon={agent.epsilon:.3f}")
            print(f"  Buffers      worker={len(agent.worker_buf):>7,}  "
                  f"manager={len(agent.manager_buf):>6,}")
            if last_worker_losses:
                print(f"  Worker (SAC)  critic={last_worker_losses.get('worker_critic_loss', 0):.5f}  "
                      f"actor={last_worker_losses.get('worker_actor_loss', 0):.4f}  "
                      f"alpha={last_worker_losses.get('worker_alpha', 0):.4f}")
            if last_manager_losses:
                print(f"  Manager (DQN) loss={last_manager_losses.get('manager_loss', 0):.5f}  "
                      f"q_mean={last_manager_losses.get('manager_q_mean', 0):.3f}")

            run_env_rewards.clear()
            run_any_success.clear()
            run_full_success.clear()
            run_tasks_completed.clear()
            run_options_per_ep.clear()

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
    parser.add_argument('--no_warmup_demo', action='store_true',
                        help='Skip loading the demo GIF during Stage A.')
    parser.add_argument('--demo_gif', type=str, default=None)
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--n_probe_eps', type=int, default=None,
                        help='Per-task probe episodes in Stage A.')
    args = parser.parse_args()

    config = Config()
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
    if args.demo_gif is not None:
        config.warmup.demo_gif_path = args.demo_gif
    if args.no_video:
        config.training.record_video = False
    if args.no_warmup_demo:
        config.warmup.use_demo_gif_for_context = False
    if args.n_probe_eps is not None:
        config.warmup.n_probe_episodes_per_task = args.n_probe_eps

    config.__post_init__()  # re-run to update encoder raw_dim if needed

    if config.training.device == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available — use --device cpu"

    train(config)
