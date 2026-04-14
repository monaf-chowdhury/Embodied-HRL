"""
Main Training Loop: Visual HRL for FrankaKitchen-v1.

Run: python train.py [--seed 42] [--device cuda] [--tasks microwave]

Key changes:
  1. No projection head — warmup encodes directly with raw R3M.
     Phase 1b (pretraining) removed entirely.
  2. No SSE — episodes continue after subgoal failures.
     Episodes end only when env_done or max_high_steps reached.
  3. Manager reward uses compute_manager_reward() — task-completion dominant.
  4. Worker reward uses compute_worker_reward() — focused task-progress.
  5. task_delta stored in low buffer for landmark biasing.
  6. Landmark updates use get_task_biased_z() as candidate pool.
  7. Eval success = n_tasks_completed >= 1 (not ep_reward > 0).
  8. Encoder diagnosis printed after warmup.
"""

import os
import sys
import time
import datetime
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from env_wrapper import FrankaKitchenImageWrapper
from agent import VisualHRLAgent, compute_task_progress_focused, _task_progress
from utils import save_image, save_video


# =============================================================================
# Logger
# =============================================================================

class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"train_log_{ts}.txt")
        self.terminal = sys.stdout
        self.file     = open(path, "w", buffering=1)
        self.log_path = path
        sys.stdout    = self

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
    def __init__(self, total_steps: int, n_checkpoints: int = 10):
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
        return f"ckpt_{self.next_idx:02d}_step{steps}"


# =============================================================================
# Video recording
# =============================================================================

def record_episode_videos(agent, config, label: str, n_episodes: int = 3):
    if not config.training.record_video:
        return

    video_dir = os.path.join(config.training.log_dir, 'videos', label)
    os.makedirs(video_dir, exist_ok=True)

    rec_env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
    )

    for ep_i in range(n_episodes):
        frames    = []
        obs_img   = rec_env.reset()
        z_current = agent.encoder.encode_numpy(obs_img).squeeze()
        proprio   = rec_env.get_state()
        frames.append(obs_img)
        done       = False
        high_steps = 0

        while not done and high_steps < config.manager.max_high_steps:
            if not agent.landmarks.is_ready:
                break
            landmark_idx = agent.select_subgoal(z_current)
            z_subgoal    = agent.landmarks.get(landmark_idx)

            for _ in range(config.manager.subgoal_horizon):
                action   = agent.get_worker_action(
                    z_current, z_subgoal, proprio, deterministic=True)
                next_img, _, done, info = rec_env.step(action)
                proprio   = info['state']
                frames.append(next_img)
                z_current = agent.encoder.encode_numpy(next_img).squeeze()
                if done:
                    break
                if np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    break

            high_steps += 1

        save_video(frames, os.path.join(video_dir, f'ep_{ep_i:03d}.mp4'), fps=15)

    rec_env.close()
    print(f"  [Video] {n_episodes} episodes saved to {video_dir}/")


# =============================================================================
# Evaluation — unambiguous success metric: n_tasks_completed >= 1
# =============================================================================

def evaluate(agent, config, n_episodes=None):
    n_episodes = n_episodes or config.training.n_eval_episodes

    eval_env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
    )

    task_completions, rewards, high_steps_list, tasks_done_list = [], [], [], []

    for _ in range(n_episodes):
        obs_img   = eval_env.reset()
        z_current = agent.encoder.encode_numpy(obs_img).squeeze()
        proprio   = eval_env.get_state()
        ep_reward = 0.0
        done      = False
        high_steps = 0
        max_tasks  = 0

        while not done and high_steps < config.manager.max_high_steps:
            if not agent.landmarks.is_ready:
                break
            with torch.no_grad():
                landmark_idx = agent.select_subgoal(z_current)
            z_subgoal = agent.landmarks.get(landmark_idx)

            for _ in range(config.manager.subgoal_horizon):
                action   = agent.get_worker_action(
                    z_current, z_subgoal, proprio, deterministic=True)
                next_img, reward, done, info = eval_env.step(action)
                proprio   = info['state']
                z_current = agent.encoder.encode_numpy(next_img).squeeze()
                ep_reward += reward
                max_tasks  = max(max_tasks, info.get('n_tasks_completed', 0))
                if done:
                    break
                if np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    break

            high_steps += 1

        rewards.append(ep_reward)
        high_steps_list.append(high_steps)
        tasks_done_list.append(max_tasks)
        # Unambiguous success: at least one task actually completed
        task_completions.append(max_tasks >= 1)

    eval_env.close()
    return {
        'task_completion_rate': float(np.mean(task_completions)),  # PRIMARY metric
        'mean_reward':          float(np.mean(rewards)),
        'std_reward':           float(np.std(rewards)),
        'mean_high_steps':      float(np.mean(high_steps_list)),
        'mean_tasks_completed': float(np.mean(tasks_done_list)),
        'n_episodes':           n_episodes,
    }


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(agent, config, name: str):
    path = os.path.join(config.training.log_dir, f'checkpoint_{name}.pt')
    torch.save({
        'manager_q':            agent.manager_q.state_dict(),
        'manager_q_target':     agent.manager_q_target.state_dict(),
        'worker_actor':         agent.worker_actor.state_dict(),
        'worker_critic':        agent.worker_critic.state_dict(),
        'worker_critic_target': agent.worker_critic_target.state_dict(),
        'reachability':         agent.reachability.state_dict(),
        'total_steps':          agent.total_steps,
        'total_episodes':       agent.total_episodes,
        'success_threshold':    agent.success_threshold,
        'epsilon':              agent.epsilon,
        'proprio_mean':         agent.low_buffer._p_mean,
        'proprio_m2':           agent.low_buffer._p_M2,
        'proprio_n':            agent.low_buffer._p_n,
    }, path)
    print(f"  [Checkpoint] Saved: {os.path.basename(path)}")


# =============================================================================
# Main training loop
# =============================================================================

def train(config: Config):
    logger = Logger(config.training.log_dir)
    print(f"Training log: {logger.log_path}")

    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)

    os.makedirs(config.training.log_dir, exist_ok=True)
    writer     = SummaryWriter(config.training.log_dir)
    ckpt_sched = CheckpointScheduler(config.training.total_timesteps)

    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        seed=config.training.seed,
        terminate_on_tasks_completed=False,
    )

    agent = VisualHRLAgent(config)

    # ---- Header ----
    sep = "=" * 70
    print(sep)
    print("  Visual HRL — FrankaKitchen-v1")
    print(sep)
    print(f"  Started:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device:         {config.training.device}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    if torch.cuda.is_available():
        print(f"  VRAM:           {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Encoder:        {config.encoder.name} ({config.encoder.raw_dim}-d, L2-normed)")
    print(f"  Tasks:          {config.training.tasks_to_complete}")
    print(f"  Landmarks:      {config.landmarks.n_landmarks}")
    print(f"  Subgoal K:      {config.manager.subgoal_horizon}")
    print(f"  Max high steps: {config.manager.max_high_steps} per episode")
    print(f"  SSE:            DISABLED (episodes continue after failures)")
    print(f"  Reachability:   DISABLED (reject_threshold=0)")
    print(f"  Manager reward: task_completion={config.manager.task_completion_bonus}  "
          f"task_progress={config.manager.task_progress_bonus}  "
          f"nav={config.manager.nav_bonus}")
    print(f"  Worker reward:  sparse={config.reward.sparse_weight}  "
          f"task_prog={config.reward.task_progress_weight}  "
          f"latent={config.reward.latent_weight}")
    print(f"  Total steps:    {config.training.total_timesteps:,}")
    print(f"  Log dir:        {config.training.log_dir}")
    print(sep)

    # =========================================================================
    # Phase 1: Random exploration warmup — raw R3M, no pretraining needed
    # =========================================================================
    print(f"\n[Phase 1] Random warmup ({config.training.n_warmup} episodes)...")
    p1_start = time.time()
    K_future = config.training.warmup_future_k

    for ep_i in range(config.training.n_warmup):
        obs_img    = env.reset()
        ep_imgs    = [obs_img]
        ep_proprios = [env.get_state()]
        ep_actions = []
        ep_rewards = []
        ep_dones   = []
        ep_n_tasks = []
        done = False
        while not done:
            action = env.action_space.sample()
            nxt_img, reward, done, info = env.step(action)
            ep_imgs.append(nxt_img)
            ep_proprios.append(info['state'])
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_n_tasks.append(info.get('n_tasks_completed', 0))

        T = len(ep_actions)

        # Encode directly with raw R3M (no projection, no caching needed)
        all_z = np.stack([
            agent.encoder.encode_numpy(img).squeeze()
            for img in ep_imgs
        ])  # (T+1, 2048)

        for t in range(T):
            z_t       = all_z[t]
            z_next_t  = all_z[t + 1]
            proprio_t = ep_proprios[t]
            proprio_n = ep_proprios[t + 1]
            future_idx = min(t + K_future, T)
            z_sub     = all_z[future_idx]
            init_dist  = float(np.linalg.norm(z_t - z_sub))

            shaped, task_delta = agent.compute_worker_reward(
                z_t, z_next_t, z_sub, ep_rewards[t],
                proprio_t, proprio_n, initial_dist=init_dist,
            )
            agent.low_buffer.add(
                z_t, proprio_t, z_sub, ep_actions[t],
                shaped, z_next_t, proprio_n, ep_dones[t],
                task_delta=task_delta,
            )
            agent._latent_dists.append(float(np.linalg.norm(z_next_t - z_t)))

            if ep_n_tasks[t] > (ep_n_tasks[t - 1] if t > 0 else 0):
                agent.landmarks.add_success_state(z_next_t)

        if (ep_i + 1) % 50 == 0:
            print(f"  Warmup {ep_i+1}/{config.training.n_warmup} "
                  f"| buffer: {len(agent.low_buffer):,} | ep_len: {T}")

    p1_elapsed = (time.time() - p1_start) / 60
    print(f"  Phase 1 done in {p1_elapsed:.1f} min")
    print(f"  Collected {len(agent.low_buffer):,} transitions (K_future={K_future})")

    # Diagnose encoder distances
    print("\n[Encoder Diagnosis]")
    diag = agent.encoder.diagnose_distances(env, n_steps=200)

    # Initial landmarks
    all_z_replay = agent.low_buffer.get_all_z()
    z_task_biased = agent.low_buffer.get_task_biased_z(
        top_pct=config.landmarks.task_progress_top_pct
        if hasattr(config.landmarks, 'task_progress_top_pct') else 0.3
    )
    agent.landmarks.update(z_replay=all_z_replay, z_task_biased=z_task_biased)
    print(f"  Computed {agent.landmarks.n_active} landmarks via FPS")

    # Demo landmark seeding
    if config.landmarks.use_demo_landmarks:
        agent.landmarks.seed_from_demo(
            agent.encoder,
            gif_path=config.landmarks.demo_gif_path,
            max_frames=config.landmarks.demo_max_frames,
        )
        agent.landmarks.update(z_replay=all_z_replay, z_task_biased=z_task_biased)
        print(f"  Re-computed {agent.landmarks.n_active} landmarks (demo + task-biased replay)")

    agent.calibrate_success_threshold()

    # =========================================================================
    # Phase 2: Hierarchical training — no SSE
    # =========================================================================
    print(f"\n[Phase 2] Hierarchical training (no SSE)...")

    best_eval   = 0.0
    train_start = time.time()
    run_rewards, run_task_comp, run_high_steps, run_subgoal_sr = [], [], [], []
    last_worker, last_manager, last_reach = {}, {}, {}

    pbar = tqdm(total=config.training.total_timesteps, desc="Training", dynamic_ncols=True)

    while agent.total_steps < config.training.total_timesteps:

        obs_img      = env.reset()
        z_current    = agent.encoder.encode_numpy(obs_img).squeeze()
        proprio      = env.get_state()
        ep_reward    = 0.0
        ep_high_steps = 0
        ep_successes  = 0
        ep_attempts   = 0
        ep_tasks_done = 0
        episode_done  = False
        agent._prev_n_tasks = 0

        while not episode_done:

            # ---- Manager selects subgoal ----
            landmark_idx = agent.select_subgoal(z_current)
            z_subgoal    = agent.landmarks.get(landmark_idx)

            z_start           = z_current.copy()
            proprio_start     = proprio.copy()
            cumulative_reward = 0.0
            subgoal_reached   = False
            ep_attempts      += 1
            initial_dist      = float(np.linalg.norm(z_current - z_subgoal))
            n_tasks_before    = agent._prev_n_tasks

            # Track task progress over this subgoal attempt
            nearest_task_start, prog_start = compute_task_progress_focused(
                proprio_start, config.training.tasks_to_complete)
            n_tasks_after_attempt = n_tasks_before

            for _ in range(config.manager.subgoal_horizon):
                action   = agent.get_worker_action(z_current, z_subgoal, proprio)
                nxt_img, env_reward, env_done, info = env.step(action)
                z_next    = agent.encoder.encode_numpy(nxt_img).squeeze()
                proprio_n = info['state']
                n_tasks   = info.get('n_tasks_completed', 0)

                agent.maybe_inject_hindsight(z_next, n_tasks)
                n_tasks_after_attempt = max(n_tasks_after_attempt, n_tasks)

                shaped, task_delta = agent.compute_worker_reward(
                    z_current, z_next, z_subgoal, env_reward,
                    proprio, proprio_n, initial_dist=initial_dist,
                )
                agent.low_buffer.add(
                    z_current, proprio, z_subgoal, action,
                    shaped, z_next, proprio_n, env_done,
                    task_delta=task_delta,
                )

                cumulative_reward += env_reward
                ep_reward         += env_reward
                agent.total_steps += 1
                pbar.update(1)
                z_current = z_next
                proprio   = proprio_n
                obs_img   = nxt_img

                if np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    subgoal_reached = True
                    break

                if env_done:
                    episode_done = True
                    break

                if (agent.total_steps % 4 == 0
                        and agent.low_buffer.size > config.buffer.batch_size):
                    last_worker = agent.update_worker()

            # ---- Manager reward (task-aware) ----
            nearest_task_end, prog_end = compute_task_progress_focused(
                proprio, config.training.tasks_to_complete)
            # Use same task as start for consistent delta
            if nearest_task_start:
                prog_end_same = _task_progress(proprio, nearest_task_start)
                task_prog_delta = prog_end_same - prog_start
            else:
                task_prog_delta = 0.0

            manager_reward = agent.compute_manager_reward(
                n_tasks_before=n_tasks_before,
                n_tasks_after=n_tasks_after_attempt,
                task_progress_before=prog_start,
                task_progress_after=prog_start + task_prog_delta,
                cumulative_env_reward=cumulative_reward,
                landmark_reached=subgoal_reached,
            )

            # Store manager transition (always — no SSE termination)
            agent.high_buffer.add(
                z_start, z_subgoal, manager_reward, z_current,
                done=float(episode_done), landmark_idx=landmark_idx,
            )

            # ---- Bookkeeping ----
            ep_high_steps += 1
            ep_tasks_done  = max(ep_tasks_done, n_tasks_after_attempt)
            agent.reach_buffer.add(z_start, z_subgoal, subgoal_reached)
            agent.landmarks.record_visit(landmark_idx, success=subgoal_reached)

            if subgoal_reached:
                ep_successes += 1

            if agent.high_buffer.size > config.buffer.batch_size:
                last_manager = agent.update_manager()

            agent._update_epsilon()

            # Episode ends only on env_done or max_high_steps
            if ep_high_steps >= config.manager.max_high_steps:
                episode_done = True

        # ---- End of episode ----
        agent.total_episodes += 1
        run_rewards.append(ep_reward)
        run_task_comp.append(ep_tasks_done >= 1)
        run_high_steps.append(ep_high_steps)
        run_subgoal_sr.append(ep_successes / max(ep_attempts, 1))

        # Reachability update (disabled by config but runs anyway for potential future use)
        if agent.total_episodes % config.reachability.update_freq == 0:
            for _ in range(3):
                last_reach = agent.update_reachability()

        # Landmark update — task-progress-biased
        if agent.total_episodes % config.landmarks.update_freq == 0:
            all_z_replay  = agent.low_buffer.get_all_z()
            z_task_biased = agent.low_buffer.get_task_biased_z(
                top_pct=getattr(config.landmarks, 'task_progress_top_pct', 0.3)
            )
            if len(all_z_replay) > config.landmarks.min_observations:
                agent.landmarks.update(
                    z_replay=all_z_replay,
                    z_task_biased=z_task_biased,
                )

        # TensorBoard
        if agent.total_episodes % 10 == 0:
            writer.add_scalar('train/episode_reward',       ep_reward,      agent.total_steps)
            writer.add_scalar('train/task_completion',      float(ep_tasks_done >= 1), agent.total_steps)
            writer.add_scalar('train/tasks_completed',      ep_tasks_done,  agent.total_steps)
            writer.add_scalar('train/high_level_steps',     ep_high_steps,  agent.total_steps)
            writer.add_scalar('train/subgoal_success_rate',
                              ep_successes / max(ep_attempts, 1), agent.total_steps)
            writer.add_scalar('train/epsilon',              agent.epsilon,  agent.total_steps)
            writer.add_scalar('train/low_buffer_size',      len(agent.low_buffer), agent.total_steps)
            writer.add_scalar('train/high_buffer_size',     len(agent.high_buffer), agent.total_steps)
            writer.add_scalar('train/success_threshold',    agent.success_threshold, agent.total_steps)
            writer.add_scalar('train/hindsight_pool_size',
                              agent.landmarks._success_pool_size, agent.total_steps)
            for k, v in last_worker.items():
                writer.add_scalar(f'worker/{k}',       v, agent.total_steps)
            for k, v in last_manager.items():
                writer.add_scalar(f'manager/{k}',      v, agent.total_steps)
            for k, v in last_reach.items():
                writer.add_scalar(f'reachability/{k}', v, agent.total_steps)

        # Terminal log every 50 episodes
        if agent.total_episodes % 50 == 0 and run_rewards:
            elapsed = time.time() - train_start
            sps     = agent.total_steps / max(elapsed, 1)
            eta     = str(datetime.timedelta(
                seconds=int((config.training.total_timesteps - agent.total_steps) / max(sps, 1))))
            pct     = 100 * agent.total_steps / config.training.total_timesteps

            print(f"\n{'─'*70}")
            print(f"  Step {agent.total_steps:>9,} / {config.training.total_timesteps:,}"
                  f"  ({pct:.1f}%)    Ep {agent.total_episodes:,}")
            print(f"  Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))}  "
                  f"ETA: {eta}  Speed: {sps:.1f} sps")
            print(f"{'─'*70}")
            n = len(run_rewards)
            print(f"  [Last {n} episodes]")
            print(f"    Reward:           mean={np.mean(run_rewards):.4f}  "
                  f"max={np.max(run_rewards):.4f}")
            print(f"    Task completion:  {np.mean(run_task_comp)*100:.1f}%  (≥1 task done)")
            print(f"    High-level steps: mean={np.mean(run_high_steps):.1f}")
            print(f"    Subgoal SR:       mean={np.mean(run_subgoal_sr)*100:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.4f}  "
                  f"Landmarks: {agent.landmarks.n_active}  "
                  f"Hindsight: {agent.landmarks._success_pool_size}  "
                  f"Threshold: {agent.success_threshold:.4f}")
            print(f"  Buffers: low={len(agent.low_buffer):,}  "
                  f"high={len(agent.high_buffer):,}")
            if last_worker:
                print(f"  Worker: critic={last_worker.get('worker_critic_loss',0):.5f}  "
                      f"actor={last_worker.get('worker_actor_loss',0):.5f}  "
                      f"alpha={last_worker.get('worker_alpha',0):.4f}")
            if last_manager:
                print(f"  Manager: loss={last_manager.get('manager_loss',0):.5f}  "
                      f"q_mean={last_manager.get('manager_q_mean',0):.4f}")
            run_rewards.clear(); run_task_comp.clear()
            run_high_steps.clear(); run_subgoal_sr.clear()

        # Evaluation
        if agent.total_steps % config.training.eval_freq < config.manager.subgoal_horizon * 20:
            t0     = time.time()
            result = evaluate(agent, config)
            # Primary metric: task_completion_rate (not ep_reward > 0)
            is_best = result['task_completion_rate'] > best_eval
            if is_best:
                best_eval = result['task_completion_rate']
                save_checkpoint(agent, config, 'best')
                record_episode_videos(agent, config, 'best', config.training.video_n_episodes)
            for k, v in result.items():
                if isinstance(v, float):
                    writer.add_scalar(f'eval/{k}', v, agent.total_steps)
            print(f"\n{'='*70}")
            print(f"  [EVAL] Step {agent.total_steps:,}  "
                  f"({result['n_episodes']} eps, {time.time()-t0:.1f}s)")
            print(f"    Task completion:    {result['task_completion_rate']*100:.1f}%"
                  + (" <-- NEW BEST" if is_best else
                     f"  (best: {best_eval*100:.1f}%)"))
            print(f"    Mean reward:        {result['mean_reward']:.4f} "
                  f"± {result['std_reward']:.4f}")
            print(f"    Mean tasks done:    {result['mean_tasks_completed']:.2f}"
                  f" / {len(config.training.tasks_to_complete)}")
            print(f"    Mean high steps:    {result['mean_high_steps']:.1f}")
            print(f"{'='*70}")

        # Periodic checkpoint
        if ckpt_sched.should_save(agent.total_steps):
            label = ckpt_sched.label(agent.total_steps)
            save_checkpoint(agent, config, label)
            record_episode_videos(agent, config, label, config.training.video_n_episodes)

    pbar.close()
    writer.close()
    total = time.time() - train_start
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Time:     {str(datetime.timedelta(seconds=int(total)))}")
    print(f"  Steps:    {agent.total_steps:,}")
    print(f"  Episodes: {agent.total_episodes:,}")
    print(f"  Best task completion rate: {best_eval*100:.1f}%")
    print(f"  Log:      {logger.log_path}")
    print(f"{'='*70}")
    logger.close()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',              type=int,  default=42)
    parser.add_argument('--device',            type=str,  default='cuda')
    parser.add_argument('--total_steps',       type=int,  default=None)
    parser.add_argument('--encoder',           type=str,  default='r3m',
                        choices=['r3m', 'dinov2'])
    parser.add_argument('--n_landmarks',       type=int,  default=None)
    parser.add_argument('--subgoal_horizon',   type=int,  default=None)
    parser.add_argument('--log_dir',           type=str,  default=None)
    parser.add_argument('--warmup_k',          type=int,  default=None)
    parser.add_argument('--tasks',             type=str,  nargs='+', default=None)
    parser.add_argument('--no_demo_landmarks', action='store_true')
    parser.add_argument('--no_hindsight',      action='store_true')
    parser.add_argument('--no_video',          action='store_true')
    parser.add_argument('--demo_gif',          type=str,  default=None)
    args = parser.parse_args()

    config = Config()  # all defaults from config.py

    if args.seed            is not None: config.training.seed              = args.seed
    if args.device          is not None: config.training.device            = args.device
    if args.total_steps     is not None: config.training.total_timesteps   = args.total_steps
    if args.log_dir         is not None: config.training.log_dir           = args.log_dir
    if args.tasks           is not None: config.training.tasks_to_complete = args.tasks
    if args.warmup_k        is not None: config.training.warmup_future_k   = args.warmup_k
    if args.encoder         is not None: config.encoder.name               = args.encoder
    if args.n_landmarks     is not None: config.landmarks.n_landmarks      = args.n_landmarks
    if args.subgoal_horizon is not None: config.manager.subgoal_horizon    = args.subgoal_horizon
    if args.demo_gif        is not None: config.landmarks.demo_gif_path    = args.demo_gif

    if args.no_video:          config.training.record_video             = False
    if args.no_demo_landmarks: config.landmarks.use_demo_landmarks      = False
    if args.no_hindsight:      config.landmarks.use_hindsight_landmarks = False

    if config.training.device == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available!"
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train(config)