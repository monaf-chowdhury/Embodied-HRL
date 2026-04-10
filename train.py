"""
Main Training Loop: Visual HRL for FrankaKitchen-v1.

Run: python train.py [--seed 42] [--device cuda] [--tasks microwave]

Key changes vs original:
  - z_goal / goal image REMOVED everywhere.
  - Warmup and Phase 2 both pass proprio to the worker and buffer.
  - Demo landmark seeding after Phase 1 warmup.
  - Hindsight injection on every task-completion event.
  - Landmark update passes z_goal=None (no goal image).
  - Episode video recording at every checkpoint save.
  - evaluate() updated: no z_goal, passes proprio.
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
from env_wrapper import FrankaKitchenImageWrapper, HierarchicalKitchenWrapper
from agent import VisualHRLAgent
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
        return f"periodic_{self.next_idx:02d}_step{steps}"


# =============================================================================
# Video recording
# =============================================================================

def record_episode_videos(agent, config, label: str, n_episodes: int = 3):
    """
    Run n_episodes deterministically and save each as an MP4 in
    logs/videos/<label>/.
    """
    if not config.training.record_video:
        return

    video_dir = os.path.join(config.training.log_dir, 'videos', label)
    os.makedirs(video_dir, exist_ok=True)

    record_env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
    )

    for ep_i in range(n_episodes):
        frames    = []
        obs_img   = record_env.reset()
        z_current = agent.encoder.encode_numpy(obs_img).squeeze()
        proprio   = record_env.get_state()
        frames.append(obs_img)
        done       = False
        high_steps = 0

        while not done and high_steps < 15:
            if not agent.landmarks.is_ready:
                break
            landmark_idx = agent.select_subgoal(z_current)
            z_subgoal    = agent.landmarks.get(landmark_idx)

            for _ in range(config.manager.subgoal_horizon):
                action   = agent.get_worker_action(
                    z_current, z_subgoal, proprio, deterministic=True)
                next_img, _, done, info = record_env.step(action)
                proprio  = info['state']
                frames.append(next_img)
                z_next   = agent.encoder.encode_numpy(next_img).squeeze()
                z_current = z_next
                if done:
                    break
                if np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    break

            high_steps += 1

        video_path = os.path.join(video_dir, f'ep_{ep_i:03d}.mp4')
        save_video(frames, video_path, fps=15)

    record_env.close()
    print(f"  [Video] {n_episodes} episodes saved to {video_dir}/")


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(agent, config, n_episodes=None):
    """Evaluate agent deterministically. Returns dict of metrics."""
    n_episodes = n_episodes or config.training.n_eval_episodes

    eval_env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
    )

    successes, rewards, high_steps_list, tasks_completed_list = [], [], [], []

    for _ in range(n_episodes):
        obs_img   = eval_env.reset()
        z_current = agent.encoder.encode_numpy(obs_img).squeeze()
        proprio   = eval_env.get_state()
        ep_reward = 0.0
        done      = False
        high_steps = 0
        max_tasks_completed = 0

        while not done and high_steps < 15:
            if not agent.landmarks.is_ready:
                break
            with torch.no_grad():
                landmark_idx = agent.select_subgoal(z_current)
            z_subgoal = agent.landmarks.get(landmark_idx)

            for _ in range(config.manager.subgoal_horizon):
                action   = agent.get_worker_action(
                    z_current, z_subgoal, proprio, deterministic=True)
                next_img, reward, done, info = eval_env.step(action)
                proprio  = info['state']
                z_next   = agent.encoder.encode_numpy(next_img).squeeze()
                ep_reward += reward
                z_current  = z_next
                max_tasks_completed = max(
                    max_tasks_completed, info.get('n_tasks_completed', 0))

                if np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    break
                if done:
                    break

            high_steps += 1
            # Strict execution: if subgoal not reached, stop
            if np.linalg.norm(z_current - z_subgoal) >= agent.success_threshold:
                break

        rewards.append(ep_reward)
        high_steps_list.append(high_steps)
        tasks_completed_list.append(max_tasks_completed)
        successes.append(ep_reward > 0)

    eval_env.close()
    return {
        'success_rate':         float(np.mean(successes)),
        'mean_reward':          float(np.mean(rewards)),
        'std_reward':           float(np.std(rewards)),
        'mean_high_steps':      float(np.mean(high_steps_list)),
        'mean_tasks_completed': float(np.mean(tasks_completed_list)),
        'n_episodes':           n_episodes,
    }


# =============================================================================
# Checkpoint save
# =============================================================================

def save_checkpoint(agent, config, name: str):
    path = os.path.join(config.training.log_dir, f'checkpoint_{name}.pt')
    torch.save({
        'encoder_projection':   agent.encoder.projection.state_dict(),
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
        # Save proprio running stats for reproducibility
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
    ckpt_sched = CheckpointScheduler(config.training.total_timesteps, n_checkpoints=10)

    # ---- Environment ----
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        seed=config.training.seed,
        terminate_on_tasks_completed=False,
    )

    # ---- Agent ----
    agent = VisualHRLAgent(config)

    # ---- Print header ----
    sep = "=" * 70
    print(sep)
    print("  Visual HRL Training — FrankaKitchen-v1")
    print(sep)
    print(f"  Started:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device:        {config.training.device}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    if torch.cuda.is_available():
        print(f"  VRAM:          "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Seed:          {config.training.seed}")
    print(f"  Encoder:       {config.encoder.name} (proj_dim={config.encoder.proj_dim})")
    print(f"  Tasks:         {config.training.tasks_to_complete}")
    print(f"  Landmarks:     {config.landmarks.n_landmarks}")
    print(f"    use_demo:     {config.landmarks.use_demo_landmarks}")
    print(f"    use_hindsight:{config.landmarks.use_hindsight_landmarks}")
    print(f"    use_curriculum:{config.landmarks.use_curriculum_landmarks}")
    print(f"  Subgoal K:     {config.manager.subgoal_horizon}")
    print(f"  Reward weights: sparse={config.reward.sparse_weight}  "
          f"task_prog={config.reward.task_progress_weight}  "
          f"latent={config.reward.latent_weight}")
    print(f"  Total steps:   {config.training.total_timesteps:,}")
    print(f"  Record video:  {config.training.record_video}")
    print(f"  Log dir:       {config.training.log_dir}")
    print(sep)

    # =========================================================================
    # Phase 1: Random exploration with K-step future subgoals
    # =========================================================================
    print("\n[Phase 1] Random exploration with K-step future subgoals...")
    p1_start = time.time()
    K_future = config.training.warmup_future_k

    n_warmup = 50
    for ep_i in range(n_warmup):

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
            next_img, reward, done, info = env.step(action)
            ep_imgs.append(next_img)
            ep_proprios.append(info['state'])
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_n_tasks.append(info.get('n_tasks_completed', 0))

        T = len(ep_actions)

        # Encode all frames at once
        all_z = np.stack([
            agent.encoder.encode_numpy(img).squeeze()
            for img in ep_imgs
        ])  # (T+1, z_dim)

        # Store transitions with K-step future subgoal
        for t in range(T):
            z_t       = all_z[t]
            z_next    = all_z[t + 1]
            proprio_t = ep_proprios[t]
            proprio_n = ep_proprios[t + 1]
            future_idx = min(t + K_future, T)
            z_sub     = all_z[future_idx]
            # Initial dist for normalised shaping
            init_dist  = np.linalg.norm(z_t - z_sub)

            shaped = agent.compute_shaped_reward(
                z_t, z_next, z_sub, ep_rewards[t],
                proprio_t, proprio_n, initial_dist=init_dist,
            )
            agent.low_buffer.add(z_t, proprio_t, z_sub, ep_actions[t],
                                  shaped, z_next, proprio_n, ep_dones[t])
            agent._latent_dists.append(np.linalg.norm(z_next - z_t))

            # Hindsight injection during warmup too
            if ep_n_tasks[t] > (ep_n_tasks[t - 1] if t > 0 else 0):
                agent.landmarks.add_success_state(z_next)

        if (ep_i + 1) % 10 == 0:
            print(f"  Warmup {ep_i+1}/{n_warmup} | buffer: {len(agent.low_buffer):,} "
                  f"| last ep len: {T} steps")

    print(f"  Phase 1 done in {(time.time()-p1_start)/60:.1f} min")
    print(f"  Collected {len(agent.low_buffer):,} transitions (K_future={K_future})")

    # ---- Initial landmark computation ----
    all_z = agent.low_buffer.get_all_z()
    agent.landmarks.update(all_z, z_goal=None)
    print(f"  Computed {agent.landmarks.n_active} landmarks via FPS")

    # ---- Demo landmark seeding ----
    if config.landmarks.use_demo_landmarks:
        agent.landmarks.seed_from_demo(
            agent.encoder,
            gif_path=config.landmarks.demo_gif_path,
            max_frames=config.landmarks.demo_max_frames,
        )
        # Recompute landmarks now that demo latents are loaded
        agent.landmarks.update(all_z, z_goal=None)
        print(f"  Re-computed {agent.landmarks.n_active} landmarks (demo + replay)")

    agent.calibrate_success_threshold()

    # =========================================================================
    # Phase 2: Hierarchical training
    # =========================================================================
    print(f"\n[Phase 2] Hierarchical training loop...")

    best_eval   = 0.0
    train_start = time.time()
    run_rewards, run_successes, run_high_steps, run_subgoal_sr = [], [], [], []
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
            initial_dist      = np.linalg.norm(z_current - z_subgoal)

            for _ in range(config.manager.subgoal_horizon):
                action   = agent.get_worker_action(z_current, z_subgoal, proprio)
                next_img, env_reward, env_done, info = env.step(action)
                z_next    = agent.encoder.encode_numpy(next_img).squeeze()
                proprio_n = info['state']

                n_tasks = info.get('n_tasks_completed', 0)

                # Hindsight landmark injection
                agent.maybe_inject_hindsight(z_next, n_tasks)

                shaped = agent.compute_shaped_reward(
                    z_current, z_next, z_subgoal, env_reward,
                    proprio, proprio_n, initial_dist=initial_dist,
                )
                agent.low_buffer.add(
                    z_current, proprio, z_subgoal, action,
                    shaped, z_next, proprio_n, env_done,
                )

                cumulative_reward += env_reward
                agent.total_steps += 1
                pbar.update(1)
                z_current = z_next
                proprio   = proprio_n
                obs_img   = next_img

                if np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    subgoal_reached = True
                    break

                if env_done:
                    break

                if (agent.total_steps % 4 == 0
                        and agent.low_buffer.size > config.buffer.batch_size):
                    last_worker = agent.update_worker()

            # ---- High-level bookkeeping ----
            ep_high_steps += 1
            agent.reach_buffer.add(z_start, z_subgoal, subgoal_reached)
            agent.landmarks.record_visit(landmark_idx, success=subgoal_reached)

            if subgoal_reached:
                agent.high_buffer.add_success(
                    z_start, z_start,    # z_goal field unused by manager now, reuse z_start
                    z_subgoal, cumulative_reward, z_current, landmark_idx)
                ep_successes += 1
                ep_reward    += cumulative_reward
            else:
                delta = (np.linalg.norm(z_start - z_subgoal)
                         - np.linalg.norm(z_current - z_subgoal))
                if delta > 0.05:
                    agent.high_buffer.add_partial(
                        z_start, z_start,
                        z_current,
                        cumulative_reward * 0.5,
                        z_current, landmark_idx,
                    )
                else:
                    agent.high_buffer.add_failure(z_start, z_start, z_subgoal, landmark_idx)
                episode_done = True

            if env_done:
                episode_done = True

            if agent.high_buffer.size > config.buffer.batch_size:
                last_manager = agent.update_manager()

            agent._update_epsilon()

        # ---- End of episode ----
        agent.total_episodes += 1
        run_rewards.append(ep_reward)
        run_successes.append(ep_reward > 0)
        run_high_steps.append(ep_high_steps)
        run_subgoal_sr.append(ep_successes / max(ep_attempts, 1))

        if agent.total_episodes % config.reachability.update_freq == 0:
            for _ in range(5):
                last_reach = agent.update_reachability()

        if agent.total_episodes % config.landmarks.update_freq == 0:
            all_z = agent.low_buffer.get_all_z()
            if len(all_z) > config.landmarks.min_observations:
                agent.landmarks.update(all_z, z_goal=None)

        # TensorBoard
        if agent.total_episodes % 10 == 0:
            writer.add_scalar('train/episode_reward',       ep_reward,     agent.total_steps)
            writer.add_scalar('train/high_level_steps',     ep_high_steps, agent.total_steps)
            writer.add_scalar('train/subgoal_success_rate',
                              ep_successes / max(ep_attempts, 1), agent.total_steps)
            writer.add_scalar('train/epsilon',              agent.epsilon, agent.total_steps)
            writer.add_scalar('train/low_buffer_size',      len(agent.low_buffer), agent.total_steps)
            writer.add_scalar('train/high_buffer_size',     len(agent.high_buffer), agent.total_steps)
            writer.add_scalar('train/success_threshold',    agent.success_threshold, agent.total_steps)
            writer.add_scalar('train/hindsight_pool_size',
                              agent.landmarks._success_pool_size, agent.total_steps)
            for k, v in last_worker.items():
                writer.add_scalar(f'worker/{k}',        v, agent.total_steps)
            for k, v in last_manager.items():
                writer.add_scalar(f'manager/{k}',       v, agent.total_steps)
            for k, v in last_reach.items():
                writer.add_scalar(f'reachability/{k}',  v, agent.total_steps)

        # Rich terminal log every 50 episodes
        if agent.total_episodes % 50 == 0 and run_rewards:
            elapsed = time.time() - train_start
            sps     = agent.total_steps / max(elapsed, 1)
            eta     = str(datetime.timedelta(
                seconds=int((config.training.total_timesteps - agent.total_steps) / max(sps, 1))))
            pct     = 100 * agent.total_steps / config.training.total_timesteps

            print(f"\n{'─'*70}")
            print(f"  Step {agent.total_steps:>9,} / {config.training.total_timesteps:,}"
                  f"  ({pct:.1f}%)    Episode {agent.total_episodes:,}")
            print(f"  Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))}  "
                  f"ETA: {eta}  Speed: {sps:.1f} steps/s")
            print(f"{'─'*70}")
            n = len(run_rewards)
            print(f"  [Episode Stats — last {n} eps]")
            print(f"    Reward:           mean={np.mean(run_rewards):.4f}  "
                  f"std={np.std(run_rewards):.4f}  max={np.max(run_rewards):.4f}")
            print(f"    Task success:     {np.mean(run_successes)*100:.1f}%  (reward > 0)")
            print(f"    High-level steps: mean={np.mean(run_high_steps):.1f}")
            print(f"    Subgoal SR:       mean={np.mean(run_subgoal_sr)*100:.1f}%")
            print(f"  [Exploration]")
            print(f"    Epsilon:          {agent.epsilon:.4f}")
            print(f"    Landmarks:        {agent.landmarks.n_active}")
            print(f"    Hindsight pool:   {agent.landmarks._success_pool_size}")
            print(f"    Success thresh:   {agent.success_threshold:.4f}")
            print(f"  [Buffers]")
            print(f"    Low-level:        {len(agent.low_buffer):,}")
            print(f"    High-level (FER): {len(agent.high_buffer):,}")
            print(f"    Reachability:     {len(agent.reach_buffer):,}")
            if last_worker:
                print(f"  [Worker — SAC]")
                print(f"    Critic loss:      {last_worker.get('worker_critic_loss',0):.6f}")
                print(f"    Actor loss:       {last_worker.get('worker_actor_loss',0):.6f}")
                print(f"    Alpha:            {last_worker.get('worker_alpha',0):.6f}")
            if last_manager:
                print(f"  [Manager — DQN]")
                print(f"    Q-loss:           {last_manager.get('manager_loss',0):.6f}")
                print(f"    Mean Q:           {last_manager.get('manager_q_mean',0):.6f}")
            if last_reach:
                print(f"  [Reachability]")
                print(f"    BCE loss:         {last_reach.get('reach_loss',0):.6f}")
                print(f"    Accuracy:         {last_reach.get('reach_accuracy',0)*100:.1f}%")
            run_rewards.clear(); run_successes.clear()
            run_high_steps.clear(); run_subgoal_sr.clear()

        # Evaluation
        if agent.total_steps % config.training.eval_freq < config.manager.subgoal_horizon * 20:
            t0     = time.time()
            result = evaluate(agent, config)
            is_best = result['success_rate'] > best_eval
            if is_best:
                best_eval = result['success_rate']
                save_checkpoint(agent, config, 'best')
                record_episode_videos(agent, config, 'best',
                                      config.training.video_n_episodes)
            for k, v in result.items():
                if isinstance(v, float):
                    writer.add_scalar(f'eval/{k}', v, agent.total_steps)
            print(f"\n{'='*70}")
            print(f"  [EVAL] Step {agent.total_steps:,}  ({result['n_episodes']} eps, "
                  f"{time.time()-t0:.1f}s)")
            print(f"    Success rate:       {result['success_rate']*100:.1f}%"
                  + (" <-- NEW BEST" if is_best else f"  (best: {best_eval*100:.1f}%)"))
            print(f"    Mean reward:        {result['mean_reward']:.4f} "
                  f"± {result['std_reward']:.4f}")
            print(f"    Mean tasks done:    {result['mean_tasks_completed']:.2f}"
                  f" / {len(config.training.tasks_to_complete)}")
            print(f"    Mean high steps:    {result['mean_high_steps']:.1f}")
            print(f"{'='*70}")

        # Periodic checkpoint + video
        if ckpt_sched.should_save(agent.total_steps):
            label = ckpt_sched.label(agent.total_steps)
            save_checkpoint(agent, config, label)
            record_episode_videos(agent, config, label,
                                  config.training.video_n_episodes)

    pbar.close()
    writer.close()
    total = time.time() - train_start
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Time:     {str(datetime.timedelta(seconds=int(total)))}")
    print(f"  Steps:    {agent.total_steps:,}")
    print(f"  Episodes: {agent.total_episodes:,}")
    print(f"  Best SR:  {best_eval*100:.1f}%")
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
    parser.add_argument('--total_steps',       type=int,  default=1_000_000)
    parser.add_argument('--encoder',           type=str,  default='r3m',
                        choices=['r3m', 'dinov2'])
    parser.add_argument('--n_landmarks',       type=int,  default=100)
    parser.add_argument('--subgoal_horizon',   type=int,  default=20)
    parser.add_argument('--log_dir',           type=str,  default='logs/')
    parser.add_argument('--warmup_k',          type=int,  default=10)
    parser.add_argument('--tasks',             type=str,  nargs='+',
                        default=['microwave'],
                        help='Tasks to complete. Start with microwave alone.')
    parser.add_argument('--no_demo_landmarks', action='store_true',
                        help='Disable demo landmark seeding (ablation).')
    parser.add_argument('--no_hindsight',      action='store_true',
                        help='Disable hindsight landmark injection (ablation).')
    parser.add_argument('--no_curriculum',     action='store_true',
                        help='Disable curriculum landmark filtering (ablation).')
    parser.add_argument('--no_video',          action='store_true',
                        help='Disable episode video recording.')
    parser.add_argument('--demo_gif',          type=str,
                        default='demo/franka_kitchen/demo.gif',
                        help='Path to demo GIF for landmark seeding.')
    args = parser.parse_args()

    config = Config()
    config.training.seed              = args.seed
    config.training.device            = args.device
    config.training.total_timesteps   = args.total_steps
    config.training.log_dir           = args.log_dir
    config.training.tasks_to_complete = args.tasks
    config.training.warmup_future_k   = args.warmup_k
    config.training.record_video      = not args.no_video
    config.encoder.name               = args.encoder
    config.landmarks.n_landmarks      = args.n_landmarks
    config.landmarks.use_demo_landmarks    = not args.no_demo_landmarks
    config.landmarks.use_hindsight_landmarks = not args.no_hindsight
    config.landmarks.use_curriculum_landmarks = not args.no_curriculum
    config.landmarks.demo_gif_path    = args.demo_gif
    config.manager.subgoal_horizon    = args.subgoal_horizon

    if args.device == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available!"
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train(config)