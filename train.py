"""
Main Training Loop: Visual HRL for Franka Kitchen.

Implements the full pipeline:
1. Frozen R3M encoder + projection head
2. FPS landmark buffer
3. Manager (DQN) selects landmarks as subgoals
4. Worker (SAC) executes with L2 delta-progress shaping
5. Strict execution: success continues, failure terminates
6. Reachability predictor filters bad subgoals

Run: python train.py [--seed 42] [--device cuda]
"""
import os
import sys
import time
import datetime
import argparse
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from env_wrapper import FrankaKitchenImageWrapper, HierarchicalKitchenWrapper
from agent import VisualHRLAgent
from utils import get_goal_image_and_encoding
import d4rl  # Must import to register kitchen envs with gym


# =============================================================================
# Logger: writes to both terminal and a log file simultaneously
# =============================================================================

class Logger:
    """Tees all print() output to both stdout and a log file."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"train_log_{timestamp}.txt")
        self.terminal = sys.stdout
        self.file = open(log_path, "w", buffering=1)  # line-buffered
        self.log_path = log_path
        sys.stdout = self  # redirect all print() calls

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.terminal
        self.file.close()


# =============================================================================
# Checkpoint logic: exactly 10 periodic saves over total training
# =============================================================================

class CheckpointScheduler:
    """
    Saves exactly N evenly-spaced checkpoints over total_steps.
    Works by tracking which checkpoint indices have already been saved,
    so it is robust to variable episode lengths.
    """

    def __init__(self, total_steps: int, n_checkpoints: int = 10):
        self.total_steps = total_steps
        self.n_checkpoints = n_checkpoints
        # Compute the step thresholds at which to save
        # e.g. for 1M steps and 10 checkpoints: 100k, 200k, ..., 1M
        self.thresholds = [
            int((i + 1) * total_steps / n_checkpoints)
            for i in range(n_checkpoints)
        ]
        self.next_idx = 0  # Index into thresholds

    def should_save(self, current_steps: int) -> bool:
        """Returns True the first time current_steps crosses the next threshold."""
        if self.next_idx >= len(self.thresholds):
            return False
        if current_steps >= self.thresholds[self.next_idx]:
            self.next_idx += 1
            return True
        return False

    def checkpoint_name(self, current_steps: int) -> str:
        idx = self.next_idx  # already incremented, so this is the one just saved
        total = self.total_steps
        return f"periodic_{idx:02d}_of_{self.n_checkpoints}_step{current_steps}"


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(agent, config, z_goal, n_episodes=None):
    """
    Evaluate the agent deterministically (no exploration).
    Returns: dict with success_rate, mean_reward, mean_subtasks, mean_high_steps
    """
    if n_episodes is None:
        n_episodes = config.training.n_eval_episodes

    eval_env = FrankaKitchenImageWrapper(
        task="kitchen-complete-v0",
        img_size=config.encoder.img_size,
    )

    successes = 0
    episode_rewards = []
    episode_high_steps_list = []

    for ep_i in range(n_episodes):
        obs_img = eval_env.reset()
        z_current = agent.encoder.encode_numpy(obs_img).squeeze()
        ep_reward = 0.0
        done = False
        high_steps = 0

        while not done and high_steps < 15:
            if agent.landmarks.is_ready:
                with torch.no_grad():
                    z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(agent.device)
                    z_g = torch.from_numpy(z_goal).float().unsqueeze(0).to(agent.device)
                    lm = torch.from_numpy(agent.landmarks.get_all()).float().to(agent.device)
                    q_vals = agent.manager_q.evaluate_all_landmarks(z_c, z_g, lm)
                    landmark_idx = q_vals.argmax(dim=1).item()
                z_subgoal = agent.landmarks.get(landmark_idx)
            else:
                break

            subgoal_reached = False
            for k in range(config.manager.subgoal_horizon):
                action = agent.get_worker_action(z_current, z_subgoal, deterministic=True)
                next_img, reward, done, info = eval_env.step(action)
                z_next = agent.encoder.encode_numpy(next_img).squeeze()
                ep_reward += reward
                z_current = z_next

                if np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    subgoal_reached = True
                    break
                if done:
                    break

            high_steps += 1
            if not subgoal_reached:
                break  # Strict execution

        episode_rewards.append(ep_reward)
        episode_high_steps_list.append(high_steps)
        # Kitchen reward > 0 means at least one subtask completed
        if ep_reward > 0:
            successes += 1

    eval_env.close()

    return {
        'success_rate': successes / n_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_high_steps': float(np.mean(episode_high_steps_list)),
        'n_episodes': n_episodes,
    }


# =============================================================================
# Checkpoint save
# =============================================================================

def save_checkpoint(agent, config, name: str):
    """Save all model weights and training state."""
    path = os.path.join(config.training.log_dir, f'checkpoint_{name}.pt')
    torch.save({
        'encoder_projection': agent.encoder.projection.state_dict(),
        'manager_q': agent.manager_q.state_dict(),
        'manager_q_target': agent.manager_q_target.state_dict(),
        'worker_actor': agent.worker_actor.state_dict(),
        'worker_critic': agent.worker_critic.state_dict(),
        'worker_critic_target': agent.worker_critic_target.state_dict(),
        'reachability': agent.reachability.state_dict(),
        'total_steps': agent.total_steps,
        'total_episodes': agent.total_episodes,
        'success_threshold': agent.success_threshold,
        'epsilon': agent.epsilon,
    }, path)
    print(f"  [Checkpoint] Saved: {os.path.basename(path)}")


# =============================================================================
# Main training loop
# =============================================================================

def train(config: Config):
    """Main training function."""

    # ---- Logger (must come first — redirects stdout to file + terminal) ----
    logger = Logger(config.training.log_dir)
    print(f"Training log: {logger.log_path}")

    # ---- Setup ----
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)

    os.makedirs(config.training.log_dir, exist_ok=True)
    writer = SummaryWriter(config.training.log_dir)

    # ---- Checkpoint scheduler: exactly 10 saves ----
    ckpt_scheduler = CheckpointScheduler(
        total_steps=config.training.total_timesteps,
        n_checkpoints=10,
    )

    # ---- Environment ----
    env = FrankaKitchenImageWrapper(
        task="kitchen-complete-v0",
        img_size=config.encoder.img_size,
        seed=config.training.seed,
    )
    hier_env = HierarchicalKitchenWrapper(env, subgoal_horizon=config.manager.subgoal_horizon)

    # ---- Agent ----
    agent = VisualHRLAgent(config)

    # Goal placeholder — manager learns goal from reward signal
    # z_goal = np.zeros(config.encoder.proj_dim, dtype=np.float32)
    print("Rendering goal state image...")
    z_goal, goal_img = get_goal_image_and_encoding(
        agent.encoder, 
        img_size=config.encoder.img_size, 
        device=config.training.device
    )
    # Save goal image for visual inspection
    cv2.imwrite(os.path.join(config.training.log_dir, 'goal_image.png'), goal_img[:,:,::-1])
    print(f"Goal image saved to {config.training.log_dir}/goal_image.png")


    # ---- Print config header ----
    sep = "=" * 70
    print(sep)
    print("  Visual HRL Training — Franka Kitchen")
    print(sep)
    print(f"  Started:          {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device:           {config.training.device} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    if torch.cuda.is_available():
        print(f"  VRAM:             "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Seed:             {config.training.seed}")
    print(f"  Encoder:          {config.encoder.name} "
          f"(frozen={config.encoder.freeze}, proj_dim={config.encoder.proj_dim})")
    print(f"  Landmarks:        {config.landmarks.n_landmarks} "
          f"(update every {config.landmarks.update_freq} eps)")
    print(f"  Subgoal horizon:  {config.manager.subgoal_horizon} low-level steps")
    print(f"  Manager gamma:    {config.manager.gamma} "
          f"(lr={config.manager.lr}, tau={config.manager.tau})")
    print(f"  Worker gamma:     {config.worker.gamma} "
          f"(actor_lr={config.worker.actor_lr}, auto_alpha={config.worker.auto_alpha})")
    print(f"  Buffer capacity:  LL={config.buffer.capacity:,} "
          f"  HL=100,000  Reach=100,000")
    print(f"  Batch size:       {config.buffer.batch_size}")
    print(f"  Total timesteps:  {config.training.total_timesteps:,}")
    print(f"  Eval every:       {config.training.eval_freq:,} steps "
          f"({config.training.n_eval_episodes} episodes)")
    print(f"  Checkpoints:      10 periodic + best")
    print(f"  Log dir:          {config.training.log_dir}")
    print(sep)

    # =========================================================================
    # Phase 1: Random exploration
    # =========================================================================
    print("\n[Phase 1] Random exploration for initial data...")
    phase1_start = time.time()

    n_warmup_episodes = 50
    for ep_i in range(n_warmup_episodes):
        obs_img = hier_env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_img, reward, done, info = env.step(action)

            z_t = agent.encoder.encode_numpy(obs_img).squeeze()
            z_next = agent.encoder.encode_numpy(next_img).squeeze()

            z_random_sub = z_next
            shaped_r = agent.compute_shaped_reward(z_t, z_next, z_random_sub, reward)
            agent.low_buffer.add(z_t, z_random_sub, action, shaped_r, z_next, done)
            agent._latent_dists.append(np.linalg.norm(z_next - z_t))

            obs_img = next_img

        if (ep_i + 1) % 10 == 0:
            print(f"  Warmup episode {ep_i+1}/{n_warmup_episodes} | "
                  f"Buffer: {len(agent.low_buffer):,} transitions")

    phase1_time = time.time() - phase1_start
    print(f"  Phase 1 complete in {phase1_time/60:.1f} min")
    print(f"  Collected {len(agent.low_buffer):,} low-level transitions")

    # Compute initial landmarks
    all_z = agent.low_buffer.get_all_z()
    agent.landmarks.update(all_z)
    print(f"  Computed {agent.landmarks.n_active} landmarks via FPS")

    # Calibrate success threshold
    agent.calibrate_success_threshold()

    # Goal: mean of all observed latents (rough placeholder)
    z_goal = all_z.mean(axis=0)

    # =========================================================================
    # Phase 2: Hierarchical training
    # =========================================================================
    print(f"\n[Phase 2] Hierarchical training loop...")
    print(f"  Checkpoint schedule: every "
          f"{config.training.total_timesteps // 10:,} steps (10 total)\n")

    best_eval_success = 0.0
    training_start = time.time()
    last_log_time = time.time()

    # Running stats for terminal logging (reset every log interval)
    running_rewards = []
    running_successes = []
    running_high_steps = []
    running_subgoal_sr = []   # subgoal success rate per episode
    last_worker_metrics = {}
    last_manager_metrics = {}
    last_reach_metrics = {}

    pbar = tqdm(
        total=config.training.total_timesteps,
        desc="Training",
        dynamic_ncols=True,
    )

    while agent.total_steps < config.training.total_timesteps:

        obs_img = hier_env.reset()
        z_current = agent.encoder.encode_numpy(obs_img).squeeze()
        episode_reward = 0.0
        episode_high_steps = 0
        episode_successes = 0
        episode_attempts = 0
        episode_done = False

        while not episode_done:
            # ---- Manager selects subgoal ----
            landmark_idx = agent.select_subgoal(z_current, z_goal)
            z_subgoal = agent.landmarks.get(landmark_idx)

            # ---- Worker executes for K steps ----
            z_start = z_current.copy()
            cumulative_env_reward = 0.0
            subgoal_reached = False
            episode_attempts += 1

            for k_step in range(config.manager.subgoal_horizon):
                action = agent.get_worker_action(z_current, z_subgoal)
                next_img, env_reward, env_done, info = env.step(action)
                z_next = agent.encoder.encode_numpy(next_img).squeeze()

                shaped_r = agent.compute_shaped_reward(
                    z_current, z_next, z_subgoal, env_reward)
                agent.low_buffer.add(
                    z_current, z_subgoal, action, shaped_r, z_next, env_done)

                cumulative_env_reward += env_reward
                agent.total_steps += 1
                pbar.update(1)

                z_current = z_next
                obs_img = next_img

                if np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    subgoal_reached = True
                    break

                if env_done:
                    break

                if (agent.total_steps % 4 == 0
                        and agent.low_buffer.size > config.buffer.batch_size):
                    last_worker_metrics = agent.update_worker()

            # ---- High-level bookkeeping ----
            episode_high_steps += 1
            agent.reach_buffer.add(z_start, z_subgoal, subgoal_reached)
            agent.landmarks.record_visit(landmark_idx, success=subgoal_reached)

            if subgoal_reached:
                agent.high_buffer.add_success(
                    z_start, z_goal, z_subgoal, cumulative_env_reward,
                    z_current, landmark_idx)
                episode_successes += 1
                episode_reward += cumulative_env_reward
            else:
                agent.high_buffer.add_failure(
                    z_start, z_goal, z_subgoal, landmark_idx)
                episode_done = True

            if env_done:
                episode_done = True

            if agent.high_buffer.size > config.buffer.batch_size:
                last_manager_metrics = agent.update_manager()

            agent._update_epsilon()

        # ---- End of episode ----
        agent.total_episodes += 1

        # Track running stats
        running_rewards.append(episode_reward)
        running_successes.append(episode_reward > 0)
        running_high_steps.append(episode_high_steps)
        sr = episode_successes / max(episode_attempts, 1)
        running_subgoal_sr.append(sr)

        # Reachability update
        if agent.total_episodes % config.reachability.update_freq == 0:
            for _ in range(5):
                last_reach_metrics = agent.update_reachability()

        # Landmark update
        if agent.total_episodes % config.landmarks.update_freq == 0:
            all_z = agent.low_buffer.get_all_z()
            if len(all_z) > config.landmarks.min_observations:
                agent.landmarks.update(all_z)

        # ---- TensorBoard logging ----
        if agent.total_episodes % 10 == 0:
            writer.add_scalar('train/episode_reward', episode_reward, agent.total_steps)
            writer.add_scalar('train/high_level_steps', episode_high_steps, agent.total_steps)
            writer.add_scalar('train/subgoal_successes', episode_successes, agent.total_steps)
            writer.add_scalar('train/subgoal_success_rate', sr, agent.total_steps)
            writer.add_scalar('train/epsilon', agent.epsilon, agent.total_steps)
            writer.add_scalar('train/low_buffer_size', len(agent.low_buffer), agent.total_steps)
            writer.add_scalar('train/high_buffer_size', len(agent.high_buffer), agent.total_steps)
            if last_worker_metrics:
                for k, v in last_worker_metrics.items():
                    writer.add_scalar(f'worker/{k}', v, agent.total_steps)
            if last_manager_metrics:
                for k, v in last_manager_metrics.items():
                    writer.add_scalar(f'manager/{k}', v, agent.total_steps)
            if last_reach_metrics:
                for k, v in last_reach_metrics.items():
                    writer.add_scalar(f'reachability/{k}', v, agent.total_steps)

        # ---- Rich terminal + file logging every 50 episodes ----
        if agent.total_episodes % 50 == 0 and len(running_rewards) > 0:
            elapsed = time.time() - training_start
            steps_per_sec = agent.total_steps / max(elapsed, 1)
            remaining_steps = config.training.total_timesteps - agent.total_steps
            eta_sec = remaining_steps / max(steps_per_sec, 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

            pct_done = 100 * agent.total_steps / config.training.total_timesteps

            print(f"\n{'─'*70}")
            print(f"  Step {agent.total_steps:>9,} / {config.training.total_timesteps:,}"
                  f"  ({pct_done:.1f}%)    Episode {agent.total_episodes:,}")
            print(f"  Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))}    "
                  f"ETA: {eta_str}    "
                  f"Speed: {steps_per_sec:.1f} steps/s")
            print(f"{'─'*70}")

            # Episode stats (last 50 episodes)
            print(f"  [Episode Stats — last {len(running_rewards)} eps]")
            print(f"    Env reward:       mean={np.mean(running_rewards):.4f}  "
                  f"std={np.std(running_rewards):.4f}  "
                  f"max={np.max(running_rewards):.4f}")
            print(f"    Task success:     {np.mean(running_successes)*100:.1f}%  "
                  f"(ep reward > 0)")
            print(f"    High-level steps: mean={np.mean(running_high_steps):.1f}  "
                  f"max={np.max(running_high_steps)}")
            print(f"    Subgoal SR:       mean={np.mean(running_subgoal_sr)*100:.1f}%  "
                  f"(successes / attempts per ep)")

            # Exploration
            print(f"  [Exploration]")
            print(f"    Epsilon:          {agent.epsilon:.4f}")
            print(f"    Landmarks active: {agent.landmarks.n_active}")
            print(f"    Success thresh:   {agent.success_threshold:.4f} (L2 in latent)")

            # Buffers
            print(f"  [Buffers]")
            print(f"    Low-level:        {len(agent.low_buffer):,} / "
                  f"{config.buffer.capacity:,}")
            print(f"    High-level (FER): {len(agent.high_buffer):,} / 100,000")
            print(f"    Reachability:     {len(agent.reach_buffer):,} / 100,000")

            # Network losses
            if last_worker_metrics:
                print(f"  [Worker — SAC]")
                print(f"    Critic loss:      "
                      f"{last_worker_metrics.get('worker_critic_loss', 0):.6f}")
                print(f"    Actor loss:       "
                      f"{last_worker_metrics.get('worker_actor_loss', 0):.6f}")
                print(f"    Alpha (entropy):  "
                      f"{last_worker_metrics.get('worker_alpha', 0):.6f}")
            if last_manager_metrics:
                print(f"  [Manager — DQN]")
                print(f"    Q-loss:           "
                      f"{last_manager_metrics.get('manager_loss', 0):.6f}")
                print(f"    Mean Q-value:     "
                      f"{last_manager_metrics.get('manager_q_mean', 0):.6f}")
            if last_reach_metrics:
                print(f"  [Reachability Predictor]")
                print(f"    BCE loss:         "
                      f"{last_reach_metrics.get('reach_loss', 0):.6f}")
                print(f"    Accuracy:         "
                      f"{last_reach_metrics.get('reach_accuracy', 0)*100:.1f}%")

            # Reset running stats
            running_rewards.clear()
            running_successes.clear()
            running_high_steps.clear()
            running_subgoal_sr.clear()

        # ---- Evaluation ----
        if agent.total_steps % config.training.eval_freq < config.manager.subgoal_horizon * 20:
            eval_start = time.time()
            eval_results = evaluate(agent, config, z_goal)
            eval_time = time.time() - eval_start

            writer.add_scalar('eval/success_rate',
                              eval_results['success_rate'], agent.total_steps)
            writer.add_scalar('eval/mean_reward',
                              eval_results['mean_reward'], agent.total_steps)
            writer.add_scalar('eval/mean_high_steps',
                              eval_results['mean_high_steps'], agent.total_steps)

            is_best = eval_results['success_rate'] > best_eval_success
            if is_best:
                best_eval_success = eval_results['success_rate']
                save_checkpoint(agent, config, 'best')

            print(f"\n{'='*70}")
            print(f"  [EVALUATION] Step {agent.total_steps:,}  "
                  f"({eval_results['n_episodes']} episodes, {eval_time:.1f}s)")
            print(f"    Success rate:     {eval_results['success_rate']*100:.1f}%"
                  f"  {'<-- NEW BEST' if is_best else f'(best: {best_eval_success*100:.1f}%)'}")
            print(f"    Mean reward:      {eval_results['mean_reward']:.4f} "
                  f"± {eval_results['std_reward']:.4f}")
            print(f"    Mean high steps:  {eval_results['mean_high_steps']:.1f}")
            print(f"{'='*70}")

        # ---- Periodic checkpoint (exactly 10 over full training) ----
        if ckpt_scheduler.should_save(agent.total_steps):
            ckpt_name = ckpt_scheduler.checkpoint_name(agent.total_steps)
            save_checkpoint(agent, config, ckpt_name)

    # ---- Training complete ----
    pbar.close()
    writer.close()

    total_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Total time:       {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"  Total steps:      {agent.total_steps:,}")
    print(f"  Total episodes:   {agent.total_episodes:,}")
    print(f"  Best eval SR:     {best_eval_success*100:.1f}%")
    print(f"  Log saved to:     {logger.log_path}")
    print(f"{'='*70}")

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--encoder', type=str, default='r3m', choices=['r3m', 'dinov2'])
    parser.add_argument('--n_landmarks', type=int, default=100)
    parser.add_argument('--subgoal_horizon', type=int, default=20)
    parser.add_argument('--log_dir', type=str, default='logs/')
    args = parser.parse_args()

    config = Config()
    config.training.seed = args.seed
    config.training.device = args.device
    config.training.total_timesteps = args.total_steps
    config.training.log_dir = args.log_dir
    config.encoder.name = args.encoder
    config.landmarks.n_landmarks = args.n_landmarks
    config.manager.subgoal_horizon = args.subgoal_horizon

    if args.device == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available!"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train(config)