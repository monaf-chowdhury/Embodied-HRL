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
from agent import VisualHRLAgent, per_task_progress
from utils import save_video


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


class CheckpointScheduler:
    def __init__(self, total_steps: int, n_checkpoints: int = 10):
        self.thresholds = [int((i + 1) * total_steps / n_checkpoints) for i in range(n_checkpoints)]
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


def record_episode_videos(agent, config, label: str, n_episodes: int = 3):
    if not config.training.record_video:
        return
    video_dir = os.path.join(config.training.log_dir, 'videos', label)
    os.makedirs(video_dir, exist_ok=True)
    env = FrankaKitchenImageWrapper(tasks_to_complete=config.training.tasks_to_complete, img_size=config.encoder.img_size)
    for ep_i in range(n_episodes):
        frames = []
        img = env.reset()
        z = agent.encoder.encode_numpy(img).squeeze()
        proprio = env.get_state()
        done = False
        frames.append(img)
        high_steps = 0
        while not done and high_steps < config.manager.max_high_level_steps:
            idx = agent.select_subgoal(z)
            z_sub = agent.landmarks.get(idx)
            task_id = agent.landmarks.get_task_id(idx)
            for _ in range(config.manager.subgoal_horizon):
                a = agent.get_worker_action(z, z_sub, proprio, task_id, deterministic=True)
                next_img, _, done, info = env.step(a)
                frames.append(next_img)
                proprio = info['state']
                z = agent.encoder.encode_numpy(next_img).squeeze()
                if done or np.linalg.norm(z - z_sub) < agent.success_threshold:
                    break
            high_steps += 1
        save_video(frames, os.path.join(video_dir, f'ep_{ep_i:03d}.mp4'), fps=15)
    env.close()


def evaluate(agent, config, n_episodes=None):
    n_episodes = n_episodes or config.training.n_eval_episodes
    env = FrankaKitchenImageWrapper(tasks_to_complete=config.training.tasks_to_complete, img_size=config.encoder.img_size)
    rewards, high_steps_list, tasks_completed_list = [], [], []
    any_task_flags, full_task_flags, env_reward_flags = [], [], []
    for _ in range(n_episodes):
        img = env.reset()
        z = agent.encoder.encode_numpy(img).squeeze()
        proprio = env.get_state()
        done = False
        ep_env_reward = 0.0
        max_tasks_completed = 0
        high_steps = 0
        while not done and high_steps < config.manager.max_high_level_steps:
            idx = agent.select_subgoal(z)
            z_sub = agent.landmarks.get(idx)
            task_id = agent.landmarks.get_task_id(idx)
            for _ in range(config.manager.subgoal_horizon):
                a = agent.get_worker_action(z, z_sub, proprio, task_id, deterministic=True)
                next_img, env_reward, done, info = env.step(a)
                proprio = info['state']
                z = agent.encoder.encode_numpy(next_img).squeeze()
                ep_env_reward += env_reward
                max_tasks_completed = max(max_tasks_completed, info.get('n_tasks_completed', 0))
                if done or np.linalg.norm(z - z_sub) < agent.success_threshold:
                    break
            high_steps += 1
        rewards.append(ep_env_reward)
        high_steps_list.append(high_steps)
        tasks_completed_list.append(max_tasks_completed)
        any_task_flags.append(max_tasks_completed >= 1)
        full_task_flags.append(max_tasks_completed >= len(config.training.tasks_to_complete))
        env_reward_flags.append(ep_env_reward > 0)
    env.close()
    return {
        'any_task_success_rate': float(np.mean(any_task_flags)),
        'full_task_success_rate': float(np.mean(full_task_flags)),
        'env_reward_success_rate': float(np.mean(env_reward_flags)),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_high_steps': float(np.mean(high_steps_list)),
        'mean_tasks_completed': float(np.mean(tasks_completed_list)),
        'n_episodes': n_episodes,
    }


def save_checkpoint(agent, config, name: str):
    path = os.path.join(config.training.log_dir, f'checkpoint_{name}.pt')
    torch.save({
        'manager_q': agent.manager_q.state_dict(),
        'manager_q_target': agent.manager_q_target.state_dict(),
        'worker_actor': agent.worker_actor.state_dict(),
        'worker_critic': agent.worker_critic.state_dict(),
        'worker_critic_target': agent.worker_critic_target.state_dict(),
        'total_steps': agent.total_steps,
        'total_episodes': agent.total_episodes,
        'success_threshold': agent.success_threshold,
        'epsilon': agent.epsilon,
        'proprio_mean': agent.low_buffer._p_mean,
        'proprio_m2': agent.low_buffer._p_M2,
        'proprio_n': agent.low_buffer._p_n,
    }, path)
    print(f"  [Checkpoint] Saved: {os.path.basename(path)}")


def train(config: Config):
    logger = Logger(config.training.log_dir)
    print(f"Training log: {logger.log_path}")
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)

    writer = SummaryWriter(config.training.log_dir)
    ckpt_sched = CheckpointScheduler(config.training.total_timesteps)
    env = FrankaKitchenImageWrapper(tasks_to_complete=config.training.tasks_to_complete,
                                    img_size=config.encoder.img_size,
                                    seed=config.training.seed,
                                    terminate_on_tasks_completed=False)
    agent = VisualHRLAgent(config)

    print("=" * 70)
    print("  Visual HRL Training — FrankaKitchen-v1")
    print("=" * 70)
    print(f"  Encoder:       {config.encoder.name} raw backbone features (dim={config.encoder.raw_dim})")
    print(f"  Tasks:         {config.training.tasks_to_complete}")
    print(f"  Landmarks:     {config.landmarks.n_landmarks}")
    print(f"  Subgoal K:     {config.manager.subgoal_horizon}")
    print(f"  Max HL steps:  {config.manager.max_high_level_steps}")
    print(f"  Total steps:   {config.training.total_timesteps:,}")
    print("=" * 70)

    print("\n[Phase 1] Random exploration warmup using raw R3M latents...")
    warmup_start = time.time()
    K_future = config.training.warmup_future_k
    for ep_i in range(config.training.n_warmup):
        img = env.reset()
        ep_imgs = [img]
        ep_props = [env.get_state()]
        ep_actions, ep_rewards, ep_dones, ep_n_tasks = [], [], [], []
        done = False
        while not done:
            a = env.action_space.sample()
            next_img, reward, done, info = env.step(a)
            ep_imgs.append(next_img)
            ep_props.append(info['state'])
            ep_actions.append(a)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_n_tasks.append(info.get('n_tasks_completed', 0))
        all_z = agent.encoder.encode_numpy(np.stack(ep_imgs))
        T = len(ep_actions)
        for t in range(T):
            z_t, z_next = all_z[t], all_z[t + 1]
            prop_t, prop_n = ep_props[t], ep_props[t + 1]
            task_prog_before = per_task_progress(prop_t, agent.tasks)
            task_prog_after = per_task_progress(prop_n, agent.tasks)
            task_deltas = task_prog_after - task_prog_before
            completed_delta = ep_n_tasks[t] - (ep_n_tasks[t - 1] if t > 0 else 0)
            future_idx = min(t + K_future, T)
            z_sub = all_z[future_idx]
            task_id = agent.choose_task_from_state(ep_props[future_idx])
            reward = agent.compute_worker_reward(
                z_t, z_next, z_sub, ep_rewards[t], task_deltas, task_id,
                completed_delta, initial_dist=np.linalg.norm(z_t - z_sub)
            )
            agent.low_buffer.add(z_t, prop_t, z_sub, task_id, ep_actions[t], reward, z_next, prop_n,
                                 ep_dones[t], task_deltas, completed_delta)
            agent._latent_dists.append(np.linalg.norm(z_next - z_t))
            if completed_delta > 0:
                success_task = int(np.argmax(task_deltas)) if task_deltas.max() > 0 else task_id
                agent.landmarks.add_success_state(z_next, success_task)
        if (ep_i + 1) % 20 == 0:
            print(f"  Collected {ep_i+1}/{config.training.n_warmup} episodes")
    print(f"  Warmup done in {(time.time()-warmup_start)/60:.1f} min")

    replay = agent.low_buffer.get_landmark_data()
    agent.landmarks.update(replay)
    if config.landmarks.use_demo_landmarks:
        agent.landmarks.seed_from_demo(agent.encoder, gif_path=config.landmarks.demo_gif_path,
                                       max_frames=config.landmarks.demo_max_frames)
        agent.landmarks.update(replay)
    agent.calibrate_success_threshold()

    print("\n[Phase 2] Hierarchical training without SSE...")
    best_any = 0.0
    last_worker, last_manager = {}, {}
    run_rewards, run_any_success, run_high_steps, run_subgoal_sr = [], [], [], []
    pbar = tqdm(total=config.training.total_timesteps, desc="Training", dynamic_ncols=True)

    while agent.total_steps < config.training.total_timesteps:
        img = env.reset()
        z_current = agent.encoder.encode_numpy(img).squeeze()
        proprio = env.get_state()
        ep_env_reward = 0.0
        ep_successes = 0
        ep_attempts = 0
        ep_high_steps = 0
        done = False
        agent._prev_n_tasks = 0
        completed_count = 0

        while not done and ep_high_steps < config.manager.max_high_level_steps:
            landmark_idx = agent.select_subgoal(z_current)
            z_subgoal = agent.landmarks.get(landmark_idx)
            task_id = agent.landmarks.get_task_id(landmark_idx)

            z_start = z_current.copy()
            start_proprio = proprio.copy()
            start_progress = per_task_progress(start_proprio, agent.tasks)
            start_dist = np.linalg.norm(z_current - z_subgoal)
            start_completed_count = completed_count
            cumulative_env_reward = 0.0
            subgoal_reached = False
            ep_attempts += 1

            for _ in range(config.manager.subgoal_horizon):
                action = agent.get_worker_action(z_current, z_subgoal, proprio, task_id)
                next_img, env_reward, done, info = env.step(action)
                z_next = agent.encoder.encode_numpy(next_img).squeeze()
                proprio_n = info['state']
                n_tasks = info.get('n_tasks_completed', 0)
                completed_count = n_tasks

                task_prog_before = per_task_progress(proprio, agent.tasks)
                task_prog_after = per_task_progress(proprio_n, agent.tasks)
                task_deltas = task_prog_after - task_prog_before
                completed_delta = max(n_tasks - agent._prev_n_tasks, 0)

                agent.maybe_inject_hindsight(z_next, n_tasks, task_id)
                shaped = agent.compute_worker_reward(z_current, z_next, z_subgoal, env_reward, task_deltas,
                                                     task_id, completed_delta, initial_dist=start_dist)
                agent.low_buffer.add(z_current, proprio, z_subgoal, task_id, action, shaped, z_next,
                                     proprio_n, done, task_deltas, completed_delta)
                agent._latent_dists.append(np.linalg.norm(z_next - z_current))

                cumulative_env_reward += env_reward
                ep_env_reward += env_reward
                agent.total_steps += 1
                pbar.update(1)
                z_current = z_next
                proprio = proprio_n

                if done or np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    subgoal_reached = np.linalg.norm(z_current - z_subgoal) < agent.success_threshold
                    break
                if agent.total_steps % 2 == 0 and len(agent.low_buffer) > config.buffer.batch_size:
                    last_worker = agent.update_worker()

            end_progress = per_task_progress(proprio, agent.tasks)
            end_dist = np.linalg.norm(z_current - z_subgoal)
            tasks_completed_delta = max(completed_count - start_completed_count, 0)
            manager_reward = agent.compute_manager_reward(start_progress, end_progress, task_id,
                                                          tasks_completed_delta, cumulative_env_reward,
                                                          subgoal_reached, start_dist, end_dist)
            agent.high_buffer.add(z_start, z_subgoal, manager_reward, z_current, done, task_id)
            agent.landmarks.record_visit(landmark_idx, success=subgoal_reached)
            ep_successes += int(subgoal_reached)
            ep_high_steps += 1

            if len(agent.high_buffer) > config.buffer.batch_size:
                last_manager = agent.update_manager()
            agent._update_epsilon()

        agent.total_episodes += 1
        run_rewards.append(ep_env_reward)
        run_any_success.append(completed_count >= 1)
        run_high_steps.append(ep_high_steps)
        run_subgoal_sr.append(ep_successes / max(ep_attempts, 1))

        if agent.total_episodes % config.landmarks.update_freq == 0 and len(agent.low_buffer) > config.landmarks.min_observations:
            agent.landmarks.update(agent.low_buffer.get_landmark_data())
            agent.calibrate_success_threshold()

        if agent.total_episodes % 10 == 0:
            writer.add_scalar('train/episode_reward', ep_env_reward, agent.total_steps)
            writer.add_scalar('train/high_level_steps', ep_high_steps, agent.total_steps)
            writer.add_scalar('train/subgoal_success_rate', ep_successes / max(ep_attempts, 1), agent.total_steps)
            writer.add_scalar('train/epsilon', agent.epsilon, agent.total_steps)
            writer.add_scalar('train/low_buffer_size', len(agent.low_buffer), agent.total_steps)
            writer.add_scalar('train/high_buffer_size', len(agent.high_buffer), agent.total_steps)
            writer.add_scalar('train/success_threshold', agent.success_threshold, agent.total_steps)
            for k, v in last_worker.items():
                writer.add_scalar(f'worker/{k}', v, agent.total_steps)
            for k, v in last_manager.items():
                writer.add_scalar(f'manager/{k}', v, agent.total_steps)

        if agent.total_episodes % 50 == 0 and run_rewards:
            print(f"\n{'─'*70}")
            print(f"  Step {agent.total_steps:,} / {config.training.total_timesteps:,}    Episode {agent.total_episodes:,}")
            print(f"  Reward mean: {np.mean(run_rewards):.4f} | Any-task success: {np.mean(run_any_success)*100:.1f}% | HL steps: {np.mean(run_high_steps):.1f} | Subgoal SR: {np.mean(run_subgoal_sr)*100:.1f}%")
            if last_worker:
                print(f"  Worker: critic={last_worker.get('worker_critic_loss',0):.4f} actor={last_worker.get('worker_actor_loss',0):.4f} alpha={last_worker.get('worker_alpha',0):.4f}")
            if last_manager:
                print(f"  Manager: qloss={last_manager.get('manager_loss',0):.4f} qmean={last_manager.get('manager_q_mean',0):.4f}")
            run_rewards.clear(); run_any_success.clear(); run_high_steps.clear(); run_subgoal_sr.clear()

        if agent.total_steps % config.training.eval_freq < config.manager.subgoal_horizon:
            result = evaluate(agent, config)
            is_best = result['any_task_success_rate'] > best_any
            if is_best:
                best_any = result['any_task_success_rate']
                save_checkpoint(agent, config, 'best')
                record_episode_videos(agent, config, 'best', config.training.video_n_episodes)
            for k, v in result.items():
                if isinstance(v, float):
                    writer.add_scalar(f'eval/{k}', v, agent.total_steps)
            print(f"\n[EVAL] step={agent.total_steps:,} any-task={result['any_task_success_rate']*100:.1f}% full-task={result['full_task_success_rate']*100:.1f}% env-reward={result['env_reward_success_rate']*100:.1f}% mean_tasks={result['mean_tasks_completed']:.2f}")

        if ckpt_sched.should_save(agent.total_steps):
            label = ckpt_sched.label(agent.total_steps)
            save_checkpoint(agent, config, label)
            record_episode_videos(agent, config, label, config.training.video_n_episodes)

    pbar.close()
    writer.close()
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    parser.add_argument('--encoder', type=str, default='r3m', choices=['r3m', 'dinov2'])
    parser.add_argument('--n_landmarks', type=int, default=None)
    parser.add_argument('--subgoal_horizon', type=int, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--warmup_k', type=int, default=None)
    parser.add_argument('--tasks', type=str, nargs='+', default=None)
    parser.add_argument('--no_demo_landmarks', action='store_true')
    parser.add_argument('--no_hindsight', action='store_true')
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--demo_gif', type=str, default=None)
    args = parser.parse_args()

    config = Config()
    config.training.seed = args.seed
    config.training.device = args.device
    config.training.total_timesteps = args.total_steps
    if args.log_dir is not None:
        config.training.log_dir = args.log_dir
    if args.tasks is not None:
        config.training.tasks_to_complete = args.tasks
    if args.warmup_k is not None:
        config.training.warmup_future_k = args.warmup_k
    config.encoder.name = args.encoder
    if args.n_landmarks is not None:
        config.landmarks.n_landmarks = args.n_landmarks
    if args.subgoal_horizon is not None:
        config.manager.subgoal_horizon = args.subgoal_horizon
    if args.demo_gif is not None:
        config.landmarks.demo_gif_path = args.demo_gif
    if args.no_video:
        config.training.record_video = False
    if args.no_demo_landmarks:
        config.landmarks.use_demo_landmarks = False
    if args.no_hindsight:
        config.landmarks.use_hindsight_landmarks = False

    train(config)
