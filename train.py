import argparse
import datetime
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import TaskGroundedHRLAgent
from config import Config
from env_wrapper import FrankaKitchenImageWrapper
from utils import completed_mask_from_info, per_task_errors, task_success_tolerance, save_video, set_seed

class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(log_dir, f'train_log_{ts}.txt')
        self.terminal = sys.stdout
        self.file = open(path, 'w', buffering=1)
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
        return f'periodic_{self.next_idx:02d}_step{steps}'


SEP = '═' * 72
SEP2 = '─' * 72
SEP3 = '·' * 72


def _fmt_steps(s: int) -> str:
    if s >= 1_000_000:
        return f'{s/1e6:.2f}M'
    if s >= 1_000:
        return f'{s/1e3:.1f}k'
    return str(s)


def _fmt_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def run_option(
    env: FrankaKitchenImageWrapper,
    agent: TaskGroundedHRLAgent,
    z_current: np.ndarray,
    proprio: np.ndarray,
    completion_mask: np.ndarray,
    prev_task: int,
    task_id: int,
    deterministic: bool = False,
    store: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Dict[str, float], list]:
    cfg = agent.config
    task_name = agent.tasks[task_id]
    start_state = agent.build_manager_state(proprio, completion_mask, prev_task, z_current=z_current)
    start_z = z_current.copy()
    start_proprio = proprio.copy()
    start_completion = completion_mask.copy()

    option_reward_env = 0.0
    option_len = 0
    option_success = False
    done = False
    hold_count = 0
    best_err = per_task_errors(proprio, [task_name])[0]
    patience = 0
    selected_error_traj = []
    frames = []

    if deterministic:
        frames.append(env._render_image())

    while option_len < cfg.manager.option_horizon:
        action = agent.select_worker_action(
            z_current=z_current,
            proprio=proprio,
            completion_mask=completion_mask,
            task_id=task_id,
            deterministic=deterministic,
        )
        if action is None:
            action = env.action_space.sample().astype(np.float32)

        next_img, env_reward, done, info = env.step(action)
        next_z = agent.encoder.encode_numpy(next_img).squeeze()
        next_proprio = info['state']
        next_completion = completed_mask_from_info(info, agent.tasks)

        prev_inputs = agent.get_worker_structured_inputs(proprio, task_id)
        next_inputs = agent.get_worker_structured_inputs(next_proprio, task_id)
        worker_reward, worker_stats = agent.compute_worker_reward(
            prev_proprio=proprio,
            next_proprio=next_proprio,
            task_id=task_id,
            prev_completion=completion_mask,
            next_completion=next_completion,
            env_reward=env_reward,
            action=action,
        )
        if store:
            agent.worker_buffer.add(
                z=z_current,
                proprio=proprio,
                task_id=task_id,
                target=prev_inputs['target'],
                value=prev_inputs['value'],
                error_vec=prev_inputs['error_vec'],
                progress=prev_inputs['progress'],
                completion=completion_mask,
                action=action,
                reward=worker_reward,
                next_z=next_z,
                next_proprio=next_proprio,
                next_value=next_inputs['value'],
                next_error_vec=next_inputs['error_vec'],
                next_progress=next_inputs['progress'],
                next_completion=next_completion,
                done=done,
            )

        option_reward_env += env_reward
        option_len += 1
        if store:
            agent.total_steps += 1
            agent.update_manager_epsilon()

        task_err = per_task_errors(next_proprio, [task_name])[0]
        selected_error_traj.append(float(task_err))
        if task_err <= task_success_tolerance(task_name):
            hold_count += 1
        else:
            hold_count = 0
        if task_err < best_err - cfg.worker.improvement_epsilon:
            best_err = float(task_err)
            patience = 0
        else:
            patience += 1

        success_now = agent.option_success(task_id, completion_mask, next_completion, task_err, hold_count)
        z_current = next_z
        proprio = next_proprio
        completion_mask = next_completion
        if deterministic:
            frames.append(next_img)

        if success_now:
            option_success = True
            break
        if done:
            break
        if option_len >= cfg.worker.option_min_steps and patience >= cfg.worker.option_patience:
            break
        if store and agent.total_steps >= cfg.training.total_timesteps:
            break

    manager_reward, manager_stats = agent.compute_manager_reward(
        start_proprio=start_proprio,
        end_proprio=proprio,
        task_id=task_id,
        start_completion=start_completion,
        end_completion=completion_mask,
        option_steps=option_len,
    )

    end_state = agent.build_manager_state(proprio, completion_mask, task_id, z_current=z_current)
    if store:
        agent.manager_buffer.add(
            z=start_z,
            proprio=start_proprio,
            progress=start_state['progress'],
            errors=start_state['errors'],
            completion=start_state['completion'],
            remaining=start_state['remaining'],
            prototype_sims=start_state['prototype_sims'],
            prev_task=start_state['prev_task'],
            task_id=task_id,
            reward=manager_reward,
            next_z=z_current,
            next_proprio=proprio,
            next_progress=end_state['progress'],
            next_errors=end_state['errors'],
            next_completion=end_state['completion'],
            next_remaining=end_state['remaining'],
            next_prototype_sims=end_state['prototype_sims'],
            done=done,
        )
        agent.total_options += 1

    stats = {
        'option_env_reward': float(option_reward_env),
        'option_len': float(option_len),
        'option_success': float(option_success),
        'manager_reward': float(manager_reward),
        'selected_task_end_error': float(manager_stats['end_err']),
        'selected_task_delta_err': float(manager_stats['delta_err']),
        'selected_task_delta_prog': float(manager_stats['delta_prog']),
        'completion_gain': float(manager_stats['completion_gain']),
        'worker_last_delta_err': float(worker_stats['delta_err']) if selected_error_traj else 0.0,
    }
    return z_current, proprio, completion_mask, done, stats, frames


def evaluate(agent: TaskGroundedHRLAgent, config: Config, n_episodes: int = None) -> Dict[str, float]:
    n_episodes = n_episodes or config.training.n_eval_episodes
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=config.training.terminate_on_tasks_completed,
    )
    any_flags, full_flags = [], []
    rewards, tasks_completed, high_steps = [], [], []

    for _ in range(n_episodes):
        img = env.reset()
        z_current = agent.encoder.encode_numpy(img).squeeze()
        proprio = env.get_state()
        completion = np.zeros(agent.n_tasks, dtype=np.float32)
        prev_task = -1
        done = False
        ep_reward = 0.0
        options = 0

        while not done and options < config.manager.max_high_level_steps and completion.sum() < agent.n_tasks:
            task_id = agent.select_task(z_current, proprio, completion, prev_task, deterministic=True)
            z_current, proprio, completion, done, stats, _ = run_option(
                env, agent, z_current, proprio, completion, prev_task, task_id, deterministic=True, store=False
            )
            ep_reward += stats['option_env_reward']
            prev_task = task_id
            options += 1

        n_done = int(round(completion.sum()))
        rewards.append(ep_reward)
        tasks_completed.append(n_done)
        high_steps.append(options)
        any_flags.append(n_done >= 1)
        full_flags.append(n_done >= agent.n_tasks)

    env.close()
    return {
        'any_task_success_rate': float(np.mean(any_flags)),
        'full_task_success_rate': float(np.mean(full_flags)),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_tasks_completed': float(np.mean(tasks_completed)),
        'mean_high_steps': float(np.mean(high_steps)),
        'n_episodes': int(n_episodes),
    }


def record_episode_videos(agent: TaskGroundedHRLAgent, config: Config, label: str, n_episodes: int = 3):
    if not config.training.record_video:
        return
    video_dir = os.path.join(config.training.log_dir, 'videos', label)
    os.makedirs(video_dir, exist_ok=True)
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=config.training.terminate_on_tasks_completed,
    )
    for ep_i in range(n_episodes):
        frames = []
        img = env.reset()
        frames.append(img)
        z_current = agent.encoder.encode_numpy(img).squeeze()
        proprio = env.get_state()
        completion = np.zeros(agent.n_tasks, dtype=np.float32)
        prev_task = -1
        done = False
        high_steps = 0
        while not done and high_steps < config.manager.max_high_level_steps and completion.sum() < agent.n_tasks:
            task_id = agent.select_task(z_current, proprio, completion, prev_task, deterministic=True)
            z_current, proprio, completion, done, _, option_frames = run_option(
                env, agent, z_current, proprio, completion, prev_task, task_id, deterministic=True, store=False
            )
            if option_frames:
                frames.extend(option_frames[1:] if len(frames) > 0 else option_frames)
            prev_task = task_id
            high_steps += 1
        save_video(frames, os.path.join(video_dir, f'ep_{ep_i:03d}.mp4'), fps=15)
    env.close()
    print(f'  [Video] {n_episodes} episodes recorded -> {video_dir}/')


def save_checkpoint(agent: TaskGroundedHRLAgent, config: Config, name: str):
    path = os.path.join(config.training.log_dir, f'checkpoint_{name}.pt')
    torch.save({
        'manager_q': agent.manager_q.state_dict(),
        'manager_q_target': agent.manager_q_target.state_dict(),
        'worker_actor': agent.worker_actor.state_dict(),
        'worker_critic': agent.worker_critic.state_dict(),
        'worker_critic_target': agent.worker_critic_target.state_dict(),
        'manager_optimizer': agent.manager_optimizer.state_dict(),
        'worker_actor_optimizer': agent.worker_actor_optimizer.state_dict(),
        'worker_critic_optimizer': agent.worker_critic_optimizer.state_dict(),
        'alpha_optimizer': agent.alpha_optimizer.state_dict() if agent.alpha_optimizer is not None else None,
        'log_alpha': agent.log_alpha.detach().cpu().item(),
        'task_lang_np': agent.task_lang_np,
        'task_goals_np': agent.task_goals_np,
        'manager_epsilon': agent.manager_epsilon,
        'total_steps': agent.total_steps,
        'total_episodes': agent.total_episodes,
        'total_options': agent.total_options,
        'manager_proprio_mean': agent.manager_buffer.proprio_norm.mean,
        'manager_proprio_M2': agent.manager_buffer.proprio_norm.M2,
        'manager_proprio_n': agent.manager_buffer.proprio_norm.n,
        'worker_proprio_mean': agent.worker_buffer.proprio_norm.mean,
        'worker_proprio_M2': agent.worker_buffer.proprio_norm.M2,
        'worker_proprio_n': agent.worker_buffer.proprio_norm.n,
        'config': config,
    }, path)
    print(f'  [Checkpoint] Saved -> {os.path.basename(path)}')


def train(config: Config):
    logger = Logger(config.training.log_dir)
    set_seed(config.training.seed)

    writer = SummaryWriter(config.training.log_dir)
    ckpt_sched = CheckpointScheduler(config.training.total_timesteps, n_checkpoints=config.training.checkpoint_every_n)

    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        seed=config.training.seed,
        terminate_on_tasks_completed=config.training.terminate_on_tasks_completed,
    )
    agent = TaskGroundedHRLAgent(config)

    print(f'\n{SEP}')
    print('  Task-Grounded Hierarchical RL — FrankaKitchen-v1')
    print(f"  Started : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(SEP)
    print(f'  Encoder          : {config.encoder.name.upper()} ({config.encoder.raw_dim}-d frozen visual context)')
    print(f'  Tasks            : {config.training.tasks_to_complete}')
    print(f'  Manager horizon  : {config.manager.option_horizon} steps / option')
    print(f'  Max HL steps     : {config.manager.max_high_level_steps} per episode')
    print(f'  Demo prototypes  : {config.semantic.use_demo_prototypes} ({config.semantic.demo_gif_path})')
    print(f'  Task language    : {config.semantic.use_task_language_embeddings} ({config.semantic.task_language_dim}-d hashed embeddings)')
    print(SEP2)
    print(f'  Manager reward   : completion={config.manager.reward_completion_bonus}  err={config.manager.reward_selected_error_reduction}  prog={config.manager.reward_selected_progress_gain}  regress={config.manager.reward_regression_penalty}')
    print(f'  Worker reward    : err={config.worker.reward_error_reduction}  prog={config.worker.reward_progress_gain}  completion={config.worker.reward_completion_bonus}  regress={config.worker.reward_regression_penalty}  env={config.worker.reward_env_weight}')
    print(SEP2)
    print(f'  Buffer start     : {config.buffer.start_learning_after:,} steps before learning')
    print(f'  Worker bootstrap : random actions for first {config.worker.bootstrap_random_action_steps:,} steps')
    print(f'  Manager bootstrap: uniform/heuristic for first {config.manager.bootstrap_uniform_steps:,} steps')
    print(f'  Total steps      : {config.training.total_timesteps:,}')
    print(f'  Batch size       : {config.buffer.batch_size}')
    print(f'  Buffer caps      : worker={config.buffer.worker_capacity:,}  manager={config.buffer.manager_capacity:,}')
    print(f'  Device           : {config.training.device}')
    if torch.cuda.is_available() and config.training.device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU              : {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB VRAM)')
    print(f'  Log dir          : {config.training.log_dir}')
    print(f'  Train log        : {logger.log_path}')
    print(f'{SEP}\n')

    best_full = 0.0
    last_worker = {}
    last_manager = {}
    train_start = time.time()
    pbar = tqdm(total=config.training.total_timesteps, desc='Training', dynamic_ncols=True, unit='step')

    run_rewards, run_tasks, run_options, run_option_sr, run_option_len = [], [], [], [], []
    run_selected_err = []

    while agent.total_steps < config.training.total_timesteps:
        img = env.reset()
        z_current = agent.encoder.encode_numpy(img).squeeze()
        proprio = env.get_state()
        completion = np.zeros(agent.n_tasks, dtype=np.float32)
        prev_task = -1
        done = False
        episode_reward = 0.0
        episode_options = 0
        episode_option_success = 0
        episode_option_lengths = []
        episode_new_completions = 0
        episode_selected_errors = []

        while not done and episode_options < config.manager.max_high_level_steps and completion.sum() < agent.n_tasks:
            task_id = agent.select_task(z_current, proprio, completion, prev_task, deterministic=False)
            z_current, proprio, completion, done, stats, _ = run_option(
                env, agent, z_current, proprio, completion, prev_task, task_id, deterministic=False, store=True
            )
            pbar.n = agent.total_steps
            pbar.refresh()
            episode_reward += stats['option_env_reward']
            episode_options += 1
            episode_option_success += int(stats['option_success'])
            episode_option_lengths.append(stats['option_len'])
            episode_new_completions += int(stats['completion_gain'])
            episode_selected_errors.append(stats['selected_task_end_error'])
            prev_task = task_id

            if agent.total_steps >= config.buffer.start_learning_after:
                for _ in range(config.buffer.worker_updates_per_step):
                    last_worker = agent.update_worker()
                for _ in range(config.buffer.manager_updates_per_option):
                    last_manager = agent.update_manager()

            if agent.total_steps >= config.training.total_timesteps:
                break

        agent.total_episodes += 1
        tasks_done = int(round(completion.sum()))
        run_rewards.append(episode_reward)
        run_tasks.append(tasks_done)
        run_options.append(episode_options)
        run_option_sr.append(episode_option_success / max(episode_options, 1))
        run_option_len.append(np.mean(episode_option_lengths) if episode_option_lengths else 0.0)
        run_selected_err.append(np.mean(episode_selected_errors) if episode_selected_errors else 0.0)

        if agent.total_episodes % 10 == 0:
            writer.add_scalar('train/episode_reward', episode_reward, agent.total_steps)
            writer.add_scalar('train/tasks_completed', float(tasks_done), agent.total_steps)
            writer.add_scalar('train/any_task_success', float(tasks_done >= 1), agent.total_steps)
            writer.add_scalar('train/full_task_success', float(tasks_done >= agent.n_tasks), agent.total_steps)
            writer.add_scalar('train/new_completions_per_episode', float(episode_new_completions), agent.total_steps)
            writer.add_scalar('train/high_level_steps', float(episode_options), agent.total_steps)
            writer.add_scalar('train/option_success_rate', episode_option_success / max(episode_options, 1), agent.total_steps)
            writer.add_scalar('train/option_length_mean', np.mean(episode_option_lengths) if episode_option_lengths else 0.0, agent.total_steps)
            writer.add_scalar('train/mean_selected_task_error', np.mean(episode_selected_errors) if episode_selected_errors else 0.0, agent.total_steps)
            writer.add_scalar('train/manager_epsilon', agent.manager_epsilon, agent.total_steps)
            writer.add_scalar('train/worker_random_prob', agent.current_worker_random_prob(), agent.total_steps)
            writer.add_scalar('train/worker_buffer_size', len(agent.worker_buffer), agent.total_steps)
            writer.add_scalar('train/manager_buffer_size', len(agent.manager_buffer), agent.total_steps)
            for k, v in last_worker.items():
                writer.add_scalar(f'worker/{k}', v, agent.total_steps)
            for k, v in last_manager.items():
                writer.add_scalar(f'manager/{k}', v, agent.total_steps)

        if agent.total_episodes % 50 == 0 and run_rewards:
            elapsed = time.time() - train_start
            sps = agent.total_steps / max(elapsed, 1.0)
            eta = (config.training.total_timesteps - agent.total_steps) / max(sps, 1.0)
            pct = 100.0 * agent.total_steps / config.training.total_timesteps
            n_ep = len(run_rewards)
            print(f'\n{SEP2}')
            print(f'  Step {agent.total_steps:>10,} / {config.training.total_timesteps:,} ({pct:5.1f}%)   Episode {agent.total_episodes:,}')
            print(f'  Elapsed : {_fmt_time(elapsed)}    ETA : {_fmt_time(eta)}    Speed : {sps:.0f} steps/s')
            print(SEP3)
            print(f'  [Last {n_ep} episodes]')
            print(f'    Episode reward       : mean = {np.mean(run_rewards):.4f}   max = {np.max(run_rewards):.4f}')
            print(f'    Tasks completed      : mean = {np.mean(run_tasks):.2f} / {agent.n_tasks}   full-task = {np.mean(np.asarray(run_tasks) >= agent.n_tasks) * 100:5.1f}%')
            print(f'    Options / episode    : mean = {np.mean(run_options):.2f}')
            print(f'    Option success rate  : mean = {np.mean(run_option_sr) * 100:5.1f}%')
            print(f'    Option length        : mean = {np.mean(run_option_len):.2f}')
            print(f'    Selected task error  : mean = {np.mean(run_selected_err):.4f}')
            print(SEP3)
            print(f'  Exploration')
            print(f'    Manager epsilon      : {agent.manager_epsilon:.4f}')
            print(f'    Worker random prob   : {agent.current_worker_random_prob():.4f}')
            print(f'  Buffers')
            print(f'    Worker replay        : {len(agent.worker_buffer):>8,}')
            print(f'    Manager replay       : {len(agent.manager_buffer):>8,}')
            if last_worker:
                print(SEP3)
                print('  Worker (SAC)')
                print(f"    Critic loss          : {last_worker.get('worker_critic_loss', 0):.5f}")
                print(f"    Actor loss           : {last_worker.get('worker_actor_loss', 0):.5f}")
                print(f"    Alpha                : {last_worker.get('worker_alpha', 0):.4f}")
            if last_manager:
                print(SEP3)
                print('  Manager (Task Q)')
                print(f"    Q loss               : {last_manager.get('manager_loss', 0):.5f}")
                print(f"    Q mean               : {last_manager.get('manager_q_mean', 0):.4f}")
            run_rewards.clear()
            run_tasks.clear()
            run_options.clear()
            run_option_sr.clear()
            run_option_len.clear()
            run_selected_err.clear()

        if agent.total_steps > 0 and agent.total_steps % config.training.eval_freq < config.manager.option_horizon:
            t0 = time.time()
            result = evaluate(agent, config)
            eval_t = time.time() - t0
            is_best = result['full_task_success_rate'] > best_full
            if is_best:
                best_full = result['full_task_success_rate']
                save_checkpoint(agent, config, 'best')
                record_episode_videos(agent, config, 'best', config.training.video_n_episodes)
            for k, v in result.items():
                if isinstance(v, (float, int)):
                    writer.add_scalar(f'eval/{k}', float(v), agent.total_steps)
            badge = '  ◀ NEW BEST' if is_best else f'  (best so far: {best_full * 100:.1f}%)'
            print(f'\n{SEP}')
            print(f"  EVALUATION — Step {_fmt_steps(agent.total_steps)} ({result['n_episodes']} episodes, {eval_t:.1f}s)")
            print(SEP2)
            print(f"  Any-task success   : {result['any_task_success_rate'] * 100:5.1f}%")
            print(f"  Full-task success  : {result['full_task_success_rate'] * 100:5.1f}%{badge}")
            print(f"  Mean tasks done    : {result['mean_tasks_completed']:.2f} / {agent.n_tasks}")
            print(f"  Mean env reward    : {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
            print(f"  Mean HL steps      : {result['mean_high_steps']:.2f}")
            print(f'{SEP}\n')

        if ckpt_sched.should_save(agent.total_steps):
            label = ckpt_sched.label(agent.total_steps)
            save_checkpoint(agent, config, label)
            record_episode_videos(agent, config, label, config.training.video_n_episodes)

    pbar.close()
    writer.close()
    env.close()
    total_time = time.time() - train_start
    print(f'\n{SEP}')
    print('  TRAINING COMPLETE')
    print(SEP2)
    print(f'  Total time        : {_fmt_time(total_time)}')
    print(f'  Total steps       : {agent.total_steps:,}')
    print(f'  Total episodes    : {agent.total_episodes:,}')
    print(f'  Total options     : {agent.total_options:,}')
    print(f'  Best full-task SR : {best_full * 100:.1f}%')
    print(f'  Final epsilon     : {agent.manager_epsilon:.4f}')
    print(f'  Train log         : {logger.log_path}')
    print(f'{SEP}\n')
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task-grounded hierarchical RL for Franka Kitchen v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--total_steps', type=int, default=1_200_000)
    parser.add_argument('--encoder', type=str, default='r3m', choices=['r3m', 'dinov2'])
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--tasks', type=str, nargs='+', default=None)
    parser.add_argument('--option_horizon', type=int, default=None)
    parser.add_argument('--max_high_level_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--eval_freq', type=int, default=None)
    parser.add_argument('--demo_gif', type=str, default=None)
    parser.add_argument('--no_demo_prototypes', action='store_true')
    parser.add_argument('--no_video', action='store_true')
    args = parser.parse_args()

    config = Config()
    config.training.seed = args.seed
    config.training.device = args.device
    config.training.total_timesteps = args.total_steps
    config.encoder.name = args.encoder
    if args.log_dir is not None:
        config.training.log_dir = args.log_dir
    if args.tasks is not None:
        config.training.tasks_to_complete = args.tasks
    if args.option_horizon is not None:
        config.manager.option_horizon = args.option_horizon
    if args.max_high_level_steps is not None:
        config.manager.max_high_level_steps = args.max_high_level_steps
    if args.batch_size is not None:
        config.buffer.batch_size = args.batch_size
    if args.eval_freq is not None:
        config.training.eval_freq = args.eval_freq
    if args.demo_gif is not None:
        config.semantic.demo_gif_path = args.demo_gif
    if args.no_demo_prototypes:
        config.semantic.use_demo_prototypes = False
    if args.no_video:
        config.training.record_video = False

    if config.training.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA not available — use --device cpu'
    train(config)
