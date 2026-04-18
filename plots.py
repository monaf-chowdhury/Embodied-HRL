"""
Diagnostic plots for the task-grounded Franka Kitchen hierarchy.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print('ERROR: tensorboard not installed. Run: pip install tensorboard')
    raise


COLORS = {
    'green': '#2ecc71',
    'blue': '#3498db',
    'orange': '#e67e22',
    'red': '#e74c3c',
    'purple': '#9b59b6',
    'teal': '#1abc9c',
    'gray': '#95a5a6',
    'gold': '#f39c12',
    'pink': '#e91e8c',
    'dark': '#2c3e50',
}

STYLE = {
    'raw_alpha': 0.20,
    'raw_lw': 0.7,
    'smooth_alpha': 0.92,
    'smooth_lw': 2.0,
    'grid_alpha': 0.25,
    'title_fs': 11,
    'label_fs': 9,
    'tick_fs': 8,
    'suptitle_fs': 13,
}


def load_tb_scalars(log_dir: str) -> dict:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = (
            np.array([e.step for e in events]),
            np.array([e.value for e in events]),
        )
        print(f'  Loaded {tag} ({len(events)} pts)')
    return data


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[: len(values)]


def _style_ax(ax):
    ax.grid(True, alpha=STYLE['grid_alpha'], linestyle='--', linewidth=0.6)
    ax.tick_params(labelsize=STYLE['tick_fs'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f'{x/1e6:.1f}M' if x >= 1_000_000 else (f'{x/1e3:.0f}k' if x >= 1_000 else str(int(x)))
        )
    )


def _plot(ax, data, tag, title=None, ylabel=None, color='steelblue', smooth_window=10, pct=False, ymin=None, ymax=None):
    if tag not in data:
        ax.text(0.5, 0.5, f'No data\n({tag})', ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=STYLE['label_fs'])
        ax.set_title(title or tag, fontsize=STYLE['title_fs'], fontweight='bold')
        _style_ax(ax)
        return
    steps, values = data[tag]
    smoothed = smooth(values, smooth_window)
    if pct:
        values = values * 100.0
        smoothed = smoothed * 100.0
    ax.plot(steps, values, alpha=STYLE['raw_alpha'], color=color, linewidth=STYLE['raw_lw'])
    ax.plot(steps, smoothed, alpha=STYLE['smooth_alpha'], color=color, linewidth=STYLE['smooth_lw'])
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_title(title or tag, fontsize=STYLE['title_fs'], fontweight='bold')
    ax.set_xlabel('Environment Steps', fontsize=STYLE['label_fs'])
    ax.set_ylabel(ylabel or ('%' if pct else ''), fontsize=STYLE['label_fs'])
    _style_ax(ax)


def _save(fig, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved -> {path}')


def plot_overview_dashboard(data: dict, out_dir: str, sw: int = 15):
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor('white')
    fig.suptitle('Task-Grounded Franka Kitchen — Training Overview', fontsize=STYLE['suptitle_fs'] + 2, fontweight='bold', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    _plot(axes[0], data, 'eval/full_task_success_rate', title='Eval Full-Task Success', ylabel='Success Rate (%)', color=COLORS['green'], pct=True, smooth_window=1, ymin=0, ymax=105)
    _plot(axes[1], data, 'eval/mean_tasks_completed', title='Eval Mean Tasks Completed', ylabel='Tasks', color=COLORS['blue'], pct=False, smooth_window=1, ymin=0, ymax=4.1)
    _plot(axes[2], data, 'train/option_success_rate', title='Train Option Success Rate', ylabel='Success Rate (%)', color=COLORS['orange'], pct=True, smooth_window=sw, ymin=0, ymax=105)
    _plot(axes[3], data, 'worker/worker_critic_loss', title='Worker Critic Loss', ylabel='MSE Loss', color=COLORS['red'], smooth_window=sw)
    _plot(axes[4], data, 'manager/manager_loss', title='Manager Loss', ylabel='MSE Loss', color=COLORS['purple'], smooth_window=sw)
    _plot(axes[5], data, 'train/episode_reward', title='Train Episode Reward', ylabel='Reward', color=COLORS['teal'], smooth_window=sw)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, '00_overview_dashboard.png')


def plot_eval_success(data: dict, out_dir: str, sw: int = 1):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Evaluation Metrics', fontsize=STYLE['suptitle_fs'], fontweight='bold')
    _plot(axes[0], data, 'eval/any_task_success_rate', title='Any-Task Success', ylabel='%', color=COLORS['green'], pct=True, smooth_window=sw, ymin=0, ymax=105)
    _plot(axes[1], data, 'eval/full_task_success_rate', title='Full-Task Success', ylabel='%', color=COLORS['blue'], pct=True, smooth_window=sw, ymin=0, ymax=105)
    _plot(axes[2], data, 'eval/mean_tasks_completed', title='Mean Tasks Completed', ylabel='Count', color=COLORS['gold'], pct=False, smooth_window=sw, ymin=0, ymax=4.1)
    fig.tight_layout()
    _save(fig, out_dir, '01_eval_success.png')


def plot_training_rewards(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Rewards & Task Progress', fontsize=STYLE['suptitle_fs'], fontweight='bold')
    _plot(axes[0], data, 'train/episode_reward', title='Episode Reward', ylabel='Reward', color=COLORS['blue'], smooth_window=sw)
    _plot(axes[1], data, 'train/tasks_completed', title='Tasks Completed per Episode', ylabel='Count', color=COLORS['gold'], smooth_window=sw, ymin=0, ymax=4.1)
    _plot(axes[2], data, 'train/mean_selected_task_error', title='Selected Task Error', ylabel='Normalized Error', color=COLORS['red'], smooth_window=sw)
    fig.tight_layout()
    _save(fig, out_dir, '02_training_rewards.png')


def plot_manager(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Manager Diagnostics', fontsize=STYLE['suptitle_fs'], fontweight='bold')
    _plot(axes[0], data, 'manager/manager_loss', title='Manager Loss', ylabel='Loss', color=COLORS['purple'], smooth_window=sw)
    _plot(axes[1], data, 'manager/manager_q_mean', title='Manager Q Mean', ylabel='Q', color=COLORS['teal'], smooth_window=sw)
    _plot(axes[2], data, 'train/manager_epsilon', title='Manager Epsilon', ylabel='ε', color=COLORS['gray'], smooth_window=1)
    fig.tight_layout()
    _save(fig, out_dir, '03_manager.png')


def plot_worker(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('Worker Diagnostics', fontsize=STYLE['suptitle_fs'], fontweight='bold')
    _plot(axes[0], data, 'worker/worker_critic_loss', title='Critic Loss', ylabel='Loss', color=COLORS['blue'], smooth_window=sw)
    _plot(axes[1], data, 'worker/worker_actor_loss', title='Actor Loss', ylabel='Loss', color=COLORS['orange'], smooth_window=sw)
    _plot(axes[2], data, 'worker/worker_alpha', title='Entropy Alpha', ylabel='α', color=COLORS['teal'], smooth_window=sw)
    _plot(axes[3], data, 'train/worker_random_prob', title='Worker Random-Action Prob', ylabel='Prob', color=COLORS['gray'], smooth_window=sw)
    fig.tight_layout()
    _save(fig, out_dir, '04_worker.png')


def plot_options_and_buffers(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle('Options, Buffers, and Replay Health', fontsize=STYLE['suptitle_fs'], fontweight='bold')
    _plot(axes[0], data, 'train/option_success_rate', title='Option Success Rate', ylabel='%', color=COLORS['orange'], pct=True, smooth_window=sw, ymin=0, ymax=105)
    _plot(axes[1], data, 'train/high_level_steps', title='High-Level Steps / Episode', ylabel='Count', color=COLORS['dark'], smooth_window=sw)
    _plot(axes[2], data, 'train/option_length_mean', title='Option Length Mean', ylabel='Steps', color=COLORS['purple'], smooth_window=sw)
    _plot(axes[3], data, 'train/worker_buffer_size', title='Worker Buffer Size', ylabel='Transitions', color=COLORS['blue'], smooth_window=sw)
    _plot(axes[4], data, 'train/manager_buffer_size', title='Manager Buffer Size', ylabel='Transitions', color=COLORS['teal'], smooth_window=sw)
    _plot(axes[5], data, 'train/new_completions_per_episode', title='New Completions / Episode', ylabel='Count', color=COLORS['green'], smooth_window=sw, ymin=0)
    fig.tight_layout()
    _save(fig, out_dir, '05_options_and_buffers.png')


def main():
    parser = argparse.ArgumentParser(description='Generate diagnostic plots from TensorBoard logs.')
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=15)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.log_dir, 'plots')
    data = load_tb_scalars(args.log_dir)
    if not data:
        print('No data found.')
        return

    plot_overview_dashboard(data, out_dir, sw=args.smooth)
    plot_eval_success(data, out_dir, sw=1)
    plot_training_rewards(data, out_dir, sw=args.smooth)
    plot_manager(data, out_dir, sw=args.smooth)
    plot_worker(data, out_dir, sw=args.smooth)
    plot_options_and_buffers(data, out_dir, sw=args.smooth)
    print(f'All plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
