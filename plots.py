import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise


def load_tb_scalars(log_dir: str) -> dict:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    data = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        data[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))
    return data


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


def _plot(ax, data, tag, title, ylabel, pct=False, smooth_window=10):
    if tag not in data:
        ax.text(0.5, 0.5, f'No data\n({tag})', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    steps, values = data[tag]
    smoothed = smooth(values, smooth_window)
    if pct:
        values = values * 100
        smoothed = smoothed * 100
    ax.plot(steps, values, alpha=0.25)
    ax.plot(steps, smoothed, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=15)
    args = parser.parse_args()
    data = load_tb_scalars(args.log_dir)
    out_dir = args.out_dir or os.path.join(args.log_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _plot(axes[0,0], data, 'eval/any_task_success_rate', 'Eval Any-Task Success', '%', pct=True, smooth_window=1)
    _plot(axes[0,1], data, 'eval/full_task_success_rate', 'Eval Full-Task Success', '%', pct=True, smooth_window=1)
    _plot(axes[0,2], data, 'train/subgoal_success_rate', 'Subgoal Success Rate', '%', pct=True, smooth_window=args.smooth)
    _plot(axes[1,0], data, 'train/episode_reward', 'Episode Reward', 'Reward', smooth_window=args.smooth)
    _plot(axes[1,1], data, 'manager/manager_q_mean', 'Manager Q Mean', 'Q', smooth_window=args.smooth)
    _plot(axes[1,2], data, 'worker/worker_critic_loss', 'Worker Critic Loss', 'Loss', smooth_window=args.smooth)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'overview.png'), dpi=120, bbox_inches='tight')


if __name__ == '__main__':
    main()
