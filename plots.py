"""
plots.py — Diagnostic plots for Visual HRL training.

Reads TensorBoard event files from a log directory and generates
PNG plots for every tracked metric. Run after (or during) training.

Usage:
    python plots.py --log_dir logs/seed42/
    python plots.py --log_dir logs/seed42/ --out_dir plots/seed42/
    python plots.py --log_dir logs/  # compares multiple runs
    python plots.py --log_dir logs/seed42/ --smooth 20 # Adjust smoothing

Plots:
    00_overview_dashboard.png — the 6 most important metrics on one page, look at this first
    01_training_progress.png — eval success, episode reward, subgoal SR
    02_manager.png — Q-loss, mean Q-value, epsilon decay
    03_worker_sac.png — critic loss, actor loss, alpha
    04_reachability.png — BCE loss and accuracy (with 50% random baseline line)
    05_buffers.png — all three buffer fill levels
    06_episode_structure.png — high-level steps per episode, eval reward
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless — no display needed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# TensorBoard log reader
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    raise


# =============================================================================
# Helper: load all scalar tags from a TensorBoard log dir
# =============================================================================

def load_tb_scalars(log_dir: str) -> dict:
    """
    Load all scalar data from a TensorBoard event file.
    Returns: { tag: (steps_array, values_array) }
    """
    ea = EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"WARNING: No scalar data found in {log_dir}")
        return {}

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps  = np.array([e.step  for e in events])
        values = np.array([e.value for e in events])
        data[tag] = (steps, values)
        print(f"  Loaded: {tag}  ({len(steps)} points)")

    return data


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


# =============================================================================
# Individual plot functions
# =============================================================================

def _plot(ax, data, tag, label=None, color='steelblue',
          smooth_window=10, ylabel=None, title=None, pct=False):
    """Plot a single tag onto an axis, with optional smoothing."""
    if tag not in data:
        ax.text(0.5, 0.5, f'No data\n({tag})',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=9)
        ax.set_title(title or tag)
        return

    steps, values = data[tag]
    smoothed = smooth(values, smooth_window)

    if pct:
        values   = values   * 100
        smoothed = smoothed * 100

    ax.plot(steps, values,   alpha=0.25, color=color, linewidth=0.8)
    ax.plot(steps, smoothed, alpha=0.9,  color=color, linewidth=1.8,
            label=label or tag)

    ax.set_xlabel('Environment Steps', fontsize=9)
    ax.set_ylabel(ylabel or ('(%)' if pct else ''), fontsize=9)
    ax.set_title(title or tag, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # Format x-axis as 'k' / 'M'
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6
                          else f'{x/1e3:.0f}k'))


def save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Plot groups
# =============================================================================

def plot_training_progress(data, out_dir, smooth_window=10):
    """Success rate, episode reward, subgoal success rate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training Progress', fontsize=13, fontweight='bold', y=1.01)

    _plot(axes[0], data, 'eval/success_rate',
          title='Eval Success Rate',
          ylabel='Success Rate (%)',
          color='green', pct=True, smooth_window=smooth_window)

    _plot(axes[1], data, 'train/episode_reward',
          title='Episode Reward (Train)',
          ylabel='Cumulative Reward',
          color='steelblue', smooth_window=smooth_window)

    _plot(axes[2], data, 'train/subgoal_success_rate',
          title='Subgoal Success Rate',
          ylabel='Success Rate (%)',
          color='darkorange', pct=True, smooth_window=smooth_window)

    fig.tight_layout()
    save(fig, out_dir, '01_training_progress.png')


def plot_manager(data, out_dir, smooth_window=10):
    """Manager Q-loss, mean Q-value, epsilon."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Manager (DQN)', fontsize=13, fontweight='bold', y=1.01)

    _plot(axes[0], data, 'manager/manager_loss',
          title='Manager Q-Loss',
          ylabel='MSE Loss',
          color='crimson', smooth_window=smooth_window)

    _plot(axes[1], data, 'manager/manager_q_mean',
          title='Mean Q-Value',
          ylabel='Q-Value',
          color='purple', smooth_window=smooth_window)

    _plot(axes[2], data, 'train/epsilon',
          title='Epsilon (Exploration)',
          ylabel='Epsilon',
          color='gray', smooth_window=1)  # no smoothing for epsilon

    fig.tight_layout()
    save(fig, out_dir, '02_manager.png')


def plot_worker(data, out_dir, smooth_window=10):
    """Worker SAC: critic loss, actor loss, entropy alpha."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Worker (SAC)', fontsize=13, fontweight='bold', y=1.01)

    _plot(axes[0], data, 'worker/worker_critic_loss',
          title='Critic Loss',
          ylabel='MSE Loss',
          color='steelblue', smooth_window=smooth_window)

    _plot(axes[1], data, 'worker/worker_actor_loss',
          title='Actor Loss',
          ylabel='Loss',
          color='darkorange', smooth_window=smooth_window)

    _plot(axes[2], data, 'worker/worker_alpha',
          title='Entropy Temperature (Alpha)',
          ylabel='Alpha',
          color='teal', smooth_window=smooth_window)

    fig.tight_layout()
    save(fig, out_dir, '03_worker_sac.png')


def plot_reachability(data, out_dir, smooth_window=10):
    """Reachability predictor: BCE loss and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Reachability Predictor', fontsize=13, fontweight='bold', y=1.01)

    _plot(axes[0], data, 'reachability/reach_loss',
          title='BCE Loss',
          ylabel='Binary Cross-Entropy',
          color='crimson', smooth_window=smooth_window)

    _plot(axes[1], data, 'reachability/reach_accuracy',
          title='Classification Accuracy',
          ylabel='Accuracy (%)',
          color='green', pct=True, smooth_window=smooth_window)

    # Add 50% reference line on accuracy (random baseline)
    if 'reachability/reach_accuracy' in data:
        axes[1].axhline(50, color='red', linestyle='--',
                        linewidth=1, alpha=0.6, label='Random (50%)')
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    save(fig, out_dir, '04_reachability.png')


def plot_buffers(data, out_dir, smooth_window=5):
    """Buffer fill levels over training."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Replay Buffer Sizes', fontsize=13, fontweight='bold', y=1.01)

    _plot(axes[0], data, 'train/low_buffer_size',
          title='Low-Level Buffer (Worker)',
          ylabel='Transitions',
          color='steelblue', smooth_window=smooth_window)

    _plot(axes[1], data, 'train/high_buffer_size',
          title='High-Level Buffer (Manager / FER)',
          ylabel='Transitions',
          color='darkorange', smooth_window=smooth_window)

    # Subgoal successes per episode
    _plot(axes[2], data, 'train/subgoal_successes',
          title='Subgoal Successes per Episode',
          ylabel='Count',
          color='green', smooth_window=smooth_window)

    fig.tight_layout()
    save(fig, out_dir, '05_buffers.png')


def plot_episode_structure(data, out_dir, smooth_window=10):
    """High-level steps per episode and eval reward."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Episode Structure', fontsize=13, fontweight='bold', y=1.01)

    _plot(axes[0], data, 'train/high_level_steps',
          title='High-Level Steps per Episode',
          ylabel='Steps',
          color='purple', smooth_window=smooth_window)

    _plot(axes[1], data, 'eval/mean_reward',
          title='Eval Mean Reward',
          ylabel='Reward',
          color='green', smooth_window=1)

    fig.tight_layout()
    save(fig, out_dir, '06_episode_structure.png')


def plot_overview_dashboard(data, out_dir, smooth_window=10):
    """
    Single-page summary dashboard with the 6 most important metrics.
    This is the first thing to look at.
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Training Overview Dashboard', fontsize=15,
                 fontweight='bold', y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    _plot(axes[0], data, 'eval/success_rate',
          title='[1] Eval Success Rate',
          ylabel='%', color='green', pct=True, smooth_window=1)

    _plot(axes[1], data, 'train/subgoal_success_rate',
          title='[2] Subgoal Success Rate',
          ylabel='%', color='darkorange', pct=True, smooth_window=smooth_window)

    _plot(axes[2], data, 'train/episode_reward',
          title='[3] Episode Reward',
          ylabel='Reward', color='steelblue', smooth_window=smooth_window)

    _plot(axes[3], data, 'worker/worker_critic_loss',
          title='[4] Worker Critic Loss',
          ylabel='Loss', color='crimson', smooth_window=smooth_window)

    _plot(axes[4], data, 'manager/manager_loss',
          title='[5] Manager Q-Loss',
          ylabel='Loss', color='purple', smooth_window=smooth_window)

    _plot(axes[5], data, 'reachability/reach_accuracy',
          title='[6] Reachability Accuracy',
          ylabel='%', color='teal', pct=True, smooth_window=smooth_window)

    if 'reachability/reach_accuracy' in data:
        axes[5].axhline(50, color='red', linestyle='--',
                        linewidth=1, alpha=0.5, label='Random')
        axes[5].legend(fontsize=7)

    save(fig, out_dir, '00_overview_dashboard.png')


# =============================================================================
# Multi-run comparison (optional)
# =============================================================================

def plot_comparison(run_dirs: list, run_labels: list, out_dir: str,
                    tags=None, smooth_window=10):
    """
    Overlay multiple runs on the same axes for seed comparison / ablations.
    """
    if tags is None:
        tags = [
            ('eval/success_rate',           'Eval Success Rate (%)',       True),
            ('train/subgoal_success_rate',  'Subgoal Success Rate (%)',    True),
            ('train/episode_reward',        'Episode Reward',              False),
            ('worker/worker_critic_loss',   'Worker Critic Loss',          False),
        ]

    colors = ['steelblue', 'darkorange', 'green', 'crimson',
              'purple', 'teal', 'brown', 'gray']

    fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 4))
    if len(tags) == 1:
        axes = [axes]
    fig.suptitle('Multi-Run Comparison', fontsize=13,
                 fontweight='bold', y=1.01)

    for run_dir, label, color in zip(run_dirs, run_labels,
                                     colors[:len(run_dirs)]):
        print(f"\nLoading run: {label} from {run_dir}")
        data = load_tb_scalars(run_dir)

        for ax, (tag, ylabel, pct) in zip(axes, tags):
            if tag not in data:
                continue
            steps, values = data[tag]
            smoothed = smooth(values, smooth_window)
            if pct:
                values   = values   * 100
                smoothed = smoothed * 100
            ax.plot(steps, values,   alpha=0.15, color=color, linewidth=0.8)
            ax.plot(steps, smoothed, alpha=0.9,  color=color,
                    linewidth=1.8, label=label)
            ax.set_title(ylabel, fontsize=10, fontweight='bold')
            ax.set_xlabel('Steps', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'
                                  if x >= 1e6 else f'{x/1e3:.0f}k'))

    for ax in axes:
        ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, out_dir, 'comparison.png')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate diagnostic plots from TensorBoard logs.')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to TensorBoard log directory (single run) '
                             'or parent dir containing multiple runs.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Where to save PNGs. Defaults to log_dir/plots/')
    parser.add_argument('--smooth', type=int, default=15,
                        help='Smoothing window size (default: 15)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all subdirectories as separate runs.')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.log_dir, 'plots')

    # ---- Comparison mode: multiple subdirs ----
    if args.compare:
        subdirs = sorted([
            os.path.join(args.log_dir, d)
            for d in os.listdir(args.log_dir)
            if os.path.isdir(os.path.join(args.log_dir, d))
        ])
        if not subdirs:
            print("No subdirectories found for comparison.")
            return
        labels = [os.path.basename(d) for d in subdirs]
        print(f"\nComparing {len(subdirs)} runs: {labels}")
        plot_comparison(subdirs, labels, out_dir, smooth_window=args.smooth)
        print(f"\nComparison plot saved to: {out_dir}/comparison.png")
        return

    # ---- Single run mode ----
    print(f"\nLoading TensorBoard data from: {args.log_dir}")
    data = load_tb_scalars(args.log_dir)

    if not data:
        print("No data found. Make sure training has started and the "
              "log_dir contains TensorBoard event files.")
        return

    print(f"\nGenerating plots -> {out_dir}/")

    plot_overview_dashboard(data,  out_dir, args.smooth)
    plot_training_progress(data,   out_dir, args.smooth)
    plot_manager(data,             out_dir, args.smooth)
    plot_worker(data,              out_dir, args.smooth)
    plot_reachability(data,        out_dir, args.smooth)
    plot_buffers(data,             out_dir, args.smooth)
    plot_episode_structure(data,   out_dir, args.smooth)

    print(f"\nAll plots saved to: {out_dir}/")
    print("Files:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.png'):
            print(f"  {f}")


if __name__ == '__main__':
    main()
