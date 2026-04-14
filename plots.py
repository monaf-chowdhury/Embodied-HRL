"""
plots.py — Diagnostic plots for Visual HRL on FrankaKitchen-v1.

Reads TensorBoard event files and generates a set of PNG plots covering
every aspect of training health.

Usage:
    python plots.py --log_dir logs/run1/
    python plots.py --log_dir logs/run1/ --out_dir plots/run1/ --smooth 20
    python plots.py --log_dir logs/ --compare          # overlay multiple runs

Output files (saved to out_dir):
    00_overview_dashboard.png   — 6-panel summary; look here first
    01_eval_success.png         — any-task / full-task / env-reward success rates
    02_training_rewards.png     — episode reward + subgoal success rate
    03_manager.png              — DQN: Q-loss, Q-mean, epsilon schedule
    04_worker_sac.png           — SAC: critic loss, actor loss, entropy alpha
    05_task_completion.png      — tasks completed per episode + hindsight pool
    06_buffers.png              — buffer fill levels + active landmarks
    07_latent_space.png         — success threshold + subgoal attempt rate
    comparison.png              — (--compare mode) multi-run overlay
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed.  Run: pip install tensorboard")
    raise


# =============================================================================
# Colour palette — consistent across all plots
# =============================================================================

COLORS = {
    'green':      '#2ecc71',
    'blue':       '#3498db',
    'orange':     '#e67e22',
    'red':        '#e74c3c',
    'purple':     '#9b59b6',
    'teal':       '#1abc9c',
    'gray':       '#95a5a6',
    'dark_blue':  '#2c3e50',
    'gold':       '#f39c12',
    'pink':       '#e91e8c',
}

STYLE = {
    'raw_alpha':    0.20,
    'raw_lw':       0.7,
    'smooth_alpha': 0.92,
    'smooth_lw':    2.0,
    'grid_alpha':   0.25,
    'title_fs':     11,
    'label_fs':     9,
    'tick_fs':      8,
    'suptitle_fs':  13,
}


# =============================================================================
# Data loading
# =============================================================================

def load_tb_scalars(log_dir: str) -> dict:
    """Load all scalar tags from a TensorBoard event directory."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"  WARNING: No scalar data in {log_dir}")
        return {}
    data = {}
    for tag in tags:
        events     = ea.Scalars(tag)
        steps      = np.array([e.step  for e in events])
        values     = np.array([e.value for e in events])
        data[tag]  = (steps, values)
        print(f"  Loaded  {tag}  ({len(steps)} pts)")
    return data


# =============================================================================
# Smoothing
# =============================================================================

def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


# =============================================================================
# Core plotting primitive
# =============================================================================

def _plot(ax, data: dict, tag: str,
          title: str  = None,
          ylabel: str = None,
          color: str  = 'steelblue',
          smooth_window: int  = 10,
          pct:   bool = False,
          hline: float = None,
          hline_label: str = None,
          hline_color: str = 'red',
          ymin:  float = None,
          ymax:  float = None):
    """Plot one TensorBoard tag onto an axes object."""
    if tag not in data:
        ax.text(0.5, 0.5, f'No data\n({tag})',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=STYLE['label_fs'])
        ax.set_title(title or tag, fontsize=STYLE['title_fs'], fontweight='bold')
        _style_ax(ax)
        return

    steps, values = data[tag]
    smoothed = smooth(values, smooth_window)

    if pct:
        values   = values   * 100.0
        smoothed = smoothed * 100.0

    ax.plot(steps, values,
            alpha=STYLE['raw_alpha'], color=color, linewidth=STYLE['raw_lw'])
    ax.plot(steps, smoothed,
            alpha=STYLE['smooth_alpha'], color=color, linewidth=STYLE['smooth_lw'])

    if hline is not None:
        ax.axhline(hline, color=hline_color, linestyle='--',
                   linewidth=1.2, alpha=0.7,
                   label=hline_label or f'y={hline}')
        if hline_label:
            ax.legend(fontsize=STYLE['tick_fs'])

    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    ax.set_title(title or tag, fontsize=STYLE['title_fs'], fontweight='bold')
    ax.set_xlabel('Environment Steps', fontsize=STYLE['label_fs'])
    ax.set_ylabel(ylabel or ('%' if pct else ''), fontsize=STYLE['label_fs'])
    _style_ax(ax)


def _style_ax(ax):
    ax.grid(True, alpha=STYLE['grid_alpha'], linestyle='--', linewidth=0.6)
    ax.tick_params(labelsize=STYLE['tick_fs'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _:
            f'{x/1e6:.1f}M' if x >= 1_000_000 else
            f'{x/1e3:.0f}k' if x >= 1_000 else str(int(x))
        )
    )


def _save(fig, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved  →  {path}")


# =============================================================================
# Plot 00 — Overview Dashboard
# =============================================================================

def plot_overview_dashboard(data: dict, out_dir: str, sw: int = 15):
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor('white')
    fig.suptitle('Visual HRL — Training Overview Dashboard',
                 fontsize=STYLE['suptitle_fs'] + 2, fontweight='bold', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    _plot(axes[0], data, 'eval/any_task_success_rate',
          title='[1]  Eval  Any-Task Success',
          ylabel='Success Rate (%)', color=COLORS['green'],
          pct=True, smooth_window=1, ymin=0, ymax=105)

    _plot(axes[1], data, 'train/subgoal_success_rate',
          title='[2]  Subgoal Success Rate',
          ylabel='Success Rate (%)', color=COLORS['orange'],
          pct=True, smooth_window=sw)

    _plot(axes[2], data, 'train/episode_reward',
          title='[3]  Episode Reward  (train)',
          ylabel='Reward', color=COLORS['blue'],
          smooth_window=sw)

    _plot(axes[3], data, 'worker/worker_critic_loss',
          title='[4]  Worker Critic Loss',
          ylabel='MSE Loss', color=COLORS['red'],
          smooth_window=sw)

    _plot(axes[4], data, 'manager/manager_loss',
          title='[5]  Manager Q-Loss',
          ylabel='MSE Loss', color=COLORS['purple'],
          smooth_window=sw)

    _plot(axes[5], data, 'manager/manager_q_mean',
          title='[6]  Manager Q-Mean',
          ylabel='Q-Value', color=COLORS['teal'],
          smooth_window=sw)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, '00_overview_dashboard.png')


# =============================================================================
# Plot 01 — Eval Success (all three metrics)
# =============================================================================

def plot_eval_success(data: dict, out_dir: str, sw: int = 1):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Evaluation Success Metrics',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'eval/any_task_success_rate',
          title='Any-Task Success  (≥1 task)',
          ylabel='%', color=COLORS['green'],
          pct=True, smooth_window=sw, ymin=0, ymax=105)

    _plot(axes[1], data, 'eval/full_task_success_rate',
          title='Full-Task Success  (all 4 tasks)',
          ylabel='%', color=COLORS['blue'],
          pct=True, smooth_window=sw, ymin=0, ymax=105)

    _plot(axes[2], data, 'eval/env_reward_success_rate',
          title='Env-Reward Success  (ep_reward > 0)\n[legacy — may be misleading]',
          ylabel='%', color=COLORS['gray'],
          pct=True, smooth_window=sw, ymin=0, ymax=105)

    fig.tight_layout()
    _save(fig, out_dir, '01_eval_success.png')


# =============================================================================
# Plot 02 — Training Rewards
# =============================================================================

def plot_training_rewards(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Rewards & Subgoal Performance',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'train/episode_reward',
          title='Episode Env Reward  (train)',
          ylabel='Cumulative Reward', color=COLORS['blue'],
          smooth_window=sw)

    _plot(axes[1], data, 'train/subgoal_success_rate',
          title='Subgoal Success Rate  (train)',
          ylabel='%', color=COLORS['orange'],
          pct=True, smooth_window=sw)

    fig.tight_layout()
    _save(fig, out_dir, '02_training_rewards.png')


# =============================================================================
# Plot 03 — Manager (DQN)
# =============================================================================

def plot_manager(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Manager  (DQN)',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'manager/manager_loss',
          title='Q-Loss',
          ylabel='MSE Loss', color=COLORS['red'],
          smooth_window=sw)

    _plot(axes[1], data, 'manager/manager_q_mean',
          title='Mean Q-Value',
          ylabel='Q-Value', color=COLORS['purple'],
          smooth_window=sw)

    _plot(axes[2], data, 'train/epsilon',
          title='Epsilon  (exploration schedule)',
          ylabel='ε', color=COLORS['gray'],
          smooth_window=1)            # no smoothing — step function

    fig.tight_layout()
    _save(fig, out_dir, '03_manager.png')


# =============================================================================
# Plot 04 — Worker (SAC)
# =============================================================================

def plot_worker_sac(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Worker  (SAC)',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'worker/worker_critic_loss',
          title='Critic Loss',
          ylabel='MSE Loss', color=COLORS['blue'],
          smooth_window=sw)

    _plot(axes[1], data, 'worker/worker_actor_loss',
          title='Actor Loss',
          ylabel='Loss', color=COLORS['orange'],
          smooth_window=sw)

    _plot(axes[2], data, 'worker/worker_alpha',
          title='Entropy Temperature  (α)',
          ylabel='Alpha', color=COLORS['teal'],
          smooth_window=sw)

    fig.tight_layout()
    _save(fig, out_dir, '04_worker_sac.png')


# =============================================================================
# Plot 05 — Task Completion
# =============================================================================

def plot_task_completion(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Task Completion & Hindsight',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'train/any_task_success',
          title='Any-Task Success  (train, per episode)',
          ylabel='%', color=COLORS['green'],
          pct=True, smooth_window=sw)

    _plot(axes[1], data, 'train/tasks_completed',
          title='Tasks Completed per Episode  (train)',
          ylabel='Count', color=COLORS['gold'],
          smooth_window=sw)

    _plot(axes[2], data, 'train/hindsight_pool_size',
          title='Hindsight Success Pool Size',
          ylabel='States in Pool', color=COLORS['pink'],
          smooth_window=5)

    fig.tight_layout()
    _save(fig, out_dir, '05_task_completion.png')


# =============================================================================
# Plot 06 — Buffers & Landmarks
# =============================================================================

def plot_buffers(data: dict, out_dir: str, sw: int = 5):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Replay Buffers & Landmark Health',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'train/low_buffer_size',
          title='Low-Level Buffer  (worker transitions)',
          ylabel='Transitions', color=COLORS['blue'],
          smooth_window=sw)

    _plot(axes[1], data, 'train/high_buffer_size',
          title='High-Level Buffer  (manager transitions)',
          ylabel='Transitions', color=COLORS['orange'],
          smooth_window=sw)

    _plot(axes[2], data, 'train/n_active_landmarks',
          title='Active Landmarks',
          ylabel='Count', color=COLORS['teal'],
          smooth_window=sw)

    fig.tight_layout()
    _save(fig, out_dir, '06_buffers.png')


# =============================================================================
# Plot 07 — Latent Space Health
# =============================================================================

def plot_latent_space(data: dict, out_dir: str, sw: int = 10):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Latent Space & Subgoal Dynamics',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'train/success_threshold',
          title='Success Threshold  (auto-calibrated)',
          ylabel='Latent Distance', color=COLORS['red'],
          smooth_window=sw)

    _plot(axes[1], data, 'train/subgoal_attempts',
          title='Subgoal Attempts per Episode',
          ylabel='Count', color=COLORS['purple'],
          smooth_window=sw)

    _plot(axes[2], data, 'train/high_level_steps',
          title='High-Level Steps per Episode',
          ylabel='Count', color=COLORS['dark_blue'],
          smooth_window=sw)

    fig.tight_layout()
    _save(fig, out_dir, '07_latent_space.png')


# =============================================================================
# Multi-run comparison (--compare mode)
# =============================================================================

def plot_comparison(run_dirs: list, run_labels: list,
                    out_dir: str, sw: int = 15):
    COMPARE_TAGS = [
        ('eval/any_task_success_rate',   'Eval Any-Task Success (%)',   True),
        ('train/subgoal_success_rate',   'Subgoal Success Rate (%)',    True),
        ('train/episode_reward',         'Episode Reward',              False),
        ('worker/worker_critic_loss',    'Worker Critic Loss',          False),
        ('manager/manager_q_mean',       'Manager Q-Mean',              False),
        ('train/success_threshold',      'Success Threshold',           False),
    ]

    palette = list(COLORS.values())
    n_tags  = len(COMPARE_TAGS)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle('Multi-Run Comparison',
                 fontsize=STYLE['suptitle_fs'] + 1, fontweight='bold')

    for run_dir, label, color in zip(run_dirs, run_labels, palette):
        print(f"  Loading run: {label}")
        d = load_tb_scalars(run_dir)
        for ax, (tag, ylabel, pct) in zip(axes, COMPARE_TAGS):
            if tag not in d:
                continue
            steps, vals = d[tag]
            s = smooth(vals, sw)
            if pct:
                vals *= 100; s *= 100
            ax.plot(steps, vals, alpha=0.12, color=color, linewidth=0.7)
            ax.plot(steps, s,    alpha=0.90, color=color, linewidth=2.0, label=label)

    for ax, (_, ylabel, pct) in zip(axes, COMPARE_TAGS):
        ax.set_title(ylabel, fontsize=STYLE['title_fs'], fontweight='bold')
        ax.set_xlabel('Environment Steps', fontsize=STYLE['label_fs'])
        ax.set_ylabel('%' if pct else '', fontsize=STYLE['label_fs'])
        ax.legend(fontsize=STYLE['tick_fs'])
        _style_ax(ax)

    fig.tight_layout()
    _save(fig, out_dir, 'comparison.png')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate diagnostic plots from TensorBoard logs.'
    )
    parser.add_argument('--log_dir', type=str, required=True,
                        help='TensorBoard log directory for a single run, '
                             'or parent directory for --compare mode.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Where to save PNGs (default: <log_dir>/plots/).')
    parser.add_argument('--smooth',  type=int, default=15,
                        help='Moving-average window (default: 15).')
    parser.add_argument('--compare', action='store_true',
                        help='Overlay all immediate subdirectories as separate runs.')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.log_dir, 'plots')
    sw      = args.smooth

    # ---- Multi-run comparison ----
    if args.compare:
        subdirs = sorted([
            os.path.join(args.log_dir, d)
            for d in os.listdir(args.log_dir)
            if os.path.isdir(os.path.join(args.log_dir, d))
        ])
        if not subdirs:
            print("No subdirectories found.  Nothing to compare.")
            return
        labels = [os.path.basename(d) for d in subdirs]
        print(f"\nComparing {len(subdirs)} runs: {labels}")
        plot_comparison(subdirs, labels, out_dir, sw=sw)
        print(f"\nComparison saved → {out_dir}/comparison.png")
        return

    # ---- Single run ----
    print(f"\nLoading TensorBoard data from: {args.log_dir}")
    data = load_tb_scalars(args.log_dir)

    if not data:
        print("No data found.  Make sure training has started.")
        return

    print(f"\nGenerating plots → {out_dir}/\n")

    plot_overview_dashboard(data, out_dir, sw=sw)
    plot_eval_success(data,       out_dir, sw=1)      # no smoothing — too few eval points
    plot_training_rewards(data,   out_dir, sw=sw)
    plot_manager(data,            out_dir, sw=sw)
    plot_worker_sac(data,         out_dir, sw=sw)
    plot_task_completion(data,    out_dir, sw=sw)
    plot_buffers(data,            out_dir, sw=5)
    plot_latent_space(data,       out_dir, sw=sw)

    print(f"\n{'─'*60}")
    print(f"  All plots saved to: {out_dir}/")
    print(f"{'─'*60}")
    pngs = sorted(f for f in os.listdir(out_dir) if f.endswith('.png'))
    for f in pngs:
        print(f"    {f}")
    print()


if __name__ == '__main__':
    main()