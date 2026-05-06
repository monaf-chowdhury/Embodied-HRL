"""
plots.py — Diagnostic plots for SMGW training runs.

Reads TensorBoard event files and generates PNG plots covering:
  * eval full-task vs any-task success rates (the headline chart)
  * task-completion dynamics during training
  * manager DQN losses + Q-values
  * worker SAC losses + entropy alpha
  * buffer fills + option length
  * stage-A warmup sanity (BC / CE losses)

Usage:
    python plots.py --log_dir logs/smgw_run1/
    python plots.py --log_dir logs/smgw_run1/ --out_dir plots/run1/ --smooth 20
    python plots.py --log_dir logs/ --compare         # overlay multiple runs

Output files:
    00_overview_dashboard.png     — 6-panel summary; look here first
    01_eval_success.png           — eval success rates over time
    02_training_episode.png       — per-episode env reward + options + tasks completed
    03_manager_dqn.png            — manager/controller diagnostics
    04_worker_sac.png             — worker critic/actor loss, alpha
    05_buffers.png                — buffer fills
    06_warmup.png                 — Stage-A BC/CE losses (if collected)
    comparison.png                — multi-run overlay (--compare mode)
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
# Colour palette
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
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"  WARNING: No scalar data in {log_dir}")
        return {}
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        data[tag] = (steps, values)
    print(f"  Loaded {len(data)} scalar tags from {log_dir}")
    return data


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


def _plot(ax, data: dict, tag: str,
          title: str = None, ylabel: str = None,
          color: str = 'steelblue', smooth_window: int = 10,
          pct: bool = False, hline: float = None,
          hline_label: str = None, hline_color: str = 'red',
          ymin: float = None, ymax: float = None):
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
        values = values * 100.0
        smoothed = smoothed * 100.0

    ax.plot(steps, values, alpha=STYLE['raw_alpha'],
            color=color, linewidth=STYLE['raw_lw'])
    ax.plot(steps, smoothed, alpha=STYLE['smooth_alpha'],
            color=color, linewidth=STYLE['smooth_lw'])

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


def _plot_first_available(ax, data: dict, tags: list[str], **kwargs):
    for tag in tags:
        if tag in data:
            return _plot(ax, data, tag, **kwargs)
    return _plot(ax, data, tags[0], **kwargs)


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
    print(f"  Saved  ->  {path}")


# =============================================================================
# Plot 00 — Overview Dashboard
# =============================================================================

def plot_overview(data: dict, out_dir: str, sw: int = 15):
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor('white')
    fig.suptitle('SMGW — Training Overview Dashboard',
                 fontsize=STYLE['suptitle_fs'] + 2, fontweight='bold', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    _plot(axes[0], data, 'eval/full_task_success_rate',
          title='[1]  Eval Full-Task Success  (HEADLINE)',
          ylabel='Success Rate (%)', color=COLORS['green'],
          pct=True, smooth_window=1, ymin=0, ymax=105)

    _plot(axes[1], data, 'eval/any_task_success_rate',
          title='[2]  Eval Any-Task Success',
          ylabel='Success Rate (%)', color=COLORS['blue'],
          pct=True, smooth_window=1, ymin=0, ymax=105)

    _plot(axes[2], data, 'eval/mean_tasks_completed',
          title='[3]  Eval Mean Tasks Completed',
          ylabel='Tasks', color=COLORS['gold'],
          smooth_window=1, ymin=0, ymax=4.1)

    _plot(axes[3], data, 'train/ep_tasks_completed',
          title='[4]  Train Tasks Completed / episode',
          ylabel='Tasks', color=COLORS['orange'], smooth_window=sw)

    _plot(axes[4], data, 'worker/worker_critic_loss',
          title='[5]  Worker Critic Loss',
          ylabel='MSE', color=COLORS['red'], smooth_window=sw)

    _plot_first_available(axes[5], data,
          ['manager/manager_loss', 'train/controller_fraction'],
          title='[6]  High-Level Controller',
          ylabel='MSE / fraction', color=COLORS['purple'], smooth_window=sw)

    fig.tight_layout()
    _save(fig, out_dir, '00_overview_dashboard.png')


# =============================================================================
# Plot 01 — Eval Success
# =============================================================================

def plot_eval_success(data: dict, out_dir: str, sw: int = 1):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Evaluation Success Rates',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'eval/full_task_success_rate',
          title='Full-Task Success (all configured tasks)',
          ylabel='%', color=COLORS['green'],
          pct=True, smooth_window=sw, ymin=0, ymax=105)
    _plot(axes[1], data, 'eval/any_task_success_rate',
          title='Any-Task Success (>=1)',
          ylabel='%', color=COLORS['blue'],
          pct=True, smooth_window=sw, ymin=0, ymax=105)
    _plot(axes[2], data, 'eval/mean_chosen_task_success',
          title='Chosen-Task Success (per option)',
          ylabel='%', color=COLORS['teal'],
          pct=True, smooth_window=sw, ymin=0, ymax=105)

    fig.tight_layout()
    _save(fig, out_dir, '01_eval_success.png')


# =============================================================================
# Plot 02 — Training Episode Dynamics
# =============================================================================

def plot_training_episode(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Episode Dynamics',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'train/ep_env_reward',
          title='Episode Env Reward',
          ylabel='Reward', color=COLORS['blue'], smooth_window=sw)
    _plot(axes[1], data, 'train/ep_tasks_completed',
          title='Tasks Completed / Episode',
          ylabel='Tasks', color=COLORS['gold'], smooth_window=sw)
    _plot(axes[2], data, 'train/ep_options',
          title='Options Used / Episode',
          ylabel='Options', color=COLORS['purple'], smooth_window=sw)

    fig.tight_layout()
    _save(fig, out_dir, '02_training_episode.png')


# =============================================================================
# Plot 03 — Manager DQN
# =============================================================================

def plot_manager(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('High-Level Controller',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot_first_available(axes[0], data,
          ['manager/manager_loss', 'train/controller_fraction'],
          title='Manager Loss / Controller Fraction', ylabel='MSE / fraction',
          color=COLORS['red'], smooth_window=sw)
    _plot_first_available(axes[1], data,
          ['manager/manager_q_mean', 'train/unlocked_task_count'],
          title='Mean Q(s,*) / Unlocked Tasks', ylabel='Q-value / count',
          color=COLORS['purple'], smooth_window=sw)
    _plot_first_available(axes[2], data,
          ['train/epsilon', 'train/scripted_manager_fraction', 'train/controller_fraction'],
          title='Exploration / Scripted Fraction', ylabel='eps / fraction',
          color=COLORS['gray'], smooth_window=sw, ymin=0, ymax=1.05)

    fig.tight_layout()
    _save(fig, out_dir, '03_manager_dqn.png')


# =============================================================================
# Plot 04 — Worker SAC
# =============================================================================

def plot_worker(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Grounded Worker (SAC)',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'worker/worker_critic_loss',
          title='Critic Loss', ylabel='MSE',
          color=COLORS['red'], smooth_window=sw)
    _plot(axes[1], data, 'worker/worker_actor_loss',
          title='Actor Loss', ylabel='Loss',
          color=COLORS['blue'], smooth_window=sw)
    _plot(axes[2], data, 'worker/worker_alpha',
          title='Entropy Temperature (alpha)',
          ylabel='alpha', color=COLORS['teal'], smooth_window=sw)

    fig.tight_layout()
    _save(fig, out_dir, '04_worker_sac.png')


# =============================================================================
# Plot 05 — Buffers
# =============================================================================

def plot_buffers(data: dict, out_dir: str, sw: int = 5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Replay Buffers', fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'train/worker_buffer_size',
          title='Worker Buffer (chunk transitions)',
          ylabel='Transitions', color=COLORS['blue'], smooth_window=sw)
    _plot(axes[1], data, 'train/manager_buffer_size',
          title='Manager Buffer (option transitions)',
          ylabel='Transitions', color=COLORS['orange'], smooth_window=sw)

    fig.tight_layout()
    _save(fig, out_dir, '05_buffers.png')


# =============================================================================
# Plot 06 — Warmup losses (if present)
# =============================================================================

def plot_warmup(data: dict, out_dir: str):
    keys = [k for k in data if k.startswith('warmup/')]
    if not keys:
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('Stage-A Warmup Metrics',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')
    for i, tag in enumerate(sorted(keys)):
        steps, values = data[tag]
        ax.plot(steps, values, marker='o', label=tag.replace('warmup/', ''))
    ax.legend(fontsize=STYLE['tick_fs'])
    _style_ax(ax)
    ax.set_xlabel('Step', fontsize=STYLE['label_fs'])
    ax.set_ylabel('Value', fontsize=STYLE['label_fs'])
    fig.tight_layout()
    _save(fig, out_dir, '06_warmup.png')


# =============================================================================
# Multi-run comparison
# =============================================================================

def plot_comparison(run_dirs, run_labels, out_dir: str, sw: int = 15):
    COMPARE_TAGS = [
        ('eval/full_task_success_rate',  'Eval Full-Task Success (%)',  True),
        ('eval/any_task_success_rate',   'Eval Any-Task Success (%)',   True),
        ('eval/mean_tasks_completed',    'Eval Mean Tasks Completed',   False),
        ('train/ep_env_reward',          'Train Episode Env Reward',    False),
        ('worker/worker_critic_loss',    'Worker Critic Loss',          False),
        ('manager/manager_q_mean',       'Manager Q-Mean',              False),
    ]

    palette = list(COLORS.values())
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
                vals = vals * 100; s = s * 100
            ax.plot(steps, vals, alpha=0.12, color=color, linewidth=0.7)
            ax.plot(steps, s, alpha=0.90, color=color, linewidth=2.0, label=label)

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
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=15)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.log_dir, 'plots')
    sw = args.smooth

    if args.compare:
        subdirs = sorted([
            os.path.join(args.log_dir, d)
            for d in os.listdir(args.log_dir)
            if os.path.isdir(os.path.join(args.log_dir, d))
        ])
        if not subdirs:
            print("No subdirectories found. Nothing to compare.")
            return
        labels = [os.path.basename(d) for d in subdirs]
        print(f"\nComparing {len(subdirs)} runs: {labels}")
        plot_comparison(subdirs, labels, out_dir, sw=sw)
        print(f"\nComparison saved -> {out_dir}/comparison.png")
        return

    print(f"\nLoading TensorBoard data from: {args.log_dir}")
    data = load_tb_scalars(args.log_dir)
    if not data:
        print("No data found. Make sure training has started.")
        return

    print(f"\nGenerating plots -> {out_dir}/\n")
    plot_overview(data, out_dir, sw=sw)
    plot_eval_success(data, out_dir, sw=1)
    plot_training_episode(data, out_dir, sw=sw)
    plot_manager(data, out_dir, sw=sw)
    plot_worker(data, out_dir, sw=sw)
    plot_buffers(data, out_dir, sw=5)
    plot_warmup(data, out_dir)

    print(f"\n{'-'*60}")
    print(f"  All plots saved to: {out_dir}/")
    print(f"{'-'*60}")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.png'):
            print(f"    {f}")
    print()


if __name__ == '__main__':
    main()
