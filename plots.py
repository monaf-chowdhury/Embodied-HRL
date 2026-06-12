"""
plots.py — Diagnostic plots for lean FrankaKitchen skill-learning runs.

Reads TensorBoard event files (with train_log text fallback) and generates
PNG plots covering offline BC/IQL+LQL training health, prefix-state model
selection, and online AWAC fine-tuning.

Usage:
    python plots.py --log_dir logs/lean_skills/
    python plots.py --log_dir logs/lean_skills/ --out_dir plots/run1/ --smooth 20
    python plots.py --log_dir logs/ --compare         # overlay multiple runs

Output files:
    00_overview_dashboard.png     — 6-panel summary; look here first
    01_eval_success.png           — chain eval success rates over time
    02_training_episode.png       — online per-episode reward / options / tasks
    03_lql_diagnostics.png        — LQL lower-bound penalty health per skill
    04_prefix_validation.png      — IQL prefix-state validation per skill
    07_skill_losses.png           — per-skill offline optimisation losses
    08_skill_eval.png             — final per-skill eval bar charts
    09_online_awac.png            — online AWAC fine-tuning diagnostics
    comparison.png                — multi-run overlay (--compare mode)
"""
import os
import argparse
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None


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
    if EventAccumulator is None:
        print("  WARNING: tensorboard not installed; using train_log text fallback only.")
        return {}
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


def _append_scalar(series: dict, tag: str, step: int, value: float):
    series.setdefault(tag, []).append((int(step), float(value)))


def _finalize_series(series: dict) -> dict:
    data = {}
    for tag, points in series.items():
        points = sorted(points, key=lambda x: x[0])
        data[tag] = (
            np.asarray([p[0] for p in points], dtype=np.int64),
            np.asarray([p[1] for p in points], dtype=np.float64),
        )
    return data


def load_text_log_scalars(log_dir: str) -> dict:
    paths = sorted(glob.glob(os.path.join(log_dir, "train_log_*.txt")))
    if not paths:
        return {}
    series = {}
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        online_matches = list(re.finditer(
            r"ONLINE EVAL step=([\d,]+): full=\s*([\d.]+)%\s+any=\s*([\d.]+)%\s+"
            r"tasks=([\d.]+)(?:/\d+)?(?:\s+chosen=\s*([\d.]+)%)?",
            text,
        ))
        for match in online_matches:
            step = int(match.group(1).replace(",", ""))
            _append_scalar(series, "online_eval/eval/full_task_success_rate", step, float(match.group(2)) / 100.0)
            _append_scalar(series, "online_eval/eval/any_task_success_rate", step, float(match.group(3)) / 100.0)
            _append_scalar(series, "online_eval/eval/mean_tasks_completed", step, float(match.group(4)))
            if match.group(5) is not None:
                _append_scalar(
                    series,
                    "online_eval/eval/mean_chosen_task_success",
                    step,
                    float(match.group(5)) / 100.0,
                )

        offline_block = _extract_block(text, "SCRIPTED CHAIN EVAL", "STAGE B")
        if offline_block:
            _append_eval_block(series, "eval", 0, offline_block)

        final_block = _extract_block(text, "FINAL SCRIPTED CHAIN EVAL AFTER ONLINE", "RUN COMPLETE")
        if final_block:
            step = int(online_matches[-1].group(1).replace(",", "")) if online_matches else 0
            _append_eval_block(series, "final_after_online/eval", step, final_block)

    data = _finalize_series(series)
    if data:
        print(f"  Parsed {len(data)} scalar tags from train_log text fallback")
    return data


def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        end = len(text)
    return text[start:end]


def _append_eval_block(series: dict, prefix: str, step: int, block: str):
    patterns = {
        "full_task_success_rate": r"Full-task success\s*:\s*([\d.]+)%",
        "any_task_success_rate": r"Any-task success\s*:\s*([\d.]+)%",
        "mean_tasks_completed": r"Mean tasks done\s*:\s*([\d.]+)",
        "mean_chosen_task_success": r"Chosen-task SR\s*:\s*([\d.]+)%",
        "final_env_done_failure_rate": r"Env-horizon fail\s*:\s*([\d.]+)%",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, block)
        if not match:
            continue
        value = float(match.group(1))
        if name.endswith("_rate") or name == "mean_chosen_task_success":
            value /= 100.0
        _append_scalar(series, f"{prefix}/{name}", step, value)


def load_run_scalars(log_dir: str) -> dict:
    data = load_tb_scalars(log_dir)
    text_data = load_text_log_scalars(log_dir)
    for tag, text_series in text_data.items():
        if tag not in data or len(data[tag][0]) <= 1:
            data[tag] = text_series
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


def _plot_eval_metric(ax, data: dict, base_tag: str,
                      online_tag: str = None, final_tag: str = None,
                      title: str = None, ylabel: str = None,
                      color: str = 'steelblue', pct: bool = False,
                      smooth_window: int = 1, ymin=None, ymax=None):
    plotted = False
    if base_tag in data:
        steps, values = data[base_tag]
        y = values * 100.0 if pct else values
        ax.plot(steps, y, marker='o', linestyle='--', color=COLORS['gray'],
                linewidth=1.2, alpha=0.8, label='offline / baseline')
        plotted = True
    if online_tag and online_tag in data:
        steps, values = data[online_tag]
        y = values * 100.0 if pct else values
        ys = smooth(y, smooth_window)
        ax.plot(steps, y, alpha=STYLE['raw_alpha'], color=color, linewidth=STYLE['raw_lw'])
        ax.plot(steps, ys, marker='o', alpha=STYLE['smooth_alpha'],
                color=color, linewidth=STYLE['smooth_lw'], label='online eval')
        plotted = True
    if final_tag and final_tag in data:
        steps, values = data[final_tag]
        y = values * 100.0 if pct else values
        ax.scatter(steps, y, color=COLORS['red'], s=36, zorder=4, label='final')
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, f'No data\n({base_tag})',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=STYLE['label_fs'])
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_title(title or base_tag, fontsize=STYLE['title_fs'], fontweight='bold')
    ax.set_xlabel('Environment Steps', fontsize=STYLE['label_fs'])
    ax.set_ylabel(ylabel or ('%' if pct else ''), fontsize=STYLE['label_fs'])
    if plotted:
        ax.legend(fontsize=STYLE['tick_fs'])
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
    print(f"  Saved  ->  {path}")


def _skill_names(data: dict) -> list[str]:
    names = set()
    for tag in data:
        parts = tag.split('/')
        if len(parts) >= 3 and parts[0] == 'skill':
            names.add(parts[1])
            continue
        if len(parts) >= 4 and parts[0] == 'online' and parts[1] == 'skill':
            names.add(parts[2])
            continue
        if len(parts) >= 2 and parts[0] == 'single_task':
            metric = parts[1]
            for suffix in ('_success_rate', '_mean_options', '_mean_env_reward', '_mean_final_error'):
                if metric.endswith(suffix):
                    names.add(metric[:-len(suffix)])
                    break
    return sorted(names)


def _plot_skill_family(ax, data: dict, metric: str, title: str, ylabel: str,
                       sw: int = 15, pct: bool = False, ymin=None, ymax=None):
    skills = _skill_names(data)
    plotted = False
    palette = list(COLORS.values())
    for i, skill in enumerate(skills):
        tag = f'skill/{skill}/{metric}'
        if tag not in data:
            continue
        steps, values = data[tag]
        y = values * 100.0 if pct else values
        ys = smooth(y, sw)
        color = palette[i % len(palette)]
        ax.plot(steps, y, alpha=STYLE['raw_alpha'], color=color, linewidth=STYLE['raw_lw'])
        ax.plot(steps, ys, alpha=STYLE['smooth_alpha'], color=color,
                linewidth=STYLE['smooth_lw'], label=skill.replace('_', ' '))
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, f'No skill data\n({metric})',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=STYLE['label_fs'])
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=STYLE['title_fs'], fontweight='bold')
    ax.set_xlabel('Optimizer Steps', fontsize=STYLE['label_fs'])
    ax.set_ylabel(ylabel, fontsize=STYLE['label_fs'])
    if plotted:
        ax.legend(fontsize=STYLE['tick_fs'])
    _style_ax(ax)


def _plot_skill_family_first_available(ax, data: dict, metrics: list[str],
                                       title: str, ylabel: str, sw: int = 15):
    skills = _skill_names(data)
    for metric in metrics:
        if any(f'skill/{skill}/{metric}' in data for skill in skills):
            return _plot_skill_family(ax, data, metric, title, ylabel, sw=sw)
    return _plot_skill_family(ax, data, metrics[0], title, ylabel, sw=sw)


def _plot_online_skill_family(ax, data: dict, metric: str, title: str, ylabel: str,
                              sw: int = 15, ymin=None, ymax=None):
    skills = _skill_names(data)
    plotted = False
    palette = list(COLORS.values())
    for i, skill in enumerate(skills):
        tag = f'online/skill/{skill}/{metric}'
        if tag not in data:
            continue
        steps, values = data[tag]
        ys = smooth(values, sw)
        color = palette[i % len(palette)]
        ax.plot(steps, values, alpha=STYLE['raw_alpha'], color=color, linewidth=STYLE['raw_lw'])
        ax.plot(steps, ys, alpha=STYLE['smooth_alpha'], color=color,
                linewidth=STYLE['smooth_lw'], label=skill.replace('_', ' '))
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, f'No online data\n({metric})',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=STYLE['label_fs'])
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=STYLE['title_fs'], fontweight='bold')
    ax.set_xlabel('Environment Steps', fontsize=STYLE['label_fs'])
    ax.set_ylabel(ylabel, fontsize=STYLE['label_fs'])
    if plotted:
        ax.legend(fontsize=STYLE['tick_fs'])
    _style_ax(ax)


# =============================================================================
# Plot 00 — Overview Dashboard
# =============================================================================

def plot_overview(data: dict, out_dir: str, sw: int = 15):
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor('white')
    fig.suptitle('Lean Skill Learning — Training Overview Dashboard',
                 fontsize=STYLE['suptitle_fs'] + 2, fontweight='bold', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    _plot_eval_metric(
        axes[0], data,
        'eval/full_task_success_rate',
        online_tag='online_eval/eval/full_task_success_rate',
        final_tag='final_after_online/eval/full_task_success_rate',
        title='[1]  Eval Full-Task Success  (HEADLINE)',
        ylabel='Success Rate (%)', color=COLORS['green'],
        pct=True, smooth_window=1, ymin=0, ymax=105)

    _plot_eval_metric(
        axes[1], data,
        'eval/any_task_success_rate',
        online_tag='online_eval/eval/any_task_success_rate',
        final_tag='final_after_online/eval/any_task_success_rate',
        title='[2]  Eval Any-Task Success',
        ylabel='Success Rate (%)', color=COLORS['blue'],
        pct=True, smooth_window=1, ymin=0, ymax=105)

    _plot_eval_metric(
        axes[2], data,
        'eval/mean_tasks_completed',
        online_tag='online_eval/eval/mean_tasks_completed',
        final_tag='final_after_online/eval/mean_tasks_completed',
        title='[3]  Eval Mean Tasks Completed',
        ylabel='Tasks', color=COLORS['gold'],
        smooth_window=1, ymin=0, ymax=4.1)

    _plot_skill_family(axes[3], data, 'iql_prefix_success_rate',
                       title='[4]  IQL Prefix-Val Success (model selection)',
                       ylabel='%', sw=1, pct=True, ymin=0, ymax=105)

    _plot_skill_family(axes[4], data, 'iql_critic_loss',
                       title='[5]  IQL Critic TD Loss', ylabel='MSE', sw=sw)

    _plot_skill_family(axes[5], data, 'iql_lb_active_frac',
                       title='[6]  LQL Active Pair Fraction',
                       ylabel='fraction', sw=sw, ymin=0, ymax=1.05)

    fig.tight_layout()
    _save(fig, out_dir, '00_overview_dashboard.png')


# =============================================================================
# Plot 01 — Eval Success
# =============================================================================

def plot_eval_success(data: dict, out_dir: str, sw: int = 1):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Evaluation Success Rates',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot_eval_metric(
        axes[0], data,
        'eval/full_task_success_rate',
        online_tag='online_eval/eval/full_task_success_rate',
        final_tag='final_after_online/eval/full_task_success_rate',
        title='Full-Task Success (all configured tasks)',
        ylabel='%', color=COLORS['green'],
        pct=True, smooth_window=sw, ymin=0, ymax=105)
    _plot_eval_metric(
        axes[1], data,
        'eval/any_task_success_rate',
        online_tag='online_eval/eval/any_task_success_rate',
        final_tag='final_after_online/eval/any_task_success_rate',
        title='Any-Task Success (>=1)',
        ylabel='%', color=COLORS['blue'],
        pct=True, smooth_window=sw, ymin=0, ymax=105)
    _plot_eval_metric(
        axes[2], data,
        'eval/mean_chosen_task_success',
        online_tag='online_eval/eval/mean_chosen_task_success',
        final_tag='final_after_online/eval/mean_chosen_task_success',
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
# Plot 03 — LQL lower-bound penalty diagnostics
# =============================================================================

def plot_lql_diagnostics(data: dict, out_dir: str, sw: int = 15):
    """Health of the LQL lower-bound critic penalty.

    What to look for:
      * active_frac should settle in a moderate band (~5-40%). Pinned at 0%
        means chains never fire (sampler/data bug); pinned near 100% means the
        critic is persistently below the demo returns (slow propagation or
        lambda too small).
      * lb_loss should decay as Q absorbs the bounds.
      * target_q_mean / q_mean climbing without bound => Q inflation; lower
        lambda_lb.
    """
    has_lql = any('/iql_lb_' in k for k in data)
    if not has_lql:
        return
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle('LQL Lower-Bound Penalty Diagnostics',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot_skill_family(axes[0], data, 'iql_lb_loss',
                       title='LB Penalty (mean hinge^2)', ylabel='Loss', sw=sw)
    _plot_skill_family(axes[1], data, 'iql_lb_active_frac',
                       title='Active Pair Fraction (hinge > 0)',
                       ylabel='fraction', sw=sw, ymin=0, ymax=1.05)
    _plot_skill_family(axes[2], data, 'iql_lb_hinge_mean',
                       title='Mean Hinge Magnitude (active pairs)',
                       ylabel='Q underestimate', sw=sw)
    _plot_skill_family(axes[3], data, 'iql_lb_chain_len_mean',
                       title='Mean Sampled Chain Length',
                       ylabel='chunk transitions', sw=sw)
    _plot_skill_family(axes[4], data, 'iql_target_q_mean',
                       title='TD Target Q Mean (inflation watch)',
                       ylabel='Q', sw=sw)
    _plot_skill_family(axes[5], data, 'iql_q_mean',
                       title='Q(s,a) Mean on Demo Batch',
                       ylabel='Q', sw=sw)

    fig.tight_layout()
    _save(fig, out_dir, '03_lql_diagnostics.png')


# =============================================================================
# Plot 04 — IQL prefix-state validation (model selection signal)
# =============================================================================

def plot_prefix_validation(data: dict, out_dir: str):
    skills = _skill_names(data)
    has_prefix = any(f'skill/{s}/iql_prefix_success_rate' in data for s in skills)
    if not has_prefix:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('IQL Prefix-State Validation (checkpoint selection)',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot_skill_family(axes[0], data, 'iql_prefix_success_rate',
                       title='Prefix-Val Success Rate', ylabel='%',
                       sw=1, pct=True, ymin=0, ymax=105)
    _plot_skill_family(axes[1], data, 'iql_prefix_mean_final_error',
                       title='Prefix-Val Final Task Error', ylabel='error', sw=1)
    _plot_skill_family(axes[2], data, 'iql_prefix_mean_options',
                       title='Prefix-Val Options Used', ylabel='options', sw=1)

    fig.tight_layout()
    _save(fig, out_dir, '04_prefix_validation.png')


# =============================================================================
# Plot 07 — Skill optimisation curves
# =============================================================================

def plot_skill_losses(data: dict, out_dir: str, sw: int = 15):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle('Per-Skill Offline Optimisation',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot_skill_family(axes[0], data, 'bc_loss',
                       title='BC Action MSE', ylabel='MSE', sw=sw)
    _plot_skill_family_first_available(
        axes[1], data,
        ['iql_value_loss', 'awr_value_loss', 'bet_cls_loss'],
        title='Value / Classification Loss', ylabel='Loss', sw=sw)
    _plot_skill_family_first_available(
        axes[2], data,
        ['iql_critic_loss', 'td3bc_critic_loss', 'awr_critic_loss', 'bet_residual_loss'],
        title='Critic / Residual Loss', ylabel='Loss', sw=sw)
    _plot_skill_family_first_available(
        axes[3], data,
        ['iql_actor_loss', 'td3bc_actor_loss', 'awr_actor_loss', 'bet_loss'],
        title='Actor / Total Loss', ylabel='Loss', sw=sw)
    _plot_skill_family_first_available(
        axes[4], data,
        ['iql_adv_mean', 'td3bc_bc_loss', 'awr_weight_mean'],
        title='Advantage / BC / Weight', ylabel='Value', sw=sw)
    _plot_skill_family_first_available(
        axes[5], data,
        ['iql_weight_mean', 'td3bc_lambda'],
        title='Policy Weight / Lambda', ylabel='Weight', sw=sw)

    fig.tight_layout()
    _save(fig, out_dir, '07_skill_losses.png')


# =============================================================================
# Plot 08 — Skill eval summary
# =============================================================================

def plot_skill_eval(data: dict, out_dir: str):
    skills = _skill_names(data)
    if not skills:
        return
    labels = [s.replace('_', ' ') for s in skills]

    def scalar(tag: str, default=np.nan):
        if tag not in data:
            return default
        return float(data[tag][1][-1])

    success = np.array([scalar(f'single_task/{s}_success_rate') for s in skills])
    options = np.array([scalar(f'single_task/{s}_mean_options') for s in skills])
    errors = np.array([scalar(f'single_task/{s}_mean_final_error') for s in skills])
    rewards = np.array([scalar(f'single_task/{s}_mean_env_reward') for s in skills])

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('Per-Skill Evaluation Summary',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')
    x = np.arange(len(skills))

    axes[0].bar(x, success * 100.0, color=COLORS['green'])
    axes[0].set_title('Single-Task Success', fontsize=STYLE['title_fs'], fontweight='bold')
    axes[0].set_ylabel('%', fontsize=STYLE['label_fs'])
    axes[0].set_ylim(0, 105)

    axes[1].bar(x, options, color=COLORS['purple'])
    axes[1].set_title('Mean Options', fontsize=STYLE['title_fs'], fontweight='bold')
    axes[1].set_ylabel('Options', fontsize=STYLE['label_fs'])

    axes[2].bar(x, errors, color=COLORS['red'])
    axes[2].set_title('Final Task Error', fontsize=STYLE['title_fs'], fontweight='bold')
    axes[2].set_ylabel('Error', fontsize=STYLE['label_fs'])

    axes[3].bar(x, rewards, color=COLORS['blue'])
    axes[3].set_title('Env Reward', fontsize=STYLE['title_fs'], fontweight='bold')
    axes[3].set_ylabel('Reward', fontsize=STYLE['label_fs'])

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right')
        _style_ax(ax)

    fig.tight_layout()
    _save(fig, out_dir, '08_skill_eval.png')


# =============================================================================
# Plot 09 — Online AWAC diagnostics
# =============================================================================

def plot_online_awac(data: dict, out_dir: str, sw: int = 15):
    has_online = any(k.startswith('online/') or k.startswith('online_eval/') for k in data)
    if not has_online:
        return
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    fig.suptitle('Online Chain-Context AWAC Fine-Tuning',
                 fontsize=STYLE['suptitle_fs'], fontweight='bold')

    _plot(axes[0], data, 'online_eval/eval/full_task_success_rate',
          title='Online Eval Full-Task Success',
          ylabel='%', color=COLORS['green'], pct=True, smooth_window=1, ymin=0, ymax=105)
    _plot(axes[1], data, 'online_eval/eval/mean_tasks_completed',
          title='Online Eval Mean Tasks Completed',
          ylabel='Tasks', color=COLORS['gold'], smooth_window=1, ymin=0, ymax=4.1)
    if 'online/demo_fraction' in data or 'online/bc_anchor_weight' in data:
        for tag, label, color in [
            ('online/demo_fraction', 'demo fraction', COLORS['gray']),
            ('online/bc_anchor_weight', 'BC anchor', COLORS['red']),
        ]:
            if tag not in data:
                continue
            steps, values = data[tag]
            axes[2].plot(steps, values, alpha=STYLE['raw_alpha'], color=color, linewidth=STYLE['raw_lw'])
            axes[2].plot(steps, smooth(values, sw), alpha=STYLE['smooth_alpha'],
                         color=color, linewidth=STYLE['smooth_lw'], label=label)
        axes[2].set_title('Demo Fraction / BC Anchor',
                          fontsize=STYLE['title_fs'], fontweight='bold')
        axes[2].set_xlabel('Environment Steps', fontsize=STYLE['label_fs'])
        axes[2].set_ylabel('value', fontsize=STYLE['label_fs'])
        axes[2].legend(fontsize=STYLE['tick_fs'])
        _style_ax(axes[2])
    else:
        _plot(axes[2], data, 'online/demo_fraction',
              title='Demo Replay Fraction',
              ylabel='fraction', color=COLORS['gray'], smooth_window=sw, ymin=0, ymax=1.05)
    _plot_online_skill_family(axes[3], data, 'online_critic_loss',
                              title='Per-Skill Online Critic Loss', ylabel='MSE', sw=sw)
    _plot_online_skill_family(axes[4], data, 'online_bc_anchor_loss',
                              title='Per-Skill Demo BC Anchor Loss', ylabel='MSE', sw=sw)
    _plot_online_skill_family(axes[5], data, 'online_weight_mean',
                              title='Per-Skill AWAC Weight Mean', ylabel='weight', sw=sw)
    for tag, label, color in [
        ('online/actor_source/demo_only_fallback_total', 'demo-only fallback', COLORS['orange']),
        ('online/actor_source/success_online_total', 'success-online actor', COLORS['green']),
        ('online/actor_source/all_attempt_total', 'all-attempt actor', COLORS['purple']),
    ]:
        if tag not in data:
            continue
        steps, values = data[tag]
        axes[6].plot(steps, values, alpha=STYLE['raw_alpha'], color=color, linewidth=STYLE['raw_lw'])
        axes[6].plot(steps, smooth(values, sw), alpha=STYLE['smooth_alpha'],
                     color=color, linewidth=STYLE['smooth_lw'], label=label)
    axes[6].set_title('Actor Update Source Counts',
                      fontsize=STYLE['title_fs'], fontweight='bold')
    axes[6].set_xlabel('Environment Steps', fontsize=STYLE['label_fs'])
    axes[6].set_ylabel('updates', fontsize=STYLE['label_fs'])
    if axes[6].has_data():
        axes[6].legend(fontsize=STYLE['tick_fs'])
    else:
        axes[6].text(0.5, 0.5, 'No actor-source data',
                     ha='center', va='center', transform=axes[6].transAxes,
                     color='gray', fontsize=STYLE['label_fs'])
    _style_ax(axes[6])
    _plot_first_available(
        axes[7], data,
        ['online/replay/light_switch_actor_quality_size',
         'online/replay/light_switch_quality_size',
         'online/replay_total'],
        title='Actor-Quality Replay Size',
        ylabel='samples', color=COLORS['blue'], smooth_window=sw)

    fig.tight_layout()
    _save(fig, out_dir, '09_online_awac.png')


# =============================================================================
# Multi-run comparison
# =============================================================================

def plot_comparison(run_dirs, run_labels, out_dir: str, sw: int = 15):
    COMPARE_TAGS = [
        ('eval/full_task_success_rate',             'Eval Full-Task Success (%)',   True),
        ('eval/any_task_success_rate',              'Eval Any-Task Success (%)',    True),
        ('eval/mean_tasks_completed',               'Eval Mean Tasks Completed',    False),
        ('online_eval/eval/full_task_success_rate', 'Online Full-Task Success (%)', True),
        ('skill/light_switch/iql_prefix_success_rate', 'Light-Switch Prefix-Val (%)', True),
        ('train/ep_env_reward',                     'Train Episode Env Reward',     False),
    ]

    palette = list(COLORS.values())
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle('Multi-Run Comparison',
                 fontsize=STYLE['suptitle_fs'] + 1, fontweight='bold')

    for run_dir, label, color in zip(run_dirs, run_labels, palette):
        print(f"  Loading run: {label}")
        d = load_run_scalars(run_dir)
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
    data = load_run_scalars(args.log_dir)
    if not data:
        print("No data found. Make sure training has started.")
        return

    print(f"\nGenerating plots -> {out_dir}/\n")
    plot_overview(data, out_dir, sw=sw)
    plot_eval_success(data, out_dir, sw=1)
    plot_training_episode(data, out_dir, sw=sw)
    plot_lql_diagnostics(data, out_dir, sw=sw)
    plot_prefix_validation(data, out_dir)
    plot_skill_losses(data, out_dir, sw=sw)
    plot_skill_eval(data, out_dir)
    plot_online_awac(data, out_dir, sw=sw)

    print(f"\n{'-'*60}")
    print(f"  All plots saved to: {out_dir}/")
    print(f"{'-'*60}")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.png'):
            print(f"    {f}")
    print()


if __name__ == '__main__':
    main()
