"""
warmup.py — Stage A demo-driven behavioural cloning warmup.

The worker is now pretrained on teleoperated kitchen demonstrations instead of
random-walk probes. Task labels now come from replaying the demo actions in
FrankaKitchen-v1 and reading the env's actual completion events. Stage A then
does BC plus offline worker RL on the demo transitions.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from agent import SMGWAgent
from config import Config
from demo_dataset import build_or_load_demo_dataset


def run_stage_a_warmup(agent: SMGWAgent,
                       config: Config,
                       verbose: bool = True) -> Dict[str, float]:
    """
    Full Stage A:
      1. Load and cache offline kitchen demos.
      2. Infer per-transition task labels from demo completion timelines.
      3. Offline-render demo states to RGB and encode them with the frozen encoder.
      4. BC-train the worker on demo actions.
      5. Optionally CE-train the manager on the same timeline labels.

    Returns a dict of stage-A metrics.
    """
    cfg = config
    results: Dict[str, float] = {}

    if verbose:
        names = ", ".join(cfg.warmup.dataset_ids)
        print(f"  [Warmup] Building demo BC dataset from: {names}")

    ds, ds_stats = build_or_load_demo_dataset(agent, cfg, verbose=verbose)
    results.update(ds_stats)
    agent.demo_dataset = ds

    if ds.n_worker() == 0:
        raise RuntimeError(
            "Stage A found zero demo-labelled worker samples. "
            "Check dataset loading and task-label timeline extraction before continuing."
        )

    # Prime the worker normalizer from the demo proprio distribution.
    agent.worker_buf.observe_proprio_batch(ds.w_p)
    results["proprio_stats_count"] = float(agent.worker_buf.proprio_stats.n)

    if cfg.warmup.seed_worker_replay_with_demos:
        if verbose:
            print(f"  [Warmup] Seeding worker replay with {ds.n_worker():,} demo samples.")
        for i in range(ds.n_worker()):
            agent.worker_buf.add(
                z=ds.w_z[i],
                proprio=ds.w_p[i],
                task_target=ds.w_tt[i],
                task_cur=ds.w_tc[i],
                task_mask=ds.w_tm[i],
                task_id=ds.w_id[i],
                action_flat=ds.w_a[i],
                reward=ds.w_r[i],
                z_next=ds.w_z_next[i],
                proprio_next=ds.w_p_next[i],
                task_cur_next=ds.w_tc_next[i],
                done=ds.w_done[i],
            )
        results["seeded_worker_replay"] = float(ds.n_worker())
    else:
        results["seeded_worker_replay"] = 0.0

    if verbose:
        print(f"  [Warmup] Worker BC: {cfg.warmup.n_worker_sl_steps} steps "
              f"on {ds.n_worker():,} demo samples.")
    bc_losses = []
    for _ in range(cfg.warmup.n_worker_sl_steps):
        batch = ds.sample_worker_batch(
            cfg.warmup.sl_batch_size,
            proprio_normalizer=agent.worker_buf.normalize_proprio,
            balance_by_task=cfg.warmup.balance_worker_task_sampling,
        )
        bc_losses.append(agent.worker_warmup_step(batch))
    results["worker_bc_loss_final"] = float(np.mean(bc_losses[-100:])) \
        if len(bc_losses) >= 100 else float(np.mean(bc_losses))
    results["worker_bc_loss_best"] = float(np.min(bc_losses))
    if verbose:
        print(f"  [Warmup] Worker BC final loss: {results['worker_bc_loss_final']:.4f}  "
              f"(best {results['worker_bc_loss_best']:.4f})")

    if cfg.warmup.n_worker_iql_steps > 0:
        if verbose:
            print(f"  [Warmup] Worker IQL: {cfg.warmup.n_worker_iql_steps} steps "
                  f"on the replay-labelled demo transitions.")
        iql_metrics = []
        for _ in range(cfg.warmup.n_worker_iql_steps):
            batch = ds.sample_worker_batch(
                cfg.warmup.sl_batch_size,
                proprio_normalizer=agent.worker_buf.normalize_proprio,
                balance_by_task=cfg.warmup.balance_worker_task_sampling,
            )
            iql_metrics.append(agent.worker_iql_step(batch))
        for key in iql_metrics[-1].keys():
            tail = [m[key] for m in iql_metrics[-100:]] if len(iql_metrics) >= 100 else [m[key] for m in iql_metrics]
            results[f"{key}_final"] = float(np.mean(tail))
        if verbose:
            print(f"  [Warmup] Worker IQL final losses: "
                  f"value={results.get('worker_iql_value_loss_final', float('nan')):.4f}  "
                  f"critic={results.get('worker_iql_critic_loss_final', float('nan')):.4f}  "
                  f"actor={results.get('worker_iql_actor_loss_final', float('nan')):.4f}")

    if ds.n_manager() > 0 and cfg.warmup.n_manager_sl_steps > 0:
        if verbose:
            print(f"  [Warmup] Manager CE: {cfg.warmup.n_manager_sl_steps} steps "
                  f"on {ds.n_manager():,} demo-labelled states.")
        ce_losses = []
        for _ in range(cfg.warmup.n_manager_sl_steps):
            batch = ds.sample_manager_batch(cfg.warmup.sl_batch_size)
            ce_losses.append(agent.manager_warmup_step(batch))
        results["manager_ce_loss_final"] = float(np.mean(ce_losses[-100:])) \
            if len(ce_losses) >= 100 else float(np.mean(ce_losses))
        if verbose:
            print(f"  [Warmup] Manager CE final loss: {results['manager_ce_loss_final']:.4f}")
    elif verbose:
        print("  [Warmup] No demo-derived manager labels found; skipping manager CE.")

    if verbose:
        print("  [Warmup] Worker labels by task:")
        for task_name in agent.tasks:
            safe_task = task_name.lower().replace(" ", "_")
            print(f"    {task_name:<14} {int(results.get(f'worker_labels/{safe_task}', 0)):,}")
        replay_mean = results.get("replay_mean_state_l2", float('nan'))
        replay_max = results.get("replay_max_state_l2", float('nan'))
        if not np.isnan(replay_mean):
            print(f"  [Warmup] Replay fidelity: mean_state_l2={replay_mean:.6f}  "
                  f"max_state_l2={replay_max:.6f}")

    return results
