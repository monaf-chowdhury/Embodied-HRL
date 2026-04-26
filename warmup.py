"""
warmup.py — Stage A: Demo BC pretraining for SMGW.

This replaces the previous random-walk warmup. The new version:
  1. Loads the Minari kitchen demos via demo_loader.
  2. BC-trains the worker actor on (state, image, task_id) -> demo_action
     with MSE over the pre-tanh action mean. Task labels come from the
     demo's completion timeline (see demo_loader.py).
  3. BC-trains the manager on (state, image, completion_mask) -> task_id
     where the label is the task about to complete in the demo.
  4. Populates the worker replay buffer with the demo transitions so Stage B
     starts with a non-empty buffer of purposeful data.

Why this is different (and better) than the previous warmup:
  - Old warmup used random actions as BC targets. A small actor trained to
    regress random actions has learned nothing useful.
  - New warmup uses real teleoperated actions that actually cause the
    objects to move toward the goals. The actor learns a real prior.
  - The manager label used to be "the task that happened to complete
    during random rollout" - almost always no label. Now it's "the task
    that completes next in a demo" - ~4 labels per episode, hundreds of
    episodes.

Verifying BC worked:
  After Stage A completes, call run_single_task_eval() (in
  single_task_eval.py) to measure how often the deterministic BC worker
  completes each task when commanded. Target: 40-70% on the easier tasks
  (microwave, slide cabinet).
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

from config import Config
from utils import TaskSpec
from agent import SMGWAgent
from demo_loader import load_demo_bc_dataset, DemoBCDataset, safe_atanh


# =============================================================================
# BC training steps
# =============================================================================

def _worker_bc_step(agent: SMGWAgent,
                    batch: Dict[str, np.ndarray],
                    lr_override: float = None) -> float:
    """
    One BC gradient step for the worker actor.

    We regress the actor's pre-tanh raw mean onto atanh(demo_action). This
    is the mathematically correct BC target for a tanh-squashed Gaussian:
    the MODE of the policy is tanh(mean), so pushing mean toward
    atanh(demo_action) pushes the mode toward demo_action.
    """
    device = agent.device
    z = torch.from_numpy(batch['z']).to(device)
    p = torch.from_numpy(batch['proprio']).to(device)
    tt = torch.from_numpy(batch['task_target']).to(device)
    tc = torch.from_numpy(batch['task_cur']).to(device)
    tm = torch.from_numpy(batch['task_mask']).to(device)
    tid = torch.from_numpy(batch['task_id']).long().to(device)
    a_demo = torch.from_numpy(batch['action']).to(device)
    te = agent.spec.text_embeddings[tid]

    h = agent.worker_actor.trunk(z, p, tt, tc, tm, te)
    mean_raw = agent.worker_actor.mean_head(h)

    # Chunked worker: tile the single-step demo action across chunk slots so
    # the BC loss at least shapes the first slot. Online Stage B will refine
    # the later chunk slots from real rollouts.
    if agent.worker_actor.H > 1:
        a_demo_flat = a_demo.repeat(1, agent.worker_actor.H)
    else:
        a_demo_flat = a_demo

    target = torch.from_numpy(
        safe_atanh(a_demo_flat.detach().cpu().numpy())
    ).to(device)

    loss = F.mse_loss(mean_raw, target)

    agent.worker_actor_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.worker_actor.parameters(), 1.0)
    if lr_override is not None:
        for g in agent.worker_actor_opt.param_groups:
            g['_saved_lr'] = g['lr']
            g['lr'] = lr_override
    agent.worker_actor_opt.step()
    if lr_override is not None:
        for g in agent.worker_actor_opt.param_groups:
            g['lr'] = g.pop('_saved_lr')
    return float(loss.item())


def _manager_bc_step(agent: SMGWAgent,
                     batch: Dict[str, np.ndarray],
                     lr_override: float = None) -> float:
    """One BC gradient step for the manager (CE over the 4 tasks)."""
    device = agent.device
    z = torch.from_numpy(batch['m_z']).to(device)
    p = torch.from_numpy(batch['m_proprio']).to(device)
    ts = torch.from_numpy(batch['m_task_state']).to(device)
    c = torch.from_numpy(batch['m_completion']).to(device)
    y = torch.from_numpy(batch['m_label']).long().to(device)

    q = agent.manager(z, p, ts, c, agent.spec.text_embeddings)
    from networks import SemanticManager
    q = q.masked_fill(c > 0.5, SemanticManager.MASK_FILL)
    logp = F.log_softmax(q, dim=-1)
    loss = F.nll_loss(logp, y)

    agent.manager_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.manager.parameters(), 1.0)
    if lr_override is not None:
        for g in agent.manager_opt.param_groups:
            g['_saved_lr'] = g['lr']
            g['lr'] = lr_override
    agent.manager_opt.step()
    if lr_override is not None:
        for g in agent.manager_opt.param_groups:
            g['lr'] = g.pop('_saved_lr')
    return float(loss.item())


# =============================================================================
# Populate worker buffer with demo transitions
# =============================================================================

def _populate_worker_buffer_from_demos(agent: SMGWAgent,
                                       ds: DemoBCDataset) -> int:
    """Push every demo transition into the worker replay buffer."""
    H = agent.H_chunk
    n = ds.n_worker
    if n == 0:
        return 0

    if H > 1:
        actions_flat = np.tile(ds.action, (1, H))
    else:
        actions_flat = ds.action

    for i in range(n):
        agent.worker_buf.add(
            z=ds.z[i],
            proprio=ds.proprio[i],
            task_target=ds.task_target[i],
            task_cur=ds.task_cur[i],
            task_mask=ds.task_mask[i],
            task_id=int(ds.task_id[i]),
            action_flat=actions_flat[i],
            reward=float(ds.reward[i]),
            z_next=ds.z_next[i],
            proprio_next=ds.proprio_next[i],
            task_cur_next=ds.task_cur[i],
            done=0.0,
        )

    return n


# =============================================================================
# Batch samplers with proper proprio normalisation
# =============================================================================

def _make_worker_sampler(ds: DemoBCDataset, agent: SMGWAgent):
    n = ds.n_worker
    def sample(bs: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, n, size=bs)
        p_raw = ds.proprio[idx]
        p_norm = np.stack([
            agent.worker_buf.normalize_proprio(p_raw[b]) for b in range(bs)
        ]).astype(np.float32)
        return {
            'z':           ds.z[idx],
            'proprio':     p_norm,
            'task_target': ds.task_target[idx],
            'task_cur':    ds.task_cur[idx],
            'task_mask':   ds.task_mask[idx],
            'task_id':     ds.task_id[idx],
            'action':      ds.action[idx],
        }
    return sample


def _make_manager_sampler(ds: DemoBCDataset):
    n = ds.n_manager
    def sample(bs: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, n, size=bs)
        return {
            'm_z':          ds.m_z[idx],
            'm_proprio':    ds.m_proprio[idx],
            'm_task_state': ds.m_task_state[idx],
            'm_completion': ds.m_completion[idx],
            'm_label':      ds.m_label[idx],
        }
    return sample


# =============================================================================
# Public entry point
# =============================================================================

def run_stage_a_warmup(agent: SMGWAgent,
                       config: Config,
                       verbose: bool = True) -> Dict[str, float]:
    """
    Stage A: demo BC pretraining.

    Outline:
      1. Load / build the demo dataset.
      2. Push demo transitions into the worker replay buffer.
      3. BC-train the worker actor.
      4. BC-train the manager.
      5. Optionally save a BC checkpoint.
    """
    cfg = config
    results: Dict[str, float] = {}

    # 1. Build / load dataset
    ds = load_demo_bc_dataset(cfg, agent.spec, agent.encoder,
                              tasks_to_complete=cfg.training.tasks_to_complete)
    results['demo_worker_samples'] = ds.n_worker
    results['demo_manager_samples'] = ds.n_manager

    # 2. Populate worker buffer (this builds proprio stats)
    if verbose:
        print(f"\n  [Warmup] Populating worker buffer with "
              f"{ds.n_worker:,} demo transitions...")
    _populate_worker_buffer_from_demos(agent, ds)
    if verbose:
        print(f"  [Warmup] Worker buffer now holds {len(agent.worker_buf):,} transitions.")
        print(f"  [Warmup] Proprio running-stats N = {agent.worker_buf.proprio_stats.n:,}")

    # 3. Worker BC
    if ds.n_worker > 0:
        if verbose:
            print(f"\n  [Warmup] Worker BC: {cfg.warmup.n_worker_bc_steps:,} steps, "
                  f"batch {cfg.warmup.bc_batch_size}, LR {cfg.warmup.bc_lr}")
        sampler = _make_worker_sampler(ds, agent)
        losses = []
        for step in range(cfg.warmup.n_worker_bc_steps):
            batch = sampler(cfg.warmup.bc_batch_size)
            losses.append(_worker_bc_step(agent, batch,
                                          lr_override=cfg.warmup.bc_lr))
            if verbose and (step + 1) % 1000 == 0:
                recent = float(np.mean(losses[-500:]))
                print(f"    step {step+1:>6,}/{cfg.warmup.n_worker_bc_steps:,}  "
                      f"BC loss (last 500) = {recent:.5f}")
        results['worker_bc_loss_final'] = float(np.mean(losses[-500:]))
        if verbose:
            print(f"  [Warmup] Worker BC final loss: "
                  f"{results['worker_bc_loss_final']:.5f}")
    else:
        results['worker_bc_loss_final'] = float('nan')

    # 4. Manager BC
    if ds.n_manager > 0:
        if verbose:
            print(f"\n  [Warmup] Manager BC: {cfg.warmup.n_manager_bc_steps:,} steps")
        sampler = _make_manager_sampler(ds)
        losses = []
        for step in range(cfg.warmup.n_manager_bc_steps):
            batch = sampler(cfg.warmup.bc_batch_size)
            losses.append(_manager_bc_step(agent, batch,
                                           lr_override=cfg.warmup.bc_lr))
            if verbose and (step + 1) % 500 == 0:
                recent = float(np.mean(losses[-200:]))
                print(f"    step {step+1:>6,}/{cfg.warmup.n_manager_bc_steps:,}  "
                      f"CE loss (last 200) = {recent:.5f}")
        results['manager_bc_loss_final'] = float(np.mean(losses[-200:]))
        if verbose:
            print(f"  [Warmup] Manager BC final loss: "
                  f"{results['manager_bc_loss_final']:.5f}")
    else:
        results['manager_bc_loss_final'] = float('nan')

    # 5. Save BC checkpoint
    if cfg.warmup.save_bc_checkpoint:
        ckpt_path = cfg.warmup.bc_checkpoint_path
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        agent.save(ckpt_path)
        if verbose:
            print(f"\n  [Warmup] Saved BC checkpoint -> {ckpt_path}")
        results['bc_checkpoint_path'] = ckpt_path

    return results