"""
warmup.py — Stage A: Demo pretraining for SMGW.

Two pretraining modes (controlled by config.warmup.use_iql):

  use_iql=True  [DEFAULT, recommended]:
    IQL (Implicit Q-Learning) offline RL on the demo dataset.
      1. Build dense per-step rewards from task-space error reduction.
      2. Train V (expectile), Q (TD backup via V), and actor
         (advantage-weighted BC) jointly for N_IQL_STEPS steps.
      3. Follow with a short BC warmup for the actor (optional polish).
    Advantage over pure BC: avoids mode-averaging on multimodal demos,
    gives the actor a signal about which demo actions are actually useful.

  use_iql=False  [fallback]:
    Original MSE-BC: regress actor mean towards atanh(demo_action).
    Simpler but prone to mode-averaging → wishy-washy near-zero actions
    when the demo distribution is multimodal.

Manager pretraining (BC, CE loss) is the same in both modes.

Target single-task SR after Stage A (IQL mode):
  microwave, slide cabinet : 40-70%
  light switch              : 20-50%
  kettle                    : 10-40%

Run:
  python train.py --warmup_only --eval_single_tasks --log_dir logs/run
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Callable

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
    """
    Push every demo transition into the worker replay buffer.

    task_cur_next is now computed from proprio_next (not reusing task_cur),
    which is the correct next-step task-state slice for SAC bootstrapping
    in Stage B.
    """
    H = agent.H_chunk
    n = ds.n_worker
    if n == 0:
        return 0

    if H > 1:
        actions_flat = np.tile(ds.action, (1, H))
    else:
        actions_flat = ds.action

    spec = agent.spec
    for i in range(n):
        k = int(ds.task_id[i])
        # Correct: task_cur_next from the NEXT state's task-space slice.
        task_cur_next = spec.padded_state_slice_for(ds.proprio_next[i], k)
        agent.worker_buf.add(
            z=ds.z[i],
            proprio=ds.proprio[i],
            task_target=ds.task_target[i],
            task_cur=ds.task_cur[i],
            task_mask=ds.task_mask[i],
            task_id=k,
            action_flat=actions_flat[i],
            reward=float(ds.reward[i]),
            z_next=ds.z_next[i],
            proprio_next=ds.proprio_next[i],
            task_cur_next=task_cur_next,
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


def _make_iql_sampler(ds: DemoBCDataset,
                      agent: SMGWAgent,
                      config: Config) -> Callable[[int], Dict[str, np.ndarray]]:
    """
    Build a batch sampler for IQL training.

    Key differences from the plain BC sampler:
      1. Computes DENSE per-step rewards using task-space error reduction
         (matches exactly what Stage B computes during online rollouts).
      2. Computes task_cur_next from proprio_next rather than reusing task_cur.
      3. Properly normalises proprio and proprio_next together.
    """
    n = ds.n_worker
    spec = agent.spec
    wcfg = config.worker

    # Pre-compute task_cur_next and dense rewards for all samples at once.
    # Both loops iterate over tasks (4 tasks × N/4 samples each) → fast.
    print(f"  [IQL sampler] Pre-computing task_cur_next and dense rewards "
          f"for {n:,} samples ...")
    task_cur_next_all = np.zeros((n, spec.max_goal_dim), dtype=np.float32)
    dense_reward_all  = np.zeros(n, dtype=np.float32)
    action_cost_all   = wcfg.action_cost * np.sum(ds.action ** 2, axis=1)

    for k in range(spec.n_tasks):
        mask_k = (ds.task_id == k)
        if not mask_k.any():
            continue
        idx_k = np.where(mask_k)[0]
        # task_cur_next: extract and pad task-state slice from proprio_next
        raw_next = ds.proprio_next[idx_k][:, spec.indices(k)]   # (N_k, d_k)
        padded = np.zeros((len(idx_k), spec.max_goal_dim), dtype=np.float32)
        padded[:, :raw_next.shape[1]] = raw_next
        task_cur_next_all[idx_k] = padded
        # dense reward: progress + completion bonus - action cost
        raw_cur = ds.proprio[idx_k][:, spec.indices(k)]
        goal_k  = spec.goal(k)
        err_cur  = np.linalg.norm(raw_cur  - goal_k, axis=1)
        err_next = np.linalg.norm(raw_next - goal_k, axis=1)
        progress = err_cur - err_next
        completion = (ds.reward[idx_k] > 0.5).astype(np.float32)
        dense_reward_all[idx_k] = (
            wcfg.progress_weight * progress
            + wcfg.completion_bonus * completion
            - action_cost_all[idx_k]
        )
    print(f"  [IQL sampler] Done. Reward stats: "
          f"mean={dense_reward_all.mean():.4f}  "
          f"min={dense_reward_all.min():.4f}  "
          f"max={dense_reward_all.max():.4f}")

    def sample(bs: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, n, size=bs)

        p_raw  = ds.proprio[idx]
        pn_raw = ds.proprio_next[idx]
        p_norm  = np.stack([agent.worker_buf.normalize_proprio(p_raw[b])
                            for b in range(bs)]).astype(np.float32)
        pn_norm = np.stack([agent.worker_buf.normalize_proprio(pn_raw[b])
                            for b in range(bs)]).astype(np.float32)

        # Tile action for chunked workers
        H = agent.H_chunk
        a = ds.action[idx]
        if H > 1:
            a = np.tile(a, (1, H))

        return {
            'z':             ds.z[idx].astype(np.float32),
            'proprio':       p_norm,
            'task_target':   ds.task_target[idx],
            'task_cur':      ds.task_cur[idx],
            'task_mask':     ds.task_mask[idx],
            'task_id':       ds.task_id[idx],
            'action':        a.astype(np.float32),
            'reward':        dense_reward_all[idx],
            'z_next':        ds.z_next[idx].astype(np.float32),
            'proprio_next':  pn_norm,
            'task_cur_next': task_cur_next_all[idx],
            'done':          np.zeros(bs, dtype=np.float32),
        }

    return sample


def _run_iql_training(agent: SMGWAgent,
                      ds: DemoBCDataset,
                      config: Config,
                      verbose: bool = True) -> Dict[str, float]:
    """
    Train worker (V, Q, actor) with IQL for n_iql_steps steps.

    Also runs a short BC polish pass on the actor afterwards to ensure the
    actor distribution mode is well-aligned with the demo actions (IQL's
    advantage weighting can leave the mode slightly off due to the clipped
    exp-weight).

    Returns a dict of final mean losses.
    """
    cfg = config.warmup
    sampler = _make_iql_sampler(ds, agent, config)
    bc_sampler = _make_worker_sampler(ds, agent)    # for the BC polish pass

    iql_losses: Dict[str, list] = {
        'iql/v_loss': [], 'iql/q_loss': [], 'iql/actor_loss': [],
        'iql/adv_mean': [], 'iql/weight_mean': [], 'iql/logp_mean': [],
    }

    log_every = 5_000

    if verbose:
        print(f"\n  [Warmup] IQL pretraining: {cfg.n_iql_steps:,} steps, "
              f"batch {cfg.bc_batch_size}, "
              f"τ={cfg.iql_tau}, β={cfg.iql_beta}, γ={cfg.iql_gamma}")

    for step in range(cfg.n_iql_steps):
        batch = sampler(cfg.bc_batch_size)
        losses = agent.update_worker_iql(
            batch,
            iql_tau=cfg.iql_tau,
            iql_beta=cfg.iql_beta,
            gamma=cfg.iql_gamma,
            lr_override=cfg.iql_lr,
        )
        for k, v in losses.items():
            if k in iql_losses:
                iql_losses[k].append(v)

        if verbose and (step + 1) % log_every == 0:
            recent = {k: float(np.mean(v[-500:])) for k, v in iql_losses.items()}
            print(f"    step {step+1:>7,}/{cfg.n_iql_steps:,}  "
                  f"V={recent['iql/v_loss']:.4f}  "
                  f"Q={recent['iql/q_loss']:.4f}  "
                  f"actor={recent['iql/actor_loss']:.4f}  "
                  f"adv={recent['iql/adv_mean']:.3f}  "
                  f"w={recent['iql/weight_mean']:.2f}  "
                  f"logp={recent['iql/logp_mean']:.3f}")

    # Optional BC polish — DISABLED by default (n_iql_bc_polish_steps=0).
    # WARNING: more than ~5k steps reintroduces MSE-BC mode-averaging and
    # undoes the advantage-weighted IQL actor.  In the v2 run we observed
    # BC polish (20k steps) returning the actor to the same 0.037 loss as
    # pure BC, destroying IQL's per-task gains for slide cabinet / light switch.
    n_polish = cfg.n_iql_bc_polish_steps
    if n_polish > 0:
        if verbose:
            print(f"\n  [Warmup] BC polish: {n_polish:,} steps")
        bc_losses = []
        for step in range(n_polish):
            bc_batch = bc_sampler(cfg.bc_batch_size)
            bc_losses.append(
                _worker_bc_step(agent, bc_batch, lr_override=cfg.bc_lr)
            )
            if verbose and (step + 1) % 1_000 == 0:
                print(f"    step {step+1:>6,}/{n_polish:,}  "
                      f"BC loss = {float(np.mean(bc_losses[-200:])):.5f}")
        final_bc = float(np.mean(bc_losses[-200:]))
        if verbose:
            print(f"  [Warmup] BC polish final loss: {final_bc:.5f}")
    else:
        final_bc = float('nan')
        if verbose:
            print(f"\n  [Warmup] BC polish skipped (n_iql_bc_polish_steps=0).")

    return {
        'iql_v_loss_final':     float(np.mean(iql_losses['iql/v_loss'][-500:])),
        'iql_q_loss_final':     float(np.mean(iql_losses['iql/q_loss'][-500:])),
        'iql_actor_loss_final': float(np.mean(iql_losses['iql/actor_loss'][-500:])),
        'bc_polish_loss_final': final_bc,
    }


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

    # 3. Worker pretraining — IQL (recommended) or plain BC (fallback)
    if ds.n_worker > 0:
        if cfg.warmup.use_iql:
            if verbose:
                print(f"\n  [Warmup] Worker pretraining: IQL mode")
            iql_results = _run_iql_training(agent, ds, cfg, verbose=verbose)
            results.update(iql_results)
            results['worker_bc_loss_final'] = iql_results.get(
                'bc_polish_loss_final', float('nan'))
        else:
            if verbose:
                print(f"\n  [Warmup] Worker pretraining: BC mode "
                      f"({cfg.warmup.n_worker_bc_steps:,} steps, "
                      f"batch {cfg.warmup.bc_batch_size}, LR {cfg.warmup.bc_lr})")
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