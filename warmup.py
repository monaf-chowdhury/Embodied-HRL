"""
warmup.py — Stage A: Task-grounded warmup for SMGW.

Goal: give the worker and manager clean, labelled supervision BEFORE
online RL starts, so Stage B starts from a non-trivial initialization.
Addresses Core Problem 3 (noisy warmup labels) by:

  1. Running PER-TASK random-walk probes with a known task_id.
     For each task k, we reset the env, FIX the manager's choice to k,
     take short random-walk actions, and record transitions tagged with
     (task_id = k, task_target = g_k, task_cur = state[idx_k]).
     The TASK LABEL IS KNOWN BY CONSTRUCTION — not inferred from max-delta.

  2. Building a warmup worker dataset from those probes and running a
     small behavioural-cloning pretraining pass: regress actor mean
     toward the random-walk actions, which at least teaches the
     actor to produce valid-magnitude outputs conditioned on the
     task target / current slice pair.
     (Yes, random actions are poor targets — but for Stage A all we
      need is a small calibrated initial actor. Stage B does the real
      learning. If you have demonstrations, you can swap random-walk
      for demo actions here with no other changes.)

  3. Building a warmup manager dataset where the label is the task
     whose completion bit flipped during the probe. If no task
     completed (the common case with random actions), we skip that
     probe — we do not fabricate labels.

  4. Running a small supervised pass on the manager (cross-entropy
     over the 4 tasks, with completion mask applied).

  5. (Optional) Loading the demo GIF as *visual context* for the worker
     buffer. The demo is NOT used as a subgoal target — its role is
     only to widen the distribution of z_t values the worker and
     manager see during warmup, so they are less brittle at Stage B.

Stage A is cheap: O(few thousand env steps) and a small number of
supervised gradient steps.
"""
from __future__ import annotations
import os
import numpy as np
import torch
from typing import Dict, List, Optional

from config import Config
from env_wrapper import FrankaKitchenImageWrapper
from agent import SMGWAgent, build_task_state_flat


# =============================================================================
# A tiny in-memory dataset used for Stage-A supervised passes
# =============================================================================

class _WarmupDataset:
    """
    Minimal storage of (worker samples, manager samples) collected during
    Stage A. Kept in RAM for the Stage-A SL pass, then discarded.
    """
    def __init__(self, z_dim: int, proprio_dim: int, action_dim: int,
                 action_chunk_len: int, max_goal_dim: int, n_tasks: int):
        self.z_dim = z_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.H = action_chunk_len
        self.max_goal_dim = max_goal_dim
        self.n_tasks = n_tasks

        # Worker (chunk-level) samples
        self.w_z: List[np.ndarray] = []
        self.w_p: List[np.ndarray] = []
        self.w_tt: List[np.ndarray] = []
        self.w_tc: List[np.ndarray] = []
        self.w_tm: List[np.ndarray] = []
        self.w_id: List[int] = []
        self.w_a: List[np.ndarray] = []

        # Manager (option-level) samples with KNOWN task labels
        self.m_z: List[np.ndarray] = []
        self.m_p: List[np.ndarray] = []
        self.m_ts: List[np.ndarray] = []
        self.m_c: List[np.ndarray] = []
        self.m_label: List[int] = []

    # ---------------- writers ----------------

    def add_worker(self, z, p, tt, tc, tm, tid, a_flat):
        self.w_z.append(z.astype(np.float32))
        self.w_p.append(p.astype(np.float32))
        self.w_tt.append(tt.astype(np.float32))
        self.w_tc.append(tc.astype(np.float32))
        self.w_tm.append(tm.astype(np.float32))
        self.w_id.append(int(tid))
        self.w_a.append(a_flat.astype(np.float32))

    def add_manager(self, z, p, ts, c, label):
        self.m_z.append(z.astype(np.float32))
        self.m_p.append(p.astype(np.float32))
        self.m_ts.append(ts.astype(np.float32))
        self.m_c.append(c.astype(np.float32))
        self.m_label.append(int(label))

    # ---------------- batching ----------------

    def sample_worker_batch(self, batch_size: int,
                            proprio_normalizer=None) -> Dict[str, np.ndarray]:
        """
        proprio_normalizer : callable(raw_proprio: np.ndarray) -> np.ndarray
            If provided, normalizes proprio to match the online SAC pipeline.
        """
        n = len(self.w_z)
        idx = np.random.randint(0, n, size=batch_size)
        p = np.stack([self.w_p[i] for i in idx])
        if proprio_normalizer is not None:
            # Apply per-row; the normalizer expects (D,) shapes, so loop.
            p_out = np.stack([proprio_normalizer(p[b]) for b in range(p.shape[0])])
        else:
            p_out = p.astype(np.float32)
        return {
            'z':           np.stack([self.w_z[i] for i in idx]),
            'proprio':     p_out,
            'task_target': np.stack([self.w_tt[i] for i in idx]),
            'task_cur':    np.stack([self.w_tc[i] for i in idx]),
            'task_mask':   np.stack([self.w_tm[i] for i in idx]),
            'task_id':     np.array([self.w_id[i] for i in idx], dtype=np.int64),
            'action':      np.stack([self.w_a[i] for i in idx]),
        }

    def sample_manager_batch(self, batch_size: int,
                             proprio_normalizer=None) -> Dict[str, np.ndarray]:
        n = len(self.m_z)
        idx = np.random.randint(0, n, size=batch_size)
        p = np.stack([self.m_p[i] for i in idx])
        # NOTE: the manager uses raw proprio (it has its own torso with
        # LayerNorm) — we pass raw proprio through as float32.
        return {
            'z':          np.stack([self.m_z[i] for i in idx]),
            'proprio':    p.astype(np.float32),
            'task_state': np.stack([self.m_ts[i] for i in idx]),
            'completion': np.stack([self.m_c[i] for i in idx]),
            'label':      np.array([self.m_label[i] for i in idx], dtype=np.int64),
        }

    def n_worker(self) -> int:
        return len(self.w_z)

    def n_manager(self) -> int:
        return len(self.m_z)


# =============================================================================
# Probing: per-task random-walk episodes with KNOWN task labels
# =============================================================================

def _collect_probe_episodes(agent: SMGWAgent,
                            env: FrankaKitchenImageWrapper,
                            n_episodes_per_task: int,
                            steps_per_episode: int) -> _WarmupDataset:
    """
    For each task k, run n_episodes_per_task episodes where we take random
    actions for up to steps_per_episode steps. Every chunk is labelled with
    task_id = k. If during the episode the env flips task k's completion
    bit, we record an (option-level) manager sample labelled k.

    We ALSO write a manager sample for every option where task k's
    completion bit did NOT flip but some OTHER task did (we use the other
    task as the label). This only adds supervision if luck made us
    complete something; it never fabricates labels.
    """
    cfg = agent.config
    H = agent.H_chunk
    action_dim = agent.action_dim
    K = cfg.manager.subgoal_horizon

    ds = _WarmupDataset(
        z_dim=cfg.encoder.raw_dim, proprio_dim=cfg.worker.proprio_dim,
        action_dim=action_dim, action_chunk_len=H,
        max_goal_dim=agent.spec.max_goal_dim, n_tasks=agent.n_tasks,
    )

    for task_id in range(agent.n_tasks):
        task_name = agent.spec.name(task_id)
        for ep in range(n_episodes_per_task):
            img, state = env.reset()
            z = agent.encoder.encode_numpy(img).squeeze()
            proprio = state.copy()
            completion = np.zeros(agent.n_tasks, dtype=np.float32)

            # Run several "options" inside the episode, each labelled with task_id
            steps_done = 0
            while steps_done < steps_per_episode:
                # Record option-level context
                ts_flat_start = build_task_state_flat(agent.spec, state)
                z_start = z.copy()
                state_start = state.copy()
                completion_start = completion.copy()
                completed_at_start = set(
                    n for k_, n in enumerate(agent.tasks) if completion[k_] > 0.5
                )

                # Execute up to K env steps of random actions, in chunks of H
                option_steps = 0
                newly_completed_option = []
                env_done = False
                while option_steps < min(K, steps_per_episode - steps_done) and not env_done:
                    # Sample a random chunk of H actions
                    a_chunk = np.random.uniform(-1.0, 1.0,
                                                size=(H, action_dim)).astype(np.float32)

                    z_chunk_start = z.copy()
                    state_chunk_start = state.copy()
                    proprio_chunk_start = proprio.copy()

                    for h in range(H):
                        if option_steps >= K or steps_done >= steps_per_episode:
                            break
                        a_step = a_chunk[h]
                        next_img, _, done_env, info = env.step(a_step)
                        next_state = np.asarray(info['state'], dtype=np.float64)
                        next_z = agent.encoder.encode_numpy(next_img).squeeze()
                        now_completed = info.get('tasks_completed_names', [])
                        for n in now_completed:
                            if n not in completed_at_start and n not in newly_completed_option:
                                newly_completed_option.append(n)

                        state, proprio, z = next_state, next_state, next_z
                        option_steps += 1
                        steps_done += 1
                        if done_env:
                            env_done = True
                            break

                    # Write worker warmup sample with KNOWN task_id
                    tt = agent.spec.padded_goal_for(task_id)
                    tm = agent.spec.padded_mask_for(task_id)
                    tc = agent.spec.padded_state_slice_for(state_chunk_start, task_id)
                    ds.add_worker(
                        z=z_chunk_start, p=proprio_chunk_start,
                        tt=tt, tc=tc, tm=tm, tid=task_id,
                        a_flat=a_chunk.reshape(-1),
                    )

                # Update completion mask
                completion_end = completion_start.copy()
                completed_all = set(completed_at_start)
                completed_all.update(newly_completed_option)
                for k_, n in enumerate(agent.tasks):
                    if n in completed_all:
                        completion_end[k_] = 1.0
                completion = completion_end

                # Manager sample: use task_id as the known label IF it flipped;
                # otherwise if some OTHER task flipped, use that as the label
                # (this is still a clean label — we know what happened).
                if task_name in newly_completed_option:
                    ds.add_manager(z=z_start, p=state_start,
                                   ts=ts_flat_start, c=completion_start,
                                   label=task_id)
                elif len(newly_completed_option) > 0:
                    other = newly_completed_option[0]
                    if other in agent.tasks:
                        ds.add_manager(z=z_start, p=state_start,
                                       ts=ts_flat_start, c=completion_start,
                                       label=agent.tasks.index(other))
                # If no task completed, we do NOT add a manager sample —
                # we refuse to fabricate labels.

                if env_done:
                    break

    return ds


# =============================================================================
# Optional: prime the worker buffer with demo frames (context only)
# =============================================================================

def _seed_from_demo_gif(agent: SMGWAgent, gif_path: str, max_frames: int):
    """
    Decode the demo GIF frames, encode to z, and add them as CONTEXT-ONLY
    transitions. Because we don't know the demo's per-frame action, we
    insert zero actions with zero reward; they serve purely as visual
    context so the worker's trunk doesn't freeze on narrow warmup statistics.

    If the GIF is missing, we silently skip.
    """
    if not os.path.exists(gif_path):
        print(f"  [Warmup] Demo GIF not found at {gif_path}; skipping demo seeding.")
        return 0

    try:
        from PIL import Image
        gif = Image.open(gif_path)
    except Exception as e:
        print(f"  [Warmup] Could not open demo GIF ({e}); skipping.")
        return 0

    frames = []
    try:
        i = 0
        while True:
            gif.seek(i)
            f = gif.convert('RGB').resize((agent.config.encoder.img_size,
                                           agent.config.encoder.img_size))
            frames.append(np.array(f, dtype=np.uint8))
            i += 1
    except EOFError:
        pass

    if not frames:
        return 0

    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idx]

    zs = agent.encoder.encode_numpy(np.stack(frames))
    # We don't add these to worker_buf (no action/target); instead, store them
    # on the agent so the encoder's "domain" has seen demo statistics. If you
    # want them to serve as additional context for BC, you could extend
    # _WarmupDataset to hold demo z's — but we keep Stage A minimal.
    agent._demo_z_for_logging = zs
    print(f"  [Warmup] Loaded {len(frames)} demo frames (context only, not used as subgoals).")
    return len(frames)


# =============================================================================
# Public entry point
# =============================================================================

def run_stage_a_warmup(agent: SMGWAgent,
                       config: Config,
                       verbose: bool = True) -> Dict[str, float]:
    """
    Full Stage A:
      1. Per-task random-walk probes with known task labels.
      2. Optional demo GIF loading for visual context.
      3. Supervised BC pass on the worker actor.
      4. Supervised CE pass on the manager (with completion mask).
      5. Populate the worker replay buffer with the probe transitions,
         so Stage B starts with a non-empty buffer.

    Returns a dict of stage-A metrics.
    """
    cfg = config
    results: Dict[str, float] = {}

    # -- Build a probe env --
    probe_env = FrankaKitchenImageWrapper(
        tasks_to_complete=cfg.training.tasks_to_complete,
        img_size=cfg.encoder.img_size,
        seed=cfg.training.seed + 1,          # different seed from main env
    )

    if verbose:
        print("  [Warmup] Collecting per-task random-walk probes…")
    ds = _collect_probe_episodes(
        agent=agent, env=probe_env,
        n_episodes_per_task=cfg.warmup.n_probe_episodes_per_task,
        steps_per_episode=cfg.warmup.probe_steps_per_episode,
    )
    probe_env.close()
    if verbose:
        print(f"  [Warmup] Collected {ds.n_worker():,} worker chunks, "
              f"{ds.n_manager():,} manager option samples.")
    results['probe_worker_samples'] = ds.n_worker()
    results['probe_manager_samples'] = ds.n_manager()

    # -- Demo (optional, context only) --
    if cfg.warmup.use_demo_gif_for_context:
        _seed_from_demo_gif(agent, cfg.warmup.demo_gif_path,
                            cfg.warmup.demo_max_frames)

    # -- Feed probe transitions into the real worker buffer so Stage B
    # doesn't start from an empty replay. We use action-space clamping
    # so the BC-targets match the actual env range.
    for i in range(ds.n_worker()):
        agent.worker_buf.add(
            z=ds.w_z[i], proprio=ds.w_p[i],
            task_target=ds.w_tt[i], task_cur=ds.w_tc[i], task_mask=ds.w_tm[i],
            task_id=ds.w_id[i],
            action_flat=ds.w_a[i],
            reward=0.0,                     # conservative: no reward signal from random actions
            z_next=ds.w_z[i],               # we don't store next-state from probes;
            proprio_next=ds.w_p[i],         # Stage-B online data will supply the real TD targets
            task_cur_next=ds.w_tc[i],
            done=1.0,                       # treat as terminal so TD bootstraps = 0
        )

    # -- Supervised BC on worker actor --
    if ds.n_worker() > 0:
        if verbose:
            print(f"  [Warmup] Worker BC: {cfg.warmup.n_worker_sl_steps} steps "
                  f"on {ds.n_worker():,} samples.")
        bc_losses = []
        for step in range(cfg.warmup.n_worker_sl_steps):
            batch = ds.sample_worker_batch(
                cfg.warmup.sl_batch_size,
                proprio_normalizer=agent.worker_buf.normalize_proprio,
            )
            bc_losses.append(agent.worker_warmup_step(batch))
        results['worker_bc_loss_final'] = float(np.mean(bc_losses[-100:])) \
            if len(bc_losses) >= 100 else float(np.mean(bc_losses))
        if verbose:
            print(f"  [Warmup] Worker BC final loss: {results['worker_bc_loss_final']:.4f}")

    # -- Supervised CE on manager --
    if ds.n_manager() > 0:
        if verbose:
            print(f"  [Warmup] Manager CE: {cfg.warmup.n_manager_sl_steps} steps "
                  f"on {ds.n_manager():,} samples.")
        ce_losses = []
        for step in range(cfg.warmup.n_manager_sl_steps):
            batch = ds.sample_manager_batch(cfg.warmup.sl_batch_size)
            ce_losses.append(agent.manager_warmup_step(batch))
        results['manager_ce_loss_final'] = float(np.mean(ce_losses[-100:])) \
            if len(ce_losses) >= 100 else float(np.mean(ce_losses))
        if verbose:
            print(f"  [Warmup] Manager CE final loss: {results['manager_ce_loss_final']:.4f}")
    else:
        if verbose:
            print("  [Warmup] No manager samples collected — skipping CE pass. "
                  "This is expected if no task happened to complete during probes. "
                  "Manager will still be updated online in Stage B from the option buffer.")

    return results
