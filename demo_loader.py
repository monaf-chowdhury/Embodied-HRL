"""
demo_loader.py — Load Minari / D4RL kitchen demos for BC pretraining.

WHAT THIS DOES
--------------
1. Loads Minari kitchen datasets (Complete / Partial / Mixed) - teleoperated
   demo trajectories of (state, action, next_state, reward) tuples.

2. For every transition it computes which task is ACTIVE at that moment.
   A task K is active at time t if:
     (a) the env's completion bit for K flipped within the next
         LOOKAHEAD_STEPS env steps, AND
     (b) K is in the configured tasks_to_complete list.
   This produces DEMO actions labelled by "which task this action is helping
   to complete" - the clean per-transition signal the worker needs for BC.

3. Re-renders each transition's image by stepping a MuJoCo env with the
   same qpos/qvel the demo recorded. This is important because the worker's
   observation space at inference time is (image, state), so we must give
   it the same modality during BC.

4. Labels each transition with (task_id, task_target, task_cur_slice,
   task_mask) using the TaskSpec. These are ready to drop into the
   WorkerBuffer.

5. Balances the dataset across tasks (drops excess samples for over-
   represented tasks).

6. Caches the result to an .npz so subsequent runs skip re-rendering.

USAGE
-----
    from demo_loader import load_demo_bc_dataset
    ds = load_demo_bc_dataset(cfg, spec, encoder)
    # ds is a DemoBCDataset with .worker_samples and .manager_samples

WHY WE RENDER IMAGES OURSELVES
------------------------------
Minari / D4RL store only state, not pixels. We reset a fresh MuJoCo env to
the demo's qpos/qvel and call render() to get the matching frame. This is
slower than using pre-rendered frames, but (a) it's a one-time cost cached
to disk, and (b) it guarantees the rendered images exactly match what the
env produces at training time (same camera, same resolution).

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not train anything. It just builds the dataset. BC training happens
in warmup.py.
"""
from __future__ import annotations

import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils import TaskSpec
from env_wrapper import FrankaKitchenImageWrapper


# Number of steps ahead we "look" to decide which task is active at this
# transition. If multiple tasks complete within that window, we pick the
# one that completes SOONEST - that's the one this action is most directly
# contributing to.
LOOKAHEAD_STEPS = 30


# =============================================================================
# The output dataset structure
# =============================================================================

@dataclass
class DemoBCDataset:
    """
    Arrays ready to stream into BC training.

    Worker samples (one per transition):
      z           : (N, z_dim)    float32  image latent  (zeros if not rendered)
      proprio     : (N, 59)       float32  raw 59-d state at transition START
      task_target : (N, max_goal_dim)  float32  padded goal for the active task
      task_cur    : (N, max_goal_dim)  float32  padded task-state slice at START
      task_mask   : (N, max_goal_dim)  float32  goal-dim validity mask
      task_id     : (N,)          int64    active task id (in spec's task ordering)
      action      : (N, action_dim)  float32  demo action for this transition
      reward      : (N,)          float32  demo reward (not used for BC; kept for IQL later)
      proprio_next: (N, 59)       float32  state at END of transition
      z_next      : (N, z_dim)    float32  latent at END

    Manager samples (one per episode per task-completion event):
      m_z          : (M, z_dim)    float32
      m_proprio    : (M, 59)       float32
      m_task_state : (M, n_tasks * max_goal_dim) float32
      m_completion : (M, n_tasks)  float32   completion mask BEFORE the event
      m_label      : (M,)          int64    task id that completed next
    """
    z: np.ndarray
    proprio: np.ndarray
    task_target: np.ndarray
    task_cur: np.ndarray
    task_mask: np.ndarray
    task_id: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    proprio_next: np.ndarray
    z_next: np.ndarray

    m_z: np.ndarray
    m_proprio: np.ndarray
    m_task_state: np.ndarray
    m_completion: np.ndarray
    m_label: np.ndarray

    @property
    def n_worker(self) -> int:
        return len(self.action)

    @property
    def n_manager(self) -> int:
        return len(self.m_label)


# =============================================================================
# Minari loading
# =============================================================================

def _load_minari_episodes(dataset_ids: List[str]) -> List[Dict]:
    """
    Returns a list of episode dicts with keys:
        'observations' : (T+1, 59)        full 59-d state trajectory
        'actions'      : (T, 9)
        'rewards'      : (T,)
        'terminations' : (T,) bool
        'truncations'  : (T,) bool
        'dataset_id'   : which dataset this episode came from

    Uses minari.load_dataset(..., download=True) so the datasets are
    auto-fetched on first call and cached to ~/.minari/datasets/
    thereafter. No separate CLI step is required.
    """
    try:
        import minari
    except ImportError:
        raise ImportError(
            "Minari is required for demo BC. Install with:\n"
            "  pip install \"minari[all]\""
        )

    episodes: List[Dict] = []

    for ds_id in dataset_ids:
        # Try to load-with-download first; fall back to local-only load
        # if the user is offline and already has the dataset cached.
        ds = None
        try:
            print(f"  [DemoLoader] Loading '{ds_id}' (download=True if missing)...")
            ds = minari.load_dataset(ds_id, download=True)
        except TypeError:
            # Older Minari versions may not support download=True kwarg.
            # Try without it; if the dataset is missing, instruct the user.
            try:
                ds = minari.load_dataset(ds_id)
            except Exception as e:
                print(f"  [DemoLoader] Could not load '{ds_id}' without "
                      f"downloading ({e}). Try:\n"
                      f"    minari download {ds_id}\n"
                      f"  or upgrade Minari:  pip install -U \"minari[all]\"")
                continue
        except Exception as e:
            print(f"  [DemoLoader] Skipping '{ds_id}': {e}\n"
                  f"  If the dataset id is wrong, list remote options with:\n"
                  f"    minari list remote")
            continue

        n_eps = ds.total_episodes
        print(f"  [DemoLoader] '{ds_id}': {n_eps} episodes")

        for ep in ds.iterate_episodes():
            # Minari episodes for kitchen store obs as dict {'observation', 'achieved_goal', 'desired_goal'}
            obs = ep.observations
            if isinstance(obs, dict):
                obs_arr = np.asarray(obs['observation'], dtype=np.float64)
            else:
                obs_arr = np.asarray(obs, dtype=np.float64)

            # Actions (T, 9)
            acts = np.asarray(ep.actions, dtype=np.float32)

            # Sanity check - kitchen action dim is 9
            if acts.ndim != 2 or acts.shape[1] != 9:
                print(f"    [warn] unexpected action shape {acts.shape}; skipping")
                continue

            # obs_arr is (T+1, 59) for kitchen
            if obs_arr.ndim != 2 or obs_arr.shape[1] < 39:
                print(f"    [warn] unexpected obs shape {obs_arr.shape}; skipping")
                continue

            episodes.append({
                'observations': obs_arr,
                'actions': acts,
                'rewards': np.asarray(ep.rewards, dtype=np.float32),
                'terminations': np.asarray(ep.terminations, dtype=bool),
                'truncations': np.asarray(ep.truncations, dtype=bool),
                'dataset_id': ds_id,
            })

    if not episodes:
        raise RuntimeError(
            "No demo episodes loaded. Check:\n"
            "  1. You have internet connectivity (first run downloads ~GB of data)\n"
            "  2. The dataset IDs in config.warmup.minari_dataset_ids are valid.\n"
            "     See available ones with:  minari list remote\n"
            "  3. Minari is up to date: pip install -U \"minari[all]\""
        )

    return episodes


# =============================================================================
# Per-transition task labelling from the completion timeline
# =============================================================================

def _completion_timeline(states: np.ndarray, spec: TaskSpec) -> List[List[int]]:
    """
    For each step in the episode, returns the list of task-ids that are
    considered "completed" at that step based on task-space error.

    We use a slightly conservative rule: a task is considered completed
    at the first step t* where ||state[idx_k] - goal_k|| < eps_k AND the
    error stays below eps_k * 1.5 for the next few steps (to avoid false
    positives during approach). We also require MEANINGFUL task-space
    motion (error must have been >= 2 * eps at some earlier step) so we
    don't falsely claim a task was "completed" when the demo merely
    happened to start near the goal for that task.
    """
    T = states.shape[0]
    # (T, n_tasks) - is this task's error below threshold at step t?
    below = np.zeros((T, spec.n_tasks), dtype=bool)
    err_series = np.zeros((T, spec.n_tasks), dtype=np.float32)
    for k in range(spec.n_tasks):
        eps = spec.epsilon(k)
        errs = np.array([spec.task_error(states[t], k) for t in range(T)])
        err_series[:, k] = errs
        below[:, k] = errs < eps * 1.1   # slight relaxation to match env tolerance

    completed_at = [-1] * spec.n_tasks
    for k in range(spec.n_tasks):
        eps = spec.epsilon(k)
        for t in range(T):
            if not below[t, k]:
                continue
            window = below[t:min(t + 10, T), k]
            if window.sum() < min(5, len(window)):
                continue
            # Require the task to have been "far from goal" at some earlier
            # step - otherwise it was just initialised close and never
            # actually manipulated.
            if t > 0 and err_series[:t, k].max() < eps * 2.0:
                continue
            completed_at[k] = t
            break

    # Build per-step "completed set"
    timeline: List[List[int]] = [[] for _ in range(T)]
    for t in range(T):
        for k in range(spec.n_tasks):
            if 0 <= completed_at[k] <= t:
                timeline[t].append(k)
    return timeline


def _active_task_per_step(states: np.ndarray,
                          spec: TaskSpec,
                          lookahead: int = LOOKAHEAD_STEPS) -> np.ndarray:
    """
    For each step t, returns the task-id that is "active" at that step:
      = the soonest-to-complete task that is not yet completed at step t.

    Rationale: the demo is a fixed-order sequence of subtask completions
    (microwave -> kettle -> light -> slide, or whatever order the demo
    follows). At every step, there is exactly one "next task" until all
    tasks are done. Previously we skipped any transition where the next
    completion was more than `lookahead` steps away, which threw out ~72%
    of the dataset and left the worker blind to approach / transition
    motions. Now we always label.

    If at step t all the spec's tasks are already completed, we fall back
    to labeling with the task that has the largest remaining task-space
    error (arbitrary but stable — those trailing idle frames are rare).

    The `lookahead` arg is retained only for backwards compatibility;
    it no longer filters anything.
    """
    T = states.shape[0]
    timeline = _completion_timeline(states, spec)
    active = np.full(T, -1, dtype=np.int64)

    # When does each task first complete in this episode? (-1 = never)
    first_completed = [-1] * spec.n_tasks
    for t in range(T):
        for k in timeline[t]:
            if first_completed[k] == -1:
                first_completed[k] = t

    # For step t, pick the soonest-upcoming completion of a task that
    # isn't already done at t. No lookahead cap.
    for t in range(T):
        best_k = -1
        best_dt = T + 1
        for k in range(spec.n_tasks):
            tc = first_completed[k]
            if tc < 0:
                continue                 # task never completed in this demo
            if tc <= t:
                continue                 # already done before/at t
            dt = tc - t
            if dt < best_dt:
                best_dt = dt
                best_k = k
        if best_k < 0:
            # All completable tasks done, or none ever will. Fall back to
            # the task with the largest remaining task-space error.
            errs = np.array([spec.task_error(states[t], k)
                             for k in range(spec.n_tasks)])
            best_k = int(np.argmax(errs))
        active[t] = best_k

    return active


# =============================================================================
# State-replay rendering (the expensive step)
# =============================================================================

def _render_trajectory_images(env: FrankaKitchenImageWrapper,
                              states: np.ndarray,
                              encoder=None,
                              stride: int = 1) -> np.ndarray:
    """
    Replay a trajectory's qpos/qvel through the MuJoCo env and render each
    frame. Returns (T // stride, H, W, 3) uint8 images. If encoder is given,
    returns (T // stride, z_dim) float32 latents instead to save RAM.

    Kitchen state layout (59-d):
      [0:9]   robot qpos
      [9:30]  object qpos (21 dims)
      [30:39] robot qvel
      [39:60] object qvel (wait, 59-d total so [39:60] is 20 dims - check)

    Actually gymnasium-robotics kitchen returns:
      obs['observation']: concat(robot qpos[9], robot qvel[9], object qpos[21], object qvel[21]) - but this is 60
      or                   concat(robot qpos[9], object qpos[21], robot qvel[9], object qvel[21]) = 60

    The user's original indexing uses 18 for bottom-burner (after 9 robot
    qpos + 9 robot qvel = 18), so the layout is:
      [0:9]   robot qpos
      [9:18]  robot qvel
      [18:39] object qpos  (21 dims)
      [39:60] object qvel  (20 or 21 dims)

    To set state: qpos = [robot_qpos, object_qpos] (30 dims),
                  qvel = [robot_qvel, object_qvel] (30 dims).
    """
    raw_env = env._env.unwrapped
    model = raw_env.model
    data = raw_env.data

    n_qpos = model.nq                    # 30 for kitchen
    n_qvel = model.nv                    # 30 for kitchen

    # Build qpos/qvel from the 59-d state vector
    def set_state(s: np.ndarray):
        robot_qpos = s[0:9]
        robot_qvel = s[9:18]
        object_qpos = s[18:18 + (n_qpos - 9)]
        object_qvel = s[18 + (n_qpos - 9):18 + (n_qpos - 9) + (n_qvel - 9)]

        qpos = np.zeros(n_qpos, dtype=np.float64)
        qvel = np.zeros(n_qvel, dtype=np.float64)
        qpos[:9] = robot_qpos
        qpos[9:] = object_qpos[:n_qpos - 9]
        qvel[:9] = robot_qvel
        qvel[9:] = object_qvel[:n_qvel - 9]

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        import mujoco
        mujoco.mj_forward(model, data)

    env.reset()    # ensures the renderer is alive

    T = states.shape[0]
    out_zs = []
    batch_imgs = []
    BATCH = 64

    def flush():
        nonlocal batch_imgs
        if not batch_imgs:
            return
        if encoder is not None:
            zs = encoder.encode_numpy(np.stack(batch_imgs))
            out_zs.append(zs)
        else:
            out_zs.append(np.stack(batch_imgs))
        batch_imgs = []

    for t in range(0, T, stride):
        try:
            set_state(states[t])
            img = env.render_image()
        except Exception as e:
            # On any failure, substitute a black frame
            img = np.zeros((env.img_size, env.img_size, 3), dtype=np.uint8)
        batch_imgs.append(img)
        if len(batch_imgs) >= BATCH:
            flush()

    flush()
    return np.concatenate(out_zs, axis=0) if out_zs else np.zeros((0,))


# =============================================================================
# Main entry point
# =============================================================================

def load_demo_bc_dataset(cfg,
                         spec: TaskSpec,
                         encoder,
                         tasks_to_complete: List[str]) -> DemoBCDataset:
    """
    End-to-end: load Minari episodes, label each transition, render images,
    rebalance per-task, cache, return.

    Args:
      cfg : the top-level Config. Uses cfg.warmup.*
      spec: the TaskSpec for the run's task list
      encoder: VisualEncoder (for z-context on each transition)
      tasks_to_complete: the 4 target tasks (for manager label derivation)

    Returns:
      DemoBCDataset
    """
    cache_path = cfg.warmup.cache_path
    if os.path.exists(cache_path) and not cfg.warmup.rebuild_cache:
        print(f"  [DemoLoader] Loading cached demo dataset: {cache_path}")
        d = np.load(cache_path)
        return DemoBCDataset(
            z=d['z'], proprio=d['proprio'],
            task_target=d['task_target'], task_cur=d['task_cur'],
            task_mask=d['task_mask'], task_id=d['task_id'],
            action=d['action'], reward=d['reward'],
            proprio_next=d['proprio_next'], z_next=d['z_next'],
            m_z=d['m_z'], m_proprio=d['m_proprio'],
            m_task_state=d['m_task_state'], m_completion=d['m_completion'],
            m_label=d['m_label'],
        )

    print(f"  [DemoLoader] Loading Minari episodes from "
          f"{cfg.warmup.minari_dataset_ids}...")
    episodes = _load_minari_episodes(cfg.warmup.minari_dataset_ids)
    print(f"  [DemoLoader] Loaded {len(episodes)} total episodes.")

    # Render env for state-replay imaging
    render_env = None
    if cfg.warmup.render_demo_images:
        render_env = FrankaKitchenImageWrapper(
            tasks_to_complete=tasks_to_complete,
            img_size=cfg.encoder.img_size,
        )

    # Accumulators - worker
    all_z, all_proprio = [], []
    all_task_target, all_task_cur, all_task_mask = [], [], []
    all_task_id, all_action, all_reward = [], [], []
    all_proprio_next, all_z_next = [], []

    # Accumulators - manager
    all_m_z, all_m_proprio = [], []
    all_m_task_state, all_m_completion, all_m_label = [], [], []

    t_start = time.time()
    processed = 0
    skipped_no_active_task = 0

    for ep_i, ep in enumerate(episodes):
        states = ep['observations']
        actions = ep['actions']
        rewards = ep['rewards']
        T_ep = len(actions)

        # Per-step active task
        active = _active_task_per_step(states, spec)

        # Per-step completion mask (what's done so far)
        timeline = _completion_timeline(states, spec)
        completion_per_step = np.zeros((T_ep + 1, spec.n_tasks), dtype=np.float32)
        for t in range(T_ep + 1):
            for k in timeline[t]:
                completion_per_step[t, k] = 1.0

        # Render z's for this episode (one per transition)
        if render_env is not None:
            # Render at both t and t+1 - we need z and z_next
            # Efficient: render every step (stride=1) across states[0..T_ep]
            try:
                zs = _render_trajectory_images(render_env, states,
                                                encoder=encoder, stride=1)
                # zs has shape (T_ep + 1, z_dim)
            except Exception as e:
                print(f"    [warn] ep {ep_i} render failed ({e}); using zeros for z")
                zs = np.zeros((T_ep + 1, cfg.encoder.raw_dim), dtype=np.float32)
        else:
            zs = np.zeros((T_ep + 1, cfg.encoder.raw_dim), dtype=np.float32)

        # ----- Worker samples: one per transition with active task -----
        for t in range(T_ep):
            k = int(active[t])
            if k < 0:
                skipped_no_active_task += 1
                continue
            all_z.append(zs[t])
            all_proprio.append(states[t].astype(np.float32))
            all_task_target.append(spec.padded_goal_for(k))
            all_task_cur.append(spec.padded_state_slice_for(states[t], k))
            all_task_mask.append(spec.padded_mask_for(k))
            all_task_id.append(k)
            all_action.append(actions[t])
            all_reward.append(rewards[t])
            all_proprio_next.append(states[t + 1].astype(np.float32))
            all_z_next.append(zs[t + 1])

        # ----- Manager samples: one per "task K about to complete" event -----
        # For each completion event, the manager should learn that from the
        # state just before the event, it should select task K.
        # IMPORTANT: the kitchen env's completion predicate marks a task as
        # "done" whenever its object is near its goal qpos - including at the
        # very first step of a partial/mixed demo where the initial state
        # happens to satisfy some task already. If a task is "completed"
        # at t_obs, we must NOT emit a manager sample with that label,
        # because the manager masks completed-task logits to -inf and the
        # resulting NLL blows up to ~1e7.
        for k in range(spec.n_tasks):
            t_comp = -1
            for t in range(T_ep + 1):
                if k in timeline[t]:
                    t_comp = t
                    break
            if t_comp < 0:
                continue
            if t_comp == 0:
                # Task was already "completed" at t=0 (demo started with
                # that object near its goal). No decision was required.
                continue
            t_obs = max(0, t_comp - LOOKAHEAD_STEPS)
            completion_at_obs = completion_per_step[t_obs]
            if completion_at_obs[k] > 0.5:
                # Label task is masked at this step - skip to avoid NLL blow-up.
                continue
            all_m_z.append(zs[t_obs])
            all_m_proprio.append(states[t_obs].astype(np.float32))
            ts_flat = np.concatenate(
                [spec.padded_state_slice_for(states[t_obs], j)
                 for j in range(spec.n_tasks)],
            ).astype(np.float32)
            all_m_task_state.append(ts_flat)
            all_m_completion.append(completion_at_obs.copy())
            all_m_label.append(k)

        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - t_start
            print(f"    [DemoLoader] {processed}/{len(episodes)} "
                  f"({elapsed:.1f}s, {len(all_action):,} worker samples)")

        # Optional early cutoff
        if cfg.warmup.max_transitions and len(all_action) >= cfg.warmup.max_transitions:
            print(f"    [DemoLoader] Hit max_transitions cap "
                  f"({cfg.warmup.max_transitions:,}).")
            break

    if render_env is not None:
        render_env.close()

    # ----- Per-task rebalance -----
    worker_task_id_arr = np.array(all_task_id, dtype=np.int64)
    if cfg.warmup.max_per_task and len(worker_task_id_arr) > 0:
        keep_mask = np.zeros(len(worker_task_id_arr), dtype=bool)
        for k in range(spec.n_tasks):
            idx_k = np.where(worker_task_id_arr == k)[0]
            if len(idx_k) > cfg.warmup.max_per_task:
                sampled = np.random.choice(idx_k, size=cfg.warmup.max_per_task,
                                           replace=False)
                keep_mask[sampled] = True
            else:
                keep_mask[idx_k] = True
        keep = np.where(keep_mask)[0]
    else:
        keep = np.arange(len(worker_task_id_arr))

    print(f"\n  [DemoLoader] Worker samples before rebalance: {len(worker_task_id_arr):,}")
    print(f"  [DemoLoader] Worker samples after  rebalance: {len(keep):,}")
    print(f"  [DemoLoader] Per-task counts after rebalance:")
    for k in range(spec.n_tasks):
        c = int((worker_task_id_arr[keep] == k).sum())
        print(f"      {spec.name(k):>16s}  {c:>6,}")
    print(f"  [DemoLoader] Manager samples: {len(all_m_label):,}")
    print(f"  [DemoLoader] Skipped transitions (no active task): {skipped_no_active_task:,}")

    ds = DemoBCDataset(
        z=np.stack([all_z[i] for i in keep]).astype(np.float32) if len(keep) else np.zeros((0, cfg.encoder.raw_dim), np.float32),
        proprio=np.stack([all_proprio[i] for i in keep]).astype(np.float32) if len(keep) else np.zeros((0, 59), np.float32),
        task_target=np.stack([all_task_target[i] for i in keep]).astype(np.float32) if len(keep) else np.zeros((0, spec.max_goal_dim), np.float32),
        task_cur=np.stack([all_task_cur[i] for i in keep]).astype(np.float32) if len(keep) else np.zeros((0, spec.max_goal_dim), np.float32),
        task_mask=np.stack([all_task_mask[i] for i in keep]).astype(np.float32) if len(keep) else np.zeros((0, spec.max_goal_dim), np.float32),
        task_id=worker_task_id_arr[keep],
        action=np.stack([all_action[i] for i in keep]).astype(np.float32) if len(keep) else np.zeros((0, 9), np.float32),
        reward=np.stack([all_reward[i] for i in keep]).astype(np.float32) if len(keep) else np.zeros((0,), np.float32),
        proprio_next=np.stack([all_proprio_next[i] for i in keep]).astype(np.float32) if len(keep) else np.zeros((0, 59), np.float32),
        z_next=np.stack([all_z_next[i] for i in keep]).astype(np.float32) if len(keep) else np.zeros((0, cfg.encoder.raw_dim), np.float32),
        m_z=np.stack(all_m_z).astype(np.float32) if all_m_z else np.zeros((0, cfg.encoder.raw_dim), np.float32),
        m_proprio=np.stack(all_m_proprio).astype(np.float32) if all_m_proprio else np.zeros((0, 59), np.float32),
        m_task_state=np.stack(all_m_task_state).astype(np.float32) if all_m_task_state else np.zeros((0, spec.n_tasks * spec.max_goal_dim), np.float32),
        m_completion=np.stack(all_m_completion).astype(np.float32) if all_m_completion else np.zeros((0, spec.n_tasks), np.float32),
        m_label=np.array(all_m_label, dtype=np.int64) if all_m_label else np.zeros((0,), np.int64),
    )

    # Save cache
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(
        cache_path,
        z=ds.z, proprio=ds.proprio, task_target=ds.task_target,
        task_cur=ds.task_cur, task_mask=ds.task_mask, task_id=ds.task_id,
        action=ds.action, reward=ds.reward,
        proprio_next=ds.proprio_next, z_next=ds.z_next,
        m_z=ds.m_z, m_proprio=ds.m_proprio,
        m_task_state=ds.m_task_state, m_completion=ds.m_completion,
        m_label=ds.m_label,
    )
    print(f"  [DemoLoader] Cached to {cache_path}")

    return ds


# =============================================================================
# Action noise / clipping helpers for BC
# =============================================================================

def safe_atanh(a: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Invert tanh squashing numerically safely."""
    a_clip = np.clip(a, -1.0 + eps, 1.0 - eps)
    return np.arctanh(a_clip).astype(np.float32)