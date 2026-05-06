"""
demo_dataset.py — Offline kitchen-demo loading, labeling, rendering, and caching.

Stage A now pivots around purposeful demonstration data:
  1. Load kitchen demos from Minari or D4RL.
  2. Replay the demo actions in FrankaKitchen-v1 and recover the env's actual
     completion timeline from benchmark completion bits.
  3. Replay recorded states through MuJoCo to render RGB observations.
  4. Encode the rendered frames with the frozen visual encoder.
  5. Cache the resulting BC-ready arrays so subsequent runs skip the render cost.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent import SMGWAgent, build_task_state_flat
from config import Config
from env_wrapper import FrankaKitchenImageWrapper


_DATASET_ALIASES: Dict[str, List[str]] = {
    "franka-complete": ["D4RL/kitchen/complete-v2", "kitchen-complete-v0"],
    "franka-mixed": ["D4RL/kitchen/mixed-v2", "kitchen-mixed-v0"],
    "franka-partial": ["D4RL/kitchen/partial-v2", "kitchen-partial-v0"],
    "kitchen-complete-v0": ["kitchen-complete-v0", "D4RL/kitchen/complete-v2"],
    "kitchen-mixed-v0": ["kitchen-mixed-v0", "D4RL/kitchen/mixed-v2"],
    "kitchen-partial-v0": ["kitchen-partial-v0", "D4RL/kitchen/partial-v2"],
    "d4rl/kitchen/complete-v2": ["D4RL/kitchen/complete-v2", "kitchen-complete-v0"],
    "d4rl/kitchen/mixed-v2": ["D4RL/kitchen/mixed-v2", "kitchen-mixed-v0"],
    "d4rl/kitchen/partial-v2": ["D4RL/kitchen/partial-v2", "kitchen-partial-v0"],
}

_CACHE_VERSION = "v3_replaylabels_balancedbc_iql"


@dataclass
class DemoEpisode:
    observations: np.ndarray
    actions: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    dataset_id: str
    episode_id: str


class DemoPretrainDataset:
    """
    BC-ready arrays collected from offline demos.
    """

    def __init__(self, z_dim: int, proprio_dim: int, action_dim: int,
                 action_chunk_len: int, max_goal_dim: int, n_tasks: int):
        self.z_dim = z_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.H = action_chunk_len
        self.max_goal_dim = max_goal_dim
        self.n_tasks = n_tasks

        self.w_z: List[np.ndarray] = []
        self.w_p: List[np.ndarray] = []
        self.w_tt: List[np.ndarray] = []
        self.w_tc: List[np.ndarray] = []
        self.w_tm: List[np.ndarray] = []
        self.w_id: List[int] = []
        self.w_a: List[np.ndarray] = []
        self.w_r: List[float] = []
        self.w_done: List[float] = []
        self.w_z_next: List[np.ndarray] = []
        self.w_p_next: List[np.ndarray] = []
        self.w_tc_next: List[np.ndarray] = []

        self.m_z: List[np.ndarray] = []
        self.m_p: List[np.ndarray] = []
        self.m_ts: List[np.ndarray] = []
        self.m_c: List[np.ndarray] = []
        self.m_label: List[int] = []

        self.worker_task_counts = np.zeros(n_tasks, dtype=np.int64)
        self.manager_task_counts = np.zeros(n_tasks, dtype=np.int64)
        self.worker_indices_by_task: List[np.ndarray] = [
            np.zeros(0, dtype=np.int64) for _ in range(n_tasks)
        ]

    def add_worker(self, z, p, tt, tc, tm, tid, a_flat, reward,
                   z_next, p_next, tc_next, done):
        self.w_z.append(np.asarray(z, dtype=np.float16))
        self.w_p.append(np.asarray(p, dtype=np.float32))
        self.w_tt.append(np.asarray(tt, dtype=np.float32))
        self.w_tc.append(np.asarray(tc, dtype=np.float32))
        self.w_tm.append(np.asarray(tm, dtype=np.float32))
        self.w_id.append(int(tid))
        self.w_a.append(np.asarray(a_flat, dtype=np.float32))
        self.w_r.append(float(reward))
        self.w_z_next.append(np.asarray(z_next, dtype=np.float16))
        self.w_p_next.append(np.asarray(p_next, dtype=np.float32))
        self.w_tc_next.append(np.asarray(tc_next, dtype=np.float32))
        self.w_done.append(float(done))
        self.worker_task_counts[int(tid)] += 1

    def add_manager(self, z, p, ts, c, label):
        self.m_z.append(np.asarray(z, dtype=np.float16))
        self.m_p.append(np.asarray(p, dtype=np.float32))
        self.m_ts.append(np.asarray(ts, dtype=np.float32))
        self.m_c.append(np.asarray(c, dtype=np.float32))
        self.m_label.append(int(label))
        self.manager_task_counts[int(label)] += 1

    def extend(self, other: "DemoPretrainDataset"):
        self.w_z.extend(other.w_z)
        self.w_p.extend(other.w_p)
        self.w_tt.extend(other.w_tt)
        self.w_tc.extend(other.w_tc)
        self.w_tm.extend(other.w_tm)
        self.w_id.extend(other.w_id)
        self.w_a.extend(other.w_a)
        self.w_r.extend(other.w_r)
        self.w_done.extend(other.w_done)
        self.w_z_next.extend(other.w_z_next)
        self.w_p_next.extend(other.w_p_next)
        self.w_tc_next.extend(other.w_tc_next)

        self.m_z.extend(other.m_z)
        self.m_p.extend(other.m_p)
        self.m_ts.extend(other.m_ts)
        self.m_c.extend(other.m_c)
        self.m_label.extend(other.m_label)

        self.worker_task_counts += other.worker_task_counts
        self.manager_task_counts += other.manager_task_counts

    def n_worker(self) -> int:
        return len(self.w_z)

    def n_manager(self) -> int:
        return len(self.m_z)

    def finalize(self):
        self.w_z = (np.stack(self.w_z).astype(np.float16)
                    if self.w_z else np.zeros((0, self.z_dim), dtype=np.float16))
        self.w_p = (np.stack(self.w_p).astype(np.float32)
                    if self.w_p else np.zeros((0, self.proprio_dim), dtype=np.float32))
        self.w_tt = (np.stack(self.w_tt).astype(np.float32)
                     if self.w_tt else np.zeros((0, self.max_goal_dim), dtype=np.float32))
        self.w_tc = (np.stack(self.w_tc).astype(np.float32)
                     if self.w_tc else np.zeros((0, self.max_goal_dim), dtype=np.float32))
        self.w_tm = (np.stack(self.w_tm).astype(np.float32)
                     if self.w_tm else np.zeros((0, self.max_goal_dim), dtype=np.float32))
        self.w_id = np.asarray(self.w_id, dtype=np.int64)
        self.w_a = (np.stack(self.w_a).astype(np.float32)
                    if self.w_a else np.zeros((0, self.action_dim * self.H), dtype=np.float32))
        self.w_r = np.asarray(self.w_r, dtype=np.float32)
        self.w_done = np.asarray(self.w_done, dtype=np.float32)
        self.w_z_next = (np.stack(self.w_z_next).astype(np.float16)
                         if self.w_z_next else np.zeros((0, self.z_dim), dtype=np.float16))
        self.w_p_next = (np.stack(self.w_p_next).astype(np.float32)
                         if self.w_p_next else np.zeros((0, self.proprio_dim), dtype=np.float32))
        self.w_tc_next = (np.stack(self.w_tc_next).astype(np.float32)
                          if self.w_tc_next else np.zeros((0, self.max_goal_dim), dtype=np.float32))

        task_state_dim = self.n_tasks * self.max_goal_dim
        self.m_z = (np.stack(self.m_z).astype(np.float16)
                    if self.m_z else np.zeros((0, self.z_dim), dtype=np.float16))
        self.m_p = (np.stack(self.m_p).astype(np.float32)
                    if self.m_p else np.zeros((0, self.proprio_dim), dtype=np.float32))
        self.m_ts = (np.stack(self.m_ts).astype(np.float32)
                     if self.m_ts else np.zeros((0, task_state_dim), dtype=np.float32))
        self.m_c = (np.stack(self.m_c).astype(np.float32)
                    if self.m_c else np.zeros((0, self.n_tasks), dtype=np.float32))
        self.m_label = np.asarray(self.m_label, dtype=np.int64)
        self.worker_indices_by_task = [
            np.where(self.w_id == k)[0].astype(np.int64)
            for k in range(self.n_tasks)
        ]

    def sample_worker_batch(self, batch_size: int,
                            proprio_normalizer=None,
                            balance_by_task: bool = True) -> Dict[str, np.ndarray]:
        idx = self._sample_worker_indices(batch_size, balance_by_task=balance_by_task)
        p = self.w_p[idx]
        p_next = self.w_p_next[idx]
        if proprio_normalizer is not None:
            p = np.stack([proprio_normalizer(row) for row in p], axis=0)
            p_next = np.stack([proprio_normalizer(row) for row in p_next], axis=0)
        return {
            "z": self.w_z[idx].astype(np.float32),
            "proprio": p.astype(np.float32),
            "task_target": self.w_tt[idx],
            "task_cur": self.w_tc[idx],
            "task_mask": self.w_tm[idx],
            "task_id": self.w_id[idx],
            "action": self.w_a[idx],
            "reward": self.w_r[idx],
            "z_next": self.w_z_next[idx].astype(np.float32),
            "proprio_next": p_next.astype(np.float32),
            "task_cur_next": self.w_tc_next[idx].astype(np.float32),
            "done": self.w_done[idx],
        }

    def _sample_worker_indices(self, batch_size: int, balance_by_task: bool) -> np.ndarray:
        if (not balance_by_task) or self.n_worker() == 0:
            return np.random.randint(0, self.n_worker(), size=batch_size)

        available = [idx for idx in self.worker_indices_by_task if len(idx) > 0]
        if not available:
            return np.random.randint(0, self.n_worker(), size=batch_size)

        per_task = batch_size // len(available)
        remainder = batch_size % len(available)
        sampled = []
        for task_slot, indices in enumerate(available):
            n_take = per_task + (1 if task_slot < remainder else 0)
            if n_take <= 0:
                continue
            chosen = np.random.choice(indices, size=n_take, replace=True)
            sampled.append(chosen.astype(np.int64))

        if not sampled:
            return np.random.randint(0, self.n_worker(), size=batch_size)

        idx = np.concatenate(sampled, axis=0)
        np.random.shuffle(idx)
        return idx

    def sample_manager_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.n_manager(), size=batch_size)
        return {
            "z": self.m_z[idx].astype(np.float32),
            "proprio": self.m_p[idx],
            "task_state": self.m_ts[idx],
            "completion": self.m_c[idx],
            "label": self.m_label[idx],
        }


def build_or_load_demo_dataset(agent: SMGWAgent,
                               config: Config,
                               verbose: bool = True) -> Tuple[DemoPretrainDataset, Dict[str, float]]:
    os.makedirs(config.warmup.cache_dir, exist_ok=True)
    stats: Dict[str, float] = {
        "demo_datasets_requested": float(len(config.warmup.dataset_ids)),
    }
    total_replay_weight = 0.0
    replay_mean_accum = 0.0
    replay_max = 0.0

    merged = DemoPretrainDataset(
        z_dim=config.encoder.raw_dim,
        proprio_dim=config.worker.proprio_dim,
        action_dim=agent.action_dim,
        action_chunk_len=agent.H_chunk,
        max_goal_dim=agent.spec.max_goal_dim,
        n_tasks=agent.n_tasks,
    )

    render_env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        seed=config.training.seed + 7,
        terminate_on_tasks_completed=False,
    )
    replay_env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        seed=config.training.seed + 11,
        terminate_on_tasks_completed=False,
        max_steps=1000,
    )
    # Initialise renderer/data once before replaying offline states.
    render_env.reset()
    replay_env.reset()

    try:
        for requested_name in config.warmup.dataset_ids:
            cache_path = _cache_path_for(config, requested_name, agent.H_chunk)
            part_stats: Dict[str, float]
            if (not config.warmup.rebuild_cache) and os.path.exists(cache_path):
                part = _load_cached_dataset(cache_path, merged)
                part_stats = {
                    "worker_samples": float(part.n_worker()),
                    "manager_samples": float(part.n_manager()),
                    "cache_hit": 1.0,
                }
                if verbose:
                    print(f"  [Demo] Cache hit for {requested_name}: {cache_path}")
            else:
                episodes, source_name, resolved_name = load_demo_episodes(
                    requested_name,
                    source_preference=config.warmup.dataset_source,
                    max_episodes=config.warmup.max_episodes_per_dataset,
                )
                if verbose:
                    print(f"  [Demo] Loaded {len(episodes)} episodes from {resolved_name} "
                          f"via {source_name}.")
                part, part_stats = _build_dataset_from_episodes(
                    agent=agent,
                    config=config,
                    episodes=episodes,
                    render_env=render_env,
                    replay_env=replay_env,
                    verbose=verbose,
                )
                _save_cached_dataset(cache_path, part)
                part_stats["cache_hit"] = 0.0
                if verbose:
                    print(f"  [Demo] Cached BC tensors -> {cache_path}")

            merged.extend(part)
            safe_name = _safe_name(requested_name)
            for key, value in part_stats.items():
                stats[f"{safe_name}/{key}"] = float(value)
            worker_samples = float(part_stats.get("worker_samples", 0.0))
            if "replay_mean_state_l2" in part_stats and worker_samples > 0:
                replay_mean_accum += worker_samples * float(part_stats["replay_mean_state_l2"])
                total_replay_weight += worker_samples
                replay_max = max(replay_max, float(part_stats.get("replay_max_state_l2", 0.0)))
    finally:
        render_env.close()
        replay_env.close()

    merged.finalize()
    stats["demo_worker_samples"] = float(merged.n_worker())
    stats["demo_manager_samples"] = float(merged.n_manager())
    if total_replay_weight > 0:
        stats["replay_mean_state_l2"] = float(replay_mean_accum / total_replay_weight)
        stats["replay_max_state_l2"] = float(replay_max)
    for k, task_name in enumerate(agent.tasks):
        safe_task = _safe_name(task_name)
        stats[f"worker_labels/{safe_task}"] = float(merged.worker_task_counts[k])
        stats[f"manager_labels/{safe_task}"] = float(merged.manager_task_counts[k])
    return merged, stats


def load_demo_episodes(requested_name: str,
                       source_preference: str = "auto",
                       max_episodes: int = 0) -> Tuple[List[DemoEpisode], str, str]:
    candidates = _DATASET_ALIASES.get(requested_name.lower(), [requested_name])
    source_order = (["minari", "d4rl"] if source_preference == "auto"
                    else [source_preference])
    errors: List[str] = []

    for source_name in source_order:
        for candidate in candidates:
            try:
                if source_name == "minari":
                    episodes = _load_from_minari(candidate)
                elif source_name == "d4rl":
                    episodes = _load_from_d4rl(candidate)
                else:
                    raise ValueError(
                        f"Unknown dataset source '{source_preference}'. "
                        "Expected one of: auto, minari, d4rl."
                    )
                if max_episodes > 0:
                    episodes = episodes[:max_episodes]
                return episodes, source_name, candidate
            except Exception as exc:
                errors.append(f"{source_name}:{candidate}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not load any requested demo dataset. Tried:\n  - "
        + "\n  - ".join(errors)
    )


def _load_from_minari(dataset_id: str) -> List[DemoEpisode]:
    import minari

    ds = minari.load_dataset(dataset_id)
    episodes: List[DemoEpisode] = []
    for ep_idx, episode in enumerate(ds.iterate_episodes()):
        observations = _extract_observation_array(episode.observations)
        actions = np.asarray(episode.actions, dtype=np.float32)
        terminations = np.asarray(
            getattr(episode, "terminations", np.zeros(len(actions), dtype=bool)),
            dtype=bool,
        )
        truncations = np.asarray(
            getattr(episode, "truncations", np.zeros(len(actions), dtype=bool)),
            dtype=bool,
        )
        episodes.append(DemoEpisode(
            observations=observations,
            actions=actions,
            terminations=terminations,
            truncations=truncations,
            dataset_id=dataset_id,
            episode_id=f"{dataset_id}:ep{ep_idx:05d}",
        ))
    if not episodes:
        raise RuntimeError(f"Dataset '{dataset_id}' contained no episodes.")
    return episodes


def _load_from_d4rl(dataset_id: str) -> List[DemoEpisode]:
    try:
        import gym
    except Exception:
        import gymnasium as gym
    import d4rl  # noqa: F401

    env = gym.make(dataset_id)
    try:
        raw = env.get_dataset()
    finally:
        env.close()

    observations = _extract_observation_array(raw["observations"])
    actions = np.asarray(raw["actions"], dtype=np.float32)
    terminals = np.asarray(raw.get("terminals", np.zeros(len(actions))), dtype=bool)
    timeouts = np.asarray(raw.get("timeouts", np.zeros(len(actions))), dtype=bool)

    episodes: List[DemoEpisode] = []
    start = 0
    for idx in range(len(actions)):
        if terminals[idx] or timeouts[idx]:
            end = idx + 1
            episodes.append(DemoEpisode(
                observations=observations[start:end],
                actions=actions[start:end],
                terminations=terminals[start:end],
                truncations=timeouts[start:end],
                dataset_id=dataset_id,
                episode_id=f"{dataset_id}:ep{len(episodes):05d}",
            ))
            start = end

    if start < len(actions):
        episodes.append(DemoEpisode(
            observations=observations[start:],
            actions=actions[start:],
            terminations=terminals[start:],
            truncations=timeouts[start:],
            dataset_id=dataset_id,
            episode_id=f"{dataset_id}:ep{len(episodes):05d}",
        ))

    if not episodes:
        raise RuntimeError(f"D4RL dataset '{dataset_id}' contained no episodes.")
    return episodes


def _build_dataset_from_episodes(agent: SMGWAgent,
                                 config: Config,
                                 episodes: Sequence[DemoEpisode],
                                 render_env: FrankaKitchenImageWrapper,
                                 replay_env: FrankaKitchenImageWrapper,
                                 verbose: bool = True) -> Tuple[DemoPretrainDataset, Dict[str, float]]:
    ds = DemoPretrainDataset(
        z_dim=config.encoder.raw_dim,
        proprio_dim=config.worker.proprio_dim,
        action_dim=agent.action_dim,
        action_chunk_len=agent.H_chunk,
        max_goal_dim=agent.spec.max_goal_dim,
        n_tasks=agent.n_tasks,
    )
    stats: Dict[str, float] = {
        "episodes": float(len(episodes)),
        "episodes_with_labels": 0.0,
        "completion_events": 0.0,
        "replay_mean_state_l2": 0.0,
        "replay_max_state_l2": 0.0,
    }
    replay_state_errors: List[np.ndarray] = []

    for ep_i, episode in enumerate(episodes):
        (cur_states,
         next_states,
         labels,
         completion_masks,
         event_count,
         replay_errors,
         completion_event_task) = _label_episode_from_replay(
            agent,
            replay_env,
            episode,
            config.warmup.min_segment_len,
        )
        valid = np.where(labels >= 0)[0]
        if valid.size == 0:
            continue
        stats["episodes_with_labels"] += 1.0
        stats["completion_events"] += float(event_count)
        if replay_errors.size > 0:
            replay_state_errors.append(replay_errors)

        seg_start, seg_end = _segment_bounds(labels)
        z_valid = _encode_states(
            agent=agent,
            render_env=render_env,
            states=cur_states[valid],
            batch_size=config.warmup.render_batch_size,
        )
        z_next_valid = _encode_states(
            agent=agent,
            render_env=render_env,
            states=next_states[valid],
            batch_size=config.warmup.render_batch_size,
        )

        for local_idx, t in enumerate(valid):
            task_id = int(labels[t])
            chunk_end = min(t + agent.H_chunk - 1, seg_end[t])
            action_flat = _chunk_actions(episode.actions, t, chunk_end, agent.H_chunk)
            state_t = cur_states[t]
            next_state = next_states[chunk_end]
            reward = _chunk_reward(
                agent=agent,
                states=cur_states,
                next_states=next_states,
                actions=episode.actions,
                completion_events=completion_event_task,
                task_id=task_id,
                start=t,
                end=chunk_end,
            )

            ds.add_worker(
                z=z_valid[local_idx],
                p=state_t,
                tt=agent.spec.padded_goal_for(task_id),
                tc=agent.spec.padded_state_slice_for(state_t, task_id),
                tm=agent.spec.padded_mask_for(task_id),
                tid=task_id,
                a_flat=action_flat,
                reward=reward,
                z_next=z_next_valid[local_idx],
                p_next=next_state,
                tc_next=agent.spec.padded_state_slice_for(next_state, task_id),
                done=float(bool(episode.terminations[min(chunk_end, len(episode.terminations) - 1)]
                                or episode.truncations[min(chunk_end, len(episode.truncations) - 1)])),
            )

            if ((t - seg_start[t]) % max(config.warmup.manager_label_stride, 1)) == 0:
                ds.add_manager(
                    z=z_valid[local_idx],
                    p=state_t,
                    ts=build_task_state_flat(agent.spec, state_t),
                    c=completion_masks[t],
                    label=task_id,
                )

        if verbose and (ep_i + 1) % 50 == 0:
            print(f"  [Demo] Processed {ep_i + 1}/{len(episodes)} episodes  "
                  f"worker={ds.n_worker():,}  manager={ds.n_manager():,}")

    ds.finalize()
    stats["worker_samples"] = float(ds.n_worker())
    stats["manager_samples"] = float(ds.n_manager())
    if replay_state_errors:
        all_errors = np.concatenate(replay_state_errors, axis=0)
        stats["replay_mean_state_l2"] = float(np.mean(all_errors))
        stats["replay_max_state_l2"] = float(np.max(all_errors))
    return ds, stats


def _label_episode_from_replay(agent: SMGWAgent,
                               replay_env: FrankaKitchenImageWrapper,
                               episode: DemoEpisode,
                               min_segment_len: int
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    cur_states, next_states = _transition_views(episode.observations, episode.actions)
    n_steps = len(episode.actions)
    labels = np.full(n_steps, -1, dtype=np.int64)
    completion_masks = np.zeros((n_steps, agent.n_tasks), dtype=np.float32)
    replay_state_errors = np.zeros(n_steps, dtype=np.float32)
    completion_event_task = np.full(n_steps, -1, dtype=np.int64)

    replay_env.reset()
    qpos0, qvel0 = replay_env.observation_to_qpos_qvel(cur_states[0])
    replay_env.set_mujoco_state(qpos0, qvel0)

    completion_state = np.zeros(agent.n_tasks, dtype=np.float32)
    events: List[Tuple[int, int]] = []
    completed_names = set()

    for t in range(n_steps):
        completion_masks[t] = completion_state.copy()
        _, _, _, info = replay_env.step(episode.actions[t])
        replay_next_state = np.asarray(info["state"], dtype=np.float32)
        replay_state_errors[t] = float(
            np.linalg.norm(replay_next_state - next_states[t].astype(np.float32))
        )

        episode_completed = info.get("tasks_completed_names", [])
        just_completed = [
            name for name in episode_completed
            if name in agent.tasks and name not in completed_names
        ]
        for name in just_completed:
            task_id = agent.tasks.index(name)
            events.append((t, task_id))
            completion_state[task_id] = 1.0
            completed_names.add(name)
            if completion_event_task[t] < 0:
                completion_event_task[t] = task_id

    cursor = 0
    completion_before = np.zeros(agent.n_tasks, dtype=np.float32)
    for hit_idx, task_id in events:
        end_t = min(hit_idx, n_steps - 1)
        if end_t >= cursor and (end_t - cursor + 1) >= max(1, min_segment_len):
            labels[cursor:end_t + 1] = task_id
            completion_masks[cursor:end_t + 1] = completion_before
        completion_before = completion_before.copy()
        completion_before[task_id] = 1.0
        cursor = max(cursor, end_t + 1)

    return (cur_states, next_states, labels, completion_masks,
            len(events), replay_state_errors, completion_event_task)


def _transition_views(observations: np.ndarray,
                      actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    obs = _extract_observation_array(observations)
    n_steps = len(actions)
    if len(obs) == n_steps + 1:
        return obs[:-1], obs[1:]
    if len(obs) == n_steps:
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
        return obs, next_obs
    raise ValueError(
        f"Could not align observations/actions: obs={len(obs)} actions={n_steps}"
    )


def _segment_bounds(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    starts = np.full(len(labels), -1, dtype=np.int64)
    ends = np.full(len(labels), -1, dtype=np.int64)
    t = 0
    while t < len(labels):
        if labels[t] < 0:
            t += 1
            continue
        end = t
        while end + 1 < len(labels) and labels[end + 1] == labels[t]:
            end += 1
        starts[t:end + 1] = t
        ends[t:end + 1] = end
        t = end + 1
    return starts, ends


def _chunk_actions(actions: np.ndarray, start: int, end: int, chunk_len: int) -> np.ndarray:
    chunk = np.asarray(actions[start:end + 1], dtype=np.float32)
    if len(chunk) < chunk_len:
        pad = np.repeat(chunk[-1:], chunk_len - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.reshape(-1)


def _chunk_reward(agent: SMGWAgent,
                  states: np.ndarray,
                  next_states: np.ndarray,
                  actions: np.ndarray,
                  completion_events: np.ndarray,
                  task_id: int,
                  start: int,
                  end: int) -> float:
    reward = 0.0
    for t in range(start, end + 1):
        err_before = agent.spec.task_error(states[t], task_id)
        err_after = agent.spec.task_error(next_states[t], task_id)
        action_t = np.asarray(actions[t], dtype=np.float32)
        completed = bool(completion_events[t] == task_id)
        reward += agent._worker_step_reward(
            err_before,
            err_after,
            action_t,
            completion_bit_flipped=completed,
        )
    return float(reward)


def _encode_states(agent: SMGWAgent,
                   render_env: FrankaKitchenImageWrapper,
                   states: np.ndarray,
                   batch_size: int) -> np.ndarray:
    z_batches: List[np.ndarray] = []
    for start in range(0, len(states), max(1, batch_size)):
        sub = states[start:start + max(1, batch_size)]
        frames = np.stack([render_env.render_from_observation(s) for s in sub], axis=0)
        z_batches.append(agent.encoder.encode_numpy(frames))
    return np.concatenate(z_batches, axis=0) if z_batches else np.zeros(
        (0, agent.config.encoder.raw_dim), dtype=np.float32
    )


def _extract_observation_array(obs) -> np.ndarray:
    if isinstance(obs, dict):
        if "observation" in obs:
            obs = obs["observation"]
        else:
            first_key = next(iter(obs.keys()))
            obs = obs[first_key]
    return np.asarray(obs, dtype=np.float32)


def _cache_path_for(config: Config, requested_name: str, chunk_len: int) -> str:
    task_sig = "-".join(_safe_name(t) for t in config.training.tasks_to_complete)
    file_name = (
        f"{_safe_name(requested_name)}__{_CACHE_VERSION}_{config.encoder.name}"
        f"_img{config.encoder.img_size}_chunk{chunk_len}_{task_sig}.npz"
    )
    return os.path.join(config.warmup.cache_dir, file_name)


def _save_cached_dataset(path: str, ds: DemoPretrainDataset):
    np.savez(
        path,
        w_z=ds.w_z,
        w_p=ds.w_p,
        w_tt=ds.w_tt,
        w_tc=ds.w_tc,
        w_tm=ds.w_tm,
        w_id=ds.w_id,
        w_a=ds.w_a,
        w_r=ds.w_r,
        w_done=ds.w_done,
        w_z_next=ds.w_z_next,
        w_p_next=ds.w_p_next,
        w_tc_next=ds.w_tc_next,
        m_z=ds.m_z,
        m_p=ds.m_p,
        m_ts=ds.m_ts,
        m_c=ds.m_c,
        m_label=ds.m_label,
        worker_task_counts=ds.worker_task_counts,
        manager_task_counts=ds.manager_task_counts,
    )


def _load_cached_dataset(path: str, like: DemoPretrainDataset) -> DemoPretrainDataset:
    cache = np.load(path, allow_pickle=False)
    ds = DemoPretrainDataset(
        z_dim=like.z_dim,
        proprio_dim=like.proprio_dim,
        action_dim=like.action_dim,
        action_chunk_len=like.H,
        max_goal_dim=like.max_goal_dim,
        n_tasks=like.n_tasks,
    )
    ds.w_z = cache["w_z"]
    ds.w_p = cache["w_p"]
    ds.w_tt = cache["w_tt"]
    ds.w_tc = cache["w_tc"]
    ds.w_tm = cache["w_tm"]
    ds.w_id = cache["w_id"]
    ds.w_a = cache["w_a"]
    ds.w_r = cache["w_r"]
    ds.w_done = cache["w_done"]
    ds.w_z_next = cache["w_z_next"]
    ds.w_p_next = cache["w_p_next"]
    ds.w_tc_next = cache["w_tc_next"]
    ds.m_z = cache["m_z"]
    ds.m_p = cache["m_p"]
    ds.m_ts = cache["m_ts"]
    ds.m_c = cache["m_c"]
    ds.m_label = cache["m_label"]
    ds.worker_task_counts = cache["worker_task_counts"]
    ds.manager_task_counts = cache["manager_task_counts"]
    ds.worker_indices_by_task = [
        np.where(ds.w_id == k)[0].astype(np.int64)
        for k in range(ds.n_tasks)
    ]
    return ds


def _safe_name(text: str) -> str:
    return text.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
