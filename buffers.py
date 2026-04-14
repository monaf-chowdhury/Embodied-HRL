"""
Replay Buffers.

Changes from previous version:
  - LowLevelBuffer now additionally stores task_delta (float32) per transition.
    This is the per-step change in task progress, used during landmark updates
    to preferentially sample transitions where task progress improved.
    The buffer exposes get_task_biased_z() for this purpose.
  - HighLevelFERBuffer: removed z_goal field (unused). Now stores
    n_tasks_before and n_tasks_after per transition for manager reward.
  - ReachabilityBuffer: unchanged.
"""
import numpy as np
from typing import Dict, Optional


# =============================================================================
# High-Level Buffer (manager transitions)
# =============================================================================

class HighLevelBuffer:
    """
    Manager replay buffer. Stores one transition per subgoal attempt:
    (z_start, z_subgoal, manager_reward, z_end, done, landmark_idx)

    No FER, no SSE-style failure handling. Every subgoal attempt is stored
    regardless of outcome. The manager_reward is computed externally in
    agent.compute_manager_reward() and encodes task-completion signal.
    """

    def __init__(self, capacity: int = 200_000, z_dim: int = 2048):
        self.capacity = capacity
        self.z_dim    = z_dim

        self.z_current    = np.zeros((capacity, z_dim), dtype=np.float32)
        self.z_subgoal    = np.zeros((capacity, z_dim), dtype=np.float32)
        self.reward       = np.zeros(capacity, dtype=np.float32)
        self.z_next       = np.zeros((capacity, z_dim), dtype=np.float32)
        self.done         = np.zeros(capacity, dtype=np.float32)
        self.landmark_idx = np.zeros(capacity, dtype=np.int64)

        self.ptr  = 0
        self.size = 0

    def add(self, z_current, z_subgoal, manager_reward, z_next, done, landmark_idx):
        i = self.ptr
        self.z_current[i]    = z_current
        self.z_subgoal[i]    = z_subgoal
        self.reward[i]       = manager_reward
        self.z_next[i]       = z_next
        self.done[i]         = float(done)
        self.landmark_idx[i] = landmark_idx
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'z_current':    self.z_current[idx],
            'z_subgoal':    self.z_subgoal[idx],
            'reward':       self.reward[idx],
            'z_next':       self.z_next[idx],
            'done':         self.done[idx],
            'landmark_idx': self.landmark_idx[idx],
        }

    def __len__(self):
        return self.size


# =============================================================================
# Low-Level Buffer (worker transitions)
# =============================================================================

class LowLevelBuffer:
    """
    Worker SAC replay buffer.
    Stores: (z_t, proprio_t, z_subgoal, action, reward, z_next, proprio_next,
             done, task_delta)

    task_delta: change in task progress at this step (computed externally).
    Used during landmark updates to bias the candidate pool toward task-
    relevant states.

    Also runs Welford online algorithm for proprio normalisation.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        z_dim: int = 2048,
        action_dim: int = 9,
        proprio_dim: int = 59,
    ):
        self.capacity    = capacity
        self.proprio_dim = proprio_dim

        self.z_current    = np.zeros((capacity, z_dim),       dtype=np.float32)
        self.proprio      = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.z_subgoal    = np.zeros((capacity, z_dim),       dtype=np.float32)
        self.action       = np.zeros((capacity, action_dim),  dtype=np.float32)
        self.reward       = np.zeros(capacity,                dtype=np.float32)
        self.z_next       = np.zeros((capacity, z_dim),       dtype=np.float32)
        self.proprio_next = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.done         = np.zeros(capacity,                dtype=np.float32)
        self.task_delta   = np.zeros(capacity,                dtype=np.float32)

        self.ptr  = 0
        self.size = 0

        # Welford running stats for proprio normalisation
        self._p_mean = np.zeros(proprio_dim, dtype=np.float64)
        self._p_M2   = np.ones(proprio_dim,  dtype=np.float64)
        self._p_n    = 0

    def _update_proprio_stats(self, x: np.ndarray):
        self._p_n += 1
        delta = x - self._p_mean
        self._p_mean += delta / self._p_n
        self._p_M2   += delta * (x - self._p_mean)

    def normalise_proprio(self, x: np.ndarray) -> np.ndarray:
        if self._p_n < 2:
            return x.astype(np.float32)
        std = np.sqrt(self._p_M2 / max(self._p_n - 1, 1)) + 1e-8
        return ((x - self._p_mean) / std).astype(np.float32)

    def add(
        self,
        z_t, proprio_t, z_subgoal, action, reward,
        z_next, proprio_next, done, task_delta: float = 0.0,
    ):
        i = self.ptr
        self.z_current[i]    = z_t
        self.proprio[i]      = proprio_t
        self.z_subgoal[i]    = z_subgoal
        self.action[i]       = action
        self.reward[i]       = reward
        self.z_next[i]       = z_next
        self.proprio_next[i] = proprio_next
        self.done[i]         = float(done)
        self.task_delta[i]   = float(task_delta)

        self._update_proprio_stats(np.asarray(proprio_t, dtype=np.float64))

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx    = np.random.randint(0, self.size, size=batch_size)
        p_raw  = self.proprio[idx]
        pn_raw = self.proprio_next[idx]
        std    = np.sqrt(self._p_M2 / max(self._p_n - 1, 1)) + 1e-8
        return {
            'z_current':    self.z_current[idx],
            'proprio':      ((p_raw  - self._p_mean) / std).astype(np.float32),
            'z_subgoal':    self.z_subgoal[idx],
            'action':       self.action[idx],
            'reward':       self.reward[idx],
            'z_next':       self.z_next[idx],
            'proprio_next': ((pn_raw - self._p_mean) / std).astype(np.float32),
            'done':         self.done[idx],
        }

    def get_all_z(self) -> np.ndarray:
        return self.z_current[:self.size].copy()

    def get_task_biased_z(self, top_pct: float = 0.3) -> np.ndarray:
        """
        Return z_next for transitions in the top `top_pct` by task_delta,
        plus all transitions where task_delta > 0 (any task progress).
        This gives the landmark FPS a task-relevant candidate pool.
        Falls back to all z if not enough high-delta transitions exist.
        """
        if self.size == 0:
            return self.z_current[:self.size].copy()

        deltas = self.task_delta[:self.size]

        # Always include states where task progress was positive
        positive_mask = deltas > 1e-4
        positive_idx  = np.where(positive_mask)[0]

        # Also include top_pct overall
        n_top = max(int(self.size * top_pct), 50)
        top_idx = np.argpartition(deltas, -n_top)[-n_top:]

        combined_idx = np.union1d(positive_idx, top_idx)

        if len(combined_idx) < 20:
            # Fallback: not enough task-relevant data yet, use all
            return self.z_current[:self.size].copy()

        return self.z_next[combined_idx].copy()

    def get_recent_z(self, fraction: float = 0.7) -> np.ndarray:
        n      = self.size
        cutoff = max(int(n * (1.0 - fraction)), 1)
        if self.size < self.capacity:
            return self.z_current[cutoff:self.size].copy()
        start = (self.ptr - int(n * fraction)) % self.capacity
        if start < self.ptr:
            return self.z_current[start:self.ptr].copy()
        return np.concatenate([self.z_current[start:], self.z_current[:self.ptr]], axis=0)

    def __len__(self):
        return self.size


# =============================================================================
# Reachability Buffer (unchanged — kept for potential future use)
# =============================================================================

class ReachabilityBuffer:
    def __init__(self, capacity: int = 100_000, z_dim: int = 2048):
        self.capacity  = capacity
        self.z_current = np.zeros((capacity, z_dim), dtype=np.float32)
        self.z_subgoal = np.zeros((capacity, z_dim), dtype=np.float32)
        self.label     = np.zeros(capacity,           dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def add(self, z_current, z_subgoal, success: bool):
        i = self.ptr
        self.z_current[i] = z_current
        self.z_subgoal[i] = z_subgoal
        self.label[i]     = 1.0 if success else 0.0
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_balanced(self, batch_size: int) -> Dict[str, np.ndarray]:
        pos = np.where(self.label[:self.size] > 0.5)[0]
        neg = np.where(self.label[:self.size] <= 0.5)[0]
        half = batch_size // 2
        if len(pos) < half or len(neg) < half:
            idx = np.random.randint(0, self.size, size=batch_size)
        else:
            idx = np.concatenate([
                np.random.choice(pos, half, replace=True),
                np.random.choice(neg, batch_size - half, replace=True),
            ])
            np.random.shuffle(idx)
        return {
            'z_current': self.z_current[idx],
            'z_subgoal': self.z_subgoal[idx],
            'label':     self.label[idx],
        }

    def __len__(self):
        return self.size
