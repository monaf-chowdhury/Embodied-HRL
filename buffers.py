"""
Replay Buffers.

Changes vs original:
  - LowLevelBuffer now also stores the raw 59-d proprioceptive state at
    each transition.  The worker needs proprio at training time for the
    dual-stream network.  The buffer also exposes running mean/std for
    online normalisation (updated during add()).

  - HighLevelFERBuffer and ReachabilityBuffer are UNCHANGED.
"""
import numpy as np
from typing import Dict, Optional


# =============================================================================
# High-Level FER Buffer (unchanged)
# =============================================================================

class HighLevelFERBuffer:
    """
    Frontier Experience Replay for the manager (SSE-style).
    Stores success, stop-on-failure, and partial-success transitions.
    """

    def __init__(self, capacity: int = 100_000, z_dim: int = 64):
        self.capacity   = capacity
        self.z_dim      = z_dim

        self.z_current       = np.zeros((capacity, z_dim), dtype=np.float32)
        self.z_goal          = np.zeros((capacity, z_dim), dtype=np.float32)
        self.z_subgoal       = np.zeros((capacity, z_dim), dtype=np.float32)
        self.reward          = np.zeros(capacity, dtype=np.float32)
        self.z_next          = np.zeros((capacity, z_dim), dtype=np.float32)
        self.done            = np.zeros(capacity, dtype=np.float32)
        self.transition_type = np.zeros(capacity, dtype=np.int32)   # 0=success,1=fail,2=partial
        self.landmark_idx    = np.zeros(capacity, dtype=np.int64)

        self.ptr  = 0
        self.size = 0

    def add_success(self, z_t, z_goal, z_subgoal, cumulative_reward, z_next, landmark_idx):
        i = self.ptr
        self.z_current[i] = z_t;       self.z_goal[i]    = z_goal
        self.z_subgoal[i] = z_subgoal; self.reward[i]    = cumulative_reward
        self.z_next[i]    = z_next;    self.done[i]      = 0.0
        self.transition_type[i] = 0;   self.landmark_idx[i] = landmark_idx
        self._advance()

    def add_failure(self, z_t, z_goal, z_subgoal, landmark_idx):
        i = self.ptr
        self.z_current[i] = z_t;       self.z_goal[i]    = z_goal
        self.z_subgoal[i] = z_subgoal; self.reward[i]    = 0.0
        self.z_next[i]    = z_t;       self.done[i]      = 1.0
        self.transition_type[i] = 1;   self.landmark_idx[i] = landmark_idx
        self._advance()

    def add_partial(self, z_t, z_goal, z_partial_reached, partial_reward,
                    z_partial_end, landmark_idx):
        i = self.ptr
        self.z_current[i] = z_t;              self.z_goal[i]    = z_goal
        self.z_subgoal[i] = z_partial_reached; self.reward[i]   = partial_reward
        self.z_next[i]    = z_partial_end;    self.done[i]      = 1.0
        self.transition_type[i] = 2;          self.landmark_idx[i] = landmark_idx
        self._advance()

    def _advance(self):
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'z_current':       self.z_current[idx],
            'z_goal':          self.z_goal[idx],
            'z_subgoal':       self.z_subgoal[idx],
            'reward':          self.reward[idx],
            'z_next':          self.z_next[idx],
            'done':            self.done[idx],
            'transition_type': self.transition_type[idx],
            'landmark_idx':    self.landmark_idx[idx],
        }

    def __len__(self):
        return self.size


# =============================================================================
# Low-Level Buffer (now stores proprio)
# =============================================================================

class LowLevelBuffer:
    """
    Standard replay buffer for the worker (SAC).
    Stores: (z_t, proprio_t, z_subgoal, action, reward_shaped, z_next, proprio_next, done)

    Also maintains running mean/std over proprio dimensions for online
    normalisation.  Call normalise_proprio(x) to get a normalised vector.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        z_dim: int = 64,
        action_dim: int = 9,
        proprio_dim: int = 59,
    ):
        self.capacity   = capacity
        self.proprio_dim = proprio_dim

        self.z_current  = np.zeros((capacity, z_dim),       dtype=np.float32)
        self.proprio    = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.z_subgoal  = np.zeros((capacity, z_dim),       dtype=np.float32)
        self.action     = np.zeros((capacity, action_dim),  dtype=np.float32)
        self.reward     = np.zeros(capacity,                dtype=np.float32)
        self.z_next     = np.zeros((capacity, z_dim),       dtype=np.float32)
        self.proprio_next = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.done       = np.zeros(capacity,                dtype=np.float32)

        self.ptr  = 0
        self.size = 0

        # Running stats for proprio normalisation (Welford online algorithm)
        self._p_mean = np.zeros(proprio_dim, dtype=np.float64)
        self._p_M2   = np.ones(proprio_dim,  dtype=np.float64)   # sum of squared deviations
        self._p_n    = 0

    # ---- Welford update ----
    def _update_proprio_stats(self, x: np.ndarray):
        """Update running mean/variance with one new sample x (1-D array)."""
        self._p_n += 1
        delta     = x - self._p_mean
        self._p_mean += delta / self._p_n
        delta2    = x - self._p_mean
        self._p_M2   += delta * delta2

    def normalise_proprio(self, x: np.ndarray) -> np.ndarray:
        """Normalise a proprio vector using running stats. Safe before any data."""
        if self._p_n < 2:
            return x.astype(np.float32)
        std = np.sqrt(self._p_M2 / max(self._p_n - 1, 1)) + 1e-8
        return ((x - self._p_mean) / std).astype(np.float32)

    # ---- Add transition ----
    def add(self, z_t, proprio_t, z_subgoal, action, reward, z_next, proprio_next, done):
        i = self.ptr
        self.z_current[i]    = z_t
        self.proprio[i]      = proprio_t
        self.z_subgoal[i]    = z_subgoal
        self.action[i]       = action
        self.reward[i]       = reward
        self.z_next[i]       = z_next
        self.proprio_next[i] = proprio_next
        self.done[i]         = float(done)

        self._update_proprio_stats(np.asarray(proprio_t, dtype=np.float64))

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        # Return normalised proprio
        p_raw      = self.proprio[idx]
        pn_raw     = self.proprio_next[idx]
        std        = np.sqrt(self._p_M2 / max(self._p_n - 1, 1)) + 1e-8
        p_norm     = ((p_raw  - self._p_mean) / std).astype(np.float32)
        pn_norm    = ((pn_raw - self._p_mean) / std).astype(np.float32)
        return {
            'z_current':    self.z_current[idx],
            'proprio':      p_norm,
            'z_subgoal':    self.z_subgoal[idx],
            'action':       self.action[idx],
            'reward':       self.reward[idx],
            'z_next':       self.z_next[idx],
            'proprio_next': pn_norm,
            'done':         self.done[idx],
        }

    def get_all_z(self) -> np.ndarray:
        """Return all stored z_current values (for landmark computation)."""
        return self.z_current[:self.size].copy()

    def get_recent_z(self, fraction: float = 0.7) -> np.ndarray:
        """Return latents from the most-recent fraction of the buffer."""
        n      = self.size
        cutoff = max(int(n * (1.0 - fraction)), 1)
        # The most recent entries are just before ptr (circular buffer)
        if self.size < self.capacity:
            return self.z_current[cutoff:self.size].copy()
        # Buffer is full — wrap around
        start = (self.ptr - int(n * fraction)) % self.capacity
        if start < self.ptr:
            return self.z_current[start:self.ptr].copy()
        else:
            return np.concatenate([
                self.z_current[start:],
                self.z_current[:self.ptr],
            ], axis=0)

    def __len__(self):
        return self.size


# =============================================================================
# Reachability Buffer (unchanged)
# =============================================================================

class ReachabilityBuffer:
    """Stores (z_current, z_subgoal, success_label) for training f_ξ."""

    def __init__(self, capacity: int = 100_000, z_dim: int = 64):
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
        pos_idx = np.where(self.label[:self.size] > 0.5)[0]
        neg_idx = np.where(self.label[:self.size] <= 0.5)[0]
        half    = batch_size // 2
        if len(pos_idx) < half or len(neg_idx) < half:
            idx = np.random.randint(0, self.size, size=batch_size)
        else:
            idx = np.concatenate([
                np.random.choice(pos_idx, size=half, replace=True),
                np.random.choice(neg_idx, size=batch_size - half, replace=True),
            ])
            np.random.shuffle(idx)
        return {
            'z_current': self.z_current[idx],
            'z_subgoal': self.z_subgoal[idx],
            'label':     self.label[idx],
        }

    def __len__(self):
        return self.size