"""
buffers.py — Replay buffers for SMGW.

Two buffers:

  * ManagerBuffer (option transitions)
    A transition is:
      (z, proprio, task_state_flat, completion_mask,
       chosen_task_id,
       option_return (scalar),
       z_next, proprio_next, task_state_flat_next, completion_mask_next,
       done)
    where option_return is the semantic-level return for the whole option.
    The manager acts at option boundaries.

  * WorkerBuffer (low-level transitions with optional action chunks)
    A transition is:
      (z, proprio, task_target, task_cur, task_mask, task_embed_idx,
       action_chunk (H, A),  reward_sum (scalar over the chunk),
       z_next, proprio_next, task_cur_next,
       done)
    If action_chunk_len == 1, the chunk dimension is size 1 and reward_sum
    is just the one-step reward — behaves as standard SAC.

  Proprio running statistics are maintained in the worker buffer and used
  to normalise proprio inputs at both write-time and sample-time.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional


# =============================================================================
# Running normalizer for proprio
# =============================================================================

class _RunningStats:
    """Welford's online algorithm for per-dim mean/std."""
    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.ones(dim, dtype=np.float64)  # init 1.0 avoids div-by-0

    def update(self, x: np.ndarray):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean) + 1e-8
        return np.sqrt(self.M2 / max(self.n - 1, 1)) + 1e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return x.astype(np.float32)
        return ((x - self.mean) / self.std()).astype(np.float32)


# =============================================================================
# Manager buffer
# =============================================================================

class ManagerBuffer:
    def __init__(self, capacity: int,
                 z_dim: int,
                 proprio_dim: int,
                 n_tasks: int,
                 max_goal_dim: int,
                 z_dtype=np.float16):
        self.capacity = capacity
        task_state_dim = n_tasks * max_goal_dim

        self.z = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.task_state = np.zeros((capacity, task_state_dim), dtype=np.float32)
        self.completion = np.zeros((capacity, n_tasks), dtype=np.float32)

        self.action = np.zeros(capacity, dtype=np.int64)         # chosen task id
        self.reward = np.zeros(capacity, dtype=np.float32)

        self.z_next = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.proprio_next = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.task_state_next = np.zeros((capacity, task_state_dim), dtype=np.float32)
        self.completion_next = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, z, proprio, task_state, completion,
            action, reward,
            z_next, proprio_next, task_state_next, completion_next, done):
        i = self.ptr
        self.z[i] = z
        self.proprio[i] = proprio
        self.task_state[i] = task_state
        self.completion[i] = completion
        self.action[i] = int(action)
        self.reward[i] = float(reward)
        self.z_next[i] = z_next
        self.proprio_next[i] = proprio_next
        self.task_state_next[i] = task_state_next
        self.completion_next[i] = completion_next
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'z':                self.z[idx].astype(np.float32),
            'proprio':          self.proprio[idx],
            'task_state':       self.task_state[idx],
            'completion':       self.completion[idx],
            'action':           self.action[idx],
            'reward':           self.reward[idx],
            'z_next':           self.z_next[idx].astype(np.float32),
            'proprio_next':     self.proprio_next[idx],
            'task_state_next':  self.task_state_next[idx],
            'completion_next':  self.completion_next[idx],
            'done':             self.done[idx],
        }

    def __len__(self):
        return self.size


# =============================================================================
# Worker buffer
# =============================================================================

class WorkerBuffer:
    def __init__(self,
                 capacity: int,
                 z_dim: int,
                 proprio_dim: int,
                 action_dim: int,
                 action_chunk_len: int,
                 max_goal_dim: int,
                 n_tasks: int,
                 z_dtype=np.float16):
        self.capacity = capacity
        self.action_dim = action_dim
        self.H = action_chunk_len
        self.max_goal_dim = max_goal_dim
        self.n_tasks = n_tasks

        self.z = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.task_target = np.zeros((capacity, max_goal_dim), dtype=np.float32)
        self.task_cur = np.zeros((capacity, max_goal_dim), dtype=np.float32)
        self.task_mask = np.zeros((capacity, max_goal_dim), dtype=np.float32)
        self.task_id = np.zeros(capacity, dtype=np.int64)

        # Action chunk stored as flat (H*A)-vector
        self.action = np.zeros((capacity, action_dim * action_chunk_len),
                               dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)

        self.z_next = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.proprio_next = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.task_cur_next = np.zeros((capacity, max_goal_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

        self.proprio_stats = _RunningStats(proprio_dim)

    # -------------------- write / sample --------------------

    def observe_proprio(self, proprio):
        self.proprio_stats.update(np.asarray(proprio, dtype=np.float64))

    def observe_proprio_batch(self, proprio_batch):
        for row in np.asarray(proprio_batch):
            self.observe_proprio(row)

    def add(self, z, proprio, task_target, task_cur, task_mask, task_id,
            action_flat, reward,
            z_next, proprio_next, task_cur_next, done):
        i = self.ptr
        self.z[i] = z
        self.proprio[i] = proprio
        self.task_target[i] = task_target
        self.task_cur[i] = task_cur
        self.task_mask[i] = task_mask
        self.task_id[i] = int(task_id)
        self.action[i] = action_flat
        self.reward[i] = float(reward)
        self.z_next[i] = z_next
        self.proprio_next[i] = proprio_next
        self.task_cur_next[i] = task_cur_next
        self.done[i] = float(done)
        self.observe_proprio(proprio)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        std = self.proprio_stats.std()
        m = self.proprio_stats.mean
        return {
            'z':             self.z[idx].astype(np.float32),
            'proprio':       ((self.proprio[idx] - m) / std).astype(np.float32),
            'task_target':   self.task_target[idx],
            'task_cur':      self.task_cur[idx],
            'task_mask':     self.task_mask[idx],
            'task_id':       self.task_id[idx],
            'action':        self.action[idx],
            'reward':        self.reward[idx],
            'z_next':        self.z_next[idx].astype(np.float32),
            'proprio_next':  ((self.proprio_next[idx] - m) / std).astype(np.float32),
            'task_cur_next': self.task_cur_next[idx],
            'done':          self.done[idx],
        }

    def normalize_proprio(self, x: np.ndarray) -> np.ndarray:
        return self.proprio_stats.normalize(x)

    def __len__(self):
        return self.size
