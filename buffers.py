import numpy as np
from typing import Dict


class HighLevelBuffer:
    def __init__(self, capacity: int = 50_000, z_dim: int = 2048, n_tasks: int = 4, z_dtype=np.float16):
        self.capacity = capacity
        self.z_current = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.z_subgoal = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.z_next = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.task_id = np.zeros(capacity, dtype=np.int64)
        self.ptr = 0
        self.size = 0

    def add(self, z_t, z_subgoal, reward, z_next, done, task_id):
        i = self.ptr
        self.z_current[i] = z_t
        self.z_subgoal[i] = z_subgoal
        self.reward[i] = reward
        self.z_next[i] = z_next
        self.done[i] = float(done)
        self.task_id[i] = int(task_id)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'z_current': self.z_current[idx].astype(np.float32),
            'z_subgoal': self.z_subgoal[idx].astype(np.float32),
            'reward': self.reward[idx],
            'z_next': self.z_next[idx].astype(np.float32),
            'done': self.done[idx],
            'task_id': self.task_id[idx],
        }

    def __len__(self):
        return self.size


class LowLevelBuffer:
    def __init__(self, capacity: int = 300_000, z_dim: int = 2048, action_dim: int = 9,
                 proprio_dim: int = 59, n_tasks: int = 4, z_dtype=np.float16):
        self.capacity = capacity
        self.proprio_dim = proprio_dim
        self.n_tasks = n_tasks

        self.z_current = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.z_subgoal = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.task_id = np.zeros(capacity, dtype=np.int64)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.z_next = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.proprio_next = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.task_deltas = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.task_completed_delta = np.zeros(capacity, dtype=np.int8)

        self.ptr = 0
        self.size = 0

        self._p_mean = np.zeros(proprio_dim, dtype=np.float64)
        self._p_M2 = np.ones(proprio_dim, dtype=np.float64)
        self._p_n = 0

    def _update_proprio_stats(self, x: np.ndarray):
        self._p_n += 1
        delta = x - self._p_mean
        self._p_mean += delta / self._p_n
        delta2 = x - self._p_mean
        self._p_M2 += delta * delta2

    def _std(self):
        return np.sqrt(self._p_M2 / max(self._p_n - 1, 1)) + 1e-8

    def normalise_proprio(self, x: np.ndarray) -> np.ndarray:
        if self._p_n < 2:
            return x.astype(np.float32)
        return ((x - self._p_mean) / self._std()).astype(np.float32)

    def add(self, z_t, proprio_t, z_subgoal, task_id, action, reward, z_next, proprio_next,
            done, task_deltas, task_completed_delta):
        i = self.ptr
        self.z_current[i] = z_t
        self.proprio[i] = proprio_t
        self.z_subgoal[i] = z_subgoal
        self.task_id[i] = int(task_id)
        self.action[i] = action
        self.reward[i] = reward
        self.z_next[i] = z_next
        self.proprio_next[i] = proprio_next
        self.done[i] = float(done)
        self.task_deltas[i] = task_deltas
        self.task_completed_delta[i] = int(task_completed_delta)
        self._update_proprio_stats(np.asarray(proprio_t, dtype=np.float64))
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        std = self._std()
        return {
            'z_current': self.z_current[idx].astype(np.float32),
            'proprio': ((self.proprio[idx] - self._p_mean) / std).astype(np.float32),
            'z_subgoal': self.z_subgoal[idx].astype(np.float32),
            'task_id': self.task_id[idx],
            'action': self.action[idx],
            'reward': self.reward[idx],
            'z_next': self.z_next[idx].astype(np.float32),
            'proprio_next': ((self.proprio_next[idx] - self._p_mean) / std).astype(np.float32),
            'done': self.done[idx],
        }

    def get_landmark_data(self) -> Dict[str, np.ndarray]:
        return {
            'z': self.z_next[:self.size].astype(np.float32),
            'proprio': self.proprio_next[:self.size].copy(),
            'task_deltas': self.task_deltas[:self.size].copy(),
            'task_completed_delta': self.task_completed_delta[:self.size].copy(),
        }

    def __len__(self):
        return self.size
