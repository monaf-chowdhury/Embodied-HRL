from typing import Dict

import numpy as np


class RunningNorm:
    def __init__(self, dim: int):
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.ones(dim, dtype=np.float64)
        self.n = 0

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> np.ndarray:
        return np.sqrt(self.M2 / max(self.n - 1, 1)) + 1e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.n < 2:
            return x.astype(np.float32)
        return ((x - self.mean) / self.std()).astype(np.float32)


class ManagerReplayBuffer:
    def __init__(self, capacity: int, z_dim: int, proprio_dim: int, n_tasks: int, z_dtype=np.float16):
        self.capacity = capacity
        self.z = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.progress = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.errors = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.completion = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.remaining = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.prototype_sims = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.prev_task = np.full(capacity, -1, dtype=np.int64)

        self.task_id = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_z = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.next_proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.next_progress = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.next_errors = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.next_completion = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.next_remaining = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.next_prototype_sims = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.proprio_norm = RunningNorm(proprio_dim)

    def add(
        self,
        z,
        proprio,
        progress,
        errors,
        completion,
        remaining,
        prototype_sims,
        prev_task,
        task_id,
        reward,
        next_z,
        next_proprio,
        next_progress,
        next_errors,
        next_completion,
        next_remaining,
        next_prototype_sims,
        done,
    ):
        i = self.ptr
        self.z[i] = z
        self.proprio[i] = proprio
        self.progress[i] = progress
        self.errors[i] = errors
        self.completion[i] = completion
        self.remaining[i] = remaining
        self.prototype_sims[i] = prototype_sims
        self.prev_task[i] = int(prev_task)
        self.task_id[i] = int(task_id)
        self.reward[i] = reward
        self.next_z[i] = next_z
        self.next_proprio[i] = next_proprio
        self.next_progress[i] = next_progress
        self.next_errors[i] = next_errors
        self.next_completion[i] = next_completion
        self.next_remaining[i] = next_remaining
        self.next_prototype_sims[i] = next_prototype_sims
        self.done[i] = float(done)

        self.proprio_norm.update(np.asarray(proprio, dtype=np.float64))
        self.proprio_norm.update(np.asarray(next_proprio, dtype=np.float64))

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'z': self.z[idx].astype(np.float32),
            'proprio': self.proprio_norm.normalize(self.proprio[idx]),
            'progress': self.progress[idx],
            'errors': self.errors[idx],
            'completion': self.completion[idx],
            'remaining': self.remaining[idx],
            'prototype_sims': self.prototype_sims[idx],
            'prev_task': self.prev_task[idx],
            'task_id': self.task_id[idx],
            'reward': self.reward[idx],
            'next_z': self.next_z[idx].astype(np.float32),
            'next_proprio': self.proprio_norm.normalize(self.next_proprio[idx]),
            'next_progress': self.next_progress[idx],
            'next_errors': self.next_errors[idx],
            'next_completion': self.next_completion[idx],
            'next_remaining': self.next_remaining[idx],
            'next_prototype_sims': self.next_prototype_sims[idx],
            'done': self.done[idx],
        }

    def __len__(self):
        return self.size


class WorkerReplayBuffer:
    def __init__(
        self,
        capacity: int,
        z_dim: int,
        action_dim: int,
        proprio_dim: int,
        n_tasks: int,
        task_goal_dim: int,
        z_dtype=np.float16,
    ):
        self.capacity = capacity
        self.z = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.task_id = np.zeros(capacity, dtype=np.int64)
        self.target = np.zeros((capacity, task_goal_dim), dtype=np.float32)
        self.value = np.zeros((capacity, task_goal_dim), dtype=np.float32)
        self.error_vec = np.zeros((capacity, task_goal_dim), dtype=np.float32)
        self.progress = np.zeros(capacity, dtype=np.float32)
        self.completion = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_z = np.zeros((capacity, z_dim), dtype=z_dtype)
        self.next_proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.next_value = np.zeros((capacity, task_goal_dim), dtype=np.float32)
        self.next_error_vec = np.zeros((capacity, task_goal_dim), dtype=np.float32)
        self.next_progress = np.zeros(capacity, dtype=np.float32)
        self.next_completion = np.zeros((capacity, n_tasks), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.proprio_norm = RunningNorm(proprio_dim)

    def add(
        self,
        z,
        proprio,
        task_id,
        target,
        value,
        error_vec,
        progress,
        completion,
        action,
        reward,
        next_z,
        next_proprio,
        next_value,
        next_error_vec,
        next_progress,
        next_completion,
        done,
    ):
        i = self.ptr
        self.z[i] = z
        self.proprio[i] = proprio
        self.task_id[i] = int(task_id)
        self.target[i] = target
        self.value[i] = value
        self.error_vec[i] = error_vec
        self.progress[i] = progress
        self.completion[i] = completion
        self.action[i] = action
        self.reward[i] = reward
        self.next_z[i] = next_z
        self.next_proprio[i] = next_proprio
        self.next_value[i] = next_value
        self.next_error_vec[i] = next_error_vec
        self.next_progress[i] = next_progress
        self.next_completion[i] = next_completion
        self.done[i] = float(done)

        self.proprio_norm.update(np.asarray(proprio, dtype=np.float64))
        self.proprio_norm.update(np.asarray(next_proprio, dtype=np.float64))

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'z': self.z[idx].astype(np.float32),
            'proprio': self.proprio_norm.normalize(self.proprio[idx]),
            'task_id': self.task_id[idx],
            'target': self.target[idx],
            'value': self.value[idx],
            'error_vec': self.error_vec[idx],
            'progress': self.progress[idx],
            'completion': self.completion[idx],
            'action': self.action[idx],
            'reward': self.reward[idx],
            'next_z': self.next_z[idx].astype(np.float32),
            'next_proprio': self.proprio_norm.normalize(self.next_proprio[idx]),
            'next_value': self.next_value[idx],
            'next_error_vec': self.next_error_vec[idx],
            'next_progress': self.next_progress[idx],
            'next_completion': self.next_completion[idx],
            'done': self.done[idx],
        }

    def __len__(self):
        return self.size
