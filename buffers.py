"""
Replay Buffers.

1. HighLevelFERBuffer: Frontier Experience Replay for the manager (SSE-style).
   Stores success, stop-on-failure, and partial-success transitions separately.
   
2. LowLevelBuffer: Standard replay buffer for worker SAC transitions.

3. ReachabilityBuffer: Stores (z_current, z_subgoal, success) for training f_ξ.
"""
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque


class HighLevelFERBuffer:
    """
    Frontier Experience Replay for the manager.
    
    Three transition types (following SSE):
    - Success: (z_t, z_goal, z_subgoal, cumulative_reward, z_next)
    - Stop-on-failure: (z_t, z_goal, z_subgoal, 0, terminal)
    - Partial-success: (z_t, z_goal, z_partial, partial_reward, z_partial_end)
    
    The manager trains on this buffer. Failures get zero reward and the
    episode terminates — this is the core credit assignment mechanism.
    """
    
    def __init__(self, capacity: int = 100_000, z_dim: int = 64):
        self.capacity = capacity
        self.z_dim = z_dim
        
        # Store all transitions in arrays
        self.z_current = np.zeros((capacity, z_dim), dtype=np.float32)
        self.z_goal = np.zeros((capacity, z_dim), dtype=np.float32)
        self.z_subgoal = np.zeros((capacity, z_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.z_next = np.zeros((capacity, z_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)  # 1.0 if terminal
        self.transition_type = np.zeros(capacity, dtype=np.int32)  # 0=success, 1=failure, 2=partial
        self.landmark_idx = np.zeros(capacity, dtype=np.int64)  # Which landmark was selected
        
        self.ptr = 0
        self.size = 0
    
    def add_success(self, z_t, z_goal, z_subgoal, cumulative_reward, z_next, landmark_idx):
        """Worker successfully reached the subgoal."""
        i = self.ptr
        self.z_current[i] = z_t
        self.z_goal[i] = z_goal
        self.z_subgoal[i] = z_subgoal
        self.reward[i] = cumulative_reward
        self.z_next[i] = z_next
        self.done[i] = 0.0  # Episode continues
        self.transition_type[i] = 0
        self.landmark_idx[i] = landmark_idx
        self._advance()
    
    def add_failure(self, z_t, z_goal, z_subgoal, landmark_idx):
        """Worker failed to reach subgoal — zero reward, terminal."""
        i = self.ptr
        self.z_current[i] = z_t
        self.z_goal[i] = z_goal
        self.z_subgoal[i] = z_subgoal
        self.reward[i] = 0.0  # Strict: zero reward on failure
        self.z_next[i] = z_t  # Next state is irrelevant (terminal)
        self.done[i] = 1.0  # Terminal
        self.transition_type[i] = 1
        self.landmark_idx[i] = landmark_idx
        self._advance()
    
    def add_partial(self, z_t, z_goal, z_partial_reached, partial_reward, 
                    z_partial_end, landmark_idx):
        """Worker made partial progress — store what was achieved."""
        i = self.ptr
        self.z_current[i] = z_t
        self.z_goal[i] = z_goal
        self.z_subgoal[i] = z_partial_reached  # Relabel to what was actually reached
        self.reward[i] = partial_reward
        self.z_next[i] = z_partial_end
        self.done[i] = 1.0  # Episode terminates on failure
        self.transition_type[i] = 2
        self.landmark_idx[i] = landmark_idx
        self._advance()
    
    def _advance(self):
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'z_current': self.z_current[indices],
            'z_goal': self.z_goal[indices],
            'z_subgoal': self.z_subgoal[indices],
            'reward': self.reward[indices],
            'z_next': self.z_next[indices],
            'done': self.done[indices],
            'transition_type': self.transition_type[indices],
            'landmark_idx': self.landmark_idx[indices],
        }
    
    def __len__(self):
        return self.size


class LowLevelBuffer:
    """
    Standard replay buffer for the worker (SAC).
    Stores: (z_t, z_subgoal, action, reward_shaped, z_next, done)
    """
    
    def __init__(self, capacity: int = 1_000_000, z_dim: int = 64, action_dim: int = 9):
        self.capacity = capacity
        
        self.z_current = np.zeros((capacity, z_dim), dtype=np.float32)
        self.z_subgoal = np.zeros((capacity, z_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.z_next = np.zeros((capacity, z_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, z_t, z_subgoal, action, reward, z_next, done):
        i = self.ptr
        self.z_current[i] = z_t
        self.z_subgoal[i] = z_subgoal
        self.action[i] = action
        self.reward[i] = reward
        self.z_next[i] = z_next
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'z_current': self.z_current[indices],
            'z_subgoal': self.z_subgoal[indices],
            'action': self.action[indices],
            'reward': self.reward[indices],
            'z_next': self.z_next[indices],
            'done': self.done[indices],
        }
    
    def get_all_z(self) -> np.ndarray:
        """Return all stored z_current values (for landmark computation)."""
        return self.z_current[:self.size].copy()
    
    def __len__(self):
        return self.size


class ReachabilityBuffer:
    """
    Stores (z_current, z_subgoal, success_label) for training the reachability predictor.
    """
    
    def __init__(self, capacity: int = 100_000, z_dim: int = 64):
        self.capacity = capacity
        
        self.z_current = np.zeros((capacity, z_dim), dtype=np.float32)
        self.z_subgoal = np.zeros((capacity, z_dim), dtype=np.float32)
        self.label = np.zeros(capacity, dtype=np.float32)  # 1.0 = success, 0.0 = failure
        
        self.ptr = 0
        self.size = 0
    
    def add(self, z_current, z_subgoal, success: bool):
        i = self.ptr
        self.z_current[i] = z_current
        self.z_subgoal[i] = z_subgoal
        self.label[i] = 1.0 if success else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample_balanced(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample with roughly equal positive/negative examples."""
        pos_mask = self.label[:self.size] > 0.5
        neg_mask = ~pos_mask
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]
        
        half = batch_size // 2
        
        if len(pos_indices) < half or len(neg_indices) < half:
            # Not enough of one class — sample uniformly
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            pos_sample = np.random.choice(pos_indices, size=half, replace=True)
            neg_sample = np.random.choice(neg_indices, size=batch_size - half, replace=True)
            indices = np.concatenate([pos_sample, neg_sample])
            np.random.shuffle(indices)
        
        return {
            'z_current': self.z_current[indices],
            'z_subgoal': self.z_subgoal[indices],
            'label': self.label[indices],
        }
    
    def __len__(self):
        return self.size
