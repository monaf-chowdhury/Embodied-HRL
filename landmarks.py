"""
Landmark Buffer: Farthest Point Sampling in latent space.

Maintains a set of N landmark latent vectors sampled from replay,
ensuring good coverage of the explored latent space.
The manager selects subgoals from this set — never generates arbitrary z.
"""
import numpy as np
from typing import Optional, Tuple
from collections import defaultdict


class LandmarkBuffer:
    """
    Maintains N landmarks via Farthest Point Sampling (FPS) over
    observed latent vectors. Tracks visit counts for exploration.
    
    Usage:
        landmarks = LandmarkBuffer(n_landmarks=100, z_dim=64)
        landmarks.update(all_z_from_replay)  # Recompute landmarks
        idx = landmarks.select_explore()     # Least-visited landmark
        z_goal = landmarks.get(idx)          # Get landmark vector
        landmarks.record_visit(idx)          # Track visits
    """
    
    def __init__(self, n_landmarks: int = 100, z_dim: int = 64):
        self.n_landmarks = n_landmarks
        self.z_dim = z_dim
        
        # Landmark storage
        self.landmarks = np.zeros((n_landmarks, z_dim), dtype=np.float32)
        self.n_active = 0  # How many landmarks are currently set
        
        # Visit counts for exploration
        self.visit_counts = np.zeros(n_landmarks, dtype=np.int64)
        
        # Success/failure statistics per landmark (for debugging)
        self.success_counts = np.zeros(n_landmarks, dtype=np.int64)
        self.attempt_counts = np.zeros(n_landmarks, dtype=np.int64)
        
        self._is_initialized = False
    
    def update(self, z_observations: np.ndarray, keep_visits: bool = True):
        """
        Recompute landmarks using FPS over a set of observed latents.
        
        Args:
            z_observations: (M, z_dim) array of all observed latent vectors
            keep_visits: If True, transfer visit counts to nearest new landmarks
        """
        M = z_observations.shape[0]
        if M < self.n_landmarks:
            # Not enough observations yet — use all of them
            self.landmarks[:M] = z_observations
            self.n_active = M
            self._is_initialized = M > 0
            return
        
        # Farthest Point Sampling
        selected_indices = self._fps(z_observations, self.n_landmarks)
        new_landmarks = z_observations[selected_indices]
        
        if keep_visits and self._is_initialized and self.n_active > 0:
            # Transfer visit counts: for each new landmark, inherit counts
            # from the nearest old landmark
            old_landmarks = self.landmarks[:self.n_active]
            new_visits = np.zeros(self.n_landmarks, dtype=np.int64)
            new_success = np.zeros(self.n_landmarks, dtype=np.int64)
            new_attempts = np.zeros(self.n_landmarks, dtype=np.int64)
            
            for i in range(self.n_landmarks):
                dists = np.linalg.norm(old_landmarks - new_landmarks[i], axis=1)
                nearest = np.argmin(dists)
                new_visits[i] = self.visit_counts[nearest]
                new_success[i] = self.success_counts[nearest]
                new_attempts[i] = self.attempt_counts[nearest]
            
            self.visit_counts = new_visits
            self.success_counts = new_success
            self.attempt_counts = new_attempts
        else:
            self.visit_counts = np.zeros(self.n_landmarks, dtype=np.int64)
            self.success_counts = np.zeros(self.n_landmarks, dtype=np.int64)
            self.attempt_counts = np.zeros(self.n_landmarks, dtype=np.int64)
        
        self.landmarks = new_landmarks
        self.n_active = self.n_landmarks
        self._is_initialized = True
    
    def _fps(self, points: np.ndarray, n_select: int) -> np.ndarray:
        """
        Farthest Point Sampling: greedily select points that maximize
        minimum distance to already-selected points.
        
        Args:
            points: (M, d) array
            n_select: number of points to select
        Returns:
            (n_select,) array of indices into points
        """
        M = points.shape[0]
        selected = np.zeros(n_select, dtype=np.int64)
        
        # Start with a random point
        selected[0] = np.random.randint(M)
        
        # Track minimum distance from each point to any selected point
        min_dists = np.full(M, np.inf)
        
        for i in range(1, n_select):
            # Update min distances with the last selected point
            last = points[selected[i - 1]]
            dists = np.linalg.norm(points - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            
            # Select the point with the largest minimum distance
            # (Exclude already selected by setting their dist to -1)
            min_dists[selected[:i]] = -1
            selected[i] = np.argmax(min_dists)
        
        return selected
    
    def get(self, idx: int) -> np.ndarray:
        """Get landmark vector by index."""
        assert idx < self.n_active, f"Landmark {idx} not active (only {self.n_active} active)"
        return self.landmarks[idx].copy()
    
    def get_all(self) -> np.ndarray:
        """Get all active landmarks. Shape: (n_active, z_dim)."""
        return self.landmarks[:self.n_active].copy()
    
    def find_nearest(self, z: np.ndarray) -> Tuple[int, float]:
        """Find nearest landmark to z. Returns (index, distance)."""
        dists = np.linalg.norm(self.landmarks[:self.n_active] - z, axis=1)
        idx = np.argmin(dists)
        return int(idx), float(dists[idx])
    
    def select_explore(self) -> int:
        """Select least-visited landmark (for exploration)."""
        if self.n_active == 0:
            return 0
        # Among active landmarks, pick the one with fewest visits
        counts = self.visit_counts[:self.n_active]
        return int(np.argmin(counts))
    
    def record_visit(self, idx: int, success: bool = False):
        """Record that landmark idx was attempted."""
        self.visit_counts[idx] += 1
        self.attempt_counts[idx] += 1
        if success:
            self.success_counts[idx] += 1
    
    def get_success_rate(self, idx: int) -> float:
        """Get empirical success rate for a landmark."""
        if self.attempt_counts[idx] == 0:
            return 0.5  # Unknown — neutral prior
        return self.success_counts[idx] / self.attempt_counts[idx]
    
    @property
    def is_ready(self) -> bool:
        return self._is_initialized and self.n_active > 0
