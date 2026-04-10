"""
Landmark Buffer: Farthest Point Sampling in latent space.

New capabilities vs. the original:
  1. Demo landmark seeding — extract latents from a demo GIF and mix them
     into every FPS update so early training has meaningful subgoals.
     Controlled by LandmarkConfig.use_demo_landmarks (ablation flag).

  2. Hindsight landmark injection — whenever the worker accidentally
     achieves a task completion, the caller adds that state's latent to
     the hindsight pool via add_success_state(). The pool is included in
     every landmark update, creating a positive feedback loop.

  3. Curriculum / approach states — instead of FPS over all replay
     uniformly, we first filter the replay to keep only states that are
     close (in proprio space) to the task goal but not yet at it, then
     run FPS over that filtered set. Falls back gracefully if fewer than
     n_landmarks candidates pass the filter.

  4. Recent-replay bias — only a configurable fraction of the FPS
     candidate pool comes from the most-recent half of replay, so old
     random-walk states do not permanently dominate.

Opinion on 4.3 (not coded, per request):
  A. Task-progress-biased landmarks: Strongly recommended. Filtering
     replay to transitions where n_tasks_completed increased or where
     task_distance decreased sharply is essentially supervised landmark
     discovery: you get landmarks on the path to task completion, not
     random-walk noise. The cost is storing per-transition task metadata
     in the low-level buffer, which is a minor change.

  B. Recent-replay bias: Also recommended but secondary. The real
     problem is not recency — it is *relevance*. If the agent is still
     mostly failing, recent replay is as bad as old replay. Combining
     (B) with (A) is better: recent + task-progress-biased.
     Implement (A) first; add (B) as a secondary knob once (A) is working.
"""
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image


# =============================================================================
# Demo GIF landmark extractor
# =============================================================================

def extract_demo_latents(
    gif_path: str,
    encoder,
    max_frames: int = 60,
    img_size: int = 224,
) -> np.ndarray:
    """
    Load a demo GIF, subsample up to max_frames frames, encode with the
    provided encoder, and return shape (N, proj_dim) float32 array.
    Returns empty array (0, proj_dim) on any failure.
    """
    try:
        gif = Image.open(gif_path)
    except Exception as e:
        print(f"  [Demo Landmarks] WARNING: could not open {gif_path}: {e}")
        return np.zeros((0, encoder.config.proj_dim), dtype=np.float32)

    frames = []
    try:
        idx = 0
        while True:
            gif.seek(idx)
            frame = gif.convert('RGB').resize((img_size, img_size))
            frames.append(np.array(frame, dtype=np.uint8))
            idx += 1
    except EOFError:
        pass

    if not frames:
        print("  [Demo Landmarks] WARNING: GIF contained 0 frames.")
        return np.zeros((0, encoder.config.proj_dim), dtype=np.float32)

    # Subsample evenly
    total = len(frames)
    if total > max_frames:
        indices = np.linspace(0, total - 1, max_frames, dtype=int)
        frames  = [frames[i] for i in indices]

    frames_np = np.stack(frames)                   # (N, H, W, 3) uint8
    z_demo    = encoder.encode_numpy(frames_np)    # (N, proj_dim)

    print(f"  [Demo Landmarks] Extracted {len(frames)} frames from {gif_path}")
    print(f"  [Demo Landmarks] Latent shape: {z_demo.shape}  "
          f"norm mean={np.linalg.norm(z_demo, axis=1).mean():.4f}")
    return z_demo.astype(np.float32)


# =============================================================================
# Landmark Buffer
# =============================================================================

class LandmarkBuffer:
    """
    Maintains N landmarks via FPS over observed latent vectors.
    Tracks visit counts for exploration.
    Optionally mixes in demo latents and hindsight success states.
    """

    def __init__(
        self,
        n_landmarks: int = 100,
        z_dim: int = 64,
        landmark_config=None,    # LandmarkConfig instance; None = defaults
    ):
        self.n_landmarks = n_landmarks
        self.z_dim = z_dim
        self.cfg   = landmark_config  # may be None if called without config

        # Landmark storage
        self.landmarks   = np.zeros((n_landmarks, z_dim), dtype=np.float32)
        self.n_active    = 0

        # Visit / success counts
        self.visit_counts   = np.zeros(n_landmarks, dtype=np.int64)
        self.success_counts = np.zeros(n_landmarks, dtype=np.int64)
        self.attempt_counts = np.zeros(n_landmarks, dtype=np.int64)

        # Demo latents (set once via seed_from_demo)
        self._demo_z: Optional[np.ndarray] = None

        # Hindsight success pool
        _pool = getattr(landmark_config, 'hindsight_pool_size', 500) if landmark_config else 500
        self._success_pool: np.ndarray = np.zeros((_pool, z_dim), dtype=np.float32)
        self._success_pool_ptr  = 0
        self._success_pool_size = 0
        self._pool_capacity     = _pool

        self._is_initialized = False

    # =========================================================================
    # Demo seeding
    # =========================================================================

    def seed_from_demo(self, encoder, gif_path: str, max_frames: int = 60):
        """
        Extract latents from the demo GIF and store them for mixing.
        Call once after encoder is ready (end of Phase 1 warmup).
        No-op if gif_path does not exist.
        """
        import os
        if not os.path.exists(gif_path):
            print(f"  [Demo Landmarks] GIF not found at {gif_path} — skipping demo seeding.")
            self._demo_z = None
            return

        self._demo_z = extract_demo_latents(
            gif_path, encoder,
            max_frames=max_frames,
            img_size=encoder.config.img_size,
        )

    # =========================================================================
    # Hindsight: add accidental task-completion states
    # =========================================================================

    def add_success_state(self, z: np.ndarray):
        """
        Add a latent vector from an accidental task-completion event to the
        hindsight pool.  The pool is a fixed-size circular buffer.
        """
        self._success_pool[self._success_pool_ptr] = z
        self._success_pool_ptr  = (self._success_pool_ptr + 1) % self._pool_capacity
        self._success_pool_size = min(self._success_pool_size + 1, self._pool_capacity)

    def _get_hindsight_pool(self) -> Optional[np.ndarray]:
        if self._success_pool_size == 0:
            return None
        return self._success_pool[:self._success_pool_size].copy()

    # =========================================================================
    # Main update
    # =========================================================================

    def update(
        self,
        z_observations: np.ndarray,
        keep_visits: bool = True,
        z_goal: Optional[np.ndarray] = None,   # for curriculum filter
    ):
        """
        Recompute landmarks using FPS over a filtered, mixed candidate set.

        Pipeline:
          1. Recent-replay bias: prefer later portion of z_observations.
          2. Curriculum filter: keep approach states near z_goal.
          3. Mix in demo latents (if seeded and flag on).
          4. Mix in hindsight success pool (if flag on).
          5. FPS over combined candidate set.
          6. Transfer visit counts to nearest new landmarks.
        """
        cfg = self.cfg

        # ---- Step 1: Recent-replay bias ----
        recent_frac = getattr(cfg, 'recent_replay_fraction', 0.7) if cfg else 0.7
        M = z_observations.shape[0]
        cutoff = max(int(M * (1.0 - recent_frac)), 1)
        # Take the last (recent_frac * M) observations
        replay_candidates = z_observations[cutoff:]

        # ---- Step 2: Curriculum filter — approach states ----
        use_curriculum = getattr(cfg, 'use_curriculum_landmarks', True) if cfg else True
        if use_curriculum and z_goal is not None and len(replay_candidates) > 10:
            replay_candidates = self._filter_approach_states(
                replay_candidates, z_goal,
                top_k_frac=getattr(cfg, 'curriculum_top_k', 0.3) if cfg else 0.3,
            )

        # ---- Step 3: Mix in demo latents ----
        parts = [replay_candidates]
        use_demo = getattr(cfg, 'use_demo_landmarks', True) if cfg else True
        if use_demo and self._demo_z is not None and len(self._demo_z) > 0:
            parts.append(self._demo_z)

        # ---- Step 4: Mix in hindsight pool ----
        use_hindsight = getattr(cfg, 'use_hindsight_landmarks', True) if cfg else True
        hindsight_pool = self._get_hindsight_pool()
        if use_hindsight and hindsight_pool is not None:
            parts.append(hindsight_pool)

        combined = np.concatenate(parts, axis=0)

        if len(combined) < self.n_landmarks:
            # Not enough candidates — use all
            self.landmarks[:len(combined)] = combined
            self.n_active = len(combined)
            self._is_initialized = len(combined) > 0
            return

        # ---- Step 5: FPS ----
        selected_indices = self._fps(combined, self.n_landmarks)
        new_landmarks = combined[selected_indices]

        # ---- Step 6: Transfer visit counts ----
        if keep_visits and self._is_initialized and self.n_active > 0:
            old_landmarks = self.landmarks[:self.n_active]
            new_visits    = np.zeros(self.n_landmarks, dtype=np.int64)
            new_success   = np.zeros(self.n_landmarks, dtype=np.int64)
            new_attempts  = np.zeros(self.n_landmarks, dtype=np.int64)
            for i in range(self.n_landmarks):
                dists   = np.linalg.norm(old_landmarks - new_landmarks[i], axis=1)
                nearest = np.argmin(dists)
                new_visits[i]   = self.visit_counts[nearest]
                new_success[i]  = self.success_counts[nearest]
                new_attempts[i] = self.attempt_counts[nearest]
            self.visit_counts   = new_visits
            self.success_counts = new_success
            self.attempt_counts = new_attempts
        else:
            self.visit_counts   = np.zeros(self.n_landmarks, dtype=np.int64)
            self.success_counts = np.zeros(self.n_landmarks, dtype=np.int64)
            self.attempt_counts = np.zeros(self.n_landmarks, dtype=np.int64)

        self.landmarks = new_landmarks
        self.n_active  = self.n_landmarks
        self._is_initialized = True

    # =========================================================================
    # Curriculum helper
    # =========================================================================

    def _filter_approach_states(
        self,
        z_obs: np.ndarray,
        z_goal: np.ndarray,
        top_k_frac: float = 0.3,
    ) -> np.ndarray:
        """
        Keep the top_k_frac fraction of z_obs that are closest to z_goal
        but exclude the very closest 5% (already-at-goal states are less
        useful as subgoals).
        """
        dists = np.linalg.norm(z_obs - z_goal[np.newaxis], axis=1)
        n     = len(dists)
        # Sort ascending; exclude bottom 5% (too close) and keep next top_k_frac
        sorted_idx = np.argsort(dists)
        near_cutoff = max(int(n * 0.05), 1)
        far_cutoff  = max(int(n * (0.05 + top_k_frac)), near_cutoff + 1)
        far_cutoff  = min(far_cutoff, n)
        approach_idx = sorted_idx[near_cutoff:far_cutoff]
        if len(approach_idx) < 10:
            return z_obs   # fallback: too few, use all
        return z_obs[approach_idx]

    # =========================================================================
    # FPS core
    # =========================================================================

    def _fps(self, points: np.ndarray, n_select: int) -> np.ndarray:
        M = points.shape[0]
        selected  = np.zeros(n_select, dtype=np.int64)
        selected[0] = np.random.randint(M)
        min_dists   = np.full(M, np.inf)

        for i in range(1, n_select):
            last = points[selected[i - 1]]
            dists = np.linalg.norm(points - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected[:i]] = -1
            selected[i] = np.argmax(min_dists)

        return selected

    # =========================================================================
    # Accessors
    # =========================================================================

    def get(self, idx: int) -> np.ndarray:
        assert idx < self.n_active, f"Landmark {idx} not active ({self.n_active} active)"
        return self.landmarks[idx].copy()

    def get_all(self) -> np.ndarray:
        return self.landmarks[:self.n_active].copy()

    def find_nearest(self, z: np.ndarray) -> Tuple[int, float]:
        dists = np.linalg.norm(self.landmarks[:self.n_active] - z, axis=1)
        idx   = np.argmin(dists)
        return int(idx), float(dists[idx])

    def select_explore(self) -> int:
        if self.n_active == 0:
            return 0
        return int(np.argmin(self.visit_counts[:self.n_active]))

    def record_visit(self, idx: int, success: bool = False):
        self.visit_counts[idx]   += 1
        self.attempt_counts[idx] += 1
        if success:
            self.success_counts[idx] += 1

    def get_success_rate(self, idx: int) -> float:
        if self.attempt_counts[idx] == 0:
            return 0.5
        return self.success_counts[idx] / self.attempt_counts[idx]

    @property
    def is_ready(self) -> bool:
        return self._is_initialized and self.n_active > 0