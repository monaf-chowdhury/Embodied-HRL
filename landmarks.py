"""
Landmark Buffer: FPS in latent space with task-progress-biased candidate pool.

Changes from previous version:
  1. update() now accepts task_biased_z separately from generic replay z.
     Task-biased states (where task progress improved) are included in the
     FPS candidate pool alongside demo and hindsight latents.
  2. Curriculum filter removed — replaced by task_biased_z.
  3. extract_demo_latents uses encoder.config.raw_dim instead of proj_dim
     since there is no projection head anymore.
  4. FPS in 2048-d is O(n*d) per step — for 200 landmarks and ~50k candidates
     this is manageable. If it becomes slow, subsample candidates first.
"""
import numpy as np
import os
from typing import Optional, Tuple
from PIL import Image


def extract_demo_latents(
    gif_path: str,
    encoder,
    max_frames: int = 150,
    img_size: int = 224,
) -> np.ndarray:
    """
    Load demo GIF, encode all frames with the (now projectionless) encoder.
    Returns (N, raw_dim) float32. Same latent space as replay buffer.
    """
    try:
        gif = Image.open(gif_path)
    except Exception as e:
        print(f"  [Demo Landmarks] WARNING: could not open {gif_path}: {e}")
        return np.zeros((0, encoder.config.raw_dim), dtype=np.float32)

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
        return np.zeros((0, encoder.config.raw_dim), dtype=np.float32)

    total = len(frames)
    if total > max_frames:
        indices = np.linspace(0, total - 1, max_frames, dtype=int)
        frames  = [frames[i] for i in indices]

    frames_np = np.stack(frames)               # (N, H, W, 3) uint8
    z_demo    = encoder.encode_numpy(frames_np)  # (N, raw_dim) L2-normed

    print(f"  [Demo Landmarks] Extracted {len(frames)} frames from {gif_path}")
    print(f"  [Demo Landmarks] Shape: {z_demo.shape}  "
          f"norm mean={np.linalg.norm(z_demo, axis=1).mean():.4f}  "
          f"(should be ~1.0 for L2-normalised)")
    return z_demo.astype(np.float32)


class LandmarkBuffer:
    """
    Maintains N landmarks via FPS. Candidate pool = task-biased replay
    + demo latents + hindsight success pool.
    """

    def __init__(self, n_landmarks=200, z_dim=2048, landmark_config=None):
        self.n_landmarks = n_landmarks
        self.z_dim       = z_dim
        self.cfg         = landmark_config

        self.landmarks    = np.zeros((n_landmarks, z_dim), dtype=np.float32)
        self.n_active     = 0
        self.visit_counts   = np.zeros(n_landmarks, dtype=np.int64)
        self.success_counts = np.zeros(n_landmarks, dtype=np.int64)
        self.attempt_counts = np.zeros(n_landmarks, dtype=np.int64)

        self._demo_z: Optional[np.ndarray] = None

        _pool = getattr(landmark_config, 'hindsight_pool_size', 1000) if landmark_config else 1000
        self._success_pool     = np.zeros((_pool, z_dim), dtype=np.float32)
        self._success_pool_ptr  = 0
        self._success_pool_size = 0
        self._pool_capacity     = _pool

        self._is_initialized = False

    # =========================================================================
    # Demo seeding
    # =========================================================================

    def seed_from_demo(self, encoder, gif_path: str, max_frames: int = 150):
        if not os.path.exists(gif_path):
            print(f"  [Demo Landmarks] GIF not found at {gif_path} — skipping.")
            self._demo_z = None
            return
        self._demo_z = extract_demo_latents(
            gif_path, encoder, max_frames=max_frames,
            img_size=encoder.config.img_size,
        )

    # =========================================================================
    # Hindsight pool
    # =========================================================================

    def add_success_state(self, z: np.ndarray):
        """Add latent from task-completion event to hindsight pool."""
        self._success_pool[self._success_pool_ptr] = z
        self._success_pool_ptr  = (self._success_pool_ptr + 1) % self._pool_capacity
        self._success_pool_size = min(self._success_pool_size + 1, self._pool_capacity)

    def _get_hindsight_pool(self) -> Optional[np.ndarray]:
        if self._success_pool_size == 0:
            return None
        return self._success_pool[:self._success_pool_size].copy()

    # =========================================================================
    # Main update — task-progress-biased
    # =========================================================================

    def update(
        self,
        z_replay: np.ndarray,                    # generic replay latents
        z_task_biased: Optional[np.ndarray] = None,  # task-progress-biased latents
        keep_visits: bool = True,
    ):
        """
        FPS over: task-biased replay + demo + hindsight.
        Generic replay is only used as fallback if the biased pool is too small.

        Pipeline:
          1. Prefer task_biased_z (states where task progress improved).
          2. Mix in demo latents (same L2-normed R3M space now).
          3. Mix in hindsight success pool.
          4. If combined < n_landmarks, pad with generic replay.
          5. FPS over combined candidate set.
          6. Transfer visit counts.
        """
        cfg = self.cfg

        # ---- Step 1: Task-biased replay ----
        parts = []
        if z_task_biased is not None and len(z_task_biased) >= 10:
            parts.append(z_task_biased)
        else:
            # Fallback: recent generic replay
            recent_frac = getattr(cfg, 'recent_replay_fraction', 0.7) if cfg else 0.7
            M = z_replay.shape[0]
            cutoff = max(int(M * (1.0 - recent_frac)), 1)
            parts.append(z_replay[cutoff:])

        # ---- Step 2: Demo latents ----
        use_demo = getattr(cfg, 'use_demo_landmarks', True) if cfg else True
        if use_demo and self._demo_z is not None and len(self._demo_z) > 0:
            parts.append(self._demo_z)

        # ---- Step 3: Hindsight pool ----
        use_hindsight = getattr(cfg, 'use_hindsight_landmarks', True) if cfg else True
        hindsight = self._get_hindsight_pool()
        if use_hindsight and hindsight is not None:
            parts.append(hindsight)

        combined = np.concatenate(parts, axis=0)

        # ---- Step 4: Pad if needed ----
        if len(combined) < self.n_landmarks:
            combined = np.concatenate([combined, z_replay], axis=0)
        if len(combined) < self.n_landmarks:
            self.landmarks[:len(combined)] = combined
            self.n_active = len(combined)
            self._is_initialized = len(combined) > 0
            return

        # ---- Step 5: FPS ----
        selected = self._fps(combined, self.n_landmarks)
        new_landmarks = combined[selected]

        # ---- Step 6: Transfer visit counts ----
        if keep_visits and self._is_initialized and self.n_active > 0:
            old = self.landmarks[:self.n_active]
            new_v = np.zeros(self.n_landmarks, dtype=np.int64)
            new_s = np.zeros(self.n_landmarks, dtype=np.int64)
            new_a = np.zeros(self.n_landmarks, dtype=np.int64)
            for i in range(self.n_landmarks):
                dists   = np.linalg.norm(old - new_landmarks[i], axis=1)
                nearest = np.argmin(dists)
                new_v[i] = self.visit_counts[nearest]
                new_s[i] = self.success_counts[nearest]
                new_a[i] = self.attempt_counts[nearest]
            self.visit_counts   = new_v
            self.success_counts = new_s
            self.attempt_counts = new_a
        else:
            self.visit_counts   = np.zeros(self.n_landmarks, dtype=np.int64)
            self.success_counts = np.zeros(self.n_landmarks, dtype=np.int64)
            self.attempt_counts = np.zeros(self.n_landmarks, dtype=np.int64)

        self.landmarks = new_landmarks
        self.n_active  = self.n_landmarks
        self._is_initialized = True

    # =========================================================================
    # FPS
    # =========================================================================

    def _fps(self, points: np.ndarray, n_select: int) -> np.ndarray:
        M = points.shape[0]
        selected    = np.zeros(n_select, dtype=np.int64)
        selected[0] = np.random.randint(M)
        min_dists   = np.full(M, np.inf)
        for i in range(1, n_select):
            last      = points[selected[i - 1]]
            dists     = np.linalg.norm(points - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected[:i]] = -1
            selected[i] = np.argmax(min_dists)
        return selected

    # =========================================================================
    # Accessors
    # =========================================================================

    def get(self, idx: int) -> np.ndarray:
        assert idx < self.n_active
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
