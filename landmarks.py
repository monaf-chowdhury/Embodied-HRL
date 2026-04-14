import numpy as np
from typing import Optional, Dict
from PIL import Image

_TASK_OBS_IDX = {
    'bottom burner': (np.array([18, 19]),   np.array([-0.88, -0.01])),
    'top burner':    (np.array([24, 25]),   np.array([-0.92, -0.01])),
    'light switch':  (np.array([26, 27]),   np.array([-0.69, -0.05])),
    'slide cabinet': (np.array([28]),       np.array([0.37])),
    'hinge cabinet': (np.array([29, 30]),   np.array([0., 1.45])),
    'microwave':     (np.array([31]),       np.array([-0.75])),
    'kettle':        (np.array([32, 33, 34, 35, 36, 37, 38]),
                      np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06])),
}


def infer_task_id_from_proprio(proprio: np.ndarray, task_names) -> int:
    dists = []
    for t in task_names:
        idx, goal = _TASK_OBS_IDX[t]
        cur = proprio[idx]
        dist = np.linalg.norm(cur - goal) / (np.linalg.norm(goal) + 1e-4)
        dists.append(dist)
    return int(np.argmin(dists))


def extract_demo_latents(gif_path: str, encoder, max_frames: int = 60, img_size: int = 224) -> np.ndarray:
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
        return np.zeros((0, encoder.config.raw_dim), dtype=np.float32)

    if len(frames) > max_frames:
        ids = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in ids]

    return encoder.encode_numpy(np.stack(frames)).astype(np.float32)


class LandmarkBuffer:
    def __init__(self, n_landmarks: int = 128, z_dim: int = 2048, landmark_config=None, task_names=None):
        self.n_landmarks = n_landmarks
        self.z_dim = z_dim
        self.cfg = landmark_config
        self.task_names = task_names or ['microwave']

        self.landmarks = np.zeros((n_landmarks, z_dim), dtype=np.float32)
        self.task_ids = np.zeros(n_landmarks, dtype=np.int64)
        self.n_active = 0
        self.visit_counts = np.zeros(n_landmarks, dtype=np.int64)
        self.success_counts = np.zeros(n_landmarks, dtype=np.int64)
        self.attempt_counts = np.zeros(n_landmarks, dtype=np.int64)
        self._demo_z = None
        self._demo_task_ids = None
        pool = getattr(landmark_config, 'hindsight_pool_size', 1000) if landmark_config else 1000
        self._success_z = np.zeros((pool, z_dim), dtype=np.float32)
        self._success_task_ids = np.zeros(pool, dtype=np.int64)
        self._success_ptr = 0
        self._success_size = 0
        self._is_initialized = False

    def seed_from_demo(self, encoder, gif_path: str, max_frames: int = 60):
        import os
        if not os.path.exists(gif_path):
            print(f"  [Demo Landmarks] GIF not found at {gif_path} — skipping.")
            self._demo_z = None
            self._demo_task_ids = None
            return
        self._demo_z = extract_demo_latents(gif_path, encoder, max_frames=max_frames, img_size=encoder.config.img_size)
        self._demo_task_ids = -np.ones(len(self._demo_z), dtype=np.int64)
        print(f"  [Demo Landmarks] Extracted {len(self._demo_z)} frames from {gif_path}")

    def add_success_state(self, z: np.ndarray, task_id: int):
        self._success_z[self._success_ptr] = z
        self._success_task_ids[self._success_ptr] = int(task_id)
        self._success_ptr = (self._success_ptr + 1) % len(self._success_z)
        self._success_size = min(self._success_size + 1, len(self._success_z))

    def _get_success_pool(self):
        if self._success_size == 0:
            return None, None
        return self._success_z[:self._success_size].copy(), self._success_task_ids[:self._success_size].copy()

    def update(self, replay: Dict[str, np.ndarray], keep_visits: bool = True):
        z_obs = replay['z']
        proprios = replay['proprio']
        task_deltas = replay['task_deltas']
        completed_delta = replay['task_completed_delta']
        if len(z_obs) == 0:
            return

        recent_frac = getattr(self.cfg, 'recent_replay_fraction', 0.6) if self.cfg else 0.6
        cutoff = max(int(len(z_obs) * (1.0 - recent_frac)), 1)
        z_recent = z_obs[cutoff:]
        p_recent = proprios[cutoff:]
        td_recent = task_deltas[cutoff:]
        cd_recent = completed_delta[cutoff:]

        task_ids_recent = np.array([
            int(np.argmax(td)) if td.max() > 1e-6 else infer_task_id_from_proprio(p, self.task_names)
            for p, td in zip(p_recent, td_recent)
        ], dtype=np.int64)

        max_delta = td_recent.max(axis=1)
        percentile = getattr(self.cfg, 'task_delta_percentile', 85.0) if self.cfg else 85.0
        thr = np.percentile(max_delta, percentile) if len(max_delta) > 10 else 0.0
        priority_mask = (cd_recent > 0) | (max_delta >= thr)
        if priority_mask.sum() < 20:
            priority_mask = max_delta > 0

        z_priority = z_recent[priority_mask]
        task_ids_priority = task_ids_recent[priority_mask]
        z_general = z_recent
        task_ids_general = task_ids_recent

        parts_z, parts_task = [], []
        priority_fraction = getattr(self.cfg, 'priority_fraction', 0.5) if self.cfg else 0.5
        if len(z_priority) > 0:
            take = max(int(self.n_landmarks * 3 * priority_fraction), min(len(z_priority), self.n_landmarks))
            idx = np.random.choice(len(z_priority), size=min(take, len(z_priority)), replace=False)
            parts_z.append(z_priority[idx])
            parts_task.append(task_ids_priority[idx])
        if len(z_general) > 0:
            take = max(int(self.n_landmarks * 3 * (1.0 - priority_fraction)), self.n_landmarks)
            idx = np.random.choice(len(z_general), size=min(take, len(z_general)), replace=False)
            parts_z.append(z_general[idx])
            parts_task.append(task_ids_general[idx])

        if getattr(self.cfg, 'use_demo_landmarks', True) and self._demo_z is not None and len(self._demo_z) > 0:
            parts_z.append(self._demo_z)
            if len(z_priority) > 0:
                d = np.linalg.norm(self._demo_z[:, None, :] - z_priority[None, :, :], axis=-1)
                demo_task_ids = task_ids_priority[np.argmin(d, axis=1)]
            else:
                demo_task_ids = np.zeros(len(self._demo_z), dtype=np.int64)
            parts_task.append(demo_task_ids)

        success_z, success_task = self._get_success_pool()
        if getattr(self.cfg, 'use_hindsight_landmarks', True) and success_z is not None:
            parts_z.append(success_z)
            parts_task.append(success_task)

        combined_z = np.concatenate(parts_z, axis=0)
        combined_task = np.concatenate(parts_task, axis=0)

        if len(combined_z) <= self.n_landmarks:
            self.landmarks[:len(combined_z)] = combined_z
            self.task_ids[:len(combined_z)] = combined_task
            self.n_active = len(combined_z)
            self._is_initialized = self.n_active > 0
            return

        selected = self._fps(combined_z, self.n_landmarks)
        new_landmarks = combined_z[selected]
        new_task_ids = combined_task[selected]

        if keep_visits and self._is_initialized and self.n_active > 0:
            old_landmarks = self.landmarks[:self.n_active]
            new_visits = np.zeros(self.n_landmarks, dtype=np.int64)
            new_success = np.zeros(self.n_landmarks, dtype=np.int64)
            new_attempts = np.zeros(self.n_landmarks, dtype=np.int64)
            for i in range(self.n_landmarks):
                nearest = int(np.argmin(np.linalg.norm(old_landmarks - new_landmarks[i], axis=1)))
                new_visits[i] = self.visit_counts[nearest]
                new_success[i] = self.success_counts[nearest]
                new_attempts[i] = self.attempt_counts[nearest]
            self.visit_counts = new_visits
            self.success_counts = new_success
            self.attempt_counts = new_attempts
        else:
            self.visit_counts[:] = 0
            self.success_counts[:] = 0
            self.attempt_counts[:] = 0

        self.landmarks = new_landmarks
        self.task_ids = new_task_ids
        self.n_active = self.n_landmarks
        self._is_initialized = True

    def _fps(self, points: np.ndarray, n_select: int) -> np.ndarray:
        M = len(points)
        selected = np.zeros(n_select, dtype=np.int64)
        selected[0] = np.random.randint(M)
        min_dists = np.full(M, np.inf)
        for i in range(1, n_select):
            last = points[selected[i - 1]]
            d = np.linalg.norm(points - last, axis=1)
            min_dists = np.minimum(min_dists, d)
            min_dists[selected[:i]] = -1
            selected[i] = int(np.argmax(min_dists))
        return selected

    def get(self, idx: int) -> np.ndarray:
        return self.landmarks[idx].copy()

    def get_task_id(self, idx: int) -> int:
        return int(self.task_ids[idx])

    def get_all(self) -> np.ndarray:
        return self.landmarks[:self.n_active].copy()

    def get_all_task_ids(self) -> np.ndarray:
        return self.task_ids[:self.n_active].copy()

    def select_explore(self) -> int:
        if self.n_active == 0:
            return 0
        scores = self.visit_counts[:self.n_active] - 0.25 * self.success_counts[:self.n_active]
        return int(np.argmin(scores))

    def record_visit(self, idx: int, success: bool = False):
        self.visit_counts[idx] += 1
        self.attempt_counts[idx] += 1
        if success:
            self.success_counts[idx] += 1

    @property
    def is_ready(self) -> bool:
        return self._is_initialized and self.n_active > 0
