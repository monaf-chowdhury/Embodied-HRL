import os
import random

import cv2
import numpy as np
import torch

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from PIL import Image


TASK_SPECS: Dict[str, Dict] = {
    'bottom burner': {
        'indices': np.array([18, 19]),
        'goal': np.array([-0.88, -0.01], dtype=np.float32),
        'description': 'turn the bottom burner knob on',
        'success_tolerance': 0.14,
    },
    'top burner': {
        'indices': np.array([24, 25]),
        'goal': np.array([-0.92, -0.01], dtype=np.float32),
        'description': 'turn the top burner knob on',
        'success_tolerance': 0.14,
    },
    'light switch': {
        'indices': np.array([26, 27]),
        'goal': np.array([-0.69, -0.05], dtype=np.float32),
        'description': 'flip the light switch on',
        'success_tolerance': 0.14,
    },
    'slide cabinet': {
        'indices': np.array([28]),
        'goal': np.array([0.37], dtype=np.float32),
        'description': 'open the slide cabinet',
        'success_tolerance': 0.12,
    },
    'hinge cabinet': {
        'indices': np.array([29, 30]),
        'goal': np.array([0.0, 1.45], dtype=np.float32),
        'description': 'open the hinge cabinet',
        'success_tolerance': 0.16,
    },
    'microwave': {
        'indices': np.array([31]),
        'goal': np.array([-0.75], dtype=np.float32),
        'description': 'open the microwave door',
        'success_tolerance': 0.12,
    },
    'kettle': {
        'indices': np.array([32, 33, 34, 35, 36, 37, 38]),
        'goal': np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06], dtype=np.float32),
        'description': 'move the kettle to the top left burner',
        'success_tolerance': 0.18,
    },
}


DEFAULT_TASKS = ['microwave', 'kettle', 'light switch', 'slide cabinet']
MAX_TASK_GOAL_DIM = max(len(v['goal']) for v in TASK_SPECS.values())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_image(img: np.ndarray, path: str):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    cv2.imwrite(path, img[:, :, ::-1])


def save_video(frames: list, path: str, fps: int = 15):
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if writer.isOpened():
        for frame in frames:
            writer.write(frame[:, :, ::-1])
        writer.release()
        return
    writer.release()
    gif_path = os.path.splitext(path)[0] + '.gif'
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=int(1000 / fps),
        )
        print(f'  [Video] Saved GIF fallback: {gif_path}')
    except Exception as e:
        print(f'  [Video] WARNING: could not save video or GIF: {e}')




@dataclass
class DemoTaskPrototypes:
    enabled: bool
    prototypes: np.ndarray
    similarities_scale: float = 1.0

    def similarities(self, z: np.ndarray) -> np.ndarray:
        if not self.enabled or self.prototypes.size == 0:
            return np.zeros(0, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        zn = z / (np.linalg.norm(z) + 1e-8)
        pn = self.prototypes / (np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-8)
        sims = pn @ zn
        return (self.similarities_scale * sims).astype(np.float32)


def task_indices(task: str) -> np.ndarray:
    return TASK_SPECS[task]['indices']


def task_goal(task: str) -> np.ndarray:
    return TASK_SPECS[task]['goal'].copy()


def task_description(task: str) -> str:
    return TASK_SPECS[task]['description']


def task_success_tolerance(task: str) -> float:
    return float(TASK_SPECS[task]['success_tolerance'])


def task_value(proprio: np.ndarray, task: str) -> np.ndarray:
    idx = task_indices(task)
    return np.asarray(proprio[idx], dtype=np.float32)


def task_error_raw(proprio: np.ndarray, task: str) -> float:
    cur = task_value(proprio, task)
    goal = task_goal(task)
    return float(np.linalg.norm(cur - goal))


def task_error_normalized(proprio: np.ndarray, task: str) -> float:
    goal = task_goal(task)
    denom = float(np.linalg.norm(goal) + 1e-4)
    return task_error_raw(proprio, task) / denom


def per_task_errors(proprio: np.ndarray, tasks: List[str]) -> np.ndarray:
    return np.asarray([task_error_normalized(proprio, t) for t in tasks], dtype=np.float32)


def per_task_progress(proprio: np.ndarray, tasks: List[str]) -> np.ndarray:
    errors = per_task_errors(proprio, tasks)
    return np.maximum(0.0, 1.0 - errors).astype(np.float32)


def completed_mask_from_info(info: Dict, tasks: List[str]) -> np.ndarray:
    completed = set(info.get('episode_task_completions', []))
    return np.asarray([1.0 if t in completed else 0.0 for t in tasks], dtype=np.float32)


def sticky_completed_mask(mask_prev: np.ndarray, info: Dict, tasks: List[str]) -> np.ndarray:
    cur = completed_mask_from_info(info, tasks)
    if mask_prev is None:
        return cur
    return np.maximum(mask_prev, cur).astype(np.float32)


def remaining_mask(completed_mask: np.ndarray) -> np.ndarray:
    return (1.0 - np.asarray(completed_mask, dtype=np.float32)).astype(np.float32)


def newly_completed_count(prev_mask: np.ndarray, next_mask: np.ndarray) -> int:
    if prev_mask is None:
        return int(np.round(np.asarray(next_mask).sum()))
    delta = np.maximum(np.asarray(next_mask) - np.asarray(prev_mask), 0.0)
    return int(np.round(delta.sum()))


def task_transition_completed(prev_mask: np.ndarray, next_mask: np.ndarray, task_id: int) -> bool:
    prev_v = 0.0 if prev_mask is None else float(prev_mask[task_id])
    return prev_v < 0.5 and float(next_mask[task_id]) > 0.5


def pad_task_vector(vec: np.ndarray, max_dim: int = MAX_TASK_GOAL_DIM) -> np.ndarray:
    out = np.zeros(max_dim, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32)
    out[: len(vec)] = vec
    return out


def task_structured_vectors(proprio: np.ndarray, task: str, max_dim: int = MAX_TASK_GOAL_DIM) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cur = task_value(proprio, task)
    goal = task_goal(task)
    err = cur - goal
    return pad_task_vector(cur, max_dim), pad_task_vector(goal, max_dim), pad_task_vector(err, max_dim)


def all_task_goal_matrix(tasks: List[str], max_dim: int = MAX_TASK_GOAL_DIM) -> np.ndarray:
    return np.stack([pad_task_vector(task_goal(t), max_dim) for t in tasks], axis=0).astype(np.float32)


def heuristic_task_choice(progress: np.ndarray, remaining: np.ndarray, prototype_sims: Optional[np.ndarray] = None) -> int:
    scores = 1.0 - np.asarray(progress, dtype=np.float32)
    if prototype_sims is not None and len(prototype_sims) == len(scores):
        scores = scores - 0.10 * np.asarray(prototype_sims, dtype=np.float32)
    valid = np.where(np.asarray(remaining) > 0.5)[0]
    if len(valid) == 0:
        return int(np.argmax(scores))
    return int(valid[np.argmax(scores[valid])])


def hashed_text_embedding(text: str, dim: int = 32) -> np.ndarray:
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    seed = int(digest[:16], 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec


def build_task_language_embeddings(tasks: List[str], dim: int = 32) -> np.ndarray:
    embs = []
    for task in tasks:
        text = f"task: {task}. instruction: {task_description(task)}"
        embs.append(hashed_text_embedding(text, dim=dim))
    return np.stack(embs, axis=0).astype(np.float32)


def _load_gif_frames(gif_path: str, max_frames: int = 120, img_size: int = 224) -> List[np.ndarray]:
    if not os.path.exists(gif_path):
        return []
    gif = Image.open(gif_path)
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
    if len(frames) > max_frames:
        ids = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in ids]
    return frames


def build_demo_task_prototypes(
    gif_path: str,
    encoder,
    tasks: List[str],
    max_frames: int = 120,
) -> DemoTaskPrototypes:
    frames = _load_gif_frames(gif_path, max_frames=max_frames, img_size=encoder.config.img_size)
    if not frames:
        return DemoTaskPrototypes(enabled=False, prototypes=np.zeros((0, encoder.config.raw_dim), dtype=np.float32))

    z = encoder.encode_numpy(np.stack(frames)).astype(np.float32)
    n_tasks = len(tasks)
    segs = np.array_split(np.arange(len(z)), n_tasks)
    protos = []
    for seg in segs:
        if len(seg) == 0:
            protos.append(np.zeros(z.shape[1], dtype=np.float32))
            continue
        tail = seg[max(0, int(0.6 * len(seg))):]
        protos.append(z[tail].mean(axis=0).astype(np.float32))
    return DemoTaskPrototypes(enabled=True, prototypes=np.stack(protos, axis=0).astype(np.float32))
