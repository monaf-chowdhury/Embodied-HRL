"""
utils.py — Small shared helpers (image/video saving).

This module is the SINGLE SOURCE OF TRUTH for:
  - Which indices into the 59-d Franka Kitchen state vector belong to each task
  - The benchmark's own goal values for each task
  - Per-task distance metrics (how "close to done" is measured)
  - Per-task success epsilons (when the worker is "close enough")
  - Frozen text embeddings of the task names (for manager/worker conditioning)

Why this file exists:
  In the old codebase, the worker aimed at "some future image latent" and
  the hierarchy was graded on latent reachability. That is why chaining
  failed. The new codebase grades everything on the quantities below —
  the same quantities the benchmark uses to decide "task done" — so
  training signal = evaluation signal, by construction.

All code elsewhere in the repo should use `task_error(...)`,
`task_goal(...)`, etc., rather than hand-indexing into proprio.
"""
import os
import numpy as np
import cv2
from __future__ import annotations
import torch
from typing import Dict, List, Tuple, Optional
from transformers import CLIPTokenizer, CLIPTextModel

# =============================================================================
# Raw benchmark indices and goals — copied from Franka Kitchen relay-policy
# =============================================================================
# Each entry: task_name -> (indices into 59-d state, goal values).
# The benchmark's "completed" predicate is roughly: ||state[idx] - goal|| < ~0.3
# (gymnasium-robotics uses a hand-tuned per-task tolerance, reported via
#  info['episode_task_completions']). We mirror that with per-task epsilons
#  below, but we treat the benchmark's own completion bit as ground truth
#  and use the epsilons only as a SECONDARY early-termination signal.

_RAW_TASK_TABLE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    'bottom burner': (np.array([18, 19]),   np.array([-0.88, -0.01])),
    'top burner':    (np.array([24, 25]),   np.array([-0.92, -0.01])),
    'light switch':  (np.array([26, 27]),   np.array([-0.69, -0.05])),
    'slide cabinet': (np.array([28]),       np.array([0.37])),
    'hinge cabinet': (np.array([29, 30]),   np.array([0., 1.45])),
    'microwave':     (np.array([31]),       np.array([-0.75])),
    'kettle':        (np.array([32, 33, 34, 35, 36, 37, 38]),
                      np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06])),
}

# NOTE on indexing convention: these indices are into the 59-dimensional
# observation vector returned by FrankaKitchen-v1 (via obs_dict['observation']).
# Layout: indices 0-8 = 9 joint positions (arm + gripper), 9-17 = joint
# velocities, 18-38 = object qpos, 39-59 = object qvel. The object-qpos
# slice (indices 18-38) is where all our task targets live. This matches
# the D4RL v0 convention that the user's original codebase uses. A runtime
# sanity check in env_wrapper verifies obs dimension == 59 on first reset.

# Per-task success epsilons (in task-space units). Intentionally a little
# looser than the benchmark's internal tolerance so the worker gets a
# "close enough" early-termination signal that aligns with completion bits
# fired by the env.
_TASK_EPS: Dict[str, float] = {
    'bottom burner': 0.30,
    'top burner':    0.30,
    'light switch':  0.30,
    'slide cabinet': 0.10,
    'hinge cabinet': 0.40,
    'microwave':     0.20,
    'kettle':        0.30,
}

# Human-readable instruction strings used by the frozen semantic encoder.
# These feed into a frozen text encoder (CLIP) at init time to produce
# fixed task embeddings. The worker and manager see these embeddings as
# context — they are NOT trained.
_TASK_INSTRUCTION: Dict[str, str] = {
    'bottom burner': "turn the bottom burner knob",
    'top burner':    "turn the top burner knob",
    'light switch':  "flip the light switch on",
    'slide cabinet': "slide the cabinet door open",
    'hinge cabinet': "open the hinge cabinet",
    'microwave':     "open the microwave door",
    'kettle':        "move the kettle onto the stove",
}

# =============================================================================
# Handy constants for the outside world
# =============================================================================

ALL_KNOWN_TASKS: List[str] = list(_RAW_TASK_TABLE.keys())
DEFAULT_FOUR_TASKS: List[str] = ['microwave', 'kettle', 'light switch', 'slide cabinet']

# =============================================================================
# TaskSpec — frozen view of the tasks for one run
# =============================================================================

class TaskSpec:
    """
    Per-run frozen view of task metadata. Given the user's selected
    tasks_to_complete list (e.g. ['microwave', 'kettle', 'light switch',
    'slide cabinet']), this exposes:

        spec.n_tasks                       -> 4
        spec.names                         -> list[str]
        spec.indices(k)                    -> np.ndarray of state indices for task k
        spec.goal(k)                       -> np.ndarray of goal values
        spec.epsilon(k)                    -> float success tolerance
        spec.task_error(state, k)          -> scalar error (L2 by default)
        spec.task_errors_all(state)        -> vector of errors for all tasks
        spec.goal_vec_padded               -> (n_tasks, MAX_DIM) padded goal matrix
        spec.goal_mask_padded              -> (n_tasks, MAX_DIM) 1/0 mask
        spec.text_embeddings               -> (n_tasks, d_text) torch tensor (frozen)

    The padded views are used so the worker can consume a fixed-size
    target vector regardless of which task was chosen.
    """

    def __init__(self, task_names: List[str], device: str = "cpu"):
        for t in task_names:
            if t not in _RAW_TASK_TABLE:
                raise KeyError(
                    f"Unknown task '{t}'. Known tasks: {list(_RAW_TASK_TABLE)}"
                )
        self.names: List[str] = list(task_names)
        self.n_tasks: int = len(task_names)
        self.device = device

        self._idx: List[np.ndarray] = [
            _RAW_TASK_TABLE[t][0].copy() for t in task_names
        ]
        self._goal: List[np.ndarray] = [
            _RAW_TASK_TABLE[t][1].copy().astype(np.float32) for t in task_names
        ]
        self._eps: List[float] = [_TASK_EPS[t] for t in task_names]

        # Padded representations for fixed-width network inputs
        self.max_goal_dim: int = max(g.size for g in self._goal)
        padded_goal = np.zeros((self.n_tasks, self.max_goal_dim), dtype=np.float32)
        padded_mask = np.zeros((self.n_tasks, self.max_goal_dim), dtype=np.float32)
        for k, g in enumerate(self._goal):
            padded_goal[k, :g.size] = g
            padded_mask[k, :g.size] = 1.0
        self.goal_vec_padded: np.ndarray = padded_goal
        self.goal_mask_padded: np.ndarray = padded_mask

        # Text embeddings — filled in by attach_text_embeddings(); default is
        # a one-hot fallback so tests can run without a heavy model download.
        self.text_embeddings: torch.Tensor = torch.eye(self.n_tasks, device=device)
        self.text_embedding_dim: int = self.n_tasks
        self._text_source: str = "onehot-fallback"

    # ---------------------- basic accessors ----------------------

    def indices(self, k: int) -> np.ndarray:
        return self._idx[k]

    def goal(self, k: int) -> np.ndarray:
        return self._goal[k]

    def epsilon(self, k: int) -> float:
        return self._eps[k]

    def name(self, k: int) -> str:
        return self.names[k]

    def instruction(self, k: int) -> str:
        return _TASK_INSTRUCTION[self.names[k]]

    # ---------------------- task-space error ----------------------

    def task_state_slice(self, full_state: np.ndarray, k: int) -> np.ndarray:
        """Extract the task-relevant slice of the state vector for task k."""
        return full_state[self._idx[k]].astype(np.float32)

    def task_error(self, full_state: np.ndarray, k: int) -> float:
        """Scalar L2 distance from current task-slice to the goal."""
        cur = self.task_state_slice(full_state, k)
        goal = self._goal[k]
        return float(np.linalg.norm(cur - goal))

    def task_errors_all(self, full_state: np.ndarray) -> np.ndarray:
        """Vector of task errors for ALL tasks in this spec (shape [n_tasks])."""
        return np.array(
            [self.task_error(full_state, k) for k in range(self.n_tasks)],
            dtype=np.float32,
        )

    def is_close(self, full_state: np.ndarray, k: int) -> bool:
        """Is task k "close enough" by our per-task epsilon?"""
        return self.task_error(full_state, k) < self._eps[k]

    # ---------------------- padded target vector for the worker ----------------------

    def padded_goal_for(self, k: int) -> np.ndarray:
        """Goal padded to max_goal_dim (shape [max_goal_dim])."""
        return self.goal_vec_padded[k].copy()

    def padded_mask_for(self, k: int) -> np.ndarray:
        """Mask indicating which dims of the padded goal are active for task k."""
        return self.goal_mask_padded[k].copy()

    def padded_state_slice_for(self, full_state: np.ndarray, k: int) -> np.ndarray:
        """The task-state slice PADDED to max_goal_dim (zeros elsewhere)."""
        out = np.zeros(self.max_goal_dim, dtype=np.float32)
        cur = self.task_state_slice(full_state, k)
        out[:cur.size] = cur
        return out

    # ---------------------- completion-mask helpers ----------------------

    def completion_mask_from_names(self, completed_names: List[str]) -> np.ndarray:
        """
        Given the env's info['episode_task_completions'] (a list of NAMES),
        produce a binary completion mask aligned to THIS spec's task order.
        """
        completed_set = set(completed_names)
        return np.array(
            [1.0 if t in completed_set else 0.0 for t in self.names],
            dtype=np.float32,
        )

    # ---------------------- text embeddings ----------------------

    def attach_text_embeddings(self, embeddings: torch.Tensor, source: str):
        """
        Install frozen text embeddings produced once at init time.
        `embeddings` must have shape (n_tasks, d_text).
        """
        assert embeddings.shape[0] == self.n_tasks
        self.text_embeddings = embeddings.detach().to(self.device)
        self.text_embedding_dim = int(embeddings.shape[1])
        self._text_source = source

    @property
    def text_source(self) -> str:
        return self._text_source


# =============================================================================
# Frozen text embedder
# =============================================================================

def build_frozen_text_embeddings(task_names: List[str],
                                 device: str = "cuda") -> Tuple[torch.Tensor, str]:
    """
    Produce one frozen embedding per task instruction.

    Tries CLIP (openai/clip-vit-base-patch32) first; if HuggingFace is
    unavailable or offline, falls back to a deterministic hash-based
    embedding so training can still proceed. The embeddings are fixed
    throughout training.
    """
    instructions = [_TASK_INSTRUCTION[t] for t in task_names]

    # ---- Attempt 1: HuggingFace CLIP ----
    try:
        
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        mdl = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        mdl = mdl.to(device).eval()
        with torch.no_grad():
            batch = tok(instructions, padding=True, return_tensors="pt").to(device)
            out = mdl(**batch)
            # Use pooled output (CLS-equivalent) as the task embedding
            emb = out.pooler_output
        for p in mdl.parameters():
            p.requires_grad = False
        return emb.detach(), "clip-vit-base-patch32"
    except Exception as e:
        print(f"  [TextEmbed] CLIP unavailable ({type(e).__name__}: {e}). "
              f"Falling back to deterministic hash embeddings.")

    # ---- Attempt 2: deterministic hash → random projection ----
    # This is a stable, reproducible fallback. Different task strings map to
    # different embeddings, so the worker still gets a distinct conditioning
    # signal per task even without a pretrained text encoder.
    EMB_DIM = 64
    rng = np.random.RandomState(0x5EED)
    proj = rng.randn(256, EMB_DIM).astype(np.float32) / np.sqrt(256)
    embs = np.zeros((len(instructions), EMB_DIM), dtype=np.float32)
    for i, s in enumerate(instructions):
        h = np.zeros(256, dtype=np.float32)
        for j, ch in enumerate(s.encode("utf-8")):
            h[ch] += 1.0 + 0.01 * j
        h = h / (np.linalg.norm(h) + 1e-6)
        embs[i] = h @ proj
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-6)
    return torch.from_numpy(embs).to(device), "hash-fallback-64d"


def save_image(img: np.ndarray, path: str):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    cv2.imwrite(path, img[:, :, ::-1])   # RGB -> BGR for OpenCV


def save_video(frames: list, path: str, fps: int = 15):
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    H, W = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))

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
            gif_path, save_all=True, append_images=pil_frames[1:],
            loop=0, duration=int(1000 / fps),
        )
        print(f"  [Video] Saved GIF fallback: {gif_path}")
    except Exception as e:
        print(f"  [Video] WARNING: could not save video or GIF: {e}")


def format_time(seconds: float) -> str:
    import datetime
    return str(datetime.timedelta(seconds=int(seconds)))


def format_steps(s: int) -> str:
    if s >= 1_000_000:
        return f"{s/1e6:.2f}M"
    if s >= 1_000:
        return f"{s/1e3:.1f}k"
    return str(s)
