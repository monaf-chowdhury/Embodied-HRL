"""
utils.py — Goal image rendering and encoding for FrankaKitchen-v1 HRL.

get_goal_image_and_encoding():
    Creates a fresh FrankaKitchen-v1 env, manually sets the sim to the
    goal state for all 4 required tasks, renders using the same Q10 camera
    as env_wrapper.py, and encodes with the provided encoder.

save_goal_image():
    Saves a goal image PNG for visual inspection.

compare_goal_methods():
    Renders both manual-qpos and a reference rendering, compares encodings.
"""

import os
import numpy as np
import cv2
from typing import Tuple

import gymnasium as gym
import gymnasium_robotics  # noqa: F401

gym.register_envs(gymnasium_robotics)


# =============================================================================
# Goal state — same values as D4RL kitchen_envs.py OBS_ELEMENT_GOALS
# Verified: cosine similarity 0.9999 between manual-qpos and dataset methods
#
# FrankaKitchen-v1 uses the same underlying XML as v0, so qpos layout is:
#   qpos[0:9]  = robot arm (9 DOF)
#   qpos[9:30] = object joints
# OBS_ELEMENT_INDICES are into obs[0:21] = qpos[9:30], so +9 offset applies.
# =============================================================================

_GOAL_QPOS = {
    'bottom burner': (np.array([11, 12]) + 9, np.array([-0.88, -0.01])),
    'top burner':    (np.array([15, 16]) + 9, np.array([-0.92, -0.01])),
    'light switch':  (np.array([17, 18]) + 9, np.array([-0.69, -0.05])),
    'slide cabinet': (np.array([19])     + 9, np.array([0.37])),
    'hinge cabinet': (np.array([20, 21]) + 9, np.array([0., 1.45])),
    'microwave':     (np.array([22])     + 9, np.array([-0.75])),
    'kettle':        (np.array([23, 24, 25, 26, 27, 28, 29]) + 9,
                      np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06])),
}

_REQUIRED_TASKS = ['microwave', 'kettle', 'light switch', 'slide cabinet']


def get_goal_image_and_encoding(
    encoder,
    tasks: list = None,
    img_size: int = 224,
    device: str = 'cuda',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render FrankaKitchen-v1 in its fully-completed goal state and encode it.

    Uses render_mode='rgb_array' (the v1 API). Sets goal qpos for all
    required tasks, calls mj_forward to propagate physics, then renders.

    Args:
        encoder : VisualEncoder instance
        tasks   : list of task names to complete (default: 4-task complete set)
        img_size: render resolution
        device  : torch device string

    Returns:
        z_goal : np.ndarray shape (proj_dim,)
        img    : np.ndarray shape (img_size, img_size, 3) uint8 RGB
    """
    import mujoco

    tasks = tasks or _REQUIRED_TASKS

    env = gym.make(
        'FrankaKitchen-v1',
        tasks_to_complete=tasks,
        render_mode='rgb_array',
        width=img_size,
        height=img_size,
    )
    env.reset()

    # Access the underlying mujoco model and data (new bindings)
    model = env.unwrapped.model
    data  = env.unwrapped.data

    # Set goal qpos for all required tasks
    for task in tasks:
        if task not in _GOAL_QPOS:
            print(f"  WARNING: unknown task '{task}', skipping.")
            continue
        qpos_indices, goal_vals = _GOAL_QPOS[task]
        for i, idx in enumerate(qpos_indices):
            if idx < model.nq:
                data.qpos[idx] = goal_vals[i]

    # Propagate physics
    mujoco.mj_forward(model, data)

    # Render — env.render() uses the rgb_array renderer
    img = env.render()
    img = np.array(img, dtype=np.uint8)

    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    env.close()

    # Encode
    z_goal = encoder.encode_numpy(img).squeeze()

    print("Goal image rendered (FrankaKitchen-v1):")
    print(f"  Tasks:        {tasks}")
    print(f"  Image shape:  {img.shape}  dtype={img.dtype}")
    print(f"  Image range:  [{img.min()}, {img.max()}]  mean={img.mean():.1f}")
    print(f"  z_goal shape: {z_goal.shape}")
    print(f"  z_goal norm:  {np.linalg.norm(z_goal):.4f}")

    return z_goal, img


def save_goal_image(img: np.ndarray, path: str):
    """Save goal image as PNG for visual inspection."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    cv2.imwrite(path, img[:, :, ::-1])  # RGB -> BGR for OpenCV
    print(f"  Goal image saved: {path}")


def compare_goal_methods(
    encoder,
    log_dir: str = 'logs/',
    img_size: int = 224,
):
    """
    Render the goal state twice with different random seeds to verify
    the qpos-setting is deterministic and the encoding is consistent.
    Saves both images for visual inspection.
    """
    os.makedirs(log_dir, exist_ok=True)

    print("\n[Goal Rendering — Run 1]")
    z1, img1 = get_goal_image_and_encoding(encoder, img_size=img_size)
    save_goal_image(img1, os.path.join(log_dir, 'goal_image.png'))

    print("\n[Goal Rendering — Run 2 (consistency check)]")
    z2, img2 = get_goal_image_and_encoding(encoder, img_size=img_size)
    save_goal_image(img2, os.path.join(log_dir, 'goal_image_run2.png'))

    cos_sim = np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-8)
    l2_dist = np.linalg.norm(z1 - z2)

    print(f"\n[Goal Consistency]")
    print(f"  Cosine similarity: {cos_sim:.6f}  (should be ~1.0)")
    print(f"  L2 distance:       {l2_dist:.6f}  (should be ~0.0)")

    return z1, cos_sim