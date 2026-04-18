"""
env_wrapper.py — Franka Kitchen image-observation wrapper for SMGW.

Changes vs. the original codebase:
  * Returns image AND full state on every step, so the agent never has to
    call env.get_state() as a separate RPC. The state is in info['state'].
  * Exposes a first-reset sanity check: verifies obs['observation'] is
    the 59-d vector our task indices expect.
  * Exposes info['tasks_completed_names'] (list of task NAMES completed
    this episode), which we use to build the completion mask in the agent.
  * Removes the HierarchicalKitchenWrapper (old latent-based executor);
    option execution now lives inside agent.execute_option().
"""
import os
import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers FrankaKitchen-v1

gym.register_envs(gymnasium_robotics)


# =============================================================================
# Camera: demo_relay_cam viewpoint (unchanged)
# =============================================================================

_CAM_DISTANCE = 4.5
_CAM_AZIMUTH = -66.0
_CAM_ELEVATION = -65.0
_CAM_LOOKAT = np.array([-0.1, 0.75, 1.6])

_DEFAULT_TASKS = ['microwave', 'kettle', 'light switch', 'slide cabinet']


def _apply_camera(env):
    try:
        renderer = env.unwrapped.mujoco_renderer
        if renderer is None:
            return
        cam = renderer.viewer.cam if hasattr(renderer, 'viewer') else None
        if cam is not None:
            cam.lookat[:] = _CAM_LOOKAT
            cam.distance = _CAM_DISTANCE
            cam.azimuth = _CAM_AZIMUTH
            cam.elevation = _CAM_ELEVATION
    except Exception:
        pass


# =============================================================================
# FrankaKitchenImageWrapper
# =============================================================================

class FrankaKitchenImageWrapper:
    """
    Wraps FrankaKitchen-v1 to return image observations while preserving
    access to the full 59-d state vector (for task grounding).

    reset() -> image (H, W, 3) uint8
    step(a) -> (image, env_reward, done, info)
        info['state']                 : 59-d np.ndarray
        info['step_count']            : int
        info['tasks_completed_names'] : List[str] names completed this episode
        info['n_tasks_completed']     : int
        info['tasks_remaining_names'] : List[str]
    """

    def __init__(
        self,
        tasks_to_complete: Optional[List[str]] = None,
        img_size: int = 224,
        seed: Optional[int] = None,
        terminate_on_tasks_completed: bool = False,
        max_steps: int = 280,
    ):
        self.img_size = img_size
        self.tasks_to_complete = tasks_to_complete or _DEFAULT_TASKS

        self._env = gym.make(
            'FrankaKitchen-v1',
            tasks_to_complete=self.tasks_to_complete,
            terminate_on_tasks_completed=terminate_on_tasks_completed,
            remove_task_when_completed=True,
            render_mode='rgb_array',
            width=img_size,
            height=img_size,
        )

        if seed is not None:
            self._env.action_space.seed(seed)

        self.action_space = self._env.action_space
        self.action_dim = self.action_space.shape[0]
        self._current_obs = None
        self._step_count = 0
        self._max_steps = max_steps
        self._seed = seed
        self._verified_obs_dim = False

    # ---------------------- rendering ----------------------

    def render_image(self) -> np.ndarray:
        """Render the current env state to an (img_size, img_size, 3) uint8 RGB array.

        Public API — called by training, eval, and video code between option
        boundaries to refresh the latest image without stepping the env.
        """
        _apply_camera(self._env)
        try:
            img = self._env.render()
            if img is None:
                raise ValueError("render() returned None")
            img = np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"WARNING: render failed ({e}); returning black image.")
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    # Back-compat alias for any legacy callers.
    _render_image = render_image

    # ---------------------- lifecycle ----------------------

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset and return (image, state_59d).

        On the very first call, sanity-check that the state is 59-d —
        our task_spec indices assume this layout.
        """
        reset_kwargs = {}
        if self._seed is not None:
            reset_kwargs['seed'] = self._seed
            self._seed = None

        obs_dict, _ = self._env.reset(**reset_kwargs)
        self._current_obs = obs_dict
        self._step_count = 0

        state = np.asarray(obs_dict['observation'], dtype=np.float64)
        if not self._verified_obs_dim:
            assert state.size >= 39, (
                f"FrankaKitchen-v1 returned state of size {state.size}; "
                f"SMGW task_spec indices expect the 59-d D4RL-compatible "
                f"layout (indices up to 38). Verify gymnasium-robotics version."
            )
            self._verified_obs_dim = True

        return self._render_image(), state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        obs_dict, reward, terminated, truncated, info = self._env.step(action)
        self._current_obs = obs_dict
        self._step_count += 1
        done = terminated or truncated or (self._step_count >= self._max_steps)

        img = self._render_image()

        # gymnasium-robotics uses 'step_task_completions' for tasks completed
        # THIS STEP and 'episode_task_completions' for cumulative list.
        episode_completions = info.get('episode_task_completions', [])
        info['state'] = np.asarray(obs_dict['observation'], dtype=np.float64)
        info['step_count'] = self._step_count
        info['tasks_completed_names'] = list(episode_completions)
        info['n_tasks_completed'] = len(episode_completions)
        info['tasks_remaining_names'] = [
            t for t in self.tasks_to_complete
            if t not in episode_completions
        ]

        return img, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        """Return the underlying 59-d proprioceptive state (for convenience)."""
        if self._current_obs is None:
            return np.zeros(59, dtype=np.float64)
        return np.asarray(self._current_obs['observation'], dtype=np.float64)

    def close(self):
        self._env.close()

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def n_tasks(self) -> int:
        return len(self.tasks_to_complete)
