"""
Franka Kitchen v1 image wrapper for task-grounded hierarchical RL.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import gymnasium_robotics  # noqa: F401

gym.register_envs(gymnasium_robotics)

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


class FrankaKitchenImageWrapper:
    def __init__(
        self,
        tasks_to_complete: Optional[List[str]] = None,
        img_size: int = 224,
        seed: Optional[int] = None,
        terminate_on_tasks_completed: bool = False,
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
        self._max_steps = 280
        self._seed = seed

    def _render_image(self) -> np.ndarray:
        _apply_camera(self._env)
        try:
            img = self._env.render()
            if img is None:
                raise RuntimeError('render() returned None')
            img = np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f'WARNING: render failed ({e}), returning black image.')
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    def reset(self) -> np.ndarray:
        kwargs = {}
        if self._seed is not None:
            kwargs['seed'] = self._seed
            self._seed = None
        obs_dict, _ = self._env.reset(**kwargs)
        self._current_obs = obs_dict
        self._step_count = 0
        return self._render_image()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        obs_dict, reward, terminated, truncated, info = self._env.step(action)
        self._current_obs = obs_dict
        self._step_count += 1
        done = terminated or truncated or (self._step_count >= self._max_steps)
        img = self._render_image()

        episode_completions = info.get('episode_task_completions', [])
        info['state'] = obs_dict['observation']
        info['step_count'] = self._step_count
        info['episode_task_completions'] = episode_completions
        info['n_tasks_completed'] = len(episode_completions)
        info['tasks_remaining'] = [t for t in self.tasks_to_complete if t not in episode_completions]
        return img, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        if self._current_obs is None:
            return np.zeros(59, dtype=np.float32)
        return self._current_obs['observation'].astype(np.float32).copy()

    def close(self):
        self._env.close()

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def n_tasks(self) -> int:
        return len(self.tasks_to_complete)
