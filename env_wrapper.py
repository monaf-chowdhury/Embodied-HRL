"""
Franka Kitchen v1 environment wrapper for image-based HRL.

Uses FrankaKitchen-v1 from gymnasium-robotics (NOT the legacy D4RL v0).

Key design decisions:
- render_mode="rgb_array" is passed at make() time — this is the v1 API.
  env.render() then returns a numpy array directly.
- The camera matches the original relay-policy-learning demo camera:
  distance=4.5, azimuth=-66°, elevation=-65°, lookat=[-0.1, 0.75, 1.6].
  Set directly via MuJoCo's spherical parameterisation (no Cartesian
  conversion) to exactly reproduce the demo_relay_cam viewpoint.
- obs from v1 is a dict: obs['observation'] is the 59-d state vector.
  We only use images as observations; the state vector is kept in info
  for debugging only.
- info['episode_task_completions'] tells us which subtasks are done.
- Step returns 5 values: (obs, reward, terminated, truncated, info).
  done = terminated or truncated.
- tasks_to_complete is configurable — default is the 4-task complete set.
"""
import os
import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers FrankaKitchen-v1

gym.register_envs(gymnasium_robotics)


# =============================================================================
# Camera: demo_relay_cam — from relay-policy-learning source XML.
# distance=4.5, azimuth=-66°, elevation=-65°, lookat=[-0.1, 0.75, 1.6]
# Set using MuJoCo's native spherical params — no Cartesian conversion.
# =============================================================================

_CAM_DISTANCE  = 4.5
_CAM_AZIMUTH   = -66.0   # degrees
_CAM_ELEVATION = -65.0   # degrees
_CAM_LOOKAT    = np.array([-0.1, 0.75, 1.6])

# Default 4-task complete set (equivalent to kitchen-complete-v0)
_DEFAULT_TASKS = ['microwave', 'kettle', 'light switch', 'slide cabinet']


def _apply_camera(env):
    """
    Override the renderer camera to use the relay-policy-learning demo viewpoint.
    Uses MuJoCo's native spherical parameterisation (distance/azimuth/elevation)
    directly — no Cartesian conversion needed or performed.
    """
    try:
        renderer = env.unwrapped.mujoco_renderer
        if renderer is None:
            return

        cam = renderer.viewer.cam if hasattr(renderer, 'viewer') else None
        if cam is not None:
            cam.lookat[:]  = _CAM_LOOKAT
            cam.distance   = _CAM_DISTANCE
            cam.azimuth    = _CAM_AZIMUTH
            cam.elevation  = _CAM_ELEVATION
    except Exception:
        pass  # Renderer not yet initialised — will be called again after reset


class FrankaKitchenImageWrapper:
    """
    Wraps FrankaKitchen-v1 (gymnasium-robotics) to output image observations.

    Usage:
        env = FrankaKitchenImageWrapper(img_size=224)
        img = env.reset()                        # (224, 224, 3) uint8
        img, reward, done, info = env.step(act)

    info contains:
        'state'                  : 59-d proprioceptive observation
        'step_count'             : steps taken this episode
        'episode_task_completions': list of task names completed so far
        'tasks_remaining'        : list of tasks not yet completed
        'n_tasks_completed'      : int count of completed tasks
    """

    def __init__(
        self,
        tasks_to_complete: Optional[List[str]] = None,
        img_size: int = 224,
        seed: Optional[int] = None,
        terminate_on_tasks_completed: bool = False,
    ):
        self.img_size = img_size
        self.tasks_to_complete = tasks_to_complete or _DEFAULT_TASKS

        # v1 render API: pass render_mode at make() time
        # width/height are passed here for the renderer
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
        self.action_dim   = self.action_space.shape[0]
        self._current_obs = None
        self._step_count  = 0
        self._max_steps   = 280
        self._seed        = seed

    def _render_image(self) -> np.ndarray:
        """
        Render using env.render() — the v1 / gymnasium API.
        Applies the demo camera before each render.
        Returns (img_size, img_size, 3) uint8 array.
        """
        _apply_camera(self._env)
        try:
            img = self._env.render()  # returns numpy array in rgb_array mode
            if img is None:
                raise ValueError("render() returned None")
            img = np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"WARNING: render failed ({e}), returning black image.")
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    def reset(self) -> np.ndarray:
        """Reset environment, return image observation."""
        reset_kwargs = {}
        if self._seed is not None:
            reset_kwargs['seed'] = self._seed
            self._seed = None  # only seed on first reset

        obs_dict, _ = self._env.reset(**reset_kwargs)
        self._current_obs = obs_dict
        self._step_count  = 0
        return self._render_image()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step environment.
        Returns: (image_obs, reward, done, info)
        """
        obs_dict, reward, terminated, truncated, info = self._env.step(action)
        self._current_obs  = obs_dict
        self._step_count  += 1
        done = terminated or truncated or (self._step_count >= self._max_steps)

        img = self._render_image()

        # Enrich info with useful fields
        episode_completions = info.get('episode_task_completions', [])
        info['state']                   = obs_dict['observation']
        info['step_count']              = self._step_count
        info['episode_task_completions'] = episode_completions
        info['n_tasks_completed']       = len(episode_completions)
        info['tasks_remaining']         = [
            t for t in self.tasks_to_complete
            if t not in episode_completions
        ]

        return img, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        """Return the underlying 59-d proprioceptive state."""
        if self._current_obs is None:
            return np.zeros(59, dtype=np.float64)
        return self._current_obs['observation'].copy()

    def close(self):
        self._env.close()

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def n_tasks(self) -> int:
        return len(self.tasks_to_complete)


# =============================================================================
# Hierarchical wrapper
# =============================================================================

class HierarchicalKitchenWrapper:
    """
    Implements the hierarchical execution protocol:
    - Manager selects a subgoal (landmark index)
    - Worker executes for K steps trying to reach it
    - Strict execution: failure terminates the high-level episode
    """

    def __init__(self, env: FrankaKitchenImageWrapper, subgoal_horizon: int = 20):
        self.env          = env
        self.K            = subgoal_horizon
        self.action_space = env.action_space
        self.action_dim   = env.action_dim

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def execute_subgoal(
        self,
        worker_policy,
        z_subgoal: np.ndarray,
        encoder,
        success_threshold: float = 5.0,
    ) -> Dict:
        """
        Execute one subgoal attempt for up to K low-level steps.

        Returns dict with:
            success, final_image, final_z, cumulative_reward,
            n_steps, trajectory, done, delta_progress,
            initial_dist, final_dist, n_tasks_completed
        """
        trajectory        = []
        cumulative_reward = 0.0
        env_done          = False
        n_tasks_completed = 0

        current_img  = self.env._render_image()
        z_current    = encoder.encode_numpy(current_img).squeeze()
        initial_dist = np.linalg.norm(z_current - z_subgoal)

        for _ in range(self.K):
            action = worker_policy(z_current, z_subgoal)
            next_img, reward, done, info = self.env.step(action)
            cumulative_reward += reward
            z_next = encoder.encode_numpy(next_img).squeeze()
            n_tasks_completed = info.get('n_tasks_completed', 0)

            trajectory.append({
                'image':     current_img,
                'z':         z_current.copy(),
                'action':    action.copy(),
                'reward':    reward,
                'next_image': next_img,
                'z_next':    z_next.copy(),
                'z_subgoal': z_subgoal.copy(),
            })

            z_current   = z_next
            current_img = next_img

            if done:
                env_done = True
                break

        final_dist     = np.linalg.norm(z_current - z_subgoal)
        success        = final_dist < success_threshold
        delta_progress = initial_dist - final_dist

        return {
            'success':           success,
            'final_image':       current_img,
            'final_z':           z_current,
            'cumulative_reward': cumulative_reward,
            'n_steps':           len(trajectory),
            'trajectory':        trajectory,
            'done':              env_done,
            'delta_progress':    delta_progress,
            'initial_dist':      initial_dist,
            'final_dist':        final_dist,
            'n_tasks_completed': n_tasks_completed,
        }

    def close(self):
        self.env.close()