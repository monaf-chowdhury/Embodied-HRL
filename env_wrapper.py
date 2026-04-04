"""
Franka Kitchen environment wrapper for image-based HRL.
"""
import numpy as np
import os
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import gym
import d4rl
import cv2
from typing import Tuple, Dict, Optional


class FrankaKitchenImageWrapper:
    def __init__(
        self,
        task: str = "kitchen-complete-v0",
        img_size: int = 224,
        camera_name: str = "fixed",
        seed: Optional[int] = None,
    ):
        self.img_size = img_size
        self.camera_name = camera_name

        # Make env but bypass gym's passive checker by accessing the unwrapped env
        _env = gym.make(task)
        # Unwrap completely to get the raw kitchen env with 4-value step API
        self._env = _env.unwrapped
        # Re-seed
        if seed is not None:
            self._env.seed(seed)
            self._env.action_space.seed(seed)

        self.action_space = self._env.action_space
        self.action_dim = self.action_space.shape[0]
        self._current_state = None
        self._step_count = 0
        self._max_steps = 280

    def _render_image(self) -> np.ndarray:
        img = None
        try:
            physics = self._env.sim
            img = physics.render(
                width=self.img_size,
                height=self.img_size,
                camera_id=0,
            )
            if img is not None:
                img = np.array(img)
        except Exception:
            img = None

        if img is None:
            try:
                result = self._env.render(mode='rgb_array')
                if isinstance(result, list):
                    result = result[0] if len(result) > 0 else None
                if result is not None:
                    img = np.array(result)
            except Exception:
                img = None

        if img is None or not isinstance(img, np.ndarray):
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)

        return img.astype(np.uint8)

    def reset(self) -> np.ndarray:
        self._current_state = self._env.reset()
        self._step_count = 0
        return self._render_image()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        result = self._env.step(action)
        # Handle both 4-value and 5-value step APIs
        if len(result) == 5:
            state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            state, reward, done, info = result

        self._current_state = state
        self._step_count += 1

        if self._step_count >= self._max_steps:
            done = True

        img = self._render_image()
        info['state'] = state
        info['step_count'] = self._step_count

        return img, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        return self._current_state.copy()

    def close(self):
        self._env.close()

    @property
    def max_steps(self):
        return self._max_steps


class HierarchicalKitchenWrapper:
    def __init__(self, env: FrankaKitchenImageWrapper, subgoal_horizon: int = 20):
        self.env = env
        self.K = subgoal_horizon
        self.action_space = env.action_space
        self.action_dim = env.action_dim

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def execute_subgoal(
        self,
        worker_policy,
        z_subgoal: np.ndarray,
        encoder,
        reachability_fn=None,
        success_threshold: float = 5.0,
    ) -> Dict:
        trajectory = []
        cumulative_reward = 0.0
        env_done = False

        current_img = self.env._render_image()
        z_current = encoder.encode_numpy(current_img).squeeze()
        initial_dist = np.linalg.norm(z_current - z_subgoal)

        for step_i in range(self.K):
            action = worker_policy(z_current, z_subgoal)
            next_img, reward, done, info = self.env.step(action)
            cumulative_reward += reward
            z_next = encoder.encode_numpy(next_img).squeeze()

            trajectory.append({
                'image': current_img,
                'z': z_current.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_image': next_img,
                'z_next': z_next.copy(),
                'z_subgoal': z_subgoal.copy(),
            })

            z_current = z_next
            current_img = next_img

            if done:
                env_done = True
                break

        final_dist = np.linalg.norm(z_current - z_subgoal)
        success = final_dist < success_threshold
        delta_progress = initial_dist - final_dist

        return {
            'success': success,
            'final_image': current_img,
            'final_z': z_current,
            'cumulative_reward': cumulative_reward,
            'n_steps': len(trajectory),
            'trajectory': trajectory,
            'done': env_done,
            'delta_progress': delta_progress,
            'initial_dist': initial_dist,
            'final_dist': final_dist,
        }

    def close(self):
        self.env.close()
