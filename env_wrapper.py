"""
Franka Kitchen environment wrapper for image-based HRL.

Wraps the D4RL Franka Kitchen to:
1. Render RGB images as observations (instead of proprioception)
2. Provide sparse task reward (original kitchen reward)
3. Support our hierarchical execution protocol
"""
import numpy as np
import gym
import cv2
from typing import Tuple, Dict, Optional


class FrankaKitchenImageWrapper:
    """
    Wraps Franka Kitchen to output image observations.
    
    The original Franka Kitchen returns a 60-d state vector.
    This wrapper renders an RGB image from the simulator camera
    and returns that as the observation.
    
    Usage:
        env = FrankaKitchenImageWrapper(img_size=224)
        obs_img = env.reset()  # (224, 224, 3) uint8
        obs_img, reward, done, info = env.step(action)
    """
    
    def __init__(
        self,
        task: str = "kitchen-complete-v0",
        img_size: int = 224,
        camera_name: str = "fixed",  # "fixed" gives a good overview
        seed: Optional[int] = None,
    ):
        self.img_size = img_size
        self.camera_name = camera_name
        
        # Create the base environment
        # D4RL kitchen environments: kitchen-complete-v0, kitchen-partial-v0, kitchen-mixed-v0
        self._env = gym.make(task)
        
        if seed is not None:
            self._env.seed(seed)
            self._env.action_space.seed(seed)
        
        # Action space (9-DoF for Franka Kitchen)
        self.action_space = self._env.action_space
        self.action_dim = self.action_space.shape[0]
        
        # Track state for rendering
        self._current_state = None
        self._step_count = 0
        self._max_steps = 280  # Standard Kitchen episode length
        
    def _render_image(self) -> np.ndarray:
        """Render an RGB image from the simulator."""
        try:
            # Try gym render
            img = self._env.render(mode='rgb_array')
        except Exception:
            try:
                # Some versions use different API
                img = self._env.sim.render(
                    width=self.img_size, 
                    height=self.img_size,
                    camera_name=self.camera_name,
                )
                # MuJoCo renders upside-down
                img = img[::-1, :, :]
            except Exception:
                # Fallback: render from dm_control-style interface
                try:
                    img = self._env.render(
                        width=self.img_size,
                        height=self.img_size,
                    )
                except Exception:
                    # Last resort: black image (for headless debugging)
                    print("WARNING: Rendering failed. Returning black image.")
                    print("Set MUJOCO_GL=egl or MUJOCO_GL=osmesa for headless rendering.")
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                    return img
        
        # Resize to target size
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), 
                           interpolation=cv2.INTER_AREA)
        
        return img.astype(np.uint8)
    
    def reset(self) -> np.ndarray:
        """Reset environment, return image observation."""
        self._current_state = self._env.reset()
        self._step_count = 0
        return self._render_image()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step environment.
        Returns: (image_obs, reward, done, info)
        
        info contains:
            - 'state': the original 60-d state vector (for debugging/evaluation)
            - 'n_completed': number of subtasks completed
        """
        state, reward, done, info = self._env.step(action)
        self._current_state = state
        self._step_count += 1
        
        # Check if episode should end
        if self._step_count >= self._max_steps:
            done = True
        
        # Render image observation
        img = self._render_image()
        
        # Add useful info
        info['state'] = state
        info['step_count'] = self._step_count
        
        return img, reward, done, info
    
    def get_state(self) -> np.ndarray:
        """Get the underlying state vector (for evaluation/debugging only)."""
        return self._current_state.copy()
    
    def close(self):
        self._env.close()
    
    @property
    def max_steps(self):
        return self._max_steps


class HierarchicalKitchenWrapper:
    """
    Wrapper that implements the hierarchical execution protocol:
    - Manager selects a subgoal (landmark index)
    - Worker executes for K steps trying to reach the subgoal
    - Strict execution: success continues, failure terminates high-level transition
    
    This wrapper manages the inner loop (worker execution for K steps)
    and reports outcomes to the outer loop (manager).
    """
    
    def __init__(self, env: FrankaKitchenImageWrapper, subgoal_horizon: int = 20):
        self.env = env
        self.K = subgoal_horizon  # Low-level steps per subgoal attempt
        self.action_space = env.action_space
        self.action_dim = env.action_dim
    
    def reset(self) -> np.ndarray:
        """Reset and return initial image."""
        return self.env.reset()
    
    def execute_subgoal(
        self, 
        worker_policy,  # Callable: (z_current, z_subgoal) -> action
        z_subgoal: np.ndarray,
        encoder,  # VisualEncoder for encoding new observations
        reachability_fn=None,  # Optional: f(z_curr, z_sub) -> [0,1]
        success_threshold: float = 5.0,  # In latent L2 distance
    ) -> Dict:
        """
        Execute one subgoal attempt for up to K low-level steps.
        
        Returns a dict with:
            - 'success': bool — did the worker reach the subgoal?
            - 'final_image': np.ndarray — image at end of attempt
            - 'final_z': np.ndarray — latent at end of attempt
            - 'cumulative_reward': float — total env reward during attempt
            - 'n_steps': int — how many steps were taken
            - 'trajectory': list of (image, action, reward, z) tuples
            - 'done': bool — did the env episode end?
            - 'delta_progress': float — total L2 progress toward subgoal
        """
        trajectory = []
        cumulative_reward = 0.0
        env_done = False
        
        # Get current latent
        current_img = self.env._render_image()
        z_current = encoder.encode_numpy(current_img).squeeze()
        initial_dist = np.linalg.norm(z_current - z_subgoal)
        
        for step_i in range(self.K):
            # Worker selects action
            action = worker_policy(z_current, z_subgoal)
            
            # Step environment
            next_img, reward, done, info = self.env.step(action)
            cumulative_reward += reward
            
            # Encode new observation
            z_next = encoder.encode_numpy(next_img).squeeze()
            
            # Store transition
            trajectory.append({
                'image': current_img,
                'z': z_current.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_image': next_img,
                'z_next': z_next.copy(),
                'z_subgoal': z_subgoal.copy(),
            })
            
            # Update current
            z_current = z_next
            current_img = next_img
            
            if done:
                env_done = True
                break
        
        # Check if subgoal was reached
        final_dist = np.linalg.norm(z_current - z_subgoal)
        success = final_dist < success_threshold
        delta_progress = initial_dist - final_dist  # Positive = got closer
        
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
