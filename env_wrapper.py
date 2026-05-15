"""
env_wrapper.py — Franka Kitchen image/state wrapper.

Changes vs. the original codebase:
  * Returns image AND full state on every step, so the agent never has to
    call env.get_state() as a separate RPC. The state is in info['state'].
  * Exposes a first-reset sanity check: verifies obs['observation'] is
    the 59-d vector our task indices expect.
  * Exposes info['tasks_completed_names'] (list of task NAMES completed
    this episode), which we use to build the completion mask in the agent.
  * Option execution lives in the active skill agent.
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

    @staticmethod
    def observation_to_qpos_qvel(observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct MuJoCo qpos/qvel from the D4RL-compatible observation
        layout used by FrankaKitchen-v1.
        """
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if obs.size < 39:
            raise ValueError(
                f"Expected kitchen observation with >=39 dims, got {obs.size}."
            )
        qpos = np.concatenate([obs[:9], obs[18:39]], axis=0)
        qvel = np.concatenate([obs[9:18], obs[39:]], axis=0)
        return qpos.astype(np.float64), qvel.astype(np.float64)

    def set_mujoco_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """
        Set the underlying MuJoCo simulator state for offline rendering.
        """
        base = self._env.unwrapped
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        qvel = np.asarray(qvel, dtype=np.float64).reshape(-1)

        if hasattr(base, "set_state"):
            base.set_state(qpos, qvel)
        elif hasattr(base, "data") and hasattr(base, "model"):
            nq = int(getattr(base.model, "nq", qpos.shape[0]))
            nv = int(getattr(base.model, "nv", qvel.shape[0]))
            if qpos.shape[0] != nq or qvel.shape[0] != nv:
                raise ValueError(
                    f"State size mismatch for MuJoCo replay: "
                    f"got qpos={qpos.shape[0]}, qvel={qvel.shape[0]}, "
                    f"expected nq={nq}, nv={nv}."
                )
            base.data.qpos[:nq] = qpos
            base.data.qvel[:nv] = qvel
            if hasattr(base.data, "act") and base.data.act is not None:
                base.data.act[:] = 0.0
        else:
            raise AttributeError(
                "FrankaKitchen-v1 env does not expose set_state(qpos, qvel) "
                "and direct MuJoCo state access is unavailable."
            )

        try:
            import mujoco
            mujoco.mj_forward(base.model, base.data)
        except Exception:
            pass

    def render_from_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Replay a dataset observation into MuJoCo and render the corresponding RGB frame.
        """
        qpos, qvel = self.observation_to_qpos_qvel(observation)
        self.set_mujoco_state(qpos, qvel)
        return self.render_image()

    # ---------------------- privileged geometry access ----------------------

    def _model_names(self, kind: str) -> List[str]:
        """
        Return MuJoCo object names for `site`, `body`, or `geom`.

        This is intentionally best-effort: different kitchen package versions
        expose slightly different names, so callers should use fuzzy matching.
        """
        base = self._env.unwrapped
        model = getattr(base, "model", None)
        if model is None:
            return []
        n = int(getattr(model, f"n{kind}", 0))
        names = []
        typed_accessor = getattr(model, kind, None)
        id2name = getattr(model, f"{kind}_id2name", None)
        raw_names = getattr(model, "names", None)
        name_adr = getattr(model, f"name_{kind}adr", None)
        for i in range(n):
            name = None
            if callable(typed_accessor):
                try:
                    name = getattr(typed_accessor(i), "name", None)
                except Exception:
                    name = None
            if name is None and callable(id2name):
                try:
                    name = id2name(i)
                except Exception:
                    name = None
            if name is None and hasattr(model, "id2name"):
                try:
                    name = model.id2name(i, kind)
                except Exception:
                    name = None
            if name is None and raw_names is not None and name_adr is not None:
                try:
                    adr = int(name_adr[i])
                    raw = bytes(raw_names[adr:])
                    name = raw.split(b"\x00", 1)[0].decode("utf-8")
                except Exception:
                    name = None
            names.append(str(name or ""))
        return names

    def list_mujoco_names(self) -> Dict[str, List[str]]:
        return {
            "site": self._model_names("site"),
            "body": self._model_names("body"),
            "geom": self._model_names("geom"),
        }

    def xpos_by_name_patterns(self, patterns: List[str]) -> Optional[np.ndarray]:
        """
        Best-effort world position lookup. Searches sites first, then bodies,
        then geoms. Returns the highest-scoring fuzzy match.
        """
        base = self._env.unwrapped
        model = getattr(base, "model", None)
        data = getattr(base, "data", None)
        if model is None or data is None:
            return None
        pats = [p.lower() for p in patterns]
        best_score = 0
        best_pos = None
        for kind, arr_name in (("site", "site_xpos"), ("body", "xpos"), ("geom", "geom_xpos")):
            names = self._model_names(kind)
            arr = getattr(data, arr_name, None)
            if arr is None:
                continue
            for i, name in enumerate(names):
                low = name.lower()
                score = sum(1 for p in pats if p in low)
                if score > best_score:
                    try:
                        best_pos = np.asarray(arr[i], dtype=np.float64).copy()
                        best_score = score
                    except Exception:
                        continue
        return best_pos

    def contact_features_by_name_patterns(self,
                                          patterns_a: List[str],
                                          patterns_b: List[str]) -> np.ndarray:
        """
        Return [contact_flag, min_contact_dist] for contacts between two fuzzy
        geom-name groups. If no matching contact exists, min_contact_dist=1.0.
        """
        base = self._env.unwrapped
        data = getattr(base, "data", None)
        if data is None:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        geom_names = self._model_names("geom")
        pats_a = [p.lower() for p in patterns_a]
        pats_b = [p.lower() for p in patterns_b]

        def matches(name: str, pats: List[str]) -> bool:
            low = name.lower()
            return any(p in low for p in pats)

        min_dist = 1.0
        found = False
        for i in range(int(getattr(data, "ncon", 0))):
            try:
                con = data.contact[i]
                g1 = int(con.geom1)
                g2 = int(con.geom2)
                n1 = geom_names[g1] if 0 <= g1 < len(geom_names) else ""
                n2 = geom_names[g2] if 0 <= g2 < len(geom_names) else ""
                pair_match = (
                    matches(n1, pats_a) and matches(n2, pats_b)
                ) or (
                    matches(n1, pats_b) and matches(n2, pats_a)
                )
                if pair_match:
                    found = True
                    min_dist = min(min_dist, float(getattr(con, "dist", 0.0)))
            except Exception:
                continue
        return np.asarray([1.0 if found else 0.0, min_dist], dtype=np.float32)

    # Back-compat alias for any legacy callers.
    _render_image = render_image

    # ---------------------- lifecycle ----------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset and return (image, state_59d).

        On the very first call, sanity-check that the state is 59-d —
        our task_spec indices assume this layout.
        """
        reset_kwargs = {}
        eff_seed = self._seed if seed is None else seed
        if eff_seed is not None:
            reset_kwargs['seed'] = eff_seed
            self._seed = None

        obs_dict, _ = self._env.reset(**reset_kwargs)
        self._current_obs = obs_dict
        self._step_count = 0

        state = np.asarray(obs_dict['observation'], dtype=np.float64)
        if not self._verified_obs_dim:
            assert state.size >= 39, (
                f"FrankaKitchen-v1 returned state of size {state.size}; "
                f"TaskSpec indices expect the 59-d D4RL-compatible "
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
