"""
VisualHRLAgent — all changes applied.

Key changes:
  1. No projection head. z_dim = 2048 (raw R3M).
  2. No SSE. Episodes are not terminated on subgoal failure.
  3. Reachability filter disabled (config.reachability.reject_threshold=0.0).
  4. Manager reward redesigned:
       - task_completion_bonus if n_tasks_completed increased (dominant)
       - task_progress_bonus * delta_task_progress (secondary)
       - env_reward_weight * cumulative_env_reward (passthrough)
       - nav_bonus if landmark reached in latent space (small)
     This breaks the bootstrapping deadlock by giving the manager a
     non-zero signal even before any task is completed.
  5. Worker task-progress reward: "focused" — computed only for the single
     task whose proprio-goal is nearest to the current proprio state.
     This avoids diluting the gradient across all 4 tasks simultaneously.
     Rationale: the worker should focus on ONE task at a time. The task
     it is nearest to is the most useful gradient to follow.
  6. calibrate_success_threshold redesigned for 2048-d L2-normalised space.
     L2-normed distances are in [0,2]. Expected step mean: 0.05-0.20.
     Threshold is set as a multiple of the step mean — not percentile of
     inter-landmark distances (which can be large in 2048-d uniform FPS).
  7. Eval metric: task_completion_rate (n_tasks >= 1) replaces ep_reward > 0.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple as Tuple_

from config import Config
from encoder import VisualEncoder
from env_wrapper import FrankaKitchenImageWrapper
from landmarks import LandmarkBuffer
from networks import ManagerQNetwork, SACActorNetwork, SACCriticNetwork, ReachabilityPredictor
from buffers import HighLevelBuffer, LowLevelBuffer, ReachabilityBuffer


# =============================================================================
# Task-progress helpers (verified indices: obs_idx = qpos_idx + 9)
# =============================================================================

_TASK_OBS_IDX = {
    'bottom burner': (np.array([18, 19]),  np.array([-0.88, -0.01])),
    'top burner':    (np.array([24, 25]),  np.array([-0.92, -0.01])),
    'light switch':  (np.array([26, 27]),  np.array([-0.69, -0.05])),
    'slide cabinet': (np.array([28]),      np.array([0.37])),
    'hinge cabinet': (np.array([29, 30]),  np.array([0., 1.45])),
    'microwave':     (np.array([31]),      np.array([-0.75])),
    'kettle':        (np.array([32, 33, 34, 35, 36, 37, 38]),
                      np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06])),
}


def _task_proprio_dist(proprio: np.ndarray, task: str) -> float:
    """Proprio L2 distance to the goal state for a single task."""
    if task not in _TASK_OBS_IDX:
        return float('inf')
    idx, goal = _TASK_OBS_IDX[task]
    return float(np.linalg.norm(proprio[idx] - goal))


def _task_progress(proprio: np.ndarray, task: str) -> float:
    """[0,1] progress for a single task."""
    if task not in _TASK_OBS_IDX:
        return 0.0
    idx, goal = _TASK_OBS_IDX[task]
    dist     = np.linalg.norm(proprio[idx] - goal)
    max_dist = np.linalg.norm(goal) + 1e-4
    return float(max(0.0, 1.0 - dist / max_dist))


def compute_task_progress_focused(
    proprio: np.ndarray,
    tasks: list,
) -> Tuple_[str, float]:
    """
    Return (nearest_task_name, progress_for_that_task).
    'Nearest' = task whose proprio-goal is closest to current proprio state.
    This focuses the worker on ONE task rather than averaging over all,
    avoiding the 4x dilution problem.
    """
    if not tasks:
        return '', 0.0
    dists = {t: _task_proprio_dist(proprio, t) for t in tasks if t in _TASK_OBS_IDX}
    if not dists:
        return '', 0.0
    nearest = min(dists, key=dists.get)
    return nearest, _task_progress(proprio, nearest)



class VisualHRLAgent:

    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        z_dim       = config.encoder.raw_dim   # 2048 — no projection
        proprio_dim = config.worker.proprio_dim
        action_dim  = 9

        # ---- Encoder (frozen, no projection) ----
        self.encoder = VisualEncoder(config.encoder, device=self.device)

        # ---- Landmarks ----
        self.landmarks = LandmarkBuffer(
            n_landmarks=config.landmarks.n_landmarks,
            z_dim=z_dim,
            landmark_config=config.landmarks,
        )

        # ---- Manager (DQN) ----
        self.manager_q = ManagerQNetwork(
            z_dim, config.manager.hidden_dim, config.manager.n_layers,
        ).to(self.device)
        self.manager_q_target = ManagerQNetwork(
            z_dim, config.manager.hidden_dim, config.manager.n_layers,
        ).to(self.device)
        self.manager_q_target.load_state_dict(self.manager_q.state_dict())
        # Only manager_q parameters — encoder has nothing trainable
        self.manager_optimizer = torch.optim.Adam(
            self.manager_q.parameters(), lr=config.manager.lr)

        # ---- Worker (SAC) ----
        self.worker_actor = SACActorNetwork(
            z_dim, action_dim,
            config.worker.hidden_dim, config.worker.n_layers, proprio_dim,
        ).to(self.device)
        self.worker_critic = SACCriticNetwork(
            z_dim, action_dim,
            config.worker.hidden_dim, config.worker.n_layers, proprio_dim,
        ).to(self.device)
        self.worker_critic_target = SACCriticNetwork(
            z_dim, action_dim,
            config.worker.hidden_dim, config.worker.n_layers, proprio_dim,
        ).to(self.device)
        self.worker_critic_target.load_state_dict(self.worker_critic.state_dict())

        self.worker_actor_optimizer  = torch.optim.Adam(
            self.worker_actor.parameters(),  lr=config.worker.actor_lr)
        self.worker_critic_optimizer = torch.optim.Adam(
            self.worker_critic.parameters(), lr=config.worker.critic_lr)

        if config.worker.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(config.worker.init_alpha),
                requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=config.worker.alpha_lr)
            self.target_entropy = -action_dim
        else:
            self.log_alpha = torch.tensor(
                np.log(config.worker.init_alpha), device=self.device)

        # ---- Reachability (kept, disabled via config) ----
        self.reachability = ReachabilityPredictor(
            z_dim, config.reachability.hidden_dim, config.reachability.n_layers,
        ).to(self.device)
        self.reachability_optimizer = torch.optim.Adam(
            self.reachability.parameters(), lr=config.reachability.lr)

        # ---- Buffers ----
        self.high_buffer  = HighLevelBuffer(capacity=200_000, z_dim=z_dim)
        self.low_buffer   = LowLevelBuffer(
            capacity=config.buffer.capacity,
            z_dim=z_dim, action_dim=action_dim, proprio_dim=proprio_dim,
        )
        self.reach_buffer = ReachabilityBuffer(capacity=100_000, z_dim=z_dim)

        # ---- Tracking ----
        self.total_steps    = 0
        self.total_episodes = 0
        self.epsilon        = config.manager.epsilon_start
        self._latent_dists  = []
        self.success_threshold = 0.3   # placeholder; calibrate_success_threshold() sets this
        self._prev_n_tasks  = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    # =========================================================================
    # Manager: subgoal selection (reachability filter disabled)
    # =========================================================================

    def select_subgoal(self, z_current: np.ndarray) -> int:
        if not self.landmarks.is_ready:
            return 0

        if np.random.random() < self.config.landmarks.explore_ratio:
            return self.landmarks.select_explore()

        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.landmarks.n_active)

        with torch.no_grad():
            z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
            lm  = torch.from_numpy(self.landmarks.get_all()).float().to(self.device)
            q_values = self.manager_q.evaluate_all_landmarks(z_c, lm)
            # Reachability filter: only active if enabled_after_steps reached
            # and reject_threshold > 0. Both are set to "never" in config.
            return q_values.argmax(dim=1).item()

    def _update_epsilon(self):
        cfg = self.config.manager
        self.epsilon = max(
            cfg.epsilon_end,
            cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end)
            * self.total_steps / cfg.epsilon_decay_steps,
        )

    # =========================================================================
    # Worker: action selection
    # =========================================================================

    def get_worker_action(
        self,
        z_current: np.ndarray,
        z_subgoal: np.ndarray,
        proprio: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
        z_s = torch.from_numpy(z_subgoal).float().unsqueeze(0).to(self.device)
        p   = torch.from_numpy(
            self.low_buffer.normalise_proprio(proprio)
        ).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                action = self.worker_actor.get_action_deterministic(z_c, z_s, p)
            else:
                action, _ = self.worker_actor(z_c, z_s, p)
        return action.cpu().numpy().squeeze()

    # =========================================================================
    # Reward computation
    # =========================================================================

    def compute_worker_reward(
        self,
        z_t: np.ndarray,
        z_next: np.ndarray,
        z_subgoal: np.ndarray,
        sparse_reward: float,
        proprio_t: np.ndarray,
        proprio_next: np.ndarray,
        initial_dist: float,
    ) -> Tuple_[float, float]:
        """
        Worker shaped reward with focused task-progress signal.
        Returns (shaped_reward, task_delta) where task_delta is the change
        in progress for the NEAREST task only (not averaged over all).
        """
        cfg = self.config.reward

        # 1. Sparse (dominant)
        r = cfg.sparse_weight * sparse_reward

        # 2. Focused task-progress (secondary)
        # Find which task is nearest BEFORE step; compute progress delta
        # for that SAME task before and after.
        nearest_task, _ = compute_task_progress_focused(
            proprio_t, self.config.training.tasks_to_complete)
        if nearest_task:
            prog_before_val = _task_progress(proprio_t,    nearest_task)
            prog_after_val  = _task_progress(proprio_next, nearest_task)
            delta_task      = prog_after_val - prog_before_val
        else:
            delta_task = 0.0
        r += cfg.task_progress_weight * delta_task

        # 3. Normalised latent progress (tertiary — small navigation hint)
        dist_before = np.linalg.norm(z_t    - z_subgoal)
        dist_after  = np.linalg.norm(z_next - z_subgoal)
        denom       = max(initial_dist, dist_before, 1e-4)
        r += cfg.latent_weight * ((dist_before - dist_after) / denom)

        return float(r), float(delta_task)

    def compute_manager_reward(
        self,
        n_tasks_before: int,
        n_tasks_after: int,
        task_progress_before: float,
        task_progress_after: float,
        cumulative_env_reward: float,
        landmark_reached: bool,
    ) -> float:
        """
        Manager reward: breaks the bootstrapping deadlock by providing a
        non-zero signal even before full task completion.

        Hierarchy:
          1. task_completion_bonus (10.0)  — n_tasks increased [DOMINANT]
          2. task_progress_bonus (3.0) * delta_task_progress [SECONDARY]
          3. env_reward_weight (1.0) * cumulative_env_reward [PASSTHROUGH]
          4. nav_bonus (0.5) if landmark reached [SMALL]
        """
        cfg = self.config.manager
        r   = 0.0

        # 1. Task completion (dominant)
        if n_tasks_after > n_tasks_before:
            r += cfg.task_completion_bonus * (n_tasks_after - n_tasks_before)

        # 2. Task progress (secondary)
        delta_prog = task_progress_after - task_progress_before
        if delta_prog > 0:
            r += cfg.task_progress_bonus * delta_prog

        # 3. Env reward passthrough
        r += cfg.env_reward_weight * cumulative_env_reward

        # 4. Navigation bonus (small)
        if landmark_reached:
            r += cfg.nav_bonus

        return float(r)

    def maybe_inject_hindsight(self, z: np.ndarray, n_tasks: int):
        if (self.config.landmarks.use_hindsight_landmarks
                and n_tasks > self._prev_n_tasks):
            self.landmarks.add_success_state(z)
        self._prev_n_tasks = n_tasks

    # =========================================================================
    # Manager update
    # =========================================================================

    def update_manager(self) -> dict:
        if self.high_buffer.size < self.config.buffer.batch_size:
            return {}
        batch  = self.high_buffer.sample(self.config.buffer.batch_size)
        z_curr = torch.from_numpy(batch['z_current']).to(self.device)
        z_sub  = torch.from_numpy(batch['z_subgoal']).to(self.device)
        reward = torch.from_numpy(batch['reward']).unsqueeze(1).to(self.device)
        z_next = torch.from_numpy(batch['z_next']).to(self.device)
        done   = torch.from_numpy(batch['done']).unsqueeze(1).to(self.device)

        q_current = self.manager_q(z_curr, z_sub)

        with torch.no_grad():
            if self.landmarks.is_ready:
                lm = torch.from_numpy(self.landmarks.get_all()).float().to(self.device)
                q_next = self.manager_q_target.evaluate_all_landmarks(z_next, lm).max(dim=1, keepdim=True)[0]
            else:
                q_next = torch.zeros_like(reward)
            target = reward + self.config.manager.gamma * (1 - done) * q_next

        loss = F.mse_loss(q_current, target)
        self.manager_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_q.parameters(), 1.0)
        self.manager_optimizer.step()
        self._soft_update(self.manager_q, self.manager_q_target, self.config.manager.tau)

        return {'manager_loss': loss.item(), 'manager_q_mean': q_current.mean().item()}

    # =========================================================================
    # Worker update
    # =========================================================================

    def update_worker(self) -> dict:
        if self.low_buffer.size < self.config.buffer.batch_size:
            return {}
        batch  = self.low_buffer.sample(self.config.buffer.batch_size)
        z_curr = torch.from_numpy(batch['z_current']).to(self.device)
        p_curr = torch.from_numpy(batch['proprio']).to(self.device)
        z_sub  = torch.from_numpy(batch['z_subgoal']).to(self.device)
        action = torch.from_numpy(batch['action']).to(self.device)
        reward = torch.from_numpy(batch['reward']).unsqueeze(1).to(self.device)
        z_next = torch.from_numpy(batch['z_next']).to(self.device)
        p_next = torch.from_numpy(batch['proprio_next']).to(self.device)
        done   = torch.from_numpy(batch['done']).unsqueeze(1).to(self.device)

        with torch.no_grad():
            na, nlp = self.worker_actor(z_next, z_sub, p_next)
            q1n, q2n = self.worker_critic_target(z_next, z_sub, p_next, na)
            q_next   = torch.min(q1n, q2n) - self.alpha * nlp
            target_q = reward + self.config.worker.gamma * (1 - done) * q_next

        q1, q2 = self.worker_critic(z_curr, z_sub, p_curr, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.worker_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(), 1.0)
        self.worker_critic_optimizer.step()

        new_a, lp = self.worker_actor(z_curr, z_sub, p_curr)
        q1n2, q2n2 = self.worker_critic(z_curr, z_sub, p_curr, new_a)
        actor_loss = (self.alpha * lp - torch.min(q1n2, q2n2)).mean()
        self.worker_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_actor.parameters(), 1.0)
        self.worker_actor_optimizer.step()

        if self.config.worker.auto_alpha:
            alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.log_alpha.data.clamp_(min=np.log(0.01))

        self._soft_update(self.worker_critic, self.worker_critic_target, self.config.worker.tau)
        return {
            'worker_critic_loss': critic_loss.item(),
            'worker_actor_loss':  actor_loss.item(),
            'worker_alpha':       self.alpha.item(),
        }

    def update_reachability(self) -> dict:
        if self.reach_buffer.size < self.config.reachability.min_buffer_size:
            return {}
        batch  = self.reach_buffer.sample_balanced(self.config.reachability.batch_size)
        z_curr = torch.from_numpy(batch['z_current']).to(self.device)
        z_sub  = torch.from_numpy(batch['z_subgoal']).to(self.device)
        label  = torch.from_numpy(batch['label']).unsqueeze(1).to(self.device)
        pred   = self.reachability(z_curr, z_sub)
        loss   = F.binary_cross_entropy(pred, label)
        self.reachability_optimizer.zero_grad()
        loss.backward()
        self.reachability_optimizer.step()
        acc = ((pred > 0.5).float() == label).float().mean().item()
        return {'reach_loss': loss.item(), 'reach_accuracy': acc}

    def _soft_update(self, source, target, tau):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    # =========================================================================
    # Success threshold calibration for L2-normalised 2048-d space
    # =========================================================================

    def calibrate_success_threshold(self):
        """
        Calibrate success_threshold for L2-normalised R3M (2048-d) space.

        All features are on the unit hypersphere so distances are in [0, 2].
        We set the threshold as a multiple of the mean step distance:
          threshold = mean_step_dist * K_steps
        where K_steps is how many directed steps the worker must take to
        count as 'reached'. We choose K_steps = subgoal_horizon / 4,
        meaning the worker must cover 25% of the typical K-step distance.

        This is more principled than percentile-of-landmark-distances for
        two reasons:
          1. In 2048-d, FPS landmarks are very spread out (the space is huge).
             p10 of inter-landmark distances would be enormous.
          2. The worker's capability is measured in steps, not landmark geometry.

        Hard bounds:
          lower: 2 * mean_step_dist (at least 2 steps of motion required)
          upper: 0.5 (half the diameter of the unit hypersphere)
        """
        if len(self._latent_dists) < 50:
            print("  [Threshold] Not enough data — keeping default threshold.")
            return

        d = np.array(self._latent_dists)
        mean_step = float(d.mean())
        std_step  = float(d.std())

        # Threshold = K/4 steps of directed motion
        K = self.config.manager.subgoal_horizon
        target = mean_step * (K / 4.0)

        # Hard bounds
        lower = mean_step * 2.0    # must take at least 2 directed steps
        upper = 0.5                # at most 0.5 (quarter of max distance 2.0)

        self.success_threshold = float(np.clip(target, lower, upper))

        print(f"  [Threshold] Calibrated success_threshold = {self.success_threshold:.4f}")
        print(f"    Step dist — mean={mean_step:.4f}  std={std_step:.4f}  "
              f"p10={np.percentile(d,10):.4f}  p90={np.percentile(d,90):.4f}")
        print(f"    K={K}  target={target:.4f}  "
              f"lower={lower:.4f}  upper={upper:.4f}")
        if mean_step < 0.01:
            print("  WARNING: step distances very small — check L2 normalisation.")
        elif mean_step > 0.3:
            print("  WARNING: step distances large — encoder may not be normalised.")
