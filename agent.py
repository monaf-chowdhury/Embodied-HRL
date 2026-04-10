"""
VisualHRLAgent — full Visual HRL agent.

Key changes vs original:
  1. z_goal REMOVED everywhere.  The manager no longer receives a global
     goal encoding.  It receives only z_current and z_landmark.
     ManagerQNetwork input is now 2*z_dim (not 3*z_dim).

  2. Worker uses dual-stream architecture: images + normalised proprio.
     get_worker_action() and update_worker() both require proprio.

  3. Reward reshaping:
       r = sparse_weight * r_sparse
           + task_progress_weight * task_progress (proprio-based)
           + latent_weight * normalised_delta_progress
     Weights: 5.0 / 0.5 / 0.1 (from RewardConfig).

  4. Hindsight landmark injection: whenever n_tasks_completed increases,
     the current latent is added to the landmark success pool.

  5. ManagerQNetwork now takes 2*z_dim input.
"""
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from encoder import VisualEncoder
from env_wrapper import FrankaKitchenImageWrapper
from landmarks import LandmarkBuffer
from networks import ManagerQNetwork, SACActorNetwork, SACCriticNetwork, ReachabilityPredictor
from buffers import HighLevelFERBuffer, LowLevelBuffer, ReachabilityBuffer


# =============================================================================
# Task-progress helper
# =============================================================================

# OBS_ELEMENT_INDICES and GOALS from D4RL kitchen_envs.py
# obs[0:21] = qpos[9:30] (object joints, zero-indexed into obs vector)
_TASK_OBS_IDX = {
    'bottom burner': (np.array([2, 3]),    np.array([-0.88, -0.01])),
    'top burner':    (np.array([6, 7]),    np.array([-0.92, -0.01])),
    'light switch':  (np.array([8, 9]),    np.array([-0.69, -0.05])),
    'slide cabinet': (np.array([10]),      np.array([0.37])),
    'hinge cabinet': (np.array([11, 12]),  np.array([0., 1.45])),
    'microwave':     (np.array([13]),      np.array([-0.75])),
    'kettle':        (np.array([14, 15, 16, 17, 18, 19, 20]),
                      np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06])),
}


def compute_task_progress(proprio: np.ndarray, tasks: list) -> float:
    """
    Compute a [0,1] progress scalar toward task completion using proprio.
    Progress = average over tasks of max(0, 1 - normalised_dist_to_goal).
    A value of 1.0 means all tasks are exactly at goal.
    """
    if not tasks:
        return 0.0
    scores = []
    for task in tasks:
        if task not in _TASK_OBS_IDX:
            continue
        idx, goal = _TASK_OBS_IDX[task]
        current   = proprio[idx]
        dist      = np.linalg.norm(current - goal)
        # Normalise by the max possible distance for this task
        max_dist  = np.linalg.norm(goal) + 1e-4
        progress  = max(0.0, 1.0 - dist / max_dist)
        scores.append(progress)
    return float(np.mean(scores)) if scores else 0.0


class VisualHRLAgent:
    """Full Visual HRL agent."""

    def __init__(self, config: Config):
        self.config  = config
        self.device  = config.training.device
        z_dim        = config.encoder.proj_dim
        proprio_dim  = config.worker.proprio_dim
        action_dim   = 9   # Franka Kitchen 9-DOF

        # ---- Encoder ----
        self.encoder = VisualEncoder(config.encoder, device=self.device)

        # ---- Landmarks ----
        self.landmarks = LandmarkBuffer(
            n_landmarks=config.landmarks.n_landmarks,
            z_dim=z_dim,
            landmark_config=config.landmarks,
        )

        # ---- Manager (DQN over landmarks, image latent only) ----
        # Input: [z_current, z_landmark] = 2 * z_dim
        self.manager_q = ManagerQNetwork(
            z_dim, config.manager.hidden_dim, config.manager.n_layers,
            input_multiplier=2,   # 2*z_dim (no z_goal)
        ).to(self.device)
        self.manager_q_target = ManagerQNetwork(
            z_dim, config.manager.hidden_dim, config.manager.n_layers,
            input_multiplier=2,
        ).to(self.device)
        self.manager_q_target.load_state_dict(self.manager_q.state_dict())
        self.manager_optimizer = torch.optim.Adam(
            list(self.manager_q.parameters()) + list(self.encoder.get_trainable_params()),
            lr=config.manager.lr,
        )

        # ---- Worker (SAC, dual-stream) ----
        self.worker_actor = SACActorNetwork(
            z_dim, action_dim,
            config.worker.hidden_dim, config.worker.n_layers,
            proprio_dim=proprio_dim,
        ).to(self.device)
        self.worker_critic = SACCriticNetwork(
            z_dim, action_dim,
            config.worker.hidden_dim, config.worker.n_layers,
            proprio_dim=proprio_dim,
        ).to(self.device)
        self.worker_critic_target = SACCriticNetwork(
            z_dim, action_dim,
            config.worker.hidden_dim, config.worker.n_layers,
            proprio_dim=proprio_dim,
        ).to(self.device)
        self.worker_critic_target.load_state_dict(self.worker_critic.state_dict())

        self.worker_actor_optimizer  = torch.optim.Adam(
            self.worker_actor.parameters(),  lr=config.worker.actor_lr)
        self.worker_critic_optimizer = torch.optim.Adam(
            self.worker_critic.parameters(), lr=config.worker.critic_lr)

        # SAC entropy temperature
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

        # ---- Reachability ----
        self.reachability = ReachabilityPredictor(
            z_dim, config.reachability.hidden_dim, config.reachability.n_layers
        ).to(self.device)
        self.reachability_optimizer = torch.optim.Adam(
            self.reachability.parameters(), lr=config.reachability.lr)

        # ---- Replay buffers ----
        self.high_buffer  = HighLevelFERBuffer(capacity=100_000, z_dim=z_dim)
        self.low_buffer   = LowLevelBuffer(
            capacity=config.buffer.capacity,
            z_dim=z_dim,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
        )
        self.reach_buffer = ReachabilityBuffer(capacity=100_000, z_dim=z_dim)

        # ---- Tracking ----
        self.total_steps    = 0
        self.total_episodes = 0
        self.epsilon        = config.manager.epsilon_start

        self._latent_dists = []
        self._success_threshold_calibrated = False
        self.success_threshold = 5.0

        # Track previous n_tasks_completed for hindsight injection
        self._prev_n_tasks = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    # =========================================================================
    # Manager — no z_goal
    # =========================================================================

    def select_subgoal(self, z_current: np.ndarray) -> int:
        """Select a landmark index (epsilon-greedy + reachability filter)."""
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

            if (self.reach_buffer.size > self.config.reachability.min_buffer_size
                    and self.total_steps > 200_000):
                z_c_exp    = z_c.expand(self.landmarks.n_active, -1)
                reach_probs = self.reachability(z_c_exp, lm).squeeze(-1)
                mask = reach_probs < self.config.reachability.reject_threshold
                q_values[0, mask] = -1e9

            return q_values.argmax(dim=1).item()

    def _update_epsilon(self):
        cfg = self.config.manager
        self.epsilon = max(
            cfg.epsilon_end,
            cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end)
            * self.total_steps / cfg.epsilon_decay_steps
        )

    # =========================================================================
    # Worker — now requires proprio
    # =========================================================================

    def get_worker_action(
        self,
        z_current: np.ndarray,
        z_subgoal: np.ndarray,
        proprio: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        z_c  = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
        z_s  = torch.from_numpy(z_subgoal).float().unsqueeze(0).to(self.device)
        p    = torch.from_numpy(
            self.low_buffer.normalise_proprio(proprio)
        ).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                action = self.worker_actor.get_action_deterministic(z_c, z_s, p)
            else:
                action, _ = self.worker_actor(z_c, z_s, p)
        return action.cpu().numpy().squeeze()

    # =========================================================================
    # Hindsight landmark injection
    # =========================================================================

    def maybe_inject_hindsight(self, z_current: np.ndarray, n_tasks_completed: int):
        """
        If n_tasks_completed just increased, add z_current to the landmark
        hindsight pool.  This captures accidental task-completion states.
        """
        if (self.config.landmarks.use_hindsight_landmarks
                and n_tasks_completed > self._prev_n_tasks):
            self.landmarks.add_success_state(z_current)
        self._prev_n_tasks = n_tasks_completed

    # =========================================================================
    # Reward shaping (task-progress-aware)
    # =========================================================================

    def compute_shaped_reward(
        self,
        z_t: np.ndarray,
        z_next: np.ndarray,
        z_subgoal: np.ndarray,
        sparse_reward: float,
        proprio_t: np.ndarray,
        proprio_next: np.ndarray,
        initial_dist: float = None,   # for normalised latent progress
    ) -> float:
        """
        r = sparse_weight * r_sparse
            + task_progress_weight * delta_task_progress
            + latent_weight * normalised_delta_progress

        delta_task_progress: change in proprioceptive distance to task goals.
        normalised_delta_progress: delta-L2 / initial_dist (scale-invariant).
        """
        cfg = self.config.reward

        # 1. Sparse (dominant)
        r = cfg.sparse_weight * sparse_reward

        # 2. Task-progress (secondary) — proprio-based
        prog_before = compute_task_progress(proprio_t,    self.config.training.tasks_to_complete)
        prog_after  = compute_task_progress(proprio_next, self.config.training.tasks_to_complete)
        delta_task  = prog_after - prog_before
        r += cfg.task_progress_weight * delta_task

        # 3. Latent progress (tertiary) — normalised by initial dist
        dist_before = np.linalg.norm(z_t    - z_subgoal)
        dist_after  = np.linalg.norm(z_next - z_subgoal)
        delta_lat   = dist_before - dist_after
        denom       = max(initial_dist, dist_before, 1e-4) if initial_dist else max(dist_before, 1e-4)
        r += cfg.latent_weight * (delta_lat / denom)

        return float(r)

    # =========================================================================
    # Manager update — no z_goal
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

        q_current = self.manager_q(z_curr, z_sub)   # (B, 1)

        with torch.no_grad():
            if self.landmarks.is_ready:
                lm = torch.from_numpy(self.landmarks.get_all()).float().to(self.device)
                q_next_all = self.manager_q_target.evaluate_all_landmarks(z_next, lm)
                q_next_max = q_next_all.max(dim=1, keepdim=True)[0]
            else:
                q_next_max = torch.zeros_like(reward)
            target = reward + self.config.manager.gamma * (1 - done) * q_next_max

        loss = F.mse_loss(q_current, target)
        self.manager_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_q.parameters(), 1.0)
        self.manager_optimizer.step()
        self._soft_update(self.manager_q, self.manager_q_target, self.config.manager.tau)

        return {'manager_loss': loss.item(), 'manager_q_mean': q_current.mean().item()}

    # =========================================================================
    # Worker update — dual stream
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

        # ---- Critic update ----
        with torch.no_grad():
            next_action, next_log_prob = self.worker_actor(z_next, z_sub, p_next)
            q1_next, q2_next = self.worker_critic_target(z_next, z_sub, p_next, next_action)
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + self.config.worker.gamma * (1 - done) * q_next

        q1, q2 = self.worker_critic(z_curr, z_sub, p_curr, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.worker_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(), 1.0)
        self.worker_critic_optimizer.step()

        # ---- Actor update ----
        new_action, log_prob = self.worker_actor(z_curr, z_sub, p_curr)
        q1_new, q2_new = self.worker_critic(z_curr, z_sub, p_curr, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        self.worker_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_actor.parameters(), 1.0)
        self.worker_actor_optimizer.step()

        # ---- Alpha update ----
        if self.config.worker.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
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
        accuracy = ((pred > 0.5).float() == label).float().mean().item()
        return {'reach_loss': loss.item(), 'reach_accuracy': accuracy}

    def _soft_update(self, source, target, tau):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    # =========================================================================
    # Calibration
    # =========================================================================

    def calibrate_success_threshold(self):
        if len(self._latent_dists) < 100:
            return
        dists = np.array(self._latent_dists)
        self.success_threshold = max(np.percentile(dists, 50), 0.5)
        self._success_threshold_calibrated = True
        print(f"Calibrated success threshold: {self.success_threshold:.4f}")
        print(f"  Latent dist — mean={dists.mean():.4f}  std={dists.std():.4f}"
              f"  min={dists.min():.4f}  max={dists.max():.4f}")