import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from encoder import VisualEncoder
from env_wrapper import FrankaKitchenImageWrapper, HierarchicalKitchenWrapper
from landmarks import LandmarkBuffer
from networks import ManagerQNetwork, SACActorNetwork, SACCriticNetwork, ReachabilityPredictor
from buffers import HighLevelFERBuffer, LowLevelBuffer, ReachabilityBuffer


class VisualHRLAgent:
    """Full Visual HRL agent with all components."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        z_dim = config.encoder.proj_dim

        # ---- Visual Encoder ----
        self.encoder = VisualEncoder(config.encoder, device=self.device)

        # ---- Landmarks ----
        self.landmarks = LandmarkBuffer(
            n_landmarks=config.landmarks.n_landmarks,
            z_dim=z_dim,
        )

        # ---- Manager (DQN over landmarks) ----
        self.manager_q = ManagerQNetwork(
            z_dim, config.manager.hidden_dim, config.manager.n_layers
        ).to(self.device)
        self.manager_q_target = ManagerQNetwork(
            z_dim, config.manager.hidden_dim, config.manager.n_layers
        ).to(self.device)
        self.manager_q_target.load_state_dict(self.manager_q.state_dict())
        self.manager_optimizer = torch.optim.Adam(
            list(self.manager_q.parameters()) + list(self.encoder.get_trainable_params()),
            lr=config.manager.lr,
        )

        # ---- Worker (SAC) ----
        action_dim = 9  # Franka Kitchen: 9-DOF joint velocity control
        self.worker_actor = SACActorNetwork(
            z_dim, action_dim, config.worker.hidden_dim, config.worker.n_layers
        ).to(self.device)
        self.worker_critic = SACCriticNetwork(
            z_dim, action_dim, config.worker.hidden_dim, config.worker.n_layers
        ).to(self.device)
        self.worker_critic_target = SACCriticNetwork(
            z_dim, action_dim, config.worker.hidden_dim, config.worker.n_layers
        ).to(self.device)
        self.worker_critic_target.load_state_dict(self.worker_critic.state_dict())

        self.worker_actor_optimizer = torch.optim.Adam(
            self.worker_actor.parameters(), lr=config.worker.actor_lr)
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

        # ---- Reachability Predictor ----
        self.reachability = ReachabilityPredictor(
            z_dim, config.reachability.hidden_dim, config.reachability.n_layers
        ).to(self.device)
        self.reachability_optimizer = torch.optim.Adam(
            self.reachability.parameters(), lr=config.reachability.lr)

        # ---- Replay Buffers ----
        self.high_buffer  = HighLevelFERBuffer(capacity=100_000, z_dim=z_dim)
        self.low_buffer   = LowLevelBuffer(
            capacity=config.buffer.capacity, z_dim=z_dim, action_dim=action_dim)
        self.reach_buffer = ReachabilityBuffer(capacity=100_000, z_dim=z_dim)

        # ---- Tracking ----
        self.total_steps   = 0
        self.total_episodes = 0
        self.epsilon       = config.manager.epsilon_start

        # Latent distance calibration
        self._latent_dists = []
        self._success_threshold_calibrated = False
        self.success_threshold = 5.0  # auto-calibrated after warmup

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    # =========================================================================
    # Manager: subgoal selection
    # =========================================================================

    def select_subgoal(self, z_current: np.ndarray, z_goal: np.ndarray) -> int:
        """Select a landmark index as the next subgoal (epsilon-greedy + reachability filter)."""
        if not self.landmarks.is_ready:
            return 0

        if np.random.random() < self.config.landmarks.explore_ratio:
            return self.landmarks.select_explore()

        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.landmarks.n_active)

        with torch.no_grad():
            z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
            z_g = torch.from_numpy(z_goal).float().unsqueeze(0).to(self.device)
            lm  = torch.from_numpy(self.landmarks.get_all()).float().to(self.device)
            q_values = self.manager_q.evaluate_all_landmarks(z_c, z_g, lm)

            if self.reach_buffer.size > self.config.reachability.min_buffer_size:
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
    # Worker: action selection
    # =========================================================================

    def get_worker_action(
        self,
        z_current: np.ndarray,
        z_subgoal: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
        z_s = torch.from_numpy(z_subgoal).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.worker_actor.get_action_deterministic(z_c, z_s)
            else:
                action, _ = self.worker_actor(z_c, z_s)
        return action.cpu().numpy().squeeze()

    # =========================================================================
    # Training updates
    # =========================================================================

    def update_manager(self) -> dict:
        if self.high_buffer.size < self.config.buffer.batch_size:
            return {}
        batch  = self.high_buffer.sample(self.config.buffer.batch_size)
        z_curr = torch.from_numpy(batch['z_current']).to(self.device)
        z_goal = torch.from_numpy(batch['z_goal']).to(self.device)
        z_sub  = torch.from_numpy(batch['z_subgoal']).to(self.device)
        reward = torch.from_numpy(batch['reward']).unsqueeze(1).to(self.device)
        z_next = torch.from_numpy(batch['z_next']).to(self.device)
        done   = torch.from_numpy(batch['done']).unsqueeze(1).to(self.device)

        q_current = self.manager_q(z_curr, z_goal, z_sub)

        with torch.no_grad():
            if self.landmarks.is_ready:
                lm = torch.from_numpy(self.landmarks.get_all()).float().to(self.device)
                q_next_all = self.manager_q_target.evaluate_all_landmarks(z_next, z_goal, lm)
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

    def update_worker(self) -> dict:
        if self.low_buffer.size < self.config.buffer.batch_size:
            return {}
        batch  = self.low_buffer.sample(self.config.buffer.batch_size)
        z_curr = torch.from_numpy(batch['z_current']).to(self.device)
        z_sub  = torch.from_numpy(batch['z_subgoal']).to(self.device)
        action = torch.from_numpy(batch['action']).to(self.device)
        reward = torch.from_numpy(batch['reward']).unsqueeze(1).to(self.device)
        z_next = torch.from_numpy(batch['z_next']).to(self.device)
        done   = torch.from_numpy(batch['done']).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.worker_actor(z_next, z_sub)
            q1_next, q2_next = self.worker_critic_target(z_next, z_sub, next_action)
            q_next    = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q  = reward + self.config.worker.gamma * (1 - done) * q_next

        q1, q2 = self.worker_critic(z_curr, z_sub, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.worker_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(), 1.0)
        self.worker_critic_optimizer.step()

        new_action, log_prob = self.worker_actor(z_curr, z_sub)
        q1_new, q2_new = self.worker_critic(z_curr, z_sub, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        self.worker_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_actor.parameters(), 1.0)
        self.worker_actor_optimizer.step()

        alpha_loss_val = 0.0
        if self.config.worker.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

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
    # Reward shaping
    # =========================================================================

    def compute_shaped_reward(
        self,
        z_t: np.ndarray,
        z_next: np.ndarray,
        z_subgoal: np.ndarray,
        sparse_reward: float,
    ) -> float:
        """L2 delta-progress potential-based shaping."""
        dist_before = np.linalg.norm(z_t    - z_subgoal)
        dist_after  = np.linalg.norm(z_next - z_subgoal)
        delta  = dist_before - dist_after
        return sparse_reward + self.config.reward.shaping_weight * delta

    # =========================================================================
    # Calibration
    # =========================================================================

    def calibrate_success_threshold(self):
        """Auto-calibrate success threshold from warmup latent distances."""
        if len(self._latent_dists) < 100:
            return
        dists = np.array(self._latent_dists)
        self.success_threshold = np.percentile(dists, 25)
        self._success_threshold_calibrated = True
        print(f"Calibrated success threshold: {self.success_threshold:.4f}")
        print(f"  Latent dist — mean={dists.mean():.4f}  std={dists.std():.4f}"
              f"  min={dists.min():.4f}  max={dists.max():.4f}")