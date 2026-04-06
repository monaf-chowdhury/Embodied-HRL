import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from encoder import VisualEncoder
from env_wrapper import FrankaKitchenImageWrapper, HierarchicalKitchenWrapper
from landmarks import LandmarkBuffer
from networks import ManagerQNetwork, SACActorNetwork, SACCriticNetwork, ReachabilityPredictor
from buffers import HighLevelFERBuffer, LowLevelBuffer, ReachabilityBuffer
import d4rl  # Must import to register kitchen envs with gym


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
        self.manager_q = ManagerQNetwork(z_dim, config.manager.hidden_dim, 
                                          config.manager.n_layers).to(self.device)
        self.manager_q_target = ManagerQNetwork(z_dim, config.manager.hidden_dim,
                                                 config.manager.n_layers).to(self.device)
        self.manager_q_target.load_state_dict(self.manager_q.state_dict())
        self.manager_optimizer = torch.optim.Adam(
            list(self.manager_q.parameters()) + list(self.encoder.get_trainable_params()),
            lr=config.manager.lr,
        )
        
        # ---- Worker (SAC) ----
        action_dim = 9  # Franka Kitchen
        self.worker_actor = SACActorNetwork(z_dim, action_dim, 
                                             config.worker.hidden_dim,
                                             config.worker.n_layers).to(self.device)
        self.worker_critic = SACCriticNetwork(z_dim, action_dim,
                                               config.worker.hidden_dim,
                                               config.worker.n_layers).to(self.device)
        self.worker_critic_target = SACCriticNetwork(z_dim, action_dim,
                                                      config.worker.hidden_dim,
                                                      config.worker.n_layers).to(self.device)
        self.worker_critic_target.load_state_dict(self.worker_critic.state_dict())
        
        self.worker_actor_optimizer = torch.optim.Adam(
            self.worker_actor.parameters(), lr=config.worker.actor_lr)
        self.worker_critic_optimizer = torch.optim.Adam(
            self.worker_critic.parameters(), lr=config.worker.critic_lr)
        
        # SAC entropy
        if config.worker.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(config.worker.init_alpha), 
                requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.worker.alpha_lr)
            self.target_entropy = -action_dim  # Standard heuristic
        else:
            self.log_alpha = torch.tensor(np.log(config.worker.init_alpha), device=self.device)
        
        # ---- Reachability Predictor ----
        self.reachability = ReachabilityPredictor(z_dim, config.reachability.hidden_dim,
                                                   config.reachability.n_layers).to(self.device)
        self.reachability_optimizer = torch.optim.Adam(
            self.reachability.parameters(), lr=config.reachability.lr)
        
        # ---- Replay Buffers ----
        self.high_buffer = HighLevelFERBuffer(capacity=100_000, z_dim=z_dim)
        self.low_buffer = LowLevelBuffer(capacity=config.buffer.capacity, 
                                          z_dim=z_dim, action_dim=action_dim)
        self.reach_buffer = ReachabilityBuffer(capacity=100_000, z_dim=z_dim)
        
        # ---- Tracking ----
        self.total_steps = 0
        self.total_episodes = 0
        self.epsilon = config.manager.epsilon_start
        
        # For calibrating success threshold in latent space
        self._latent_dists = []
        self._success_threshold_calibrated = False
        self.success_threshold = 5.0  # Will be auto-calibrated
    
    @property
    def alpha(self):
        return self.log_alpha.exp().detach()
    
    # =====================================================================
    # Manager: select landmark subgoal
    # =====================================================================
    
    def select_subgoal(self, z_current: np.ndarray, z_goal: np.ndarray) -> int:
        """
        Select a landmark index as the next subgoal.
        Uses epsilon-greedy with the Q-network + reachability filtering.
        """
        if not self.landmarks.is_ready:
            return 0  # Fallback
        
        # Exploration: select least-visited landmark
        if np.random.random() < self.config.landmarks.explore_ratio:
            return self.landmarks.select_explore()
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.landmarks.n_active)
        
        # Greedy: evaluate Q for all landmarks
        with torch.no_grad():
            z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
            z_g = torch.from_numpy(z_goal).float().unsqueeze(0).to(self.device)
            lm = torch.from_numpy(self.landmarks.get_all()).float().to(self.device)
            
            q_values = self.manager_q.evaluate_all_landmarks(z_c, z_g, lm)  # (1, N)
            
            # Filter by reachability
            if self.reach_buffer.size > self.config.reachability.min_buffer_size:
                z_c_exp = z_c.expand(self.landmarks.n_active, -1)
                reach_probs = self.reachability(z_c_exp, lm).squeeze(-1)  # (N,)
                # Mask out unreachable landmarks
                mask = reach_probs < self.config.reachability.reject_threshold
                q_values[0, mask] = -1e9
            
            return q_values.argmax(dim=1).item()
    
    def _update_epsilon(self):
        """Decay epsilon."""
        cfg = self.config.manager
        self.epsilon = max(
            cfg.epsilon_end,
            cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) 
            * self.total_steps / cfg.epsilon_decay_steps
        )
    
    # =====================================================================
    # Worker: get action
    # =====================================================================
    
    def get_worker_action(self, z_current: np.ndarray, z_subgoal: np.ndarray, 
                          deterministic: bool = False) -> np.ndarray:
        """Get low-level action from SAC actor."""
        z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
        z_s = torch.from_numpy(z_subgoal).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action = self.worker_actor.get_action_deterministic(z_c, z_s)
            else:
                action, _ = self.worker_actor(z_c, z_s)
        
        return action.cpu().numpy().squeeze()
    
    # =====================================================================
    # Training updates
    # =====================================================================
    
    def update_manager(self):
        """Update manager Q-network from FER buffer."""
        if self.high_buffer.size < self.config.buffer.batch_size:
            return {}
        
        batch = self.high_buffer.sample(self.config.buffer.batch_size)
        
        z_curr = torch.from_numpy(batch['z_current']).to(self.device)
        z_goal = torch.from_numpy(batch['z_goal']).to(self.device)
        z_sub = torch.from_numpy(batch['z_subgoal']).to(self.device)
        reward = torch.from_numpy(batch['reward']).unsqueeze(1).to(self.device)
        z_next = torch.from_numpy(batch['z_next']).to(self.device)
        done = torch.from_numpy(batch['done']).unsqueeze(1).to(self.device)
        
        # Current Q
        q_current = self.manager_q(z_curr, z_goal, z_sub)
        
        # Target Q (for non-terminal transitions, evaluate best landmark from z_next)
        with torch.no_grad():
            if self.landmarks.is_ready:
                lm = torch.from_numpy(self.landmarks.get_all()).float().to(self.device)
                q_next_all = self.manager_q_target.evaluate_all_landmarks(
                    z_next, z_goal, lm)  # (B, N)
                q_next_max = q_next_all.max(dim=1, keepdim=True)[0]
            else:
                q_next_max = torch.zeros_like(reward)
            
            target = reward + self.config.manager.gamma * (1 - done) * q_next_max
        
        loss = F.mse_loss(q_current, target)
        
        self.manager_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_q.parameters(), 1.0)
        self.manager_optimizer.step()
        
        # Soft update target
        self._soft_update(self.manager_q, self.manager_q_target, self.config.manager.tau)
        
        return {'manager_loss': loss.item(), 'manager_q_mean': q_current.mean().item()}
    
    def update_worker(self):
        """Update worker SAC (actor + critic)."""
        if self.low_buffer.size < self.config.buffer.batch_size:
            return {}
        
        batch = self.low_buffer.sample(self.config.buffer.batch_size)
        
        z_curr = torch.from_numpy(batch['z_current']).to(self.device)
        z_sub = torch.from_numpy(batch['z_subgoal']).to(self.device)
        action = torch.from_numpy(batch['action']).to(self.device)
        reward = torch.from_numpy(batch['reward']).unsqueeze(1).to(self.device)
        z_next = torch.from_numpy(batch['z_next']).to(self.device)
        done = torch.from_numpy(batch['done']).unsqueeze(1).to(self.device)
        
        # ---- Critic update ----
        with torch.no_grad():
            next_action, next_log_prob = self.worker_actor(z_next, z_sub)
            q1_next, q2_next = self.worker_critic_target(z_next, z_sub, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + self.config.worker.gamma * (1 - done) * q_next
        
        q1, q2 = self.worker_critic(z_curr, z_sub, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.worker_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(), 1.0)
        self.worker_critic_optimizer.step()
        
        # ---- Actor update ----
        new_action, log_prob = self.worker_actor(z_curr, z_sub)
        q1_new, q2_new = self.worker_critic(z_curr, z_sub, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.worker_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_actor.parameters(), 1.0)
        self.worker_actor_optimizer.step()
        
        # ---- Alpha update ----
        alpha_loss_val = 0.0
        if self.config.worker.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()
        
        # Soft update target
        self._soft_update(self.worker_critic, self.worker_critic_target, self.config.worker.tau)
        
        return {
            'worker_critic_loss': critic_loss.item(),
            'worker_actor_loss': actor_loss.item(),
            'worker_alpha': self.alpha.item(),
        }
    
    def update_reachability(self):
        """Update reachability predictor from collected data."""
        if self.reach_buffer.size < self.config.reachability.min_buffer_size:
            return {}
        
        batch = self.reach_buffer.sample_balanced(self.config.reachability.batch_size)
        
        z_curr = torch.from_numpy(batch['z_current']).to(self.device)
        z_sub = torch.from_numpy(batch['z_subgoal']).to(self.device)
        label = torch.from_numpy(batch['label']).unsqueeze(1).to(self.device)
        
        pred = self.reachability(z_curr, z_sub)
        loss = F.binary_cross_entropy(pred, label)
        
        self.reachability_optimizer.zero_grad()
        loss.backward()
        self.reachability_optimizer.step()
        
        accuracy = ((pred > 0.5).float() == label).float().mean().item()
        
        return {'reach_loss': loss.item(), 'reach_accuracy': accuracy}
    
    def _soft_update(self, source, target, tau):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
    
    # =====================================================================
    # Reward shaping
    # =====================================================================
    
    def compute_shaped_reward(self, z_t, z_next, z_subgoal, sparse_reward):
        """
        L2 delta-progress potential-based shaping.
        r = r_sparse + alpha * (||z_t - z_sub|| - ||z_next - z_sub||)
        """
        dist_before = np.linalg.norm(z_t - z_subgoal)
        dist_after = np.linalg.norm(z_next - z_subgoal)
        delta = dist_before - dist_after  # Positive = got closer
        
        shaped = sparse_reward + self.config.reward.shaping_weight * delta
        return shaped
    
    # =====================================================================
    # Calibration
    # =====================================================================
    
    def calibrate_success_threshold(self):
        """
        Auto-calibrate the success threshold based on observed latent distances.
        Called after collecting initial data.
        """
        if len(self._latent_dists) < 100:
            return
        
        dists = np.array(self._latent_dists)
        # Success threshold: ~25th percentile of K-step latent distances
        self.success_threshold = np.percentile(dists, 25)
        self._success_threshold_calibrated = True
        print(f"Calibrated success threshold: {self.success_threshold:.4f}")
        print(f"  Latent dist stats: mean={dists.mean():.4f}, "
              f"std={dists.std():.4f}, min={dists.min():.4f}, max={dists.max():.4f}")

