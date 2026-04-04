"""
Main Training Loop: Visual HRL for Franka Kitchen.

Implements the full pipeline:
1. Frozen R3M encoder + projection head
2. FPS landmark buffer
3. Manager (DQN) selects landmarks as subgoals
4. Worker (SAC) executes with L2 delta-progress shaping
5. Strict execution: success continues, failure terminates
6. Reachability predictor filters bad subgoals

Run: python train.py [--seed 42] [--device cuda]
"""
import os
import time
import argparse
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


def train(config: Config):
    """Main training function."""
    
    # ---- Setup ----
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    
    os.makedirs(config.training.log_dir, exist_ok=True)
    writer = SummaryWriter(config.training.log_dir)
    
    # Environment
    env = FrankaKitchenImageWrapper(
        task="kitchen-complete-v0",
        img_size=config.encoder.img_size,
        seed=config.training.seed,
    )
    hier_env = HierarchicalKitchenWrapper(env, subgoal_horizon=config.manager.subgoal_horizon)
    
    # Agent
    agent = VisualHRLAgent(config)
    
    # We need a goal image. In Kitchen, the goal is implicit (complete all 4 tasks).
    # We'll use a dummy goal encoding — the manager learns what "goal" means from rewards.
    # In practice, you could render a goal image or use the final observation from demos.
    z_goal = np.zeros(config.encoder.proj_dim, dtype=np.float32)  # Placeholder
    # TODO: Replace with actual goal encoding from a demonstration or target image
    
    print("=" * 60)
    print("Visual HRL Training — Franka Kitchen")
    print(f"Device: {config.training.device}")
    print(f"Encoder: {config.encoder.name} (frozen={config.encoder.freeze})")
    print(f"Landmarks: {config.landmarks.n_landmarks}")
    print(f"Subgoal horizon: {config.manager.subgoal_horizon} steps")
    print(f"Total timesteps: {config.training.total_timesteps}")
    print("=" * 60)
    
    # ---- Phase 1: Random exploration to collect initial data ----
    print("\nPhase 1: Random exploration for initial data...")
    
    n_warmup_episodes = 50
    for ep_i in range(n_warmup_episodes):
        obs_img = hier_env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_img, reward, done, info = env.step(action)
            
            z_t = agent.encoder.encode_numpy(obs_img).squeeze()
            z_next = agent.encoder.encode_numpy(next_img).squeeze()
            
            # Store in low-level buffer (random subgoal for now)
            z_random_sub = z_next  # Self-supervised: future state as subgoal
            shaped_r = agent.compute_shaped_reward(z_t, z_next, z_random_sub, reward)
            agent.low_buffer.add(z_t, z_random_sub, action, shaped_r, z_next, done)
            
            # Track latent distances for calibration
            agent._latent_dists.append(np.linalg.norm(z_next - z_t))
            
            obs_img = next_img
    
    print(f"  Collected {len(agent.low_buffer)} low-level transitions")
    
    # Compute initial landmarks
    all_z = agent.low_buffer.get_all_z()
    agent.landmarks.update(all_z)
    print(f"  Computed {agent.landmarks.n_active} landmarks via FPS")
    
    # Calibrate success threshold
    agent.calibrate_success_threshold()
    
    # Get a meaningful goal encoding
    # Use the last observation of a successful demo or just the mean of all z
    z_goal = all_z.mean(axis=0)  # Rough placeholder
    
    # ---- Phase 2: Hierarchical training ----
    print("\nPhase 2: Hierarchical training loop...")
    
    best_eval_success = 0.0
    pbar = tqdm(total=config.training.total_timesteps, desc="Training")
    
    while agent.total_steps < config.training.total_timesteps:
        obs_img = hier_env.reset()
        z_current = agent.encoder.encode_numpy(obs_img).squeeze()
        episode_reward = 0.0
        episode_high_steps = 0
        episode_successes = 0
        episode_done = False
        
        while not episode_done:
            # ---- Manager selects subgoal ----
            landmark_idx = agent.select_subgoal(z_current, z_goal)
            z_subgoal = agent.landmarks.get(landmark_idx)
            
            # ---- Worker executes for K steps ----
            z_start = z_current.copy()
            cumulative_env_reward = 0.0
            worker_steps = 0
            subgoal_reached = False
            
            for k_step in range(config.manager.subgoal_horizon):
                action = agent.get_worker_action(z_current, z_subgoal)
                next_img, env_reward, env_done, info = env.step(action)
                z_next = agent.encoder.encode_numpy(next_img).squeeze()
                
                # Shaped reward for worker
                shaped_r = agent.compute_shaped_reward(z_current, z_next, z_subgoal, env_reward)
                
                # Store worker transition
                agent.low_buffer.add(z_current, z_subgoal, action, shaped_r, z_next, env_done)
                
                cumulative_env_reward += env_reward
                worker_steps += 1
                agent.total_steps += 1
                pbar.update(1)
                
                z_current = z_next
                obs_img = next_img
                
                # Check subgoal reached
                dist_to_subgoal = np.linalg.norm(z_current - z_subgoal)
                if dist_to_subgoal < agent.success_threshold:
                    subgoal_reached = True
                    break
                
                if env_done:
                    break
                
                # Update worker periodically
                if agent.total_steps % 4 == 0 and agent.low_buffer.size > config.buffer.batch_size:
                    worker_metrics = agent.update_worker()
            
            # ---- Store high-level transition (FER) ----
            episode_high_steps += 1
            
            # Store reachability data
            agent.reach_buffer.add(z_start, z_subgoal, subgoal_reached)
            agent.landmarks.record_visit(landmark_idx, success=subgoal_reached)
            
            if subgoal_reached:
                # Success: store with cumulative reward, episode continues
                agent.high_buffer.add_success(
                    z_start, z_goal, z_subgoal, cumulative_env_reward,
                    z_current, landmark_idx)
                episode_successes += 1
                episode_reward += cumulative_env_reward
            else:
                # Failure: store with zero reward, TERMINATE episode (strict execution)
                agent.high_buffer.add_failure(z_start, z_goal, z_subgoal, landmark_idx)
                episode_done = True
            
            if env_done:
                episode_done = True
            
            # ---- Update manager ----
            if agent.high_buffer.size > config.buffer.batch_size:
                manager_metrics = agent.update_manager()
            
            # ---- Update epsilon ----
            agent._update_epsilon()
        
        # ---- End of episode ----
        agent.total_episodes += 1
        
        # Update reachability predictor
        if agent.total_episodes % config.reachability.update_freq == 0:
            for _ in range(5):  # Multiple gradient steps
                reach_metrics = agent.update_reachability()
        
        # Update landmarks
        if agent.total_episodes % config.landmarks.update_freq == 0:
            all_z = agent.low_buffer.get_all_z()
            if len(all_z) > config.landmarks.min_observations:
                agent.landmarks.update(all_z)
        
        # Logging
        if agent.total_episodes % 10 == 0:
            writer.add_scalar('train/episode_reward', episode_reward, agent.total_steps)
            writer.add_scalar('train/high_level_steps', episode_high_steps, agent.total_steps)
            writer.add_scalar('train/subgoal_successes', episode_successes, agent.total_steps)
            writer.add_scalar('train/epsilon', agent.epsilon, agent.total_steps)
            writer.add_scalar('train/low_buffer_size', len(agent.low_buffer), agent.total_steps)
            
            if hasattr(agent, '_last_reach_metrics') and agent._last_reach_metrics:
                for k, v in agent._last_reach_metrics.items():
                    writer.add_scalar(f'reachability/{k}', v, agent.total_steps)
        
        # ---- Evaluation ----
        if agent.total_steps % config.training.eval_freq < config.manager.subgoal_horizon * 20:
            eval_success = evaluate(agent, config, z_goal)
            writer.add_scalar('eval/success_rate', eval_success, agent.total_steps)
            
            if eval_success > best_eval_success:
                best_eval_success = eval_success
                save_checkpoint(agent, config, 'best')
            
            print(f"\n[Step {agent.total_steps}] Eval success: {eval_success:.3f} "
                  f"(best: {best_eval_success:.3f}) | "
                  f"eps: {agent.epsilon:.3f} | "
                  f"buffers: HL={len(agent.high_buffer)} LL={len(agent.low_buffer)}")
        
        # Save periodic checkpoint
        if agent.total_steps % config.training.save_freq < config.manager.subgoal_horizon * 20:
            save_checkpoint(agent, config, f'step_{agent.total_steps}')
    
    pbar.close()
    writer.close()
    print(f"\nTraining complete. Best eval success: {best_eval_success:.3f}")


def evaluate(agent, config, z_goal, n_episodes=None):
    """Evaluate the agent (deterministic, no exploration)."""
    if n_episodes is None:
        n_episodes = config.training.n_eval_episodes
    
    eval_env = FrankaKitchenImageWrapper(
        task="kitchen-complete-v0",
        img_size=config.encoder.img_size,
    )
    
    successes = 0
    total_rewards = []
    
    for ep_i in range(n_episodes):
        obs_img = eval_env.reset()
        z_current = agent.encoder.encode_numpy(obs_img).squeeze()
        ep_reward = 0.0
        done = False
        high_steps = 0
        
        while not done and high_steps < 15:  # Max 15 high-level steps
            # Greedy subgoal selection
            if agent.landmarks.is_ready:
                with torch.no_grad():
                    z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(agent.device)
                    z_g = torch.from_numpy(z_goal).float().unsqueeze(0).to(agent.device)
                    lm = torch.from_numpy(agent.landmarks.get_all()).float().to(agent.device)
                    q_vals = agent.manager_q.evaluate_all_landmarks(z_c, z_g, lm)
                    landmark_idx = q_vals.argmax(dim=1).item()
                z_subgoal = agent.landmarks.get(landmark_idx)
            else:
                break
            
            # Execute
            subgoal_reached = False
            for k in range(config.manager.subgoal_horizon):
                action = agent.get_worker_action(z_current, z_subgoal, deterministic=True)
                next_img, reward, done, info = eval_env.step(action)
                z_next = agent.encoder.encode_numpy(next_img).squeeze()
                ep_reward += reward
                z_current = z_next
                
                if np.linalg.norm(z_current - z_subgoal) < agent.success_threshold:
                    subgoal_reached = True
                    break
                if done:
                    break
            
            high_steps += 1
            if not subgoal_reached:
                break  # Strict execution: stop on failure
        
        total_rewards.append(ep_reward)
        if ep_reward > 0:  # Kitchen gives reward > 0 only for completing subtasks
            successes += 1
    
    eval_env.close()
    return successes / n_episodes


def save_checkpoint(agent, config, name):
    """Save all model weights."""
    path = os.path.join(config.training.log_dir, f'checkpoint_{name}.pt')
    torch.save({
        'encoder_projection': agent.encoder.projection.state_dict(),
        'manager_q': agent.manager_q.state_dict(),
        'manager_q_target': agent.manager_q_target.state_dict(),
        'worker_actor': agent.worker_actor.state_dict(),
        'worker_critic': agent.worker_critic.state_dict(),
        'worker_critic_target': agent.worker_critic_target.state_dict(),
        'reachability': agent.reachability.state_dict(),
        'total_steps': agent.total_steps,
        'total_episodes': agent.total_episodes,
        'success_threshold': agent.success_threshold,
    }, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--encoder', type=str, default='r3m', choices=['r3m', 'dinov2'])
    parser.add_argument('--n_landmarks', type=int, default=100)
    parser.add_argument('--subgoal_horizon', type=int, default=20)
    parser.add_argument('--log_dir', type=str, default='logs/')
    args = parser.parse_args()
    
    config = Config()
    config.training.seed = args.seed
    config.training.device = args.device
    config.training.total_timesteps = args.total_steps
    config.training.log_dir = args.log_dir
    config.encoder.name = args.encoder
    config.landmarks.n_landmarks = args.n_landmarks
    config.manager.subgoal_horizon = args.subgoal_horizon
    
    # Verify GPU
    if args.device == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available!"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    train(config)
