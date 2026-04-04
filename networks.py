"""
Neural network modules for all learnable components.

- Manager: DQN-style Q-network that scores landmark candidates
- Worker: SAC (actor-critic) goal-conditioned policy
- ReachabilityPredictor: binary classifier f(z_curr, z_sub) -> [0,1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple


# =============================================================================
# Shared building blocks
# =============================================================================

def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, 
              n_layers: int = 3, activation: str = "relu") -> nn.Sequential:
    """Build a simple MLP with LayerNorm."""
    act = nn.ReLU if activation == "relu" else nn.Tanh
    layers = []
    dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # No activation/norm on output layer
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


# =============================================================================
# Manager: Q-network over discrete landmark actions
# =============================================================================

class ManagerQNetwork(nn.Module):
    """
    Evaluates Q(z_current, z_goal, z_landmark) for each landmark candidate.
    
    Input: z_current (d,), z_goal (d,), z_landmark (d,)
    Output: scalar Q-value
    
    The manager calls this for each landmark and picks argmax (or epsilon-greedy).
    """
    
    def __init__(self, z_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        # Input: [z_current, z_goal, z_landmark] = 3 * z_dim
        self.net = build_mlp(3 * z_dim, hidden_dim, 1, n_layers)
    
    def forward(self, z_current: torch.Tensor, z_goal: torch.Tensor, 
                z_landmark: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_current: (B, z_dim)
            z_goal: (B, z_dim)  
            z_landmark: (B, z_dim)
        Returns:
            q_values: (B, 1)
        """
        x = torch.cat([z_current, z_goal, z_landmark], dim=-1)
        return self.net(x)
    
    def evaluate_all_landmarks(self, z_current: torch.Tensor, z_goal: torch.Tensor,
                                landmarks: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Q-values for all landmarks at once.
        
        Args:
            z_current: (B, z_dim)
            z_goal: (B, z_dim)
            landmarks: (N, z_dim) — all N landmark vectors
        Returns:
            q_values: (B, N)
        """
        B = z_current.shape[0]
        N = landmarks.shape[0]
        
        # Expand for broadcasting: (B, N, z_dim)
        z_curr_exp = z_current.unsqueeze(1).expand(B, N, -1)
        z_goal_exp = z_goal.unsqueeze(1).expand(B, N, -1)
        lm_exp = landmarks.unsqueeze(0).expand(B, N, -1)
        
        # Reshape to (B*N, 3*z_dim)
        x = torch.cat([z_curr_exp, z_goal_exp, lm_exp], dim=-1)
        x = x.reshape(B * N, -1)
        
        q_vals = self.net(x)  # (B*N, 1)
        return q_vals.reshape(B, N)


# =============================================================================
# Worker: Goal-conditioned SAC
# =============================================================================

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SACActorNetwork(nn.Module):
    """
    SAC actor: outputs a squashed Gaussian action given (z_current, z_subgoal).
    """
    
    def __init__(self, z_dim: int, action_dim: int, 
                 hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        input_dim = 2 * z_dim  # [z_current, z_subgoal]
        
        # Shared trunk
        trunk_layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1)
        for i in range(len(dims) - 1):
            trunk_layers.append(nn.Linear(dims[i], dims[i + 1]))
            trunk_layers.append(nn.LayerNorm(dims[i + 1]))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, z_current: torch.Tensor, z_subgoal: torch.Tensor):
        """
        Returns: (action, log_prob) with action squashed to [-1, 1]
        """
        x = torch.cat([z_current, z_subgoal], dim=-1)
        h = self.trunk(x)
        
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        
        # Sample from Gaussian
        dist = Normal(mean, std)
        x_t = dist.rsample()  # Reparameterization trick
        
        # Squash to [-1, 1] via tanh
        action = torch.tanh(x_t)
        
        # Log probability with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def get_action_deterministic(self, z_current: torch.Tensor, 
                                  z_subgoal: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean, squashed)."""
        x = torch.cat([z_current, z_subgoal], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)
        return torch.tanh(mean)


class SACCriticNetwork(nn.Module):
    """
    SAC twin Q-networks for the worker.
    Input: (z_current, z_subgoal, action) -> two Q-values
    """
    
    def __init__(self, z_dim: int, action_dim: int,
                 hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        input_dim = 2 * z_dim + action_dim
        
        self.q1 = build_mlp(input_dim, hidden_dim, 1, n_layers)
        self.q2 = build_mlp(input_dim, hidden_dim, 1, n_layers)
    
    def forward(self, z_current: torch.Tensor, z_subgoal: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_current, z_subgoal, action], dim=-1)
        return self.q1(x), self.q2(x)


# =============================================================================
# Reachability Predictor
# =============================================================================

class ReachabilityPredictor(nn.Module):
    """
    Predicts P(worker can reach z_subgoal from z_current within K steps).
    
    Trained as a binary classifier on rollout success/failure data.
    """
    
    def __init__(self, z_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        input_dim = 2 * z_dim  # [z_current, z_subgoal]
        self.net = build_mlp(input_dim, hidden_dim, 1, n_layers)
    
    def forward(self, z_current: torch.Tensor, z_subgoal: torch.Tensor) -> torch.Tensor:
        """Returns reachability probability in [0, 1]."""
        x = torch.cat([z_current, z_subgoal], dim=-1)
        return torch.sigmoid(self.net(x))
    
    def predict_numpy(self, z_current: np.ndarray, z_subgoal: np.ndarray,
                      device: str = "cuda") -> float:
        """Convenience: numpy in, float out."""
        z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(device)
        z_s = torch.from_numpy(z_subgoal).float().unsqueeze(0).to(device)
        with torch.no_grad():
            prob = self.forward(z_c, z_s)
        return prob.item()
