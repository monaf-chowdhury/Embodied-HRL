"""
Neural network modules for all learnable components.

Changes vs original:
  - Worker (SAC) now uses a DUAL-STREAM architecture:
      stream_1: image latent (64d) + subgoal latent (64d) = 128d
      stream_2: normalised proprio (59d)
      merged  : concat(stream_1_features, stream_2_features) -> action/Q-value
    This keeps image-space reasoning and motor control signals separate
    until the final merge, preventing proprio scale from dominating.

  - Manager is UNCHANGED: it only receives image latents (64d + 64d = 128d).
    The manager's job is geometric reasoning in image-latent space.

  - ReachabilityPredictor is UNCHANGED.
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

def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int = 3,
    activation: str = "relu",
) -> nn.Sequential:
    act = nn.ReLU if activation == "relu" else nn.Tanh
    layers = []
    dims   = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


# =============================================================================
# Manager: Q-network over discrete landmark actions (image latent only)
# =============================================================================

class ManagerQNetwork(nn.Module):
    """
    Q(z_current, z_landmark) — image latents only, no z_goal, no proprio.
    input_multiplier=2 means input is [z_current, z_landmark] = 2 * z_dim.

    The manager no longer uses a global z_goal. Its job is to pick which
    landmark to visit next from the current state — that is purely a
    geometric problem in image-latent space.
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        input_multiplier: int = 2,   # 2 = [z_curr, z_lm]; 3 = legacy [z_curr, z_goal, z_lm]
    ):
        super().__init__()
        self.input_multiplier = input_multiplier
        self.net = build_mlp(input_multiplier * z_dim, hidden_dim, 1, n_layers)

    def forward(
        self,
        z_current: torch.Tensor,
        z_landmark: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_current  : (B, z_dim)
            z_landmark : (B, z_dim)
        Returns:
            q_values   : (B, 1)
        """
        x = torch.cat([z_current, z_landmark], dim=-1)
        return self.net(x)

    def evaluate_all_landmarks(
        self,
        z_current: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate Q-values for all landmarks efficiently.

        Args:
            z_current : (B, z_dim)
            landmarks : (N, z_dim)
        Returns:
            q_values  : (B, N)
        """
        B = z_current.shape[0]
        N = landmarks.shape[0]

        z_curr_exp = z_current.unsqueeze(1).expand(B, N, -1)
        lm_exp     = landmarks.unsqueeze(0).expand(B, N, -1)

        x = torch.cat([z_curr_exp, lm_exp], dim=-1).reshape(B * N, -1)
        return self.net(x).reshape(B, N)


# =============================================================================
# Worker: Dual-stream goal-conditioned SAC
# =============================================================================

LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Stream sizes
_IMG_STREAM_DIM   = 128   # processes [z_current, z_subgoal] (each 64d)
_PROPRIO_STREAM_DIM = 64  # processes normalised proprio (59d)
_MERGE_DIM        = _IMG_STREAM_DIM + _PROPRIO_STREAM_DIM


def _build_stream(input_dim: int, out_dim: int, hidden_dim: int) -> nn.Sequential:
    """Two-layer stream encoder with LayerNorm."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
    )


class SACActorNetwork(nn.Module):
    """
    Dual-stream SAC actor.

    Stream 1 (image): [z_current (64d), z_subgoal (64d)] → 128d features
    Stream 2 (proprio): normalised proprio (59d) → 64d features
    Merged: 192d → squashed Gaussian action
    """

    def __init__(
        self,
        z_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        proprio_dim: int = 59,
    ):
        super().__init__()
        self.z_dim       = z_dim
        self.proprio_dim = proprio_dim

        # Stream 1: image latents
        self.img_stream = _build_stream(2 * z_dim, _IMG_STREAM_DIM, hidden_dim)

        # Stream 2: proprio
        self.proprio_stream = _build_stream(proprio_dim, _PROPRIO_STREAM_DIM, hidden_dim)

        # Merged trunk (remaining n_layers - 2 layers after the two stream layers)
        merge_hidden = hidden_dim
        trunk_layers = []
        for _ in range(max(n_layers - 2, 1)):
            trunk_layers += [
                nn.Linear(merge_hidden if trunk_layers else _MERGE_DIM, merge_hidden),
                nn.LayerNorm(merge_hidden),
                nn.ReLU(),
            ]
        # Fix: always start from _MERGE_DIM
        trunk_layers = [
            nn.Linear(_MERGE_DIM, merge_hidden),
            nn.LayerNorm(merge_hidden),
            nn.ReLU(),
        ]
        for _ in range(max(n_layers - 3, 0)):
            trunk_layers += [
                nn.Linear(merge_hidden, merge_hidden),
                nn.LayerNorm(merge_hidden),
                nn.ReLU(),
            ]
        self.trunk = nn.Sequential(*trunk_layers)

        self.mean_head    = nn.Linear(merge_hidden, action_dim)
        self.log_std_head = nn.Linear(merge_hidden, action_dim)

    def _encode(
        self,
        z_current: torch.Tensor,
        z_subgoal: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        img_feat    = self.img_stream(torch.cat([z_current, z_subgoal], dim=-1))
        proprio_feat = self.proprio_stream(proprio)
        merged      = torch.cat([img_feat, proprio_feat], dim=-1)
        return self.trunk(merged)

    def forward(
        self,
        z_current: torch.Tensor,
        z_subgoal: torch.Tensor,
        proprio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(z_current, z_subgoal, proprio)
        mean    = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        dist  = Normal(mean, std)
        x_t   = dist.rsample()
        action = torch.tanh(x_t)

        log_prob  = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob  = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def get_action_deterministic(
        self,
        z_current: torch.Tensor,
        z_subgoal: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        h = self._encode(z_current, z_subgoal, proprio)
        return torch.tanh(self.mean_head(h))


class SACCriticNetwork(nn.Module):
    """
    Dual-stream twin Q-networks for the worker.
    Input: z_current, z_subgoal, proprio, action
    """

    def __init__(
        self,
        z_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        proprio_dim: int = 59,
    ):
        super().__init__()

        # Q1
        self.img_stream_1    = _build_stream(2 * z_dim, _IMG_STREAM_DIM, hidden_dim)
        self.proprio_stream_1 = _build_stream(proprio_dim, _PROPRIO_STREAM_DIM, hidden_dim)
        self.q1_trunk = build_mlp(
            _MERGE_DIM + action_dim, hidden_dim, 1, n_layers)

        # Q2
        self.img_stream_2    = _build_stream(2 * z_dim, _IMG_STREAM_DIM, hidden_dim)
        self.proprio_stream_2 = _build_stream(proprio_dim, _PROPRIO_STREAM_DIM, hidden_dim)
        self.q2_trunk = build_mlp(
            _MERGE_DIM + action_dim, hidden_dim, 1, n_layers)

    def forward(
        self,
        z_current: torch.Tensor,
        z_subgoal: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_cat = torch.cat([z_current, z_subgoal], dim=-1)

        f1 = torch.cat([
            self.img_stream_1(img_cat),
            self.proprio_stream_1(proprio),
            action,
        ], dim=-1)
        f2 = torch.cat([
            self.img_stream_2(img_cat),
            self.proprio_stream_2(proprio),
            action,
        ], dim=-1)

        return self.q1_trunk(f1), self.q2_trunk(f2)


# =============================================================================
# Reachability Predictor (unchanged — operates in image latent space only)
# =============================================================================

class ReachabilityPredictor(nn.Module):
    """P(worker can reach z_subgoal from z_current within K steps)."""

    def __init__(self, z_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.net = build_mlp(2 * z_dim, hidden_dim, 1, n_layers)

    def forward(
        self,
        z_current: torch.Tensor,
        z_subgoal: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([z_current, z_subgoal], dim=-1)))

    def predict_numpy(
        self,
        z_current: np.ndarray,
        z_subgoal: np.ndarray,
        device: str = "cuda",
    ) -> float:
        z_c = torch.from_numpy(z_current).float().unsqueeze(0).to(device)
        z_s = torch.from_numpy(z_subgoal).float().unsqueeze(0).to(device)
        with torch.no_grad():
            prob = self.forward(z_c, z_s)
        return prob.item()