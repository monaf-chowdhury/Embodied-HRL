"""
Neural network modules.

Changes from previous version:
  - All z_dim inputs are now 2048 (raw R3M). Networks compress internally.
  - ManagerQNetwork: input = 2*2048 = 4096. First layer compresses to hidden_dim.
  - SACActorNetwork / SACCriticNetwork: image streams compress 2*2048 → 256,
    proprio stream compresses 59 → 64. Merge dim = 320.
  - hidden_dim defaults raised to 512 throughout.
  - ReachabilityPredictor kept but disabled via config.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple


def build_mlp(input_dim, hidden_dim, output_dim, n_layers=3):
    layers = []
    dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# =============================================================================
# Manager Q-Network
# Input: [z_current (2048), z_landmark (2048)] = 4096
# =============================================================================

class ManagerQNetwork(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 512, n_layers: int = 3):
        super().__init__()
        # First compress the large input before the MLP
        self.input_compress = nn.Sequential(
            nn.Linear(2 * z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.net = build_mlp(hidden_dim, hidden_dim, 1, max(n_layers - 1, 2))

    def forward(self, z_current: torch.Tensor, z_landmark: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_current, z_landmark], dim=-1)
        return self.net(self.input_compress(x))

    def evaluate_all_landmarks(
        self,
        z_current: torch.Tensor,   # (B, z_dim)
        landmarks: torch.Tensor,   # (N, z_dim)
    ) -> torch.Tensor:             # (B, N)
        B, N = z_current.shape[0], landmarks.shape[0]
        z_exp  = z_current.unsqueeze(1).expand(B, N, -1)
        lm_exp = landmarks.unsqueeze(0).expand(B, N, -1)
        x = torch.cat([z_exp, lm_exp], dim=-1).reshape(B * N, -1)
        compressed = self.input_compress(x)
        return self.net(compressed).reshape(B, N)


# =============================================================================
# Worker: Dual-stream SAC
# Stream 1: [z_current (2048), z_subgoal (2048)] → compressed 256
# Stream 2: normalised proprio (59) → compressed 64
# Merge: 320 → action
# =============================================================================

LOG_STD_MIN = -20
LOG_STD_MAX = 2

_IMG_STREAM_OUT  = 256
_PROP_STREAM_OUT = 64
_MERGE_DIM       = _IMG_STREAM_OUT + _PROP_STREAM_OUT  # 320


def _img_stream(z_dim: int, hidden_dim: int) -> nn.Sequential:
    """Compress [z_current, z_subgoal] = 4096 → _IMG_STREAM_OUT."""
    return nn.Sequential(
        nn.Linear(2 * z_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, _IMG_STREAM_OUT),
        nn.LayerNorm(_IMG_STREAM_OUT),
        nn.ReLU(),
    )


def _prop_stream(proprio_dim: int, hidden_dim: int) -> nn.Sequential:
    mid = max(hidden_dim // 4, 64)
    return nn.Sequential(
        nn.Linear(proprio_dim, mid),
        nn.LayerNorm(mid),
        nn.ReLU(),
        nn.Linear(mid, _PROP_STREAM_OUT),
        nn.LayerNorm(_PROP_STREAM_OUT),
        nn.ReLU(),
    )


class SACActorNetwork(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_dim=512, n_layers=3, proprio_dim=59):
        super().__init__()
        self.img_stream  = _img_stream(z_dim, hidden_dim)
        self.prop_stream = _prop_stream(proprio_dim, hidden_dim)

        # Trunk after merge: _MERGE_DIM → hidden_dim → ... → hidden_dim
        trunk = [nn.Linear(_MERGE_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(max(n_layers - 3, 0)):
            trunk += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*trunk)

        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _encode(self, z_c, z_s, p):
        img  = self.img_stream(torch.cat([z_c, z_s], dim=-1))
        prop = self.prop_stream(p)
        return self.trunk(torch.cat([img, prop], dim=-1))

    def forward(self, z_c, z_s, p):
        h = self._encode(z_c, z_s, p)
        mean    = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        dist    = Normal(mean, log_std.exp())
        x_t     = dist.rsample()
        action  = torch.tanh(x_t)
        lp      = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, lp.sum(-1, keepdim=True)

    def get_action_deterministic(self, z_c, z_s, p):
        return torch.tanh(self.mean_head(self._encode(z_c, z_s, p)))


class SACCriticNetwork(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_dim=512, n_layers=3, proprio_dim=59):
        super().__init__()
        self.img1  = _img_stream(z_dim, hidden_dim)
        self.prop1 = _prop_stream(proprio_dim, hidden_dim)
        self.q1    = build_mlp(_MERGE_DIM + action_dim, hidden_dim, 1, n_layers)

        self.img2  = _img_stream(z_dim, hidden_dim)
        self.prop2 = _prop_stream(proprio_dim, hidden_dim)
        self.q2    = build_mlp(_MERGE_DIM + action_dim, hidden_dim, 1, n_layers)

    def forward(self, z_c, z_s, p, action):
        img_cat = torch.cat([z_c, z_s], dim=-1)
        f1 = torch.cat([self.img1(img_cat), self.prop1(p), action], dim=-1)
        f2 = torch.cat([self.img2(img_cat), self.prop2(p), action], dim=-1)
        return self.q1(f1), self.q2(f2)


# =============================================================================
# Reachability Predictor (kept but disabled via config)
# =============================================================================

class ReachabilityPredictor(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 512, n_layers: int = 3):
        super().__init__()
        self.input_compress = nn.Sequential(
            nn.Linear(2 * z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.net = build_mlp(hidden_dim, hidden_dim, 1, max(n_layers - 1, 2))

    def forward(self, z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_c, z_s], dim=-1)
        return torch.sigmoid(self.net(self.input_compress(x)))
