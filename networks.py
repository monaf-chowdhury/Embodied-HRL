"""Small neural-network helpers for the lean QC-FQL skill learner.

This module provides:
  * build_mlp           — the shared MLP backbone (LayerNorm + ReLU, optional dropout)
  * FlowActor           — a flow-matching BC policy (velocity field) plus a
                          one-step distilled actor, following Flow Q-Learning
                          (Park, Li, Levine; ICML 2025, arXiv:2502.02538).
  * TwinQ               — twin chunk critic Q(s, a_chunk)

The flow policy models the (multimodal) behavior action distribution; the
one-step actor is distilled from it and trained to maximize Q under a behavior
constraint, so policy improvement does not require within-state action
diversity in the data (which is exactly what broke the Gaussian+IQL setup).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(input_dim: int,
              hidden_dim: int,
              output_dim: int,
              n_layers: int = 3,
              use_layernorm: bool = True,
              dropout: float = 0.0) -> nn.Sequential:
    """Simple MLP used by actors and critics (LayerNorm + ReLU + optional dropout)."""
    assert n_layers >= 2, "MLP needs at least 2 linear layers"
    dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class FlowActor(nn.Module):
    """Flow-matching BC policy + one-step distilled actor (FQL).

    Two sub-networks, trained jointly by a single optimizer:
      velocity v_theta(s, x_t, t)  -> BC flow, learned by flow matching
      onestep  mu_omega(s, noise)  -> deployed one-step policy, distilled from
                                      the flow ODE and pushed to maximize Q

    `state_dim`  conditioning feature dim (z + proprio + task features)
    `action_dim` chunk action dim (env_action_dim * H)
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 flow_steps: int = 10,
                 use_layernorm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.action_dim = int(action_dim)
        self.flow_steps = int(flow_steps)
        # Velocity field: [state, noisy_action, time] -> velocity in action space.
        self.velocity = build_mlp(state_dim + action_dim + 1, hidden_dim, action_dim,
                                  n_layers, use_layernorm, dropout)
        # One-step actor: [state, noise] -> action chunk.
        self.onestep = build_mlp(state_dim + action_dim, hidden_dim, action_dim,
                                 n_layers, use_layernorm, dropout)

    def velocity_field(self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.velocity(torch.cat([state, x_t, t], dim=-1))

    def bc_flow_loss(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Conditional flow-matching loss with linear interpolation paths.

        x_0 ~ N(0, I),  x_1 = action,  x_t = (1-t) x_0 + t x_1,
        target velocity = x_1 - x_0,  loss = || v_theta(s, x_t, t) - (x_1 - x_0) ||^2.
        """
        x0 = torch.randn_like(action)
        t = torch.rand(action.shape[0], 1, device=action.device)
        x_t = (1.0 - t) * x0 + t * action
        target_v = action - x0
        pred_v = self.velocity_field(state, x_t, t)
        return ((pred_v - target_v) ** 2).mean()

    def flow_action(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Integrate dx/dt = v_theta(s, x, t) from noise (t=0) to action (t=1) by Euler."""
        x = noise
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full((x.shape[0], 1), i * dt, device=x.device)
            x = x + dt * self.velocity_field(state, x, t)
        return x

    def onestep_action(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.onestep(torch.cat([state, noise], dim=-1))


class TwinQ(nn.Module):
    """Twin chunk critic Q(s, a_chunk)."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, action_dim: int,
                 use_layernorm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.q1 = build_mlp(input_dim + action_dim, hidden_dim, 1, n_layers, use_layernorm, dropout)
        self.q2 = build_mlp(input_dim + action_dim, hidden_dim, 1, n_layers, use_layernorm, dropout)

    def forward(self, x: torch.Tensor, action: torch.Tensor):
        xa = torch.cat([x, action], dim=-1)
        return self.q1(xa), self.q2(xa)
