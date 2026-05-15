"""Small neural-network helpers for the lean skill learner."""
from __future__ import annotations

import torch.nn as nn


def build_mlp(input_dim: int,
              hidden_dim: int,
              output_dim: int,
              n_layers: int = 3,
              use_layernorm: bool = True) -> nn.Sequential:
    """Simple MLP used by per-skill actors, critics, and value functions."""
    assert n_layers >= 2, "MLP needs at least 2 linear layers"
    dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
