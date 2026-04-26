"""
networks.py — Neural architectures for SMGW.

Two models:

  * SemanticManager  : maps (z, proprio, task-state, completion-mask, task-embeds)
                       to a Q-value per task. At action time we mask the
                       logits of already-completed tasks so the manager is
                       STRUCTURALLY unable to re-pick a finished task.

  * GroundedWorker   : SAC actor/critic conditioned on the chosen task via
                       FiLM over a frozen text embedding + a compact task
                       target vector (padded goal slice from task_spec).
                       Optionally outputs an action chunk of length H_chunk.

Design choices worth flagging in review:

  - The worker's subgoal input is the BENCHMARK'S OWN GOAL (padded to
    max_goal_dim). There is NO latent subgoal anywhere in these networks.
  - The image latent z is still a context channel, because what's on screen
    (e.g., where the kettle currently sits) carries information that the
    indexed task-state slice alone may miss for non-selected tasks.
  - Action chunks use a single head that emits H_chunk * action_dim outputs,
    with per-step log_std. The chunk is executed open-loop by the env
    runner; the next policy query happens after the chunk ends.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional


# =============================================================================
# Small helpers
# =============================================================================

def build_mlp(input_dim: int, hidden_dim: int, output_dim: int,
              n_layers: int = 3, use_layernorm: bool = True) -> nn.Sequential:
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


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: given a conditioning vector c,
    produce (gamma, beta) and apply gamma * h + beta to features h.
    Used to inject task identity into worker features.
    """
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)
        # Initialise so FiLM starts close to identity
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, feats: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma(cond) * feats + self.beta(cond)


# =============================================================================
# Semantic Manager
# =============================================================================

class SemanticManager(nn.Module):
    """
    Q-function over the N_tasks discrete action space.

    Inputs (all per-batch):
      z              : (B, z_dim)          image latent (context)
      proprio        : (B, P)              full 59-d state (context)
      task_state     : (B, n_tasks * max_goal_dim)   flattened per-task slices
      completion     : (B, n_tasks)        binary completion mask
      task_embeds    : (n_tasks, d_text)   FROZEN text embeds (shared, not per-batch)

    Output:
      q_values       : (B, n_tasks)        Q-values per task; apply
                                           completion mask at ACTION time
                                           by setting q[:, k] = -inf where
                                           completion[:, k] == 1.

    Why we concatenate task_embeds into the manager's hidden representation
    rather than using them as FiLM conditioning: the manager's OUTPUT is
    already indexed per-task (one Q-value per task), so we just need the
    embeddings to differentiate tasks in the head. We tile and concat.
    """

    MASK_FILL = -1e9  # used to mask completed tasks

    def __init__(self,
                 z_dim: int,
                 proprio_dim: int,
                 n_tasks: int,
                 max_goal_dim: int,
                 text_embed_dim: int,
                 hidden_dim: int = 256,
                 n_layers: int = 3):
        super().__init__()
        self.n_tasks = n_tasks
        self.max_goal_dim = max_goal_dim

        task_state_dim = n_tasks * max_goal_dim

        # Torso: maps (z, proprio, task_state, completion) -> a shared hidden
        in_dim = z_dim + proprio_dim + task_state_dim + n_tasks
        self.torso = build_mlp(in_dim, hidden_dim, hidden_dim, n_layers)

        # Per-task head: concat(torso_out, text_embed_k) -> scalar Q
        self.head = build_mlp(hidden_dim + text_embed_dim, hidden_dim, 1,
                              n_layers=2)

    def forward(self,
                z: torch.Tensor,
                proprio: torch.Tensor,
                task_state: torch.Tensor,
                completion: torch.Tensor,
                task_embeds: torch.Tensor) -> torch.Tensor:
        """Return Q-values (B, n_tasks) UNMASKED (caller masks)."""
        B = z.shape[0]
        x = torch.cat([z, proprio, task_state, completion], dim=-1)
        h = self.torso(x)                                    # (B, H)

        # Broadcast torso output over tasks, concat per-task text embedding
        h_exp = h.unsqueeze(1).expand(B, self.n_tasks, h.shape[-1])
        t_exp = task_embeds.unsqueeze(0).expand(B, self.n_tasks,
                                                task_embeds.shape[-1])
        head_in = torch.cat([h_exp, t_exp], dim=-1)          # (B, K, H+d_text)
        head_in = head_in.reshape(B * self.n_tasks, -1)
        q = self.head(head_in).reshape(B, self.n_tasks)      # (B, K)
        return q

    def q_masked(self,
                 z: torch.Tensor,
                 proprio: torch.Tensor,
                 task_state: torch.Tensor,
                 completion: torch.Tensor,
                 task_embeds: torch.Tensor) -> torch.Tensor:
        """Q-values with completed tasks set to -inf for argmax."""
        q = self.forward(z, proprio, task_state, completion, task_embeds)
        q = q.masked_fill(completion > 0.5, self.MASK_FILL)
        return q


# =============================================================================
# Grounded Worker
# =============================================================================

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class WorkerTrunk(nn.Module):
    """
    Shared torso used by actor and critic. Produces a conditioning-modulated
    feature vector given:
      z           : (B, z_dim)         image latent
      proprio     : (B, P)             raw proprio (normalized outside)
      task_target : (B, max_goal_dim)  padded goal slice for chosen task
      task_cur    : (B, max_goal_dim)  padded current task-state slice
      task_mask   : (B, max_goal_dim)  1/0 mask over active goal dims
      task_embed  : (B, d_text)        frozen text embed for chosen task

    The FiLM layer injects task identity into the fused features so the
    same shared trunk can behave very differently per task.
    """
    def __init__(self,
                 z_dim: int,
                 proprio_dim: int,
                 max_goal_dim: int,
                 text_embed_dim: int,
                 hidden_dim: int,
                 n_layers: int):
        super().__init__()
        # Individual streams -> a compact feature each
        self.z_stream = build_mlp(z_dim, hidden_dim, hidden_dim, 2)
        self.prop_stream = build_mlp(proprio_dim, hidden_dim, hidden_dim, 2)
        # Task-grounding stream: current task-slice + goal + mask -> features
        # Feeding (cur, goal, cur-goal) so the net sees the DELTA directly.
        self.task_stream = build_mlp(3 * max_goal_dim + max_goal_dim,
                                     hidden_dim, hidden_dim, 2)

        fused_dim = 3 * hidden_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.film = FiLMLayer(text_embed_dim, hidden_dim)

        # Post-FiLM head
        self.post = build_mlp(hidden_dim, hidden_dim, hidden_dim,
                              max(n_layers - 1, 2))

    def forward(self,
                z: torch.Tensor,
                proprio: torch.Tensor,
                task_target: torch.Tensor,
                task_cur: torch.Tensor,
                task_mask: torch.Tensor,
                task_embed: torch.Tensor) -> torch.Tensor:
        delta = (task_target - task_cur) * task_mask
        task_in = torch.cat([task_cur, task_target, delta, task_mask], dim=-1)

        fz = self.z_stream(z)
        fp = self.prop_stream(proprio)
        ft = self.task_stream(task_in)

        h = self.fuse(torch.cat([fz, fp, ft], dim=-1))
        h = self.film(h, task_embed)
        h = F.relu(h)
        return self.post(h)


class GroundedWorkerActor(nn.Module):
    """
    SAC actor with optional action-chunk output.

    If action_chunk_len == 1 : outputs a single-step squashed-Gaussian policy
                               over action_dim dims.
    If action_chunk_len  > 1 : outputs H_chunk step actions, each squashed-
                               Gaussian, stacked as (B, H_chunk, action_dim).
                               log_prob is summed across chunk steps AND
                               across action dims.
    """
    def __init__(self,
                 z_dim: int,
                 proprio_dim: int,
                 action_dim: int,
                 max_goal_dim: int,
                 text_embed_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 action_chunk_len: int = 1):
        super().__init__()
        assert action_chunk_len >= 1
        self.action_dim = action_dim
        self.H = action_chunk_len
        self.out_dim = action_dim * action_chunk_len

        self.trunk = WorkerTrunk(z_dim, proprio_dim, max_goal_dim,
                                 text_embed_dim, hidden_dim, n_layers)
        self.mean_head = nn.Linear(hidden_dim, self.out_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.out_dim)

    def _dist(self, z, proprio, task_target, task_cur, task_mask, task_embed):
        h = self.trunk(z, proprio, task_target, task_cur, task_mask, task_embed)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def forward(self, z, proprio, task_target, task_cur, task_mask, task_embed
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (action, log_prob).
          action   : (B, out_dim)  in [-1, 1]
          log_prob : (B, 1)        summed over all action dims (and chunk steps)
        """
        dist = self._dist(z, proprio, task_target, task_cur, task_mask, task_embed)
        x = dist.rsample()
        action = torch.tanh(x)
        # Tanh-squashed log-prob correction
        logp = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        logp = logp.sum(-1, keepdim=True)
        return action, logp

    def get_action_deterministic(self, z, proprio, task_target, task_cur,
                                 task_mask, task_embed) -> torch.Tensor:
        h = self.trunk(z, proprio, task_target, task_cur, task_mask, task_embed)
        return torch.tanh(self.mean_head(h))

    def log_prob_of(self,
                    z: torch.Tensor,
                    proprio: torch.Tensor,
                    task_target: torch.Tensor,
                    task_cur: torch.Tensor,
                    task_mask: torch.Tensor,
                    task_embed: torch.Tensor,
                    action_flat: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of a given action under the current policy.

        action_flat : (B, out_dim) — tanh-squashed, in [-1, 1].
        Returns     : (B,) log-probability summed over all action/chunk dims.

        Used by IQL advantage-weighted BC:
            loss = -mean( exp(beta * A(s,a)) * log_prob_of(s, a_demo) )
        """
        dist = self._dist(z, proprio, task_target, task_cur, task_mask, task_embed)
        a = action_flat.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        x = torch.atanh(a)                                  # invert tanh squashing
        logp = dist.log_prob(x) - torch.log(1.0 - a.pow(2) + 1e-6)
        return logp.sum(-1)                                  # (B,)

    def action_to_chunk(self, action: torch.Tensor) -> torch.Tensor:
        """
        Reshape a flat action output (B, H*A) into (B, H, A). When H==1 this
        is just (B, 1, A).
        """
        B = action.shape[0]
        return action.reshape(B, self.H, self.action_dim)


class GroundedWorkerCritic(nn.Module):
    """
    Twin-Q critic. For chunked workers the critic Q(s, a_chunk) treats the
    flat action vector (B, H*A) as the action, so the TD target is an
    n-step return sum over the chunk.
    """
    def __init__(self,
                 z_dim: int,
                 proprio_dim: int,
                 action_dim: int,
                 max_goal_dim: int,
                 text_embed_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 action_chunk_len: int = 1):
        super().__init__()
        self.H = action_chunk_len
        self.action_dim = action_dim
        flat_action_dim = action_dim * action_chunk_len

        self.trunk_1 = WorkerTrunk(z_dim, proprio_dim, max_goal_dim,
                                   text_embed_dim, hidden_dim, n_layers)
        self.trunk_2 = WorkerTrunk(z_dim, proprio_dim, max_goal_dim,
                                   text_embed_dim, hidden_dim, n_layers)
        self.q1 = build_mlp(hidden_dim + flat_action_dim, hidden_dim, 1,
                            n_layers=3)
        self.q2 = build_mlp(hidden_dim + flat_action_dim, hidden_dim, 1,
                            n_layers=3)

    def forward(self, z, proprio, task_target, task_cur, task_mask, task_embed,
                action) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.trunk_1(z, proprio, task_target, task_cur, task_mask, task_embed)
        h2 = self.trunk_2(z, proprio, task_target, task_cur, task_mask, task_embed)
        q1 = self.q1(torch.cat([h1, action], dim=-1))
        q2 = self.q2(torch.cat([h2, action], dim=-1))
        return q1, q2


# =============================================================================
# Grounded Worker Value — V(s, task) for IQL offline pretraining
# =============================================================================

class GroundedWorkerValue(nn.Module):
    """
    Twin state-value function V(s, task) used during IQL offline pretraining.
    Identical inputs to the actor/critic trunk but NO action input.

    Twin architecture (V1, V2) mirrors the twin-Q critic for stability:
    the IQL V-update uses min(Q1, Q2) as target and the Q-update uses
    min(V1_target, V2_target) for bootstrapping.
    """
    def __init__(self,
                 z_dim: int,
                 proprio_dim: int,
                 max_goal_dim: int,
                 text_embed_dim: int,
                 hidden_dim: int = 256,
                 n_layers: int = 3):
        super().__init__()
        self.trunk_1 = WorkerTrunk(z_dim, proprio_dim, max_goal_dim,
                                   text_embed_dim, hidden_dim, n_layers)
        self.trunk_2 = WorkerTrunk(z_dim, proprio_dim, max_goal_dim,
                                   text_embed_dim, hidden_dim, n_layers)
        self.v1 = nn.Linear(hidden_dim, 1)
        self.v2 = nn.Linear(hidden_dim, 1)

    def forward(self,
                z: torch.Tensor,
                proprio: torch.Tensor,
                task_target: torch.Tensor,
                task_cur: torch.Tensor,
                task_mask: torch.Tensor,
                task_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (v1, v2), each shape (B,)."""
        h1 = self.trunk_1(z, proprio, task_target, task_cur, task_mask, task_embed)
        h2 = self.trunk_2(z, proprio, task_target, task_cur, task_mask, task_embed)
        return self.v1(h1).squeeze(-1), self.v2(h2).squeeze(-1)
