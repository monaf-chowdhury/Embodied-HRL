from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 3) -> nn.Sequential:
    dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


class ManagerQNetwork(nn.Module):
    def __init__(self, z_dim: int, proprio_dim: int, n_tasks: int, task_lang_dim: int, task_goal_dim: int, hidden_dim: int = 384, n_layers: int = 3):
        super().__init__()
        self.n_tasks = n_tasks
        self.state_stream = build_mlp(
            z_dim + proprio_dim + 5 * n_tasks + (n_tasks + 1),
            hidden_dim,
            hidden_dim,
            3,
        )
        token_dim = task_lang_dim + task_goal_dim + 6
        self.score_net = build_mlp(hidden_dim + token_dim, hidden_dim, 1, n_layers)

    def _prev_task_onehot(self, prev_task: torch.Tensor) -> torch.Tensor:
        prev = torch.full((prev_task.shape[0], self.n_tasks + 1), 0.0, device=prev_task.device)
        clamped = prev_task.long().clamp(min=-1, max=self.n_tasks - 1)
        none_mask = clamped < 0
        prev[none_mask, self.n_tasks] = 1.0
        idx = (~none_mask).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            prev[idx, clamped[idx]] = 1.0
        return prev

    def encode_state(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        progress: torch.Tensor,
        errors: torch.Tensor,
        completion: torch.Tensor,
        remaining: torch.Tensor,
        prototype_sims: torch.Tensor,
        prev_task: torch.Tensor,
    ) -> torch.Tensor:
        prev_oh = self._prev_task_onehot(prev_task)
        x = torch.cat([z, proprio, progress, errors, completion, remaining, prototype_sims, prev_oh], dim=-1)
        return self.state_stream(x)

    def evaluate_all_tasks(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        progress: torch.Tensor,
        errors: torch.Tensor,
        completion: torch.Tensor,
        remaining: torch.Tensor,
        prototype_sims: torch.Tensor,
        prev_task: torch.Tensor,
        task_lang: torch.Tensor,
        task_goals: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = z.shape[0]
        state_h = self.encode_state(z, proprio, progress, errors, completion, remaining, prototype_sims, prev_task)
        per_task = torch.stack([
            progress,
            errors,
            completion,
            remaining,
            prototype_sims,
            (prev_task.unsqueeze(1) == torch.arange(self.n_tasks, device=z.device).view(1, -1)).float(),
        ], dim=-1)
        lang = task_lang.unsqueeze(0).expand(B, -1, -1)
        goals = task_goals.unsqueeze(0).expand(B, -1, -1)
        state_expand = state_h.unsqueeze(1).expand(B, self.n_tasks, -1)
        token = torch.cat([lang, goals, per_task], dim=-1)
        q = self.score_net(torch.cat([state_expand, token], dim=-1)).squeeze(-1)
        if valid_mask is not None:
            q = q.masked_fill(valid_mask <= 0.5, -1e9)
        return q

    def forward(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        progress: torch.Tensor,
        errors: torch.Tensor,
        completion: torch.Tensor,
        remaining: torch.Tensor,
        prototype_sims: torch.Tensor,
        prev_task: torch.Tensor,
        task_lang: torch.Tensor,
        task_goals: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        q_all = self.evaluate_all_tasks(z, proprio, progress, errors, completion, remaining, prototype_sims, prev_task, task_lang, task_goals)
        return q_all.gather(1, task_id.long().unsqueeze(1))


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class WorkerActorNetwork(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, n_tasks: int, task_lang_dim: int, task_goal_dim: int, proprio_dim: int = 59, hidden_dim: int = 384, n_layers: int = 3):
        super().__init__()
        self.visual = build_mlp(z_dim, hidden_dim, hidden_dim // 2, 2)
        self.proprio = build_mlp(proprio_dim, hidden_dim // 2, hidden_dim // 2, 2)
        cond_dim = task_lang_dim + 3 * task_goal_dim + 1 + n_tasks
        self.cond = build_mlp(cond_dim, hidden_dim // 2, hidden_dim // 2, 2)
        self.trunk = build_mlp(hidden_dim + hidden_dim // 2, hidden_dim, hidden_dim, max(n_layers - 1, 2))
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _encode(self, z, proprio, task_lang, target, value, error_vec, progress, completion):
        zv = self.visual(z)
        pv = self.proprio(proprio)
        cond = self.cond(torch.cat([task_lang, target, value, error_vec, progress.unsqueeze(-1), completion], dim=-1))
        return self.trunk(torch.cat([zv, pv, cond], dim=-1))

    def forward(self, z, proprio, task_lang, target, value, error_vec, progress, completion) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(z, proprio, task_lang, target, value, error_vec, progress, completion)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    def get_action_deterministic(self, z, proprio, task_lang, target, value, error_vec, progress, completion):
        h = self._encode(z, proprio, task_lang, target, value, error_vec, progress, completion)
        return torch.tanh(self.mean_head(h))


class WorkerCriticNetwork(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, n_tasks: int, task_lang_dim: int, task_goal_dim: int, proprio_dim: int = 59, hidden_dim: int = 384, n_layers: int = 3):
        super().__init__()
        cond_dim = task_lang_dim + 3 * task_goal_dim + 1 + n_tasks

        self.z1 = build_mlp(z_dim, hidden_dim, hidden_dim // 2, 2)
        self.p1 = build_mlp(proprio_dim, hidden_dim // 2, hidden_dim // 2, 2)
        self.c1 = build_mlp(cond_dim, hidden_dim // 2, hidden_dim // 2, 2)
        self.q1 = build_mlp(hidden_dim + hidden_dim // 2 + action_dim, hidden_dim, 1, n_layers)

        self.z2 = build_mlp(z_dim, hidden_dim, hidden_dim // 2, 2)
        self.p2 = build_mlp(proprio_dim, hidden_dim // 2, hidden_dim // 2, 2)
        self.c2 = build_mlp(cond_dim, hidden_dim // 2, hidden_dim // 2, 2)
        self.q2 = build_mlp(hidden_dim + hidden_dim // 2 + action_dim, hidden_dim, 1, n_layers)

    def forward(self, z, proprio, task_lang, target, value, error_vec, progress, completion, action):
        cond = torch.cat([task_lang, target, value, error_vec, progress.unsqueeze(-1), completion], dim=-1)
        f1 = torch.cat([self.z1(z), self.p1(proprio), self.c1(cond), action], dim=-1)
        f2 = torch.cat([self.z2(z), self.p2(proprio), self.c2(cond), action], dim=-1)
        return self.q1(f1), self.q2(f2)
