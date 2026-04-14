import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 3) -> nn.Sequential:
    dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


class ManagerQNetwork(nn.Module):
    def __init__(self, z_dim: int, n_tasks: int, hidden_dim: int = 512, n_layers: int = 3):
        super().__init__()
        self.n_tasks = n_tasks
        self.net = build_mlp(2 * z_dim + n_tasks, hidden_dim, 1, n_layers)

    def forward(self, z_current: torch.Tensor, z_landmark: torch.Tensor, task_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_current, z_landmark, task_onehot], dim=-1)
        return self.net(x)

    def evaluate_all_landmarks(self, z_current: torch.Tensor, landmarks: torch.Tensor, task_onehots: torch.Tensor) -> torch.Tensor:
        B, N = z_current.shape[0], landmarks.shape[0]
        zc = z_current.unsqueeze(1).expand(B, N, -1)
        lm = landmarks.unsqueeze(0).expand(B, N, -1)
        to = task_onehots.unsqueeze(0).expand(B, N, -1)
        x = torch.cat([zc, lm, to], dim=-1).reshape(B * N, -1)
        return self.net(x).reshape(B, N)


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SACActorNetwork(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, n_tasks: int, hidden_dim: int = 384, n_layers: int = 3, proprio_dim: int = 59):
        super().__init__()
        self.img_stream = build_mlp(2 * z_dim, hidden_dim, hidden_dim, 2)
        self.proprio_stream = build_mlp(proprio_dim, hidden_dim // 2, hidden_dim // 2, 2)
        self.task_stream = build_mlp(n_tasks, 32, 32, 2)
        trunk_in = hidden_dim + hidden_dim // 2 + 32
        self.trunk = build_mlp(trunk_in, hidden_dim, hidden_dim, max(n_layers - 1, 2))
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _encode(self, z_current, z_subgoal, proprio, task_onehot):
        img = self.img_stream(torch.cat([z_current, z_subgoal], dim=-1))
        prop = self.proprio_stream(proprio)
        task = self.task_stream(task_onehot)
        return self.trunk(torch.cat([img, prop, task], dim=-1))

    def forward(self, z_current, z_subgoal, proprio, task_onehot) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(z_current, z_subgoal, proprio, task_onehot)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    def get_action_deterministic(self, z_current, z_subgoal, proprio, task_onehot):
        h = self._encode(z_current, z_subgoal, proprio, task_onehot)
        return torch.tanh(self.mean_head(h))


class SACCriticNetwork(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, n_tasks: int, hidden_dim: int = 384, n_layers: int = 3, proprio_dim: int = 59):
        super().__init__()
        self.img_stream_1 = build_mlp(2 * z_dim, hidden_dim, hidden_dim, 2)
        self.prop_stream_1 = build_mlp(proprio_dim, hidden_dim // 2, hidden_dim // 2, 2)
        self.task_stream_1 = build_mlp(n_tasks, 32, 32, 2)
        self.q1 = build_mlp(hidden_dim + hidden_dim // 2 + 32 + action_dim, hidden_dim, 1, n_layers)

        self.img_stream_2 = build_mlp(2 * z_dim, hidden_dim, hidden_dim, 2)
        self.prop_stream_2 = build_mlp(proprio_dim, hidden_dim // 2, hidden_dim // 2, 2)
        self.task_stream_2 = build_mlp(n_tasks, 32, 32, 2)
        self.q2 = build_mlp(hidden_dim + hidden_dim // 2 + 32 + action_dim, hidden_dim, 1, n_layers)

    def forward(self, z_current, z_subgoal, proprio, task_onehot, action):
        img = torch.cat([z_current, z_subgoal], dim=-1)
        f1 = torch.cat([self.img_stream_1(img), self.prop_stream_1(proprio), self.task_stream_1(task_onehot), action], dim=-1)
        f2 = torch.cat([self.img_stream_2(img), self.prop_stream_2(proprio), self.task_stream_2(task_onehot), action], dim=-1)
        return self.q1(f1), self.q2(f2)
