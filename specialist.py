"""
specialist.py -- Lean per-skill BC/IQL learner.

Current research path:
  1. Load replay-labelled FrankaKitchen demonstrations.
  2. Train one visual low-level policy per task with the same protocol.
  3. Evaluate skills alone and under a scripted task order.

No hierarchy, no teacher/student, no residual stack. Keep this file boring.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from config import Config
from demo_dataset import build_or_load_demo_dataset
from encoder import VisualEncoder
from env_wrapper import FrankaKitchenImageWrapper
from networks import build_mlp
from offline_algorithms import make_offline_algorithm
from utils import TaskSpec, build_frozen_text_embeddings, build_task_state_flat


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def lql_lower_bound_penalty(q_chain: torch.Tensor,
                            v_next_chain: torch.Tensor,
                            rewards: torch.Tensor,
                            dones: torch.Tensor,
                            valid: torch.Tensor,
                            gamma: float,
                            min_gap: int) -> Dict[str, torch.Tensor]:
    """LQL lower-bound hinge penalty over chunk-aligned demo chains.

    For a chain of N valid transitions (states s_0..s_N), every pair (k, l)
    with l - k >= min_gap enforces the optimality lower bound

        Q(s_k, a_k) >= G_{k:l} + gamma^(l-k) * (1 - done_{l-1}) * V_target(s_l)

    where G_{k:l} = sum_{j=k}^{l-1} gamma^(j-k) r_j is the realized discounted
    partial return. Violations are penalised with hinge^2 (paper: arXiv
    2605.05812; continuation via V_target instead of Q(s, pi(s)) to stay
    in-distribution for IQL).

    Args (all shaped (B, L); only q_chain may carry gradient):
      q_chain      Q(s_k, a_k) for each chain row.
      v_next_chain V_target at each row's NEXT state, i.e. chain state s_{k+1}.
      rewards      per-chunk rewards (already zeroed where invalid).
      dones        per-row env-termination flags.
      valid        bool prefix mask of real rows.

    Returns dict: "penalty" (scalar, mean hinge^2 over valid pairs) plus
    detached diagnostics ("active_frac", "hinge_mean", "n_pairs").
    """
    B, L = rewards.shape
    device = rewards.device
    min_gap = max(1, int(min_gap))
    zero = torch.zeros((), device=device)
    if L < min_gap:
        return {"penalty": zero, "active_frac": zero,
                "hinge_mean": zero, "n_pairs": zero}

    valid_f = valid.float()
    r = rewards * valid_f
    disc = torch.pow(torch.full((L,), float(gamma), device=device),
                     torch.arange(L, device=device, dtype=torch.float32))
    # c[:, m] = sum_{j < m} gamma^j r_j  -> G_{k:l} = (c[:, l] - c[:, k]) / gamma^k
    c = torch.cat([torch.zeros(B, 1, device=device),
                   torch.cumsum(r * disc.unsqueeze(0), dim=1)], dim=1)

    k_idx = torch.arange(L, device=device)            # k = 0..L-1
    l_idx = torch.arange(1, L + 1, device=device)     # l = 1..L (state s_l)
    G = (c[:, l_idx].unsqueeze(1) - c[:, k_idx].unsqueeze(2)) / disc[k_idx].view(1, L, 1)

    span = l_idx.view(1, 1, L) - k_idx.view(1, L, 1)  # l - k
    gamma_span = torch.pow(torch.full_like(G, float(gamma)), span.float())
    # Continuation at s_l == next state of row l-1; drop it past termination.
    boot = gamma_span * (1.0 - dones[:, l_idx - 1].unsqueeze(1)) * v_next_chain[:, l_idx - 1].unsqueeze(1)

    hinge = F.relu(G + boot - q_chain.unsqueeze(2))
    # Prefix-contiguous validity: rows k..l-1 valid <=> row l-1 valid (k < l).
    pair_valid = (span >= min_gap) & valid[:, l_idx - 1].unsqueeze(1)
    pair_valid_f = pair_valid.float()
    n_pairs = pair_valid_f.sum().clamp(min=1.0)

    penalty = (hinge.pow(2) * pair_valid_f).sum() / n_pairs
    with torch.no_grad():
        active = ((hinge > 0) & pair_valid).float().sum()
        active_frac = active / n_pairs
        hinge_mean = (hinge * pair_valid_f).sum() / active.clamp(min=1.0)
    return {"penalty": penalty, "active_frac": active_frac,
            "hinge_mean": hinge_mean, "n_pairs": pair_valid_f.sum()}


@dataclass
class OptionResult:
    z_start: np.ndarray
    proprio_start: np.ndarray
    task_state_start: np.ndarray
    completion_start: np.ndarray
    z_end: np.ndarray
    proprio_end: np.ndarray
    task_state_end: np.ndarray
    completion_end: np.ndarray
    chosen_task: int
    chosen_task_completed: bool
    any_task_completed: bool
    new_completions: int
    offtask_completions: int
    steps_taken: int
    env_done: bool
    termination_reason: str
    option_return: float
    env_reward_sum: float
    task_error_reduction: float
    frames: List[np.ndarray]
    last_worker_losses: Dict[str, float]


class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.mean = np.zeros(dim, dtype=np.float32)
        self.std = np.ones(dim, dtype=np.float32)

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = (x.std(axis=0) + self.eps).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


class SkillActor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, action_dim: int):
        super().__init__()
        self.trunk = build_mlp(input_dim, hidden_dim, hidden_dim, n_layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _dist(self, x: torch.Tensor) -> Normal:
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def forward(self, x: torch.Tensor):
        dist = self._dist(x)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logp.sum(-1, keepdim=True)

    def deterministic(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.tanh(self.mean_head(h))

    def log_prob_from_action(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self._dist(x)
        clipped = action.clamp(-0.999, 0.999)
        pre_tanh = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        logp = dist.log_prob(pre_tanh) - torch.log(1 - clipped.pow(2) + 1e-6)
        return logp.sum(-1, keepdim=True)


class TwinQ(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, action_dim: int):
        super().__init__()
        self.q1 = build_mlp(input_dim + action_dim, hidden_dim, 1, n_layers)
        self.q2 = build_mlp(input_dim + action_dim, hidden_dim, 1, n_layers)

    def forward(self, x: torch.Tensor, action: torch.Tensor):
        xa = torch.cat([x, action], dim=-1)
        return self.q1(xa), self.q2(xa)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.v = build_mlp(input_dim, hidden_dim, 1, n_layers)

    def forward(self, x: torch.Tensor):
        return self.v(x)


@dataclass
class Skill:
    actor: SkillActor
    critic: TwinQ
    critic_target: TwinQ
    value: ValueNet
    value_target: ValueNet
    actor_opt: torch.optim.Optimizer
    critic_opt: torch.optim.Optimizer
    value_opt: torch.optim.Optimizer


class SkillAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        self.tasks = list(config.training.tasks_to_complete)
        self.n_tasks = len(self.tasks)
        self.spec = TaskSpec(self.tasks, device=self.device)
        text_embs, text_src = build_frozen_text_embeddings(self.tasks, device=self.device)
        self.spec.attach_text_embeddings(text_embs, text_src)
        print(f"  [TaskSpec] text embeddings: {self.spec.text_source}  (dim={self.spec.text_embedding_dim})")

        self.encoder = VisualEncoder(config.encoder, device=self.device)
        self.z_dim = config.encoder.raw_dim
        self.proprio_dim = config.worker.proprio_dim
        self.max_goal_dim = self.spec.max_goal_dim
        self.H_chunk = max(1, config.worker.action_chunk_len)
        self.env_action_dim = 9
        self.action_dim = self.env_action_dim * self.H_chunk
        self.policy_input_dim = self.z_dim + self.proprio_dim + 4 * self.max_goal_dim

        self.proprio_norm = RunningNorm(self.proprio_dim)
        self.demo_dataset = None

        hidden = config.specialist.hidden_dim
        layers = config.specialist.n_layers
        self.skills: List[Skill] = []
        for _ in self.tasks:
            actor = SkillActor(self.policy_input_dim, hidden, layers, self.action_dim).to(self.device)
            critic = TwinQ(self.policy_input_dim, hidden, layers, self.action_dim).to(self.device)
            critic_target = copy.deepcopy(critic).to(self.device).eval()
            for p in critic_target.parameters():
                p.requires_grad_(False)
            value = ValueNet(self.policy_input_dim, hidden, layers).to(self.device)
            value_target = copy.deepcopy(value).to(self.device).eval()
            for p in value_target.parameters():
                p.requires_grad_(False)
            self.skills.append(
                Skill(
                    actor=actor,
                    critic=critic,
                    critic_target=critic_target,
                    value=value,
                    value_target=value_target,
                    actor_opt=torch.optim.Adam(actor.parameters(), lr=config.worker.actor_lr),
                    critic_opt=torch.optim.Adam(critic.parameters(), lr=config.worker.critic_lr),
                    value_opt=torch.optim.Adam(value.parameters(), lr=config.worker.critic_lr),
                )
            )

        self.stage_a_task_success = np.zeros(self.n_tasks, dtype=np.float32)
        self.curriculum_task_order = list(range(self.n_tasks))
        self.total_env_steps = 0
        self.total_options = 0
        self.total_episodes = 0
        self.epsilon = 0.0

    def normalize_proprio(self, p: np.ndarray) -> np.ndarray:
        return self.proprio_norm(p)

    def reset_optimizers(self, task_ids: Optional[List[int]] = None):
        """Re-create per-skill optimizers (fresh Adam moments).

        Checkpoints store network weights only, so after agent.load() the
        optimizers still hold first/second-moment estimates accumulated for
        the *discarded* weights. Call this after any load that is followed by
        further training (e.g. online rollback) so the first post-restore
        updates are not driven by stale momenta.
        """
        ids = list(range(self.n_tasks)) if task_ids is None else [int(k) for k in task_ids]
        for k in ids:
            skill = self.skills[k]
            skill.actor_opt = torch.optim.Adam(
                skill.actor.parameters(), lr=self.config.worker.actor_lr)
            skill.critic_opt = torch.optim.Adam(
                skill.critic.parameters(), lr=self.config.worker.critic_lr)
            skill.value_opt = torch.optim.Adam(
                skill.value.parameters(), lr=self.config.worker.critic_lr)

    def _input_arrays(self,
                      z: np.ndarray,
                      proprio: np.ndarray,
                      task_target: np.ndarray,
                      task_cur: np.ndarray,
                      task_mask: np.ndarray) -> np.ndarray:
        delta = (task_target - task_cur) * task_mask
        return np.concatenate([z, proprio, task_target, task_cur, delta, task_mask], axis=-1).astype(np.float32)

    def _input_from_batch(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        x = self._input_arrays(
            batch["z"].astype(np.float32),
            batch["proprio"].astype(np.float32),
            batch["task_target"].astype(np.float32),
            batch["task_cur"].astype(np.float32),
            batch["task_mask"].astype(np.float32),
        )
        return torch.from_numpy(x).to(self.device)

    def _input_next_from_batch(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        x = self._input_arrays(
            batch["z_next"].astype(np.float32),
            batch["proprio_next"].astype(np.float32),
            batch["task_target"].astype(np.float32),
            batch["task_cur_next"].astype(np.float32),
            batch["task_mask"].astype(np.float32),
        )
        return torch.from_numpy(x).to(self.device)

    def _input_from_state(self, z: np.ndarray, state: np.ndarray, task_id: int) -> torch.Tensor:
        p = self.normalize_proprio(state)
        tt = self.spec.padded_goal_for(task_id)
        tc = self.spec.padded_state_slice_for(state, task_id)
        tm = self.spec.padded_mask_for(task_id)
        x = self._input_arrays(
            z.astype(np.float32)[None, :],
            p.astype(np.float32)[None, :],
            tt[None, :],
            tc[None, :],
            tm[None, :],
        )
        return torch.from_numpy(x).to(self.device)

    def bc_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> float:
        skill = self.skills[task_id]
        x = self._input_from_batch(batch)
        a = torch.from_numpy(batch["action"]).to(self.device).clamp(-0.999, 0.999)
        pred = skill.actor.deterministic(x)
        loss = F.mse_loss(pred, a)
        skill.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.actor.parameters(), 1.0)
        skill.actor_opt.step()
        return float(loss.item())

    @staticmethod
    def _soft_update(src: nn.Module, dst: nn.Module, tau: float):
        with torch.no_grad():
            for p, p_targ in zip(src.parameters(), dst.parameters()):
                p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

    def iql_step(self,
                 task_id: int,
                 batch: Dict[str, np.ndarray],
                 segments: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        skill = self.skills[task_id]
        x = self._input_from_batch(batch)
        xn = self._input_next_from_batch(batch)
        a = torch.from_numpy(batch["action"]).to(self.device).clamp(-0.999, 0.999)
        r = torch.from_numpy(batch["reward"]).unsqueeze(1).to(self.device)
        d = torch.from_numpy(batch["done"]).unsqueeze(1).to(self.device)

        with torch.no_grad():
            q1_det, q2_det = skill.critic(x, a)
            q_det = torch.min(q1_det, q2_det)
        v = skill.value(x)
        adv = q_det - v
        expectile = self.config.specialist.iql_expectile
        weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        value_loss = (weight * adv.pow(2)).mean()
        skill.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.value.parameters(), 1.0)
        skill.value_opt.step()

        with torch.no_grad():
            next_v = skill.value_target(xn) if self.config.specialist.iql_use_value_target else skill.value(xn)
            target_q = r + self.config.worker.gamma * (1.0 - d) * next_v
        q1, q2 = skill.critic(x, a)
        critic_td_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        lql_metrics: Dict[str, float] = {}
        lambda_lb = float(self.config.specialist.lql_lambda_lb)
        if (segments is not None
                and bool(self.config.specialist.lql_enabled)
                and lambda_lb > 0.0):
            lb = self._lql_lb_loss(skill, segments)
            critic_loss = critic_td_loss + lambda_lb * lb["penalty"]
            lql_metrics = {
                "iql_lb_loss": float(lb["penalty"].item()),
                "iql_lb_active_frac": float(lb["active_frac"].item()),
                "iql_lb_hinge_mean": float(lb["hinge_mean"].item()),
                "iql_lb_pairs": float(lb["n_pairs"].item()),
                "iql_lb_chain_len_mean": float(lb["chain_len_mean"]),
            }
        else:
            critic_loss = critic_td_loss

        skill.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.critic.parameters(), 1.0)
        skill.critic_opt.step()

        with torch.no_grad():
            q1_pi, q2_pi = skill.critic(x, a)
            adv_pi = torch.min(q1_pi, q2_pi) - skill.value(x)
            adv_for_weight = adv_pi
            if self.config.specialist.iql_normalize_advantage:
                adv_std = adv_pi.std(unbiased=False).clamp(min=1e-6)
                adv_for_weight = (adv_pi - adv_pi.mean()) / adv_std
            exp_adv = torch.exp(self.config.specialist.iql_adv_beta * adv_for_weight).clamp(
                max=self.config.specialist.iql_max_weight
            )
        logp = skill.actor.log_prob_from_action(x, a)
        actor_loss = -(exp_adv * logp).mean()
        skill.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.actor.parameters(), 1.0)
        skill.actor_opt.step()
        if self.config.specialist.iql_use_value_target:
            tau = float(self.config.specialist.iql_value_target_tau)
            self._soft_update(skill.value, skill.value_target, tau)

        out = {
            "iql_value_loss": float(value_loss.item()),
            # iql_critic_loss stays the pure TD term for comparability with
            # pre-LQL baselines; the optimized total is logged separately.
            "iql_critic_loss": float(critic_td_loss.item()),
            "iql_critic_total_loss": float(critic_loss.item()),
            "iql_actor_loss": float(actor_loss.item()),
            "iql_adv_mean": float(adv_pi.mean().item()),
            "iql_adv_std": float(adv_pi.std(unbiased=False).item()),
            "iql_weight_mean": float(exp_adv.mean().item()),
            "iql_weight_max": float(exp_adv.max().item()),
            "iql_target_q_mean": float(target_q.mean().item()),
            "iql_q_mean": float(q_det.mean().item()),
            "iql_v_mean": float(v.mean().item()),
        }
        out.update(lql_metrics)
        return out

    def _lql_lb_loss(self, skill: Skill, segments: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Evaluate the LQL lower-bound penalty on a sampled chain batch.

        Network passes mirror the TD step: Q with gradient at every chain row,
        V_target (detached) at every row's next state (== chain state s_{l}).
        The hinge is applied to each twin head separately, matching how the TD
        loss treats q1/q2, then averaged.
        """
        x_seg = self._input_from_batch(segments)        # (B, L, in_dim)
        xn_seg = self._input_next_from_batch(segments)  # (B, L, in_dim)
        a_seg = torch.from_numpy(segments["action"]).to(self.device).clamp(-0.999, 0.999)
        r_seg = torch.from_numpy(segments["reward"]).to(self.device)
        d_seg = torch.from_numpy(segments["done"]).to(self.device)
        valid = torch.from_numpy(segments["valid"]).to(self.device)

        q1_seg, q2_seg = skill.critic(x_seg, a_seg)
        with torch.no_grad():
            v_next = skill.value_target(xn_seg).squeeze(-1)

        gamma = float(self.config.worker.gamma)
        min_gap = int(self.config.specialist.lql_min_gap)
        out1 = lql_lower_bound_penalty(
            q1_seg.squeeze(-1), v_next, r_seg, d_seg, valid, gamma, min_gap)
        out2 = lql_lower_bound_penalty(
            q2_seg.squeeze(-1), v_next, r_seg, d_seg, valid, gamma, min_gap)
        return {
            "penalty": 0.5 * (out1["penalty"] + out2["penalty"]),
            "active_frac": 0.5 * (out1["active_frac"] + out2["active_frac"]),
            "hinge_mean": 0.5 * (out1["hinge_mean"] + out2["hinge_mean"]),
            "n_pairs": out1["n_pairs"],
            "chain_len_mean": float(valid.float().sum(dim=1).mean().item()),
        }

    def online_awac_step(self,
                         task_id: int,
                         train_batch: Dict[str, np.ndarray],
                         demo_anchor_batch: Dict[str, np.ndarray],
                         actor_batch: Optional[Dict[str, np.ndarray]] = None,
                         bc_anchor_weight: Optional[float] = None) -> Dict[str, float]:
        """Conservative online AWAC update on mixed demo/online data.

        Critic: TD backup over mixed replay.
        Actor: advantage-weighted regression on high-quality actor replay + explicit demo BC anchor.
        """
        skill = self.skills[task_id]
        x = self._input_from_batch(train_batch)
        xn = self._input_next_from_batch(train_batch)
        a = torch.from_numpy(train_batch["action"]).to(self.device).clamp(-0.999, 0.999)
        r = torch.from_numpy(train_batch["reward"]).unsqueeze(1).to(self.device)
        d = torch.from_numpy(train_batch["done"]).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_a = skill.actor.deterministic(xn).clamp(-0.999, 0.999)
            tq1, tq2 = skill.critic_target(xn, next_a)
            target_q = r + self.config.worker.gamma * (1.0 - d) * torch.min(tq1, tq2)

        q1, q2 = skill.critic(x, a)
        if bool(self.config.online.critic_huber_loss):
            beta = max(float(self.config.online.critic_huber_delta), 1e-6)
            critic_loss = F.smooth_l1_loss(q1, target_q, beta=beta) + F.smooth_l1_loss(q2, target_q, beta=beta)
        else:
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        skill.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.critic.parameters(), 1.0)
        skill.critic_opt.step()

        actor_batch = actor_batch if actor_batch is not None else train_batch
        xa = self._input_from_batch(actor_batch)
        aa = torch.from_numpy(actor_batch["action"]).to(self.device).clamp(-0.999, 0.999)
        with torch.no_grad():
            q1_data, q2_data = skill.critic(xa, aa)
            q_data = torch.min(q1_data, q2_data)
            pi = skill.actor.deterministic(xa).clamp(-0.999, 0.999)
            q1_pi, q2_pi = skill.critic(xa, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            adv = q_data - q_pi
            adv_for_weight = adv
            if self.config.online.normalize_advantage:
                adv_for_weight = (adv - adv.mean()) / adv.std(unbiased=False).clamp(min=1e-6)
            weights = torch.exp(adv_for_weight / max(float(self.config.online.awac_temperature), 1e-6)).clamp(
                max=float(self.config.online.awac_max_weight)
            )

        logp = skill.actor.log_prob_from_action(xa, aa)
        awac_loss = -(weights * logp).mean()

        demo_x = self._input_from_batch(demo_anchor_batch)
        demo_a = torch.from_numpy(demo_anchor_batch["action"]).to(self.device).clamp(-0.999, 0.999)
        demo_pred = skill.actor.deterministic(demo_x)
        bc_anchor_loss = F.mse_loss(demo_pred, demo_a)
        anchor_weight = (
            float(self.config.online.bc_anchor_weight)
            if bc_anchor_weight is None
            else float(bc_anchor_weight)
        )
        actor_loss = awac_loss + anchor_weight * bc_anchor_loss

        skill.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.actor.parameters(), 1.0)
        skill.actor_opt.step()
        self._soft_update(skill.critic, skill.critic_target, float(self.config.online.critic_target_tau))

        return {
            "online_critic_loss": float(critic_loss.item()),
            "online_actor_loss": float(actor_loss.item()),
            "online_awac_loss": float(awac_loss.item()),
            "online_bc_anchor_loss": float(bc_anchor_loss.item()),
            "online_bc_anchor_weight": float(anchor_weight),
            "online_adv_mean": float(adv.mean().item()),
            "online_adv_std": float(adv.std(unbiased=False).item()),
            "online_weight_mean": float(weights.mean().item()),
            "online_weight_max": float(weights.max().item()),
            "online_target_q_mean": float(target_q.mean().item()),
        }

    @torch.no_grad()
    def get_worker_chunk(self,
                         z: np.ndarray,
                         proprio: np.ndarray,
                         full_state: np.ndarray,
                         task_id: int,
                         deterministic: bool = True) -> np.ndarray:
        del full_state
        x = self._input_from_state(z, proprio, task_id)
        skill = self.skills[task_id]
        action = skill.actor.deterministic(x) if deterministic else skill.actor(x)[0]
        return action.cpu().numpy().reshape(self.H_chunk, self.env_action_dim).clip(-1.0, 1.0)

    def _worker_step_reward(self,
                            spec_err_before: float,
                            spec_err_after: float,
                            action_step: np.ndarray,
                            completion_bit_flipped: bool,
                            task_id: int) -> float:
        cfg = self.config.worker
        eps = float(self.spec.epsilon(task_id))
        sigma = float(cfg.sigma)
        # Potential-based shaping: phi(s) = exp(-(e/eps)/sigma) in (0, 1].
        # Normalising by per-task eps makes sigma dimensionless and uniform
        # across tasks. The shaping term is gamma*phi(s') - phi(s), which
        # preserves the optimal policy (Ng et al. 1999).
        phi_before = float(np.exp(-(spec_err_before / eps) / sigma))
        phi_after  = float(np.exp(-(spec_err_after  / eps) / sigma))
        reward = cfg.progress_weight * (cfg.gamma * phi_after - phi_before)
        reward += cfg.completion_bonus * float(completion_bit_flipped)
        reward -= cfg.action_cost * float(np.sum(action_step ** 2))
        return float(reward)

    def execute_option(self,
                       env,
                       task_id: int,
                       start_img: np.ndarray,
                       start_state: np.ndarray,
                       start_z: np.ndarray,
                       completion: np.ndarray,
                       deterministic_worker: bool = True,
                       collect_frames: bool = False,
                       train_worker_online: bool = False,
                       update_every_n_env_steps: int = 1) -> OptionResult:
        del train_worker_online, update_every_n_env_steps
        chosen_name = self.spec.name(task_id)
        state = start_state.copy()
        z = start_z.copy()
        completion_mask = completion.copy()
        already_completed = {self.tasks[i] for i, v in enumerate(completion_mask) if v > 0.5}
        new_completion_names: List[str] = []
        start_error = self.spec.task_error(state, task_id)
        frames = [start_img.copy()] if collect_frames else []
        option_return = 0.0
        option_env_reward = 0.0
        env_done = False
        chosen_completed = False
        termination_reason = "budget"
        steps_taken = 0

        while steps_taken < self.config.manager.subgoal_horizon and not env_done and not chosen_completed:
            chunk = self.get_worker_chunk(z, state, state, task_id, deterministic=deterministic_worker)
            for h in range(self.H_chunk):
                if steps_taken >= self.config.manager.subgoal_horizon:
                    break
                action_step = chunk[h]
                next_img, env_reward, done_env, info = env.step(action_step)
                next_state = np.asarray(info["state"], dtype=np.float64)
                next_z = self.encoder.encode_numpy(next_img).squeeze()
                raw_completion = self.spec.completion_mask_from_names(info.get("tasks_completed_names", []))
                completion_next = np.maximum(completion_mask, raw_completion)

                just_completed = [
                    name for name in info.get("tasks_completed_names", [])
                    if name not in already_completed
                ]
                for name in just_completed:
                    already_completed.add(name)
                    if name not in new_completion_names:
                        new_completion_names.append(name)
                if chosen_name in just_completed:
                    chosen_completed = True
                    termination_reason = "completed"

                err_before = self.spec.task_error(state, task_id)
                err_after = self.spec.task_error(next_state, task_id)
                option_return += self._worker_step_reward(
                    err_before, err_after, action_step,
                    chosen_name in just_completed, task_id,
                )
                option_env_reward += float(env_reward)
                state = next_state
                z = next_z
                completion_mask = completion_next
                env_done = bool(done_env)
                steps_taken += 1
                self.total_env_steps += 1
                if collect_frames:
                    frames.append(next_img.copy())
                if env_done:
                    termination_reason = "env_done"
                    break
                if not chosen_completed and self.spec.is_close(state, task_id):
                    chosen_completed = True
                    termination_reason = "close_enough"
                    break

        self.total_options += 1
        return OptionResult(
            z_start=start_z.copy(),
            proprio_start=start_state.copy(),
            task_state_start=build_task_state_flat(self.spec, start_state),
            completion_start=completion.copy(),
            z_end=z.copy(),
            proprio_end=state.copy(),
            task_state_end=build_task_state_flat(self.spec, state),
            completion_end=completion_mask.copy(),
            chosen_task=task_id,
            chosen_task_completed=bool(chosen_completed),
            any_task_completed=bool(new_completion_names),
            new_completions=int(np.sum(completion_mask > completion)),
            offtask_completions=int(sum(1 for n in new_completion_names if n != chosen_name)),
            steps_taken=steps_taken,
            env_done=env_done,
            termination_reason=termination_reason,
            option_return=float(option_return),
            env_reward_sum=float(option_env_reward),
            task_error_reduction=float(start_error - self.spec.task_error(state, task_id)),
            frames=frames,
            last_worker_losses={},
        )

    def save(self, path: str):
        torch.save(
            {
                "tasks": self.tasks,
                "proprio_mean": self.proprio_norm.mean,
                "proprio_std": self.proprio_norm.std,
                "stage_a_task_success": self.stage_a_task_success,
                "curriculum_task_order": self.curriculum_task_order,
                "skills": [
                    {
                        "actor": s.actor.state_dict(),
                        "critic": s.critic.state_dict(),
                        "critic_target": s.critic_target.state_dict(),
                        "value": s.value.state_dict(),
                        "value_target": s.value_target.state_dict(),
                    }
                    for s in self.skills
                ],
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        saved_tasks = list(ckpt.get("tasks", []))
        if saved_tasks and saved_tasks != self.tasks:
            raise ValueError(f"Checkpoint tasks {saved_tasks} do not match current tasks {self.tasks}.")
        self.proprio_norm.mean = np.asarray(ckpt["proprio_mean"], dtype=np.float32)
        self.proprio_norm.std = np.asarray(ckpt["proprio_std"], dtype=np.float32)
        self.stage_a_task_success = np.asarray(
            ckpt.get("stage_a_task_success", np.zeros(self.n_tasks, dtype=np.float32)),
            dtype=np.float32,
        )
        self.curriculum_task_order = list(ckpt.get("curriculum_task_order", list(range(self.n_tasks))))
        for skill, saved in zip(self.skills, ckpt["skills"]):
            skill.actor.load_state_dict(saved["actor"])
            skill.critic.load_state_dict(saved["critic"])
            if "critic_target" in saved:
                skill.critic_target.load_state_dict(saved["critic_target"])
            else:
                skill.critic_target.load_state_dict(saved["critic"])
            skill.value.load_state_dict(saved["value"])
            if "value_target" in saved:
                skill.value_target.load_state_dict(saved["value_target"])
            else:
                skill.value_target.load_state_dict(saved["value"])
            skill.actor_opt = torch.optim.Adam(skill.actor.parameters(), lr=self.config.worker.actor_lr)
            skill.critic_opt = torch.optim.Adam(skill.critic.parameters(), lr=self.config.worker.critic_lr)
            skill.value_opt = torch.optim.Adam(skill.value.parameters(), lr=self.config.worker.critic_lr)


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def train_offline_skills(agent: SkillAgent,
                         config: Config,
                         verbose: bool = True,
                         writer=None) -> Dict[str, float]:
    if verbose:
        print(f"  [Stage A] Building demo dataset from: {', '.join(config.warmup.dataset_ids)}")
    ds, stats = build_or_load_demo_dataset(agent, config, verbose=verbose)
    agent.demo_dataset = ds
    agent.proprio_norm.fit(ds.w_p)

    results: Dict[str, float] = dict(stats)
    for task_id, task_name in enumerate(agent.tasks):
        safe = _safe_name(task_name)
        n = int(ds.worker_task_counts[task_id])
        if verbose:
            print(f"  [Skill] {task_name}: {n:,} demo transitions")
        if n == 0:
            continue

        algo = make_offline_algorithm(
            config.specialist.offline_algo,
            agent,
            config,
            writer=writer,
            verbose=verbose,
        )
        algo_result = algo.train_task(ds, task_id, task_name)
        results.update(algo_result.metrics)
        if verbose:
            _print_skill_metrics(task_name, safe, config.specialist.offline_algo, algo_result.metrics)

    if verbose:
        print("  [Stage A] Worker labels by task:")
        for k, name in enumerate(agent.tasks):
            print(f"    {name:<14} {int(ds.worker_task_counts[k]):,}")
    return results


def _print_skill_metrics(task_name: str, safe: str, algo: str, metrics: Dict[str, float]):
    del task_name
    if f"bc/{safe}_loss_final" in metrics:
        print(f"    BC final={metrics[f'bc/{safe}_loss_final']:.4f}  "
              f"best={metrics[f'bc/{safe}_loss_best']:.4f} @ step {int(metrics[f'bc/{safe}_best_step'])}")
    algo = algo.lower().replace("-", "_")
    if algo in ("bc_iql", "iql") and f"iql_value_loss/{safe}_final" in metrics:
        print(f"    IQL value={metrics[f'iql_value_loss/{safe}_final']:.4f}  "
              f"critic={metrics[f'iql_critic_loss/{safe}_final']:.4f}  "
              f"actor={metrics[f'iql_actor_loss/{safe}_final']:.4f}")
        if f"iql_adv_std/{safe}_final" in metrics:
            print(f"    IQL weights: adv_mean={metrics[f'iql_adv_mean/{safe}_final']:.4f}  "
                  f"adv_std={metrics[f'iql_adv_std/{safe}_final']:.4f}  "
                  f"weight_mean={metrics[f'iql_weight_mean/{safe}_final']:.4f}  "
                  f"weight_max={metrics[f'iql_weight_max/{safe}_final']:.4f}")
        if f"iql_prefix_success/{safe}_best" in metrics:
            print(f"    IQL prefix-val: success={metrics[f'iql_prefix_success/{safe}_best']*100:.1f}%  "
                  f"err={metrics[f'iql_prefix_error/{safe}_best']:.4f}  "
                  f"best_step={int(metrics[f'iql_prefix_best_step/{safe}'])}")
    elif algo in ("td3bc", "td3_bc") and f"td3bc_critic_loss/{safe}_final" in metrics:
        print(f"    TD3+BC critic={metrics[f'td3bc_critic_loss/{safe}_final']:.4f}  "
              f"actor={metrics[f'td3bc_actor_loss/{safe}_final']:.4f}  "
              f"bc={metrics[f'td3bc_bc_loss/{safe}_final']:.4f}")
    elif algo == "awr" and f"awr_critic_loss/{safe}_final" in metrics:
        print(f"    AWR value={metrics[f'awr_value_loss/{safe}_final']:.4f}  "
              f"critic={metrics[f'awr_critic_loss/{safe}_final']:.4f}  "
              f"actor={metrics[f'awr_actor_loss/{safe}_final']:.4f}")
    elif algo in ("bet", "behavior_transformer", "sequence_bc") and f"bet/{safe}_loss_final" in metrics:
        print(f"    BeT loss={metrics[f'bet/{safe}_loss_final']:.4f}  "
              f"best={metrics[f'bet/{safe}_loss_best']:.4f} @ step {int(metrics[f'bet/{safe}_best_step'])}")


__all__ = ["SkillAgent", "train_offline_skills", "OptionResult"]
