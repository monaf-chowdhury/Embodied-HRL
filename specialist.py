"""
specialist.py -- Per-skill specialist teacher/student pipeline.

This module implements the new default research path:

1. Shared frozen visual encoder.
2. One specialist policy per task.
3. Uniform offline training protocol for every skill:
   - privileged teacher trained with BC + IQL on demos
   - visual student distilled from the teacher on the same demos
4. Frozen scripted chaining evaluation after Stage A.

The specialist agent intentionally matches the subset of the old SMGWAgent API
used by train.py's evaluation code:
  * tasks, n_tasks, spec, encoder
  * execute_option(...)
  * save(path) / load(path)
  * total_env_steps / total_options / total_episodes
  * stage_a_task_success / curriculum_task_order
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

from agent import OptionResult, build_task_state_flat
from buffers import WorkerBuffer
from config import Config
from demo_dataset import build_or_load_demo_dataset, sample_oracle_prefix_states
from encoder import VisualEncoder
from env_wrapper import FrankaKitchenImageWrapper
from networks import build_mlp
from utils import TaskSpec, build_frozen_text_embeddings


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _zero_actor_mean(actor: "SkillActor"):
    nn.init.zeros_(actor.mean_head.weight)
    nn.init.zeros_(actor.mean_head.bias)


class _NullBuffer:
    def __len__(self):
        return 0


class RolloutSupervisionDataset:
    def __init__(self, input_dim: int, action_dim: int):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.inputs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []

    def add(self, inp: np.ndarray, action: np.ndarray):
        self.inputs.append(np.asarray(inp, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))

    def extend(self, other: "RolloutSupervisionDataset"):
        self.inputs.extend(other.inputs)
        self.actions.extend(other.actions)

    def finalize(self):
        self.inputs = np.stack(self.inputs).astype(np.float32) if self.inputs else np.zeros((0, self.input_dim), dtype=np.float32)
        self.actions = np.stack(self.actions).astype(np.float32) if self.actions else np.zeros((0, self.action_dim), dtype=np.float32)

    def __len__(self):
        if isinstance(self.inputs, list):
            return len(self.inputs)
        return int(self.inputs.shape[0])

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if len(self) == 0:
            raise RuntimeError("RolloutSupervisionDataset is empty.")
        idx = np.random.randint(0, len(self), size=batch_size)
        return {
            "input": self.inputs[idx],
            "action": self.actions[idx],
        }


class OnlineTeacherDataset:
    def __init__(self):
        self.rows: List[Dict[str, np.ndarray]] = []

    def add(self, row: Dict[str, np.ndarray]):
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if not self.rows:
            raise RuntimeError("OnlineTeacherDataset is empty.")
        idx = np.random.randint(0, len(self.rows), size=batch_size)
        keys = self.rows[0].keys()
        return {
            key: np.stack([self.rows[int(i)][key] for i in idx], axis=0)
            for key in keys
        }


class PrivilegedSkillReward:
    EE_PATTERNS = ["end_effector", "ee", "eef", "gripper", "panda_hand", "right_hand"]
    TASK_PATTERNS = {
        "microwave": ["micro", "microwave", "handle"],
        "kettle": ["kettle"],
        "light switch": ["light", "switch"],
        "slide cabinet": ["slide", "cabinet", "handle"],
    }

    def __init__(self, agent: "SpecialistSkillAgent", config: Config, env: FrankaKitchenImageWrapper, task_id: int):
        self.agent = agent
        self.config = config
        self.env = env
        self.task_id = task_id
        self.task_name = agent.tasks[task_id]
        self._warned_missing_geom = False

    def _ee_pos(self) -> Optional[np.ndarray]:
        return self.env.xpos_by_name_patterns(self.EE_PATTERNS)

    def _target_pos(self) -> Optional[np.ndarray]:
        patterns = list(self.TASK_PATTERNS.get(self.task_name, [self.task_name]))
        if "handle" not in patterns and self.task_name in ("microwave", "slide cabinet"):
            patterns.append("handle")
        return self.env.xpos_by_name_patterns(patterns)

    def positions(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._ee_pos(), self._target_pos()

    def approach_distance(self) -> Optional[float]:
        ee = self._ee_pos()
        target = self._target_pos()
        if ee is None or target is None:
            if not self._warned_missing_geom:
                self._warned_missing_geom = True
                print(f"    [TeacherOnline] geometry fallback for '{self.task_name}': "
                      "could not resolve end-effector or target affordance name.")
            return None
        return float(np.linalg.norm(ee - target))

    def step_reward(self,
                    state_before: np.ndarray,
                    state_after: np.ndarray,
                    approach_before: Optional[float],
                    approach_after: Optional[float],
                    action: np.ndarray,
                    completed: bool) -> float:
        cfg = self.config.specialist
        task_progress = self.agent.spec.task_error(state_before, self.task_id) - self.agent.spec.task_error(state_after, self.task_id)
        reward = cfg.teacher_online_reward_task_weight * task_progress
        if approach_before is not None and approach_after is not None:
            reward += cfg.teacher_online_reward_approach_weight * (approach_before - approach_after)
        reward += cfg.teacher_online_reward_completion * float(completed)
        reward -= cfg.teacher_online_reward_action_cost * float(np.sum(action ** 2))
        return float(reward)


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

    def get_action_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.tanh(self.mean_head(h))

    def log_prob_from_action(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self._dist(x)
        clipped = action.clamp(-0.999, 0.999)
        pre_tanh = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        logp = dist.log_prob(pre_tanh) - torch.log(1 - clipped.pow(2) + 1e-6)
        return logp.sum(-1, keepdim=True)


class SkillTwinQ(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, action_dim: int):
        super().__init__()
        self.q1 = build_mlp(input_dim + action_dim, hidden_dim, 1, n_layers)
        self.q2 = build_mlp(input_dim + action_dim, hidden_dim, 1, n_layers)

    def forward(self, x: torch.Tensor, action: torch.Tensor):
        qa = torch.cat([x, action], dim=-1)
        return self.q1(qa), self.q2(qa)


class SkillValue(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.v = build_mlp(input_dim, hidden_dim, 1, n_layers)

    def forward(self, x: torch.Tensor):
        return self.v(x)


@dataclass
class SkillModules:
    teacher_actor: SkillActor
    teacher_base_actor: SkillActor
    teacher_residual_actor: SkillActor
    teacher_critic: SkillTwinQ
    teacher_value: SkillValue
    student_actor: SkillActor
    teacher_actor_opt: torch.optim.Optimizer
    teacher_residual_opt: torch.optim.Optimizer
    teacher_critic_opt: torch.optim.Optimizer
    teacher_value_opt: torch.optim.Optimizer
    student_actor_opt: torch.optim.Optimizer


class SpecialistSkillAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        self.tasks: List[str] = list(config.training.tasks_to_complete)
        self.n_tasks = len(self.tasks)

        self.spec = TaskSpec(self.tasks, device=self.device)
        text_embs, text_src = build_frozen_text_embeddings(self.tasks, device=self.device)
        self.spec.attach_text_embeddings(text_embs, text_src)
        print(f"  [TaskSpec] text embeddings: {self.spec.text_source}  "
              f"(dim={self.spec.text_embedding_dim})")

        self.encoder = VisualEncoder(config.encoder, device=self.device)

        self.proprio_dim = config.worker.proprio_dim
        self.z_dim = config.encoder.raw_dim
        self.max_goal_dim = self.spec.max_goal_dim
        self.H_chunk = max(1, config.worker.action_chunk_len)
        self.env_action_dim = 9
        self.action_dim = self.env_action_dim * self.H_chunk

        self.worker_buf = WorkerBuffer(
            capacity=config.buffer.worker_capacity,
            z_dim=self.z_dim,
            proprio_dim=self.proprio_dim,
            action_dim=self.env_action_dim,
            action_chunk_len=self.H_chunk,
            max_goal_dim=self.max_goal_dim,
            n_tasks=self.n_tasks,
            z_dtype=np.float16 if config.buffer.z_storage_dtype == "float16" else np.float32,
        )
        self.manager_buf = _NullBuffer()
        self.demo_dataset = None

        self.teacher_affordance_dim = 16 + self.max_goal_dim
        teacher_input_dim = self.proprio_dim + 4 * self.max_goal_dim + self.teacher_affordance_dim
        student_input_dim = self.z_dim + self.proprio_dim
        hidden = config.specialist.hidden_dim
        layers = config.specialist.n_layers

        self.skills: List[SkillModules] = []
        for _ in self.tasks:
            teacher_actor = SkillActor(teacher_input_dim, hidden, layers, self.action_dim).to(self.device)
            teacher_base_actor = SkillActor(teacher_input_dim, hidden, layers, self.action_dim).to(self.device)
            teacher_residual_actor = SkillActor(teacher_input_dim, hidden, layers, self.action_dim).to(self.device)
            _zero_actor_mean(teacher_residual_actor)
            teacher_critic = SkillTwinQ(teacher_input_dim, hidden, layers, self.action_dim).to(self.device)
            teacher_value = SkillValue(teacher_input_dim, hidden, layers).to(self.device)
            student_actor = SkillActor(student_input_dim, hidden, layers, self.action_dim).to(self.device)
            self.skills.append(
                SkillModules(
                    teacher_actor=teacher_actor,
                    teacher_base_actor=teacher_base_actor,
                    teacher_residual_actor=teacher_residual_actor,
                    teacher_critic=teacher_critic,
                    teacher_value=teacher_value,
                    student_actor=student_actor,
                    teacher_actor_opt=torch.optim.Adam(teacher_actor.parameters(), lr=config.worker.actor_lr),
                    teacher_residual_opt=torch.optim.Adam(teacher_residual_actor.parameters(), lr=config.worker.actor_lr),
                    teacher_critic_opt=torch.optim.Adam(teacher_critic.parameters(), lr=config.worker.critic_lr),
                    teacher_value_opt=torch.optim.Adam(teacher_value.parameters(), lr=config.worker.critic_lr),
                    student_actor_opt=torch.optim.Adam(student_actor.parameters(), lr=config.worker.actor_lr),
                )
            )

        self.stage_a_task_success = np.zeros(self.n_tasks, dtype=np.float32)
        self.curriculum_task_order = list(range(self.n_tasks))
        self.rollout_policy_source = "student"
        self.total_env_steps = 0
        self.total_options = 0
        self.total_episodes = 0
        self.epsilon = config.manager.epsilon_end

    def _teacher_input_arrays(self,
                              proprio: np.ndarray,
                              task_target: np.ndarray,
                              task_cur: np.ndarray,
                              task_mask: np.ndarray,
                              affordance: Optional[np.ndarray] = None) -> np.ndarray:
        delta = (task_target - task_cur) * task_mask
        if affordance is None:
            affordance = self._fallback_affordance_arrays(proprio, task_target, task_cur, task_mask)
        affordance = np.asarray(affordance, dtype=np.float32)
        if affordance.ndim == 1:
            affordance = affordance[None, :]
        return np.concatenate(
            [proprio, task_target, task_cur, delta, task_mask, affordance],
            axis=-1,
        ).astype(np.float32)

    def _fallback_affordance_arrays(self,
                                    proprio: np.ndarray,
                                    task_target: np.ndarray,
                                    task_cur: np.ndarray,
                                    task_mask: np.ndarray) -> np.ndarray:
        proprio = np.asarray(proprio, dtype=np.float32)
        task_target = np.asarray(task_target, dtype=np.float32)
        task_cur = np.asarray(task_cur, dtype=np.float32)
        task_mask = np.asarray(task_mask, dtype=np.float32)
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        if task_target.ndim == 1:
            task_target = task_target[None, :]
            task_cur = task_cur[None, :]
            task_mask = task_mask[None, :]

        n = proprio.shape[0]
        delta = (task_target - task_cur) * task_mask
        task_error = np.linalg.norm(delta, axis=-1, keepdims=True).astype(np.float32)
        gripper = np.zeros((n, 2), dtype=np.float32)
        if proprio.shape[1] >= 9:
            gripper = proprio[:, 7:9].astype(np.float32)
        gripper_open = np.mean(gripper, axis=-1, keepdims=True).astype(np.float32)
        zeros3 = np.zeros((n, 3), dtype=np.float32)
        return np.concatenate(
            [
                zeros3,              # ee_to_target xyz, geometry unavailable
                task_error,          # distance proxy in task coordinates
                zeros3,              # target xyz
                zeros3,              # ee xyz
                gripper,
                gripper_open,
                task_error,
                np.zeros((n, 1), dtype=np.float32),  # contact flag
                np.ones((n, 1), dtype=np.float32),   # min contact distance fallback
                delta.astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

    def affordance_features_from_env(self,
                                     env: FrankaKitchenImageWrapper,
                                     full_state: np.ndarray,
                                     task_id: int) -> np.ndarray:
        p = self.worker_buf.normalize_proprio(full_state).astype(np.float32)
        tt = self.spec.padded_goal_for(task_id)
        tc = self.spec.padded_state_slice_for(full_state, task_id)
        tm = self.spec.padded_mask_for(task_id)
        features = self._fallback_affordance_arrays(
            p[None, :], tt[None, :], tc[None, :], tm[None, :]
        )[0]

        rewarder = PrivilegedSkillReward(self, self.config, env, task_id)
        ee, target = rewarder.positions()
        if ee is not None and target is not None:
            ee = np.asarray(ee, dtype=np.float32)
            target = np.asarray(target, dtype=np.float32)
            ee_to_target = target - ee
            features[0:3] = ee_to_target
            features[3] = float(np.linalg.norm(ee_to_target))
            features[4:7] = target
            features[7:10] = ee
        if len(full_state) >= 9:
            features[10:12] = np.asarray(full_state[7:9], dtype=np.float32)
            features[12] = float(np.mean(full_state[7:9]))
        features[13] = float(self.spec.task_error(full_state, task_id))
        patterns = PrivilegedSkillReward.TASK_PATTERNS.get(self.tasks[task_id], [self.tasks[task_id]])
        contact = env.contact_features_by_name_patterns(
            PrivilegedSkillReward.EE_PATTERNS,
            list(patterns),
        )
        features[14:16] = contact.astype(np.float32)
        features[16:] = ((tt - tc) * tm).astype(np.float32)
        return features.astype(np.float32)

    def _teacher_input_from_batch(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        x = self._teacher_input_arrays(
            batch["proprio"], batch["task_target"], batch["task_cur"], batch["task_mask"],
            batch.get("affordance"),
        )
        return torch.from_numpy(x).to(self.device)

    def _teacher_input_next_from_batch(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        x = self._teacher_input_arrays(
            batch["proprio_next"], batch["task_target"], batch["task_cur_next"], batch["task_mask"],
            batch.get("affordance_next"),
        )
        return torch.from_numpy(x).to(self.device)

    def _student_input_from_batch(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        x = np.concatenate([batch["z"], batch["proprio"]], axis=-1).astype(np.float32)
        return torch.from_numpy(x).to(self.device)

    def _teacher_input_from_state(self,
                                  proprio: np.ndarray,
                                  full_state: np.ndarray,
                                  task_id: int,
                                  affordance: Optional[np.ndarray] = None) -> torch.Tensor:
        p = self.worker_buf.normalize_proprio(proprio).astype(np.float32)
        tt = self.spec.padded_goal_for(task_id)
        tc = self.spec.padded_state_slice_for(full_state, task_id)
        tm = self.spec.padded_mask_for(task_id)
        x = self._teacher_input_arrays(
            p[None, :],
            tt[None, :],
            tc[None, :],
            tm[None, :],
            None if affordance is None else np.asarray(affordance, dtype=np.float32)[None, :],
        )
        return torch.from_numpy(x).to(self.device)

    def freeze_teacher_base(self, task_id: int):
        skill = self.skills[task_id]
        skill.teacher_base_actor.load_state_dict(copy.deepcopy(skill.teacher_actor.state_dict()))
        skill.teacher_base_actor.eval()
        for p in skill.teacher_base_actor.parameters():
            p.requires_grad = False
        _zero_actor_mean(skill.teacher_residual_actor)

    def _student_input_from_state(self, z: np.ndarray, proprio: np.ndarray) -> torch.Tensor:
        p = self.worker_buf.normalize_proprio(proprio).astype(np.float32)
        x = np.concatenate([z.astype(np.float32), p], axis=0)[None, :]
        return torch.from_numpy(x).to(self.device)

    @torch.no_grad()
    def get_teacher_action_deterministic(self,
                                         proprio: np.ndarray,
                                         full_state: np.ndarray,
                                         task_id: int,
                                         affordance: Optional[np.ndarray] = None) -> np.ndarray:
        skill = self.skills[task_id]
        x = self._teacher_input_from_state(proprio, full_state, task_id, affordance=affordance)
        if self.config.specialist.teacher_residual_online:
            base = skill.teacher_base_actor.get_action_deterministic(x)
            residual = skill.teacher_residual_actor.get_action_deterministic(x)
            action = torch.clamp(
                base + float(self.config.specialist.teacher_residual_scale) * residual,
                -0.999,
                0.999,
            )
        else:
            action = skill.teacher_actor.get_action_deterministic(x)
        return action.cpu().numpy().reshape(self.H_chunk, self.env_action_dim)

    def teacher_bc_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> float:
        skill = self.skills[task_id]
        x = self._teacher_input_from_batch(batch)
        a = torch.from_numpy(batch["action"]).to(self.device)
        pred = skill.teacher_actor.get_action_deterministic(x)
        loss = F.mse_loss(pred, a.clamp(-0.999, 0.999))
        skill.teacher_actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.teacher_actor.parameters(), 1.0)
        skill.teacher_actor_opt.step()
        return float(loss.item())

    def teacher_iql_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        skill = self.skills[task_id]
        x = self._teacher_input_from_batch(batch)
        xn = self._teacher_input_next_from_batch(batch)
        a = torch.from_numpy(batch["action"]).to(self.device)
        r = torch.from_numpy(batch["reward"]).unsqueeze(1).to(self.device)
        d = torch.from_numpy(batch["done"]).unsqueeze(1).to(self.device)

        with torch.no_grad():
            q1_det, q2_det = skill.teacher_critic(x, a)
            q_det = torch.min(q1_det, q2_det)
        v = skill.teacher_value(x)
        adv = q_det - v
        expectile = self.config.specialist.iql_expectile
        weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        value_loss = (weight * adv.pow(2)).mean()
        skill.teacher_value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.teacher_value.parameters(), 1.0)
        skill.teacher_value_opt.step()

        with torch.no_grad():
            v_next = skill.teacher_value(xn)
            gamma_eff = self.config.worker.gamma ** self.H_chunk
            target_q = r + gamma_eff * (1.0 - d) * v_next
        q1, q2 = skill.teacher_critic(x, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        skill.teacher_critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.teacher_critic.parameters(), 1.0)
        skill.teacher_critic_opt.step()

        with torch.no_grad():
            q1_pi, q2_pi = skill.teacher_critic(x, a)
            v_pi = skill.teacher_value(x)
            adv_pi = torch.min(q1_pi, q2_pi) - v_pi
            exp_adv = torch.exp(self.config.specialist.iql_adv_beta * adv_pi).clamp(
                max=self.config.specialist.iql_max_weight
            )
        logp = skill.teacher_actor.log_prob_from_action(x, a)
        actor_loss = -(exp_adv * logp).mean()
        skill.teacher_actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.teacher_actor.parameters(), 1.0)
        skill.teacher_actor_opt.step()

        return {
            "teacher_iql_value_loss": float(value_loss.item()),
            "teacher_iql_critic_loss": float(critic_loss.item()),
            "teacher_iql_actor_loss": float(actor_loss.item()),
            "teacher_iql_adv_mean": float(adv_pi.mean().item()),
            "teacher_iql_weight_mean": float(exp_adv.mean().item()),
        }

    def teacher_residual_iql_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        skill = self.skills[task_id]
        x = self._teacher_input_from_batch(batch)
        xn = self._teacher_input_next_from_batch(batch)
        a = torch.from_numpy(batch["action"]).to(self.device)
        r = torch.from_numpy(batch["reward"]).unsqueeze(1).to(self.device)
        d = torch.from_numpy(batch["done"]).unsqueeze(1).to(self.device)

        with torch.no_grad():
            q1_det, q2_det = skill.teacher_critic(x, a)
            q_det = torch.min(q1_det, q2_det)
        v = skill.teacher_value(x)
        adv = q_det - v
        expectile = self.config.specialist.iql_expectile
        weight = torch.where(adv > 0, expectile, 1.0 - expectile)
        value_loss = (weight * adv.pow(2)).mean()
        skill.teacher_value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.teacher_value.parameters(), 1.0)
        skill.teacher_value_opt.step()

        with torch.no_grad():
            v_next = skill.teacher_value(xn)
            gamma_eff = self.config.worker.gamma ** self.H_chunk
            target_q = r + gamma_eff * (1.0 - d) * v_next
        q1, q2 = skill.teacher_critic(x, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        skill.teacher_critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.teacher_critic.parameters(), 1.0)
        skill.teacher_critic_opt.step()

        with torch.no_grad():
            base = skill.teacher_base_actor.get_action_deterministic(x)
            scale = max(float(self.config.specialist.teacher_residual_scale), 1e-6)
            residual_target = ((a - base) / scale).clamp(-0.999, 0.999)
            q1_pi, q2_pi = skill.teacher_critic(x, a)
            v_pi = skill.teacher_value(x)
            adv_pi = torch.min(q1_pi, q2_pi) - v_pi
            exp_adv = torch.exp(self.config.specialist.iql_adv_beta * adv_pi).clamp(
                max=self.config.specialist.iql_max_weight
            )
        logp = skill.teacher_residual_actor.log_prob_from_action(x, residual_target)
        actor_loss = -(exp_adv * logp).mean()
        skill.teacher_residual_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.teacher_residual_actor.parameters(), 1.0)
        skill.teacher_residual_opt.step()

        with torch.no_grad():
            residual_pred = skill.teacher_residual_actor.get_action_deterministic(x)
            action_pred = (base + scale * residual_pred).clamp(-0.999, 0.999)
            action_mse = F.mse_loss(action_pred, a.clamp(-0.999, 0.999))
            residual_l2 = residual_pred.pow(2).mean()

        return {
            "teacher_residual_value_loss": float(value_loss.item()),
            "teacher_residual_critic_loss": float(critic_loss.item()),
            "teacher_residual_actor_loss": float(actor_loss.item()),
            "teacher_residual_action_mse": float(action_mse.item()),
            "teacher_residual_l2": float(residual_l2.item()),
            "teacher_residual_adv_mean": float(adv_pi.mean().item()),
            "teacher_residual_weight_mean": float(exp_adv.mean().item()),
        }

    def student_distill_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        skill = self.skills[task_id]
        teacher_x = self._teacher_input_from_batch(batch)
        student_x = self._student_input_from_batch(batch)
        demo_action = torch.from_numpy(batch["action"]).to(self.device)
        with torch.no_grad():
            teacher_action = skill.teacher_actor.get_action_deterministic(teacher_x)
        student_action = skill.student_actor.get_action_deterministic(student_x)
        distill_loss = F.mse_loss(student_action, teacher_action)
        demo_loss = F.mse_loss(student_action, demo_action.clamp(-0.999, 0.999))
        loss = distill_loss + self.config.specialist.student_demo_bc_weight * demo_loss
        skill.student_actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.student_actor.parameters(), 1.0)
        skill.student_actor_opt.step()
        return {
            "student_distill_loss": float(distill_loss.item()),
            "student_demo_bc_loss": float(demo_loss.item()),
            "student_total_loss": float(loss.item()),
        }

    def student_rollout_distill_step(self, task_id: int, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        skill = self.skills[task_id]
        x = torch.from_numpy(batch["input"]).to(self.device)
        target_action = torch.from_numpy(batch["action"]).to(self.device)
        pred_action = skill.student_actor.get_action_deterministic(x)
        loss = F.mse_loss(pred_action, target_action.clamp(-0.999, 0.999))
        skill.student_actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(skill.student_actor.parameters(), 1.0)
        skill.student_actor_opt.step()
        return {"student_rollout_distill_loss": float(loss.item())}

    @torch.no_grad()
    def get_worker_chunk(self,
                         z: np.ndarray,
                         proprio: np.ndarray,
                         full_state: np.ndarray,
                         task_id: int,
                         deterministic: bool = False,
                         teacher_affordance: Optional[np.ndarray] = None) -> np.ndarray:
        if self.rollout_policy_source == "teacher":
            return np.clip(
                self.get_teacher_action_deterministic(
                    proprio, full_state, task_id, affordance=teacher_affordance
                ),
                -1.0,
                1.0,
            )

        del full_state
        skill = self.skills[task_id]
        x = self._student_input_from_state(z, proprio)
        if deterministic:
            action = skill.student_actor.get_action_deterministic(x)
        else:
            action, _ = skill.student_actor(x)
        chunk = action.cpu().numpy().reshape(1, self.H_chunk, self.env_action_dim)
        return np.clip(chunk.squeeze(0), -1.0, 1.0)

    def _worker_step_reward(self,
                            spec_err_before: float,
                            spec_err_after: float,
                            action_step: np.ndarray,
                            completion_bit_flipped: bool) -> float:
        cfg = self.config.worker
        progress = spec_err_before - spec_err_after
        reward = cfg.progress_weight * progress
        reward += cfg.completion_bonus * (1.0 if completion_bit_flipped else 0.0)
        reward -= cfg.action_cost * float(np.sum(action_step ** 2))
        return float(reward)

    def execute_option(self,
                       env,
                       task_id: int,
                       start_img: np.ndarray,
                       start_state: np.ndarray,
                       start_z: np.ndarray,
                       completion: np.ndarray,
                       deterministic_worker: bool = False,
                       collect_frames: bool = False,
                       train_worker_online: bool = False,
                       update_every_n_env_steps: int = 1) -> OptionResult:
        del train_worker_online, update_every_n_env_steps

        cfg = self.config
        task_state_start_flat = build_task_state_flat(self.spec, start_state)
        chosen_name = self.spec.name(task_id)

        state = start_state.copy()
        z = start_z.copy()
        proprio = state.copy()
        completion_mask = completion.copy()

        steps_taken = 0
        env_done = False
        chosen_completed = False
        termination_reason = "budget"
        option_return = 0.0
        option_env_reward = 0.0
        last_worker_losses: Dict[str, float] = {}
        frames = [start_img.copy()] if collect_frames else []

        already_completed = set(
            name for idx, name in enumerate(self.tasks) if completion_mask[idx] > 0.5
        )
        new_completion_names: List[str] = []
        start_error = self.spec.task_error(start_state, task_id)

        while steps_taken < cfg.manager.subgoal_horizon and not env_done and not chosen_completed:
            teacher_affordance = (
                self.affordance_features_from_env(env, state, task_id)
                if self.rollout_policy_source == "teacher" else None
            )
            chunk = self.get_worker_chunk(
                z=z, proprio=proprio, full_state=state, task_id=task_id,
                deterministic=deterministic_worker,
                teacher_affordance=teacher_affordance,
            )
            for h in range(self.H_chunk):
                if steps_taken >= cfg.manager.subgoal_horizon:
                    break
                action_step = chunk[h]
                next_img, env_reward, done_env, info = env.step(action_step)
                next_state = np.asarray(info["state"], dtype=np.float64)
                next_z = self.encoder.encode_numpy(next_img).squeeze()
                completion_names_now = info.get("tasks_completed_names", [])
                raw_completion_next = self.spec.completion_mask_from_names(completion_names_now)
                # Kitchen observations can report currently satisfied goals,
                # not necessarily the episode history. Chaining requires
                # episode-level memory: once a task is achieved, keep it done.
                completion_next = np.maximum(completion_mask, raw_completion_next)

                just_completed = [name for name in completion_names_now if name not in already_completed]
                if just_completed:
                    already_completed.update(just_completed)
                    for name in just_completed:
                        if name not in new_completion_names:
                            new_completion_names.append(name)
                    if chosen_name in just_completed:
                        chosen_completed = True
                        termination_reason = "completed"

                err_before = self.spec.task_error(state, task_id)
                err_after = self.spec.task_error(next_state, task_id)
                option_return += self._worker_step_reward(
                    spec_err_before=err_before,
                    spec_err_after=err_after,
                    action_step=action_step,
                    completion_bit_flipped=(chosen_name in just_completed),
                )
                option_env_reward += float(env_reward)

                state = next_state
                proprio = next_state
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

        if not chosen_completed and not env_done and steps_taken >= cfg.manager.subgoal_horizon:
            termination_reason = "budget"

        end_task_state = build_task_state_flat(self.spec, state)
        task_error_reduction = start_error - self.spec.task_error(state, task_id)
        new_completions = int(np.sum(completion_mask > completion) if completion_mask.shape == completion.shape else 0)
        offtask_completions = int(sum(1 for name in new_completion_names if name != chosen_name))

        self.total_options += 1
        return OptionResult(
            z_start=start_z.copy(),
            proprio_start=start_state.copy(),
            task_state_start=task_state_start_flat,
            completion_start=completion.copy(),
            z_end=z.copy(),
            proprio_end=state.copy(),
            task_state_end=end_task_state,
            completion_end=completion_mask.copy(),
            chosen_task=task_id,
            chosen_task_completed=bool(chosen_name in new_completion_names or termination_reason == "close_enough"),
            any_task_completed=bool(len(new_completion_names) > 0),
            new_completions=new_completions,
            offtask_completions=offtask_completions,
            steps_taken=steps_taken,
            env_done=env_done,
            termination_reason=termination_reason,
            option_return=float(option_return),
            env_reward_sum=float(option_env_reward),
            task_error_reduction=float(task_error_reduction),
            frames=frames,
            last_worker_losses=last_worker_losses,
        )

    def save(self, path: str):
        ckpt = {
            "total_env_steps": self.total_env_steps,
            "total_options": self.total_options,
            "total_episodes": self.total_episodes,
            "stage_a_task_success": self.stage_a_task_success,
            "curriculum_task_order": self.curriculum_task_order,
            "proprio_mean": self.worker_buf.proprio_stats.mean,
            "proprio_M2": self.worker_buf.proprio_stats.M2,
            "proprio_n": self.worker_buf.proprio_stats.n,
            "skills": [],
        }
        for skill in self.skills:
            ckpt["skills"].append({
                "teacher_actor": skill.teacher_actor.state_dict(),
                "teacher_base_actor": skill.teacher_base_actor.state_dict(),
                "teacher_residual_actor": skill.teacher_residual_actor.state_dict(),
                "teacher_critic": skill.teacher_critic.state_dict(),
                "teacher_value": skill.teacher_value.state_dict(),
                "student_actor": skill.student_actor.state_dict(),
            })
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.total_env_steps = ckpt.get("total_env_steps", 0)
        self.total_options = ckpt.get("total_options", 0)
        self.total_episodes = ckpt.get("total_episodes", 0)
        self.stage_a_task_success = ckpt.get("stage_a_task_success", self.stage_a_task_success)
        self.curriculum_task_order = ckpt.get("curriculum_task_order", self.curriculum_task_order)
        self.worker_buf.proprio_stats.mean = ckpt["proprio_mean"]
        self.worker_buf.proprio_stats.M2 = ckpt["proprio_M2"]
        self.worker_buf.proprio_stats.n = ckpt["proprio_n"]
        for state, skill in zip(ckpt["skills"], self.skills):
            skill.teacher_actor.load_state_dict(state["teacher_actor"])
            if "teacher_base_actor" in state:
                skill.teacher_base_actor.load_state_dict(state["teacher_base_actor"])
            else:
                skill.teacher_base_actor.load_state_dict(state["teacher_actor"])
            if "teacher_residual_actor" in state:
                skill.teacher_residual_actor.load_state_dict(state["teacher_residual_actor"])
            skill.teacher_critic.load_state_dict(state["teacher_critic"])
            skill.teacher_value.load_state_dict(state["teacher_value"])
            skill.student_actor.load_state_dict(state["student_actor"])


def _evaluate_specialist_skill(agent: SpecialistSkillAgent,
                               config: Config,
                               task_id: int,
                               source: str = "student",
                               n_episodes: int = 10) -> Dict[str, float]:
    task_name = agent.tasks[task_id]
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=[task_name],
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=True,
    )
    prev_source = agent.rollout_policy_source
    agent.rollout_policy_source = source
    try:
        success = []
        option_counts = []
        env_rewards = []
        for ep_idx in range(n_episodes):
            img, state = env.reset(seed=config.training.seed + 40_000 + 1000 * task_id + ep_idx)
            z = agent.encoder.encode_numpy(img).squeeze()
            completion = np.zeros(agent.n_tasks, dtype=np.float32)
            n_options = 0
            ep_reward = 0.0
            done = False
            while (not done
                   and completion[task_id] < 0.5
                   and n_options < config.manager.max_high_level_steps):
                result = agent.execute_option(
                    env=env,
                    task_id=task_id,
                    start_img=img,
                    start_state=state,
                    start_z=z,
                    completion=completion,
                    deterministic_worker=True,
                    collect_frames=False,
                    train_worker_online=False,
                )
                state = result.proprio_end
                z = result.z_end
                completion = result.completion_end
                ep_reward += result.env_reward_sum
                n_options += 1
                done = result.env_done
                img = env.render_image()
            success.append(float(completion[task_id] > 0.5))
            option_counts.append(float(n_options))
            env_rewards.append(float(ep_reward))
    finally:
        agent.rollout_policy_source = prev_source
        env.close()

    return {
        "success_rate": float(np.mean(success)) if success else 0.0,
        "mean_options": float(np.mean(option_counts)) if option_counts else 0.0,
        "mean_env_reward": float(np.mean(env_rewards)) if env_rewards else 0.0,
    }


def _collect_supervision_from_starts(agent: SpecialistSkillAgent,
                                     config: Config,
                                     task_id: int,
                                     starts: List[Dict[str, np.ndarray]],
                                     execution_source: str,
                                     query_source: str) -> RolloutSupervisionDataset:
    dataset = RolloutSupervisionDataset(
        input_dim=agent.z_dim + agent.proprio_dim,
        action_dim=agent.action_dim,
    )
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=False,
    )
    prev_source = agent.rollout_policy_source
    try:
        for sample_idx, sample in enumerate(starts):
            env.reset(seed=config.training.seed + 50_000 + 100 * task_id + sample_idx)
            if "state" in sample and "completion" in sample:
                qpos, qvel = env.observation_to_qpos_qvel(sample["state"])
                env.set_mujoco_state(qpos, qvel)
                env._current_obs = {"observation": np.asarray(sample["state"], dtype=np.float64).copy()}
                env._step_count = 0
                state = np.asarray(sample["state"], dtype=np.float64).copy()
                completion = np.asarray(sample["completion"], dtype=np.float32).copy()
                img = env.render_image()
            else:
                img, state = env.reset(seed=config.training.seed + 50_000 + 100 * task_id + sample_idx)
                completion = np.zeros(agent.n_tasks, dtype=np.float32)
            z = agent.encoder.encode_numpy(img).squeeze()
            steps_taken = 0
            done = False
            while (not done
                   and completion[task_id] < 0.5
                   and steps_taken < config.manager.subgoal_horizon):
                student_input = np.concatenate(
                    [z.astype(np.float32), agent.worker_buf.normalize_proprio(state).astype(np.float32)],
                    axis=0,
                )
                teacher_affordance = agent.affordance_features_from_env(env, state, task_id)
                teacher_action = agent.get_teacher_action_deterministic(
                    state, state, task_id, affordance=teacher_affordance
                ).reshape(-1)
                dataset.add(student_input, teacher_action)

                agent.rollout_policy_source = execution_source
                chunk = agent.get_worker_chunk(
                    z=z, proprio=state, full_state=state, task_id=task_id,
                    deterministic=True,
                    teacher_affordance=teacher_affordance if execution_source == "teacher" else None,
                )
                for h in range(agent.H_chunk):
                    next_img, _, done_env, info = env.step(chunk[h])
                    next_state = np.asarray(info["state"], dtype=np.float64)
                    z = agent.encoder.encode_numpy(next_img).squeeze()
                    raw_completion = agent.spec.completion_mask_from_names(
                        info.get("tasks_completed_names", [])
                    )
                    completion = np.maximum(completion, raw_completion)
                    state = next_state
                    steps_taken += 1
                    done = bool(done_env)
                    if done or completion[task_id] > 0.5 or steps_taken >= config.manager.subgoal_horizon:
                        break
    finally:
        agent.rollout_policy_source = prev_source
        env.close()
    dataset.finalize()
    return dataset


def _build_teacher_supervision_dataset(agent: SpecialistSkillAgent,
                                       config: Config,
                                       task_id: int) -> RolloutSupervisionDataset:
    starts: List[Dict[str, np.ndarray]] = [{"kind": "reset"} for _ in range(config.specialist.teacher_rollout_episodes)]
    prefix_tasks = agent.tasks[:task_id]
    if prefix_tasks and config.specialist.teacher_prefix_states > 0:
        try:
            prefix_samples, _ = sample_oracle_prefix_states(
                agent=agent,
                config=config,
                prefix_tasks=prefix_tasks,
                target_task=agent.tasks[task_id],
                max_states=config.specialist.teacher_prefix_states,
                verbose=False,
            )
            starts.extend(prefix_samples)
        except RuntimeError:
            pass
    return _collect_supervision_from_starts(
        agent=agent,
        config=config,
        task_id=task_id,
        starts=starts,
        execution_source="teacher",
        query_source="teacher",
    )


def _build_dagger_dataset(agent: SpecialistSkillAgent,
                          config: Config,
                          task_id: int) -> RolloutSupervisionDataset:
    starts: List[Dict[str, np.ndarray]] = [{"kind": "reset"} for _ in range(config.specialist.dagger_rollout_episodes)]
    prefix_tasks = agent.tasks[:task_id]
    if prefix_tasks and config.specialist.dagger_prefix_states > 0:
        try:
            prefix_samples, _ = sample_oracle_prefix_states(
                agent=agent,
                config=config,
                prefix_tasks=prefix_tasks,
                target_task=agent.tasks[task_id],
                max_states=config.specialist.dagger_prefix_states,
                verbose=False,
            )
            starts.extend(prefix_samples)
        except RuntimeError:
            pass
    return _collect_supervision_from_starts(
        agent=agent,
        config=config,
        task_id=task_id,
        starts=starts,
        execution_source="student",
        query_source="teacher",
    )


def _online_teacher_starts(agent: SpecialistSkillAgent,
                           config: Config,
                           task_id: int,
                           n_reset: int,
                           n_prefix: int) -> List[Dict[str, np.ndarray]]:
    starts: List[Dict[str, np.ndarray]] = [{"kind": "reset"} for _ in range(n_reset)]
    prefix_tasks = agent.tasks[:task_id]
    if prefix_tasks and n_prefix > 0:
        try:
            prefix_samples, _ = sample_oracle_prefix_states(
                agent=agent,
                config=config,
                prefix_tasks=prefix_tasks,
                target_task=agent.tasks[task_id],
                max_states=n_prefix,
                verbose=False,
            )
            starts.extend(prefix_samples)
        except RuntimeError:
            pass
    return starts


def _collect_teacher_online_dataset(agent: SpecialistSkillAgent,
                                    config: Config,
                                    task_id: int) -> OnlineTeacherDataset:
    dataset = OnlineTeacherDataset()
    starts = _online_teacher_starts(
        agent=agent,
        config=config,
        task_id=task_id,
        n_reset=config.specialist.teacher_online_rollout_episodes,
        n_prefix=config.specialist.teacher_online_prefix_states,
    )
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=False,
    )
    rewarder = PrivilegedSkillReward(agent, config, env, task_id)
    try:
        for sample_idx, sample in enumerate(starts):
            env.reset(seed=config.training.seed + 60_000 + 100 * task_id + sample_idx)
            if "state" in sample and "completion" in sample:
                qpos, qvel = env.observation_to_qpos_qvel(sample["state"])
                env.set_mujoco_state(qpos, qvel)
                env._current_obs = {"observation": np.asarray(sample["state"], dtype=np.float64).copy()}
                env._step_count = 0
                state = np.asarray(sample["state"], dtype=np.float64).copy()
                completion = np.asarray(sample["completion"], dtype=np.float32).copy()
            else:
                _, state = env.reset(seed=config.training.seed + 60_000 + 100 * task_id + sample_idx)
                completion = np.zeros(agent.n_tasks, dtype=np.float32)

            done = False
            steps_taken = 0
            while (not done
                   and completion[task_id] < 0.5
                   and steps_taken < config.manager.subgoal_horizon):
                p = agent.worker_buf.normalize_proprio(state).astype(np.float32)
                task_target = agent.spec.padded_goal_for(task_id)
                task_cur = agent.spec.padded_state_slice_for(state, task_id)
                task_mask = agent.spec.padded_mask_for(task_id)
                approach_before = rewarder.approach_distance()
                affordance = agent.affordance_features_from_env(env, state, task_id)
                action = agent.get_teacher_action_deterministic(
                    state, state, task_id, affordance=affordance
                ).reshape(-1)
                noise_std = float(config.specialist.teacher_online_exploration_std)
                if noise_std > 0.0:
                    action = np.clip(
                        action + np.random.normal(0.0, noise_std, size=action.shape).astype(np.float32),
                        -0.999,
                        0.999,
                    )

                next_img, _, done_env, info = env.step(action.reshape(agent.H_chunk, agent.env_action_dim)[0])
                del next_img
                next_state = np.asarray(info["state"], dtype=np.float64)
                raw_completion_next = agent.spec.completion_mask_from_names(
                    info.get("tasks_completed_names", [])
                )
                completion_next = np.maximum(completion, raw_completion_next)
                approach_after = rewarder.approach_distance()
                affordance_next = agent.affordance_features_from_env(env, next_state, task_id)
                completed = bool(completion_next[task_id] > completion[task_id])
                reward = rewarder.step_reward(
                    state_before=state,
                    state_after=next_state,
                    approach_before=approach_before,
                    approach_after=approach_after,
                    action=action,
                    completed=completed,
                )
                dataset.add({
                    "proprio": p,
                    "task_target": task_target.astype(np.float32),
                    "task_cur": task_cur.astype(np.float32),
                    "task_mask": task_mask.astype(np.float32),
                    "affordance": affordance.astype(np.float32),
                    "action": action.astype(np.float32),
                    "reward": np.asarray(reward, dtype=np.float32),
                    "proprio_next": agent.worker_buf.normalize_proprio(next_state).astype(np.float32),
                    "task_cur_next": agent.spec.padded_state_slice_for(next_state, task_id).astype(np.float32),
                    "affordance_next": affordance_next.astype(np.float32),
                    "done": np.asarray(float(done_env or completed), dtype=np.float32),
                })
                state = next_state
                completion = completion_next
                done = bool(done_env)
                steps_taken += 1
    finally:
        env.close()
    return dataset


def _teacher_online_finetune(agent: SpecialistSkillAgent,
                             config: Config,
                             task_id: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    online_ds = _collect_teacher_online_dataset(agent, config, task_id)
    metrics["teacher_online_samples"] = float(len(online_ds))
    if len(online_ds) > 0:
        rewards = np.asarray([float(row["reward"]) for row in online_ds.rows], dtype=np.float32)
        dones = np.asarray([float(row["done"]) for row in online_ds.rows], dtype=np.float32)
        distances = np.asarray([float(row["affordance"][3]) for row in online_ds.rows], dtype=np.float32)
        contacts = np.asarray([float(row["affordance"][14]) for row in online_ds.rows], dtype=np.float32)
        metrics["teacher_online_reward_mean"] = float(np.mean(rewards))
        metrics["teacher_online_reward_max"] = float(np.max(rewards))
        metrics["teacher_online_done_frac"] = float(np.mean(dones))
        metrics["teacher_online_affordance_dist_mean"] = float(np.mean(distances))
        metrics["teacher_online_affordance_dist_min"] = float(np.min(distances))
        metrics["teacher_online_contact_frac"] = float(np.mean(contacts))
    if len(online_ds) == 0 or config.specialist.n_teacher_online_steps <= 0:
        return metrics
    losses = []
    for _ in range(config.specialist.n_teacher_online_steps):
        batch = online_ds.sample(config.specialist.batch_size)
        if config.specialist.teacher_residual_online:
            losses.append(agent.teacher_residual_iql_step(task_id, batch))
        else:
            losses.append(agent.teacher_iql_step(task_id, batch))
    for key in losses[-1].keys():
        tail = [m[key] for m in (losses[-100:] if len(losses) >= 100 else losses)]
        metrics[f"online_{key}"] = float(np.mean(tail))
    return metrics


def run_specialist_stage_a_warmup(agent: SpecialistSkillAgent,
                                  config: Config,
                                  verbose: bool = True) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if verbose:
        names = ", ".join(config.warmup.dataset_ids)
        print(f"  [Warmup] Building specialist demo dataset from: {names}")

    ds, ds_stats = build_or_load_demo_dataset(agent, config, verbose=verbose)
    results.update(ds_stats)
    agent.demo_dataset = ds

    if ds.n_worker() == 0:
        raise RuntimeError("Stage A found zero worker demo samples for specialist training.")

    agent.worker_buf.observe_proprio_batch(ds.w_p)
    results["proprio_stats_count"] = float(agent.worker_buf.proprio_stats.n)

    if verbose:
        print(f"  [Warmup] Seeding running proprio statistics from {ds.n_worker():,} demo samples.")

    for task_id, task_name in enumerate(agent.tasks):
        safe_task = task_name.lower().replace(" ", "_")
        n_task_samples = len(ds.worker_indices_by_task[task_id])
        if verbose:
            print(f"  [Warmup] Specialist skill '{task_name}': {n_task_samples:,} demo transitions.")

        bc_losses = []
        for _ in range(config.specialist.n_teacher_bc_steps):
            batch = ds.sample_worker_task_batch(
                task_id, config.specialist.batch_size,
                proprio_normalizer=agent.worker_buf.normalize_proprio,
            )
            bc_losses.append(agent.teacher_bc_step(task_id, batch))
        results[f"teacher_bc/{safe_task}_loss_final"] = float(np.mean(bc_losses[-100:])) if bc_losses else 0.0
        results[f"teacher_bc/{safe_task}_loss_best"] = float(np.min(bc_losses)) if bc_losses else 0.0
        if verbose:
            print(f"    teacher BC final: {results[f'teacher_bc/{safe_task}_loss_final']:.4f}  "
                  f"(best {results[f'teacher_bc/{safe_task}_loss_best']:.4f})")

        iql_metrics = []
        for _ in range(config.specialist.n_teacher_iql_steps):
            batch = ds.sample_worker_task_batch(
                task_id, config.specialist.batch_size,
                proprio_normalizer=agent.worker_buf.normalize_proprio,
            )
            iql_metrics.append(agent.teacher_iql_step(task_id, batch))
        if iql_metrics:
            for key in iql_metrics[-1].keys():
                tail = [m[key] for m in (iql_metrics[-100:] if len(iql_metrics) >= 100 else iql_metrics)]
                results[f"{key}/{safe_task}_final"] = float(np.mean(tail))
        if verbose:
            print(f"    teacher IQL final: "
                  f"value={results[f'teacher_iql_value_loss/{safe_task}_final']:.4f}  "
                  f"critic={results[f'teacher_iql_critic_loss/{safe_task}_final']:.4f}  "
                  f"actor={results[f'teacher_iql_actor_loss/{safe_task}_final']:.4f}")

        if config.specialist.teacher_residual_online:
            agent.freeze_teacher_base(task_id)
            if verbose:
                print(f"    teacher residual base: frozen from offline teacher  "
                      f"(scale={config.specialist.teacher_residual_scale:.3f})")

        online_metrics = _teacher_online_finetune(agent, config, task_id)
        for key, value in online_metrics.items():
            results[f"teacher_online/{safe_task}_{key}"] = float(value)
        if verbose:
            print(f"    teacher online samples: {int(online_metrics.get('teacher_online_samples', 0)):,}  "
                  f"reward_mean={online_metrics.get('teacher_online_reward_mean', float('nan')):.4f}  "
                  f"reward_max={online_metrics.get('teacher_online_reward_max', float('nan')):.4f}  "
                  f"done_frac={online_metrics.get('teacher_online_done_frac', float('nan')):.3f}  "
                  f"dist_mean={online_metrics.get('teacher_online_affordance_dist_mean', float('nan')):.4f}  "
                  f"dist_min={online_metrics.get('teacher_online_affordance_dist_min', float('nan')):.4f}  "
                  f"contact_frac={online_metrics.get('teacher_online_contact_frac', float('nan')):.3f}")
            if "online_teacher_iql_actor_loss" in online_metrics:
                print(f"    teacher online IQL final: "
                      f"value={online_metrics.get('online_teacher_iql_value_loss', float('nan')):.4f}  "
                      f"critic={online_metrics.get('online_teacher_iql_critic_loss', float('nan')):.4f}  "
                      f"actor={online_metrics.get('online_teacher_iql_actor_loss', float('nan')):.4f}")
            if "online_teacher_residual_actor_loss" in online_metrics:
                print(f"    teacher residual online final: "
                      f"value={online_metrics.get('online_teacher_residual_value_loss', float('nan')):.4f}  "
                      f"critic={online_metrics.get('online_teacher_residual_critic_loss', float('nan')):.4f}  "
                      f"actor={online_metrics.get('online_teacher_residual_actor_loss', float('nan')):.4f}  "
                      f"action_mse={online_metrics.get('online_teacher_residual_action_mse', float('nan')):.4f}  "
                      f"res_l2={online_metrics.get('online_teacher_residual_l2', float('nan')):.4f}")

        teacher_eval = _evaluate_specialist_skill(
            agent=agent,
            config=config,
            task_id=task_id,
            source="teacher",
            n_episodes=config.specialist.teacher_eval_episodes,
        )
        for key, value in teacher_eval.items():
            results[f"teacher_eval/{safe_task}_{key}"] = float(value)
        if verbose:
            print(f"    teacher eval: success={teacher_eval['success_rate']*100:5.1f}%  "
                  f"mean_options={teacher_eval['mean_options']:.1f}  "
                  f"env_reward={teacher_eval['mean_env_reward']:.3f}")

        teacher_rollout_ds = _build_teacher_supervision_dataset(agent, config, task_id)
        results[f"teacher_rollout/{safe_task}_states"] = float(len(teacher_rollout_ds))
        if verbose:
            print(f"    teacher rollout states: {len(teacher_rollout_ds):,}")

        distill_metrics = []
        for _ in range(config.specialist.n_student_distill_steps):
            batch = ds.sample_worker_task_batch(
                task_id, config.specialist.batch_size,
                proprio_normalizer=agent.worker_buf.normalize_proprio,
            )
            distill_metrics.append(agent.student_distill_step(task_id, batch))
        if distill_metrics:
            for key in distill_metrics[-1].keys():
                tail = [m[key] for m in (distill_metrics[-100:] if len(distill_metrics) >= 100 else distill_metrics)]
                results[f"{key}/{safe_task}_final"] = float(np.mean(tail))
            if verbose:
                print(f"    student distill final: "
                      f"teacher_bc={results[f'student_distill_loss/{safe_task}_final']:.4f}  "
                      f"demo_bc={results[f'student_demo_bc_loss/{safe_task}_final']:.4f}  "
                      f"total={results[f'student_total_loss/{safe_task}_final']:.4f}")

        rollout_distill_metrics = []
        if len(teacher_rollout_ds) > 0 and config.specialist.n_student_rollout_distill_steps > 0:
            for _ in range(config.specialist.n_student_rollout_distill_steps):
                batch = teacher_rollout_ds.sample(config.specialist.batch_size)
                rollout_distill_metrics.append(agent.student_rollout_distill_step(task_id, batch))
            results[f"student_rollout_distill/{safe_task}_loss_final"] = float(
                np.mean([m["student_rollout_distill_loss"] for m in rollout_distill_metrics[-100:]])
            )
            if verbose:
                print(f"    student rollout-distill final: "
                      f"{results[f'student_rollout_distill/{safe_task}_loss_final']:.4f}")

        dagger_ds = _build_dagger_dataset(agent, config, task_id)
        results[f"dagger/{safe_task}_states"] = float(len(dagger_ds))
        if verbose:
            print(f"    dagger correction states: {len(dagger_ds):,}")

        dagger_metrics = []
        if len(dagger_ds) > 0 and config.specialist.n_student_dagger_steps > 0:
            for _ in range(config.specialist.n_student_dagger_steps):
                batch = dagger_ds.sample(config.specialist.batch_size)
                dagger_metrics.append(agent.student_rollout_distill_step(task_id, batch))
            results[f"student_dagger/{safe_task}_loss_final"] = float(
                np.mean([m["student_rollout_distill_loss"] for m in dagger_metrics[-100:]])
            )
            if verbose:
                print(f"    student dagger final: "
                      f"{results[f'student_dagger/{safe_task}_loss_final']:.4f}")

    if verbose:
        print("  [Warmup] Worker labels by task:")
        for task_name in agent.tasks:
            safe_task = task_name.lower().replace(" ", "_")
            print(f"    {task_name:<14} {int(results.get(f'worker_labels/{safe_task}', 0)):,}")
        replay_mean = results.get("replay_mean_state_l2", float("nan"))
        replay_max = results.get("replay_max_state_l2", float("nan"))
        if not np.isnan(replay_mean):
            print(f"  [Warmup] Replay fidelity: mean_state_l2={replay_mean:.6f}  "
                  f"max_state_l2={replay_max:.6f}")

    return results
