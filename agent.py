from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from buffers import ManagerReplayBuffer, WorkerReplayBuffer
from config import Config
from encoder import VisualEncoder
from networks import ManagerQNetwork, WorkerActorNetwork, WorkerCriticNetwork
from utils import (
    MAX_TASK_GOAL_DIM,
    all_task_goal_matrix,
    build_demo_task_prototypes,
    build_task_language_embeddings,
    heuristic_task_choice,
    newly_completed_count,
    per_task_errors,
    per_task_progress,
    remaining_mask,
    task_structured_vectors,
    task_transition_completed,
)
# Avoid circular import at function definition time.
from utils import task_success_tolerance as task_success_threshold


class TaskGroundedHRLAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        self.tasks = config.training.tasks_to_complete
        self.n_tasks = len(self.tasks)
        self.z_dim = config.encoder.raw_dim
        self.action_dim = 9

        self.encoder = VisualEncoder(config.encoder, device=self.device)

        task_lang_np = build_task_language_embeddings(self.tasks, dim=config.semantic.task_language_dim)
        task_goals_np = all_task_goal_matrix(self.tasks, max_dim=config.worker.task_goal_dim)
        self.task_lang_np = task_lang_np.astype(np.float32)
        self.task_goals_np = task_goals_np.astype(np.float32)
        self.task_lang = torch.from_numpy(self.task_lang_np).float().to(self.device)
        self.task_goals = torch.from_numpy(self.task_goals_np).float().to(self.device)

        if config.semantic.use_demo_prototypes:
            self.demo_prototypes = build_demo_task_prototypes(
                gif_path=config.semantic.demo_gif_path,
                encoder=self.encoder,
                tasks=self.tasks,
                max_frames=config.semantic.demo_max_frames,
            )
        else:
            self.demo_prototypes = build_demo_task_prototypes('', self.encoder, [], 0)

        self.manager_q = ManagerQNetwork(
            z_dim=self.z_dim,
            proprio_dim=config.worker.proprio_dim,
            n_tasks=self.n_tasks,
            task_lang_dim=config.semantic.task_language_dim,
            task_goal_dim=config.worker.task_goal_dim,
            hidden_dim=config.manager.hidden_dim,
            n_layers=config.manager.n_layers,
        ).to(self.device)
        self.manager_q_target = ManagerQNetwork(
            z_dim=self.z_dim,
            proprio_dim=config.worker.proprio_dim,
            n_tasks=self.n_tasks,
            task_lang_dim=config.semantic.task_language_dim,
            task_goal_dim=config.worker.task_goal_dim,
            hidden_dim=config.manager.hidden_dim,
            n_layers=config.manager.n_layers,
        ).to(self.device)
        self.manager_q_target.load_state_dict(self.manager_q.state_dict())
        self.manager_optimizer = torch.optim.Adam(self.manager_q.parameters(), lr=config.manager.lr)

        self.worker_actor = WorkerActorNetwork(
            z_dim=self.z_dim,
            action_dim=self.action_dim,
            n_tasks=self.n_tasks,
            task_lang_dim=config.semantic.task_language_dim,
            task_goal_dim=config.worker.task_goal_dim,
            proprio_dim=config.worker.proprio_dim,
            hidden_dim=config.worker.hidden_dim,
            n_layers=config.worker.n_layers,
        ).to(self.device)
        self.worker_critic = WorkerCriticNetwork(
            z_dim=self.z_dim,
            action_dim=self.action_dim,
            n_tasks=self.n_tasks,
            task_lang_dim=config.semantic.task_language_dim,
            task_goal_dim=config.worker.task_goal_dim,
            proprio_dim=config.worker.proprio_dim,
            hidden_dim=config.worker.hidden_dim,
            n_layers=config.worker.n_layers,
        ).to(self.device)
        self.worker_critic_target = WorkerCriticNetwork(
            z_dim=self.z_dim,
            action_dim=self.action_dim,
            n_tasks=self.n_tasks,
            task_lang_dim=config.semantic.task_language_dim,
            task_goal_dim=config.worker.task_goal_dim,
            proprio_dim=config.worker.proprio_dim,
            hidden_dim=config.worker.hidden_dim,
            n_layers=config.worker.n_layers,
        ).to(self.device)
        self.worker_critic_target.load_state_dict(self.worker_critic.state_dict())
        self.worker_actor_optimizer = torch.optim.Adam(self.worker_actor.parameters(), lr=config.worker.actor_lr)
        self.worker_critic_optimizer = torch.optim.Adam(self.worker_critic.parameters(), lr=config.worker.critic_lr)

        if config.worker.auto_alpha:
            self.log_alpha = torch.tensor(np.log(config.worker.init_alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.worker.alpha_lr)
            self.target_entropy = -self.action_dim
        else:
            self.log_alpha = torch.tensor(np.log(config.worker.init_alpha), device=self.device)
            self.alpha_optimizer = None
            self.target_entropy = -self.action_dim

        z_dtype = np.float16 if config.buffer.z_storage_dtype == 'float16' else np.float32
        self.manager_buffer = ManagerReplayBuffer(
            capacity=config.buffer.manager_capacity,
            z_dim=self.z_dim,
            proprio_dim=config.worker.proprio_dim,
            n_tasks=self.n_tasks,
            z_dtype=z_dtype,
        )
        self.worker_buffer = WorkerReplayBuffer(
            capacity=config.buffer.worker_capacity,
            z_dim=self.z_dim,
            action_dim=self.action_dim,
            proprio_dim=config.worker.proprio_dim,
            n_tasks=self.n_tasks,
            task_goal_dim=config.worker.task_goal_dim,
            z_dtype=z_dtype,
        )

        self.total_steps = 0
        self.total_episodes = 0
        self.total_options = 0
        self.manager_epsilon = config.manager.epsilon_start

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def prototype_similarities(self, z_current: np.ndarray) -> np.ndarray:
        sims = self.demo_prototypes.similarities(z_current)
        if sims.size == 0:
            return np.zeros(self.n_tasks, dtype=np.float32)
        if len(sims) != self.n_tasks:
            return np.zeros(self.n_tasks, dtype=np.float32)
        return sims.astype(np.float32)

    def build_manager_state(self, proprio: np.ndarray, completion_mask: np.ndarray, prev_task: int, z_current: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        progress = per_task_progress(proprio, self.tasks)
        errors = per_task_errors(proprio, self.tasks)
        remain = remaining_mask(completion_mask)
        sims = np.zeros(self.n_tasks, dtype=np.float32) if z_current is None else self.prototype_similarities(z_current)
        return {
            'progress': progress.astype(np.float32),
            'errors': errors.astype(np.float32),
            'completion': np.asarray(completion_mask, dtype=np.float32),
            'remaining': remain.astype(np.float32),
            'prototype_sims': sims.astype(np.float32),
            'prev_task': int(prev_task),
        }

    def current_worker_random_prob(self) -> float:
        cfg = self.config.worker
        frac = self.total_steps / max(cfg.random_action_prob_decay_steps, 1)
        return float(max(cfg.random_action_prob_end, cfg.random_action_prob_start - (cfg.random_action_prob_start - cfg.random_action_prob_end) * frac))

    def update_manager_epsilon(self):
        cfg = self.config.manager
        frac = self.total_steps / max(cfg.epsilon_decay_steps, 1)
        self.manager_epsilon = float(max(cfg.epsilon_end, cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) * frac))

    def select_task(
        self,
        z_current: np.ndarray,
        proprio: np.ndarray,
        completion_mask: np.ndarray,
        prev_task: int,
        deterministic: bool = False,
    ) -> int:
        state = self.build_manager_state(proprio, completion_mask, prev_task, z_current=z_current)
        remaining = state['remaining']
        valid = np.where(remaining > 0.5)[0]
        if len(valid) == 0:
            return int(np.argmin(state['errors']))

        if deterministic:
            return self._greedy_manager_choice(z_current, proprio, state)

        if self.total_steps < self.config.manager.bootstrap_uniform_steps:
            if np.random.rand() < self.config.manager.heuristic_mix_prob:
                return heuristic_task_choice(state['progress'], state['remaining'], state['prototype_sims'])
            return int(np.random.choice(valid))

        if np.random.rand() < self.manager_epsilon:
            if np.random.rand() < self.config.manager.heuristic_mix_prob:
                return heuristic_task_choice(state['progress'], state['remaining'], state['prototype_sims'])
            return int(np.random.choice(valid))
        return self._greedy_manager_choice(z_current, proprio, state)

    def _greedy_manager_choice(self, z_current: np.ndarray, proprio: np.ndarray, state: Dict[str, np.ndarray]) -> int:
        with torch.no_grad():
            z = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
            p = torch.from_numpy(self.manager_buffer.proprio_norm.normalize(proprio)).float().unsqueeze(0).to(self.device)
            progress = torch.from_numpy(state['progress']).float().unsqueeze(0).to(self.device)
            errors = torch.from_numpy(state['errors']).float().unsqueeze(0).to(self.device)
            completion = torch.from_numpy(state['completion']).float().unsqueeze(0).to(self.device)
            remain = torch.from_numpy(state['remaining']).float().unsqueeze(0).to(self.device)
            sims = torch.from_numpy(state['prototype_sims']).float().unsqueeze(0).to(self.device)
            prev_task = torch.tensor([state['prev_task']], dtype=torch.long, device=self.device)
            q = self.manager_q.evaluate_all_tasks(
                z, p, progress, errors, completion, remain, sims, prev_task, self.task_lang, self.task_goals, valid_mask=remain
            )
            return int(q.argmax(dim=1).item())

    def get_worker_structured_inputs(self, proprio: np.ndarray, task_id: int) -> Dict[str, np.ndarray]:
        task = self.tasks[task_id]
        value, target, error_vec = task_structured_vectors(proprio, task, max_dim=self.config.worker.task_goal_dim)
        progress = per_task_progress(proprio, [task])[0]
        return {
            'value': value.astype(np.float32),
            'target': target.astype(np.float32),
            'error_vec': error_vec.astype(np.float32),
            'progress': np.float32(progress),
        }

    def select_worker_action(
        self,
        z_current: np.ndarray,
        proprio: np.ndarray,
        completion_mask: np.ndarray,
        task_id: int,
        deterministic: bool = False,
        force_random: bool = False,
    ) -> np.ndarray:
        if force_random or self.total_steps < self.config.worker.bootstrap_random_action_steps or (not deterministic and np.random.rand() < self.current_worker_random_prob()):
            return None
        inputs = self.get_worker_structured_inputs(proprio, task_id)
        with torch.no_grad():
            z = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
            p = torch.from_numpy(self.worker_buffer.proprio_norm.normalize(proprio)).float().unsqueeze(0).to(self.device)
            task_lang = self.task_lang[task_id].unsqueeze(0)
            target = torch.from_numpy(inputs['target']).float().unsqueeze(0).to(self.device)
            value = torch.from_numpy(inputs['value']).float().unsqueeze(0).to(self.device)
            error_vec = torch.from_numpy(inputs['error_vec']).float().unsqueeze(0).to(self.device)
            progress = torch.tensor([inputs['progress']], dtype=torch.float32, device=self.device)
            completion = torch.from_numpy(completion_mask).float().unsqueeze(0).to(self.device)
            if deterministic:
                a = self.worker_actor.get_action_deterministic(z, p, task_lang, target, value, error_vec, progress, completion)
            else:
                a, _ = self.worker_actor(z, p, task_lang, target, value, error_vec, progress, completion)
        return a.cpu().numpy().squeeze().astype(np.float32)

    def compute_worker_reward(
        self,
        prev_proprio: np.ndarray,
        next_proprio: np.ndarray,
        task_id: int,
        prev_completion: np.ndarray,
        next_completion: np.ndarray,
        env_reward: float,
        action: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self.config.worker
        task = self.tasks[task_id]
        prev_err = per_task_errors(prev_proprio, [task])[0]
        next_err = per_task_errors(next_proprio, [task])[0]
        prev_prog = per_task_progress(prev_proprio, [task])[0]
        next_prog = per_task_progress(next_proprio, [task])[0]
        delta_err = float(prev_err - next_err)
        delta_prog = float(next_prog - prev_prog)
        selected_completed = 1.0 if task_transition_completed(prev_completion, next_completion, task_id) else 0.0

        regression = 0.0
        if prev_completion is not None:
            completed_ids = np.where(np.asarray(prev_completion) > 0.5)[0]
            for j in completed_ids:
                if j == task_id:
                    continue
                before = per_task_errors(prev_proprio, [self.tasks[j]])[0]
                after = per_task_errors(next_proprio, [self.tasks[j]])[0]
                regression += max(0.0, float(after - before))

        reward = 0.0
        reward += cfg.reward_error_reduction * delta_err
        reward += cfg.reward_progress_gain * delta_prog
        reward += cfg.reward_completion_bonus * selected_completed
        reward -= cfg.reward_regression_penalty * regression
        reward -= cfg.reward_action_penalty * float(np.square(action).mean())
        reward += cfg.reward_env_weight * float(env_reward)

        stats = {
            'delta_err': delta_err,
            'delta_prog': delta_prog,
            'selected_completed': selected_completed,
            'regression': regression,
            'next_err': float(next_err),
        }
        return float(reward), stats

    def compute_manager_reward(
        self,
        start_proprio: np.ndarray,
        end_proprio: np.ndarray,
        task_id: int,
        start_completion: np.ndarray,
        end_completion: np.ndarray,
        option_steps: int,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self.config.manager
        task = self.tasks[task_id]
        start_err = per_task_errors(start_proprio, [task])[0]
        end_err = per_task_errors(end_proprio, [task])[0]
        start_prog = per_task_progress(start_proprio, [task])[0]
        end_prog = per_task_progress(end_proprio, [task])[0]
        completion_gain = newly_completed_count(start_completion, end_completion)
        selected_completed = 1.0 if task_transition_completed(start_completion, end_completion, task_id) else 0.0
        regression = 0.0
        completed_ids = np.where(np.asarray(start_completion) > 0.5)[0]
        for j in completed_ids:
            if j == task_id:
                continue
            before = per_task_errors(start_proprio, [self.tasks[j]])[0]
            after = per_task_errors(end_proprio, [self.tasks[j]])[0]
            regression += max(0.0, float(after - before))

        delta_err = float(start_err - end_err)
        delta_prog = float(end_prog - start_prog)
        reward = 0.0
        reward += cfg.reward_completion_bonus * completion_gain
        reward += cfg.reward_selected_error_reduction * delta_err
        reward += cfg.reward_selected_progress_gain * delta_prog
        reward -= cfg.reward_regression_penalty * regression
        reward -= cfg.reward_efficiency_penalty * (option_steps / max(cfg.option_horizon, 1))

        stats = {
            'delta_err': delta_err,
            'delta_prog': delta_prog,
            'completion_gain': float(completion_gain),
            'selected_completed': selected_completed,
            'regression': regression,
            'end_err': float(end_err),
        }
        return float(reward), stats

    def option_success(self, task_id: int, prev_completion: np.ndarray, next_completion: np.ndarray, task_error: float, hold_count: int) -> bool:
        task_name = self.tasks[task_id]
        if task_transition_completed(prev_completion, next_completion, task_id):
            return True
        return task_error <= task_success_threshold(task_name) and hold_count >= self.config.worker.success_hold_steps

    def update_worker(self) -> Dict[str, float]:
        if len(self.worker_buffer) < self.config.buffer.batch_size:
            return {}
        b = self.worker_buffer.sample(self.config.buffer.batch_size)
        z = torch.from_numpy(b['z']).to(self.device)
        p = torch.from_numpy(b['proprio']).to(self.device)
        task_id = torch.from_numpy(b['task_id']).long().to(self.device)
        task_lang = self.task_lang[task_id]
        target = torch.from_numpy(b['target']).to(self.device)
        value = torch.from_numpy(b['value']).to(self.device)
        error_vec = torch.from_numpy(b['error_vec']).to(self.device)
        progress = torch.from_numpy(b['progress']).to(self.device)
        completion = torch.from_numpy(b['completion']).to(self.device)
        action = torch.from_numpy(b['action']).to(self.device)
        reward = torch.from_numpy(b['reward']).unsqueeze(1).to(self.device)
        next_z = torch.from_numpy(b['next_z']).to(self.device)
        next_p = torch.from_numpy(b['next_proprio']).to(self.device)
        next_value = torch.from_numpy(b['next_value']).to(self.device)
        next_error_vec = torch.from_numpy(b['next_error_vec']).to(self.device)
        next_progress = torch.from_numpy(b['next_progress']).to(self.device)
        next_completion = torch.from_numpy(b['next_completion']).to(self.device)
        done = torch.from_numpy(b['done']).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_logp = self.worker_actor(next_z, next_p, task_lang, target, next_value, next_error_vec, next_progress, next_completion)
            q1n, q2n = self.worker_critic_target(next_z, next_p, task_lang, target, next_value, next_error_vec, next_progress, next_completion, next_action)
            q_next = torch.min(q1n, q2n) - self.alpha * next_logp
            target_q = reward + self.config.worker.gamma * (1 - done) * q_next

        q1, q2 = self.worker_critic(z, p, task_lang, target, value, error_vec, progress, completion, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.worker_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(), 1.0)
        self.worker_critic_optimizer.step()

        new_action, logp = self.worker_actor(z, p, task_lang, target, value, error_vec, progress, completion)
        q1a, q2a = self.worker_critic(z, p, task_lang, target, value, error_vec, progress, completion, new_action)
        actor_loss = (self.alpha * logp - torch.min(q1a, q2a)).mean()
        self.worker_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_actor.parameters(), 1.0)
        self.worker_actor_optimizer.step()

        if self.config.worker.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.log_alpha.data.clamp_(min=np.log(self.config.worker.min_alpha))

        self._soft_update(self.worker_critic, self.worker_critic_target, self.config.worker.tau)
        return {
            'worker_critic_loss': float(critic_loss.item()),
            'worker_actor_loss': float(actor_loss.item()),
            'worker_alpha': float(self.alpha.item()),
        }

    def update_manager(self) -> Dict[str, float]:
        if len(self.manager_buffer) < self.config.buffer.batch_size:
            return {}
        b = self.manager_buffer.sample(self.config.buffer.batch_size)
        z = torch.from_numpy(b['z']).to(self.device)
        p = torch.from_numpy(b['proprio']).to(self.device)
        progress = torch.from_numpy(b['progress']).to(self.device)
        errors = torch.from_numpy(b['errors']).to(self.device)
        completion = torch.from_numpy(b['completion']).to(self.device)
        remain = torch.from_numpy(b['remaining']).to(self.device)
        sims = torch.from_numpy(b['prototype_sims']).to(self.device)
        prev_task = torch.from_numpy(b['prev_task']).long().to(self.device)
        task_id = torch.from_numpy(b['task_id']).long().to(self.device)
        reward = torch.from_numpy(b['reward']).unsqueeze(1).to(self.device)
        next_z = torch.from_numpy(b['next_z']).to(self.device)
        next_p = torch.from_numpy(b['next_proprio']).to(self.device)
        next_progress = torch.from_numpy(b['next_progress']).to(self.device)
        next_errors = torch.from_numpy(b['next_errors']).to(self.device)
        next_completion = torch.from_numpy(b['next_completion']).to(self.device)
        next_remain = torch.from_numpy(b['next_remaining']).to(self.device)
        next_sims = torch.from_numpy(b['next_prototype_sims']).to(self.device)
        done = torch.from_numpy(b['done']).unsqueeze(1).to(self.device)

        q = self.manager_q(z, p, progress, errors, completion, remain, sims, prev_task, self.task_lang, self.task_goals, task_id)
        with torch.no_grad():
            next_prev_task = task_id
            q_next_all = self.manager_q_target.evaluate_all_tasks(
                next_z, next_p, next_progress, next_errors, next_completion, next_remain, next_sims, next_prev_task, self.task_lang, self.task_goals, valid_mask=next_remain,
            )
            has_valid = (next_remain.sum(dim=1, keepdim=True) > 0.5).float()
            q_next = q_next_all.max(dim=1, keepdim=True)[0] * has_valid
            target = reward + self.config.manager.gamma * (1 - done) * q_next

        loss = F.mse_loss(q, target)
        self.manager_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_q.parameters(), 1.0)
        self.manager_optimizer.step()
        self._soft_update(self.manager_q, self.manager_q_target, self.config.manager.tau)
        return {
            'manager_loss': float(loss.item()),
            'manager_q_mean': float(q.mean().item()),
        }

    def _soft_update(self, source, target, tau: float):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)



