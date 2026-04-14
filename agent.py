import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from encoder import VisualEncoder
from landmarks import LandmarkBuffer
from networks import ManagerQNetwork, SACActorNetwork, SACCriticNetwork
from buffers import HighLevelBuffer, LowLevelBuffer

_TASK_OBS_IDX = {
    'bottom burner': (np.array([18, 19]),   np.array([-0.88, -0.01])),
    'top burner':    (np.array([24, 25]),   np.array([-0.92, -0.01])),
    'light switch':  (np.array([26, 27]),   np.array([-0.69, -0.05])),
    'slide cabinet': (np.array([28]),       np.array([0.37])),
    'hinge cabinet': (np.array([29, 30]),   np.array([0., 1.45])),
    'microwave':     (np.array([31]),       np.array([-0.75])),
    'kettle':        (np.array([32, 33, 34, 35, 36, 37, 38]),
                      np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06])),
}


def per_task_progress(proprio: np.ndarray, tasks: list[str]) -> np.ndarray:
    vals = []
    for task in tasks:
        idx, goal = _TASK_OBS_IDX[task]
        cur = proprio[idx]
        dist = np.linalg.norm(cur - goal)
        norm = np.linalg.norm(goal) + 1e-4
        vals.append(max(0.0, 1.0 - dist / norm))
    return np.asarray(vals, dtype=np.float32)


class VisualHRLAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.training.device
        self.tasks = config.training.tasks_to_complete
        self.n_tasks = len(self.tasks)
        z_dim = config.encoder.raw_dim
        action_dim = 9

        self.encoder = VisualEncoder(config.encoder, device=self.device)
        self.landmarks = LandmarkBuffer(
            n_landmarks=config.landmarks.n_landmarks,
            z_dim=z_dim,
            landmark_config=config.landmarks,
            task_names=self.tasks,
        )

        self.manager_q = ManagerQNetwork(z_dim, self.n_tasks, config.manager.hidden_dim, config.manager.n_layers).to(self.device)
        self.manager_q_target = ManagerQNetwork(z_dim, self.n_tasks, config.manager.hidden_dim, config.manager.n_layers).to(self.device)
        self.manager_q_target.load_state_dict(self.manager_q.state_dict())
        self.manager_optimizer = torch.optim.Adam(self.manager_q.parameters(), lr=config.manager.lr)

        self.worker_actor = SACActorNetwork(z_dim, action_dim, self.n_tasks, config.worker.hidden_dim, config.worker.n_layers, config.worker.proprio_dim).to(self.device)
        self.worker_critic = SACCriticNetwork(z_dim, action_dim, self.n_tasks, config.worker.hidden_dim, config.worker.n_layers, config.worker.proprio_dim).to(self.device)
        self.worker_critic_target = SACCriticNetwork(z_dim, action_dim, self.n_tasks, config.worker.hidden_dim, config.worker.n_layers, config.worker.proprio_dim).to(self.device)
        self.worker_critic_target.load_state_dict(self.worker_critic.state_dict())
        self.worker_actor_optimizer = torch.optim.Adam(self.worker_actor.parameters(), lr=config.worker.actor_lr)
        self.worker_critic_optimizer = torch.optim.Adam(self.worker_critic.parameters(), lr=config.worker.critic_lr)

        if config.worker.auto_alpha:
            self.log_alpha = torch.tensor(np.log(config.worker.init_alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.worker.alpha_lr)
            self.target_entropy = -action_dim
        else:
            self.log_alpha = torch.tensor(np.log(config.worker.init_alpha), device=self.device)

        z_dtype = np.float16 if config.buffer.z_storage_dtype == 'float16' else np.float32
        self.high_buffer = HighLevelBuffer(config.buffer.high_capacity, z_dim=z_dim, n_tasks=self.n_tasks, z_dtype=z_dtype)
        self.low_buffer = LowLevelBuffer(config.buffer.capacity, z_dim=z_dim, action_dim=action_dim,
                                         proprio_dim=config.worker.proprio_dim, n_tasks=self.n_tasks, z_dtype=z_dtype)

        self.total_steps = 0
        self.total_episodes = 0
        self.epsilon = config.manager.epsilon_start
        self._latent_dists = []
        self.success_threshold = 10.0
        self._prev_n_tasks = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def task_onehot(self, task_ids: np.ndarray | torch.Tensor):
        if isinstance(task_ids, np.ndarray):
            ids = torch.from_numpy(task_ids).long().to(self.device)
        elif isinstance(task_ids, torch.Tensor):
            ids = task_ids.long().to(self.device)
        else:
            ids = torch.tensor(task_ids, dtype=torch.long, device=self.device)
        return F.one_hot(ids, num_classes=self.n_tasks).float()

    def choose_task_from_state(self, proprio: np.ndarray, incomplete_only: bool = False, completed_tasks=None) -> int:
        candidate_tasks = self.tasks
        if incomplete_only and completed_tasks is not None:
            candidate_tasks = [t for t in self.tasks if t not in completed_tasks] or self.tasks
        progresses = per_task_progress(proprio, candidate_tasks)
        chosen_name = candidate_tasks[int(np.argmin(progresses))]
        return self.tasks.index(chosen_name)

    def select_subgoal(self, z_current: np.ndarray) -> int:
        if not self.landmarks.is_ready:
            return 0
        if np.random.random() < self.config.landmarks.explore_ratio:
            return self.landmarks.select_explore()
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.landmarks.n_active)
        with torch.no_grad():
            zc = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
            lms = torch.from_numpy(self.landmarks.get_all()).float().to(self.device)
            task_oh = self.task_onehot(self.landmarks.get_all_task_ids())
            q = self.manager_q.evaluate_all_landmarks(zc, lms, task_oh)
            return int(q.argmax(dim=1).item())

    def _update_epsilon(self):
        cfg = self.config.manager
        frac = self.total_steps / max(cfg.epsilon_decay_steps, 1)
        self.epsilon = max(cfg.epsilon_end, cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) * frac)

    def get_worker_action(self, z_current: np.ndarray, z_subgoal: np.ndarray, proprio: np.ndarray, task_id: int, deterministic: bool = False) -> np.ndarray:
        zc = torch.from_numpy(z_current).float().unsqueeze(0).to(self.device)
        zs = torch.from_numpy(z_subgoal).float().unsqueeze(0).to(self.device)
        p = torch.from_numpy(self.low_buffer.normalise_proprio(proprio)).float().unsqueeze(0).to(self.device)
        t = self.task_onehot(np.array([task_id], dtype=np.int64))
        with torch.no_grad():
            if deterministic:
                a = self.worker_actor.get_action_deterministic(zc, zs, p, t)
            else:
                a, _ = self.worker_actor(zc, zs, p, t)
        return a.cpu().numpy().squeeze()

    def maybe_inject_hindsight(self, z_current: np.ndarray, n_tasks_completed: int, task_id: int):
        if self.config.landmarks.use_hindsight_landmarks and n_tasks_completed > self._prev_n_tasks:
            self.landmarks.add_success_state(z_current, task_id)
        self._prev_n_tasks = n_tasks_completed

    def compute_worker_reward(self, z_t: np.ndarray, z_next: np.ndarray, z_subgoal: np.ndarray,
                              sparse_reward: float, task_deltas: np.ndarray, task_id: int,
                              task_completed_delta: int, initial_dist: float | None = None) -> float:
        cfg = self.config.reward
        r = cfg.sparse_weight * sparse_reward
        r += cfg.selected_task_progress_weight * float(task_deltas[task_id])
        r += cfg.any_task_progress_weight * float(np.maximum(task_deltas, 0.0).sum())
        r += cfg.completion_bonus * float(task_completed_delta)
        dist_before = np.linalg.norm(z_t - z_subgoal)
        dist_after = np.linalg.norm(z_next - z_subgoal)
        delta_lat = dist_before - dist_after
        denom = max(initial_dist if initial_dist is not None else dist_before, dist_before, 1e-6)
        r += cfg.latent_weight * max(delta_lat / denom, -1.0)
        return float(r)

    def compute_manager_reward(self, start_progress: np.ndarray, end_progress: np.ndarray,
                               selected_task_id: int, tasks_completed_delta: int,
                               cumulative_env_reward: float, subgoal_reached: bool,
                               start_dist: float, end_dist: float) -> float:
        cfg = self.config.manager
        delta = end_progress - start_progress
        latent_progress = (start_dist - end_dist) / max(start_dist, 1e-6)
        reward = 0.0
        reward += cfg.completion_bonus * float(tasks_completed_delta)
        reward += cfg.selected_task_progress_weight * float(delta[selected_task_id])
        reward += cfg.any_task_progress_weight * float(np.maximum(delta, 0.0).sum())
        reward += cfg.env_reward_weight * float(cumulative_env_reward)
        reward += cfg.latent_progress_weight * float(max(latent_progress, -1.0))
        if subgoal_reached:
            reward += cfg.reach_bonus
        return float(reward)

    def update_manager(self) -> dict:
        if len(self.high_buffer) < self.config.buffer.batch_size:
            return {}
        b = self.high_buffer.sample(self.config.buffer.batch_size)
        zc = torch.from_numpy(b['z_current']).to(self.device)
        zs = torch.from_numpy(b['z_subgoal']).to(self.device)
        r = torch.from_numpy(b['reward']).unsqueeze(1).to(self.device)
        zn = torch.from_numpy(b['z_next']).to(self.device)
        d = torch.from_numpy(b['done']).unsqueeze(1).to(self.device)
        t = self.task_onehot(b['task_id'])
        q = self.manager_q(zc, zs, t)
        with torch.no_grad():
            lms = torch.from_numpy(self.landmarks.get_all()).float().to(self.device)
            task_oh = self.task_onehot(self.landmarks.get_all_task_ids())
            q_next = self.manager_q_target.evaluate_all_landmarks(zn, lms, task_oh).max(dim=1, keepdim=True)[0]
            target = r + self.config.manager.gamma * (1 - d) * q_next
        loss = F.mse_loss(q, target)
        self.manager_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_q.parameters(), 1.0)
        self.manager_optimizer.step()
        self._soft_update(self.manager_q, self.manager_q_target, self.config.manager.tau)
        return {'manager_loss': float(loss.item()), 'manager_q_mean': float(q.mean().item())}

    def update_worker(self) -> dict:
        if len(self.low_buffer) < self.config.buffer.batch_size:
            return {}
        b = self.low_buffer.sample(self.config.buffer.batch_size)
        zc = torch.from_numpy(b['z_current']).to(self.device)
        p = torch.from_numpy(b['proprio']).to(self.device)
        zs = torch.from_numpy(b['z_subgoal']).to(self.device)
        t = self.task_onehot(b['task_id'])
        a = torch.from_numpy(b['action']).to(self.device)
        r = torch.from_numpy(b['reward']).unsqueeze(1).to(self.device)
        zn = torch.from_numpy(b['z_next']).to(self.device)
        pn = torch.from_numpy(b['proprio_next']).to(self.device)
        d = torch.from_numpy(b['done']).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_a, next_logp = self.worker_actor(zn, zs, pn, t)
            q1n, q2n = self.worker_critic_target(zn, zs, pn, t, next_a)
            qn = torch.min(q1n, q2n) - self.alpha * next_logp
            target_q = r + self.config.worker.gamma * (1 - d) * qn

        q1, q2 = self.worker_critic(zc, zs, p, t, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.worker_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(), 1.0)
        self.worker_critic_optimizer.step()

        new_a, logp = self.worker_actor(zc, zs, p, t)
        q1a, q2a = self.worker_critic(zc, zs, p, t, new_a)
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

    def _soft_update(self, source, target, tau):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    def calibrate_success_threshold(self):
        if not self.landmarks.is_ready:
            return
        lm = self.landmarks.get_all()
        if len(lm) < 2:
            return
        pairwise = np.linalg.norm(lm[:, None, :] - lm[None, :, :], axis=-1)
        np.fill_diagonal(pairwise, np.inf)
        nn_dists = pairwise.min(axis=1)
        step_mean = float(np.mean(self._latent_dists)) if self._latent_dists else 0.0
        target = np.percentile(nn_dists, 10) * 0.35
        lower = max(step_mean * 1.5, 1e-3)
        upper = np.percentile(nn_dists, 50) * 0.75
        self.success_threshold = float(np.clip(target, lower, upper))
        print(f"Calibrated success threshold: {self.success_threshold:.4f}")
        print(f"  NN landmark dist — p10={np.percentile(nn_dists, 10):.4f}  median={np.percentile(nn_dists, 50):.4f}  mean={nn_dists.mean():.4f}")
        if self._latent_dists:
            sd = np.asarray(self._latent_dists)
            print(f"  Step dist        — mean={sd.mean():.4f}  std={sd.std():.4f}  min={sd.min():.4f}  max={sd.max():.4f}")
            if not (1.0 <= sd.mean() <= 20.0):
                print("  [Check] Raw R3M step distance mean is outside the requested 1-20 range. Calibration still adapts, but inspect videos and threshold logs.")
