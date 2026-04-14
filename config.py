"""
Configuration for Visual HRL on Franka Kitchen.

Summary of changes from previous version:
  1. Projection head REMOVED. z_dim = raw_dim = 2048 (R3M output directly).
  2. SSE removed. No strict termination on subgoal failure.
  3. Reachability filter disabled (reject_threshold=0.0).
  4. Manager reward redesigned: task-completion dominant, task-progress
     secondary, latent-nav tertiary.
  5. Worker reward: task-progress focused on nearest task to subgoal.
  6. subgoal_horizon=20 (more manager transitions per episode).
  7. warmup_future_k=20 matches subgoal_horizon.
  8. hidden_dim=512 to handle 2*2048=4096 manager input.
  9. batch_size=256 (2048-d vectors are memory-heavy).
  10. LandmarkConfig: task-progress-biased candidate pool replaces curriculum.
  11. total_timesteps=2_000_000 (more budget needed).
  12. max_high_steps=14 per episode (replaces strict termination).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig:
    name: str = "r3m"
    freeze: bool = True
    raw_dim: int = 2048     # auto-set; this IS z_dim — no projection
    img_size: int = 224


@dataclass
class ManagerConfig:
    # Input: [z_current, z_landmark] = 2*2048 = 4096 → needs large hidden
    hidden_dim: int = 512
    n_layers: int = 3
    lr: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 800_000
    subgoal_horizon: int = 20       # reduced from 50 — more manager data

    # Manager reward composition (computed in agent.compute_manager_reward):
    #   total_reward = task_completion_bonus  [if n_tasks_completed increased]
    #                + task_progress_bonus * max(0, delta_task_progress)
    #                + env_reward_weight * cumulative_env_reward
    #                + nav_bonus  [if landmark latent reached]
    task_completion_bonus: float = 10.0   # dominant: a subtask was completed
    task_progress_bonus: float = 3.0      # secondary: proprioceptive progress
    env_reward_weight: float = 1.0        # passthrough of sparse env reward
    nav_bonus: float = 0.5               # small: pure landmark navigation

    # Max subgoal attempts per episode (replaces SSE hard termination)
    max_high_steps: int = 14


@dataclass
class WorkerConfig:
    # Input: [z_current(2048), z_subgoal(2048)] + proprio(59) — large hidden needed
    hidden_dim: int = 512
    n_layers: int = 3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    auto_alpha: bool = True
    proprio_dim: int = 59


@dataclass
class ReachabilityConfig:
    hidden_dim: int = 512
    n_layers: int = 3
    lr: float = 1e-3
    batch_size: int = 256
    min_buffer_size: int = 10_000
    update_freq: int = 10
    # DISABLED: filter is degenerate. Set to 0.0 to skip all filtering.
    reject_threshold: float = 0.0
    # Only re-enable after this many steps (effectively never for now)
    enable_after_steps: int = 5_000_000


@dataclass
class LandmarkConfig:
    n_landmarks: int = 200
    update_freq: int = 10          # more frequent updates
    min_observations: int = 500

    # Demo seeding — now in same latent space as replay (both raw R3M)
    use_demo_landmarks: bool = True
    demo_gif_path: str = "demo/franka_kitchen.gif"
    demo_max_frames: int = 150

    # Hindsight injection (task-completion states go directly into pool)
    use_hindsight_landmarks: bool = True
    hindsight_pool_size: int = 1000

    # Task-progress-biased candidate pool
    # Prefer transitions where task progress improved or n_tasks increased
    use_task_progress_bias: bool = True
    task_progress_top_pct: float = 0.3   # top 30% of replay by task_delta

    # Curriculum disabled — replaced by task-progress bias above
    use_curriculum_landmarks: bool = False

    # Recent replay bias
    recent_replay_fraction: float = 0.7

    explore_ratio: float = 0.2


@dataclass
class RewardConfig:
    """
    Worker shaped reward:
      r = sparse_weight * r_sparse
          + task_progress_weight * delta_task_progress_focused
          + latent_weight * normalised_delta_latent

    'focused' = computed only for the SINGLE task whose proprio-goal is
    nearest to the current subgoal landmark in proprio space.
    This avoids diluting the gradient across all 4 tasks simultaneously.
    """
    sparse_weight: float = 5.0
    task_progress_weight: float = 2.0   # stronger than before
    latent_weight: float = 0.05         # very small — navigation hint only


@dataclass
class BufferConfig:
    capacity: int = 1_000_000
    batch_size: int = 256   # smaller — 2048-d vectors are memory-heavy


@dataclass
class TrainingConfig:
    total_timesteps: int = 2_000_000   # more budget
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    log_dir: str = "logs/"
    seed: int = 42
    device: str = "cuda"
    warmup_future_k: int = 20          # matches subgoal_horizon
    n_warmup: int = 200

    record_video: bool = True
    video_n_episodes: int = 3

    tasks_to_complete: List[str] = field(default_factory=lambda: [
        'microwave', 'kettle', 'light switch', 'slide cabinet'
    ])


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if self.encoder.name == "r3m":
            self.encoder.raw_dim = 2048
        elif self.encoder.name == "dinov2":
            self.encoder.raw_dim = 384
