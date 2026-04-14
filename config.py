from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig:
    name: str = "r3m"
    freeze: bool = True
    raw_dim: int = 2048
    img_size: int = 224
    use_raw_backbone_features: bool = True


@dataclass
class ManagerConfig:
    hidden_dim: int = 512
    n_layers: int = 3
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_steps: int = 400_000
    subgoal_horizon: int = 40
    max_high_level_steps: int = 8
    # High-level reward shaping
    completion_bonus: float = 25.0
    selected_task_progress_weight: float = 8.0
    any_task_progress_weight: float = 3.0
    env_reward_weight: float = 5.0
    reach_bonus: float = 0.5
    latent_progress_weight: float = 0.25


@dataclass
class WorkerConfig:
    hidden_dim: int = 384
    n_layers: int = 3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.1
    min_alpha: float = 0.03
    auto_alpha: bool = True
    proprio_dim: int = 59


@dataclass
class LandmarkConfig:
    n_landmarks: int = 128
    update_freq: int = 10
    min_observations: int = 1_000

    use_demo_landmarks: bool = True
    demo_gif_path: str = "demo/franka_kitchen.gif"
    demo_max_frames: int = 120
    demo_fraction: float = 0.25

    use_hindsight_landmarks: bool = True
    hindsight_pool_size: int = 1000

    recent_replay_fraction: float = 0.6
    priority_fraction: float = 0.5
    task_delta_percentile: float = 85.0
    explore_ratio: float = 0.15


@dataclass
class RewardConfig:
    sparse_weight: float = 5.0
    selected_task_progress_weight: float = 4.0
    any_task_progress_weight: float = 1.0
    latent_weight: float = 0.05
    completion_bonus: float = 10.0


@dataclass
class BufferConfig:
    capacity: int = 300_000
    high_capacity: int = 50_000
    batch_size: int = 512
    z_storage_dtype: str = "float16"


@dataclass
class TrainingConfig:
    total_timesteps: int = 1_000_000
    worker_updates_per_step: int = 1
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    log_dir: str = "logs/"
    seed: int = 42
    device: str = "cuda"
    warmup_future_k: int = 40
    n_warmup: int = 120
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
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if self.encoder.name == "r3m":
            self.encoder.raw_dim = 2048
        elif self.encoder.name == "dinov2":
            self.encoder.raw_dim = 384
