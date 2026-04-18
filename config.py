from dataclasses import dataclass, field
from typing import List
from utils import DEFAULT_TASKS, MAX_TASK_GOAL_DIM


@dataclass
class EncoderConfig:
    name: str = 'r3m'
    freeze: bool = True
    raw_dim: int = 2048
    img_size: int = 224


@dataclass
class SemanticConfig:
    use_task_language_embeddings: bool = True
    task_language_dim: int = 32
    use_demo_prototypes: bool = True
    demo_gif_path: str = 'demo/franka_kitchen.gif'
    demo_max_frames: int = 120


@dataclass
class ManagerConfig:
    hidden_dim: int = 384
    n_layers: int = 3
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.01
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500_000
    option_horizon: int = 18
    max_high_level_steps: int = 20
    bootstrap_uniform_steps: int = 20_000
    heuristic_mix_prob: float = 0.70
    update_every_options: int = 1
    reward_completion_bonus: float = 18.0
    reward_selected_error_reduction: float = 10.0
    reward_selected_progress_gain: float = 4.0
    reward_regression_penalty: float = 8.0
    reward_efficiency_penalty: float = 0.15


@dataclass
class WorkerConfig:
    hidden_dim: int = 384
    n_layers: int = 3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.01
    init_alpha: float = 0.2
    min_alpha: float = 0.03
    auto_alpha: bool = True
    proprio_dim: int = 59
    task_goal_dim: int = MAX_TASK_GOAL_DIM
    bootstrap_random_action_steps: int = 15_000
    random_action_prob_start: float = 0.60
    random_action_prob_end: float = 0.05
    random_action_prob_decay_steps: int = 250_000
    reward_error_reduction: float = 8.0
    reward_progress_gain: float = 3.0
    reward_completion_bonus: float = 12.0
    reward_regression_penalty: float = 6.0
    reward_action_penalty: float = 1e-3
    reward_env_weight: float = 1.0
    option_patience: int = 6
    option_min_steps: int = 4
    success_hold_steps: int = 2
    improvement_epsilon: float = 1e-3


@dataclass
class BufferConfig:
    worker_capacity: int = 400_000
    manager_capacity: int = 80_000
    batch_size: int = 256
    z_storage_dtype: str = 'float16'
    start_learning_after: int = 8_000
    worker_updates_per_step: int = 1
    manager_updates_per_option: int = 1


@dataclass
class TrainingConfig:
    total_timesteps: int = 1_200_000
    eval_freq: int = 20_000
    n_eval_episodes: int = 20
    log_dir: str = 'logs/'
    seed: int = 42
    device: str = 'cuda'
    record_video: bool = True
    video_n_episodes: int = 3
    checkpoint_every_n: int = 10
    tasks_to_complete: List[str] = field(default_factory=lambda: DEFAULT_TASKS.copy())
    terminate_on_tasks_completed: bool = False


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if self.encoder.name == 'r3m':
            self.encoder.raw_dim = 2048
        elif self.encoder.name == 'dinov2':
            self.encoder.raw_dim = 384
        else:
            raise ValueError(f'Unknown encoder: {self.encoder.name}')
