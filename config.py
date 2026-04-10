"""
Configuration for Visual HRL on Franka Kitchen.
All hyperparameters in one place for easy tuning.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EncoderConfig:
    name: str = "r3m"          # "r3m" or "dinov2"
    freeze: bool = True
    raw_dim: int = 2048         # R3M=2048, DINOv2-ViT-S=384 (auto-set below)
    proj_dim: int = 64          # Projection head output (subgoal space)
    proj_hidden: int = 256
    img_size: int = 224         # R3M expects 224x224


@dataclass
class ManagerConfig:
    """High-level policy: selects landmarks as subgoals (image latent only)."""
    hidden_dim: int = 256
    n_layers: int = 3
    lr: float = 3e-4
    gamma: float = 0.4          # Low gamma (SSE): sharpens credit assignment
    tau: float = 0.005
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    # Subgoal budget
    subgoal_horizon: int = 20   # K low-level steps per subgoal


@dataclass
class WorkerConfig:
    """
    Low-level policy: goal-conditioned SAC.
    Input: image latent (64d) + proprio (59d) + subgoal latent (64d) = 187d
    """
    hidden_dim: int = 256
    n_layers: int = 3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.5
    auto_alpha: bool = True
    # Proprio normalisation — running stats tracked during warmup
    proprio_dim: int = 59


@dataclass
class ReachabilityConfig:
    hidden_dim: int = 256
    n_layers: int = 3
    lr: float = 1e-3
    success_threshold: float = 0.0   # auto-calibrated
    failure_threshold: float = 0.0   # auto-calibrated
    auto_calibrate: bool = True
    batch_size: int = 256
    min_buffer_size: int = 5000
    update_freq: int = 5
    reject_threshold: float = 0.3


@dataclass
class LandmarkConfig:
    """FPS landmark buffer with demo seeding and quality improvements."""
    n_landmarks: int = 100
    update_freq: int = 20
    min_observations: int = 500

    # ---- Demo landmark seeding ----
    use_demo_landmarks: bool = True         # ablation flag
    demo_gif_path: str = "demo/franka_kitchen/demo.gif"
    demo_landmark_ratio: float = 0.3        # fraction of landmarks from demo
    # How many frames to subsample from the GIF (None = all)
    demo_max_frames: int = 60

    # ---- Hindsight landmark injection ----
    use_hindsight_landmarks: bool = True    # add accidental task-completion states
    hindsight_pool_size: int = 500          # max latents in the success pool

    # ---- Curriculum / approach states ----
    use_curriculum_landmarks: bool = True   # prioritise approach states
    curriculum_top_k: float = 0.3          # fraction of replay to consider

    # ---- Recent replay bias ----
    recent_replay_fraction: float = 0.7    # fraction from recent half of buffer

    # Exploration
    explore_ratio: float = 0.2


@dataclass
class RewardConfig:
    """
    Reward shaping for the worker.
    r = sparse_weight * r_sparse
        + task_progress_weight * task_progress
        + latent_weight * normalised_latent_progress
    """
    sparse_weight: float = 5.0           # dominant signal
    task_progress_weight: float = 0.5    # proprioceptive task-progress (secondary)
    latent_weight: float = 0.1           # normalised delta-progress (tertiary)
    sparse_success_reward: float = 1.0


@dataclass
class BufferConfig:
    capacity: int = 1_000_000
    batch_size: int = 1024


@dataclass
class TrainingConfig:
    total_timesteps: int = 1_000_000
    manager_update_freq: int = 1
    worker_updates_per_step: int = 1
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    log_dir: str = "logs/"
    save_freq: int = 100_000
    seed: int = 42
    device: str = "cuda"
    warmup_future_k: int = 10          # K-step lookahead during warmup

    # Episode video recording at every checkpoint
    record_video: bool = True
    video_n_episodes: int = 3          # episodes to record per checkpoint

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