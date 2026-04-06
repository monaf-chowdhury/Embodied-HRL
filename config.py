"""
Configuration for Visual HRL on Franka Kitchen.
All hyperparameters in one place for easy tuning.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EncoderConfig:
    name: str = "r3m"  # "r3m" or "dinov2"
    freeze: bool = True
    # R3M outputs 2048-d; DINOv2-ViT-S outputs 384-d
    raw_dim: int = 2048  # set automatically based on name
    # Projection head output dimension (the actual subgoal space)
    proj_dim: int = 64
    proj_hidden: int = 256
    # Image preprocessing
    img_size: int = 224  # R3M expects 224x224
    

@dataclass
class ManagerConfig:
    """High-level policy: selects landmarks as subgoals."""
    hidden_dim: int = 256
    n_layers: int = 3
    lr: float = 3e-4
    gamma: float = 0.4  # Low gamma like SSE — sharpens credit assignment
    tau: float = 0.005  # Target network EMA
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    # Subgoal budget: how many low-level steps per subgoal
    subgoal_horizon: int = 20  # K in the paper
    

@dataclass
class WorkerConfig:
    """Low-level policy: goal-conditioned SAC."""
    hidden_dim: int = 256
    n_layers: int = 3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4  # SAC entropy temperature
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2  # Initial entropy coefficient
    auto_alpha: bool = True  # Automatically tune alpha


@dataclass
class ReachabilityConfig:
    """Learned reachability predictor f(z_curr, z_subgoal) -> [0,1]."""
    hidden_dim: int = 256
    n_layers: int = 3
    lr: float = 1e-3
    # Thresholds for labeling (set after measuring latent scale)
    # These are relative to the latent space scale — calibrate in first 10K steps
    success_threshold: float = 0.0  # auto-calibrated
    failure_threshold: float = 0.0  # auto-calibrated
    auto_calibrate: bool = True
    # Training
    batch_size: int = 256
    min_buffer_size: int = 1000  # Don't train until this many transitions
    update_freq: int = 5  # Update every N episodes
    # Filtering
    reject_threshold: float = 0.3  # Reject subgoals with f < this


@dataclass
class LandmarkConfig:
    """FPS landmark buffer for grounded subgoal selection."""
    n_landmarks: int = 100  # Number of landmarks to maintain
    update_freq: int = 20  # Re-compute landmarks every N episodes
    min_observations: int = 500  # Min replay size before computing landmarks
    # Exploration: fraction of time to select least-visited landmark
    explore_ratio: float = 0.2


@dataclass
class RewardConfig:
    """Dense reward shaping for the worker."""
    # L2 delta-progress in latent space (potential-based shaping)
    shaping_weight: float = 1.0  # alpha in: r = r_sparse + alpha * delta_progress
    sparse_success_reward: float = 1.0


@dataclass 
class BufferConfig:
    capacity: int = 1_000_000
    batch_size: int = 1024


@dataclass
class TrainingConfig:
    total_timesteps: int = 1_000_000
    # How often to do things
    manager_update_freq: int = 1  # Update manager every N high-level steps
    worker_updates_per_step: int = 1  # Gradient steps per env step
    eval_freq: int = 10_000  # Evaluate every N timesteps
    n_eval_episodes: int = 10
    # Logging
    log_dir: str = "logs/"
    save_freq: int = 100_000  # Save checkpoint every N steps
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"

    # Tasks to complete — configurable for curriculum / ablations
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
        # Auto-set encoder raw dim
        if self.encoder.name == "r3m":
            self.encoder.raw_dim = 2048
        elif self.encoder.name == "dinov2":
            self.encoder.raw_dim = 384
