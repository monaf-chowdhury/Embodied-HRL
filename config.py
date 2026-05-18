"""Configuration for the lean FrankaKitchen skill-learning branch."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig:
    name: str = "r3m"            # "r3m" or "dinov2"
    freeze: bool = True
    raw_dim: int = 2048
    img_size: int = 224


@dataclass
class ManagerConfig:
    """Option-budget settings. There is no learned manager in this branch."""
    subgoal_horizon: int = 40
    max_high_level_steps: int = 12


@dataclass
class WorkerConfig:
    proprio_dim: int = 59
    action_chunk_len: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99

    # Demo-transition reward for IQL. Dense progress should stay auxiliary.
    progress_weight: float = 0.5
    completion_bonus: float = 10.0
    action_cost: float = 0.001
    failure_penalty: float = 0.1


@dataclass
class BufferConfig:
    # Only used by demo cache metadata and compatibility paths.
    worker_capacity: int = 300_000
    manager_capacity: int = 50_000
    batch_size: int = 256
    z_storage_dtype: str = "float16"


@dataclass
class WarmupConfig:
    dataset_ids: List[str] = field(default_factory=lambda: [
        "franka-complete", "franka-mixed", "franka-partial"
    ])
    dataset_source: str = "auto"      # "auto" | "minari" | "d4rl"
    cache_dir: str = "demo/cache"
    rebuild_cache: bool = False
    max_episodes_per_dataset: int = 0
    render_batch_size: int = 256
    manager_label_stride: int = 4
    min_segment_len: int = 1


@dataclass
class EvalConfig:
    n_eval_episodes: int = 15
    n_single_task_episodes: int = 20


@dataclass
class SpecialistConfig:
    hidden_dim: int = 256
    n_layers: int = 3
    batch_size: int = 256

    # Per-skill visual policy: BC first, optional IQL second.
    n_teacher_bc_steps: int = 30_000
    n_teacher_iql_steps: int = 100_000
    iql_expectile: float = 0.7
    iql_adv_beta: float = 3.0
    iql_max_weight: float = 20.0
    log_interval: int = 500


@dataclass
class TrainingConfig:
    mode: str = "lean_skills"
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "logs/lean_skills"
    tasks_to_complete: List[str] = field(default_factory=lambda: [
        "microwave", "kettle", "light switch", "slide cabinet"
    ])
    deterministic_torch: bool = True
    controller_order_mode: str = "given_order"   # "given_order" | "stage_a_rank"

    prefix_eval_only: bool = False
    prefix_target_task: str = ""
    prefix_condition_tasks: List[str] = field(default_factory=list)
    prefix_eval_n_states: int = 20

    record_video: bool = True
    video_n_episodes: int = 3
    video_fps: int = 15


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    specialist: SpecialistConfig = field(default_factory=SpecialistConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if self.encoder.name == "r3m":
            self.encoder.raw_dim = 2048
        elif self.encoder.name == "dinov2":
            self.encoder.raw_dim = 768
        else:
            raise ValueError(f"Unknown encoder '{self.encoder.name}'")
