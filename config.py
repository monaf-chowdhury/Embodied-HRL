"""Configuration for the lean FrankaKitchen skill-learning branch."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig:
    name: str = "dinov2"         # "r3m", "dinov2", or "dinov3"
    freeze: bool = True
    raw_dim: int = 2048
    img_size: int = 224
    dinov3_model: str = "dinov3_vits16plus"
    dinov3_weights: str = "dinov3/dinov3_weights/dinov3_vits16plus_pretrain_lvd1689m.pth"     # Relative to project root, absolute, or set DINOV3_WEIGHTS env var.
    dinov3_repo_or_dir: str = "dinov3"
    dinov3_source: str = "local"  # "github" or "local"


@dataclass
class ManagerConfig:
    """Option-budget settings. There is no learned manager in this branch."""
    subgoal_horizon: int = 40
    max_high_level_steps: int = 12


@dataclass
class WorkerConfig:
    proprio_dim: int = 59
    action_chunk_len: int = 4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99

    # Demo-transition reward for IQL. Dense progress should stay auxiliary.
    progress_weight: float = 0.5
    completion_bonus: float = 10.0
    action_cost: float = 0.001
    failure_penalty: float = 0.1
    # Potential-based shaping: phi(s) = exp(-(e/eps) / sigma), bounded in (0, 1].
    # sigma=1.0 concentrates gradient near the success boundary (e_norm < 2)
    sigma: float = 1.0


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
    n_eval_episodes: int = 100
    n_single_task_episodes: int = 100
    prefix_sample_seed: int = 30_000
    chain_context_eval: bool = True
    chain_context_eval_states: int = 100
    # Final reported eval: repeated runs on episode seeds DISJOINT from the
    # online model-selection stream (99_999+ep), reported as mean +/- std.
    # Chain evals of an identical policy vary by ~4-5pp (renderer/GPU
    # nondeterminism compounds chaotically over long rollouts), so a single
    # eval is a noisy draw and max-over-evals is upward-biased.
    final_eval_repeats: int = 3
    final_eval_seed_base: int = 200_000


@dataclass
class SpecialistConfig:
    hidden_dim: int = 256
    n_layers: int = 3
    batch_size: int = 256
    offline_algo: str = "bc_iql"  # "bc" | "bc_iql" | "td3bc" | "awr" | "bet"

    # Per-skill visual policy: BC first, optional IQL second.
    n_teacher_bc_steps: int = 30_000
    n_offline_rl_steps: int = 100_000
    n_teacher_iql_steps: int = 100_000  # Backward-compatible alias.
    iql_expectile: float = 0.7
    iql_adv_beta: float = 2.0
    iql_max_weight: float = 20.0
    iql_use_value_target: bool = True
    iql_value_target_tau: float = 0.005
    iql_normalize_advantage: bool = False
    iql_eval_interval: int = 10_000
    iql_eval_prefix_states: int = 100

    # LQL lower-bound critic penalty (arXiv 2605.05812, adapted to IQL).
    # Q(s_k,a_k) >= G_{k:l} + gamma^(l-k) * V_target(s_l) over same-task
    # chunk-aligned demo chains; violations are penalised with hinge^2.
    # One chain step == one action chunk (H env steps), matching the 1-step
    # TD semantics where gamma is applied per chunk, not per env step.
    lql_enabled: bool = True
    lql_lambda_lb: float = 0.33  # Codex set it to 0.5; For safety, I set it 1/3 i.e. 0.33 
    lql_n_transitions: int = 8   # max chunk transitions per sampled chain
    lql_min_gap: int = 2         # only pairs spanning >= this many chunks (1-step is TD's job)

    # TD3+BC.
    td3bc_alpha: float = 2.5
    td3bc_tau: float = 0.005
    td3bc_policy_noise: float = 0.2
    td3bc_noise_clip: float = 0.5
    td3bc_policy_freq: int = 2

    # AWR.
    awr_temperature: float = 1.0
    awr_max_weight: float = 20.0

    # BeT-style discretized action-chunk BC.
    bet_steps: int = 80_000
    bet_num_bins: int = 64
    bet_offset_weight: float = 5.0

    log_interval: int = 500


@dataclass
class OnlineConfig:
    enabled: bool = False
    mode: str = "skill_repair"  # "skill_repair" | "chain"
    total_env_steps: int = 200_000
    eval_interval_steps: int = 25_000
    log_interval_episodes: int = 20
    updates_per_env_step: float = 0.75
    batch_size: int = 256
    online_buffer_capacity_per_skill: int = 75_000

    # Mixed replay: keep demo-heavy by default to prevent online drift.
    demo_fraction_start: float = 0.85
    demo_fraction_end: float = 0.75
    demo_fraction_decay_steps: int = 50_000

    # Conservative AWAC-style update.
    awac_temperature: float = 1.0
    awac_max_weight: float = 20.0
    bc_anchor_weight: float = 10.0
    bc_anchor_weight_end: float = 5.0
    bc_anchor_decay_steps: int = 50_000
    critic_target_tau: float = 0.005
    critic_huber_loss: bool = True
    critic_huber_delta: float = 10.0
    normalize_advantage: bool = False
    actor_success_only: bool = True
    actor_include_high_return_failures: bool = False
    collect_next_on_success: bool = True
    min_actor_success_samples: int = 64
    freeze_success_threshold: float = 0.85
    next_skill_collection_threshold: float = 0.60
    rollback_drop_tolerance: float = 0.12

    # Chain-context collection.
    exploration_noise: float = 0.05
    failure_priority: float = 2.0

    load_checkpoint: str = ""
    skip_offline_training: bool = False


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
    online: OnlineConfig = field(default_factory=OnlineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self.refresh_encoder_dim()
        self.specialist.n_teacher_iql_steps = self.specialist.n_offline_rl_steps

    def refresh_encoder_dim(self):
        if self.encoder.name == "r3m":
            self.encoder.raw_dim = 2048
        elif self.encoder.name == "dinov2":
            self.encoder.raw_dim = 384
        elif self.encoder.name == "dinov3":
            self.encoder.raw_dim = _dinov3_output_dim(self.encoder.dinov3_model)
        else:
            raise ValueError(f"Unknown encoder '{self.encoder.name}'")


def _dinov3_output_dim(model_name: str) -> int:
    model = model_name.lower()
    if "vitb" in model:
        return 768
    if "vitl" in model:
        return 1024
    if "vith" in model:
        return 1280
    if "vit7b" in model:
        return 4096
    if "convnext_tiny" in model:
        return 768
    if "convnext_small" in model:
        return 768
    if "convnext_base" in model:
        return 1024
    if "convnext_large" in model:
        return 1536
    return 384
