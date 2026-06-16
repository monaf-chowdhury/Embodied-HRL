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

    # Per-skill shaped reward weights. Dense progress stays auxiliary to the
    # sparse completion bonus; the chunk return is a discounted sum of these.
    progress_weight: float = 0.5
    completion_bonus: float = 10.0
    action_cost: float = 0.001
    failure_penalty: float = 0.1
    # Potential-based shaping: phi(s) = exp(-(e/eps) / sigma), bounded in (0, 1].
    # sigma=1.0 concentrates gradient near the success boundary (e_norm < 2)
    sigma: float = 1.0


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
    """Per-skill QC-FQL: chunked twin critic + flow BC policy + one-step actor.

    Q-chunking (Li, Zhou, Levine; NeurIPS 2025, arXiv:2507.07969) +
    Flow Q-Learning (Park, Li, Levine; ICML 2025, arXiv:2502.02538).
    """
    hidden_dim: int = 256
    n_layers: int = 3
    batch_size: int = 256
    use_layernorm: bool = True
    dropout: float = 0.0

    # "flow_bc"  -> flow-matching BC only (no critic; staged experiment 1)
    # "qc_fql"   -> full QC-FQL (chunked critic + flow BC + one-step Q actor)
    offline_algo: str = "qc_fql"

    n_flow_bc_steps: int = 30_000      # flow-BC steps (and total for flow_bc mode)
    n_offline_rl_steps: int = 100_000  # QC-FQL joint steps

    # Flow Q-Learning knobs.
    flow_steps: int = 10           # Euler integration steps for the BC flow ODE
    fql_alpha: float = 10.0        # distillation / behavior-constraint coefficient
    fql_normalize_q: bool = True   # scale-invariant Q term (alpha is the real dial)
    best_of_n: int = 1             # >1: sample N one-step actions, pick argmax Q at eval
    target_tau: float = 0.005      # critic target soft-update rate

    # Per-skill prefix-state validation (model selection during training).
    eval_interval: int = 10_000
    eval_prefix_states: int = 100

    log_interval: int = 500


@dataclass
class OnlineConfig:
    """Online QC-FQL fine-tuning: continue the offline update on a growing
    mixed (demo + online) replay, collecting chained rollouts."""
    enabled: bool = False
    total_env_steps: int = 200_000
    eval_interval_steps: int = 25_000
    log_interval_episodes: int = 20
    updates_per_env_step: float = 1.0
    batch_size: int = 256
    online_buffer_capacity_per_skill: int = 100_000

    # Mixed replay: fraction of each batch drawn from the demo dataset.
    demo_fraction: float = 0.5
    # Exploration: stochastic one-step actor (noise ~ N(0, exploration_noise)).
    exploration_noise: float = 1.0
    # Confirmed-eval rollback (eval noise is real; confirm before acting).
    rollback_drop_tolerance: float = 0.12

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
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    specialist: SpecialistConfig = field(default_factory=SpecialistConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self.refresh_encoder_dim()

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
