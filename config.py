"""
config.py — Configuration for SMGW (Semantic Manager, Grounded Worker).

Design principles that drive these defaults:

1. The manager's action space is a 4-way discrete choice over the
   tasks_to_complete list. Its value head is NOT a Q-over-landmarks;
   it is a Q-over-tasks that gets MASKED by the completion bits so
   finished tasks cannot be re-selected.

2. The worker's "subgoal" is the benchmark's own task-space goal slice, 
   NOT an image latent. The image latent remains
   an INPUT channel for both manager and worker, but it is never a
   termination target.

3. Option termination ties to: task completion bit flipped, or
   task-space error < epsilon_k, or K_max steps elapsed. No latent
   distance anywhere.

4. Action chunks (H_chunk > 1) are an optional ablation toggle on
   the worker, following the π0/OpenVLA decomposition idea without
   the scale. Set worker.action_chunk_len = 1 to disable.
"""

from dataclasses import dataclass, field
from typing import List


# =============================================================================
# Encoder — frozen visual backbone. We only use it to produce z_t for context.
# =============================================================================

@dataclass
class EncoderConfig:
    name: str = "r3m"            # "r3m" or "dinov2"
    freeze: bool = True
    raw_dim: int = 2048          # set in __post_init__ based on name
    img_size: int = 224


# =============================================================================
# Semantic Manager
# =============================================================================

@dataclass
class ManagerConfig:
    # Architecture
    hidden_dim: int = 256
    n_layers: int = 3

    # Optimization
    lr: float = 3e-4
    gamma: float = 0.99          # option-level discount
    tau: float = 0.005
    target_update_every: int = 1
    online_demo_ce_weight: float = 1.0
    online_demo_ce_steps: int = 1_000_000

    # Exploration over the 4 tasks (ε-greedy over MASKED logits)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000

    # Option horizon and episode budget
    subgoal_horizon: int = 40        # K — max low-level steps per option
    max_high_level_steps: int = 12   # up to 12 options per episode (enough for 4 tasks × 3 retries)

    # Rewards — ALL grounded in benchmark completion bits, not latents.
    # Option return = completion_bonus * (# completion bits that flipped during option)
    #               + dense_shaping * (task-space error reduction for selected task)
    #               - option_cost  per option step (small, encourages brevity)
    completion_bonus: float = 10.0
    offtask_completion_bonus: float = 0.0
    dense_shaping_weight: float = 1.0
    option_cost: float = 0.02
    chosen_failure_penalty: float = 0.1
    # Reward when the episode finishes all tasks (applied to final option)
    all_done_bonus: float = 20.0


# =============================================================================
# Grounded Worker
# =============================================================================

@dataclass
class WorkerConfig:
    # Architecture
    hidden_dim: int = 256
    n_layers: int = 3
    proprio_dim: int = 59            # full env obs size; we feed it raw + normalised
    task_embed_dim: int = 32         # FiLM conditioning dim per task

    # Optimization (SAC)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 1e-4
    gamma: float = 0.99              # per-low-level-step discount
    tau: float = 0.005
    init_alpha: float = 0.1
    min_alpha: float = 0.01
    auto_alpha: bool = True
    online_demo_bc_weight: float = 20.0
    online_demo_bc_steps: int = 200_000
    online_demo_batch_size: int = 256

    # ----- Action chunk toggle -----
    # action_chunk_len = 1  → standard single-step SAC (baseline ablation)
    # action_chunk_len > 1  → π0-style short-horizon chunk executed open-loop,
    #                          then re-queried. Chunk-level TD with n-step return.
    action_chunk_len: int = 1        # flip to 4 or 8 for the chunked ablation
    chunk_exec_mode: str = "open_loop"  # "open_loop" only for now

    # Worker reward weights (dense)
    # Reward per low-level step (or per chunk step, aggregated):
    #   r_w = progress_weight * Δ task_error   (DECREASE in error is positive)
    #       + completion_bonus * 1[task_k* completion bit flipped]
    #       - action_cost * ||a||^2
    #       - failure_penalty * 1[option ended in failure]
    progress_weight: float = 5.0
    completion_bonus: float = 5.0
    action_cost: float = 0.005
    failure_penalty: float = 0.1

    # Success criterion on task-space error (per-task ε is in utils.py)
    # If the task-space error drops below TASK_EPS[k] we also terminate the
    # option early. Completion bit flip still takes precedence.


# =============================================================================
# Buffer
# =============================================================================

@dataclass
class BufferConfig:
    worker_capacity: int = 300_000
    manager_capacity: int = 50_000
    batch_size: int = 256
    z_storage_dtype: str = "float16"
    # For chunked worker, we store (obs, chunk, reward_sum, next_obs_after_chunk)
    # The chunk length is fixed at config.worker.action_chunk_len.


# =============================================================================
# Warmup (Stage A)
# =============================================================================

@dataclass
class WarmupConfig:
    # Stage A is now demo-driven, not random-walk-driven.
    # We load offline kitchen demos, replay them in the env to recover the
    # benchmark's own completion events, render the recorded states through
    # MuJoCo, encode them with the frozen visual encoder, and BC-train the
    # worker on the teleoperated actions.
    dataset_source: str = "auto"   # "auto" | "minari" | "d4rl"
    dataset_ids: List[str] = field(default_factory=lambda: [
        "franka-complete",
        "franka-mixed",
        "franka-partial",
    ])
    cache_dir: str = "demo/cache"
    rebuild_cache: bool = False
    max_episodes_per_dataset: int = 0   # 0 -> use all episodes
    render_batch_size: int = 128
    manager_label_stride: int = 4       # subsample dense manager labels
    min_segment_len: int = 1
    seed_worker_replay_with_demos: bool = True
    balance_worker_task_sampling: bool = True

    # Stage-A supervised pretraining passes
    n_worker_sl_steps: int = 20_000
    n_worker_iql_steps: int = 100_000
    n_manager_sl_steps: int = 4_000
    sl_batch_size: int = 256
    iql_expectile: float = 0.7
    iql_adv_beta: float = 3.0
    iql_max_weight: float = 20.0


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class EvalConfig:
    n_eval_episodes: int = 15
    eval_every_env_steps: int = 15_000
    deterministic_worker: bool = True
    n_single_task_episodes: int = 20


# =============================================================================
# Training
# =============================================================================

@dataclass
class TrainingConfig:
    total_env_steps: int = 1_000_000
    worker_updates_per_env_step: int = 1
    manager_updates_per_option: int = 1
    stage_a_only: bool = False
    deterministic_torch: bool = True
    deterministic_worker_rollout_steps: int = 200_000
    worker_update_start_steps: int = 1_000_000
    manager_freeze_steps: int = 200_000
    scripted_manager_steps: int = 400_000
    scripted_manager_prob_start: float = 1.0
    scripted_manager_prob_end: float = 0.10
    scripted_manager_mode: str = "stage_a_rank"
    min_stage_a_task_success: float = 0.25
    unlock_remaining_tasks_steps: int = 1_000_000
    log_every_episodes: int = 20
    tb_every_episodes: int = 5

    # Checkpointing
    n_periodic_checkpoints: int = 10
    save_best: bool = True
    record_video: bool = True
    video_n_episodes: int = 3
    video_fps: int = 15

    # IO
    log_dir: str = "logs/smgw_run1"
    seed: int = 42
    device: str = "cuda"

    # The 4 tasks we care about (kitchen-complete-v0 set)
    tasks_to_complete: List[str] = field(default_factory=lambda: [
        'microwave', 'kettle', 'light switch', 'slide cabinet',
    ])


# =============================================================================
# Top-level config
# =============================================================================

@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # Encoder output dim depends on backbone
        if self.encoder.name == "r3m":
            self.encoder.raw_dim = 2048
        elif self.encoder.name == "dinov2":
            self.encoder.raw_dim = 384

    # Convenience: number of tasks
    @property
    def n_tasks(self) -> int:
        return len(self.training.tasks_to_complete)
