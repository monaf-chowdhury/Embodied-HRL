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
    dense_shaping_weight: float = 1.0
    option_cost: float = 0.02
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
# Warmup (Stage A) — DEMO-BASED behavioural cloning
# =============================================================================
# Step 2 of the fix plan: replace random-walk warmup with BC pretraining on
# real teleoperated demonstrations from Minari / D4RL. The key change vs. the
# old WarmupConfig is that the actions the worker regresses toward are
# purposeful demo actions (not random noise), so BC actually teaches the
# worker how to approach and manipulate the objects.

@dataclass
class WarmupConfig:
    # ---- Demo sources ----
    # Minari dataset ids. We include all three kitchen variants because
    # Complete alone is only ~130k transitions and the worker benefits
    # enormously from seeing partial (unsuccessful-chain) trajectories for
    # the constituent sub-skills.
    minari_dataset_ids: List[str] = field(default_factory=lambda: [
        "D4RL/kitchen/complete-v2",
        "D4RL/kitchen/partial-v2",
        "D4RL/kitchen/mixed-v2",
    ])
    # Cap total transitions loaded across all datasets. Set to 0 = unlimited.
    max_transitions: int = 0
    # Cap per-task transitions after labelling (keeps dataset balanced across
    # tasks). 0 = no cap.
    max_per_task: int = 50_000
    # Cache decoded demo dataset here so we don't redo the label-and-render
    # pass on every run.
    cache_path: str = "cache/demo_bc.npz"
    # Force rebuild of the cache even if present. Useful after you change
    # index conventions or the task list.
    rebuild_cache: bool = False
    # Render images during demo loading? Required for training the worker's
    # z-context; if False, z is filled with zeros (faster but the worker
    # won't be able to use visual context).
    render_demo_images: bool = True

    # ---- BC training ----
    n_worker_bc_steps: int = 20_000
    n_manager_bc_steps: int = 3_000
    bc_batch_size: int = 256
    bc_lr: float = 3e-4           # we use a dedicated LR for BC; SAC's actor_lr can differ

    # Save a worker checkpoint right after BC so you can resume Stage B later
    # without redoing warmup.
    save_bc_checkpoint: bool = True
    bc_checkpoint_path: str = "logs/smgw_bc/bc_worker.pt"


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class EvalConfig:
    n_eval_episodes: int = 15
    eval_every_env_steps: int = 15_000
    deterministic_worker: bool = True


# =============================================================================
# Training
# =============================================================================

@dataclass
class TrainingConfig:
    total_env_steps: int = 1_000_000
    worker_updates_per_env_step: int = 1
    manager_updates_per_option: int = 1
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