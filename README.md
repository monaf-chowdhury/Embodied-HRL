# Visual HRL for Long-Horizon Manipulation

**Feasibility-Frontier Visual HRL with Strict Execution for Franka Kitchen**

## What This Is

A minimal viable implementation of image-based Hierarchical Reinforcement Learning
for long-horizon manipulation. The system:

1. **Frozen R3M encoder + learned projection head** maps images to a compact latent space
2. **FPS landmark buffer** maintains grounded subgoal candidates from real observations
3. **Manager (DQN)** selects landmark subgoals — never generates arbitrary latent vectors
4. **Worker (SAC)** executes goal-conditioned actions with L2 delta-progress shaping
5. **Strict execution** (SSE-style): success continues the episode, failure terminates it
6. **Reachability predictor** filters unreachable subgoals before execution

## File Structure

```
hrl_visual_manipulation/
├── setup.sh          # Environment setup script (conda/pip)
├── config.py         # All hyperparameters in one place
├── encoder.py        # Frozen R3M + projection head
├── env_wrapper.py    # Franka Kitchen image wrapper + hierarchical protocol
├── landmarks.py      # FPS landmark buffer with visit tracking
├── networks.py       # Manager (DQN), Worker (SAC), Reachability predictor
├── buffers.py        # FER buffer (manager), standard buffer (worker), reachability buffer
├── train.py          # Main training loop
└── README.md         # This file
```

## Quick Start

```bash
# 1. Setup environment
chmod +x setup.sh
./setup.sh

# 2. For headless servers (no display), set rendering backend:
export MUJOCO_GL=egl    # Preferred for NVIDIA GPUs
# or: export MUJOCO_GL=osmesa  # Fallback

# 3. Run training
python train.py --seed 42 --device cuda --total_steps 2000000

# 4. Monitor with tensorboard
tensorboard --logdir logs/
```

## Multi-GPU Usage

Each run is single-GPU. Run different seeds/ablations on different GPUs:

```bash
# GPU 0 (RTX 5090): main experiment
CUDA_VISIBLE_DEVICES=0 python train.py --seed 42 --log_dir logs/seed42/

# GPU 1 (RTX 4090): seed 2
CUDA_VISIBLE_DEVICES=1 python train.py --seed 123 --log_dir logs/seed123/

# GPU 2 (RTX 4090): seed 3
CUDA_VISIBLE_DEVICES=2 python train.py --seed 456 --log_dir logs/seed456/

# GPU 3-6 (RTX 3090s): baselines and ablations
CUDA_VISIBLE_DEVICES=3 python train.py --seed 42 --n_landmarks 0 --log_dir logs/no_landmarks/
```

## Key Design Decisions

### Why landmark selection, not free-form z generation?
The manager selects from observed latent vectors (landmarks) rather than outputting
arbitrary z. This prevents three failure modes: out-of-manifold subgoals, gaming of
the temporal distance predictor, and shortcuts through semantically meaningless latents.

### Why strict execution?
When the worker fails to reach a subgoal, the high-level episode terminates with zero
reward. This prevents credit from spreading across failed subgoal attempts (the core
insight from SSE, ICLR 2026). Without this, the manager learns from relabeled
"successes" that were actually failures.

### Why L2 delta-progress, not temporal distance?
L2 in frozen latent space is stable and well-calibrated. Temporal distance predictors
can drift and be gamed. We use L2 as the default and temporal distance as an optional
ablation (to be added later).

## Known Limitations / TODOs

- [ ] Goal encoding is a placeholder (mean of all z). Replace with demo-derived goal.
- [ ] No DPO for manager yet (this is the Q-learning baseline). Add as ablation.
- [ ] No temporal distance predictor yet. Add as ablation.
- [ ] No projection head contrastive loss yet. Add for better latent structure.
- [ ] Single camera view. Multi-view would help with occlusion.
- [ ] No real-robot validation.

## Ablations to Run (for the paper)

1. **Full method** vs **no reachability filter** (most important)
2. **Full method** vs **no strict execution** (HER at high level instead)
3. **Full method** vs **no landmarks** (free-form z generation)
4. **Full method** vs **no reward shaping** (sparse only)
5. **R3M** vs **DINOv2** encoder
6. **With projection head** vs **raw encoder features**
7. Later: **+ DPO manager** vs **Q-learning manager**
8. Later: **+ temporal distance** vs **L2 distance** shaping

## GPU Compatibility Notes

| GPU | CUDA | VRAM | Notes |
|-----|------|------|-------|
| RTX 3090 | 12.x | 24GB | Works fine. Use for baselines/sweeps. |
| RTX 4090 | 12.x | 24GB | Primary training. Faster than 3090. |
| RTX 5090 | 12.x | 32GB | Best card. Use for development + biggest experiments. |

All three require NVIDIA driver >= 550 for CUDA 12.x. The torch build
(cu121) works on all three architectures (sm_86, sm_89, sm_100).
