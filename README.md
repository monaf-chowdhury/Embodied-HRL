# Visual Long-Horizon Manipulation

**Flat task-conditioned offline-to-online training is now the recommended path; hierarchy is retained as an ablation**

## What This Is

The codebase supports two modes:

- `flat_scripted`:
  a task-conditioned policy is pretrained offline on demos and then
  fine-tuned online while a scripted controller selects the next incomplete
  task.
- `hierarchical`:
  the older semantic-manager / grounded-worker SMGW path, kept for ablations.

At the moment, the evidence points to skill learning and skill retention as
the main scientific bottleneck, so `flat_scripted` is the default.

## File layout

```
smgw/
├── config.py         # Dataclass config; action-chunk toggle is worker.action_chunk_len
├── demo_dataset.py   # D4RL/Minari demo loading + offline render/encode cache for Stage A
├── env_wrapper.py    # FrankaKitchen-v1 image wrapper; reset() returns (img, state)
├── encoder.py        # Frozen R3M / DINOv2 visual backbone (context only, not subgoal)
├── networks.py       # SemanticManager (mask-aware Q) + GroundedWorker (FiLM + chunks)
├── buffers.py        # ManagerBuffer (option transitions) + WorkerBuffer (chunk tx)
├── agent.py          # SMGWAgent: option execution, task-grounded rewards, updates
├── warmup.py         # Stage A: demo BC on worker + optional CE on manager
├── train.py          # Main loop: Stage A → Stage B + checkpoints + videos
├── plots.py          # TensorBoard → PNG diagnostics
└── utils.py          # save_video, formatters, per-task indices, goals, ε, text embeds
```


## Quickstart

```bash
# Stage A only: offline BC/IQL + deterministic single-task worker eval
python train.py --bc_only \
  --seed 42 \
  --log_dir logs/bc_only \
  --demo_datasets franka-complete franka-mixed franka-partial

# Recommended full run: flat scripted chaining + online worker fine-tuning
python train.py --mode flat_scripted --seed 42 --log_dir logs/flat_scripted

# Hierarchical ablation
python train.py --mode hierarchical --seed 42 --log_dir logs/hierarchical_ablation

# Action-chunk ablation (π0-style chunks of length 4)
python train.py --action_chunk 4 --log_dir logs/smgw_chunk4

# Rebuild the offline demo render/latent cache
python train.py --bc_only --rebuild_demo_cache --log_dir logs/bc_recache

# Generate plots
python plots.py --log_dir logs/smgw_baseline
python plots.py --log_dir logs/ --compare    # overlay subdirs as separate runs
```


## What each component optimises

| Component | Input | Output | Objective |
|---|---|---|---|
| SemanticManager | `(z, proprio, task_state, completion_mask, text_embeds)` | Q-values over 4 tasks with completed-task logits masked to −∞ | DQN TD on `option_return = 10·(#new completions) + shaping·Δerr − cost + 20·all_done_bonus` |
| GroundedWorker (actor) | `(z, proprio, padded_cur, padded_goal, mask, text_embed[k])` | Tanh-squashed Gaussian over `H_chunk × 9` dims | SAC on `r = 5·Δtask_err + 5·completion_bit − 0.005·‖a‖² − 0.1·failure` |

**The invariant that makes this work:** every reward and every termination
criterion is a function of benchmark completion bits or task-space errors.
Zero latent distances. Training signal = evaluation signal by construction.

## Headline metric

`eval/full_task_success_rate` — fraction of eval episodes where all 4 tasks
completed. This is the chaining metric the old codebase could not move; the
whole point of this redesign is to move it.

## Key knobs you might want to sweep

- `worker.action_chunk_len` — 1 vs 4 vs 8. Primary ablation.
- `manager.subgoal_horizon` — option budget K. Default 40; try 60 if the
  worker needs more steps per task.
- `worker.progress_weight` vs `worker.completion_bonus` — relative strength of
  dense shaping vs terminal bonus. Raise completion_bonus if the worker
  hovers at medium error without closing.
- `manager.option_cost` — raise to 0.1 if the manager is running too many
  wasted options before completing the episode.

## Residual caveats

1. Stage A labels task segments by task-space completion order reconstructed
   from demo states. If your environment version changes task tolerances,
   re-run the timeline sanity check before trusting the labels.
2. Chunked mode stores full `H×9` action tensors even when the chunk
   terminated early (e.g. task completed on step 2 of a 4-step chunk). For
   `H=1` this is a no-op; for `H≥4` it is a small bias in the critic.
3. The env wrapper assumes `info['episode_task_completions']` contains
   exactly the task names in our `tasks_to_complete` list. If that mapping
   breaks, the completion mask stays zero — print the info dict on first
   completion to verify.


--- 

## Embodied-HRL: Setup Guide (FrankaKitchen-v1)

**Environment:** FrankaKitchen-v1 from `gymnasium-robotics` (NOT legacy D4RL v0)  
**Hardware tested on:** RTX 3090 Ti · AMD Ryzen 9 7900X · 32GB RAM · Ubuntu 24.04  
**Python:** 3.10  

---


---

## Step 1: Clone the repo

```bash
git clone https://github.com/monaf-chowdhury/Embodied-HRL
cd Embodied-HRL
```

---

## Step 2: Create conda environment

```bash
conda create -n hrl python=3.10 -y
conda activate hrl
which python
# Must show: .../anaconda3/envs/hrl/bin/python
```

---

## Step 3: Install PyTorch (CUDA 12.1)

Works on RTX 3090 Ti (sm_86), 4090 (sm_89), 5090 (sm_100) — any driver ≥ 525.

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

For RTX 5090 Ti (sm_120) 
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.3.1+cu121  True  NVIDIA GeForce RTX 3090 Ti
```

---

## Step 4: Set rendering backend (EGL for headless NVIDIA)

`libgl1-mesa-glx` does not exist on Ubuntu 24 — skip it. Use EGL.

```bash
sudo apt-get install -y \
    libgl1-mesa-dev \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 libglfw3-dev \
    patchelf ffmpeg

# Permanently set EGL in the conda env
mkdir -p ~/anaconda3/envs/hrl/etc/conda/activate.d/
cat > ~/anaconda3/envs/hrl/etc/conda/activate.d/env_vars.sh << 'EOF'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
EOF

conda deactivate && conda activate hrl
echo $MUJOCO_GL   # Must print: egl
```

---

## Step 5: Install gymnasium-robotics and MuJoCo

```bash
pip install gymnasium==1.0.0
pip install gymnasium-robotics==1.4.2
pip install mujoco==3.6.0
```

Verify:
```bash
python -c "
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
env = gym.make('FrankaKitchen-v1',
               tasks_to_complete=['microwave', 'kettle'],
               render_mode='rgb_array')
obs, info = env.reset()
print('obs keys:', list(obs.keys()))
print('observation shape:', obs['observation'].shape)
print('action space:', env.action_space)
env.close()
print('FrankaKitchen-v1: OK')
"
# Expected:
# obs keys: ['observation', 'achieved_goal', 'desired_goal']
# observation shape: (59,)
# action space: Box(-1.0, 1.0, (9,), float32)
# FrankaKitchen-v1: OK
```

---

## Step 6: Install remaining dependencies

```bash
pip install \
    numpy==1.26.4 \
    scipy==1.13.1 \
    opencv-python==4.10.0.84 \
    matplotlib==3.9.2 \
    tensorboard==2.17.1 \
    tqdm==4.66.5 \
    scikit-learn==1.5.1 \
    timm==0.9.10 \
    tokenizers==0.19.1 \
    transformers==4.40.1 \
    minari==0.5.3
```

---

## Step 7: Install R3M encoder

```bash
pip install git+https://github.com/facebookresearch/r3m.git
```

Pre-download weights (~100MB, saved to `~/.r3m/`):
```bash
python -c "from r3m import load_r3m; load_r3m('resnet50'); print('R3M weights ready.')"
```

---

## Step 8: Full sanity check

Run this entire block — every section must pass before training.

```bash
python -c "
import os, torch, numpy as np

print('=== GPU ===')
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')

print('\n=== FrankaKitchen-v1 ===')
import gymnasium as gym, gymnasium_robotics
gym.register_envs(gymnasium_robotics)
env = gym.make('FrankaKitchen-v1',
               tasks_to_complete=['microwave','kettle','light switch','slide cabinet'],
               render_mode='rgb_array', width=224, height=224)
obs, _ = env.reset()
print('obs shape:', obs['observation'].shape)   # (59,)
img = env.render()
print('render shape:', img.shape, img.dtype)    # (224, 224, 3) uint8
obs2, r, term, trunc, info = env.step(env.action_space.sample())
print('step ok — reward:', r, 'tasks done:', info.get('episode_task_completions'))
env.close()

print('\n=== Image Wrapper ===')
print('MUJOCO_GL =', os.environ.get('MUJOCO_GL', 'NOT SET'))
from env_wrapper import FrankaKitchenImageWrapper
img_env = FrankaKitchenImageWrapper(img_size=224)
img = img_env.reset()
print('image obs:', img.shape, img.dtype, 'min=%d max=%d' % (img.min(), img.max()))
img_env.close()

print('\n=== R3M Encoder ===')
from config import EncoderConfig
from encoder import VisualEncoder
enc = VisualEncoder(EncoderConfig(name='r3m'), device='cuda')
z = enc.encode_numpy(img)
print('latent z:', z.shape)   # (1, 64)

print('\n=== Networks ===')
from networks import ManagerQNetwork, SACActorNetwork, SACCriticNetwork, ReachabilityPredictor
z_dim, action_dim = 64, 9
z_t = torch.zeros(4, z_dim).cuda()
z_g = torch.zeros(4, z_dim).cuda()
lm  = torch.zeros(10, z_dim).cuda()
act = torch.zeros(4, action_dim).cuda()
q   = ManagerQNetwork(z_dim).cuda().evaluate_all_landmarks(z_t, z_g, lm)
print('manager Q:', q.shape)   # [4, 10]
a, lp = SACActorNetwork(z_dim, action_dim).cuda()(z_t, z_g)
print('actor:', a.shape)       # [4, 9]
q1, q2 = SACCriticNetwork(z_dim, action_dim).cuda()(z_t, z_g, act)
print('critic:', q1.shape)     # [4, 1]
rp = ReachabilityPredictor(z_dim).cuda()(z_t, z_g)
print('reachability:', rp.shape)  # [4, 1]

print('\n=== ALL CHECKS PASSED ===')
"
```

---

## Step 9: Verify goal image

Before training, check that the goal image looks correct (open microwave, moved kettle, light on, slide cabinet open):

```bash
python -c "
from config import EncoderConfig
from encoder import VisualEncoder
from utils import compare_goal_methods

enc = VisualEncoder(EncoderConfig(name='r3m'), device='cuda')
compare_goal_methods(enc, log_dir='logs/goal_check/')
print('Open logs/goal_check/goal_image.png to verify it looks correct.')
"
```

---

## Step 10: Run training

**Default (4 tasks, 1M steps):**
```bash
mkdir -p logs/seed42

CUDA_VISIBLE_DEVICES=0 python train.py \
    --seed 42 \
    --device cuda \
    --total_steps 1000000 \
    --encoder r3m \
    --subgoal_horizon 20 \
    --log_dir logs/seed42/ \
    --demo_datasets franka-complete franka-mixed franka-partial \
    --tasks microwave kettle 'light switch' 'slide cabinet'
```

**BC only (recommended first signal after the pivot):**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --bc_only \
    --seed 42 \
    --device cuda \
    --encoder r3m \
    --worker_bc_steps 20000 \
    --single_task_eval_episodes 20 \
    --log_dir logs/bc_only/ \
    --demo_datasets franka-complete franka-mixed franka-partial \
    --tasks microwave kettle 'light switch' 'slide cabinet'
```

**Easier (2 tasks — good for debugging):**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --seed 42 \
    --total_steps 500000 \
    --log_dir logs/2task/ \
    --tasks microwave kettle
```

**3 tasks:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --seed 42 \
    --total_steps 750000 \
    --log_dir logs/3task/ \
    --tasks microwave kettle 'light switch'
```

**Phase 1 timing:** ~8–15 minutes (50 warmup episodes × ~280 steps, encoding every frame through R3M on GPU).

---

## Step 11: Monitor

```bash
# TensorBoard (separate terminal)
tensorboard --logdir logs/ --port 6006
# Open http://localhost:6006

# GPU usage
watch -n 5 nvidia-smi
```

---

## Multiple seeds / ablations

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --seed 42  --log_dir logs/seed42/ &
CUDA_VISIBLE_DEVICES=1 python train.py --seed 123 --log_dir logs/seed123/ &
CUDA_VISIBLE_DEVICES=2 python train.py --seed 456 --log_dir logs/seed456/ &
```

---

## Generate diagnostic plots

```bash
# Single run
python plots.py --log_dir logs/seed42/ --smooth 15

# Compare seeds
python plots.py --log_dir logs/ --compare
```

---

## Files changed vs v0

| File | Status | Why |
|---|---|---|
| `env_wrapper.py` | **Replaced** | gymnasium API, `render_mode='rgb_array'`, obs dict |
| `utils.py` | **Replaced** | Uses `mujoco.mj_forward` (new bindings), no d4rl |
| `agent.py` | **Replaced** | Removed `import d4rl` |
| `train.py` | **Replaced** | v1 step API, task-aware logging, `--tasks` arg |
| `config.py` | **Replaced** | Added `tasks_to_complete` field |
| `encoder.py` | **Unchanged** | No dependency on env version |
| `networks.py` | **Unchanged** | Pure PyTorch |
| `buffers.py` | **Unchanged** | Pure NumPy |
| `landmarks.py` | **Unchanged** | Pure NumPy |
| `plots.py` | **Unchanged** | Reads TensorBoard, no env dependency |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'gymnasium_robotics'`**
```bash
pip install gymnasium-robotics
```

**`gym.error.NameNotFound: FrankaKitchen-v1 not found`**
You must call `gym.register_envs(gymnasium_robotics)` before `gym.make`. This is done automatically inside `env_wrapper.py` and `utils.py`.

**Render returns None or black image**
Make sure `MUJOCO_GL=egl` is set. Run `echo $MUJOCO_GL`. If not set, `conda deactivate && conda activate hrl`.

**`gymnasium` version conflict**
`gymnasium-robotics` requires `gymnasium>=1.0`. If you have an older version: `pip install gymnasium==1.0.0 gymnasium-robotics`.

**R3M download fails**
Weights go to `~/.r3m/`. If behind a firewall, download from the R3M GitHub manually and place at `~/.r3m/r3m_50/model.pt`.

**`libgl1-mesa-glx` not found**
This package was removed in Ubuntu 24. It is not needed — EGL works without it. The Step 4 apt command already excludes it.
