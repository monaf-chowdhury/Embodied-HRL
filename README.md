# Embodied-HRL: Lean Skill Learning Branch

This branch intentionally drops the learned hierarchy and teacher/student
curriculum. The current research path is simpler:

```text
FrankaKitchen demos -> replay labels -> rendered images -> per-task BC/IQL skills -> scripted chaining eval
```
---

# How to set the environment

Environment: FrankaKitchen-v1 from gymnasium-robotics (NOT legacy D4RL v0)
Hardware tested on: RTX 3090 Ti and RTX 5090 Ti · AMD Ryzen 9 7900X · 32GB RAM · Ubuntu 24.04
Python: 3.10

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

## Step 7: Download Minari Datasets
```bash
conda activate hrl

pip install "minari[all]"
minari download D4RL/kitchen/complete-v2
minari download D4RL/kitchen/mixed-v2
minari download D4RL/kitchen/partial-v2
minari list local
```

---

## Step 8: Install R3M encoder

```bash
pip install git+https://github.com/facebookresearch/r3m.git
```

Pre-download weights (~100MB, saved to `~/.r3m/`):
```bash
python -c "from r3m import load_r3m; load_r3m('resnet50'); print('R3M weights ready.')"
```

---


---

## Active Files

```text
config.py        Dataclass config
demo_dataset.py  D4RL/Minari loading, replay labels, render/encode cache
encoder.py       Frozen R3M or DINOv2 image encoder
env_wrapper.py   FrankaKitchen image/state wrapper
networks.py      Minimal MLP helper
specialist.py    One visual BC/IQL policy per task
train.py         Main training/evaluation entrypoint
plots.py         TensorBoard-to-PNG diagnostics
utils.py         task indices, goals, eval helpers, video saving
```

## What Is Trained

For each configured task, the code trains a separate visual policy.

Input:

```text
image latent z + normalized proprio + task goal/current/delta/mask
```

Output:

```text
9-D Franka action, or H x 9 if action_chunk > 1
```

Training:

```text
1. BC on replay-labelled demo segments
2. Optional IQL on the same demo transitions
3. Deterministic single-task evaluation
4. Deterministic scripted-chain evaluation
```

There is no learned manager in this branch.

## Main Command

```bash
python train.py \
  --seed 42 \
  --device cuda \
  --encoder dinov2 \
  --bc_steps 30000 \
  --iql_steps 100000 \
  --batch_size 256 \
  --single_task_eval_episodes 20 \
  --chain_eval_episodes 15 \
  --log_interval 500 \
  --rebuild_demo_cache \
  --demo_datasets franka-complete franka-mixed franka-partial \
  --tasks microwave kettle "light switch" "slide cabinet" \
  --log_dir logs/lean_skills_sparse_dominant_dinov2 
```

## Prefix-State Diagnostic

```bash
python train.py \
  --seed 42 \
  --device cuda \
  --encoder r3m \
  --bc_steps 30000 \
  --iql_steps 100000 \
  --prefix_eval_only \
  --prefix_target_task "light switch" \
  --prefix_condition_tasks microwave kettle \
  --prefix_eval_states 20 \
  --log_dir logs/prefix_light_after_mw_kettle \
  --demo_datasets franka-complete franka-mixed franka-partial \
  --tasks microwave kettle "light switch" "slide cabinet"
```

## Plotting

```bash
python plots.py --log_dir logs/lean_skills_sparse_dominant --smooth 15
python plots.py --log_dir logs/prefix_light_sparse_dominant --smooth 15
python plots.py --log_dir logs/prefix_slide_sparse_dominant --smooth 15
```

## What To Watch

Primary metrics:

```text
single_task/<task>_success_rate
eval/full_task_success_rate
eval/mean_tasks_completed
prefix/success_rate
```

If microwave/kettle work but light switch/cabinet fail, the bottleneck is
skill competence, not hierarchy.

## Research Direction

First make each individual skill competent from images. Then test scripted
composition. Only after that should online fine-tuning or a learned manager be
added back.
