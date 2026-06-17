# Embodied-HRL: Skill Learning Q Chunking Flow QL Branch
## Shared Skill-Conditioned QC-FQL

This branch replaces the Gaussian-actor + IQL/LQL stack and the isolated
per-task specialist design with **one shared skill-conditioned QC-FQL agent**:
an expressive **flow-matching** policy class (Flow Q-Learning) trained with a
**chunked, unbiased n-step critic** (Q-chunking). Task identity enters through
goal/current/mask conditioning plus learned task-ID embeddings, and composition
uses a fixed next-incomplete predicate planner.

References:
- Flow Q-Learning (FQL) — Park, Li, Levine, **ICML 2025**, arXiv:2502.02538
- Reinforcement Learning with Action Chunking (QC) — Li, Zhou, Levine, **NeurIPS 2025**, arXiv:2507.07969


The shared learner is **QC-FQL** (Q-chunking + Flow Q-Learning). See
[QCFQL.md](QCFQL.md) for the architecture, equations, and staged experiments.

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

## Step 8.1: Install R3M encoder

```bash
pip install git+https://github.com/facebookresearch/r3m.git
```

Pre-download weights (~100MB, saved to `~/.r3m/`):
```bash
python -c "from r3m import load_r3m; load_r3m('resnet50'); print('R3M weights ready.')"
```

## Step 8.2: Download DinoV3 Weights

Request access from **[Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)** . Download links will be provided to the email 

```bash
# Download dinov3 repo whereever you want...
git clone https://github.com/facebookresearch/dinov3.git

cd dinov3
mkdir dinov3_weights
cd dinov3_weights

wget -c -O dinov3_vits16plus_pretrain_lvd1689m.pth 'PASTE_THE_LINK_INSIDE'
```


## Active Files

```text
config.py            Dataclass config
demo_dataset.py      D4RL/Minari loading, replay labels, render/encode cache
encoder.py           Frozen R3M, DINOv2, or DINOv3 image encoder
env_wrapper.py       FrankaKitchen image/state wrapper
networks.py          FlowActor (flow BC + one-step actor) and TwinQ chunk critic
specialist.py        One visual QC-FQL policy per task
offline_algorithms.py  FlowBCAlgorithm + QCFQLAlgorithm (+ prefix-val selection)
online_finetune.py   Online QC-FQL fine-tuning on a mixed demo+online replay
train.py             Main training/evaluation entrypoint
plots.py             TensorBoard-to-PNG diagnostics
utils.py             task indices, goals, eval helpers, video saving
```

## What Is Trained

For each configured task, the code trains a separate visual QC-FQL policy.

Input (conditioning feature):

```text
image latent z + normalized proprio + task goal/current/delta/mask
```

Output:

```text
H x 9 Franka action chunk (action_chunk=H, env action is 9-D)
```

Training:

```text
1. flow_bc : flow-matching BC on replay-labelled demo segments
2. qc_fql  : chunked twin critic + flow BC + one-step Q-maximizing actor
3. Prefix-state validation for per-skill checkpoint selection
4. Single-task and scripted-chain evaluation (optionally online fine-tuning)
```

There is no learned manager in this branch.

# Main Command

`Cache Rule`
If you change encoder, action_chunk, tasks, image size, reward weights, then rebuild the cache. Just add `--rebuild_demo_cache` this flag towards the end of the command. 

Algorithm choice alone does not require rebuilding. You do not need --rebuild_demo_cache for every algorithm ablations. Reuse cache if these are unchanged:

Encoder choices:

```bash
--encoder r3m
--encoder dinov2
--encoder dinov3 --dinov3_weights /path/to/dinov3_checkpoint.pth
```

DINOv3 can also read the checkpoint from `DINOV3_WEIGHTS`. Rebuild demo caches when switching between encoders or DINOv3 model variants.

## Staged experiments

**Train on all three datasets** (`franka-complete franka-mixed franka-partial`,
the `run.sh` default). `complete` provides the canonical-order, fresh-start
trajectories each per-skill policy needs to *initiate* its task; `partial`/`mixed`
add suboptimal coverage. Training on `partial` alone starves the chain (light
switch / slide cabinet are almost never demonstrated from a fresh kitchen there)
and the scripted chain collapses even when each skill looks fine from oracle
prefix states.

**0. Pre-flight (seconds, CPU, no env/encoder):**
```bash
python test.py
```

**1. Does an expressive policy class alone beat the old Gaussian BC? (flow BC)**
```bash
bash run.sh --offline_algo flow_bc --log_dir logs/flowbc_all_seed0 --seed 0 --no_video
```

**2. Does QC-FQL improve over flow BC? (Q-guidance + chunked critic)**
```bash
bash run.sh --offline_algo qc_fql --log_dir logs/qcfql_all_seed0 --seed 0 --no_video
```
Watch `03_qc_diagnostics.png`: `qc_q_mean` should settle near the reward scale
(a few × completion_bonus), **not** blow up. Note: with success-segment labeling
the one-step Q-actor can fall *below* flow BC (no critic contrast) — if so, that
is the signal for the full per-skill relabel, not a tuning issue.

**3. Online QC-FQL fine-tuning (only after offline is confirmed):**
```bash
bash run.sh --offline_algo qc_fql --online_finetune \
  --online_steps 200000 --online_eval_interval 25000 \
  --log_dir logs/qcfql_online_seed0 --seed 0
```
Resume from a saved offline checkpoint instead of retraining:
```bash
bash run.sh --offline_algo qc_fql --online_finetune \
  --load_checkpoint logs/YOUR_OFFLINE_RUN/checkpoints/checkpoint_final.pt \
  --skip_offline_training \
  --online_steps 200000 --online_eval_interval 25000 \
  --log_dir logs/qcfql_online_from_ckpt_seed0 --seed 0
```

Tuning order if needed: `--fql_alpha` (start 10, lower to loosen the behavior
constraint), then `--best_of_n` (e.g. 4–8 at eval), then `--flow_steps`.

## From the terminal (no shell wrapper)

```bash
python train.py \
  --seed 42 \
  --device cuda \
  --encoder dinov3 \
  --offline_algo qc_fql \
  --flow_bc_steps 30000 \
  --offline_rl_steps 100000 \
  --batch_size 256 \
  --single_task_eval_episodes 100 \
  --chain_eval_episodes 100 \
  --log_interval 500 \
  --demo_datasets franka-complete franka-mixed franka-partial \
  --tasks microwave kettle "light switch" "slide cabinet" \
  --log_dir logs/qcfql_dinov3
```

## Prefix-State Diagnostic

```bash
python train.py \
  --seed 42 \
  --device cuda \
  --encoder dinov3 \
  --offline_algo qc_fql \
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
python plots.py --log_dir logs/qcfql_dinov3 --smooth 15
python plots.py --log_dir logs/flowbc_partial_seed0 --smooth 15
```

## What To Watch

Primary metrics:

```text
single_task/<task>_success_rate
eval/full_task_success_rate
eval/mean_tasks_completed
skill/<task>/prefix_success_rate
qc_q_mean   (inflation watch — should stay near the reward scale)
```

If microwave/kettle work but light switch/cabinet fail, the bottleneck is
skill competence, not hierarchy.

## Research Direction

First make each individual skill competent from images. Then test scripted
composition. Only after that should online fine-tuning or a learned manager be
added back.
