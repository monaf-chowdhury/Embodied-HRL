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


---


---

## Active Files

```text
config.py        Dataclass config
demo_dataset.py  D4RL/Minari loading, replay labels, render/encode cache
encoder.py       Frozen R3M, DINOv2, or DINOv3 image encoder
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

## Using shell

### Online RL Finetuning
Recommended first run, end-to-end offline `BC+IQL+Value_Target_0.05` + online AWAC:
```bash
bash run.sh --offline_algo bc_iql \
  --seed 2000 \
  --log_dir logs/online_awac_seed2000 \
  --iql_use_value_target \
  --iql_value_target_tau 0.05 \
  --online_finetune \
  --online_steps 200000 \
  --online_eval_interval 25000 \
  --chain_eval_episodes 50
```
If you already have the best offline checkpoint and want to skip offline retraining:
```bash
bash run.sh --offline_algo bc_iql \
  --seed 2000 \
  --log_dir logs/online_awac_from_ckpt_seed2000 \
  --iql_use_value_target \
  --iql_value_target_tau 0.05 \
  --load_checkpoint logs/YOUR_OFFLINE_RUN/checkpoints/checkpoint_final.pt \
  --skip_offline_training \
  --online_finetune \
  --online_steps 200000 \
  --online_eval_interval 25000 \
  --chain_eval_episodes 50
```

### Running Offline Pretraining
Baseline offline runs
```bash
# 1. Previous best baseline: no target value, no advantage norm
bash run.sh --offline_algo bc_iql --seed 42 --log_dir logs/bc_iql_baseline_seed42

# 2. Advantage normalization only
bash run.sh --offline_algo bc_iql --seed 42 --log_dir logs/bc_iql_advnorm_seed42 --iql_normalize_advantage

# 3. (Best offline performance) Target value only, larger tau 
bash run.sh --offline_algo bc_iql --seed 42 --log_dir logs/bc_iql_vtarget_tau005_seed42 --iql_use_value_target --iql_value_target_tau 0.05

# 4. Target value + advantage normalization
bash run.sh --offline_algo bc_iql --seed 42 --log_dir logs/bc_iql_vtarget_tau005_advnorm_seed42 --iql_use_value_target --iql_value_target_tau 0.05 --iql_normalize_advantage
```

Important: `--iql_value_target_tau 0.05` alone does nothing unless you also pass `--iql_use_value_target.`


Trying different offline algo
```bash
bash run.sh --offline_algo bc
bash run.sh --offline_algo bc_iql
bash run.sh --offline_algo td3bc
bash run.sh --offline_algo awr
bash run.sh --offline_algo bet
```
You can also pass extra train.py args through the script 
```bash
bash run.sh --offline_algo bc_iql --rebuild_demo_cache
bash run.sh --offline_algo td3bc --td3bc_alpha 2.0
bash run.sh --offline_algo awr --awr_temperature 0.5
bash run.sh --offline_algo bet --bet_num_bins 128 --bet_steps 100000
bash run.sh --offline_algo bc_iql --chain_eval_episodes 100 --no_video
```

## From the terminal
### Running offline pretraining 

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
