# Embodied-HRL: Complete Setup Guide

**Hardware this was tested on:**
- CPU: AMD Ryzen 9 7900X
- GPU: NVIDIA RTX 3090 Ti (24GB VRAM)
- RAM: 32GB DDR5
- OS: Ubuntu 24.04 LTS
- NVIDIA Driver: 580.x, CUDA 13.0 (max supported)

**Note:** All `torch` installs use CUDA 12.1 binaries (`cu121`). These run fine on any driver that supports CUDA 12.x or higher.

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
# Must show: /home/<user>/anaconda3/envs/hrl/bin/python
```

---

## Step 3: Install PyTorch

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: 2.3.1+cu121 / True / NVIDIA GeForce RTX 3090 Ti
```

---

## Step 4: Set MuJoCo rendering backend permanently

On Ubuntu 24.04, `libgl1-mesa-glx` no longer exists. Use EGL instead (works with NVIDIA GPUs). Do not include `libgl1-mesa-glx` in the apt command — it will cause an error.

```bash
sudo apt-get install -y \
    libgl1-mesa-dev \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    ffmpeg

# Set env vars permanently in the conda env activation script
mkdir -p /home/$USER/anaconda3/envs/hrl/etc/conda/activate.d/
cat > /home/$USER/anaconda3/envs/hrl/etc/conda/activate.d/env_vars.sh << 'EOF'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
EOF

# Reload env so vars take effect
conda deactivate
conda activate hrl
echo $MUJOCO_GL
# Must print: egl
```

---

## Step 5: Install core Python dependencies

**Critical:** Pin `mujoco==2.3.7`. The D4RL kitchen XML uses MuJoCo 2.x syntax that MuJoCo 3.x rejects with `ValueError: XML Error: top-level default class 'main' cannot be renamed`.

```bash
pip install mujoco==2.3.7
pip install numpy==1.26.4
pip install scipy==1.13.1
pip install h5py==3.11.0
pip install gym==0.26.2
```

---

## Step 6: Install D4RL (kitchen only, without mujoco_py)

This is the most complex step. The standard D4RL install pulls in `mujoco_py` (requires a MuJoCo 2.1 binary installation) and downgrades gym. We avoid this by installing D4RL with `--no-deps`, then manually installing all the deps it actually needs.

**6a. Install all dm_control dependencies explicitly first:**
```bash
pip install \
    tqdm==4.67.3 \
    lxml==6.0.2 \
    protobuf==7.34.1 \
    pyparsing==3.3.2 \
    requests==2.33.1 \
    labmaze==1.0.6 \
    dm_env==1.6 \
    dm-tree==0.1.10 \
    pybullet==3.2.7 \
    termcolor==3.3.0 \
    click==8.3.2
```

**6b. Install dm_control 1.0.3 without deps (to avoid mujoco 3.x being pulled in):**
```bash
pip install dm_control==1.0.3 --no-deps
```

**6c. Re-pin mujoco to 2.3.7 (dm_control may have upgraded it):**
```bash
pip install mujoco==2.3.7
```

**6d. Install D4RL without deps:**
```bash
pip install git+https://github.com/Farama-Foundation/D4RL@master#egg=d4rl --no-deps
```

**6e. Patch D4RL's `__init__.py`** to only load the kitchen env (removes all broken locomotion/bullet/flow imports that require mujoco_py):

```bash
python - << 'EOF'
import site, os
site_packages = site.getsitepackages()[0]
init_file = os.path.join(site_packages, 'd4rl', '__init__.py')
print(f"Patching: {init_file}")
new_content = '''import os
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

from d4rl.kitchen import *
from d4rl import offline_env
'''
with open(init_file, 'w') as f:
    f.write(new_content)
print("Patched successfully.")
EOF
```

**6f. Verify kitchen works:**
```bash
python -c "
import gym, d4rl
env = gym.make('kitchen-complete-v0')
obs = env.reset()
print('Kitchen obs shape:', obs.shape)   # Expected: (60,)
print('Action space:', env.action_space) # Expected: Box(-1, 1, (9,))
env.close()
print('D4RL kitchen: OK')
"
```

All the gym deprecation warnings are harmless. Only worry if you see a traceback.

---

## Step 7: Install R3M encoder

```bash
pip install git+https://github.com/facebookresearch/r3m.git
```

Pre-download the R3M weights now (weights download from Google Drive on first call, ~100MB, saved to `~/.r3m/`):

```bash
python -c "from r3m import load_r3m; model = load_r3m('resnet50'); print('R3M weights ready.')"
```

---

## Step 8: Install remaining dependencies

```bash
pip install \
    opencv-python==4.10.0.84 \
    matplotlib==3.9.2 \
    tensorboard==2.17.1 \
    scikit-learn==1.5.1
```

---

## Step 9: Full sanity check

Run this entire block. Every section must succeed before starting training.

```bash
python -c "
import os, torch, numpy as np

print('=== GPU ===')
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')

print('\n=== D4RL Kitchen (state) ===')
import gym, d4rl
env = gym.make('kitchen-complete-v0')
obs = env.reset()
print('state obs:', obs.shape)
env.close()

print('\n=== Image Rendering ===')
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
print('latent z:', z.shape)  # Must be (1, 64)

print('\n=== Networks ===')
from networks import ManagerQNetwork, SACActorNetwork, SACCriticNetwork, ReachabilityPredictor
z_dim, action_dim = 64, 9
z_t = torch.zeros(4, z_dim).cuda()
z_g = torch.zeros(4, z_dim).cuda()
lm  = torch.zeros(10, z_dim).cuda()
act = torch.zeros(4, action_dim).cuda()
q = ManagerQNetwork(z_dim).cuda().evaluate_all_landmarks(z_t, z_g, lm)
print('manager Q:', q.shape)        # Must be torch.Size([4, 10])
a, lp = SACActorNetwork(z_dim, action_dim).cuda()(z_t, z_g)
print('actor:', a.shape)            # Must be torch.Size([4, 9])
q1, q2 = SACCriticNetwork(z_dim, action_dim).cuda()(z_t, z_g, act)
print('critic:', q1.shape)          # Must be torch.Size([4, 1])
rp = ReachabilityPredictor(z_dim).cuda()(z_t, z_g)
print('reachability:', rp.shape)    # Must be torch.Size([4, 1])

print('\n=== ALL CHECKS PASSED ===')
"
```

Expected final output:
```
=== GPU ===
2.3.1+cu121 True NVIDIA GeForce RTX 3090 Ti
VRAM: 25.3 GB

=== D4RL Kitchen (state) ===
state obs: (60,)

=== Image Rendering ===
MUJOCO_GL = egl
image obs: (224, 224, 3) uint8 min=9 max=255

=== R3M Encoder ===
latent z: (1, 64)

=== Networks ===
manager Q: torch.Size([4, 10])
actor: torch.Size([4, 9])
critic: torch.Size([4, 1])
reachability: torch.Size([4, 1])

=== ALL CHECKS PASSED ===
```

---

## Step 10: Run training

```bash
mkdir -p logs/seed42

CUDA_VISIBLE_DEVICES=0 python train.py \
    --seed 42 \
    --device cuda \
    --total_steps 2000000 \
    --encoder r3m \
    --n_landmarks 100 \
    --subgoal_horizon 20 \
    --log_dir logs/seed42/
```

**Phase 1 (random exploration)** runs 50 warmup episodes × up to 280 steps each. Each step encodes two images through R3M on GPU. Expect **8–15 minutes** before Phase 2 begins. You will see:
```
Phase 1: Random exploration for initial data...
  Collected 14000 low-level transitions
  Computed 100 landmarks via FPS
Phase 2: Hierarchical training loop...
```

Monitor training in a separate terminal:
```bash
tensorboard --logdir logs/ --port 6006
# Open http://localhost:6006 in browser
```

Monitor GPU usage:
```bash
watch -n 2 nvidia-smi
```

---

## Running multiple seeds / ablations

Each run is single-GPU. Use `CUDA_VISIBLE_DEVICES` to assign:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --seed 42  --log_dir logs/seed42/  &
CUDA_VISIBLE_DEVICES=1 python train.py --seed 123 --log_dir logs/seed123/ &
CUDA_VISIBLE_DEVICES=2 python train.py --seed 456 --log_dir logs/seed456/ &
```

---

## Known warnings (all harmless)

| Warning | Why it appears | Action |
|---|---|---|
| `Gym has been unmaintained since 2022` | gym 0.26 is old but D4RL requires it | Ignore |
| `Box bound precision lowered by casting to float32` | gym internal | Ignore |
| `Future gym versions will require seed` | API mismatch between gym versions | Ignore |
| `size_average and reduce args will be deprecated` | R3M uses old torch API internally | Ignore |
| `pretrained is deprecated` | R3M uses old torchvision API | Ignore |

---

## Troubleshooting

**`ValueError: XML Error: top-level default class 'main' cannot be renamed`**
MuJoCo version got upgraded to 3.x. Force reinstall: `pip install mujoco==2.3.7`

**`ValueError: not enough values to unpack (expected 5, got 4)`**
gym's `TimeLimit` wrapper is intercepting the step call. The `env_wrapper.py` in the repo uses `_env.unwrapped` to bypass this. Make sure you have the latest code: `git pull`

**`gym.error.NameNotFound: Environment kitchen-complete doesn't exist`**
`d4rl` was not imported before `gym.make`. Make sure `import d4rl` appears in `train.py`.

**`No module named 'mujoco_py'`**
D4RL's `__init__.py` was not patched. Re-run the patch command from Step 6e.

**`ModuleNotFoundError: No module named 'tqdm'` (or lxml, protobuf, dm_env, etc.)**
Some dm_control dependencies are missing. Re-run the full install block from Step 6a — it is safe to re-run.

**`AttributeError: 'list' object has no attribute 'shape'` in rendering**
Old `env_wrapper.py`. Pull latest: `git pull`

**`libgl1-mesa-glx` not found on Ubuntu 24**
This package was removed in Ubuntu 24. It is not needed — skip it. The apt commands in Step 4 already exclude it.

**R3M weights download fails**
Weights download from Google Drive to `~/.r3m/`. If behind a firewall, download manually from the R3M GitHub and place at `~/.r3m/r3m_50/model.pt`.
