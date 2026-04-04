#!/bin/bash
# =============================================================================
# Visual HRL for Long-Horizon Manipulation — Environment Setup
# =============================================================================
# Tested on: Ubuntu 22.04 / 24.04 LTS
# GPUs: NVIDIA RTX 3090, 4090, 5090
# 
# IMPORTANT NOTES:
# - Franka Kitchen (from D4RL) uses MuJoCo. The modern mujoco pip package 
#   (>=2.3.0) bundles its own binaries — no separate MuJoCo install needed.
# - R3M requires torch. We pin torch to a version compatible with all three GPUs.
# - RTX 5090 needs CUDA 12.x. RTX 3090/4090 work with CUDA 11.8+.
#   We use CUDA 12.1 torch builds which work on all three.
# =============================================================================

set -e

echo "============================================="
echo "Step 1: Creating conda environment"
echo "============================================="

# If conda not available, use venv instead
if command -v conda &> /dev/null; then
    conda create -n hrl_visual python=3.10 -y
    conda activate hrl_visual
    echo "Using conda environment"
else
    python3.10 -m venv hrl_visual_env
    source hrl_visual_env/bin/activate
    echo "Using venv environment"
    echo "NOTE: If python3.10 is not available, install it via:"
    echo "  sudo apt install python3.10 python3.10-venv python3.10-dev"
fi

echo "============================================="
echo "Step 2: Installing PyTorch (CUDA 12.1)"
echo "============================================="
# This torch version works on RTX 3090 (sm_86), 4090 (sm_89), 5090 (sm_100)
# All require CUDA 12.x driver (>=525.60.13)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

echo "============================================="
echo "Step 3: Installing MuJoCo and Franka Kitchen"
echo "============================================="
# Modern mujoco pip package — no manual install needed
pip install mujoco==3.1.6

# dm_control for rendering
pip install dm_control==1.0.18

# D4RL contains Franka Kitchen
# NOTE: D4RL's official pip package can be finicky. We install from git.
# If this fails, see TROUBLESHOOTING below.
pip install git+https://github.com/Farama-Foundation/D4RL@master

# Gymnasium (D4RL uses older gym API, we bridge it)
pip install gymnasium==0.29.1
pip install gym==0.26.2

echo "============================================="
echo "Step 4: Installing R3M visual encoder"
echo "============================================="
# R3M from Meta Research
pip install git+https://github.com/facebookresearch/r3m.git

echo "============================================="
echo "Step 5: Installing remaining dependencies"
echo "============================================="
pip install numpy==1.26.4
pip install scipy==1.13.1
pip install scikit-learn==1.5.1  # For FPS landmark selection
pip install opencv-python==4.10.0.84
pip install matplotlib==3.9.2
pip install tensorboard==2.17.1
pip install tqdm==4.66.5
pip install h5py==3.11.0

echo "============================================="
echo "Step 6: Verifying installation"
echo "============================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')

import mujoco
print(f'MuJoCo: {mujoco.__version__}')

try:
    import gym
    env = gym.make('kitchen-complete-v0')
    obs = env.reset()
    print(f'Franka Kitchen loaded. Obs shape: {obs.shape}')
    env.close()
    print('Franka Kitchen: OK')
except Exception as e:
    print(f'Franka Kitchen: FAILED — {e}')
    print('See TROUBLESHOOTING section in README.')

try:
    from r3m import load_r3m
    print('R3M: importable (weights download on first use)')
except Exception as e:
    print(f'R3M: FAILED — {e}')
"

echo "============================================="
echo "Setup complete!"
echo "============================================="

# =============================================================================
# TROUBLESHOOTING
# =============================================================================
#
# 1. D4RL install fails:
#    Try: pip install git+https://github.com/Farama-Foundation/D4RL@master --no-deps
#    Then manually install its deps: pip install h5py pybullet
#
# 2. MuJoCo rendering issues (headless server):
#    export MUJOCO_GL=egl
#    If EGL doesn't work: export MUJOCO_GL=osmesa
#    Install: sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
#
# 3. R3M download fails:
#    R3M weights download from torch hub on first call. If behind firewall:
#    manually download resnet50 weights and place in ~/.torch/hub/
#
# 4. RTX 5090 + CUDA:
#    Ensure NVIDIA driver >= 550.54.14 for CUDA 12.x
#    Check: nvidia-smi
#
# 5. Ubuntu 24.04 + gym:
#    If gym.make fails with "module 'collections' has no attribute 'MutableMapping'":
#    This is a Python 3.10+ compatibility issue in older gym.
#    Fix: pip install setuptools==65.5.0
# =============================================================================
