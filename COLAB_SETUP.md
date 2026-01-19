# NS-MDMPI Colab Setup

## Step 1: Mount Drive & Clone Repo
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/hungtrab/NS_MD_MPI.git
%cd NS_MD_MPI
!git checkout att_19
```

## Step 2: Install Dependencies (ORDER MATTERS!)
```python
# MuJoCo dependencies
!apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev

# Install in correct order
!pip install gymnasium[mujoco]==1.2.2
!pip install stable-baselines3==2.6.0
!pip install sb3-contrib==2.6.0
!pip install wandb==0.23.1
!pip install pyyaml numpy

# Verify
import gymnasium as gym
import mujoco
print("MuJoCo version:", mujoco.__version__)
env = gym.make("HalfCheetah-v4")
print("HalfCheetah OK!")
env.close()
```

## Step 3: Login WandB
```python
import wandb
wandb.login()
```

## Step 4: Run Training
```python
!python scripts/train.py --config configs/experiments/halfcheetah/C01_friction_sine.yaml --seed 42
```

## Quick One-liner Install
```python
!apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev && \
pip install gymnasium[mujoco]==1.2.2 stable-baselines3==2.6.0 sb3-contrib==2.6.0 wandb==0.23.1 pyyaml
```
