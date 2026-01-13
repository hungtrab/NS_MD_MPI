# NS-MDMPI: Non-Stationary Meta-Learning for Deep Reinforcement Learning

**Adaptive RL for Changing Environments using Variation Budgets & Meta-Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Running Experiments](#running-experiments)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Results Analysis](#results-analysis)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [Project Structure](#project-structure)

---

## 🎯 Overview

NS-MDMPI (Non-Stationary Meta-Learning with Variation Budgets) is an adaptive deep RL algorithm designed for non-stationary environments where dynamics, rewards, or policies change over time.

### **Key Features**

- ✅ **Variation Budget Tracking** - Monitor V_R (reward), V_P (transition), V_π* (policy) budgets
- ✅ **Adaptive Trust Regions** - Dynamically adjust policy update constraints based on drift
- ✅ **Meta-Learning** - Learn to adapt quickly to environment changes
- ✅ **Multi-Algorithm Support** - PPO, SAC, TRPO implementations
- ✅ **Drift Detection** - Automatic detection of environment changes
- ✅ **WandB Integration** - Comprehensive experiment tracking

### **Supported Environments**

- **MuJoCo:** Hopper, HalfCheetah, Walker2D (planned), Swimmer (planned), Humanoid (planned)
- **Gym:** LunarLander, CartPole (legacy), MountainCar (legacy)

---

## 🚀 Installation

### **1. Clone Repository**

```bash
git clone https://github.com/hungtrab/NS_MD_MPI.git
cd NS_MD_MPI
```

### **2. Create Conda Environment**

```bash
conda create -n rl_hf_course python=3.10
conda activate rl_hf_course
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt

# For hyperparameter tuning (optional)
pip install optuna optuna-dashboard
```

### **4. Verify Installation**

```bash
python scripts/test.sh
```

Expected output: All tests pass ✅

---

## ⚡ Quick Start

### **Run Your First Experiment**

```bash
# Activate environment
conda activate rl_hf_course

# Run Hopper with moderate drift (baseline)
bash scripts/run_hopper_moderate_ppo.sh
```

This will run ~12 experiments (6 drift configs × 2 methods: baseline + NS-MDMPI).

### **Monitor Progress**

**Check logs:**
```bash
tail -f logs/hopper_friction_sine_baseline_ppo.log
```

**View on WandB:**
- Navigate to: `https://wandb.ai/<your-username>/att_3_Hopper_Moderate_Comparison`

---

## ⚙️ Configuration

All experiments are defined by YAML config files in `configs/{Algorithm}/{DriftType}/`.

### **Config Structure**

```yaml
# configs/PPO/moderate/hopper_friction_sine_nsmdmpi_ppo.yaml

env_id: Hopper-v4

env:
  parameter: friction      # What changes: friction, mass_scale, damping, etc.
  drift_type: sine        # How it changes: sine, linear, jump, random_walk
  magnitude: 0.27         # Drift strength
  period: 10000           # Timesteps per cycle (for periodic drift)
  base_value: 0.9         # Starting value
  bounds: [0.45, 1.35]    # Min/max range

wandb:
  project: att_3_Hopper_Moderate_Comparison
  tags: [hopper, moderate, ppo, friction, sine, nsmdmpi]
  mode: online

train:
  algorithm: PPO
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  total_timesteps: 1000000
  seed: 42

nsmdmpi:
  enabled: true           # Enable NS-MDMPI (false for baseline)
  V_R: 10.0              # Reward variation budget
  V_P: 10.0              # Transition variation budget
  V_pi_star: 5.0         # Policy variation budget
  kappa_base: 0.01       # Base trust region size
  lambda_base: 0.01      # Base regularization
  kappa_adaptive: true   # Enable adaptive trust region
  lambda_adaptive: true  # Enable adaptive regularization
  sensitivity: 0.1       # Drift detection sensitivity

paths:
  log_dir: logs/
  model_dir: models/
  video_dir: videos/
```

### **Key Parameters Explained**

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **env.parameter** | Which environment property drifts | `friction`, `mass_scale`, `gravity`, `damping` |
| **env.drift_type** | Drift pattern | `sine`, `linear`, `jump`, `random_walk` |
| **env.magnitude** | Strength of drift | 0.1 (mild) to 0.5 (extreme) |
| **nsmdmpi.V_R** | Reward variation budget | 5-20 (moderate), 20-50 (extreme) |
| **nsmdmpi.V_P** | Transition variation budget | 5-20 (moderate), 20-50 (extreme) |
| **nsmdmpi.V_pi_star** | Policy variation budget | 2.5-10 (moderate), 10-25 (extreme) |

---

## 🧪 Running Experiments

### **Option 1: Individual Scripts** (Recommended)

Run specific env-algo-type combinations:

```bash
# Moderate drift experiments
bash scripts/run_hopper_moderate_ppo.sh     # 12 experiments (6 configs × 2 methods)
bash scripts/run_halfcheetah_moderate_ppo.sh
bash scripts/run_lunarlander_moderate_ppo.sh

# Extreme drift experiments
bash scripts/run_hopper_extreme_ppo.sh      # 4 experiments (2 configs × 2 methods)
bash scripts/run_halfcheetah_extreme_ppo.sh

# Multi-parameter drift
bash scripts/run_hopper_multi_ppo.sh        # 2 experiments (1 config × 2 methods)

# Vanilla baselines (no drift)
bash scripts/run_hopper_vanilla_ppo.sh      # 1 experiment
```

### **Option 2: Custom Experiment**

```bash
python scripts/train.py --config path/to/your_config.yaml
```

### **Option 3: Batch Experiments**

```bash
# Run ALL moderate experiments (across all envs & algos)
bash scripts/run_all_moderate.sh

# Run ALL extreme experiments
bash scripts/run_all_extreme.sh
```

⚠️ **Note:** Batch scripts run many experiments in parallel. Monitor VRAM usage!

---

## 🎛️ Hyperparameter Tuning

NS-MDMPI has many hyperparameters. Use Optuna for automated tuning.

### **Install Optuna**

```bash
pip install optuna optuna-dashboard
```

### **Run Tuning**

```bash
# Tune moderate drift parameters
bash scripts/tune_moderate.sh

# Tune extreme drift parameters
bash scripts/tune_extreme.sh

# Tune multi-parameter drift
bash scripts/tune_multi.sh
```

### **Monitor Tuning Progress**

```bash
# In a separate terminal
optuna-dashboard results/optuna_studies/moderate_hopper_friction_sine.db

# Open browser: http://localhost:8080
```

### **Apply Tuned Parameters**

After tuning completes:

1. Check best parameters:
   ```bash
   cat results/tuned_params/moderate_hopper_friction_sine_best_params.yaml
   ```

2. Update your config files with the best values for `V_R`, `V_P`, `V_pi_star`, `alpha`, `beta`

See `TUNING_GUIDE.md` for detailed instructions.

---

## 📊 Results Analysis

### **View Logs**

```bash
# Real-time monitoring
tail -f logs/hopper_friction_sine_baseline_ppo.log

# Check all running experiments
watch -n 5 'ps aux | grep train.py | grep -v grep | wc -l'
```

### **WandB Dashboard**

All experiments log to WandB:
- **Moderate:** `att_3_{Env}_Moderate_Comparison`
- **Extreme:** `att_3_{Env}_Extreme_Comparison`
- **Multi:** `att_3_{Env}_Multi_Comparison`

Key metrics to watch:
- `rollout/ep_rew_mean` - Episode reward
- `nsmdmpi/V_R_remaining` - Reward budget left
- `nsmdmpi/V_P_remaining` - Transition budget left
- `nsmdmpi/kappa_t` - Current trust region size
- `nsmdmpi/lambda_t` - Current regularization

### **Analysis Scripts**

Old experiment analysis:
```bash
# Already generated in analysis/
cat analysis/RESULTS_SUMMARY.md
```

---

## 🔧 Advanced Usage

### **Creating Custom Environments**

See `src/envs/multi_env_wrappers.py` for examples.

```python
class MyCustomWrapper(MuJoCoNonStationaryWrapper):
    VALID_PARAMS = ['my_param', 'another_param']
    
    def _apply_drift(self, param, new_value):
        # Implement parameter modification
        pass
```

### **Adding New Drift Patterns**

Edit `src/envs/nonstationary_wrapper.py`:

```python
def my_custom_drift(self, timestep: int) -> float:
    # Return drift value based on timestep
    return custom_function(timestep)
```

### **Custom NS-MDMPI Variants**

Modify `src/callbacks/nsmdmpi_callback.py` to experiment with:
- Different drift estimators
- Alternative adaptation strategies
- Custom meta-learning approaches

---

## 🐛 Troubleshooting

### **Common Issues**

| Issue | Solution |
|-------|----------|
| **CUDA OOM** | Reduce `--n-jobs` in tuning or use fewer parallel experiments |
| **Pickle errors** | Already fixed! Model saves parameters (`.pt`) not callbacks |
| **Config errors** | Validate YAML syntax, check bounds min < max |
| **Slow training** | Use CPU (`device='cpu'`), reduce logging frequency |
| **WandB offline** | Set `wandb.mode: offline` in config |
| **Import errors** | `conda activate rl_hf_course` and reinstall requirements |

### **Bounds Error Example**

❌ **Wrong:**
```yaml
bounds: [0.1, 0.0]  # min > max!
```

✅ **Correct:**
```yaml
bounds: [0.0, 0.1]  # min < max
```

### **Check Logs**

```bash
# Find recent errors
grep -i "error\|exception" logs/*.log | tail -20

# Check crashed experiments
find logs -name "*.log" -size -50k
```

---

## 📁 Project Structure

```
NS_MD_MPI/
├── configs/                    # Experiment configurations
│   ├── PPO/
│   │   ├── vanilla/           # No drift (3 envs)
│   │   ├── moderate/          # Gradual drift (36 configs)
│   │   ├── extreme/           # Rapid drift (12 configs)
│   │   └── multi/             # Multi-parameter (6 configs)
│   ├── SAC/                   # SAC algorithm configs
│   └── TRPO/                  # TRPO algorithm configs
├── scripts/
│   ├── train.py               # Main training script
│   ├── tune_hyperparameters.py  # Optuna tuning
│   ├── run_*.sh               # Launcher scripts (36 total)
│   └── tune_*.sh              # Tuning launchers
├── src/
│   ├── callbacks/
│   │   └── nsmdmpi_callback.py  # NS-MDMPI implementation
│   ├── envs/
│   │   ├── nonstationary_wrapper.py  # Drift wrapper
│   │   └── multi_env_wrappers.py     # Env-specific wrappers
│   └── evaluation/
│       ├── variation_budgets.py      # Budget tracking
│       └── drift_estimators.py       # Drift detection
├── analysis/                  # Analysis results
│   ├── RESULTS_SUMMARY.md    # Old experiment analysis
│   └── figures/              # Visualizations
├── logs/                     # Training logs
├── models/                   # Saved models
├── results/
│   ├── optuna_studies/       # Tuning databases
│   └── tuned_params/         # Best hyperparameters
├── README.md                 # This file
├── TUNING_GUIDE.md          # Hyperparameter tuning guide
└── requirements.txt         # Python dependencies
```

---

## 📚 Additional Resources

- **Tuning Guide:** `TUNING_GUIDE.md`
- **Old Results:** `analysis/RESULTS_SUMMARY.md`
- **Implementation Plan:** See artifacts in `.gemini/antigravity/brain/`

---

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes
3. Test: `python scripts/test.sh`
4. Commit: `git commit -m "Add my feature"`
5. Push: `git push origin feature/my-feature`

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 📧 Contact

**Author:** Hung Tran  
**GitHub:** https://github.com/hungtrab/NS_MD_MPI

---

## 🙏 Acknowledgments

- Stable-Baselines3 team
- WandB for experiment tracking
- Optuna for hyperparameter optimization

---

**Happy Training! 🚀**
