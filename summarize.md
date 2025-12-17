# Implementation Summary: Non-Stationary MDP Research Framework

## Overview

This document summarizes the implementation of a research framework for studying Reinforcement Learning in Non-Stationary Markov Decision Processes (MDPs), based on the paper "Regularized Non-Stationary Markov Decision Processes".

---

## What Was Built

### 1. Drift Generator (`src/envs/drift_generator.py`)

A modular system for generating parameter drift patterns:

| Drift Type | Description | Use Case |
|------------|-------------|----------|
| `static` | No change | Baseline comparison |
| `jump` | Sudden regime shift | System faults, mode changes |
| `linear` | Gradual ramp (triangular wave) | Wear and tear, degradation |
| `sine` | Periodic oscillation | Seasonal/cyclical changes |
| `random_walk` | Bounded stochastic drift | Unpredictable environments |

**Key Classes:**
- `DriftConfig`: Dataclass holding drift parameters
- `DriftGenerator`: Computes parameter values at any timestep
- `create_drift_generator()`: Factory function for easy instantiation

---

### 2. Non-Stationary Environment Wrapper (`src/envs/wrappers.py`)

Wraps standard Gymnasium environments to introduce dynamics drift:

```python
drift_conf = {
    'parameter': 'gravity',
    'drift_type': 'sine',
    'magnitude': 5.0,
    'period': 10000,
}
env = NonStationaryCartPoleWrapper(gym.make('CartPole-v1'), drift_conf)
```

**Features:**
- Supports all CartPole physics parameters: `gravity`, `masscart`, `masspole`, `length`
- Tracks `total_steps` across episodes for continuous drift
- Logs drift info in step `info` dict for monitoring
- Recalculates dependent physics quantities automatically

---

### 3. Drift Estimators (`src/estimators/__init__.py`)

Three estimators for detecting non-stationarity:

| Estimator | What It Tracks | Method |
|-----------|---------------|--------|
| `RewardDriftEstimator` | $\hat{\Delta}_t^R$ | Sliding window mean comparison |
| `TransitionDriftEstimator` | $\hat{\Delta}_t^P$ | Environment drift proxy / TD errors |
| `BellmanCommutatorEstimator` | $\|\mathcal{C}_t\|$ | TD-error variance |

**Combined Interface:**
```python
estimator = CombinedDriftEstimator()
estimator.update(reward, info, value_pred, value_target)
drift = estimator.get_total_drift()
```

---

### 4. Adaptive Callbacks (`src/callbacks/drift_callback.py`)

**Algorithm-Specific Hyperparameter Adaptation:**

The callback now supports PPO, SAC, and TRPO with algorithm-specific adaptations:

| Algorithm | Hyperparameter | Formula | Intuition |
|-----------|---------------|---------|-----------|
| **All** | Learning Rate | $\eta_t = \eta_0 \cdot (1 + c \cdot \Delta_t)$ | Faster updates when drifting |
| **PPO** | Clip Range | $\epsilon_t = \epsilon_0 / (1 + c \cdot \Delta_t)$ | Smaller trust region = conservative |
| **PPO** | Entropy Coef | $\alpha_t = \alpha_0 \cdot (1 + c \cdot \Delta_t)$ | More exploration when drifting |
| **SAC** | Entropy/Temp | $\alpha_t = \alpha_0 \cdot (1 + c \cdot \Delta_t)$ | Higher entropy = more exploration |
| **TRPO** | Target KL | $KL_t = KL_0 / (1 + c \cdot \Delta_t)$ | Stricter constraint when unstable |

**Key Features:**
- Auto-detects algorithm type from model
- Configurable per-hyperparameter adaptation
- Logs all adapted values to WandB/TensorBoard

---

### 5. Oracle Evaluator (`src/evaluation/oracle.py`)

Computes ground-truth optimal performance $v_t^*$:

1. Snapshots environment parameters at timestep $t$
2. Trains fresh PPO agent on stationary environment
3. Evaluates to get optimal value
4. **Caches results** to avoid redundant training

```python
oracle = OracleEvaluator(config)
v_star, std = oracle.get_optimal_value({'gravity': 15.0})
```

---

### 6. Dynamic Regret Calculator (`src/evaluation/dynamic_regret.py`)

Computes the key metric:
$$DynReg(T) = \sum_{t=1}^{T} \rho_t [v_t^* - v_t^{\pi_t}]$$

**Features:**
- Uniform or discounted weighting
- Cumulative regret curves
- JSON save/load for analysis
- Summary statistics (total, average, max, min regret)

---

### 7. Multi-Environment Support (`src/envs/multi_env_wrappers.py`)

Extended framework to support multiple Gymnasium environments:

| Environment | Wrapper Class | Driftable Parameters |
|-------------|---------------|---------------------|
| **CartPole** | `NonStationaryCartPoleWrapper` | gravity, length, masspole, masscart |
| **MountainCar** | `NonStationaryMountainCarWrapper` | gravity, force, goal_position |
| **FrozenLake** | `NonStationaryFrozenLakeWrapper` | slip_prob, reward_goal, reward_hole |
| **MiniGrid** | `NonStationaryMiniGridWrapper` | goal_pos, agent_start, obstacles |
| **HalfCheetah** | `NonStationaryHalfCheetahWrapper` | friction, damping, head_mass, gravity |

**Factory Functions:**
```python
from src.envs import make_nonstationary_env, get_wrapper_for_env

# Automatic wrapper selection based on env_id
env = make_nonstationary_env("MountainCar-v0", drift_conf)
```

---

### 8. Multi-Algorithm Support

The framework now supports three RL algorithms:

| Algorithm | Import | Key Hyperparameters |
|-----------|--------|---------------------|
| **PPO** | `stable_baselines3.PPO` | learning_rate, clip_range, ent_coef |
| **SAC** | `stable_baselines3.SAC` | learning_rate, ent_coef, buffer_size |
| **TRPO** | `sb3_contrib.TRPO` | learning_rate, target_kl |

**Usage:**
```bash
python scripts/train.py --config configs/cartpole_adaptive.yaml --algo PPO
python scripts/train.py --config configs/mountaincar_adaptive.yaml --algo SAC
python scripts/train.py --config configs/cartpole_adaptive.yaml --algo TRPO
```

---

### 9. Benchmark Scripts

Shell scripts for automated experiments:

| Script | Description |
|--------|-------------|
| `scripts/benchmark.sh` | Full benchmark suite across all algos/envs/drifts |
| `scripts/compare_single.sh` | Quick adaptive vs baseline comparison |
| `scripts/run_all_algos.sh` | Compare PPO vs SAC vs TRPO on same env |

**Usage:**
```bash
# Quick comparison
./scripts/compare_single.sh --env mountaincar --algo PPO --drift sine

# Full benchmark
./scripts/benchmark.sh --quick

# Compare all algorithms
./scripts/run_all_algos.sh --env cartpole
```

---

### 10. Updated Scripts

#### `scripts/train.py`
- Fixed API mismatch with wrapper
- Proper drift configuration handling
- Integrated adaptive callbacks
- **Multi-algorithm support** (PPO, SAC, TRPO via `--algo` flag)
- Algorithm factory with auto-configuration

#### `scripts/eval.py`
- Standard evaluation mode
- `--regret` flag for dynamic regret computation
- Oracle integration
- **Multi-algorithm support** with auto-detection from model path

#### `scripts/render.py`
- Video recording with drift visualization
- **Multi-algorithm support** (auto-detects or specify with `--algo`)

---

### 11. Configuration System

Comprehensive YAML configuration with inline documentation:

```yaml
env:
  parameter: "gravity"     # What to drift
  drift_type: "jump"       # How it drifts
  magnitude: 10.0          # How much
  period: 30000            # How fast

adaptive:
  enabled: true            # Enable our method
  scale_factor: 0.1        # Sensitivity
  
  # PPO-specific
  adapt_clip_range: true
  base_clip_range: 0.2
  
  # Entropy adaptation (PPO/SAC)
  adapt_entropy: true
  base_ent_coef: 0.0       # 0 = auto-detect
  
  # TRPO-specific
  adapt_target_kl: true
  base_target_kl: 0.01
```

**Config files for each environment:**
- `configs/cartpole_adaptive.yaml`
- `configs/mountaincar_adaptive.yaml`
- `configs/frozenlake_adaptive.yaml`
- `configs/minigrid_adaptive.yaml`
- `configs/halfcheetah_adaptive.yaml`

---

## Project Structure

```
Deep-RL/
├── src/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── drift_generator.py      # Drift pattern generation
│   │   ├── wrappers.py             # CartPole non-stationary wrapper
│   │   └── multi_env_wrappers.py   # MountainCar, FrozenLake, MiniGrid, HalfCheetah
│   ├── callbacks/
│   │   ├── __init__.py
│   │   └── drift_callback.py       # Algorithm-specific adaptive callbacks
│   ├── estimators/
│   │   └── __init__.py             # Drift estimators
│   └── evaluation/
│       ├── __init__.py
│       ├── oracle.py               # Oracle baseline
│       └── dynamic_regret.py       # Regret computation
├── scripts/
│   ├── train.py                    # Training script (PPO/SAC/TRPO)
│   ├── eval.py                     # Evaluation script (multi-algo)
│   ├── render.py                   # Video recording (multi-algo)
│   ├── benchmark.sh                # Full benchmark suite
│   ├── compare_single.sh           # Quick adaptive vs baseline
│   └── run_all_algos.sh            # Compare all algorithms
├── configs/
│   ├── cartpole_adaptive.yaml      # CartPole config
│   ├── mountaincar_adaptive.yaml   # MountainCar config
│   ├── frozenlake_adaptive.yaml    # FrozenLake config
│   ├── minigrid_adaptive.yaml      # MiniGrid config
│   └── halfcheetah_adaptive.yaml   # HalfCheetah config
├── requirements.txt                # Dependencies (incl. sb3-contrib, mujoco)
└── CONTEXT.md                      # Project context
```

---

## Usage Examples

### Train with Different Algorithms
```bash
# PPO (default)
python scripts/train.py --config configs/cartpole_adaptive.yaml --algo PPO

# SAC
python scripts/train.py --config configs/mountaincar_adaptive.yaml --algo SAC

# TRPO
python scripts/train.py --config configs/cartpole_adaptive.yaml --algo TRPO
```

### Train on Different Environments
```bash
python scripts/train.py --config configs/cartpole_adaptive.yaml
python scripts/train.py --config configs/mountaincar_adaptive.yaml
python scripts/train.py --config configs/halfcheetah_adaptive.yaml
```

### Train Baseline (No Adaptation)
```yaml
# In config: set adaptive.enabled: false
```
```bash
python scripts/train.py --config configs/cartpole_adaptive.yaml --exp_name baseline_run
```

### Evaluate Model (Auto-detects Algorithm)
```bash
python scripts/eval.py --model models/cartpole_PPO_sine.zip --config configs/cartpole_adaptive.yaml
```

### Evaluate with Dynamic Regret
```bash
python scripts/eval.py --model models/YOUR_MODEL.zip --regret --eval-interval 5000
```

### Record Video
```bash
python scripts/render.py --model models/cartpole_PPO.zip --config configs/cartpole_adaptive.yaml
```

### Run Benchmarks
```bash
# Quick comparison (adaptive vs baseline)
./scripts/compare_single.sh --env mountaincar --algo PPO --drift sine

# Compare all algorithms
./scripts/run_all_algos.sh --env cartpole --drift jump

# Full benchmark suite
./scripts/benchmark.sh --quick
```

---

## Key Equations Implemented

| Equation | Implementation |
|----------|----------------|
| Dynamic Regret: $DynReg(T) = \sum_t [v_t^* - v_t^{\pi_t}]$ | `DynamicRegretCalculator` |
| Adaptive LR: $\eta_t = \eta_0(1 + c\|\Delta_t\|)$ | `DriftAdaptiveCallback` |
| PPO Clip: $\epsilon_t = \epsilon_0 / (1 + c\|\Delta_t\|)$ | `DriftAdaptiveCallback._adapt_ppo()` |
| SAC Entropy: $\alpha_t = \alpha_0(1 + c\|\Delta_t\|)$ | `DriftAdaptiveCallback._adapt_sac()` |
| TRPO KL: $KL_t = KL_0 / (1 + c\|\Delta_t\|)$ | `DriftAdaptiveCallback._adapt_trpo()` |
| Sinusoidal drift: $p_t = p_0 + A\sin(2\pi t/T)$ | `DriftGenerator._sinusoidal()` |
| Jump drift: $p_t = p_0 + \Delta \cdot \mathbb{1}[t > T]$ | `DriftGenerator._piecewise_constant_jump()` |

---

## Completed Features

- [x] DriftGenerator with 5 drift patterns (static, jump, linear, sine, random_walk)
- [x] CartPole non-stationary wrapper
- [x] Multi-environment wrappers (MountainCar, FrozenLake, MiniGrid, HalfCheetah)
- [x] Drift estimators (Reward, Transition, Bellman Commutator)
- [x] Algorithm-specific adaptive callbacks (PPO, SAC, TRPO)
- [x] Oracle evaluator with caching
- [x] Dynamic regret calculator
- [x] Multi-algorithm training (PPO, SAC, TRPO)
- [x] Multi-algorithm evaluation and rendering
- [x] Benchmark shell scripts
- [x] Config files for all environments

---

## Next Steps

- [ ] Run full Adaptive vs Baseline experiments across all envs
- [ ] Generate regret curve visualizations
- [ ] Hyperparameter sweep for scale_factor $c$
- [ ] Add DQN support for discrete action spaces
- [ ] Implement automatic drift detection (without oracle knowledge)


# Evaluate with auto-detection (looks at model path)
python scripts/eval.py --model models/cartpole_SAC_adaptive.zip --config configs/cartpole_adaptive.yaml

# Explicitly specify algorithm
python scripts/eval.py --model models/my_model.zip --config configs/cartpole_adaptive.yaml --algo TRPO

# Record video with SAC model
python scripts/render.py --model models/mountaincar_SAC.zip --config configs/mountaincar_adaptive.yaml --algo SAC

# Compute dynamic regret with TRPO
python scripts/eval.py --model models/cartpole_TRPO.zip --config configs/cartpole_adaptive.yaml --algo TRPO --regret


# Test CartPole with sine drift (default)
python scripts/test_env.py

# Test MountainCar with jump drift
python scripts/test_env.py --env MountainCar-v0 --drift jump

# Test with custom parameters
python scripts/test_env.py --env CartPole-v1 --param gravity --mag 8.0 --period 300

# Slower rendering (easier to see)
python scripts/test_env.py --delay 0.05

# More episodes
python scripts/test_env.py --episodes 5 --max-steps 1000

# List supported environments
python scripts/test_env.py --list-envs