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

Two callbacks for drift-adaptive training:

#### DriftAdaptiveCallback
Adjusts learning rate based on drift:
$$\eta_t = \eta_0 \cdot (1 + c_1 \cdot \|\Delta_t\|)$$

- Large drift → Larger learning rate (faster adaptation)
- Small drift → Smaller learning rate (stable policy)

#### DriftAwareClipRangeCallback
Adjusts PPO clip range inversely:
$$\epsilon_t = \epsilon_0 / (1 + c_2 \cdot \|\Delta_t\|)$$

- Large drift → Smaller clip (conservative updates)
- Small drift → Larger clip (aggressive updates)

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

### 7. Updated Scripts

#### `scripts/train.py`
- Fixed API mismatch with wrapper
- Proper drift configuration handling
- Integrated adaptive callbacks

#### `scripts/eval.py`
- Standard evaluation mode
- `--regret` flag for dynamic regret computation
- Oracle integration

---

### 8. Configuration System

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
```

---

## Project Structure

```
Deep-RL/
├── src/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── drift_generator.py    # Drift pattern generation
│   │   └── wrappers.py           # Non-stationary wrappers
│   ├── callbacks/
│   │   ├── __init__.py
│   │   └── drift_callback.py     # Adaptive LR/clip callbacks
│   ├── estimators/
│   │   └── __init__.py           # Drift estimators
│   └── evaluation/
│       ├── __init__.py
│       ├── oracle.py             # Oracle baseline
│       └── dynamic_regret.py     # Regret computation
├── scripts/
│   ├── train.py                  # Training script
│   ├── eval.py                   # Evaluation script
│   └── render.py                 # Video recording
├── configs/
│   └── cartpole_adaptive.yaml    # Experiment config
├── requirements.txt              # Dependencies
└── CONTEXT.md                    # Project context
```

---

## Usage Examples

### Train Baseline (No Adaptation)
```bash
python scripts/train.py --config configs/cartpole_adaptive.yaml
```

### Train Adaptive Agent
```yaml
# In config: set adaptive.enabled: true
```
```bash
python scripts/train.py --config configs/cartpole_adaptive.yaml
```

### Evaluate with Dynamic Regret
```bash
python scripts/eval.py --model models/YOUR_MODEL.zip --regret --eval-interval 5000
```

---

## Key Equations Implemented

| Equation | Implementation |
|----------|----------------|
| Dynamic Regret: $DynReg(T) = \sum_t [v_t^* - v_t^{\pi_t}]$ | `DynamicRegretCalculator` |
| Adaptive LR: $\eta_t = \eta_0(1 + c_1\|\Delta_t\|)$ | `DriftAdaptiveCallback` |
| Sinusoidal drift: $p_t = p_0 + A\sin(2\pi t/T)$ | `DriftGenerator._sinusoidal()` |
| Jump drift: $p_t = p_0 + \Delta \cdot \mathbb{1}[t > T]$ | `DriftGenerator._piecewise_constant_jump()` |

---

## Next Steps

- [ ] Add TRPO and SAC baselines
- [ ] Implement MuJoCo wrapper (HalfCheetah)
- [ ] Run Adaptive vs Baseline experiments
- [ ] Generate regret curve visualizations
- [ ] Hyperparameter sweep for $c_1$
