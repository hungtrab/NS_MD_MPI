# Project Context: Regularized Non-Stationary MDPs Implementation

I am implementing the algorithms and experiments from the paper "Regularized Non-Stationary Markov Decision Processes" (see attached file: draft.pdf).

## 1. Core Problem
We are dealing with Non-Stationary MDPs where Transition Dynamics ($P_t$) and Reward Functions ($R_t$) drift over time.
The goal is to minimize **Dynamic Regret**: $DynReg(T) := \sum_{t=1}^{T} \rho_t [v_t^* - v_t^{\pi_t}]$.

## 2. Environment Setup ✅ IMPLEMENTED
Custom Gym Wrappers introduce drift into standard environments (CartPole).

### Drift Patterns (src/envs/drift_generator.py):
1. **static:** No drift (baseline)
2. **jump:** Piecewise-constant jumps (regime shifts)
3. **linear:** Linear ramps with triangular wave
4. **sine:** Sinusoidal periodic drift
5. **random_walk:** Bounded random walk with variance σ²

### Driftable Parameters:
- CartPole: `gravity`, `masscart`, `masspole`, `length`

## 3. Algorithms ✅ IMPLEMENTED
1. **Baselines:** PPO (using Stable Baselines 3) with fixed hyperparameters
2. **Adaptive (Ours):** `DriftAdaptiveCallback` adjusts learning rate based on drift magnitude
   - Formula: $\eta_t = \eta_0 \cdot (1 + c_1 \cdot \|\Delta_t\|)$
   - Also includes `DriftAwareClipRangeCallback` for adaptive PPO clip range

## 4. Drift Estimation ✅ IMPLEMENTED (src/estimators/)
- `RewardDriftEstimator`: Tracks reward distribution changes
- `TransitionDriftEstimator`: Tracks environment dynamics changes  
- `BellmanCommutatorEstimator`: Estimates policy staleness via TD-error variance
- `CombinedDriftEstimator`: Unified interface for all estimators

## 5. Evaluation Protocol ✅ IMPLEMENTED (src/evaluation/)
- **Oracle Evaluator:** Trains fresh agents on stationary snapshots to get $v_t^*$
- **Dynamic Regret Calculator:** Computes $DynReg(T) = \sum_t [v_t^* - v_t^{\pi_t}]$
- **Caching:** Oracle results are cached to avoid redundant training

## 6. Project Structure
```
src/
├── envs/
│   ├── drift_generator.py    # Drift pattern generators
│   └── wrappers.py           # Non-stationary wrappers
├── callbacks/
│   └── drift_callback.py     # Adaptive LR scheduling
├── estimators/
│   └── __init__.py           # Drift estimators
└── evaluation/
    ├── oracle.py             # Oracle baseline computation
    └── dynamic_regret.py     # Regret calculation

scripts/
├── train.py                  # Main training script
├── eval.py                   # Evaluation with regret computation
└── render.py                 # Video recording

configs/
└── cartpole_adaptive.yaml    # Experiment configuration
```

## 7. Usage

### Training:
```bash
python scripts/train.py --config configs/cartpole_adaptive.yaml
```

### Evaluation (standard):
```bash
python scripts/eval.py --model models/YOUR_MODEL.zip --config configs/cartpole_adaptive.yaml
```

### Evaluation (with dynamic regret):
```bash
python scripts/eval.py --model models/YOUR_MODEL.zip --regret --eval-interval 5000
```

### Record Video:
```bash
python scripts/render.py --model models/YOUR_MODEL.zip --episodes 1
```

## 8. Next Steps
- [ ] Add TRPO and SAC baselines (requires sb3-contrib)
- [ ] Implement MuJoCo wrapper (HalfCheetah friction/damping drift)
- [ ] Run comparison experiments: Adaptive vs Baseline
- [ ] Generate regret curves and visualizations
- [ ] Hyperparameter sweep for scale_factor (c₁)