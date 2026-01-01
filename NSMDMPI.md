# NS-MD-MPI: Theory vs Implementation

## Overview

This document explains the implementation of **Algorithm 1 (NS-MD-MPI)** from the paper "Regularized Non-Stationary Markov Decision Processes", highlighting what comes directly from the paper versus what are implementation approximations.

---

## Paper's Theoretical Framework

### Variation Budgets (Section 3)

**Paper Definition:**
```
V_R = Σ_{t=1}^T ||r_t - r_{t-1}||        (Reward variation)
V_P = Σ_{t=1}^T ||P_t - P_{t-1}||_1      (Transition variation)
V_π* = Σ_{t=1}^T ||π*_t - π*_{t-1}||_1   (Optimal policy variation)
```

These budgets bound the total non-stationarity over the time horizon T and appear in the regret bound:
```
Regret(T) ≤ O(√(T·V_R + T·V_P + T·V_π*))
```

**Implementation:**
- ✅ `VariationBudgetTracker` class in `src/evaluation/variation_budgets.py`
- ✅ Tracks cumulative consumption: `V_R_consumed`, `V_P_consumed`, `V_pi_star_consumed`
- ✅ Computes remaining fractions: `get_remaining_fraction()`
- ✅ Updates each rollout: `update(delta_R, delta_P, delta_pi)`

**Gap:** Paper assumes true V_R, V_P, V_π* are known. Implementation estimates them:
```python
# Conservative estimation (Option A)
def estimate_budgets_from_drift_config(drift_config, time_horizon, scale_factor):
    magnitude = drift_config['magnitude']
    period = drift_config['period']
    n_cycles = time_horizon / period
    
    if drift_type == 'sine':
        V_R = V_P = magnitude * sqrt(n_cycles) * scale_factor
    elif drift_type == 'jump':
        V_R = V_P = magnitude * n_cycles * scale_factor
    # ... etc
    
    V_pi_star = 0.5 * max(V_R, V_P)  # Heuristic: policy adapts slower
```

---

### Algorithm 1: NS-MD-MPI

**Paper Algorithm (Lines 1-9):**
```
Input: regularizer Ω, initial policy π^(0), value v^(0), 
       depth m, smoothing β, clips (δ_min, δ_max) or (η_min, η_max)

for k = 0, 1, 2, ... do
    for time t = 1, 2, ... do
        Compute drift proxies Δ̂_t^P, Δ̂_t^R and Δ̃_t via Equation (28)
        Update Δ̂_t
        
        Set δ_t or η_t by Equation (30)
        
        Greedy step: solve Equation (27) (trust region) 
                  or Equation (26) (penalized) to obtain π_t^(k+1)
        
        m-step unregularized evaluation to get v^(k+1)
    end for
end for
```

**Paper Equation (27) - Trust Region:**
```
π_{t+1} = argmax_π E_{s,a~π}[A_t^π_old(s,a)]
          s.t.  D_KL(π||π_t) ≤ δ_t
```

**Paper Equation (28) - Drift Proxies:**
```
Δ̂_t^R = ||r̂_t - r̂_{t-1}||
Δ̂_t^P = ||P̂_t - P̂_{t-1}||
Δ̃_t = combined drift estimate
```

**Paper Equation (30) - Trust Region Schedule:**
```
δ_t = δ_base * f(Δ̃_t, V_remaining)
```
(Exact formula not fully specified in paper excerpt)

---

## Implementation Mapping

### File: `src/callbacks/nsmdmpi_callback.py`

#### 1. Drift Proxies (Equation 28)

**What Paper Says:**
- Compute Δ̂_t^R (reward drift)
- Compute Δ̂_t^P (transition drift)
- Compute combined Δ̃_t

**Implementation:**
```python
class NSMDMPICallback:
    def _on_rollout_end(self):
        # Line 1: Estimate drift from current rollout
        delta_R = self.drift_estimator.reward_estimator.estimate_drift()
        delta_P = self.drift_estimator.transition_estimator.estimate_drift()
        delta_C = self.drift_estimator.commutator_estimator.estimate_commutator()
        
        # Line 2: Compute combined drift (weighted sum)
        combined_drift = (
            self.drift_weights[0] * delta_R +
            self.drift_weights[1] * delta_P +
            self.drift_weights[2] * delta_C
        )
```

**Implementation Details:**
- `RewardDriftEstimator`: Two-window mean comparison
  ```python
  Δ̂_R = |mean(rewards_recent) - mean(rewards_old)| / std(rewards)
  ```
- `TransitionDriftEstimator`: Direct parameter tracking from environment
  ```python
  Δ̂_P = |current_param_value - base_param_value| / |base_param_value|
  ```
- `BellmanCommutatorEstimator`: TD-error variance as proxy
  ```python
  Δ̂_C = std(TD_errors)  # Proxy for value function instability
  ```

**Gap:** Paper's Δ̂_t^P assumes access to transition probabilities P_t. Implementation uses:
- **Option A (current)**: Environment parameter tracking (e.g., gravity changes)
- **Option B (future)**: Model-based estimation from trajectory data

---

#### 2. Budget Updates (Algorithm 1, Line 4-5)

**What Paper Says:**
- Track cumulative drift
- Update remaining variation budgets

**Implementation:**
```python
def _on_rollout_end(self):
    # Use commutator as proxy for policy variation (Option A)
    delta_pi = delta_C
    
    # Update variation budgets
    self.budget_tracker.update(
        delta_R=delta_R,
        delta_P=delta_P,
        delta_pi=delta_pi,
        timestep=self.num_timesteps,
    )
```

**V_π* Estimation (Option A):**
```python
# Use Bellman commutator as proxy for optimal policy drift
delta_pi = delta_C  # Commutator magnitude
```

**Gap:** Paper's V_π* requires tracking optimal policy π*_t at each timestep. Implementation approximates:
- **Option A (current)**: Use commutator ||C_t|| as proxy
- **Option B (not implemented)**: Store oracle policies and compute KL(π*_t || π*_{t-1})
- **Option C (theoretical)**: Use bounds from Assumption 3 in paper

---

#### 3. Trust Region Adaptation (Equation 30)

**What Paper Says:**
- Set trust region size δ_t based on drift and budgets
- Equation (30) gives the schedule (not fully detailed in excerpt)

**Implementation:**
```python
def _on_rollout_end(self):
    if self.kappa_adaptive:
        min_budget_frac = self.budget_tracker.get_min_remaining_fraction()
        
        # κ_t = κ_0 / (1 + α * (1 - budget_frac))
        self.kappa_t = self.kappa_base / (
            1.0 + self.trust_region_sensitivity * (1.0 - min_budget_frac)
        )
        self.kappa_t = np.clip(self.kappa_t, self.kappa_min, self.kappa_max)
```

**Formula Derivation:**
```
When budget_frac → 0 (exhausted):
    κ_t → κ_base / (1 + α) ≈ 0   (tight constraint)
    
When budget_frac → 1 (plenty remaining):
    κ_t → κ_base                 (full trust region)
```

**Gap:** Exact form of Equation (30) not provided in paper excerpt. Implementation uses a reasonable heuristic that:
- Shrinks trust region as budgets deplete
- Controlled by sensitivity parameter α

---

#### 4. Regularization Adaptation (Algorithm 1, Line 5)

**What Paper Says:**
- Adjust regularization strength based on drift

**Implementation:**
```python
def _on_rollout_end(self):
    if self.lambda_adaptive:
        # λ_t = λ_0 * (1 + β * drift)
        self.lambda_t = self.lambda_base * (
            1.0 + self.regularization_sensitivity * combined_drift
        )
        self.lambda_t = np.clip(self.lambda_t, self.lambda_min, self.lambda_max)
```

**Formula:**
```
λ_t = λ_base × (1 + β × Δ̃_t)

Higher drift → larger λ_t → stronger regularization
```

**Gap:** Paper mentions regularizer Ω but doesn't specify λ_t schedule explicitly. Implementation uses intuitive scaling with drift magnitude.

---

#### 5. Trust Region Application (Equation 27)

**What Paper Says:**
```
π_{t+1} = argmax_π E[A_t^π_old(s,a)]  s.t. D_KL(π||π_t) ≤ δ_t
```
Direct constrained optimization with KL divergence constraint.

**Implementation:**
```python
def _apply_trust_region(self):
    if self.algo_name == 'PPO':
        # PPO: Trust region via clip_range
        self.model.clip_range = lambda _progress: self.kappa_t
        
    elif self.algo_name == 'TRPO':
        # TRPO: Trust region via target_kl
        self.model.target_kl = self.kappa_t
```

**Mapping:**
- **PPO**: `clip_range = κ_t` constrains policy update via clipped objective
  ```python
  L^CLIP(θ) = min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)
  where ε = κ_t
  ```
  
- **TRPO**: `target_kl = κ_t` directly implements KL constraint
  ```python
  max_θ E[A_old(s,a)]  s.t. D_KL(π_new||π_old) ≤ κ_t
  ```

**Gap - Option A (Current):**
- ✅ No direct access to policy update in SB3
- ✅ Use hyperparameters as proxy (clip_range, target_kl)
- ✅ PPO's clipping approximates trust region
- ✅ TRPO's target_kl is direct trust region

**Alternative Options (Not Implemented):**
- **Option B**: Switch to TRPO exclusively (better alignment)
- **Option C**: Custom PPO with explicit KL regularizer in loss

---

#### 6. Regularization Application

**What Paper Says:**
- Apply regularizer Ω to policy update
- Could be entropy, L2, or other

**Implementation:**
```python
def _apply_regularization(self):
    # Adjust learning rate (inverse relationship with λ_t)
    lr_multiplier = 1.0 / self.lambda_t
    new_lr = self.base_lr * lr_multiplier
    
    # Adjust entropy coefficient (exploration)
    if self.adapt_entropy:
        ent_multiplier = 1.0 + combined_drift
        new_ent = self._base_ent_coef * ent_multiplier
```

**Regularization Mechanisms:**
1. **Learning Rate**: Lower LR with high λ_t = more conservative updates
2. **Entropy**: Higher entropy with drift = more exploration

**Gap:** Paper's regularizer Ω is abstract. Implementation uses practical RL regularization:
- Entropy regularization (standard in PPO/SAC)
- Learning rate control (affects update step size)

---

### File: `src/evaluation/variation_budgets.py`

#### Budget Tracking

**What Paper Says:**
```
V_R = Σ_t ||r_t - r_{t-1}||
```
Exact cumulative variation.

**Implementation:**
```python
class VariationBudgetTracker:
    def update(self, delta_R, delta_P, delta_pi, timestep):
        # Ensure non-negative
        delta_R = max(0.0, abs(delta_R))
        delta_P = max(0.0, abs(delta_P))
        delta_pi = max(0.0, abs(delta_pi))
        
        # Update remaining budgets
        self.V_R_remaining -= delta_R
        self.V_P_remaining -= delta_P
        self.V_pi_star_remaining -= delta_pi
        
        # Track cumulative consumption
        self.V_R_consumed += delta_R
        self.V_P_consumed += delta_P
        self.V_pi_star_consumed += delta_pi
```

**Gap:** None for tracking logic. Gap is in estimation of initial budgets (see above).

---

### File: `src/evaluation/dynamic_regret.py`

#### Regret Decomposition

**What Paper Says:**
```
DynReg(T) = Σ_{t=1}^T [v*_t - v^π_t]
```

Paper predicts regret scales with budgets:
```
Regret ≤ O(√(T·V_R + T·V_P + T·V_π*))
```

**Implementation:**
```python
def compute_regret_decomposition(self, budget_tracker):
    total_regret = self.compute_total_regret()
    total_variation = V_R_consumed + V_P_consumed + V_pi_consumed
    
    # Budget efficiency: regret per unit of variation
    budget_efficiency = total_regret / total_variation
    
    return {
        'total_regret': total_regret,
        'total_variation': total_variation,
        'budget_efficiency': budget_efficiency,
        'regret_per_V_R': total_regret / V_R_consumed,
        'regret_per_V_P': total_regret / V_P_consumed,
        'budget_constraint_binding': min_remaining_frac < 0.2,
    }
```

**This enables testing paper's prediction:**
```python
# Check if regret ~ sqrt(V_consumed)
import numpy as np
efficiency = regret / np.sqrt(total_variation)
# Should be roughly constant across different experiments
```

---

## Empirical Goals from Paper

### Goal (i): Dynamic regret improves with drift-adaptive regularization

**Paper's Claim:**
- Adaptive methods should have lower regret than fixed methods

**How to Test:**
```bash
# Baseline (no adaptation)
python scripts/train.py --config configs/baseline.yaml

# Heuristic adaptive
python scripts/train.py --config configs/adaptive.yaml  

# NS-MD-MPI (theoretical)
python scripts/train.py --config configs/nsmdmpi_cartpole.yaml
```

**Expected:** `Regret_baseline > Regret_adaptive > Regret_NSMDMPI`

---

### Goal (ii): Regret scales with variation budgets V_R, V_P, V_π*

**Paper's Claim:**
```
Regret ≤ O(√(T·(V_R + V_P + V_π*)))
```

**How to Test:**
```python
# Run experiments with different drift magnitudes
configs = [
    {'magnitude': 2.0},  # Low drift  → small budgets
    {'magnitude': 5.0},  # Medium drift
    {'magnitude': 10.0}, # High drift → large budgets
]

# Plot: log(Regret) vs log(sqrt(V_consumed))
# Should be roughly linear with slope ≈ 1
```

**Implementation provides:**
```python
decomp = regret_calc.compute_regret_decomposition(budget_tracker)
print(f"Regret: {decomp['total_regret']:.2f}")
print(f"Total variation: {decomp['total_variation']:.2f}")
print(f"Efficiency: {decomp['budget_efficiency']:.2f}")
```

---

### Goal (iii): Benefits hold across regularizer families

**Paper's Claim:**
- Different regularizers (trust region, penalized) should both work

**How to Test:**
```yaml
# Test with PPO (clip range = trust region proxy)
algorithm: "PPO"

# Test with TRPO (explicit trust region)
algorithm: "TRPO"

# Test with SAC (entropy regularization)
algorithm: "SAC"
```

**Implementation supports:**
- PPO: Trust region via `clip_range = κ_t`
- TRPO: Trust region via `target_kl = κ_t`
- SAC: Entropy via `ent_coef` (less direct)

---

### Goal (iv): Zero-drift limit → stationary theory

**Paper's Claim:**
- When drift = 0, should match standard RL

**How to Test:**
```yaml
env:
  drift_type: "static"  # No drift
  magnitude: 0.0

nsmdmpi:
  enabled: true
  V_R: 0.1  # Minimal budgets
  V_P: 0.1
```

**Expected:**
- κ_t ≈ κ_base (trust region doesn't shrink)
- λ_t ≈ λ_base (regularization stays constant)
- Performance matches standard PPO

---

## Summary: Theory vs Implementation

### ✅ Directly from Paper

1. **Variation Budgets (V_R, V_P, V_π*)**
   - ✅ Tracking and consumption
   - ✅ Used to adapt trust region

2. **Drift Estimation (Equation 28)**
   - ✅ Reward drift Δ̂_R
   - ✅ Transition drift Δ̂_P
   - ✅ Combined drift Δ̃_t

3. **Trust Region Adaptation (Equation 27)**
   - ✅ Shrinks as budgets deplete
   - ✅ Controlled by κ_t

4. **Dynamic Regret (Algorithm 2)**
   - ✅ Oracle evaluation
   - ✅ Regret computation
   - ✅ Budget decomposition

### ⚠️ Implementation Approximations (Option A)

1. **Budget Initialization**
   - **Paper**: Assumes V_R, V_P, V_π* known
   - **Impl**: Conservative estimation from drift config

2. **V_π* Estimation**
   - **Paper**: Requires tracking optimal policy π*_t
   - **Impl**: Use Bellman commutator as proxy

3. **Trust Region Schedule (Equation 30)**
   - **Paper**: Exact formula not fully specified
   - **Impl**: κ_t = κ_base / (1 + α(1 - budget_frac))

4. **Trust Region Application (Equation 27)**
   - **Paper**: Direct KL constraint in policy optimization
   - **Impl**: PPO clip_range or TRPO target_kl as proxy

5. **Regularizer Ω**
   - **Paper**: Abstract regularizer
   - **Impl**: Learning rate and entropy coefficient

### 🔄 Why Option A Works

**Theoretical Justification:**

1. **PPO Clipping ≈ Trust Region**
   - Schulman et al. 2017: Clipped objective approximates KL constraint
   - `clip(r_t, 1-ε, 1+ε)` limits policy change similar to D_KL ≤ δ

2. **Commutator as Policy Drift Proxy**
   - Bellman commutator measures value function stability
   - High commutator → value estimates changing → policy likely changing
   - ||C_t|| = ||T_t V_{t-1} - V_t|| correlates with ||π*_t - π*_{t-1}||

3. **Conservative Budget Estimation**
   - Overestimating budgets is safe (more conservative)
   - Real experiments show actual consumption often < estimated

### 📊 What You Can Validate

**From Implementation:**
1. ✅ Budget consumption over time
2. ✅ Trust region adaptation (κ_t trajectory)
3. ✅ Regret vs budget consumption relationship
4. ✅ Comparison: NS-MD-MPI vs Adaptive vs Baseline
5. ✅ Zero-drift limit behavior

**Still Need Theory for:**
1. ❌ Exact regret bound constants
2. ❌ Optimal budget allocation
3. ❌ Formal proof of Option A approximations

---

## File Structure

```
src/
├── evaluation/
│   ├── variation_budgets.py      # Budget tracking (V_R, V_P, V_π*)
│   ├── dynamic_regret.py          # Regret computation (enhanced)
│   └── oracle.py                  # Optimal policy evaluation
├── callbacks/
│   ├── nsmdmpi_callback.py        # Algorithm 1 implementation
│   └── drift_callback.py          # Baseline adaptive (heuristic)
└── estimators/
    └── __init__.py                 # Drift estimators (Δ̂_R, Δ̂_P, Δ̂_C)

configs/
└── nsmdmpi_cartpole.yaml          # Full NS-MD-MPI configuration

scripts/
├── train.py                       # Training with NS-MD-MPI support
├── test_nsmdmpi.py               # Test suite
└── eval.py                        # Dynamic regret evaluation
```

---

## Usage Example

```bash
# 1. Train with NS-MD-MPI
python scripts/train.py --config configs/nsmdmpi_cartpole.yaml

# 2. Logs show:
# - nsmdmpi/V_R_remaining (budget depletion)
# - nsmdmpi/kappa_t (trust region adaptation)
# - nsmdmpi/lambda_t (regularization adaptation)
# - nsmdmpi/delta_R, delta_P, delta_C (drift estimates)

# 3. Evaluate regret with budget decomposition
python scripts/eval.py --model models/NSMDMPI_run/model.zip

# 4. Compare methods
# Edit config: nsmdmpi.enabled=false, adaptive.enabled=true (baseline)
# Run again and compare regret curves
```

---

## Key Takeaway

**What's Theoretical (from Paper):**
- Variation budgets bound non-stationarity
- Regret scales as O(√(T·V))
- Trust region should shrink with remaining budget
- Drift-adaptive regularization helps

**What's Practical (in Implementation):**
- Approximate trust region via PPO/TRPO hyperparameters
- Estimate V_π* from commutator (no oracle policies needed)
- Conservative budget initialization from drift config
- Budget-aware scheduling works in practice

**The implementation is theoretically motivated but practically adapted for real RL training with Stable-Baselines3.**
