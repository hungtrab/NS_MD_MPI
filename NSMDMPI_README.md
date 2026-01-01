# NS-MD-MPI Implementation

Implementation of **Algorithm 1 (NS-MD-MPI)** from the paper "Regularized Non-Stationary Markov Decision Processes" with variation budget tracking and adaptive trust region optimization.

## 📁 New Files Created

### 1. Core Implementation
- **`src/evaluation/variation_budgets.py`** - Variation budget tracker (V_R, V_P, V_π*)
  - `VariationBudgetTracker`: Tracks and depletes budgets as drift is observed
  - `estimate_budgets_from_drift_config()`: Auto-estimate budgets from drift configuration
  
- **`src/callbacks/nsmdmpi_callback.py`** - NS-MD-MPI callback implementing Algorithm 1
  - Adaptive trust region κ_t based on remaining budgets
  - Adaptive regularization λ_t based on drift magnitude
  - Budget-aware hyperparameter scheduling
  - Integration with PPO/SAC/TRPO

### 2. Configuration
- **`configs/nsmdmpi_cartpole.yaml`** - Complete NS-MD-MPI configuration template
  - Variation budget parameters (V_R, V_P, V_π*)
  - Trust region settings (κ_base, sensitivity α)
  - Regularization settings (λ_base, sensitivity β)
  - Drift estimation weights
  - Comparison mode flags

### 3. Testing
- **`scripts/test_nsmdmpi.py`** - Test suite for NS-MD-MPI implementation
  - Budget tracker functionality tests
  - Budget auto-estimation tests
  - Callback initialization tests
  - Environment integration tests

## 🔧 Modified Files

### Core Modules
- **`src/evaluation/dynamic_regret.py`**
  - Added `compute_regret_decomposition()` method
  - Budget efficiency metrics
  - Regret attribution analysis

- **`src/evaluation/__init__.py`**
  - Export `VariationBudgetTracker`, `VariationBudgetConfig`
  - Export `estimate_budgets_from_drift_config`

- **`src/callbacks/__init__.py`**
  - Export `NSMDMPICallback`

### Training Scripts
- **`scripts/train.py`**
  - Import `NSMDMPICallback`
  - Check for `nsmdmpi.enabled` in config
  - Instantiate NS-MD-MPI callback with all parameters
  - Add "_NSMDMPI" suffix to run name
  - Fallback to adaptive/baseline modes

## 🚀 Usage

### 1. Run NS-MD-MPI Experiment
```bash
python scripts/train.py --config configs/nsmdmpi_cartpole.yaml
```

### 2. Test Implementation
```bash
python scripts/test_nsmdmpi.py
```

### 3. Compare Methods

**Baseline (no adaptation):**
```yaml
nsmdmpi:
  enabled: false
adaptive:
  enabled: false
```

**Heuristic Adaptive:**
```yaml
nsmdmpi:
  enabled: false
adaptive:
  enabled: true
```

**NS-MD-MPI (Algorithm 1):**
```yaml
nsmdmpi:
  enabled: true
```

## 📊 Key Concepts

### Variation Budgets
Bound total environment non-stationarity:
- **V_R**: Cumulative reward function changes
- **V_P**: Cumulative transition dynamics changes
- **V_π\***: Cumulative optimal policy changes

### Trust Region Adaptation
```
κ_t = κ_base / (1 + α × (1 - min_budget_fraction))
```
- Shrinks as budgets are consumed
- More conservative updates when budgets low
- Prevents overfitting to transient states

### Regularization Adaptation
```
λ_t = λ_base × (1 + β × combined_drift)
```
- Increases with detected drift
- Penalizes large policy changes
- Smooths learning in non-stationary environments

### Drift Estimation
```
Δ̂_t = w_R × Δ̂_R + w_P × Δ̂_P + w_C × Δ̂_C
```
- **Δ̂_R**: Reward drift (statistical estimation)
- **Δ̂_P**: Transition drift (parameter tracking)
- **Δ̂_C**: Policy drift proxy (Bellman commutator)

## 🎯 Empirical Goals (from Paper)

The implementation tests the following predictions:

1. **Dynamic regret improves** with drift-adaptive regularization
2. **Regret scales** with variation budgets V_R, V_P, V_π*
3. **Benefits hold** across regularizer families (PPO clip_range, TRPO target_kl)
4. **Zero-drift limit**: Results collapse to stationary theory

## 📈 Logged Metrics

NS-MD-MPI logs the following to WandB/TensorBoard:

### Budget Metrics
- `nsmdmpi/V_R_remaining`, `nsmdmpi/V_P_remaining`, `nsmdmpi/V_pi_remaining`
- `nsmdmpi/V_R_fraction`, `nsmdmpi/V_P_fraction`, `nsmdmpi/V_pi_fraction`
- `nsmdmpi/min_budget_fraction`

### Adaptation Metrics
- `nsmdmpi/kappa_t` - Current trust region size
- `nsmdmpi/lambda_t` - Current regularization coefficient

### Drift Estimates
- `nsmdmpi/delta_R` - Reward drift
- `nsmdmpi/delta_P` - Transition drift
- `nsmdmpi/delta_C` - Commutator (policy drift proxy)

### Hyperparameters
- `nsmdmpi/learning_rate`
- `nsmdmpi/clip_range` (PPO)
- `nsmdmpi/target_kl` (TRPO)
- `nsmdmpi/ent_coef` (PPO/SAC)

## 🔬 Experiment Variants

### 1. Zero-Drift Baseline
```yaml
env:
  drift_type: "static"
nsmdmpi:
  enabled: false
```
Expected: Matches stationary PPO performance

### 2. NS-MD-MPI with Sine Drift
```yaml
env:
  drift_type: "sine"
  magnitude: 5.0
nsmdmpi:
  enabled: true
  V_R: 25.0
  V_P: 25.0
```
Expected: Low regret, efficient budget usage

### 3. NS-MD-MPI with Jump Drift
```yaml
env:
  drift_type: "jump"
  magnitude: 10.0
nsmdmpi:
  enabled: true
```
Expected: Trust region adapts to sudden change

### 4. Budget Exhaustion Test
```yaml
env:
  magnitude: 10.0  # High drift
nsmdmpi:
  V_R: 10.0  # Small budget
  V_P: 10.0
```
Expected: κ_t → 0, very conservative updates

### 5. Comparison: NS-MD-MPI vs Adaptive
Run both configurations and compare:
- Dynamic regret curves
- Budget utilization efficiency
- Hyperparameter evolution
- Final performance

## 📝 Implementation Details

### Option A Choices (as requested)
1. **Policy Update Regularization**: Use `clip_range` (PPO) / `target_kl` (TRPO) as trust region proxy
2. **V_π* Estimation**: Use Bellman commutator as proxy for policy variation
3. **Budget Initialization**: Conservative estimation from `drift_magnitude × cycles × scale_factor`

### Algorithm Mapping
- **PPO**: Trust region via `clip_range = κ_t`
- **TRPO**: Trust region via `target_kl = κ_t`
- **All**: Regularization via learning rate and entropy coefficient

### Budget Auto-Estimation
For drift type with magnitude M, period P, horizon T:
- **Static**: V_R = V_P = 0.1
- **Jump**: V_R = V_P = M × (T/P)
- **Linear**: V_R = V_P = 0.5 × M × (T/P)
- **Sine/Random**: V_R = V_P = M × √(T/P)
- **V_π\***: 0.5 × max(V_R, V_P)

## 🔍 Evaluation

Use the regret decomposition method:
```python
from src.evaluation import DynamicRegretCalculator

regret_calc = DynamicRegretCalculator()
# ... add oracle and policy values ...

# Get budget-aware decomposition
decomposition = regret_calc.compute_regret_decomposition(budget_tracker)
print(decomposition['budget_efficiency'])  # Regret per unit variation
print(decomposition['budget_constraint_binding'])  # Were budgets exhausted?
```

## 📚 References

- Paper: "Regularized Non-Stationary Markov Decision Processes"
- Algorithm 1: NS-MD-MPI with Drift-Adaptive Trust Region
- Algorithm 2: Estimating Dynamic Regret (Oracle/Fitted-Model)

## ✅ Status

- [x] VariationBudgetTracker implementation
- [x] NSMDMPICallback implementation  
- [x] DynamicRegretCalculator enhancement
- [x] Config template creation
- [x] Training script integration
- [x] Module exports updated
- [x] Test suite created
- [ ] Run experiments and validate empirical goals
- [ ] Compare NS-MD-MPI vs Adaptive vs Baseline
- [ ] Analyze budget efficiency and regret scaling
