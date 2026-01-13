# Hyperparameter Tuning Guide for NS-MDMPI

This guide shows you how to systematically find optimal hyperparameters for your NS-MDMPI experiments.

---

## 🎯 Why Tune?

**Analysis of old experiments showed NS-MDMPI performed WORSE than baseline!**

Likely reasons:
- Default params (V_R=10, V_P=10, V_π*=5) are too conservative
- Trust region/regularization not optimized
- Sub optimal drift detection sensitivity

**Tuning can dramatically improve performance!**

---

## 📋 Step-by-Step Tuning Process

### **Step 1: Install Optuna**

```bash
conda activate rl_hf_course
pip install optuna optuna-dashboard
```

### **Step 2: Choose What to Tune**

Start with **one environment and drift type** (faster iteration):

```bash
# Recommended: Start with Hopper Moderate
bash scripts/tune_moderate.sh
```

This will:
1. Run 5 validation trials (~30 min)
2. Ask for confirmation
3. Run 50 full trials (~6-8 hours)

### **Step 3: Monitor Progress**

**Option A: Real-time Dashboard**

```bash
# In a separate terminal
optuna-dashboard results/optuna_studies/moderate_hopper_friction_sine.db

# Open browser: http://localhost:8080
```

**Option B: Check study status**

```python
import optuna

study = optuna.load_study(
    study_name="moderate_hopper_friction_sine",
    storage="sqlite:///results/optuna_studies/moderate_hopper_friction_sine.db"
)

print(f"Trials completed: {len(study.trials)}")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
```

### **Step 4: Review Results**

After tuning completes, check best parameters:

```bash
cat results/tuned_params/moderate_hopper_friction_sine_best_params.yaml
```

Example output:
```yaml
V_R: 15.2
V_P: 18.7
V_pi_star: 7.3
alpha: 6.8        # trust_region_sensitivity
beta: 3.2         # regularization_sensitivity
max_ent_coef: 0.065
drift_window: 1200
```

### **Step 5: Apply to Configs**

Update your NS-MDMPI configs with best values:

```bash
# Edit the config
nano configs/PPO/moderate/hopper_friction_sine_nsmdmpi_ppo.yaml
```

Update the `nsmdmpi` section:
```yaml
nsmdmpi:
  enabled: true
  V_R: 15.2          # ← Updated!
  V_P: 18.7          # ← Updated!
  V_pi_star: 7.3     # ← Updated!
  # ... other params
```

### **Step 6: Validate**

Run a validation experiment with tuned params:

```bash
python scripts/train.py \
  --config configs/PPO/moderate/hopper_friction_sine_nsmdmpi_ppo.yaml
```

Compare results on WandB!

---

## 🎛️ What Gets Tuned?

### **Moderate Drift Search Space**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `V_R` | 5.0 - 20.0 | Reward budget |
| `V_P` | 5.0 - 20.0 | Transition budget |
| `V_pi_star` | 2.5 - 10.0 | Policy budget |
| `kappa_base` | 0.1 - 0.3 | Base trust region |
| `alpha` | 2.0 - 10.0 | Trust region sensitivity |
| `lambda_base` | 0.5 - 2.0 | Base regularization |
| `beta` | 1.0 - 5.0 | Regularization sensitivity |
| `max_ent_coef` | 0.01 - 0.1 | Max entropy |
| `drift_window` | 500 - 2000 | Drift detection window |

### **Extreme Drift Search Space**

Higher ranges for more aggressive adaptation:

| Parameter | Range | Notes |
|-----------|-------|-------|
| `V_R` | 20.0 - 50.0 | Higher budgets |
| `V_P` | 20.0 - 50.0 | Higher budgets |
| `V_pi_star` | 10.0 - 25.0 | Higher budgets |
| `alpha` | 5.0 - 20.0 | More sensitive |
| `beta` | 3.0 - 10.0 | Stronger regularization |

---

## ⚡ Speed Tips

### **Quick Tuning (for testing)**

```bash
python scripts/tune_hyper parameters.py \
  --env "Hopper-v4" \
  --config "configs/PPO/moderate/hopper_friction_sine_baseline_ppo.yaml" \
  --type moderate \
  --n-trials 10 \      # Fewer trials
  --n-jobs 4 \         # Parallel
  --quick \            # 100k timesteps instead of 1M
  --study-name "test_moderate"
```

### **Full Tuning (production)**

```bash
# Use the launcher script
bash scripts/tune_moderate.sh
```

### **Resume Interrupted Tuning**

Optuna studies auto-save, so just re-run:

```bash
bash scripts/tune_moderate.sh
# Will continue from where it left off!
```

---

## 📊 Understanding Results

### **Optimization Metric**

The tuning maximizes a weighted score:

```python
# For moderate drift
score = (
    0.4 * mean_reward / 1000.0 +      # Normalize reward
    0.3 * budget_remaining_avg +       # Prefer not exhausting budgets
    0.2 * (1.0 - abs(budget_remaining_avg - 0.7))  # Target ~70% remaining
)
```

Higher score = better!

### **Key Metrics to Watch**

1. **Best Value**: Highest score achieved
2. **Best Trial**: Which trial number was best
3. **Convergence**: Are scores improving or plateauing?
4. **Budget Efficiency**: Are budgets being used wisely?

---

## 🔄 Transfer Learning Approach

**Save time by tuning once and transferring!**

1. Tune thoroughly on **Hopper Moderate** (~8 hours)
2. Apply same V_R, V_P, V_π*, α, β to:
   - Other Hopper configs (friction_linear, mass_sine, etc.)
   - Similar environments (HalfCheetah, Walker2D)
3. Only fine-tune if needed

**Assumption:** Hyperparameters generalize across similar envs/drift types.

---

## 🐛 Troubleshooting

### **Out of Memory**

```bash
# Reduce parallel jobs
--n-jobs 2  # instead of 4

# Use quick mode
--quick
```

### **Trials Pruning Too Aggressively**

Edit `scripts/tune_hyperparameters.py`:

```python
pruner=MedianPruner(
    n_warmup_steps=5,  # Increase to 10
    interval_steps=2    # Increase to 5
)
```

### **Study Not Found**

```bash
# List existing studies
ls results/optuna_studies/

# Create new if needed (just run tune script)
bash scripts/tune_moderate.sh
```

---

## 📈 Expected Improvements

Based on hyperparameter tuning literature:

- **Conservative:** +5-10% reward improvement
- **Moderate:** +10-20% improvement
- **Optimistic:** +20-50% improvement in best cases

Given old experiments showed **-9% to -20%** performance, even getting to **0% (equal to baseline)** would be success!

---

##🎯 Next Steps After Tuning

1. ✅ Validate tuned params on held-out seeds
2. ✅ Apply to all configs of same type
3. ✅ Re-run full experiments with tuned params
4. ✅ Compare new results vs old (analysis/)
5. ✅ Document improvements in paper/report

---

## 📚 Additional Resources

- **Optuna Docs:** https://optuna.readthedocs.io/
- **TPE Sampler:** https://arxiv.org/abs/1703.01041
- **Dashboard:** Local at http://localhost:8080

---

**Good luck with tuning! 🎛️**
