# NS-MDMPI Extreme Drift Tuning Guide

## 🎯 Why Extreme Drift Tuning?

Extreme drift scenarios (random walk, sudden jumps) are where **NS-MDMPI should show its biggest advantage** over baseline methods. These require:
- **Larger variation budgets** (V_R, V_P, V_π*) to handle rapid changes
- **More aggressive adaptation** (higher α, β sensitivity)
- **Faster drift detection** (smaller windows)

## 📊 Expected Search Spaces (Extreme vs Moderate)

| Parameter | Moderate Range | Extreme Range | Why Different? |
|-----------|---------------|---------------|----------------|
| V_R | 5-20 | **20-50** | Need more rollout budget for rapid changes |
| V_P | 5-20 | **20-50** | More policy variation needed |
| V_π* | 2.5-10 | **10-25** | Larger policy update budget |
| α (trust region) | 2-10 | **5-20** | More aggressive constraint adaptation |
| β (regularization) | 1-5 | **3-10** | Stronger regularization updates |
| max_ent_coef | 0.01-0.1 | **0.05-0.2** | More exploration for changing dynamics |
| drift_window | 500-2000 | **200-1000** | Faster detection of rapid changes |

## 🚀 Usage

### Quick Start - Single Environment

```bash
# Tune Hopper extreme (2 configs: random_walk, jump)
bash scripts/tune_hopper_extreme.sh

# Tune HalfCheetah extreme
bash scripts/tune_halfcheetah_extreme.sh

# Tune LunarLander extreme
bash scripts/tune_lunarlander_extreme.sh
```

### Full Extreme Tuning - All Environments

```bash
# Tune all environments for extreme drift
# WARNING: Takes 10-15 hours!
bash scripts/tune_all_extreme.sh
```

### Custom Tuning

```bash
python scripts/tune_hyperparameters.py \
    --env "Hopper-v4" \
    --config configs/PPO/extreme/hopper_friction_jump_baseline_ppo.yaml \
    --type extreme \
    --n-trials 50 \
    --n-jobs 4 \
    --study-name "my_extreme_study"
```

## 📁 Results Location

After tuning completes:

**Optuna Studies:**
```
results/optuna_studies/extreme_hopper_friction_random_walk.db
results/optuna_studies/extreme_hopper_friction_jump.db
results/optuna_studies/extreme_halfcheetah_*.db
results/optuna_studies/extreme_lunarlander_*.db
```

**Best Parameters:**
```
results/tuned_params/extreme_hopper_friction_random_walk_best_params.yaml
results/tuned_params/extreme_hopper_friction_jump_best_params.yaml
...
```

## 🔍 Monitoring Progress

### Optuna Dashboard
```bash
# While tuning is running
optuna-dashboard results/optuna_studies/extreme_hopper_friction_jump.db

# Open browser: http://localhost:8080
```

### Check Results
```bash
# View best parameters
cat results/tuned_params/extreme_hopper_*_best_params.yaml

# Compare with moderate tuning
diff results/tuned_params/moderate_hopper_*_best_params.yaml \
     results/tuned_params/extreme_hopper_*_best_params.yaml
```

## 📊 Expected Results

**Hypothesis:** Extreme drift tuning should find:
- **Higher budgets** (2-3x moderate values)
- **More aggressive adaptation** (α, β ~ 2x moderate)
- **Smaller drift windows** (~50% of moderate)

This reflects that extreme drift requires:
1. More resources to track rapid changes
2. Faster adaptation to sudden shifts
3. Quicker detection of distribution changes

## 🎯 Next Steps After Tuning

1. **Apply tuned params:**
   ```bash
   # Script will auto-generate tuned configs
   # Or manually update nsmdmpi section in configs
   ```

2. **Run validation:**
   ```bash
   bash scripts/run_hopper_extreme_tuned.sh
   ```

3. **Compare in WandB:**
   - Filter: `tuned` tag
   - Compare learning curves
   - Check if performance improves over default params

4. **Analyze budget usage:**
   - Do budgets last longer?
   - Better adaptation timing?
   - Improved stability?

## 💡 Optimization Tips

### Speed Up Tuning
```bash
# Fewer trials for initial exploration
python scripts/tune_hyperparameters.py ... --n-trials 25

# More parallel jobs (if you have resources)
python scripts/tune_hyperparameters.py ... --n-jobs 8

# Quick mode (100k timesteps instead of 500k)
python scripts/tune_hyperparameters.py ... --quick
```

### Resume Interrupted Tuning
Optuna automatically resumes! Just re-run the same command:
```bash
bash scripts/tune_hopper_extreme.sh
# Picks up where it left off using existing .db file
```

## ⏱️ Time Estimates

| Environment | Configs | Time per Config | Total |
|-------------|---------|-----------------|-------|
| Hopper | 2 | 2-3 hours | 4-6 hours |
| HalfCheetah | 2 | 2-3 hours | 4-6 hours |
| LunarLander | 2 | 1-2 hours | 2-4 hours |
| **TOTAL** | **6** | - | **10-16 hours** |

**Recommendation:** Run overnight or in parallel on multiple machines.

## 🐛 Troubleshooting

**Issue: All trials fail**
- Check config file exists and is valid
- Verify environment can be created
- Test with `--quick --n-trials 5` first

**Issue: Very slow**
- Use `--quick` mode
- Reduce `--n-trials`
- Increase `--n-jobs` (if you have CPU/GPU resources)

**Issue: Out of memory**
- Reduce `--n-jobs`
- Use `--quick` mode (smaller networks)
- Close other applications

**Issue: Poor hyperparameters found**
- Increase `--n-trials` (50 → 100)
- Check if search space is appropriate
- Validate fitness function makes sense
