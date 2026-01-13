# Optuna Hyperparameter Tuning - Quick Reference

## 🚀 Quick Start

### 1. Install Optuna (if needed)
```bash
conda activate rl_hf_course
pip install optuna optuna-dashboard
```

### 2. Run Tuning

**Moderate Drift (recommended to start):**
```bash
bash scripts/tune_moderate.sh
```

**Extreme Drift:**
```bash
bash scripts/tune_extreme.sh
```

**Multi-Parameter:**
```bash
bash scripts/tune_multi.sh
```

## 📊 Monitor Progress

### View Optuna Dashboard (optional)
```bash
# In separate terminal
optuna-dashboard results/optuna_studies/moderate_hopper_friction_sine.db
# Open browser: http://localhost:8080
```

### Check study status
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

## 🎯 Best Practices

1. **Start small:** Run 5 validation trials first
2. **Monitor:** Use optuna-dashboard to track progress
3. **Resume:** Studies auto-save, safe to interrupt (Ctrl+C)
4. **Parallel:** Adjust `--n-jobs` based on available cores

## ⏱️ Time Estimates

| Type | Quick Mode (100k) | Full Mode (500k) |
|------|-------------------|------------------|
| 5 trials | ~30 min | ~2 hours |
| 50 trials | ~6 hours | ~20 hours |

With 4 parallel jobs (divide by 4).

## 📁 Output Files

After tuning completes:
- **Database:** `results/optuna_studies/{study_name}.db`
- **Best Params:** `results/tuned_params/{study_name}_best_params.yaml`

## 🔄 Apply Tuned Parameters

```bash
# Manual: Copy best params to configs
cat results/tuned_params/moderate_hopper_friction_sine_best_params.yaml

# Then update configs/PPO/moderate/*_nsmdmpi_ppo.yaml
```

## ⚡ Speed Tips

1. Use `--quick` flag (100k timesteps instead of 500k)
2. Increase `--n-jobs` for more parallelism
3. Run on CPU (`device='cpu'` in script)
4. Use smaller networks during search

## 🐛 Troubleshooting

**Out of Memory:**
- Reduce `--n-jobs`
- Use `--quick` mode

**Trial Pruning Too Aggressive:**
- Increase `n_warmup_steps` in script

**Slow Progress:**
- Check CPU usage (`htop`)
- Reduce timesteps temporarily
