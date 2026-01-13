# Quick Reference: Hyperparameter Tuning Scripts

## 🎯 Individual Environment Tuning

### Moderate Drift
```bash
bash scripts/tune_moderate_hopper.sh          # Hopper-v4
bash scripts/tune_moderate_halfcheetah.sh     # HalfCheetah-v4
bash scripts/tune_moderate_walker2d.sh        # Walker2d-v4
bash scripts/tune_moderate_swimmer.sh         # Swimmer-v4
bash scripts/tune_moderate_humanoid.sh        # Humanoid-v4
bash scripts/tune_moderate_lunarlander.sh     # LunarLander-v2
```

### Extreme Drift
```bash
bash scripts/tune_extreme_hopper.sh           # Hopper-v4
bash scripts/tune_extreme_halfcheetah.sh      # HalfCheetah-v4
bash scripts/tune_extreme_walker2d.sh         # Walker2d-v4
bash scripts/tune_extreme_lunarlander.sh      # LunarLander-v2
```

## 🚀 Batch Tuning

### All Environments (Sequential)
```bash
bash scripts/tune_all.sh
```
**⚠️ Warning:** Takes 10-20 hours total!

### Parallel Tuning (Advanced - if you have GPUs)
```bash
# Terminal 1
bash scripts/tune_moderate_hopper.sh

# Terminal 2 
bash scripts/tune_moderate_halfcheetah.sh

# Terminal 3
bash scripts/tune_extreme_hopper.sh
```

## 📊 Monitor Progress

### Optuna Dashboard
```bash
# While tuning is running
optuna-dashboard results/optuna_studies/moderate_hopper_full.db

# Open browser: http://localhost:8080
```

### Check Results
```bash
# View best parameters
cat results/tuned_params/moderate_hopper_full_best_params.yaml

# List all tuned params
ls -lh results/tuned_params/
```

## 🔄 Resume Interrupted Tuning

If tuning is interrupted, just re-run the same script:
```bash
bash scripts/tune_moderate_hopper.sh
```

Optuna automatically resumes from where it left off!

## ⏱️ Estimated Times

| Environment | Moderate (50 trials) | Extreme (50 trials) |
|-------------|---------------------|---------------------|
| Hopper      | ~2-3 hours          | ~2-3 hours          |
| HalfCheetah | ~2-3 hours          | ~2-3 hours          |
| Walker2D    | ~2-3 hours          | ~2-3 hours          |
| Swimmer     | ~1-2 hours          | N/A                 |
| Humanoid    | ~4-6 hours          | N/A                 |
| LunarLander | ~1-2 hours          | ~1-2 hours          |

**Total if sequential: 15-25 hours**

## 💡 Tips

1. **Start with one env** - Test the process works
2. **Run overnight** - Let it tune while you sleep
3. **Use transfer learning** - Apply Hopper params to similar envs first
4. **Parallel if possible** - Use multiple terminals/GPUs
5. **Monitor early trials** - Stop if all trials fail

## 🎛️ Custom Tuning

For custom parameters:
```bash
python scripts/tune_hyperparameters.py \
    --env "Hopper-v4" \
    --config configs/PPO/moderate/hopper_friction_sine_baseline_ppo.yaml \
    --type moderate \
    --n-trials 100 \    # More trials
    --n-jobs 8 \        # More parallel jobs
    --quick             # Fast mode (100k timesteps)
```
