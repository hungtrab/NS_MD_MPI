# Hyperparameter Tuning Guide

## 🌟 Recommended: W&B Sweep

### Quick Start (All-in-One)
```bash
# Tạo sweep + chạy 50 trials tự động
python scripts/wandb_sweep_agent.py --create-sweep --type moderate --env Hopper-v4 --count 50
```

### Step-by-Step

**1. Tạo sweep từ config:**
```bash
wandb sweep configs/sweep/sweep_moderate.yaml
# Output: Created sweep with ID: entity/NS-MDMPI-Sweep/abc123
```

**2. Chạy agent:**
```bash
python scripts/wandb_sweep_agent.py --sweep-id entity/NS-MDMPI-Sweep/abc123 --count 50
```

### 🚀 Parallel Tuning (Multiple Machines)
```bash
# Terminal 1 (or Machine 1)
python scripts/wandb_sweep_agent.py --sweep-id <SWEEP_ID> --count 25

# Terminal 2 (or Machine 2) - cùng sweep!
python scripts/wandb_sweep_agent.py --sweep-id <SWEEP_ID> --count 25
```

### 📁 Sweep Configs
| Config | Environment | Drift | Use Case |
|--------|------------|-------|----------|
| `configs/sweep/sweep_moderate.yaml` | Hopper-v4 | friction/sine | Moderate drift |
| `configs/sweep/sweep_extreme.yaml` | Hopper-v4 | friction/random_walk | Extreme drift |
| `configs/sweep/sweep_lunarlander.yaml` | LunarLander-v2 | gravity/sine | LunarLander |

### 📊 Monitor on W&B Dashboard
1. Go to: https://wandb.ai/<your-entity>/NS-MDMPI-Sweep
2. Click Sweeps → Your sweep
3. View parallel coordinates, importance, and best runs

---

## 🔧 Alternative: Optuna

### Individual Environment Tuning

**Moderate Drift:**
```bash
bash scripts/tune_moderate_hopper.sh          # Hopper-v4
bash scripts/tune_moderate_halfcheetah.sh     # HalfCheetah-v4
bash scripts/tune_moderate_walker2d.sh        # Walker2d-v4
bash scripts/tune_moderate_lunarlander.sh     # LunarLander-v2
```

**Extreme Drift:**
```bash
bash scripts/tune_extreme_hopper.sh           # Hopper-v4
bash scripts/tune_extreme_halfcheetah.sh      # HalfCheetah-v4
```

### Monitor Optuna
```bash
# Dashboard (while running)
optuna-dashboard results/optuna_studies/moderate_hopper_full.db
# Open: http://localhost:8080

# Check results
cat results/tuned_params/moderate_hopper_full_best_params.yaml
```

---

## ⚖️ W&B Sweep vs Optuna

| Feature | W&B Sweep | Optuna |
|---------|-----------|--------|
| Cloud Dashboard | ✅ wandb.ai | ❌ |
| Multi-Machine | ✅ Easy | ⚠️ Need DB |
| Bayesian | ✅ | ✅ |
| Early Stop | ✅ Hyperband | ✅ MedianPruner |
| Experiment Tracking | ✅ Built-in | ❌ |

**Recommendation:** Use W&B Sweep for cloud visibility and easy distributed tuning.

---

## ⏱️ Estimated Times

| Environment | 50 Trials (200k steps) | 50 Trials (500k steps) |
|-------------|------------------------|------------------------|
| Hopper      | ~3-4 hours             | ~8-10 hours            |
| HalfCheetah | ~3-4 hours             | ~8-10 hours            |
| LunarLander | ~2-3 hours             | ~5-6 hours             |

---

## 💡 Tips

1. **Start small**: Use `--type moderate --count 10` first
2. **Monitor early**: Check W&B dashboard after 5 runs
3. **Distributed**: Run multiple agents on the same sweep
4. **Custom config**: Edit YAMLs in `configs/sweep/`
