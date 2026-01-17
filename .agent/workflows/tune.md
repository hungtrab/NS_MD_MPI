---
description: Hyperparameter tuning với W&B Sweep hoặc Optuna cho NS-MDMPI
---

# Hyperparameter Tuning Workflow

## 🎯 Option 1: W&B Sweep (Recommended)

### Quick Start - Auto Mode
// turbo
```bash
# Tạo sweep + chạy agent tự động (50 trials)
python scripts/wandb_sweep_agent.py --create-sweep --type moderate --env Hopper-v4 --count 50
```

### Manual Mode - Tạo sweep trước

// turbo
1. Tạo sweep từ config YAML:
```bash
wandb sweep configs/sweep/sweep_moderate.yaml
# Output: Created sweep with ID: xxx/NS-MDMPI-Sweep/abc123
```

// turbo
2. Chạy sweep agent:
```bash
python scripts/wandb_sweep_agent.py --sweep-id <ENTITY>/NS-MDMPI-Sweep/<SWEEP_ID> --count 50
```

### Parallel Agents (Distributed)
```bash
# Terminal 1
python scripts/wandb_sweep_agent.py --sweep-id <SWEEP_ID> --count 25

# Terminal 2 (same machine or different)
python scripts/wandb_sweep_agent.py --sweep-id <SWEEP_ID> --count 25
```

### Available Sweep Configs
| Config | Environment | Drift Type | Description |
|--------|------------|------------|-------------|
| `sweep_moderate.yaml` | Hopper-v4 | friction/sine | Moderate drift |
| `sweep_extreme.yaml` | Hopper-v4 | friction/random_walk | Extreme drift |
| `sweep_lunarlander.yaml` | LunarLander-v2 | gravity/sine | LunarLander |

---

## 🔧 Option 2: Optuna

// turbo
1. Chạy tuning script:
```bash
python scripts/tune_optuna.py --env <ENV_NAME> --drift <DRIFT_TYPE> --n-trials 50
```

// turbo
2. Tuning NS-MDMPI hyperparameters:
```bash
python scripts/tune_nsmdmpi.py --config <CONFIG_PATH> --n-trials 50
```

---

## 📊 W&B Sweep vs Optuna

| Feature | W&B Sweep | Optuna |
|---------|-----------|--------|
| Dashboard | ✅ Cloud (wandb.ai) | ❌ Local only |
| Distributed | ✅ Easy multi-machine | ⚠️ Requires DB |
| Visualization | ✅ Built-in plots | ⚠️ optuna-dashboard |
| Bayesian | ✅ bayes method | ✅ TPESampler |
| Early Stop | ✅ Hyperband | ✅ MedianPruner |
| Results | ✅ Auto-synced | 📁 Local .db |

---

## 📈 Các Parameters cần tune

### NS-MDMPI Parameters
| Parameter | Moderate Range | Extreme Range | Mô tả |
|-----------|---------------|---------------|-------|
| `V_R` | 5-20 | 20-50 | Reward variation budget |
| `V_P` | 5-20 | 20-50 | Dynamics variation budget |
| `V_pi_star` | 2.5-10 | 10-25 | Policy budget |
| `kappa_base` | 0.1-0.3 | 0.15-0.4 | Base trust region |
| `lambda_base` | 0.5-2 | 1-5 | Base regularization |
| `trust_region_sensitivity` | 2-10 | 5-20 | κ adaptation speed |
| `regularization_sensitivity` | 1-5 | 3-10 | λ adaptation speed |
| `max_ent_coef` | 0.01-0.1 | 0.05-0.2 | Max entropy coefficient |
| `drift_window_size` | 500-2000 | 200-1000 | Drift estimation window |

---

## 📁 Output

### W&B Sweep
- Results: https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>
- Best params: Visible in Sweep dashboard → Best Run

### Optuna
- Best parameters: `results/tuned_params/<study_name>_best_params.yaml`
- Database: `results/optuna_studies/<study_name>.db`
- Dashboard: `optuna-dashboard results/optuna_studies/<db_file>.db`
