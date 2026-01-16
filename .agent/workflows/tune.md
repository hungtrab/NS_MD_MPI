---
description: Hyperparameter tuning với Optuna cho NS-MDMPI
---

# Hyperparameter Tuning Workflow

## Tuning với Optuna

// turbo
1. Chạy tuning script:
```bash
python scripts/tune_optuna.py --env <ENV_NAME> --drift <DRIFT_TYPE> --n-trials 50
```

**Ví dụ:**
```bash
python scripts/tune_optuna.py --env HalfCheetah-v4 --drift friction_random_walk --n-trials 100
```

## Tuning cho NS-MDMPI

// turbo
2. Tuning NS-MDMPI hyperparameters:
```bash
python scripts/tune_nsmdmpi.py --config <CONFIG_PATH> --n-trials 50
```

## Tuning cho Extreme scenarios

// turbo
3. Chạy batch tuning cho extreme:
```bash
bash scripts/tune_all_extreme.sh
```

## Các parameters quan trọng cần tune

### NS-MDMPI Parameters
| Parameter | Range | Mô tả |
|-----------|-------|-------|
| `V_pi_star` | 0.1-1.0 | Policy budget |
| `commutator_threshold` | 0.01-0.1 | Threshold để trigger adaptation |
| `adaptation_rate` | 0.1-1.0 | Tốc độ adaptation |

### PPO Parameters  
| Parameter | Range | Mô tả |
|-----------|-------|-------|
| `learning_rate` | 1e-5 - 1e-3 | Learning rate |
| `clip_range` | 0.1-0.3 | PPO clip range |
| `n_steps` | 1024-4096 | Steps per update |

## Output

- Best parameters được lưu vào file `.db` (SQLite)
- Có thể visualize với Optuna Dashboard
