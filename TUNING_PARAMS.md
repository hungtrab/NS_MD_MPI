# NS-MDMPI Tuning Parameter Ranges

## Hopper-v4

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| V_R | 30 | 150 | Reward variation budget |
| V_P | 50 | 250 | Dynamics variation budget |
| V_pi_star | 300 | 1500 | Policy budget |
| kappa_base | 0.1 | 0.35 | Base trust region |
| kappa_min | 0.02 | 0.08 | Min trust region |
| kappa_max | 0.3 | 0.5 | Max trust region |
| lambda_base | 0.5 | 2.5 | Base regularization |
| lambda_min | 0.05 | 0.15 | Min regularization |
| lambda_max | 5.0 | 12.0 | Max regularization |
| trust_region_sensitivity | 3.0 | 20.0 | κ adaptation speed |
| regularization_sensitivity | 1.5 | 8.0 | λ adaptation speed |
| min_ent_coef | 0.0 | 0.015 | Min entropy coef |
| max_ent_coef | 0.02 | 0.12 | Max entropy coef |
| drift_window_size | 400 | 1500 | Drift estimation window |
| drift_min_samples | 50 | 150 | Min samples for drift |

---

## HalfCheetah-v4

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| V_R | 50 | 200 | Reward variation budget |
| V_P | 100 | 400 | Dynamics variation budget |
| V_pi_star | 500 | 2000 | Policy budget |
| kappa_base | 0.1 | 0.4 | Base trust region |
| kappa_min | 0.02 | 0.1 | Min trust region |
| kappa_max | 0.3 | 0.6 | Max trust region |
| lambda_base | 0.5 | 3.0 | Base regularization |
| lambda_min | 0.05 | 0.2 | Min regularization |
| lambda_max | 5.0 | 15.0 | Max regularization |
| trust_region_sensitivity | 5.0 | 25.0 | κ adaptation speed |
| regularization_sensitivity | 2.0 | 10.0 | λ adaptation speed |
| min_ent_coef | 0.0 | 0.02 | Min entropy coef |
| max_ent_coef | 0.03 | 0.15 | Max entropy coef |
| drift_window_size | 500 | 2000 | Drift estimation window |
| drift_min_samples | 50 | 200 | Min samples for drift |

---

## Tune Commands

### Hopper (Colab)
```bash
python scripts/wandb_sweep_agent.py --create-sweep \
    --base-config configs/experiments/hopper/C01_friction_sine.yaml \
    --type moderate --count 10
```

### HalfCheetah (Local)
```bash
python scripts/wandb_sweep_agent.py --create-sweep \
    --base-config configs/experiments/halfcheetah/C01_friction_sine.yaml \
    --type moderate --count 10
```

---

## Training (2M steps)
```bash
python scripts/train.py --config configs/experiments/hopper/C01_friction_sine.yaml --seed 42
```
