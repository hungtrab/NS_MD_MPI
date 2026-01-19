# TUNING_GUIDE.md - Hướng Dẫn Tune NS-MDMPI

## Tổng Quan

Hiện tại có 2 script tuning:
1. `scripts/wandb_sweep_agent.py` - Tune **NS-MDMPI** params
2. `scripts/tune_baseline.py` - Tune **Baseline PPO** params

---

## 1. Tune NS-MDMPI với `wandb_sweep_agent.py`

### Cách dùng:
```bash
python scripts/wandb_sweep_agent.py --create-sweep \
    --type <TYPE> \
    --env <ENV> \
    --drift-param <PARAM> \
    --drift-type <DRIFT> \
    --count <N>
```

### Arguments:
| Arg | Mô tả | Values |
|-----|-------|--------|
| `--type` | Loại experiment | `moderate`, `extreme`, `multi` |
| `--env` | Environment | `Hopper-v4`, `HalfCheetah-v4`, etc |
| `--drift-param` | Parameter bị drift | `friction`, `mass_scale`, `damping`, `gravity` |
| `--drift-type` | Loại drift | `sine`, `linear`, `random_walk`, `jump` |
| `--count` | Số trials | 20, 50, etc |
| `--config` | Custom YAML (optional) | path to yaml |

### Ví dụ tune HalfCheetah với damping sine:
```bash
python scripts/wandb_sweep_agent.py --create-sweep \
    --type moderate \
    --env HalfCheetah-v4 \
    --drift-param damping \
    --drift-type sine \
    --count 20
```

---

## 2. Tune theo từng Drift Config

### Option A: Tạo YAML config riêng

Tạo file `configs/sweep/halfcheetah_damping_sine.yaml`:
```yaml
program: scripts/wandb_sweep_agent.py
method: bayes
name: HalfCheetah Damping Sine Tuning

metric:
  name: optimization_score
  goal: maximize

parameters:
  # NS-MDMPI params (TUNE)
  V_R:
    distribution: uniform
    min: 5.0
    max: 20.0
  V_P:
    distribution: uniform
    min: 5.0
    max: 20.0
  V_pi_star:
    distribution: uniform
    min: 2.5
    max: 10.0
  # ... other NS-MDMPI params ...

  # Fixed params
  env_id:
    value: HalfCheetah-v4
  drift_parameter:
    value: damping          # <-- THAY ĐỔI 
  drift_type:
    value: sine             # <-- THAY ĐỔI 
  drift_magnitude:
    value: 0.3
  drift_period:
    value: 50000
  # ... etc ...
```

Rồi chạy:
```bash
python scripts/wandb_sweep_agent.py --create-sweep --config configs/sweep/halfcheetah_damping_sine.yaml --count 20
```

### Option B: Sửa script để hỗ trợ drift args (RECOMMENDED)

Thêm args `--drift-param` và `--drift-type` vào `wandb_sweep_agent.py`.

---

## 3. Params Được Tune (NS-MDMPI)

| Parameter | Moderate Range | Extreme Range | Mô tả |
|-----------|----------------|---------------|-------|
| `V_R` | 5-20 | 20-50 | Reward variation budget |
| `V_P` | 5-20 | 20-50 | Dynamics variation budget |
| `V_pi_star` | 2.5-10 | 10-25 | Policy budget |
| `kappa_base` | 0.1-0.3 | 0.15-0.4 | Base trust region |
| `lambda_base` | 0.5-2 | 1-5 | Base regularization |
| `trust_region_sensitivity` | 2-10 | 5-20 | κ adaptation speed |
| `regularization_sensitivity` | 1-5 | 3-10 | λ adaptation speed |
| `max_ent_coef` | 0.01-0.1 | 0.05-0.2 | Max entropy coef |
| `drift_window_size` | 500-2000 | 200-1000 | Drift estimation window |

---

## 4. Tune Baseline PPO

```bash
python scripts/tune_baseline.py --env <ENV> --drift-param <PARAM> --drift-type <TYPE> --count <N>
```

### Params Được Tune (PPO):
- `learning_rate`: 5e-5 - 1e-3
- `n_steps`: 1024, 2048, 4096
- `batch_size`: 32, 64, 128, 256
- `gamma`: 0.99, 0.995, 0.999
- `clip_range`: 0.1 - 0.3
- `ent_coef`: 1e-4 - 1e-2
- `vf_coef`: 0.3 - 0.7

---

## 5. Workflow Hoàn Chỉnh

### Bước 1: Tune NS-MDMPI cho từng config
```bash
# friction sine
python scripts/wandb_sweep_agent.py --create-sweep --type moderate --env HalfCheetah-v4 --count 20

# Cho các drift khác: tạo YAML config riêng (xem Option A)
```

### Bước 2: Lấy best params từ WandB
1. Vào WandB dashboard → Sweeps
2. Chọn sweep → Best run
3. Copy config values

### Bước 3: Tạo tuned config
Tạo file `configs/tuned/<config>_tuned.yaml` với best params.

### Bước 4: Chạy experiments với tuned config
```bash
python scripts/train.py --config configs/tuned/halfcheetah_friction_sine_nsmdmpi_ppo_tuned.yaml
```

---

## 6. Link WandB Dashboard

https://wandb.ai/hungtrab-hanoi-university-of-science-and-technology
