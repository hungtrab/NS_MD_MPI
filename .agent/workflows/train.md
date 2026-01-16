---
description: Chạy training experiment với config file
---

# Training Workflow

## Cách chạy training cơ bản

// turbo
1. Chạy training với config file:
```bash
python scripts/train.py --config <CONFIG_PATH> --seed <SEED>
```

**Ví dụ:**
```bash
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_sine_baseline_ppo.yaml --seed 42
```

## Chạy với nhiều seeds

// turbo
2. Chạy training với 3 seeds (42, 43, 44):
```bash
for seed in 42 43 44; do
  python scripts/train.py --config <CONFIG_PATH> --seed $seed
done
```

## Resume training

3. Resume từ checkpoint:
```bash
# Resume baseline (.zip)
python scripts/train.py --config <CONFIG_PATH> --resume models/<MODEL_NAME>

# Resume NS-MDMPI (.pt)
python scripts/train.py --config <CONFIG_PATH> --resume models/<MODEL_NAME>_params.pt --remaining_steps 500000
```

## Các loại experiment

| Loại | Thư mục config | Mô tả |
|------|----------------|-------|
| Moderate | `configs/PPO/moderate/` | 2M steps, 100k period |
| Extreme | `configs/PPO/extreme/` | Drift khó hơn |
| Multi | `configs/PPO/multi/` | Nhiều parameters drift cùng lúc |

## Environments

- **HalfCheetah-v4**: friction, damping, gravity, mass_scale
- **Hopper-v4**: friction, damping, gravity, mass_scale  
- **Walker2d-v4**: friction, damping, gravity, mass_scale
- **LunarLander-v3**: gravity, wind_power
