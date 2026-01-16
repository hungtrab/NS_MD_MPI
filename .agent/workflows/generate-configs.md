---
description: Tạo config files cho experiments
---

# Generate Configs Workflow

## Generate tất cả configs

// turbo
1. Generate config files cho experiments:
```bash
python generate_configs.py
```

Configs được tạo theo cấu trúc:
```
configs/
├── PPO/
│   ├── moderate/    # 2M steps, 100k drift period
│   ├── extreme/     # Khó hơn
│   └── multi/       # Multi-parameter drift
├── SAC/
│   └── ...
└── TRPO/
    └── ...
```

## Generate scripts chạy batch

// turbo
2. Generate bash scripts để chạy hàng loạt:
```bash
python generate_scripts.py
```

Scripts được tạo tại `scripts/run_*.sh`

## Config file format

```yaml
# Ví dụ config file
env:
  name: "HalfCheetah-v4"
  drift_params:
    - name: friction
      drift_type: sine
      period: 100000
      magnitude: 0.3

algorithm:
  name: PPO
  total_timesteps: 2000000
  
nsmdmpi:
  enabled: true
  V_pi_star: 0.5
  commutator_threshold: 0.05
```

## Naming convention

Format: `{env}_{param}_{drift_type}_{baseline|nsmdmpi}_{algo}.yaml`

**Ví dụ:**
- `halfcheetah_friction_sine_baseline_ppo.yaml`
- `halfcheetah_friction_sine_nsmdmpi_ppo.yaml`
