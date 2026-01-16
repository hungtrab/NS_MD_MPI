# Project Context

## Mục tiêu
Implement và thử nghiệm thuật toán **NS-MDMPI** (Non-Stationary Mirror Descent Modified Policy Iteration) cho Reinforcement Learning trong môi trường non-stationary.

## Cấu trúc project

```
Deep-RL/
├── src/                    # Source code chính
│   ├── envs/              # Environment wrappers cho drift
│   ├── callbacks/         # Training callbacks
│   └── estimators/        # Drift estimators
├── scripts/               # Scripts training, eval, tuning
├── configs/               # Config files cho experiments
│   ├── PPO/              # PPO experiments
│   ├── SAC/              # SAC experiments
│   └── TRPO/             # TRPO experiments
├── models/                # Saved models
├── logs/                  # Training logs
├── results/               # Experiment results
└── analysis/              # Analysis scripts
```

## Các file quan trọng

- `scripts/train.py` - Main training script
- `scripts/eval.py` - Evaluation script  
- `generate_configs.py` - Generate experiment configs
- `EXPERIMENT_COMMANDS.md` - Commands để chạy experiments
- `TUNING_GUIDE.md` - Hướng dẫn tuning hyperparameters
- `NSMDMPI.md` - Chi tiết về thuật toán NS-MDMPI

## Thuật toán

### Baseline
- **PPO** (Proximal Policy Optimization)
- **SAC** (Soft Actor-Critic)
- **TRPO** (Trust Region Policy Optimization)

### Adaptive (Ours)
- **NS-MDMPI**: Sử dụng Bellman Commutator Estimator để detect drift và adapt policy

## Environments

Sử dụng MuJoCo và Box2D environments với custom drift wrappers:
- HalfCheetah-v4
- Hopper-v4  
- Walker2d-v4
- LunarLander-v3

## WandB Project
- Project name: `NSMDMPI_EXP`
