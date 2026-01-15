# Experiment Commands Reference

## Quick Start

```bash
# Basic usage
python scripts/train.py --config CONFIG_PATH --seed SEED

# Example
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_sine_baseline_ppo.yaml --seed 42
```

---

## HalfCheetah-v4

### Moderate (2M steps, 100k period)

#### Friction
```bash
# Sine
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_sine_nsmdmpi_ppo.yaml --seed 42

# Random Walk
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_random_walk_nsmdmpi_ppo.yaml --seed 42

# Linear
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_linear_nsmdmpi_ppo.yaml --seed 42

# Jump
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_jump_nsmdmpi_ppo.yaml --seed 42
```

#### Damping
```bash
# Sine
python scripts/train.py --config configs/PPO/moderate/halfcheetah_damping_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_damping_sine_nsmdmpi_ppo.yaml --seed 42

# Random Walk
python scripts/train.py --config configs/PPO/moderate/halfcheetah_damping_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_damping_random_walk_nsmdmpi_ppo.yaml --seed 42

# Linear
python scripts/train.py --config configs/PPO/moderate/halfcheetah_damping_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_damping_linear_nsmdmpi_ppo.yaml --seed 42

# Jump
python scripts/train.py --config configs/PPO/moderate/halfcheetah_damping_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_damping_jump_nsmdmpi_ppo.yaml --seed 42
```

#### Gravity
```bash
# Sine
python scripts/train.py --config configs/PPO/moderate/halfcheetah_gravity_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_gravity_sine_nsmdmpi_ppo.yaml --seed 42

# Random Walk
python scripts/train.py --config configs/PPO/moderate/halfcheetah_gravity_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_gravity_random_walk_nsmdmpi_ppo.yaml --seed 42

# Linear
python scripts/train.py --config configs/PPO/moderate/halfcheetah_gravity_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_gravity_linear_nsmdmpi_ppo.yaml --seed 42

# Jump
python scripts/train.py --config configs/PPO/moderate/halfcheetah_gravity_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/halfcheetah_gravity_jump_nsmdmpi_ppo.yaml --seed 42
```

### Extreme

```bash
# Friction Random Walk
python scripts/train.py --config configs/PPO/extreme/halfcheetah_friction_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/halfcheetah_friction_random_walk_nsmdmpi_ppo.yaml --seed 42

# Friction Jump
python scripts/train.py --config configs/PPO/extreme/halfcheetah_friction_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/halfcheetah_friction_jump_nsmdmpi_ppo.yaml --seed 42

# Mass Scale Random Walk
python scripts/train.py --config configs/PPO/extreme/halfcheetah_mass_scale_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/halfcheetah_mass_scale_random_walk_nsmdmpi_ppo.yaml --seed 42

# Mass Scale Jump
python scripts/train.py --config configs/PPO/extreme/halfcheetah_mass_scale_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/halfcheetah_mass_scale_jump_nsmdmpi_ppo.yaml --seed 42

# Gravity Random Walk
python scripts/train.py --config configs/PPO/extreme/halfcheetah_gravity_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/halfcheetah_gravity_random_walk_nsmdmpi_ppo.yaml --seed 42

# Gravity Jump
python scripts/train.py --config configs/PPO/extreme/halfcheetah_gravity_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/halfcheetah_gravity_jump_nsmdmpi_ppo.yaml --seed 42
```

### Multi-Parameter

```bash
# 2 Parameters (friction + damping)
python scripts/train.py --config configs/PPO/multi/halfcheetah_2param_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/halfcheetah_2param_randomwalk_nsmdmpi_ppo.yaml --seed 42

# 3 Parameters (friction + damping + mass_scale)
python scripts/train.py --config configs/PPO/multi/halfcheetah_3param_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/halfcheetah_3param_randomwalk_nsmdmpi_ppo.yaml --seed 42

# 4 Parameters (friction + damping + mass_scale + gravity)
python scripts/train.py --config configs/PPO/multi/halfcheetah_4param_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/halfcheetah_4param_randomwalk_nsmdmpi_ppo.yaml --seed 42
```

---

## Hopper-v4

### Moderate (2M steps, 100k period)

#### Friction
```bash
python scripts/train.py --config configs/PPO/moderate/hopper_friction_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_friction_sine_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_friction_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_friction_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_friction_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_friction_linear_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_friction_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_friction_jump_nsmdmpi_ppo.yaml --seed 42
```

#### Damping
```bash
python scripts/train.py --config configs/PPO/moderate/hopper_damping_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_damping_sine_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_damping_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_damping_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_damping_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_damping_linear_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_damping_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_damping_jump_nsmdmpi_ppo.yaml --seed 42
```

#### Gravity
```bash
python scripts/train.py --config configs/PPO/moderate/hopper_gravity_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_gravity_sine_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_gravity_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_gravity_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_gravity_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_gravity_linear_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_gravity_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/hopper_gravity_jump_nsmdmpi_ppo.yaml --seed 42
```

### Extreme

```bash
python scripts/train.py --config configs/PPO/extreme/hopper_friction_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_friction_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_friction_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_friction_jump_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_mass_scale_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_mass_scale_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_mass_scale_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_mass_scale_jump_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_gravity_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_gravity_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_gravity_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/hopper_gravity_jump_nsmdmpi_ppo.yaml --seed 42
```

### Multi-Parameter

```bash
python scripts/train.py --config configs/PPO/multi/hopper_2param_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/hopper_2param_randomwalk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/hopper_3param_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/hopper_3param_randomwalk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/hopper_4param_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/hopper_4param_randomwalk_nsmdmpi_ppo.yaml --seed 42
```

---

## Walker2d-v4

### Moderate (2M steps, 100k period)

#### Friction
```bash
python scripts/train.py --config configs/PPO/moderate/walker2d_friction_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_friction_sine_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_friction_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_friction_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_friction_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_friction_linear_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_friction_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_friction_jump_nsmdmpi_ppo.yaml --seed 42
```

#### Damping
```bash
python scripts/train.py --config configs/PPO/moderate/walker2d_damping_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_damping_sine_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_damping_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_damping_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_damping_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_damping_linear_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_damping_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_damping_jump_nsmdmpi_ppo.yaml --seed 42
```

#### Gravity
```bash
python scripts/train.py --config configs/PPO/moderate/walker2d_gravity_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_gravity_sine_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_gravity_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_gravity_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_gravity_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_gravity_linear_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_gravity_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/walker2d_gravity_jump_nsmdmpi_ppo.yaml --seed 42
```

### Extreme

```bash
python scripts/train.py --config configs/PPO/extreme/walker2d_friction_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_friction_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_friction_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_friction_jump_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_mass_scale_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_mass_scale_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_mass_scale_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_mass_scale_jump_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_gravity_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_gravity_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_gravity_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/walker2d_gravity_jump_nsmdmpi_ppo.yaml --seed 42
```

### Multi-Parameter

```bash
python scripts/train.py --config configs/PPO/multi/walker2d_2param_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/walker2d_2param_randomwalk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/walker2d_3param_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/walker2d_3param_randomwalk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/walker2d_4param_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/walker2d_4param_randomwalk_nsmdmpi_ppo.yaml --seed 42
```

---

## LunarLander-v3 (1M steps, 10k period)

### Moderate

#### Gravity
```bash
python scripts/train.py --config configs/PPO/moderate/lunarlander_gravity_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_gravity_sine_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_gravity_randomwalk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_gravity_randomwalk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_gravity_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_gravity_linear_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_gravity_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_gravity_jump_nsmdmpi_ppo.yaml --seed 42
```

#### Wind Power
```bash
python scripts/train.py --config configs/PPO/moderate/lunarlander_wind_power_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_wind_power_sine_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_wind_power_linear_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_wind_power_linear_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_wind_power_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_wind_power_jump_nsmdmpi_ppo.yaml --seed 42
```

#### Wind (Sine only)
```bash
python scripts/train.py --config configs/PPO/moderate/lunarlander_wind_sine_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/moderate/lunarlander_wind_sine_nsmdmpi_ppo.yaml --seed 42
```

### Extreme

```bash
python scripts/train.py --config configs/PPO/extreme/lunarlander_gravity_random_walk_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/lunarlander_gravity_random_walk_nsmdmpi_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/lunarlander_gravity_jump_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/extreme/lunarlander_gravity_jump_nsmdmpi_ppo.yaml --seed 42
```

### Multi-Parameter

```bash
python scripts/train.py --config configs/PPO/multi/lunarlander_gravity_wind_power_baseline_ppo.yaml --seed 42
python scripts/train.py --config configs/PPO/multi/lunarlander_gravity_wind_power_nsmdmpi_ppo.yaml --seed 42
```

---

## Running with Multiple Seeds

For each experiment, run with seeds 42, 43, 44:

```bash
# Example: HalfCheetah friction sine
for seed in 42 43 44; do
  python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_sine_baseline_ppo.yaml --seed $seed
  python scripts/train.py --config configs/PPO/moderate/halfcheetah_friction_sine_nsmdmpi_ppo.yaml --seed $seed
done
```

---

## Resume Training

```bash
# Resume from .zip (baseline)
python scripts/train.py --config CONFIG --resume models/MODEL_NAME

# Resume from .pt (NS-MDMPI)
python scripts/train.py --config CONFIG --resume models/MODEL_NAME_params.pt --remaining_steps 500000
```

---

## Summary

| Environment | Moderate | Extreme | Multi | Total |
|-------------|----------|---------|-------|-------|
| HalfCheetah | 24 | 12 | 6 | 42 |
| Hopper | 24 | 12 | 6 | 42 |
| Walker2d | 24 | 12 | 6 | 42 |
| LunarLander | 20 | 4 | 2 | 26 |
| **Total** | **92** | **40** | **20** | **152** |

With 3 seeds each: **152 × 3 = 456 total runs**

