# Environment Parameters Reference

This document lists all driftable parameters for each supported environment, including their default values, valid ranges, and physical interpretations.

---

## CartPole-v1

**Description**: Balance a pole on a moving cart. The agent must keep the pole upright by moving the cart left or right.

### Driftable Parameters

| Parameter | Default Value | Unit | Physical Meaning | Effect on Difficulty |
|-----------|--------------|------|-----------------|---------------------|
| `gravity` | 9.8 | m/s² | Gravitational acceleration | Higher → Harder to balance |
| `masscart` | 1.0 | kg | Mass of the cart | Higher → More inertia, slower response |
| `masspole` | 0.1 | kg | Mass of the pole | Higher → Harder to balance |
| `length` | 0.5 | m | Half-length of the pole | Longer → Easier to balance |

### Typical Drift Configurations

```yaml
# Moderate gravity drift (recommended)
parameter: "gravity"
drift_type: "sine"
magnitude: 5.0          # ±5.0 m/s² oscillation
period: 10000           # One cycle per 10k steps

# Mass cart drift
parameter: "masscart"
drift_type: "linear"
magnitude: 0.5          # ±0.5 kg change
period: 25000

# Sudden gravity change
parameter: "gravity"
drift_type: "jump"
magnitude: 8.0          # +8.0 m/s² after jump
period: 50000           # Jump at 50k steps
```

### Valid Ranges
- `gravity`: [0.1, 30.0] m/s²
- `masscart`: [0.1, 5.0] kg
- `masspole`: [0.01, 1.0] kg
- `length`: [0.1, 2.0] m

---

## MountainCar-v0

**Description**: Drive an underpowered car up a steep mountain. The agent must build momentum by rocking back and forth.

### Driftable Parameters

| Parameter | Default Value | Unit | Physical Meaning | Effect on Difficulty |
|-----------|--------------|------|-----------------|---------------------|
| `gravity` | 0.0025 | - | Gravitational constant | Higher → Harder to climb |
| `force` | 0.001 | - | Engine force multiplier | Lower → Weaker acceleration |
| `goal_position` | 0.5 | - | Goal location (x-coordinate) | Higher → Further goal |
| `goal_velocity` | 0.0 | - | Required velocity at goal | Higher → Need more momentum |

### Typical Drift Configurations

```yaml
# Subtle gravity drift (recommended)
parameter: "gravity"
drift_type: "sine"
magnitude: 0.001        # ±0.001 oscillation (subtle)
period: 20000           # One cycle per 20k steps

# Engine force degradation
parameter: "force"
drift_type: "linear"
magnitude: 0.0005       # ±0.0005 change
period: 250000

# Sudden force change
parameter: "force"
drift_type: "jump"
magnitude: 0.0005       # +0.0005 after jump
period: 500000
```

### Valid Ranges
- `gravity`: [0.001, 0.005]
- `force`: [0.0005, 0.003]
- `goal_position`: [0.4, 0.7]
- `goal_velocity`: [0.0, 0.1]

---

## FrozenLake-v1

**Description**: Navigate a frozen lake to reach the goal while avoiding holes. Slippery ice causes stochastic transitions.

### Driftable Parameters

| Parameter | Default Value | Unit | Physical Meaning | Effect on Difficulty |
|-----------|--------------|------|-----------------|---------------------|
| `slip_prob` | 0.67 (if slippery) / 0.0 (if not) | probability | Probability of slipping | Higher → More stochastic |
| `reward_scale` | 1.0 | - | Reward multiplier | Lower → Weaker signal |

### Typical Drift Configurations

```yaml
# Oscillating ice conditions (recommended)
parameter: "slip_prob"
drift_type: "sine"
magnitude: 0.15         # ±0.15 oscillation
period: 20000           # One cycle per 20k steps

# Sudden ice thaw
parameter: "slip_prob"
drift_type: "jump"
magnitude: 0.25         # +0.25 after jump
period: 250000

# Gradual ice melting
parameter: "slip_prob"
drift_type: "linear"
magnitude: 0.2          # ±0.2 change
period: 30000
```

### Valid Ranges
- `slip_prob`: [0.0, 1.0]
- `reward_scale`: [0.1, 5.0]

### Notes
- When `is_slippery=True`, base slip_prob = 2/3 (FrozenLake default behavior)
- When `is_slippery=False`, base slip_prob = 0.0 (deterministic)
- Slip probability is implemented by intercepting actions, not by modifying the transition matrix

---

## MiniGrid-Empty-8x8-v0

**Description**: Navigate a grid world to reach the goal. Simple environment with procedural generation.

### Driftable Parameters

| Parameter | Default Value | Unit | Physical Meaning | Effect on Difficulty |
|-----------|--------------|------|-----------------|---------------------|
| `reward_scale` | 1.0 | - | Reward multiplier | Higher → Stronger signal |
| `max_steps` | 100 | steps | Episode time limit | Lower → Less time |
| `step_penalty` | 0.0 | - | Penalty per step | Higher → Encourages speed |

### Typical Drift Configurations

```yaml
# Oscillating reward signal (recommended)
parameter: "reward_scale"
drift_type: "sine"
magnitude: 0.5          # ±0.5 oscillation
period: 15000           # One cycle per 15k steps

# Sudden reward change
parameter: "reward_scale"
drift_type: "jump"
magnitude: 0.7          # +0.7 after jump
period: 250000

# Gradual reward scaling
parameter: "reward_scale"
drift_type: "linear"
magnitude: 0.6          # ±0.6 change
period: 100000
```

### Valid Ranges
- `reward_scale`: [0.1, 3.0]
- `max_steps`: [50, 300]
- `step_penalty`: [0.0, 0.1]

---

## HalfCheetah-v4 (MuJoCo)

**Description**: Control a 2D cheetah robot to run as fast as possible. Continuous control with complex physics.

### Driftable Parameters

| Parameter | Default Value | Unit | Physical Meaning | Effect on Difficulty |
|-----------|--------------|------|-----------------|---------------------|
| `friction` | 0.4 | - | Ground friction coefficient | Lower → More slippery |
| `damping` | 1.0 | - | Joint damping multiplier | Higher → Stiffer joints |
| `mass_scale` | 1.0 | - | Body mass multiplier | Higher → More inertia |
| `gravity` | -9.81 | m/s² | Gravitational acceleration | More negative → Stronger gravity |

### Typical Drift Configurations

```yaml
# Oscillating friction (recommended)
parameter: "friction"
drift_type: "sine"
magnitude: 0.3          # ±0.3 oscillation
period: 50000           # One cycle per 50k steps

# Sudden damping change
parameter: "damping"
drift_type: "jump"
magnitude: 0.5          # +0.5 after jump
period: 500000

# Gradual gravity shift
parameter: "gravity"
drift_type: "linear"
magnitude: 3.0          # ±3.0 m/s² change
period: 250000
```

### Valid Ranges
- `friction`: [0.1, 2.0]
- `damping`: [0.5, 3.0]
- `mass_scale`: [0.5, 2.0]
- `gravity`: [-20.0, -5.0] m/s²

### Notes
- Friction affects floor geom only (index 0)
- Damping scales all joint damping values uniformly
- Mass scaling affects all body masses uniformly
- Gravity is z-component only (downward)

---

## Summary Table

| Environment | Parameters | Default Algorithm | Typical Timesteps |
|-------------|-----------|-------------------|-------------------|
| **CartPole** | gravity, masscart, masspole, length | PPO | 100,000 |
| **MountainCar** | gravity, force, goal_position, goal_velocity | PPO | 1,000,000 |
| **FrozenLake** | slip_prob, reward_scale | PPO | 500,000 |
| **MiniGrid** | reward_scale, max_steps, step_penalty | PPO | 500,000 |
| **HalfCheetah** | friction, damping, mass_scale, gravity | SAC | 1,000,000 |

---

## Drift Type Reference

### Available Drift Types

1. **`static`**: No drift (baseline)
   - Parameter remains constant throughout training
   
2. **`jump`**: Sudden regime shift
   - Parameter jumps to `base_value + magnitude` at step `period`
   - Useful for testing adaptation to sudden changes

3. **`linear`**: Triangular wave
   - Parameter increases linearly to `base_value + magnitude` over `period` steps
   - Then decreases back to `base_value` over next `period` steps
   - Repeats indefinitely

4. **`sine`**: Sinusoidal oscillation
   - Parameter oscillates smoothly: `base_value + magnitude * sin(2π * t / period)`
   - Smooth, predictable drift for testing adaptation

5. **`random_walk`**: Brownian motion
   - Parameter changes by small random steps each timestep
   - Bounded by `bounds` parameter
   - Unpredictable, stochastic drift

### Choosing Drift Magnitude

**Rule of Thumb**: Magnitude should change the parameter by 20-50% of its default value for moderate difficulty.

| Environment | Parameter | Conservative | Moderate | Aggressive |
|-------------|-----------|-------------|----------|------------|
| CartPole | gravity | 3.0 | 5.0 | 10.0 |
| CartPole | masscart | 0.3 | 0.5 | 1.0 |
| MountainCar | gravity | 0.0005 | 0.001 | 0.002 |
| FrozenLake | slip_prob | 0.1 | 0.15 | 0.3 |
| MiniGrid | reward_scale | 0.3 | 0.5 | 1.0 |
| HalfCheetah | friction | 0.2 | 0.3 | 0.5 |

---

## Implementation Notes

### Accessing Drift Information

All wrappers provide drift information in the `info` dict returned by `step()`:

```python
obs, reward, terminated, truncated, info = env.step(action)

# Drift metadata
drift_step = info['drift/step']                    # Total steps
drift_params = info['drift/params']                # Parameter values
current_value = info['drift/current_value']        # Current parameter value
```

### Multi-Parameter Drift

You can drift multiple parameters simultaneously:

```python
drift_conf = [
    {'parameter': 'gravity', 'drift_type': 'sine', 'magnitude': 5.0, 'period': 10000},
    {'parameter': 'masscart', 'drift_type': 'linear', 'magnitude': 0.5, 'period': 20000},
]
env = NonStationaryCartPoleWrapper(gym.make('CartPole-v1'), drift_conf)
```

### Base Value Override

By default, the wrapper uses the environment's current value as `base_value`. You can override this:

```python
drift_conf = {
    'parameter': 'gravity',
    'drift_type': 'sine',
    'magnitude': 5.0,
    'period': 10000,
    'base_value': 12.0,  # Custom base (default is 9.8)
}
```

---

## Recommended Experimental Configurations

### Baseline (No Drift)
```yaml
env:
  drift_type: "static"
  magnitude: 0.0
```

### Moderate Drift (Recommended Starting Point)
```yaml
# CartPole
env:
  parameter: "gravity"
  drift_type: "sine"
  magnitude: 5.0
  period: 10000

# MountainCar
env:
  parameter: "gravity"
  drift_type: "sine"
  magnitude: 0.001
  period: 20000

# FrozenLake
env:
  parameter: "slip_prob"
  drift_type: "sine"
  magnitude: 0.15
  period: 20000

# MiniGrid
env:
  parameter: "reward_scale"
  drift_type: "sine"
  magnitude: 0.5
  period: 15000

# HalfCheetah
env:
  parameter: "friction"
  drift_type: "sine"
  magnitude: 0.3
  period: 50000
```

### Aggressive Drift (Stress Testing)
- Increase `magnitude` by 2x
- Decrease `period` by 2x
- Use `random_walk` or `jump` for unpredictable drift

---

## References

- **CartPole**: `src/envs/multi_env_wrappers.py` (lines 1-180)
- **MountainCar**: `src/envs/multi_env_wrappers.py` (lines 181-289)
- **FrozenLake**: `src/envs/multi_env_wrappers.py` (lines 290-429)
- **MiniGrid**: `src/envs/multi_env_wrappers.py` (lines 430-569)
- **HalfCheetah**: `src/envs/multi_env_wrappers.py` (lines 570-881)
- **Drift Generator**: `src/envs/drift_generator.py`
