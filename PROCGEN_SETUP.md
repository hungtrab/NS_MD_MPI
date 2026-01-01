# Procgen Environment Setup

## Installation

```bash
# Install Procgen (requires compilation)
pip install procgen

# Note: Procgen may require specific dependencies
# If you get SIGIOT/core dump errors, try:
pip uninstall procgen
pip install procgen==0.10.7  # Known stable version
```

## Known Issues

### SIGIOT/Core Dump Error

If you see:
```
IOT instruction (core dumped)
```

**Possible causes:**
1. **Procgen not installed**: Run `pip install procgen`
2. **Version mismatch**: Procgen may have compatibility issues with your system
3. **Missing system dependencies**: Procgen requires Qt5 on Linux
4. **gym vs gymnasium**: Procgen uses old `gym` (not `gymnasium`)

**Solutions:**

1. **Check installation:**
   ```bash
   python -c "from procgen import ProcgenEnv; print('Procgen OK')"
   ```

2. **Install system dependencies (Linux):**
   ```bash
   sudo apt-get install qt5-default  # Ubuntu/Debian
   ```

3. **Use compatible version:**
   ```bash
   pip install procgen==0.10.7
   pip install gym==0.23.1  # Procgen requires old gym
   ```

4. **Skip Procgen**: If issues persist, use CartPole/MountainCar/FrozenLake instead

## Environment Configuration

Procgen environments use **procedural generation** for non-stationarity instead of physics drift:

```yaml
env_id: "procgen-coinrun-v0"

env:
  distribution_mode: "easy"    # easy/medium/hard
  num_levels: 500              # Number of unique levels (more = more diversity)
  use_backgrounds: true        # Random backgrounds (visual shift)
```

**Key difference from other environments:**
- CartPole/MountainCar: Physics parameters drift (gravity, mass, etc.)
- Procgen: Layout/visuals change via procedural generation
- No `drift_type` or `magnitude` parameters needed

## Supported Procgen Games

- `procgen-coinrun-v0`: Collect coins while avoiding enemies
- `procgen-starpilot-v0`: Shoot enemies in space
- `procgen-bossfight-v0`: Defeat bosses
- `procgen-bigfish-v0`: Eat smaller fish

## Configuration Notes

1. **Vectorized Environment**: Procgen requires `num_envs >= 1`
2. **Large Batch Size**: Use `batch_size: 2048` or higher
3. **More Timesteps**: Procgen needs 5M+ timesteps to learn
4. **No Drift Injection**: Procgen's natural diversity provides non-stationarity

## Example Training

```bash
# Baseline (no adaptation)
python scripts/train.py --config configs/Procgen_baseline.yaml

# NS-MD-MPI (with adaptation)
python scripts/train.py --config configs/Procgen_nsmdmpi.yaml
```

## Troubleshooting

### Error: "procgen package not installed"
```bash
pip install procgen
```

### Error: SIGIOT/core dump
Try older version:
```bash
pip install procgen==0.10.7 gym==0.23.1
```

### Error: "Qt platform plugin error"
Install Qt5:
```bash
sudo apt-get install qt5-default libqt5gui5
```

### Still failing?
**Recommendation**: Skip Procgen and use simpler environments:
- CartPole: Fast, stable, works everywhere
- MountainCar: Medium complexity
- FrozenLake: Discrete, stochastic

Comment out or delete Procgen configs if not needed.
