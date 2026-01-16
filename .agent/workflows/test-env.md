---
description: Test environment và drift wrappers
---

# Test Environment Workflow

## Test environment setup

// turbo
1. Chạy test script để kiểm tra environment:
```bash
python scripts/test_env.py
```

## Test NS-MDMPI components

// turbo
2. Chạy test cho NS-MDMPI:
```bash
python scripts/test_nsmdmpi.py
```

## Test individual environment

// turbo
3. Test một environment cụ thể:
```bash
python test_swimmer.py
python test_lunarlander_wrapper.py
```

## Kiểm tra environment info

4. Xem thông tin chi tiết về các environments tại `env_info.md`

## Drift Patterns có sẵn

| Pattern | Mô tả |
|---------|-------|
| `static` | Không drift (baseline) |
| `sine` | Sinusoidal periodic |
| `linear` | Linear ramp |
| `jump` | Piecewise constant |
| `random_walk` | Bounded random walk |

## Driftable Parameters theo Environment

| Environment | Parameters |
|-------------|------------|
| HalfCheetah | friction, damping, gravity, mass_scale |
| Hopper | friction, damping, gravity, mass_scale |
| Walker2d | friction, damping, gravity, mass_scale |
| LunarLander | gravity, wind_power |
