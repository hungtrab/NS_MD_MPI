# Implementation Plan: Regularized Non-Stationary MDPs

## Phase 1: Environment Infrastructure ✅ (Partially Complete)

### 1.1 Drift Generator Class
- [ ] Create `src/envs/drift_generator.py` with a `DriftGenerator` class
  - [ ] Implement `piecewise_constant_jump(t, period, magnitude)` method
  - [ ] Implement `linear_ramp(t, rate, max_value)` method
  - [ ] Implement `sinusoidal(t, amplitude, frequency, phase=0)` method
  - [ ] Implement `bounded_random_walk(t, sigma, bounds)` method
  - [ ] Add unit tests in `tests/test_drift_generator.py`

### 1.2 Non-Stationary Wrapper Improvements
- [x] Base `NonStationaryCartPoleWrapper` exists in `src/envs/wrappers.py`
- [ ] Refactor wrapper to use `DriftGenerator` class instead of inline logic
- [ ] Add support for **multiple simultaneous drifting parameters**
- [ ] Implement `static` drift mode (no drift, for sanity checks)
- [ ] Add `bounded_random_walk` drift pattern
- [ ] Create `NonStationaryMuJoCoWrapper` for HalfCheetah
  - [ ] Support friction drift
  - [ ] Support damping drift

### 1.3 Configuration Schema Update
- [ ] Update `configs/cartpole_adaptive.yaml` to match new wrapper API
  - [ ] Fix mismatch: wrapper expects `drift_conf` dict, but `train.py` passes individual args
- [ ] Create `configs/halfcheetah_adaptive.yaml`

---

## Phase 2: Baseline Setup ✅ (Partially Complete)

### 2.1 Training Scripts
- [x] `scripts/train.py` with WandB integration exists
- [ ] Fix API mismatch in `make_env()`:
  ```python
  # Current (broken):
  NonStationaryCartPoleWrapper(env, drift_type=..., change_period=...)
  # Expected by wrapper:
  NonStationaryCartPoleWrapper(env, drift_conf={...})