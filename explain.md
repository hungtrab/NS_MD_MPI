# Drift Magnitude — Explanation and Formulas

This document explains how the training code computes the "drift magnitude" used by the drift-adaptive callback, and how that quantity is converted into hyperparameter adaptations (learning rate, PPO clip range, entropy coefficient, TRPO target KL).

## 1) What is the drift magnitude?

Drift magnitude is a unitless scalar that measures how far the current value of the environment parameter being drifted (e.g., gravity, force, goal_position) is from its base (nominal) value.

The implementation in `DriftAdaptiveCallback` obtains the current parameter value from the running environment and computes a relative change from the base value. The core formula used is:

```
drift_magnitude = abs(current_value - base_value) / max(abs(base_value), 1e-8)
```

- `current_value` is read from the environment (for vectorized envs the first worker's value is used). If the attribute is missing, the callback falls back to `base_value` (so drift becomes 0).
- The `max(abs(base_value), 1e-8)` denominator avoids division by zero when `base_value` is 0 or extremely small.

Interpretation: `drift_magnitude` is the fractional change of the parameter relative to its baseline. Example: if `base_value = 0.0025` (gravity in MountainCar) and `current_value = 0.005`, then

```
drift_magnitude = (0.005 - 0.0025) / 0.0025 = 1.0
```

This means the parameter doubled (100% increase) relative to the base.

## 2) How the drift magnitude is turned into adaptation factors

The callback uses a simple linear scaling (the "follow the drift" heuristic) to compute an adaptation factor from the drift magnitude:

```
adaptation_factor = 1 + scale_factor * drift_magnitude
```

- `scale_factor` (named `c` in the code/config) controls sensitivity. Typical values in configs are 0 (disabled), 0.05, 0.1, etc.
- `adaptation_factor` > 1 increases the parameter multiplicatively; values < 1 are possible if `scale_factor` is negative (not common).

After computing `adaptation_factor`, each hyperparameter is updated according to an algorithm-specific rule and then clipped to prevent extreme changes.

## 3) Algorithm-specific adaptation formulas

- Learning rate (all algorithms):

```
new_lr = clip(base_lr * adaptation_factor, base_lr * min_lr_multiplier, base_lr * max_lr_multiplier)
```

This increases learning rate when drift is high (more aggressive updates), and lowers it when the config sets `scale_factor` < 0 or drift is negative (rare).

- PPO clip range (trust region):

```
new_clip = clip(base_clip_range / (1 + scale_factor * drift_magnitude), min_clip_range, max_clip_range)
```

Rationale: when the environment is changing a lot, shrink the clip range (i.e., take more conservative policy updates) by dividing the base clip by the same factor that increases LR. The code ensures the result stays within `[min_clip_range, max_clip_range]`.

- Entropy coefficient (PPO and SAC):

```
new_ent_coef = clip(base_ent_coef * (1 + scale_factor * drift_magnitude), min_ent_coef, max_ent_coef)
```

This increases entropy weight (encourages exploration) when drift is high. Note: if `base_ent_coef == 0.0` meaning "auto", adapt_entropy may do nothing or must be handled specially (the callback supports a 0 = auto-detect convention).

- TRPO target KL:

```
new_target_kl = clip(base_target_kl / (1 + scale_factor * drift_magnitude), min_target_kl, max_target_kl)
```

Rationale: stricter KL constraint (smaller KL) when drift is high to avoid large policy steps that may destabilize training in a changing environment.

## 4) Example numeric calculation

Given:
- `base_lr = 3e-4`
- `base_clip_range = 0.2`
- `scale_factor = 0.1`
- `base_value = 0.0025` (gravity)
- `current_value = 0.005` → `drift_magnitude = 1.0`

Compute:

- `adaptation_factor = 1 + 0.1 * 1.0 = 1.1`
- `new_lr = 3e-4 * 1.1 = 3.3e-4` (then clipped to the configured min/max multipliers if needed)
- `new_clip = 0.2 / (1 + 0.1 * 1.0) = 0.2 / 1.1 ≈ 0.1818`

So the learning rate is modestly increased and the PPO clip is slightly tightened.

## 5) Implementation notes and caveats

- The current `DriftAdaptiveCallback` constructs a `CombinedDriftEstimator` (and other estimators), but in the present code the `_get_drift_magnitude()` function reads the environment parameter directly (the simple relative-change formula above). That means adaptation is driven by the explicit parameter drift (e.g., gravity), not by observed reward/transition statistics.
- For a more robust signal you can (and should) use the estimator outputs: update the estimator from observed transitions and use its magnitude (e.g., combined reward+transition drift). This captures functional change even when the underlying parameter is not directly exposed or when multiple parameters drift together.
- Vectorized environments: when training uses multiple envs (VecEnv), the callback typically reads the attribute from the first worker. If workers drift independently, consider aggregating (mean/max) the attribute across workers.
- Base-value near zero: the division uses `max(abs(base_value), 1e-8)` to avoid a divide-by-zero; when `base_value` is zero this makes `drift_magnitude` effectively `abs(current_value)/1e-8` which is enormous — such cases should be handled by choosing a sensible base or switching to absolute-difference-based magnitude for parameters that naturally sit near zero.

## 6) Practical tips

- Tune `scale_factor` conservatively (e.g., 0.05–0.2) to avoid oscillatory, overly large hyperparameter changes.
- Use clipping limits (`min_*` / `max_*`) to bound the adaptation and ensure stable training.
- If the parameter being drifted has very different scales across environments (e.g., gravity vs. goal_position), prefer normalizing externally or using learned/statistical estimators rather than raw relative change.

---
If you want, I can:
- Patch `DriftAdaptiveCallback` to use `CombinedDriftEstimator` output (observed-drift) instead of direct parameter read, and add a small test script demonstrating adaptation values over a few steps.
- Add a helper that computes a safe, bounded drift magnitude when `base_value` ≈ 0.

End of document.
