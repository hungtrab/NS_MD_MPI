# NS-MDMPI Old Experiment Results Analysis

**Analysis Date:** 2026-01-13  
**Data Source:** nsmdmpi_exp.csv (100 completed experiments)

---

## 📊 Executive Summary

> [!WARNING]
> **Key Finding:** NS-MDMPI showed **NO statistically significant improvement** over baseline methods in these experiments. In fact, performance was slightly worse across all environments.

### Quick Stats
- **Environments Tested:** 4 (MountainCar, CartPole, HalfCheetah, MiniGrid)
- **Total Experiments:** 100 (50 Baseline + 50 NS-MDMPI)
- **Seeds per Config:** 5
- **Mean Improvement:** **-154.43** (worse)
- **Statistical Significance:** **None** (all p-values > 0.05)

---

## 📈 Detailed Results

### Performance Comparison by Environment

| Environment | Drift Type | Parameter | Baseline Mean | NS-MDMPI Mean | Improvement | Change (%) | p-value | Significance |
|-------------|------------|-----------|---------------|---------------|-------------|------------|---------|--------------|
| **CartPole** | jump | gravity | 500.00 | 500.00 | 0.00 | 0.00% | 0.069 | ns |
| **CartPole** | sine | gravity | 500.00 | 498.73 | -1.27 | -0.25% | 0.069 | ns |
| **HalfCheetah** | linear | gravity | 9428.56 | 9175.38 | -253.18 | -2.69% | 0.177 | ns |
| **HalfCheetah** | sine | friction | 10281.12 | 9349.51 | **-931.61** | **-9.06%** | 0.177 | ns |
| **MiniGrid** | jump | reward | 0.00 | 0.08 | +0.08 | - | 0.715 | ns |
| **MiniGrid** | sine | reward | 0.00 | -0.01 | -0.01 | - | 0.715 | ns |
| **MountainCar** | jump | force | -108.82 | -131.59 | -22.76 | -20.92% | 0.312 | ns |
| **MountainCar** | sine | gravity | -156.38 | -183.08 | -26.70 | -17.07% | 0.312 | ns |

**Legend:** ns = not significant (p > 0.05)

---

## 🔍 Statistical Analysis

### T-Test Results (Per Environment)

**MountainCar:**
```
t-statistic: -1.041
p-value: 0.3118 (ns)
Cohen's d: -0.491 (medium negative effect)
Baseline: -132.60 ± 47.82
NS-MDMPI: -157.33 ± 52.87
```

**MiniGrid:**
```
t-statistic: 0.371
p-value: 0.7147 (ns)
Cohen's d: 0.175 (small positive effect)
Baseline: 0.00 ± 0.00
NS-MDMPI: 0.03 ± 0.27
```

**HalfCheetah:**
```
t-statistic: -1.405
p-value: 0.1770 (ns)
Cohen's d: -0.662 (medium-large negative effect)
Baseline: 9854.84 ± 773.10
NS-MDMPI: 9262.44 ± 1001.12
```

**CartPole:**
```
t-statistic: -1.931
p-value: 0.0694 (ns, close to significance)
Cohen's d: -0.910 (large negative effect)
Baseline: 500.00 ± 0.00
NS-MDMPI: 499.37 ± 0.98
```

---

## 💡 Key Insights

### 1. **No Significant Improvement**
- All p-values > 0.05 (no statistical significance)
- Mean improvement across all experiments: **-154.43** (negative)
- Median improvement: **-12.02** (negative)

### 2. **Worst Case: HalfCheetah**
- **-931.61 reward drop** with sine friction drift
- **-9.06% performance degradation**
- Suggests NS-MDMPI may struggle with high-dimensional continuous control

### 3. **Best Case: MiniGrid**
- Tiny positive improvement (+0.08)
- But baseline already at 0, so improvement is minimal
- Not statistically significant

### 4. **CartPole: Almost Significant**
- p-value = 0.0694 (close to 0.05 threshold)
- But improvement is **negative** (499.37 vs 500.00)
- Large negative effect size (Cohen's d = -0.910)

---

## 🤔 Possible Reasons for Poor Performance

1. **Hyperparameter Suboptimal**
   - Variation budgets (V_R=10, V_P=10, V_π*=5) may be too conservative
   - Trust region/regularization params not well-tuned
   - This is why hyperparameter tuning (Objective 2 in plan) is critical!

2. **Algorithm Implementation Issues**
   - Budget consumption might be inefficient
   - Drift estimators may not be sensitive enough
   - Adaptation mechanisms might be too slow

3. **Environment Mismatch**
   - NS-MDMPI designed for non-stationary environments
   - Some test environments may not have strong enough drift
   - Algorithm overhead without benefits in mild drift scenarios

4. **Insufficient Training**
   - Total timesteps may be too short for adaptation to kick in
   - NS-MDMPI needs time to learn drift patterns

---

## 📁 Generated Files

- `nsmdmpi_exp_processed.csv` - Cleaned experiment data
- `analysis/comparison_summary.csv` - Statistical comparison table
- `analysis/figures/performance_comparison.png` - 4-panel visualization
- `analysis/figures/improvement_summary.png` - Improvement bar chart

---

## 🎯 Next Steps & Recommendations

> [!IMPORTANT]
> **Critical Action Items:**
> 
> 1. **Run Hyperparameter Tuning** (as planned in Objective 1)
>    - Current hyperparameters are clearly not optimal
>    - Use Optuna to find better budgets and sensitivity params
> 
> 2. **Validate on Current Codebase**
>    - These old experiments may be from buggy/outdated code
>    - Re-run key experiments with current fixed codebase
>    - Verify NS-MDMPI callback is working correctly
> 
> 3. **Test on Stronger Drift**
>    - Try more extreme drift magnitudes
>    - Test multi-parameter drift (more complex scenarios)
>    - Verify NS-MDMPI shines in high-drift environments
> 
> 4. **Analyze Budget Consumption**
>    - Check if budgets are being used efficiently
>    - Verify drift estimators are detecting changes
>    - Monitor kappa_t and lambda_t adaptation

---

## 📊 Visualizations

See `analysis/figures/` for detailed plots:
- Performance comparison across environments
- Improvement percentage breakdown
- Distribution analysis
- Baseline vs NS-MDMPI scatter plots

---

**Conclusion:** Based on these old experiments, NS-MDMPI did not outperform baselines. However, this may be due to:
1. Suboptimal hyperparameters
2. Outdated/buggy implementation
3. Insufficiently challenging test scenarios

**Proceed with hyperparameter tuning and re-validation** before drawing final conclusions.
