"""
Evaluation module for Non-Stationary MDPs.

Contains:
- Oracle Evaluator: Ground-truth optimal performance
- Dynamic Regret Calculator: Regret computation and tracking
- Variation Budget Tracker: Track V_R, V_P, V_π* for NS-MD-MPI
"""

from .oracle import OracleEvaluator, OracleConfig, compute_oracle_baseline
from .dynamic_regret import (
    DynamicRegretCalculator,
    RegretConfig,
    RegretTracker,
    compare_algorithms,
)
from .variation_budgets import (
    VariationBudgetTracker,
    VariationBudgetConfig,
    estimate_budgets_from_drift_config,
)

__all__ = [
    'OracleEvaluator',
    'OracleConfig', 
    'compute_oracle_baseline',
    'DynamicRegretCalculator',
    'RegretConfig',
    'RegretTracker',
    'compare_algorithms',
    'VariationBudgetTracker',
    'VariationBudgetConfig',
    'estimate_budgets_from_drift_config',
]
