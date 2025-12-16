"""
Evaluation module for Non-Stationary MDPs.

Contains:
- Oracle Evaluator: Ground-truth optimal performance
- Dynamic Regret Calculator: Regret computation and tracking
"""

from .oracle import OracleEvaluator, OracleConfig, compute_oracle_baseline
from .dynamic_regret import (
    DynamicRegretCalculator,
    RegretConfig,
    RegretTracker,
    compare_algorithms,
)

__all__ = [
    'OracleEvaluator',
    'OracleConfig', 
    'compute_oracle_baseline',
    'DynamicRegretCalculator',
    'RegretConfig',
    'RegretTracker',
    'compare_algorithms',
]
