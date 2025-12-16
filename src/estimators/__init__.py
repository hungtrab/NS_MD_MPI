"""
Drift Estimators for Non-Stationary MDPs.

Implements estimators for:
- Reward Drift (Δ_R): Changes in reward distribution
- Transition Drift (Δ_P): Changes in environment dynamics
- Bellman Commutator: Proxy for policy staleness

These estimators are used by adaptive algorithms to detect non-stationarity
and adjust hyperparameters accordingly.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DriftEstimatorConfig:
    """Configuration for drift estimators."""
    window_size: int = 1000          # Sliding window for statistics
    min_samples: int = 100           # Minimum samples before estimation
    ema_alpha: float = 0.01          # EMA smoothing factor
    threshold: float = 0.1           # Significance threshold


class RewardDriftEstimator:
    """
    Estimates reward drift Δ̂_t^R.
    
    Tracks changes in the reward distribution over time using a sliding window.
    
    Methods:
        - Sliding window mean comparison
        - Exponential Moving Average (EMA) tracking
        - Variance-normalized drift detection
    
    Reference: Section 7.2 - Reward drift estimation
    """
    
    def __init__(self, config: Optional[DriftEstimatorConfig] = None):
        self.config = config or DriftEstimatorConfig()
        
        # Recent rewards (current window)
        self.recent_rewards = deque(maxlen=self.config.window_size)
        # Historical rewards (previous window for comparison)
        self.historical_rewards = deque(maxlen=self.config.window_size)
        
        # EMA tracking
        self.ema_reward = None
        self.ema_variance = None
        
        # Statistics
        self.total_rewards = 0
        self.n_samples = 0
        
    def update(self, reward: float) -> None:
        """
        Update estimator with new reward observation.
        
        Args:
            reward: The observed reward
        """
        # Move oldest recent to historical before adding new
        if len(self.recent_rewards) == self.config.window_size:
            oldest = self.recent_rewards[0]
            self.historical_rewards.append(oldest)
        
        self.recent_rewards.append(reward)
        self.total_rewards += reward
        self.n_samples += 1
        
        # Update EMA
        alpha = self.config.ema_alpha
        if self.ema_reward is None:
            self.ema_reward = reward
            self.ema_variance = 0.0
        else:
            delta = reward - self.ema_reward
            self.ema_reward = self.ema_reward + alpha * delta
            self.ema_variance = (1 - alpha) * (self.ema_variance + alpha * delta ** 2)
    
    def estimate_drift(self) -> float:
        """
        Estimate current reward drift magnitude.
        
        Returns:
            Δ̂_t^R: Estimated reward drift (normalized)
        """
        if len(self.recent_rewards) < self.config.min_samples:
            return 0.0
        
        if len(self.historical_rewards) < self.config.min_samples:
            return 0.0
        
        # Compute means
        recent_mean = np.mean(self.recent_rewards)
        historical_mean = np.mean(self.historical_rewards)
        
        # Compute pooled standard deviation for normalization
        recent_std = np.std(self.recent_rewards)
        historical_std = np.std(self.historical_rewards)
        pooled_std = np.sqrt((recent_std**2 + historical_std**2) / 2)
        
        if pooled_std < 1e-8:
            return 0.0
        
        # Normalized drift (similar to effect size)
        drift = abs(recent_mean - historical_mean) / pooled_std
        
        return drift
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        return {
            'n_samples': self.n_samples,
            'ema_reward': self.ema_reward or 0.0,
            'ema_variance': self.ema_variance or 0.0,
            'recent_mean': np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            'drift_magnitude': self.estimate_drift(),
        }
    
    def reset(self) -> None:
        """Reset the estimator state."""
        self.recent_rewards.clear()
        self.historical_rewards.clear()
        self.ema_reward = None
        self.ema_variance = None
        self.total_rewards = 0
        self.n_samples = 0


class TransitionDriftEstimator:
    """
    Estimates transition drift Δ̂_t^P.
    
    Uses environment-provided drift information (Oracle) or estimates
    from observed state transitions.
    
    Methods:
        - Direct proxy from environment info (if available)
        - TD-error based estimation
        - State distribution shift detection
    
    Reference: Section 7.2 - Transition drift estimation
    """
    
    def __init__(self, config: Optional[DriftEstimatorConfig] = None):
        self.config = config or DriftEstimatorConfig()
        
        # Track drift proxy values from environment
        self.drift_values = deque(maxlen=self.config.window_size)
        
        # Track TD errors as proxy for transition changes
        self.td_errors = deque(maxlen=self.config.window_size)
        
        # Base value for computing relative drift
        self.base_value = None
        
        # EMA of drift
        self.ema_drift = 0.0
        
    def update_from_info(self, info: Dict[str, Any]) -> None:
        """
        Update estimator using environment info dict.
        
        Args:
            info: Step info dict from environment (contains drift/current_value)
        """
        if 'drift/current_value' in info:
            current_val = info['drift/current_value']
            
            if self.base_value is None:
                self.base_value = current_val
            
            drift = abs(current_val - self.base_value)
            self.drift_values.append(drift)
            
            # Update EMA
            alpha = self.config.ema_alpha
            self.ema_drift = (1 - alpha) * self.ema_drift + alpha * drift
    
    def update_from_td_error(self, td_error: float) -> None:
        """
        Update estimator using TD error.
        
        Large TD errors may indicate transition distribution changes.
        
        Args:
            td_error: Temporal difference error from value function update
        """
        self.td_errors.append(abs(td_error))
    
    def estimate_drift(self) -> float:
        """
        Estimate current transition drift magnitude.
        
        Returns:
            Δ̂_t^P: Estimated transition drift
        """
        if len(self.drift_values) > 0:
            # Use direct drift values if available
            return self.ema_drift
        
        if len(self.td_errors) < self.config.min_samples:
            return 0.0
        
        # Use TD error as proxy
        recent_td = list(self.td_errors)[-self.config.min_samples:]
        return np.mean(recent_td)
    
    def estimate_drift_rate(self) -> float:
        """
        Estimate rate of drift change (second derivative).
        
        Useful for detecting acceleration in non-stationarity.
        
        Returns:
            Rate of drift change
        """
        if len(self.drift_values) < self.config.min_samples * 2:
            return 0.0
        
        values = list(self.drift_values)
        half = len(values) // 2
        
        first_half_mean = np.mean(values[:half])
        second_half_mean = np.mean(values[half:])
        
        return second_half_mean - first_half_mean
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        return {
            'ema_drift': self.ema_drift,
            'drift_magnitude': self.estimate_drift(),
            'drift_rate': self.estimate_drift_rate(),
            'base_value': self.base_value or 0.0,
            'n_samples': len(self.drift_values),
        }
    
    def reset(self) -> None:
        """Reset the estimator state."""
        self.drift_values.clear()
        self.td_errors.clear()
        self.base_value = None
        self.ema_drift = 0.0


class BellmanCommutatorEstimator:
    """
    Estimates the Bellman Commutator magnitude.
    
    The commutator C_t = T_{t+1} V_t - T_t V_t measures how much the
    optimal value function changes due to environment dynamics shift.
    
    We estimate this using:
    - TD error variance (proxy for value function instability)
    - Value function prediction error across time
    
    Reference: Section 6 - Bellman Commutator and Eq. 27
    """
    
    def __init__(self, config: Optional[DriftEstimatorConfig] = None):
        self.config = config or DriftEstimatorConfig()
        
        # Track value predictions and actuals
        self.value_predictions = deque(maxlen=self.config.window_size)
        self.value_targets = deque(maxlen=self.config.window_size)
        
        # TD errors for variance estimation
        self.td_errors = deque(maxlen=self.config.window_size)
        
        # EMA of commutator proxy
        self.ema_commutator = 0.0
        
    def update(
        self, 
        value_pred: float, 
        value_target: float,
        td_error: Optional[float] = None
    ) -> None:
        """
        Update estimator with value function data.
        
        Args:
            value_pred: Predicted value V(s)
            value_target: Target value (r + γV(s'))
            td_error: Optional pre-computed TD error
        """
        self.value_predictions.append(value_pred)
        self.value_targets.append(value_target)
        
        if td_error is None:
            td_error = value_target - value_pred
        
        self.td_errors.append(td_error)
        
        # Update EMA of TD error magnitude
        alpha = self.config.ema_alpha
        self.ema_commutator = (1 - alpha) * self.ema_commutator + alpha * abs(td_error)
    
    def estimate_commutator(self) -> float:
        """
        Estimate Bellman Commutator magnitude ||C_t||.
        
        Uses TD error variance as proxy.
        
        Returns:
            Estimated commutator magnitude
        """
        if len(self.td_errors) < self.config.min_samples:
            return 0.0
        
        # Commutator proxy: variance of TD errors
        # High variance indicates unstable value function (due to drift)
        td_variance = np.var(self.td_errors)
        td_mean = np.mean(np.abs(self.td_errors))
        
        # Combine mean and variance for robust estimate
        return td_mean + 0.5 * np.sqrt(td_variance)
    
    def estimate_from_drift(self, drift_magnitude: float, lipschitz_const: float = 1.0) -> float:
        """
        Estimate commutator from transition drift.
        
        Uses the bound: ||C_t|| ≤ L * Δ_t^P (Lipschitz assumption)
        
        Args:
            drift_magnitude: Estimated transition drift
            lipschitz_const: Lipschitz constant of value function
            
        Returns:
            Upper bound on commutator magnitude
        """
        return lipschitz_const * drift_magnitude
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        if len(self.td_errors) > 0:
            td_errors = list(self.td_errors)
            return {
                'ema_commutator': self.ema_commutator,
                'commutator_magnitude': self.estimate_commutator(),
                'td_mean': np.mean(np.abs(td_errors)),
                'td_std': np.std(td_errors),
                'n_samples': len(self.td_errors),
            }
        return {
            'ema_commutator': 0.0,
            'commutator_magnitude': 0.0,
            'td_mean': 0.0,
            'td_std': 0.0,
            'n_samples': 0,
        }
    
    def reset(self) -> None:
        """Reset the estimator state."""
        self.value_predictions.clear()
        self.value_targets.clear()
        self.td_errors.clear()
        self.ema_commutator = 0.0


class CombinedDriftEstimator:
    """
    Combined estimator that aggregates all drift signals.
    
    Provides a unified interface for drift detection using:
    - Reward drift (Δ_R)
    - Transition drift (Δ_P)
    - Bellman Commutator (C_t)
    """
    
    def __init__(self, config: Optional[DriftEstimatorConfig] = None):
        self.config = config or DriftEstimatorConfig()
        
        self.reward_estimator = RewardDriftEstimator(self.config)
        self.transition_estimator = TransitionDriftEstimator(self.config)
        self.commutator_estimator = BellmanCommutatorEstimator(self.config)
    
    def update(
        self,
        reward: float,
        info: Dict[str, Any],
        value_pred: Optional[float] = None,
        value_target: Optional[float] = None,
    ) -> None:
        """
        Update all estimators with step data.
        
        Args:
            reward: Observed reward
            info: Environment info dict
            value_pred: Optional value prediction
            value_target: Optional value target
        """
        self.reward_estimator.update(reward)
        self.transition_estimator.update_from_info(info)
        
        if value_pred is not None and value_target is not None:
            self.commutator_estimator.update(value_pred, value_target)
    
    def get_total_drift(self, weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """
        Get weighted sum of all drift signals.
        
        Args:
            weights: (w_R, w_P, w_C) weights for reward, transition, commutator
            
        Returns:
            Combined drift magnitude
        """
        w_r, w_p, w_c = weights
        
        drift_r = self.reward_estimator.estimate_drift()
        drift_p = self.transition_estimator.estimate_drift()
        drift_c = self.commutator_estimator.estimate_commutator()
        
        return w_r * drift_r + w_p * drift_p + w_c * drift_c
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all estimators."""
        return {
            'reward': self.reward_estimator.get_stats(),
            'transition': self.transition_estimator.get_stats(),
            'commutator': self.commutator_estimator.get_stats(),
            'total_drift': self.get_total_drift(),
        }
    
    def reset(self) -> None:
        """Reset all estimators."""
        self.reward_estimator.reset()
        self.transition_estimator.reset()
        self.commutator_estimator.reset()
