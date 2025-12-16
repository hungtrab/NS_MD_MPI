"""
Drift-Adaptive Callback for Non-Stationary RL.

Implements adaptive hyperparameter scheduling based on detected environment drift.
Adjusts learning rate, trust region, and other parameters dynamically.
"""

import numpy as np
from typing import Optional, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.estimators import (
    CombinedDriftEstimator,
    TransitionDriftEstimator,
    RewardDriftEstimator,
    DriftEstimatorConfig,
)


class DriftAdaptiveCallback(BaseCallback):
    """
    Implements Drift-Adaptive Scheduling for PPO/SAC.
    
    Theoretical Basis:
        - Adjusts the learning step (trust region/temperature) based on environmental drift.
        - Follows the "follow the drift" heuristic: large steps when MDP moves, small steps when stable.
        - Corresponds to Algorithm 1 in the paper.
    
    Adaptive Rule (Eq. 30):
        η_t = η_0 * (1 + c₁ * ||C_t||)
        
    Where:
        - η_0: Base learning rate
        - c₁: Scale factor (sensitivity to drift)
        - ||C_t||: Bellman Commutator magnitude (drift proxy)
    """
    
    def __init__(
        self,
        target_param: str = "gravity",
        base_value: float = 9.8,
        scale_factor: float = 0.1,
        min_lr_multiplier: float = 0.5,
        max_lr_multiplier: float = 3.0,
        log_freq: int = 100,
        verbose: int = 0,
    ):
        """
        Initialize the adaptive callback.
        
        Args:
            target_param: Environment parameter being drifted (for logging)
            base_value: Base value of the drifting parameter
            scale_factor: c₁ in the adaptive formula
            min_lr_multiplier: Minimum LR as fraction of base
            max_lr_multiplier: Maximum LR as fraction of base
            log_freq: How often to log metrics (in steps)
            verbose: Verbosity level
        """
        super(DriftAdaptiveCallback, self).__init__(verbose)
        
        self.target_param = target_param
        self.base_value = base_value
        self.scale_factor = scale_factor
        self.min_lr_multiplier = min_lr_multiplier
        self.max_lr_multiplier = max_lr_multiplier
        self.log_freq = log_freq
        
        # Will be set on training start
        self.base_lr = 0.0
        self.current_lr = 0.0
        
        # Drift estimator
        estimator_config = DriftEstimatorConfig(
            window_size=1000,
            min_samples=100,
            ema_alpha=0.01,
        )
        self.drift_estimator = CombinedDriftEstimator(estimator_config)
        
        # Tracking for logging
        self._last_drift_magnitude = 0.0
        self._last_adaptation_factor = 1.0

    def _on_training_start(self) -> None:
        """
        Retrieve the initial learning rate from the optimizer.
        Called once at the beginning of training.
        """
        # Get base LR from optimizer
        self.base_lr = self.model.policy.optimizer.param_groups[0]["lr"]
        self.current_lr = self.base_lr
        
        # Try to get base value from environment
        try:
            env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
            if hasattr(env, 'unwrapped'):
                self.base_value = getattr(env.unwrapped, self.target_param, self.base_value)
        except Exception:
            pass
        
        if self.verbose > 0:
            print(f">>> [Callback] Adaptive Training Started")
            print(f"    Base LR: {self.base_lr}")
            print(f"    Target Param: {self.target_param} (base={self.base_value})")
            print(f"    Scale Factor: {self.scale_factor}")

    def _on_step(self) -> bool:
        """
        Executed at every step. 
        1. Detects current drift level (Oracle/Proxy).
        2. Adjusts Hyperparameters (Learning Rate).
        3. Logs to WandB/TensorBoard.
        """
        # 1. Get drift information from environment
        drift_magnitude = self._get_drift_magnitude()
        
        # 2. Update drift estimator with reward info
        # Note: Full estimator update would require value predictions
        # For now, we use the direct drift proxy from the environment
        
        # 3. Compute adaptive learning rate
        # Formula: η_t = η_0 * (1 + c₁ * drift_magnitude)
        adaptation_factor = 1.0 + (self.scale_factor * drift_magnitude)
        
        # Clip to bounds
        adaptation_factor = np.clip(
            adaptation_factor,
            self.min_lr_multiplier,
            self.max_lr_multiplier
        )
        
        new_lr = self.base_lr * adaptation_factor
        
        # 4. Update optimizer learning rate
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = new_lr
        
        self.current_lr = new_lr
        self._last_drift_magnitude = drift_magnitude
        self._last_adaptation_factor = adaptation_factor

        # 5. Logging (throttled to reduce overhead)
        if self.n_calls % self.log_freq == 0:
            self._log_metrics(drift_magnitude, adaptation_factor)
        
        return True
    
    def _get_drift_magnitude(self) -> float:
        """
        Get current drift magnitude from environment.
        
        Uses the drift proxy from the NonStationaryWrapper if available,
        otherwise estimates from observed data.
        
        Returns:
            Drift magnitude (absolute deviation from base value)
        """
        try:
            # Try to get from VecEnv
            if hasattr(self.training_env, 'get_attr'):
                current_values = self.training_env.get_attr(self.target_param)
                current_val = current_values[0] if current_values else self.base_value
            else:
                # Single env
                current_val = getattr(self.training_env.unwrapped, self.target_param, self.base_value)
            
            # Compute relative drift magnitude
            drift_magnitude = abs(current_val - self.base_value) / max(abs(self.base_value), 1e-8)
            return drift_magnitude
            
        except Exception as e:
            if self.verbose > 1:
                print(f"Warning: Could not get drift magnitude: {e}")
            return 0.0
    
    def _log_metrics(self, drift_magnitude: float, adaptation_factor: float) -> None:
        """Log adaptation metrics to the logger."""
        # Get current parameter value for logging
        try:
            if hasattr(self.training_env, 'get_attr'):
                current_values = self.training_env.get_attr(self.target_param)
                current_val = current_values[0] if current_values else self.base_value
            else:
                current_val = getattr(self.training_env.unwrapped, self.target_param, self.base_value)
        except Exception:
            current_val = self.base_value
        
        # Log to SB3 logger (syncs to WandB via TensorBoard)
        self.logger.record("adaptive/learning_rate", self.current_lr)
        self.logger.record("adaptive/base_lr", self.base_lr)
        self.logger.record("adaptive/adaptation_factor", adaptation_factor)
        self.logger.record("adaptive/drift_magnitude", drift_magnitude)
        self.logger.record(f"env/{self.target_param}", current_val)
        self.logger.record("env/base_value", self.base_value)
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print(f">>> [Callback] Adaptive Training Ended")
            print(f"    Final LR: {self.current_lr}")
            print(f"    Last Drift Magnitude: {self._last_drift_magnitude}")


class DriftAwareClipRangeCallback(BaseCallback):
    """
    Adaptive clip range for PPO based on drift magnitude.
    
    Adjusts the PPO clip range (ε) inversely to drift:
    - High drift → smaller clip range (more conservative updates)
    - Low drift → larger clip range (more aggressive updates)
    
    Formula: ε_t = ε_0 / (1 + c₂ * Δ_t^P)
    """
    
    def __init__(
        self,
        target_param: str = "gravity",
        base_value: float = 9.8,
        scale_factor: float = 0.5,
        min_clip: float = 0.05,
        max_clip: float = 0.3,
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        
        self.target_param = target_param
        self.base_value = base_value
        self.scale_factor = scale_factor
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.log_freq = log_freq
        
        self.base_clip_range = 0.2  # PPO default
        self.current_clip_range = 0.2
    
    def _on_training_start(self) -> None:
        # Get base clip range from model
        if hasattr(self.model, 'clip_range'):
            clip_range = self.model.clip_range
            if callable(clip_range):
                self.base_clip_range = clip_range(1.0)
            else:
                self.base_clip_range = clip_range
        
        if self.verbose > 0:
            print(f">>> [Callback] Adaptive Clip Range Started. Base: {self.base_clip_range}")
    
    def _on_step(self) -> bool:
        # Get drift magnitude
        drift_magnitude = self._get_drift_magnitude()
        
        # Inverse relationship: higher drift → smaller clip
        adaptation_factor = 1.0 / (1.0 + self.scale_factor * drift_magnitude)
        
        new_clip = self.base_clip_range * adaptation_factor
        new_clip = np.clip(new_clip, self.min_clip, self.max_clip)
        
        # Update model's clip range (if supported)
        if hasattr(self.model, 'clip_range'):
            self.model.clip_range = lambda _: new_clip
        
        self.current_clip_range = new_clip
        
        if self.n_calls % self.log_freq == 0:
            self.logger.record("adaptive/clip_range", new_clip)
        
        return True
    
    def _get_drift_magnitude(self) -> float:
        try:
            if hasattr(self.training_env, 'get_attr'):
                current_values = self.training_env.get_attr(self.target_param)
                current_val = current_values[0] if current_values else self.base_value
            else:
                current_val = getattr(self.training_env.unwrapped, self.target_param, self.base_value)
            
            return abs(current_val - self.base_value) / max(abs(self.base_value), 1e-8)
        except Exception:
            return 0.0