"""
Drift-Adaptive Callback for Non-Stationary RL.

Implements adaptive hyperparameter scheduling based on detected environment drift.
Each algorithm has specific hyperparameters adapted:
    - PPO: learning_rate, clip_range, ent_coef
    - SAC: learning_rate, ent_coef (temperature)
    - TRPO: learning_rate, target_kl
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
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
    Implements Drift-Adaptive Scheduling for PPO/SAC/TRPO.
    
    Theoretical Basis:
        - Adjusts hyperparameters based on environmental drift magnitude.
        - Follows the "follow the drift" heuristic from the paper.
    
    Algorithm-Specific Adaptations:
    
    PPO:
        - Learning Rate: η_t = η_0 * (1 + c * Δ_t)  [increase when drifting]
        - Clip Range: ε_t = ε_0 / (1 + c * Δ_t)    [decrease when drifting]
        - Entropy Coef: α_t = α_0 * (1 + c * Δ_t)  [increase exploration when drifting]
    
    SAC:
        - Learning Rate: η_t = η_0 * (1 + c * Δ_t)
        - Entropy Coef: α_t = α_0 * (1 + c * Δ_t)  [higher entropy = more exploration]
    
    TRPO:
        - Learning Rate: η_t = η_0 * (1 + c * Δ_t)
        - Target KL: KL_t = KL_0 / (1 + c * Δ_t)   [stricter constraint when drifting]
    """
    
    def __init__(
        self,
        target_param: str = "gravity",
        base_value: float = 9.8,
        scale_factor: float = 0.1,
        min_lr_multiplier: float = 0.5,
        max_lr_multiplier: float = 3.0,
        # PPO-specific
        adapt_clip_range: bool = True,
        base_clip_range: float = 0.2,
        min_clip_range: float = 0.05,
        max_clip_range: float = 0.4,
        # Entropy adaptation (PPO/SAC)
        adapt_entropy: bool = True,
        base_ent_coef: float = 0.0,  # 0.0 = auto-detect from model
        min_ent_coef: float = 0.0,
        max_ent_coef: float = 0.1,
        # TRPO-specific
        adapt_target_kl: bool = True,
        base_target_kl: float = 0.01,
        min_target_kl: float = 0.001,
        max_target_kl: float = 0.05,
        # General
        log_freq: int = 100,
        verbose: int = 0,
    ):
        """
        Initialize the drift-adaptive callback.
        
        Args:
            target_param: Environment parameter being drifted (for logging)
            base_value: Base value of the drifting parameter
            scale_factor: c₁ in the adaptive formulas (sensitivity to drift)
            min_lr_multiplier: Minimum LR as fraction of base
            max_lr_multiplier: Maximum LR as fraction of base
            
            adapt_clip_range: Whether to adapt PPO clip range
            base_clip_range: Base clip range for PPO (default 0.2)
            min_clip_range: Minimum clip range
            max_clip_range: Maximum clip range
            
            adapt_entropy: Whether to adapt entropy coefficient
            base_ent_coef: Base entropy coefficient (0 = auto-detect)
            min_ent_coef: Minimum entropy coefficient
            max_ent_coef: Maximum entropy coefficient
            
            adapt_target_kl: Whether to adapt TRPO target KL
            base_target_kl: Base target KL for TRPO
            min_target_kl: Minimum target KL
            max_target_kl: Maximum target KL
            
            log_freq: How often to log metrics (in steps)
            verbose: Verbosity level
        """
        super(DriftAdaptiveCallback, self).__init__(verbose)
        
        # Environment tracking
        self.target_param = target_param
        self.base_value = base_value
        self.scale_factor = scale_factor
        
        # Learning rate bounds
        self.min_lr_multiplier = min_lr_multiplier
        self.max_lr_multiplier = max_lr_multiplier
        
        # PPO clip range config
        self.adapt_clip_range = adapt_clip_range
        self._base_clip_range = base_clip_range
        self.min_clip_range = min_clip_range
        self.max_clip_range = max_clip_range
        
        # Entropy config
        self.adapt_entropy = adapt_entropy
        self._base_ent_coef = base_ent_coef
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        
        # TRPO target KL config
        self.adapt_target_kl = adapt_target_kl
        self._base_target_kl = base_target_kl
        self.min_target_kl = min_target_kl
        self.max_target_kl = max_target_kl
        
        # Logging
        self.log_freq = log_freq
        
        # Runtime state (set on training start)
        self.algo_name: Optional[str] = None
        self.base_lr: float = 0.0
        self.current_lr: float = 0.0
        self.current_clip_range: float = 0.2
        self.current_ent_coef: float = 0.0
        self.current_target_kl: float = 0.01
        
        # Drift estimator
        estimator_config = DriftEstimatorConfig(
            window_size=1000,
            min_samples=100,
            ema_alpha=0.01,
        )
        self.drift_estimator = CombinedDriftEstimator(estimator_config)
        
        # Tracking
        self._last_drift_magnitude = 0.0
        self._last_adaptation_factor = 1.0

    def _on_training_start(self) -> None:
        """Initialize base values from the model."""
        # Detect algorithm type
        self.algo_name = self.model.__class__.__name__.upper()
        
        # Get base learning rate
        self.base_lr = self.model.policy.optimizer.param_groups[0]["lr"]
        self.current_lr = self.base_lr
        
        # Algorithm-specific initialization
        if self.algo_name == 'PPO':
            self._init_ppo()
        elif self.algo_name == 'SAC':
            self._init_sac()
        elif self.algo_name == 'TRPO':
            self._init_trpo()
        
        # Try to get base value from environment
        try:
            env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
            if hasattr(env, 'unwrapped'):
                self.base_value = getattr(env.unwrapped, self.target_param, self.base_value)
        except Exception:
            pass
        
        if self.verbose > 0:
            self._print_init_summary()
    
    def _init_ppo(self) -> None:
        """Initialize PPO-specific base values."""
        # Clip range
        if hasattr(self.model, 'clip_range'):
            clip_fn = self.model.clip_range
            if callable(clip_fn):
                self._base_clip_range = clip_fn(1.0)
            else:
                self._base_clip_range = float(clip_fn)
        self.current_clip_range = self._base_clip_range
        
        # Entropy coefficient
        if hasattr(self.model, 'ent_coef'):
            if self._base_ent_coef == 0.0:  # Auto-detect
                self._base_ent_coef = self.model.ent_coef
        self.current_ent_coef = self._base_ent_coef
    
    def _init_sac(self) -> None:
        """Initialize SAC-specific base values."""
        # SAC entropy coefficient (can be learned or fixed)
        if hasattr(self.model, 'ent_coef_tensor') and self.model.ent_coef_tensor is not None:
            import torch
            self._base_ent_coef = self.model.ent_coef_tensor.item()
        elif hasattr(self.model, 'ent_coef'):
            self._base_ent_coef = float(self.model.ent_coef)
        self.current_ent_coef = self._base_ent_coef
    
    def _init_trpo(self) -> None:
        """Initialize TRPO-specific base values."""
        if hasattr(self.model, 'target_kl'):
            self._base_target_kl = self.model.target_kl
        self.current_target_kl = self._base_target_kl
    
    def _print_init_summary(self) -> None:
        """Print initialization summary."""
        print(f">>> [DriftAdaptiveCallback] Training Started")
        print(f"    Algorithm: {self.algo_name}")
        print(f"    Target Param: {self.target_param} (base={self.base_value})")
        print(f"    Scale Factor: {self.scale_factor}")
        print(f"    ")
        print(f"    Adaptive Hyperparameters:")
        print(f"      - Learning Rate: {self.base_lr:.6f}")
        
        if self.algo_name == 'PPO':
            print(f"      - Clip Range: {self._base_clip_range:.3f} (adapt={self.adapt_clip_range})")
            print(f"      - Entropy Coef: {self._base_ent_coef:.4f} (adapt={self.adapt_entropy})")
        elif self.algo_name == 'SAC':
            print(f"      - Entropy Coef: {self._base_ent_coef:.4f} (adapt={self.adapt_entropy})")
        elif self.algo_name == 'TRPO':
            print(f"      - Target KL: {self._base_target_kl:.4f} (adapt={self.adapt_target_kl})")

    def _on_step(self) -> bool:
        """
        Called at every step. Adapts hyperparameters based on drift.
        """
        # 1. Get drift magnitude
        drift_magnitude = self._get_drift_magnitude()
        
        # 2. Compute adaptation factor
        adaptation_factor = 1.0 + (self.scale_factor * drift_magnitude)
        adaptation_factor = np.clip(
            adaptation_factor,
            self.min_lr_multiplier,
            self.max_lr_multiplier
        )
        
        # 3. Adapt learning rate (all algorithms)
        self._adapt_learning_rate(adaptation_factor)
        
        # 4. Algorithm-specific adaptations
        if self.algo_name == 'PPO':
            self._adapt_ppo(drift_magnitude, adaptation_factor)
        elif self.algo_name == 'SAC':
            self._adapt_sac(drift_magnitude, adaptation_factor)
        elif self.algo_name == 'TRPO':
            self._adapt_trpo(drift_magnitude, adaptation_factor)
        
        # Track
        self._last_drift_magnitude = drift_magnitude
        self._last_adaptation_factor = adaptation_factor
        
        # 5. Log periodically
        if self.n_calls % self.log_freq == 0:
            self._log_metrics(drift_magnitude, adaptation_factor)
        
        return True
    
    def _adapt_learning_rate(self, adaptation_factor: float) -> None:
        """Adapt learning rate for all algorithms."""
        new_lr = self.base_lr * adaptation_factor
        
        # Update policy optimizer
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = new_lr
        
        # SAC has separate actor/critic optimizers
        if self.algo_name == 'SAC':
            if hasattr(self.model, 'actor') and hasattr(self.model.actor, 'optimizer'):
                for param_group in self.model.actor.optimizer.param_groups:
                    param_group["lr"] = new_lr
            if hasattr(self.model, 'critic') and hasattr(self.model.critic, 'optimizer'):
                for param_group in self.model.critic.optimizer.param_groups:
                    param_group["lr"] = new_lr
        
        self.current_lr = new_lr
    
    def _adapt_ppo(self, drift_magnitude: float, adaptation_factor: float) -> None:
        """
        Adapt PPO-specific hyperparameters.
        
        Clip Range: ε_t = ε_0 / (1 + c * Δ_t)
            - Smaller trust region when environment is drifting
            - Prevents overcommitting to outdated value estimates
        
        Entropy Coef: α_t = α_0 * (1 + c * Δ_t)
            - More exploration when environment is changing
            - Helps discover new optimal actions
        """
        # Adapt clip range (inverse relationship)
        if self.adapt_clip_range and hasattr(self.model, 'clip_range'):
            # Smaller clip when drifting = more conservative updates
            inverse_factor = 1.0 / adaptation_factor
            new_clip = self._base_clip_range * inverse_factor
            new_clip = np.clip(new_clip, self.min_clip_range, self.max_clip_range)
            
            # PPO's clip_range can be a callable or float
            self.model.clip_range = lambda _progress: new_clip
            self.current_clip_range = new_clip
        
        # Adapt entropy coefficient (direct relationship)
        if self.adapt_entropy and self._base_ent_coef > 0:
            new_ent = self._base_ent_coef * adaptation_factor
            new_ent = np.clip(new_ent, self.min_ent_coef, self.max_ent_coef)
            
            if hasattr(self.model, 'ent_coef'):
                self.model.ent_coef = new_ent
            self.current_ent_coef = new_ent
    
    def _adapt_sac(self, drift_magnitude: float, adaptation_factor: float) -> None:
        """
        Adapt SAC-specific hyperparameters.
        
        Entropy Coef (Temperature): α_t = α_0 * (1 + c * Δ_t)
            - Higher temperature = more exploration
            - When environment drifts, increase exploration to find new optima
        """
        if not self.adapt_entropy:
            return
        
        # SAC entropy can be learned (log_ent_coef) or fixed
        if hasattr(self.model, 'ent_coef_tensor') and self.model.ent_coef_tensor is not None:
            import torch
            # For learned entropy, we adjust the target entropy instead
            # This is more stable than directly modifying ent_coef
            if hasattr(self.model, 'target_entropy'):
                base_target = self.model.target_entropy
                # More negative = higher entropy = more exploration
                new_target = base_target * adaptation_factor
                self.model.target_entropy = new_target
                self.current_ent_coef = new_target
        else:
            # Fixed entropy coefficient
            new_ent = self._base_ent_coef * adaptation_factor
            new_ent = np.clip(new_ent, self.min_ent_coef, self.max_ent_coef)
            
            if hasattr(self.model, 'ent_coef'):
                self.model.ent_coef = new_ent
            self.current_ent_coef = new_ent
    
    def _adapt_trpo(self, drift_magnitude: float, adaptation_factor: float) -> None:
        """
        Adapt TRPO-specific hyperparameters.
        
        Target KL: KL_t = KL_0 / (1 + c * Δ_t)
            - Stricter KL constraint when environment is drifting
            - Prevents policy from changing too fast based on outdated data
        """
        if not self.adapt_target_kl:
            return
        
        if hasattr(self.model, 'target_kl'):
            # Inverse relationship: tighter constraint when drifting
            inverse_factor = 1.0 / adaptation_factor
            new_kl = self._base_target_kl * inverse_factor
            new_kl = np.clip(new_kl, self.min_target_kl, self.max_target_kl)
            
            self.model.target_kl = new_kl
            self.current_target_kl = new_kl
    
    def _get_drift_magnitude(self) -> float:
        """Get current drift magnitude from environment."""
        try:
            if hasattr(self.training_env, 'get_attr'):
                current_values = self.training_env.get_attr(self.target_param)
                current_val = current_values[0] if current_values else self.base_value
            else:
                current_val = getattr(self.training_env.unwrapped, self.target_param, self.base_value)
            
            # Relative drift magnitude
            drift_magnitude = abs(current_val - self.base_value) / max(abs(self.base_value), 1e-8)
            return drift_magnitude
            
        except Exception as e:
            if self.verbose > 1:
                print(f"Warning: Could not get drift magnitude: {e}")
            return 0.0
    
    def _log_metrics(self, drift_magnitude: float, adaptation_factor: float) -> None:
        """Log all adaptive metrics."""
        # Get current parameter value
        try:
            if hasattr(self.training_env, 'get_attr'):
                current_values = self.training_env.get_attr(self.target_param)
                current_val = current_values[0] if current_values else self.base_value
            else:
                current_val = getattr(self.training_env.unwrapped, self.target_param, self.base_value)
        except Exception:
            current_val = self.base_value
        
        # Common metrics
        self.logger.record("adaptive/learning_rate", self.current_lr)
        self.logger.record("adaptive/base_lr", self.base_lr)
        self.logger.record("adaptive/adaptation_factor", adaptation_factor)
        self.logger.record("adaptive/drift_magnitude", drift_magnitude)
        self.logger.record("adaptive/algorithm", self.algo_name)
        self.logger.record(f"env/{self.target_param}", current_val)
        self.logger.record("env/base_value", self.base_value)
        
        # Algorithm-specific metrics
        if self.algo_name == 'PPO':
            self.logger.record("adaptive/clip_range", self.current_clip_range)
            self.logger.record("adaptive/base_clip_range", self._base_clip_range)
            if self.adapt_entropy:
                self.logger.record("adaptive/ent_coef", self.current_ent_coef)
        elif self.algo_name == 'SAC':
            self.logger.record("adaptive/ent_coef", self.current_ent_coef)
        elif self.algo_name == 'TRPO':
            self.logger.record("adaptive/target_kl", self.current_target_kl)
            self.logger.record("adaptive/base_target_kl", self._base_target_kl)
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print(f">>> [DriftAdaptiveCallback] Training Ended")
            print(f"    Final LR: {self.current_lr:.6f}")
            print(f"    Last Drift Magnitude: {self._last_drift_magnitude:.4f}")
            if self.algo_name == 'PPO':
                print(f"    Final Clip Range: {self.current_clip_range:.4f}")
            elif self.algo_name == 'TRPO':
                print(f"    Final Target KL: {self.current_target_kl:.4f}")


# Backward compatibility alias
DriftAwareClipRangeCallback = DriftAdaptiveCallback
