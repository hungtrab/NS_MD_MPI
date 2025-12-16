"""
Drift Generator Module for Non-Stationary MDPs.

Implements various drift patterns as described in the paper:
- Piecewise-constant jumps (regime shifts)
- Linear ramps (gradual drift)
- Sinusoidal (periodic drift)
- Bounded random walk (stochastic drift)
"""

import numpy as np
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DriftConfig:
    """Configuration for drift patterns."""
    drift_type: str = "static"      # static, jump, linear, sine, random_walk
    parameter: str = "gravity"       # Target parameter to drift
    base_value: float = 9.8          # Original parameter value
    magnitude: float = 0.0           # Drift amplitude/magnitude
    period: int = 10000              # Steps per cycle or change point
    rate: float = 0.0                # Rate for linear drift
    sigma: float = 0.1               # Std dev for random walk
    bounds: Tuple[float, float] = (0.0, 20.0)  # Bounds for random walk
    
    @classmethod
    def from_dict(cls, config: dict) -> "DriftConfig":
        """Create DriftConfig from a dictionary."""
        return cls(
            drift_type=config.get("drift_type", "static"),
            parameter=config.get("parameter", "gravity"),
            base_value=config.get("base_value", 9.8),
            magnitude=config.get("magnitude", 0.0),
            period=config.get("period", 10000),
            rate=config.get("rate", 0.0),
            sigma=config.get("sigma", 0.1),
            bounds=tuple(config.get("bounds", [0.0, 20.0])),
        )


class DriftGenerator:
    """
    Generates parameter drift values based on configured patterns.
    
    Theoretical Basis:
        - Implements drift patterns from Section 8.1 of the paper
        - Supports multiple drift regimes for comprehensive evaluation
    """
    
    def __init__(self, config: DriftConfig, seed: Optional[int] = None):
        """
        Initialize the drift generator.
        
        Args:
            config: DriftConfig object with drift parameters
            seed: Random seed for reproducibility (used in random walk)
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # State for random walk
        self._random_walk_value = config.base_value
        self._last_step = -1
        
    def get_value(self, t: int) -> float:
        """
        Get the parameter value at timestep t.
        
        Args:
            t: Current timestep
            
        Returns:
            The drifted parameter value
        """
        drift_type = self.config.drift_type
        
        if drift_type == "static":
            return self._static(t)
        elif drift_type == "jump":
            return self._piecewise_constant_jump(t)
        elif drift_type == "linear":
            return self._linear_ramp(t)
        elif drift_type == "sine":
            return self._sinusoidal(t)
        elif drift_type == "random_walk":
            return self._bounded_random_walk(t)
        else:
            raise ValueError(f"Unknown drift type: {drift_type}")
    
    def _static(self, t: int) -> float:
        """No drift - returns base value."""
        return self.config.base_value
    
    def _piecewise_constant_jump(self, t: int) -> float:
        """
        Piecewise-constant jumps (regime shifts).
        
        Pattern: Value jumps by magnitude after every period steps.
        Alternates between base_value and base_value + magnitude.
        
        Reference: Section 8.1 - "piecewise-constant jumps"
        """
        period = self.config.period
        magnitude = self.config.magnitude
        base = self.config.base_value
        
        # Determine which regime we're in
        regime = (t // period) % 2
        
        if regime == 0:
            return base
        else:
            return base + magnitude
    
    def _linear_ramp(self, t: int) -> float:
        """
        Linear ramp drift (slow gradual change).
        
        Pattern: Value increases linearly over time with optional reset.
        
        Reference: Section 8.1 - "Linear ramps with rate α/τ"
        """
        period = self.config.period
        magnitude = self.config.magnitude
        base = self.config.base_value
        
        # Calculate rate from magnitude and period
        rate = magnitude / period if period > 0 else 0
        
        # Progress within current period (for cyclical linear)
        t_in_period = t % period
        
        # Triangular wave: go up then down
        half_period = period // 2
        if t_in_period < half_period:
            return base + rate * t_in_period
        else:
            return base + magnitude - rate * (t_in_period - half_period)
    
    def _sinusoidal(self, t: int) -> float:
        """
        Sinusoidal drift (periodic oscillation).
        
        Pattern: base + magnitude * sin(2π * t / period)
        
        Reference: Section 8.1 - "Sinusoidal A sin(ωt)"
        """
        period = self.config.period
        magnitude = self.config.magnitude
        base = self.config.base_value
        
        return base + magnitude * math.sin(2 * math.pi * t / period)
    
    def _bounded_random_walk(self, t: int) -> float:
        """
        Bounded random walk drift (stochastic).
        
        Pattern: Value performs random walk with Gaussian steps, clipped to bounds.
        
        Reference: Section 8.1 - "Bounded random walk with variance σ²"
        """
        sigma = self.config.sigma
        bounds = self.config.bounds
        
        # Only update if we're moving forward in time
        if t > self._last_step:
            steps_forward = t - self._last_step
            
            # Take random walk steps
            for _ in range(steps_forward):
                step = self.rng.normal(0, sigma)
                self._random_walk_value += step
                # Clip to bounds
                self._random_walk_value = np.clip(
                    self._random_walk_value, bounds[0], bounds[1]
                )
            
            self._last_step = t
        
        return self._random_walk_value
    
    def reset(self):
        """Reset the generator state (for new episodes)."""
        self._random_walk_value = self.config.base_value
        self._last_step = -1
    
    def get_drift_info(self, t: int) -> dict:
        """
        Get detailed drift information for logging.
        
        Returns:
            Dictionary with drift metadata
        """
        current_value = self.get_value(t)
        return {
            "drift_type": self.config.drift_type,
            "parameter": self.config.parameter,
            "base_value": self.config.base_value,
            "current_value": current_value,
            "delta": current_value - self.config.base_value,
            "timestep": t,
        }


# Convenience factory function
def create_drift_generator(config: dict, seed: Optional[int] = None) -> DriftGenerator:
    """
    Factory function to create a DriftGenerator from a config dict.
    
    Args:
        config: Dictionary with drift configuration
        seed: Optional random seed
        
    Returns:
        Configured DriftGenerator instance
    """
    drift_config = DriftConfig.from_dict(config)
    return DriftGenerator(drift_config, seed=seed)
