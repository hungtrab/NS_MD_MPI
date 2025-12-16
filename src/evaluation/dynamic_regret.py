"""
Dynamic Regret Calculator for Non-Stationary MDPs.

Computes the Dynamic Regret metric:
    DynReg(T) = Σ_{t=1}^{T} ρ_t [v_t^* - v_t^{π_t}]

Where:
    - v_t^*: Optimal value at time t (from Oracle)
    - v_t^{π_t}: Value achieved by policy π_t
    - ρ_t: Weighting factor (uniform, discounted, etc.)

Reference: Section 3 - Dynamic Regret Definition
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class RegretConfig:
    """Configuration for regret computation."""
    weighting: str = "uniform"          # uniform, discounted, or custom
    discount_factor: float = 0.99       # γ for discounted weighting
    eval_interval: int = 1000           # Interval for policy evaluation
    normalize: bool = True              # Normalize by time horizon


class DynamicRegretCalculator:
    """
    Calculates Dynamic Regret for non-stationary RL experiments.
    
    Tracks:
    - Oracle values v_t^* at evaluation points
    - Policy values v_t^{π_t} at evaluation points
    - Cumulative regret over time
    """
    
    def __init__(self, config: Optional[RegretConfig] = None):
        """
        Initialize the regret calculator.
        
        Args:
            config: RegretConfig with computation settings
        """
        self.config = config or RegretConfig()
        
        # Stored values: timestep -> value
        self.oracle_values: Dict[int, float] = {}
        self.policy_values: Dict[int, float] = {}
        
        # Instantaneous regret at each timestep
        self.instantaneous_regret: Dict[int, float] = {}
        
        # Cumulative regret (running sum)
        self.cumulative_regret: List[Tuple[int, float]] = []
        
    def set_oracle_values(self, oracle_values: Dict[int, float]) -> None:
        """
        Set the oracle (optimal) values.
        
        Args:
            oracle_values: Dictionary mapping timestep -> v_t^*
        """
        self.oracle_values = oracle_values.copy()
    
    def add_policy_value(self, timestep: int, value: float) -> float:
        """
        Add a policy evaluation result.
        
        Args:
            timestep: The timestep of evaluation
            value: The achieved value v_t^{π_t}
            
        Returns:
            Instantaneous regret at this timestep
        """
        self.policy_values[timestep] = value
        
        # Compute instantaneous regret if oracle value exists
        if timestep in self.oracle_values:
            oracle_val = self.oracle_values[timestep]
            regret = oracle_val - value
            self.instantaneous_regret[timestep] = regret
            
            # Update cumulative
            if self.cumulative_regret:
                _, prev_cumulative = self.cumulative_regret[-1]
            else:
                prev_cumulative = 0.0
            
            weighted_regret = self._get_weight(timestep) * regret
            self.cumulative_regret.append((timestep, prev_cumulative + weighted_regret))
            
            return regret
        
        return 0.0
    
    def _get_weight(self, timestep: int) -> float:
        """
        Get the weight ρ_t for a timestep.
        
        Args:
            timestep: The timestep
            
        Returns:
            Weight factor
        """
        if self.config.weighting == "uniform":
            return 1.0
        
        elif self.config.weighting == "discounted":
            return self.config.discount_factor ** timestep
        
        else:
            # Default to uniform
            return 1.0
    
    def compute_total_regret(self) -> float:
        """
        Compute the total dynamic regret.
        
        Returns:
            DynReg(T) = Σ_t ρ_t [v_t^* - v_t^{π_t}]
        """
        if not self.instantaneous_regret:
            return 0.0
        
        total = 0.0
        for t, regret in self.instantaneous_regret.items():
            weight = self._get_weight(t)
            total += weight * regret
        
        if self.config.normalize:
            total /= len(self.instantaneous_regret)
        
        return total
    
    def compute_average_regret(self) -> float:
        """
        Compute average regret per evaluation point.
        
        Returns:
            Average instantaneous regret
        """
        if not self.instantaneous_regret:
            return 0.0
        
        return np.mean(list(self.instantaneous_regret.values()))
    
    def get_regret_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the cumulative regret curve.
        
        Returns:
            Tuple of (timesteps, cumulative_regret) arrays
        """
        if not self.cumulative_regret:
            return np.array([]), np.array([])
        
        timesteps = np.array([t for t, _ in self.cumulative_regret])
        regrets = np.array([r for _, r in self.cumulative_regret])
        
        return timesteps, regrets
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get a summary of regret statistics.
        
        Returns:
            Dictionary with regret metrics
        """
        if not self.instantaneous_regret:
            return {
                'total_regret': 0.0,
                'average_regret': 0.0,
                'max_regret': 0.0,
                'min_regret': 0.0,
                'final_cumulative': 0.0,
                'n_evaluations': 0,
            }
        
        regrets = list(self.instantaneous_regret.values())
        
        return {
            'total_regret': self.compute_total_regret(),
            'average_regret': self.compute_average_regret(),
            'max_regret': max(regrets),
            'min_regret': min(regrets),
            'final_cumulative': self.cumulative_regret[-1][1] if self.cumulative_regret else 0.0,
            'n_evaluations': len(regrets),
        }
    
    def save(self, filepath: str) -> None:
        """
        Save regret data to file.
        
        Args:
            filepath: Path to save JSON file
        """
        data = {
            'config': {
                'weighting': self.config.weighting,
                'discount_factor': self.config.discount_factor,
                'eval_interval': self.config.eval_interval,
                'normalize': self.config.normalize,
            },
            'oracle_values': self.oracle_values,
            'policy_values': self.policy_values,
            'instantaneous_regret': self.instantaneous_regret,
            'cumulative_regret': self.cumulative_regret,
            'summary': self.get_summary(),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DynamicRegretCalculator':
        """
        Load regret data from file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            DynamicRegretCalculator instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = RegretConfig(**data['config'])
        calculator = cls(config)
        
        calculator.oracle_values = {int(k): v for k, v in data['oracle_values'].items()}
        calculator.policy_values = {int(k): v for k, v in data['policy_values'].items()}
        calculator.instantaneous_regret = {int(k): v for k, v in data['instantaneous_regret'].items()}
        calculator.cumulative_regret = [(t, r) for t, r in data['cumulative_regret']]
        
        return calculator
    
    def reset(self) -> None:
        """Reset all stored values."""
        self.oracle_values.clear()
        self.policy_values.clear()
        self.instantaneous_regret.clear()
        self.cumulative_regret.clear()


class RegretTracker:
    """
    Online regret tracker that can be used during training.
    
    Tracks episode returns and computes running regret against oracle.
    """
    
    def __init__(
        self,
        oracle_values: Optional[Dict[int, float]] = None,
        eval_interval: int = 1000,
    ):
        """
        Initialize the tracker.
        
        Args:
            oracle_values: Pre-computed oracle values (can be set later)
            eval_interval: Interval for regret computation
        """
        self.oracle_values = oracle_values or {}
        self.eval_interval = eval_interval
        
        # Episode tracking
        self.episode_returns: List[Tuple[int, float]] = []
        self.current_episode_return = 0.0
        self.current_timestep = 0
        
        # Regret tracking
        self.regret_calculator = DynamicRegretCalculator()
        if oracle_values:
            self.regret_calculator.set_oracle_values(oracle_values)
    
    def set_oracle_values(self, oracle_values: Dict[int, float]) -> None:
        """Set oracle values for regret computation."""
        self.oracle_values = oracle_values
        self.regret_calculator.set_oracle_values(oracle_values)
    
    def on_step(self, reward: float) -> None:
        """
        Called at each environment step.
        
        Args:
            reward: Reward received
        """
        self.current_episode_return += reward
        self.current_timestep += 1
    
    def on_episode_end(self) -> Optional[float]:
        """
        Called at the end of an episode.
        
        Returns:
            Regret at this point if at evaluation interval, else None
        """
        # Record episode return
        self.episode_returns.append((self.current_timestep, self.current_episode_return))
        
        # Check if we should compute regret
        regret = None
        if self.current_timestep % self.eval_interval < 500:  # Near interval
            # Find closest oracle value
            closest_t = min(
                self.oracle_values.keys(),
                key=lambda t: abs(t - self.current_timestep),
                default=None
            )
            
            if closest_t is not None:
                regret = self.regret_calculator.add_policy_value(
                    closest_t,
                    self.current_episode_return
                )
        
        # Reset episode return
        self.current_episode_return = 0.0
        
        return regret
    
    def get_current_cumulative_regret(self) -> float:
        """Get current cumulative regret."""
        if self.regret_calculator.cumulative_regret:
            return self.regret_calculator.cumulative_regret[-1][1]
        return 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """Get regret summary."""
        return self.regret_calculator.get_summary()


def compare_algorithms(
    results: Dict[str, DynamicRegretCalculator],
    metric: str = "total_regret",
) -> Dict[str, float]:
    """
    Compare multiple algorithms by their regret.
    
    Args:
        results: Dictionary mapping algorithm name to its regret calculator
        metric: Which metric to compare
        
    Returns:
        Dictionary mapping algorithm name to metric value
    """
    comparison = {}
    
    for name, calculator in results.items():
        summary = calculator.get_summary()
        comparison[name] = summary.get(metric, 0.0)
    
    return comparison
