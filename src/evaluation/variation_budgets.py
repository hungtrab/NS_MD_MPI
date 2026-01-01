"""
Variation Budget Tracker for NS-MD-MPI.

Tracks variation budgets V_R, V_P, V_π* as defined in the paper:
    - V_R: Reward variation budget (cumulative reward function changes)
    - V_P: Transition variation budget (cumulative dynamics changes)
    - V_π*: Optimal policy variation budget (cumulative optimal policy changes)

These budgets bound the total non-stationarity over the time horizon and
are used to adapt the trust region and regularization in Algorithm 1.

Reference: Section 3 - Variation Budgets
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class VariationBudgetConfig:
    """Configuration for variation budgets."""
    V_R: float = 10.0           # Reward variation budget
    V_P: float = 10.0           # Transition variation budget
    V_pi_star: float = 5.0      # Optimal policy variation budget
    
    # Budget initialization strategy
    auto_scale: bool = True     # Auto-scale based on drift config
    scale_factor: float = 1.0   # Multiplier for auto-scaling


class VariationBudgetTracker:
    """
    Tracks variation budgets V_R, V_P, V_π* over the training horizon.
    
    The budgets represent cumulative bounds on environment non-stationarity:
    - V_R = Σ_t ||r_t - r_{t-1}||
    - V_P = Σ_t ||P_t - P_{t-1}||
    - V_π* = Σ_t ||π*_t - π*_{t-1}||
    
    As drift is observed, budgets are consumed. The remaining budget
    informs the trust region size in Algorithm 1.
    """
    
    def __init__(
        self,
        V_R: float,
        V_P: float,
        V_pi_star: float,
        time_horizon: Optional[int] = None,
    ):
        """
        Initialize variation budget tracker.
        
        Args:
            V_R: Total reward variation budget
            V_P: Total transition variation budget
            V_pi_star: Total optimal policy variation budget
            time_horizon: Expected time horizon (for normalization)
        """
        # Total budgets
        self.V_R_total = V_R
        self.V_P_total = V_P
        self.V_pi_star_total = V_pi_star
        
        # Remaining budgets (depleted as drift is observed)
        self.V_R_remaining = V_R
        self.V_P_remaining = V_P
        self.V_pi_star_remaining = V_pi_star
        
        # Consumption history: timestep -> (delta_R, delta_P, delta_pi)
        self.consumption_history: List[Tuple[int, float, float, float]] = []
        
        # Cumulative consumption
        self.V_R_consumed = 0.0
        self.V_P_consumed = 0.0
        self.V_pi_star_consumed = 0.0
        
        # Time tracking
        self.current_timestep = 0
        self.time_horizon = time_horizon
        
    def update(
        self,
        delta_R: float,
        delta_P: float,
        delta_pi: float,
        timestep: Optional[int] = None,
    ) -> None:
        """
        Consume budget based on observed drift.
        
        Args:
            delta_R: Observed reward drift magnitude
            delta_P: Observed transition drift magnitude
            delta_pi: Observed policy drift magnitude (proxy via commutator)
            timestep: Current timestep (optional, auto-increments if None)
        """
        # Ensure non-negative drift
        delta_R = max(0.0, abs(delta_R))
        delta_P = max(0.0, abs(delta_P))
        delta_pi = max(0.0, abs(delta_pi))
        
        # Update remaining budgets (don't go negative)
        self.V_R_remaining = max(0.0, self.V_R_remaining - delta_R)
        self.V_P_remaining = max(0.0, self.V_P_remaining - delta_P)
        self.V_pi_star_remaining = max(0.0, self.V_pi_star_remaining - delta_pi)
        
        # Update consumed
        self.V_R_consumed += delta_R
        self.V_P_consumed += delta_P
        self.V_pi_star_consumed += delta_pi
        
        # Track timestep
        if timestep is not None:
            self.current_timestep = timestep
        else:
            self.current_timestep += 1
        
        # Record history
        self.consumption_history.append((
            self.current_timestep,
            delta_R,
            delta_P,
            delta_pi
        ))
    
    def get_remaining_fraction(self) -> Tuple[float, float, float]:
        """
        Get fraction of budget remaining for each type.
        
        Returns:
            Tuple of (V_R_frac, V_P_frac, V_pi_frac) ∈ [0, 1]
        """
        if self.V_R_total > 0:
            V_R_frac = self.V_R_remaining / self.V_R_total
        else:
            V_R_frac = 1.0
            
        if self.V_P_total > 0:
            V_P_frac = self.V_P_remaining / self.V_P_total
        else:
            V_P_frac = 1.0
            
        if self.V_pi_star_total > 0:
            V_pi_frac = self.V_pi_star_remaining / self.V_pi_star_total
        else:
            V_pi_frac = 1.0
        
        return (V_R_frac, V_P_frac, V_pi_frac)
    
    def get_consumed_fraction(self) -> Tuple[float, float, float]:
        """
        Get fraction of budget consumed for each type.
        
        Returns:
            Tuple of (V_R_consumed_frac, V_P_consumed_frac, V_pi_consumed_frac)
        """
        remaining = self.get_remaining_fraction()
        return (1.0 - remaining[0], 1.0 - remaining[1], 1.0 - remaining[2])
    
    def get_min_remaining_fraction(self) -> float:
        """
        Get the minimum remaining fraction across all budgets.
        
        This represents the "tightest" constraint and is used for
        trust region adaptation in Algorithm 1.
        
        Returns:
            min(V_R_frac, V_P_frac, V_pi_frac)
        """
        fracs = self.get_remaining_fraction()
        return min(fracs)
    
    def is_budget_exhausted(self, threshold: float = 0.1) -> bool:
        """
        Check if any budget is nearly exhausted.
        
        Args:
            threshold: Threshold below which budget is considered exhausted
            
        Returns:
            True if any budget remaining < threshold
        """
        return self.get_min_remaining_fraction() < threshold
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of budget utilization.
        
        Returns:
            Dictionary with budget metrics
        """
        remaining = self.get_remaining_fraction()
        consumed = self.get_consumed_fraction()
        
        return {
            # Total budgets
            'V_R_total': self.V_R_total,
            'V_P_total': self.V_P_total,
            'V_pi_star_total': self.V_pi_star_total,
            
            # Remaining
            'V_R_remaining': self.V_R_remaining,
            'V_P_remaining': self.V_P_remaining,
            'V_pi_star_remaining': self.V_pi_star_remaining,
            
            # Consumed
            'V_R_consumed': self.V_R_consumed,
            'V_P_consumed': self.V_P_consumed,
            'V_pi_star_consumed': self.V_pi_star_consumed,
            
            # Fractions
            'V_R_fraction_remaining': remaining[0],
            'V_P_fraction_remaining': remaining[1],
            'V_pi_fraction_remaining': remaining[2],
            
            'V_R_fraction_consumed': consumed[0],
            'V_P_fraction_consumed': consumed[1],
            'V_pi_fraction_consumed': consumed[2],
            
            # Overall
            'min_fraction_remaining': self.get_min_remaining_fraction(),
            'timesteps': self.current_timestep,
        }
    
    def get_consumption_history(self) -> np.ndarray:
        """
        Get consumption history as numpy array.
        
        Returns:
            Array of shape (n_updates, 4) with columns [timestep, delta_R, delta_P, delta_pi]
        """
        if not self.consumption_history:
            return np.array([]).reshape(0, 4)
        
        return np.array(self.consumption_history)
    
    def save(self, filepath: str) -> None:
        """
        Save budget tracker state to JSON file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'config': {
                'V_R_total': self.V_R_total,
                'V_P_total': self.V_P_total,
                'V_pi_star_total': self.V_pi_star_total,
                'time_horizon': self.time_horizon,
            },
            'state': {
                'current_timestep': self.current_timestep,
                'V_R_remaining': self.V_R_remaining,
                'V_P_remaining': self.V_P_remaining,
                'V_pi_star_remaining': self.V_pi_star_remaining,
                'V_R_consumed': self.V_R_consumed,
                'V_P_consumed': self.V_P_consumed,
                'V_pi_star_consumed': self.V_pi_star_consumed,
            },
            'consumption_history': [
                {'timestep': t, 'delta_R': dR, 'delta_P': dP, 'delta_pi': dpi}
                for t, dR, dP, dpi in self.consumption_history
            ],
            'summary': self.get_summary(),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'VariationBudgetTracker':
        """
        Load budget tracker from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            VariationBudgetTracker instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = data['config']
        tracker = cls(
            V_R=config['V_R_total'],
            V_P=config['V_P_total'],
            V_pi_star=config['V_pi_star_total'],
            time_horizon=config.get('time_horizon'),
        )
        
        # Restore state
        state = data['state']
        tracker.current_timestep = state['current_timestep']
        tracker.V_R_remaining = state['V_R_remaining']
        tracker.V_P_remaining = state['V_P_remaining']
        tracker.V_pi_star_remaining = state['V_pi_star_remaining']
        tracker.V_R_consumed = state['V_R_consumed']
        tracker.V_P_consumed = state['V_P_consumed']
        tracker.V_pi_star_consumed = state['V_pi_star_consumed']
        
        # Restore history
        tracker.consumption_history = [
            (h['timestep'], h['delta_R'], h['delta_P'], h['delta_pi'])
            for h in data['consumption_history']
        ]
        
        return tracker
    
    def reset(self) -> None:
        """Reset budgets to initial values."""
        self.V_R_remaining = self.V_R_total
        self.V_P_remaining = self.V_P_total
        self.V_pi_star_remaining = self.V_pi_star_total
        self.V_R_consumed = 0.0
        self.V_P_consumed = 0.0
        self.V_pi_star_consumed = 0.0
        self.consumption_history.clear()
        self.current_timestep = 0
    
    def __repr__(self) -> str:
        """String representation."""
        remaining = self.get_remaining_fraction()
        return (
            f"VariationBudgetTracker(\n"
            f"  V_R: {self.V_R_remaining:.2f}/{self.V_R_total:.2f} ({remaining[0]:.1%})\n"
            f"  V_P: {self.V_P_remaining:.2f}/{self.V_P_total:.2f} ({remaining[1]:.1%})\n"
            f"  V_π*: {self.V_pi_star_remaining:.2f}/{self.V_pi_star_total:.2f} ({remaining[2]:.1%})\n"
            f"  timesteps: {self.current_timestep}\n"
            f")"
        )


def estimate_budgets_from_drift_config(
    drift_config: Dict,
    time_horizon: int,
    scale_factor: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Estimate variation budgets from drift configuration.
    
    Conservative estimation strategy (Option A):
        V_R ≈ drift_magnitude × (time_horizon / period) × scale_factor
        V_P ≈ drift_magnitude × (time_horizon / period) × scale_factor
        V_π* ≈ 0.5 × max(V_R, V_P)  (heuristic: policy adapts slower)
    
    Args:
        drift_config: Drift configuration dict with 'magnitude', 'period', 'drift_type'
        time_horizon: Total training timesteps
        scale_factor: Safety multiplier (>1 = more conservative)
        
    Returns:
        Tuple of (V_R, V_P, V_pi_star)
    """
    magnitude = drift_config.get('magnitude', 1.0)
    period = drift_config.get('period', 10000)
    drift_type = drift_config.get('drift_type', 'static')
    
    # Number of drift cycles
    n_cycles = time_horizon / period
    
    # Base estimation
    if drift_type == 'static':
        # No drift
        V_R = 0.1 * scale_factor
        V_P = 0.1 * scale_factor
        
    elif drift_type == 'jump':
        # Discrete jumps
        V_R = magnitude * n_cycles * scale_factor
        V_P = magnitude * n_cycles * scale_factor
        
    elif drift_type == 'linear':
        # Linear drift accumulates
        V_R = 0.5 * magnitude * n_cycles * scale_factor
        V_P = 0.5 * magnitude * n_cycles * scale_factor
        
    elif drift_type in ['sine', 'random_walk']:
        # Oscillating/stochastic drift
        V_R = magnitude * np.sqrt(n_cycles) * scale_factor
        V_P = magnitude * np.sqrt(n_cycles) * scale_factor
        
    else:
        # Default conservative
        V_R = magnitude * n_cycles * scale_factor
        V_P = magnitude * n_cycles * scale_factor
    
    # Policy variation is typically smaller (adapts slower than environment)
    V_pi_star = 0.5 * max(V_R, V_P)
    
    return (V_R, V_P, V_pi_star)
