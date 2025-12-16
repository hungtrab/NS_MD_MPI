"""
Oracle Evaluator for Dynamic Regret Computation.

The Oracle provides ground-truth optimal performance v_t^* at each timestep
by training a fresh agent on the exact environment parameters at time t.

This is used to compute Dynamic Regret:
    DynReg(T) = Σ_{t=1}^{T} ρ_t [v_t^* - v_t^{π_t}]

Reference: Section 4 - Evaluation Protocol
"""

import os
import pickle
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.envs.wrappers import NonStationaryCartPoleWrapper


@dataclass
class OracleConfig:
    """Configuration for Oracle training."""
    env_id: str = "CartPole-v1"
    algorithm: str = "PPO"
    train_timesteps: int = 50000          # Steps to train oracle policy
    n_eval_episodes: int = 20              # Episodes for evaluation
    cache_dir: str = "oracle_cache/"       # Directory to cache trained oracles
    use_cache: bool = True                 # Whether to use cached oracles
    verbose: int = 0


class OracleEvaluator:
    """
    Oracle Evaluator for computing ground-truth optimal values.
    
    For each evaluation point t, the Oracle:
    1. Snapshots the current environment parameters
    2. Trains a fresh policy on a stationary environment with those parameters
    3. Evaluates the trained policy to get v_t^*
    
    Caching is used to avoid redundant training for the same parameter settings.
    """
    
    def __init__(self, config: OracleConfig):
        """
        Initialize the Oracle Evaluator.
        
        Args:
            config: OracleConfig with training and evaluation settings
        """
        self.config = config
        self.cache: Dict[str, Tuple[float, float]] = {}  # param_hash -> (mean, std)
        
        # Create cache directory
        self.cache_path = Path(config.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached oracle results from disk."""
        cache_file = self.cache_path / "oracle_cache.pkl"
        if cache_file.exists() and self.config.use_cache:
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f">>> [Oracle] Loaded {len(self.cache)} cached evaluations")
            except Exception as e:
                print(f">>> [Oracle] Could not load cache: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        cache_file = self.cache_path / "oracle_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f">>> [Oracle] Could not save cache: {e}")
    
    def _get_param_hash(self, params: Dict[str, float]) -> str:
        """
        Create a hash key from environment parameters.
        
        Args:
            params: Dictionary of parameter names to values
            
        Returns:
            String hash key
        """
        # Round to 3 decimal places to handle floating point issues
        rounded = {k: round(v, 3) for k, v in sorted(params.items())}
        return str(rounded)
    
    def get_optimal_value(
        self, 
        params: Dict[str, float],
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Get the optimal value v_t^* for a given parameter configuration.
        
        This trains a fresh agent on a stationary environment with the
        specified parameters and returns its performance.
        
        Args:
            params: Dictionary mapping parameter names to values
                    e.g., {'gravity': 15.0, 'masscart': 1.0}
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        param_hash = self._get_param_hash(params)
        
        # Check cache
        if param_hash in self.cache and self.config.use_cache:
            if self.config.verbose > 0:
                print(f">>> [Oracle] Cache hit for params: {params}")
            return self.cache[param_hash]
        
        if self.config.verbose > 0:
            print(f">>> [Oracle] Training oracle for params: {params}")
        
        # Create stationary environment with specified parameters
        env = self._create_stationary_env(params)
        
        # Train fresh policy
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
        )
        model.learn(total_timesteps=self.config.train_timesteps)
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
        )
        
        # Cache result
        self.cache[param_hash] = (mean_reward, std_reward)
        self._save_cache()
        
        # Cleanup
        env.close()
        del model
        
        return mean_reward, std_reward
    
    def _create_stationary_env(self, params: Dict[str, float]) -> gym.Env:
        """
        Create a stationary environment with fixed parameters.
        
        Args:
            params: Dictionary of parameter values to set
            
        Returns:
            Gymnasium environment with fixed parameters
        """
        env = gym.make(self.config.env_id)
        
        # Set parameters directly on unwrapped environment
        for param_name, value in params.items():
            if hasattr(env.unwrapped, param_name):
                setattr(env.unwrapped, param_name, value)
                
                # Recalculate dependent quantities for CartPole
                if param_name in ['masscart', 'masspole']:
                    env.unwrapped.total_mass = env.unwrapped.masspole + env.unwrapped.masscart
                if param_name in ['length', 'masspole']:
                    env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
        
        return env
    
    def evaluate_at_timesteps(
        self,
        wrapped_env: NonStationaryCartPoleWrapper,
        timesteps: List[int],
        seed: Optional[int] = None,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Evaluate oracle at multiple timesteps.
        
        Args:
            wrapped_env: The non-stationary wrapped environment
            timesteps: List of timesteps to evaluate at
            seed: Random seed
            
        Returns:
            Dictionary mapping timestep -> (mean_reward, std_reward)
        """
        results = {}
        
        for t in timesteps:
            # Get parameters at timestep t
            wrapped_env.set_total_steps(t)
            params = {}
            
            for param, gen in wrapped_env.drift_generators.items():
                params[param] = gen.get_value(t)
            
            # Get oracle value
            mean_reward, std_reward = self.get_optimal_value(params, seed=seed)
            results[t] = (mean_reward, std_reward)
            
            if self.config.verbose > 0:
                print(f">>> [Oracle] t={t}: v*={mean_reward:.2f} ± {std_reward:.2f}")
        
        return results
    
    def precompute_oracle_trajectory(
        self,
        drift_conf: Dict[str, Any],
        total_timesteps: int,
        eval_interval: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Precompute oracle values along a drift trajectory.
        
        This is useful for setting up evaluation before training.
        
        Args:
            drift_conf: Drift configuration (same format as wrapper)
            total_timesteps: Total timesteps in the trajectory
            eval_interval: Interval between oracle evaluations
            seed: Random seed
            
        Returns:
            Dictionary mapping timestep -> (mean_reward, std_reward)
        """
        # Create temporary wrapped env to get drift values
        env = gym.make(self.config.env_id)
        wrapped_env = NonStationaryCartPoleWrapper(env, drift_conf, seed=seed)
        
        timesteps = list(range(0, total_timesteps, eval_interval))
        results = self.evaluate_at_timesteps(wrapped_env, timesteps, seed=seed)
        
        wrapped_env.close()
        return results


def compute_oracle_baseline(
    env_id: str,
    drift_conf: Dict[str, Any],
    total_timesteps: int,
    eval_interval: int = 5000,
    oracle_train_steps: int = 50000,
    cache_dir: str = "oracle_cache/",
    verbose: int = 1,
) -> Dict[int, float]:
    """
    Convenience function to compute oracle baseline for an experiment.
    
    Args:
        env_id: Gymnasium environment ID
        drift_conf: Drift configuration dictionary
        total_timesteps: Total timesteps to evaluate over
        eval_interval: Steps between oracle evaluations
        oracle_train_steps: Steps to train each oracle policy
        cache_dir: Directory for caching
        verbose: Verbosity level
        
    Returns:
        Dictionary mapping timestep -> optimal_value (v_t^*)
    """
    config = OracleConfig(
        env_id=env_id,
        train_timesteps=oracle_train_steps,
        cache_dir=cache_dir,
        verbose=verbose,
    )
    
    oracle = OracleEvaluator(config)
    
    results = oracle.precompute_oracle_trajectory(
        drift_conf=drift_conf,
        total_timesteps=total_timesteps,
        eval_interval=eval_interval,
    )
    
    # Return just the mean values
    return {t: mean for t, (mean, _) in results.items()}
