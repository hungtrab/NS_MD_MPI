"""
Evaluation Script for Non-Stationary MDP Experiments.

Supports:
- Standard policy evaluation
- Oracle baseline computation
- Dynamic regret calculation
"""

import sys
import os
import yaml
import argparse
import numpy as np
import gymnasium as gym
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.envs.wrappers import NonStationaryCartPoleWrapper
from src.evaluation import (
    OracleEvaluator, 
    OracleConfig,
    DynamicRegretCalculator,
    RegretConfig,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_drift_conf(cfg: dict) -> dict:
    """Build drift configuration from YAML config."""
    return {
        'parameter': cfg['env'].get('parameter', 'gravity'),
        'drift_type': cfg['env'].get('drift_type', 'static'),
        'magnitude': cfg['env'].get('magnitude', 0.0),
        'period': cfg['env'].get('period', 10000),
        'sigma': cfg['env'].get('sigma', 0.1),
        'bounds': cfg['env'].get('bounds', [0.0, 20.0]),
    }


def evaluate_model(
    model_path: str,
    config_path: str,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> tuple:
    """
    Evaluate a trained model on the non-stationary environment.
    
    Args:
        model_path: Path to saved model
        config_path: Path to config YAML
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        
    Returns:
        Tuple of (mean_reward, std_reward)
    """
    cfg = load_config(config_path)
    drift_conf = build_drift_conf(cfg)
    
    # Create environment
    env = gym.make(cfg['env_id'])
    env = NonStationaryCartPoleWrapper(env, drift_conf)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)
    
    # Evaluate
    print(f"Evaluating over {n_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model, env, 
        n_eval_episodes=n_episodes, 
        deterministic=deterministic
    )
    
    env.close()
    return mean_reward, std_reward


def compute_dynamic_regret(
    model_path: str,
    config_path: str,
    eval_interval: int = 5000,
    oracle_train_steps: int = 50000,
    n_eval_episodes: int = 10,
) -> dict:
    """
    Compute dynamic regret for a trained model.
    
    Args:
        model_path: Path to saved model
        config_path: Path to config YAML
        eval_interval: Steps between evaluations
        oracle_train_steps: Steps to train each oracle
        n_eval_episodes: Episodes per evaluation point
        
    Returns:
        Dictionary with regret metrics
    """
    cfg = load_config(config_path)
    drift_conf = build_drift_conf(cfg)
    total_timesteps = cfg['train']['total_timesteps']
    
    print("=" * 50)
    print("Computing Dynamic Regret")
    print("=" * 50)
    
    # 1. Compute Oracle values
    print("\n[1/3] Computing Oracle baseline...")
    oracle_config = OracleConfig(
        env_id=cfg['env_id'],
        train_timesteps=oracle_train_steps,
        n_eval_episodes=n_eval_episodes,
        cache_dir="oracle_cache/",
        verbose=1,
    )
    oracle = OracleEvaluator(oracle_config)
    
    # Create wrapped env to get drift trajectory
    env = gym.make(cfg['env_id'])
    wrapped_env = NonStationaryCartPoleWrapper(env, drift_conf)
    
    eval_points = list(range(0, total_timesteps, eval_interval))
    oracle_values = oracle.evaluate_at_timesteps(wrapped_env, eval_points)
    oracle_dict = {t: mean for t, (mean, _) in oracle_values.items()}
    
    # 2. Evaluate policy at each point
    print("\n[2/3] Evaluating policy at each timestep...")
    model = PPO.load(model_path)
    
    regret_calc = DynamicRegretCalculator(RegretConfig(
        weighting="uniform",
        eval_interval=eval_interval,
        normalize=False,
    ))
    regret_calc.set_oracle_values(oracle_dict)
    
    for t in eval_points:
        # Set environment to this timestep
        wrapped_env.set_total_steps(t)
        
        # Evaluate policy
        mean_reward, _ = evaluate_policy(
            model, wrapped_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        
        regret = regret_calc.add_policy_value(t, mean_reward)
        print(f"  t={t:6d}: π={mean_reward:.1f}, v*={oracle_dict[t]:.1f}, regret={regret:.1f}")
    
    # 3. Summary
    print("\n[3/3] Computing summary...")
    summary = regret_calc.get_summary()
    
    # Save results
    output_path = Path(model_path).parent / "regret_results.json"
    regret_calc.save(str(output_path))
    print(f"\nResults saved to: {output_path}")
    
    wrapped_env.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--config", type=str, default="configs/cartpole_adaptive.yaml", help="Path to config")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--regret", action="store_true", help="Compute dynamic regret")
    parser.add_argument("--oracle-steps", type=int, default=50000, help="Oracle training steps")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Evaluation interval for regret")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    if args.regret:
        # Compute dynamic regret
        summary = compute_dynamic_regret(
            model_path=args.model,
            config_path=args.config,
            eval_interval=args.eval_interval,
            oracle_train_steps=args.oracle_steps,
            n_eval_episodes=args.episodes,
        )
        
        print("\n" + "=" * 50)
        print("DYNAMIC REGRET SUMMARY")
        print("=" * 50)
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")
    else:
        # Standard evaluation
        mean_reward, std_reward = evaluate_model(
            model_path=args.model,
            config_path=args.config,
            n_episodes=args.episodes,
        )
        
        cfg = load_config(args.config)
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Drift Type: {cfg['env']['drift_type']}")
        print(f"  Parameter: {cfg['env']['parameter']}")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
    main()