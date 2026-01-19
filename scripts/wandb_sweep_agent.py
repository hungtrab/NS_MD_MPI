#!/usr/bin/env python3
"""
NS-MDMPI Hyperparameter Tuning with Weights & Biases Sweep

Usage:
  1. Create sweep: wandb sweep configs/sweep/sweep_moderate.yaml
  2. Run agent: python scripts/wandb_sweep_agent.py --sweep-id <SWEEP_ID>
  
Or use the all-in-one script:
  python scripts/wandb_sweep_agent.py --create-sweep --config configs/sweep/sweep_moderate.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

import wandb
import gymnasium as gym

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from src.callbacks.nsmdmpi_callback import NSMDMPICallback
from src.envs.multi_env_wrappers import make_nonstationary_env


def create_env(env_id: str, drift_config: Optional[Dict], n_envs: int = 4, seed: int = 42):
    """Create vectorized environment for training"""
    def make_env(rank):
        def _init():
            if drift_config is None:
                env = gym.make(env_id)
            else:
                env = make_nonstationary_env(env_id, drift_config, seed=seed + rank)
            env.reset(seed=seed + rank)
            return env
        return _init
    
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    return env


def train_with_hyperparameters():
    """
    Training function to be called by W&B sweep agent.
    Hyperparameters are pulled from wandb.config.
    """
    # Initialize W&B run
    run = wandb.init()
    config = wandb.config
    
    print(f"\n{'='*60}")
    print(f"Starting sweep trial: {run.name}")
    print(f"{'='*60}")
    
    # Extract hyperparameters from sweep config
    env_id = config.get('env_id', 'Hopper-v4')
    
    # Drift config
    drift_config = {
        'parameter': config.get('drift_parameter', 'friction'),
        'drift_type': config.get('drift_type', 'sine'),
        'magnitude': config.get('drift_magnitude', 0.27),
        'period': config.get('drift_period', 10000),
        'base_value': config.get('drift_base_value', 0.9),
    }
    
    # Training settings
    total_timesteps = config.get('total_timesteps', 200_000)
    n_envs = config.get('n_envs', 4)
    seed = config.get('seed', 42)
    
    set_random_seed(seed)
    
    # NS-MDMPI hyperparameters (these are what we're tuning!)
    V_R = config.get('V_R', 10.0)
    V_P = config.get('V_P', 10.0)
    V_pi_star = config.get('V_pi_star', 5.0)
    kappa_base = config.get('kappa_base', 0.2)
    lambda_base = config.get('lambda_base', 1.0)
    trust_region_sensitivity = config.get('trust_region_sensitivity', 5.0)
    regularization_sensitivity = config.get('regularization_sensitivity', 2.0)
    max_ent_coef = config.get('max_ent_coef', 0.05)
    drift_window_size = config.get('drift_window_size', 1000)
    
    # PPO hyperparameters (optional tuning)
    learning_rate = config.get('learning_rate', 3e-4)
    n_steps = config.get('n_steps', 2048)
    batch_size = config.get('batch_size', 64)
    gamma = config.get('gamma', 0.99)
    
    print(f"\nHyperparameters:")
    print(f"  V_R={V_R:.2f}, V_P={V_P:.2f}, V_pi_star={V_pi_star:.2f}")
    print(f"  kappa_base={kappa_base:.3f}, lambda_base={lambda_base:.2f}")
    print(f"  trust_region_sensitivity={trust_region_sensitivity:.2f}")
    print(f"  regularization_sensitivity={regularization_sensitivity:.2f}")
    print(f"  max_ent_coef={max_ent_coef:.3f}")
    print(f"  drift_window_size={drift_window_size}")
    
    # Create environments
    env = create_env(env_id, drift_config, n_envs=n_envs, seed=seed)
    eval_env = create_env(env_id, drift_config, n_envs=1, seed=seed + 1000)
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        verbose=0,
        device='auto',
    )
    
    # NS-MDMPI callback
    nsmdmpi_callback = NSMDMPICallback(
        V_R=V_R,
        V_P=V_P,
        V_pi_star=V_pi_star,
        kappa_base=kappa_base,
        kappa_adaptive=True,
        trust_region_sensitivity=trust_region_sensitivity,
        lambda_base=lambda_base,
        lambda_adaptive=True,
        regularization_sensitivity=regularization_sensitivity,
        max_ent_coef=max_ent_coef,
        drift_window_size=drift_window_size,
        log_freq=500,
        save_budget_history=False,
        verbose=0,
    )
    
    # Evaluation callback
    eval_freq = total_timesteps // 10  # Evaluate 10 times
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=5,
        deterministic=False,
        verbose=0,
    )
    
    callbacks = [nsmdmpi_callback, eval_callback]
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except Exception as e:
        print(f"Training failed: {e}")
        wandb.log({'error': str(e), 'mean_reward': -10000})
        wandb.finish()
        return
    
    # Compute metrics
    mean_reward = eval_callback.last_mean_reward if hasattr(eval_callback, 'last_mean_reward') else 0
    
    budget_summary = nsmdmpi_callback.budget_tracker.get_summary()
    budget_remaining_avg = np.mean([
        budget_summary.get('V_R_fraction_remaining', 0),
        budget_summary.get('V_P_fraction_remaining', 0),
        budget_summary.get('V_pi_fraction_remaining', 0)
    ])
    
    # Log final metrics
    wandb.log({
        'final/mean_reward': mean_reward,
        'final/budget_remaining_avg': budget_remaining_avg,
        'final/V_R_remaining': budget_summary.get('V_R_fraction_remaining', 0),
        'final/V_P_remaining': budget_summary.get('V_P_fraction_remaining', 0),
        'final/V_pi_remaining': budget_summary.get('V_pi_fraction_remaining', 0),
    })
    
    # Compute optimization score (same as Optuna objective)
    score = (
        0.5 * mean_reward / 1000.0 +
        0.3 * budget_remaining_avg +
        0.2 * (1.0 - abs(budget_remaining_avg - 0.7))
    )
    
    wandb.log({'optimization_score': score})
    
    print(f"\n{'='*60}")
    print(f"Trial Complete!")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Budget Remaining: {budget_remaining_avg:.2%}")
    print(f"Score: {score:.4f}")
    print(f"{'='*60}")
    
    # Cleanup
    env.close()
    eval_env.close()
    wandb.finish()


def create_sweep_config(experiment_type: str, env_id: str, drift_config: Dict) -> Dict:
    """Generate sweep configuration based on experiment type and environment"""
    
    # Environment-specific base ranges
    if 'HalfCheetah' in env_id:
        base_ranges = {
            'V_R': {'distribution': 'uniform', 'min': 50.0, 'max': 200.0},
            'V_P': {'distribution': 'uniform', 'min': 100.0, 'max': 400.0},
            'V_pi_star': {'distribution': 'uniform', 'min': 500.0, 'max': 2000.0},
            'kappa_base': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
            'lambda_base': {'distribution': 'uniform', 'min': 0.5, 'max': 3.0},
            'trust_region_sensitivity': {'distribution': 'uniform', 'min': 5.0, 'max': 25.0},
            'regularization_sensitivity': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0},
            'max_ent_coef': {'distribution': 'uniform', 'min': 0.01, 'max': 0.15},
            'drift_window_size': {'distribution': 'int_uniform', 'min': 500, 'max': 2000},
        }
    elif 'Hopper' in env_id:
        base_ranges = {
            'V_R': {'distribution': 'uniform', 'min': 30.0, 'max': 150.0},
            'V_P': {'distribution': 'uniform', 'min': 50.0, 'max': 250.0},
            'V_pi_star': {'distribution': 'uniform', 'min': 300.0, 'max': 1500.0},
            'kappa_base': {'distribution': 'uniform', 'min': 0.1, 'max': 0.35},
            'lambda_base': {'distribution': 'uniform', 'min': 0.5, 'max': 2.5},
            'trust_region_sensitivity': {'distribution': 'uniform', 'min': 3.0, 'max': 20.0},
            'regularization_sensitivity': {'distribution': 'uniform', 'min': 1.5, 'max': 8.0},
            'max_ent_coef': {'distribution': 'uniform', 'min': 0.01, 'max': 0.12},
            'drift_window_size': {'distribution': 'int_uniform', 'min': 400, 'max': 1500},
        }
    elif 'LunarLander' in env_id:
        base_ranges = {
            'V_R': {'distribution': 'uniform', 'min': 5.0, 'max': 30.0},
            'V_P': {'distribution': 'uniform', 'min': 10.0, 'max': 50.0},
            'V_pi_star': {'distribution': 'uniform', 'min': 20.0, 'max': 150.0},
            'kappa_base': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
            'lambda_base': {'distribution': 'uniform', 'min': 0.5, 'max': 2.0},
            'trust_region_sensitivity': {'distribution': 'uniform', 'min': 2.0, 'max': 12.0},
            'regularization_sensitivity': {'distribution': 'uniform', 'min': 1.0, 'max': 6.0},
            'max_ent_coef': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1},
            'drift_window_size': {'distribution': 'int_uniform', 'min': 300, 'max': 1000},
        }
    else:
        # Default for unknown envs
        base_ranges = {
            'V_R': {'distribution': 'uniform', 'min': 20.0, 'max': 100.0},
            'V_P': {'distribution': 'uniform', 'min': 30.0, 'max': 150.0},
            'V_pi_star': {'distribution': 'uniform', 'min': 100.0, 'max': 500.0},
            'kappa_base': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
            'lambda_base': {'distribution': 'uniform', 'min': 0.5, 'max': 2.0},
            'trust_region_sensitivity': {'distribution': 'uniform', 'min': 2.0, 'max': 15.0},
            'regularization_sensitivity': {'distribution': 'uniform', 'min': 1.0, 'max': 7.0},
            'max_ent_coef': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1},
            'drift_window_size': {'distribution': 'int_uniform', 'min': 500, 'max': 1500},
        }
    
    # Apply experiment type modifiers
    if experiment_type == 'extreme':
        # Increase ranges by 50% for extreme drifts
        for key in ['V_R', 'V_P', 'V_pi_star']:
            base_ranges[key]['min'] *= 1.5
            base_ranges[key]['max'] *= 1.5
        base_ranges['trust_region_sensitivity']['max'] *= 1.3
        base_ranges['regularization_sensitivity']['max'] *= 1.3
    elif experiment_type == 'multi':
        # Slightly increase for multi-param drifts
        for key in ['V_R', 'V_P', 'V_pi_star']:
            base_ranges[key]['min'] *= 1.2
            base_ranges[key]['max'] *= 1.2
    
    param_ranges = base_ranges
    
    # Fixed parameters (not tuned)
    fixed_params = {
        'env_id': {'value': env_id},
        'drift_parameter': {'value': drift_config.get('parameter', 'friction')},
        'drift_type': {'value': drift_config.get('drift_type', 'sine')},
        'drift_magnitude': {'value': drift_config.get('magnitude', 0.27)},
        'drift_period': {'value': drift_config.get('period', 10000)},
        'drift_base_value': {'value': drift_config.get('base_value', 0.9)},
        'total_timesteps': {'value': 200_000},
        'n_envs': {'value': 4},
        'seed': {'value': 42},
        'learning_rate': {'value': 3e-4},
        'n_steps': {'value': 2048},
        'batch_size': {'value': 64},
        'gamma': {'value': 0.99},
    }
    
    sweep_config = {
        'program': 'scripts/wandb_sweep_agent.py',
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'optimization_score',
            'goal': 'maximize'
        },
        'parameters': {**param_ranges, **fixed_params},
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            's': 2
        }
    }
    
    return sweep_config


def main():
    parser = argparse.ArgumentParser(description='W&B Sweep Agent for NS-MDMPI')
    parser.add_argument('--sweep-id', type=str, default=None, 
                        help='W&B sweep ID (format: entity/project/sweep_id)')
    parser.add_argument('--count', type=int, default=50, 
                        help='Number of runs for this agent')
    parser.add_argument('--create-sweep', action='store_true',
                        help='Create a new sweep from config')
    parser.add_argument('--config', type=str, default=None,
                        help='Sweep config YAML path (for --create-sweep)')
    parser.add_argument('--env', type=str, default='Hopper-v4',
                        help='Environment ID (for auto-generated sweep)')
    parser.add_argument('--type', type=str, default='moderate',
                        choices=['moderate', 'extreme', 'multi'],
                        help='Experiment type')
    parser.add_argument('--drift-param', type=str, default='friction',
                        help='Drift parameter: friction, mass_scale, damping, gravity')
    parser.add_argument('--drift-type', type=str, default='sine',
                        help='Drift type: sine, linear, random_walk, jump')
    parser.add_argument('--project', type=str, default='NS-MDMPI-Sweep',
                        help='W&B project name')
    args = parser.parse_args()
    
    if args.create_sweep:
        # Load or generate sweep config
        if args.config:
            with open(args.config, 'r') as f:
                sweep_config = yaml.safe_load(f)
        else:
            # Auto-generate config with user-specified drift params
            drift_config = {
                'parameter': args.drift_param,
                'drift_type': args.drift_type,
                'magnitude': 0.3,
                'period': 50000,
                'base_value': 1.0,
            }
            sweep_config = create_sweep_config(args.type, args.env, drift_config)
            sweep_config['name'] = f"NS-MDMPI_{args.env}_{args.drift_param}_{args.drift_type}"
        
        # Create sweep
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"\n{'='*60}")
        print(f"Sweep created successfully!")
        print(f"Sweep ID: {sweep_id}")
        print(f"\nTo run agents, use:")
        print(f"  python scripts/wandb_sweep_agent.py --sweep-id {args.project}/{sweep_id}")
        print(f"{'='*60}")
        
        # Also run agent automatically
        print(f"\nStarting sweep agent with {args.count} runs...")
        wandb.agent(sweep_id, function=train_with_hyperparameters, 
                    count=args.count, project=args.project)
    
    elif args.sweep_id:
        # Run sweep agent
        print(f"\n{'='*60}")
        print(f"Starting W&B Sweep Agent")
        print(f"Sweep ID: {args.sweep_id}")
        print(f"Number of runs: {args.count}")
        print(f"{'='*60}")
        
        wandb.agent(args.sweep_id, function=train_with_hyperparameters, count=args.count)
    
    else:
        # Run single training (for sweep agent internal use)
        train_with_hyperparameters()


if __name__ == "__main__":
    main()
