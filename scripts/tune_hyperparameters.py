#!/usr/bin/env python3
"""
NS-MDMPI Hyperparameter Tuning with Optuna
Optimized for fast experimentation with early stopping
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gymnasium as gym

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from src.callbacks.nsmdmpi_callback import NSMDMPICallback
from src.envs.multi_env_wrappers import make_nonstationary_env



def create_fast_env(env_id: str, drift_config: Optional[Dict], n_envs: int = 4, seed: int = 42):
    """Create vectorized environment for faster training"""
    def make_env(rank):
        def _init():
            if drift_config is None:
                env = gym.make(env_id)
            else:
                env = make_nonstationary_env(env_id, drift_config, seed=seed + rank)
            env.reset(seed=seed + rank)
            return env
        return _init
    
    # Use SubprocVecEnv for parallel execution
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    
    return env


def objective_function(
    trial: optuna.Trial,
    env_id: str,
    drift_config: Dict,
    experiment_type: str,
    quick_mode: bool = True
) -> float:
    """
    Optuna objective function
    
    Args:
        trial: Optuna trial object
        env_id: Environment ID
        drift_config: Drift configuration
        experiment_type: 'moderate', 'extreme', or 'multi'
        quick_mode: If True, use fast training settings
    
    Returns:
        Optimization score (higher is better)
    """
    
    # ============================================
    # HYPERPARAMETER SEARCH SPACE
    # ============================================
    
    if experiment_type == 'moderate':
        # Moderate drift: balanced adaptation
        V_R = trial.suggest_float('V_R', 5.0, 20.0)
        V_P = trial.suggest_float('V_P', 5.0, 20.0)
        V_pi_star = trial.suggest_float('V_pi_star', 2.5, 10.0)
        
        kappa_base = trial.suggest_float('kappa_base', 0.1, 0.3)
        trust_region_sensitivity = trial.suggest_float('alpha', 2.0, 10.0)
        
        lambda_base = trial.suggest_float('lambda_base', 0.5, 2.0)
        regularization_sensitivity = trial.suggest_float('beta', 1.0, 5.0)
        
        max_ent_coef = trial.suggest_float('max_ent_coef', 0.01, 0.1)
        drift_window_size = trial.suggest_int('drift_window', 500, 2000, step=100)
        
    elif experiment_type == 'extreme':
        # Extreme drift: fast adaptation, high robustness
        V_R = trial.suggest_float('V_R', 20.0, 50.0)
        V_P = trial.suggest_float('V_P', 20.0, 50.0)
        V_pi_star = trial.suggest_float('V_pi_star', 10.0, 25.0)
        
        kappa_base = trial.suggest_float('kappa_base', 0.15, 0.4)
        trust_region_sensitivity = trial.suggest_float('alpha', 5.0, 20.0)
        
        lambda_base = trial.suggest_float('lambda_base', 1.0, 5.0)
        regularization_sensitivity = trial.suggest_float('beta', 3.0, 10.0)
        
        max_ent_coef = trial.suggest_float('max_ent_coef', 0.05, 0.2)
        drift_window_size = trial.suggest_int('drift_window', 200, 1000, step=100)
        
    else:  # multi
        # Multi-parameter: handle complex correlations
        V_R = trial.suggest_float('V_R', 10.0, 30.0)
        V_P = trial.suggest_float('V_P', 10.0, 30.0)
        V_pi_star = trial.suggest_float('V_pi_star', 5.0, 15.0)
        
        kappa_base = trial.suggest_float('kappa_base', 0.1, 0.3)
        trust_region_sensitivity = trial.suggest_float('alpha', 3.0, 12.0)
        
        lambda_base = trial.suggest_float('lambda_base', 0.5, 3.0)
        regularization_sensitivity = trial.suggest_float('beta', 1.5, 7.0)
        
        max_ent_coef = trial.suggest_float('max_ent_coef', 0.02, 0.15)
        drift_window_size = trial.suggest_int('drift_window', 500, 1500, step=100)
    
    # ============================================
    # TRAINING SETTINGS
    # ============================================
    
    if quick_mode:
        total_timesteps = 100_000  # Fast evaluation
        n_envs = 4  # Parallel envs
        eval_freq = 10_000
        n_eval_episodes = 5
    else:
        total_timesteps = 500_000  # More thorough
        n_envs = 1
        eval_freq = 20_000
        n_eval_episodes = 10
    
    # ============================================
    # CREATE ENVIRONMENT
    # ============================================
    
    seed = trial.number  # Different seed per trial
    env = create_fast_env(env_id, drift_config, n_envs=n_envs, seed=seed)
    eval_env = create_fast_env(env_id, drift_config, n_envs=1, seed=seed + 1000)
    
    # ============================================
    # CREATE MODEL
    # ============================================
    
    # Smaller network for faster training
    policy_kwargs = dict(
        net_arch=[64, 64] if quick_mode else [256, 256]
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=0,
        policy_kwargs=policy_kwargs,
        device='cpu',  # Often faster for small networks
    )
    
    # ============================================
    # SETUP CALLBACKS
    # ============================================
    
    # NS-MDMPI callback with trial hyperparameters
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
        log_freq=1000,
        save_budget_history=False,  # Disable to save time
        verbose=0,
    )
    
    # Evaluation callback with pruning
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=False,  # Faster
        verbose=0,
        callback_after_eval=lambda: trial.report(eval_callback.last_mean_reward, model.num_timesteps),
    )
    
    callbacks = [nsmdmpi_callback, eval_callback]
    
    # ============================================
    # TRAIN
    # ============================================
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
    except optuna.exceptions.TrialPruned:
        # Trial was pruned
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -np.inf
    
    # ============================================
    # COMPUTE OPTIMIZATION SCORE
    # ============================================
    
    # Get final evaluation reward
    mean_reward = eval_callback.last_mean_reward
    
    # Get budget efficiency
    budget_summary = nsmdmpi_callback.budget_tracker.get_summary()
    budget_remaining_avg = np.mean([
        budget_summary['V_R_fraction_remaining'],
        budget_summary['V_P_fraction_remaining'],
        budget_summary['V_pi_fraction_remaining']
    ])
    
    # Compute score based on experiment type
    if experiment_type == 'moderate':
        score = (
            0.5 * mean_reward / 1000.0 +  # Normalize reward
            0.3 * budget_remaining_avg +    # Prefer not exhausting budgets
            0.2 * (1.0 - abs(budget_remaining_avg - 0.7))  # Target ~70% remaining
        )
    elif experiment_type == 'extreme':
        score = (
            0.4 * mean_reward / 1000.0 +
            0.3 * budget_remaining_avg +    # High drift needs budgets
            0.3 * (mean_reward > 0)         # Binary: did it survive?
        )
    else:  # multi
        score = (
            0.45 * mean_reward / 1000.0 +
            0.35 * budget_remaining_avg +
            0.2 * (1.0 - abs(budget_remaining_avg - 0.65))
        )
    
    env.close()
    eval_env.close()
    
    return score


def main():
    parser = argparse.ArgumentParser(description='NS-MDMPI Hyperparameter Tuning')
    parser.add_argument('--env', type=str, required=True, help='Environment ID')
    parser.add_argument('--config', type=str, required=True, help='Path to drift config YAML')
    parser.add_argument('--type', type=str, required=True, choices=['moderate', 'extreme', 'multi'])
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--n-jobs', type=int, default=4, help='Parallel jobs')
    parser.add_argument('--study-name', type=str, default=None, help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage (SQLite DB path)')
    parser.add_argument('--quick', action='store_true', help='Quick mode (100k timesteps)')
    args = parser.parse_args()
    
    # Load drift config
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
        drift_config = full_config.get('env')
    
    # Setup study name and storage
    if args.study_name is None:
        args.study_name = f"nsmdmpi_{args.type}_{args.env.replace('-', '_')}"
    
    if args.storage is None:
        os.makedirs('results/optuna_studies', exist_ok=True)
        args.storage = f"sqlite:///results/optuna_studies/{args.study_name}.db"
    
    print(f"=" * 60)
    print(f"NS-MDMPI Hyperparameter Tuning")
    print(f"=" * 60)
    print(f"Environment: {args.env}")
    print(f"Experiment Type: {args.type}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Parallel Jobs: {args.n_jobs}")
    print(f"Study Name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Quick Mode: {args.quick}")
    print(f"=" * 60)
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=10,
            multivariate=True,
            seed=42
        ),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,  # Prune early
            interval_steps=2
        ),
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective_function(
            trial,
            args.env,
            drift_config,
            args.type,
            quick_mode=args.quick
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\n" + "=" * 60)
    print(f"OPTIMIZATION COMPLETE!")
    print(f"=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best params to YAML
    output_dir = Path('results/tuned_params')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.study_name}_best_params.yaml"
    
    with open(output_file, 'w') as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    
    print(f"\nBest parameters saved to: {output_file}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
