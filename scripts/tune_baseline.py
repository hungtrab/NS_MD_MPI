#!/usr/bin/env python3
"""
Baseline PPO Tuning Script
Creates and runs sweeps for baseline PPO across all drift scenarios.
Usage: python scripts/tune_baseline.py --env Hopper-v4 --drift-param friction --drift-type sine --count 20
"""

import argparse
import wandb
import gymnasium as gym
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from src.envs.nonstationary_wrapper import make_nonstationary_env


def train_baseline():
    """Train baseline PPO with wandb sweep config."""
    # Initialize wandb
    run = wandb.init()
    config = wandb.config
    
    # Create environment
    def make_env(rank):
        def _init():
            env = make_nonstationary_env(
                env_id=config.env_id,
                parameter=config.drift_parameter,
                drift_type=config.drift_type,
                magnitude=config.get('drift_magnitude', 0.3),
                period=config.get('drift_period', 50000),
                base_value=config.get('drift_base_value', 1.0),
                bounds=config.get('bounds', None),
                sigma=config.get('sigma', 0.01),
                seed=config.seed + rank
            )
            return env
        return _init
    
    n_envs = config.get('n_envs', 4)
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    # Create eval env
    eval_env = DummyVecEnv([make_env(100)])
    
    # Create model with sweep params
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        clip_range=config.get('clip_range', 0.2),
        ent_coef=config.get('ent_coef', 0.0),
        vf_coef=config.get('vf_coef', 0.5),
        verbose=0,
        device='cpu'
    )
    
    # Train
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            progress_bar=True
        )
        
        # Evaluate
        rewards = []
        for _ in range(10):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        wandb.log({"mean_reward": mean_reward})
        
    except Exception as e:
        wandb.log({"mean_reward": -10000, "error": str(e)})
    
    finally:
        env.close()
        eval_env.close()
        wandb.finish()


def create_sweep_config(env_id, drift_param, drift_type, bounds=None, base_value=1.0, sigma=0.01):
    """Create sweep config for a specific drift scenario."""
    
    sweep_config = {
        "method": "bayes",
        "name": f"Baseline_{env_id}_{drift_param}_{drift_type}",
        "metric": {"name": "mean_reward", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 3, "s": 2},
        "parameters": {
            # Tunable PPO params
            "learning_rate": {"distribution": "log_uniform_values", "min": 5e-5, "max": 1e-3},
            "n_steps": {"values": [1024, 2048, 4096]},
            "batch_size": {"values": [32, 64, 128, 256]},
            "gamma": {"values": [0.99, 0.995, 0.999]},
            "clip_range": {"distribution": "uniform", "min": 0.1, "max": 0.3},
            "ent_coef": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
            "vf_coef": {"distribution": "uniform", "min": 0.3, "max": 0.7},
            # Fixed params
            "env_id": {"value": env_id},
            "drift_parameter": {"value": drift_param},
            "drift_type": {"value": drift_type},
            "drift_magnitude": {"value": 0.3},
            "drift_period": {"value": 50000},
            "drift_base_value": {"value": base_value},
            "sigma": {"value": sigma},
            "total_timesteps": {"value": 200000},
            "n_envs": {"value": 4},
            "seed": {"value": 42},
        }
    }
    
    if bounds:
        sweep_config["parameters"]["bounds"] = {"value": bounds}
    
    return sweep_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment ID")
    parser.add_argument("--drift-param", type=str, required=True, help="Drift parameter")
    parser.add_argument("--drift-type", type=str, required=True, help="Drift type")
    parser.add_argument("--count", type=int, default=20, help="Number of trials")
    parser.add_argument("--project", type=str, default="att_19_Baseline_Tuning")
    args = parser.parse_args()
    
    # Create sweep
    sweep_config = create_sweep_config(
        env_id=args.env,
        drift_param=args.drift_param,
        drift_type=args.drift_type
    )
    sweep_config["program"] = "scripts/tune_baseline.py"
    
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"Created sweep: {sweep_id}")
    
    # Run agent
    wandb.agent(sweep_id, function=train_baseline, count=args.count, project=args.project)


if __name__ == "__main__":
    # Check if running as sweep agent
    if wandb.run is not None:
        train_baseline()
    else:
        main()
