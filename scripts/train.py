import sys
import os
import yaml
import datetime
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.envs.wrappers import NonStationaryCartPoleWrapper
from src.callbacks.drift_callback import DriftAdaptiveCallback

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def make_env(config, log_dir=None):
    env = gym.make(config['env_id'])
    
    env = NonStationaryCartPoleWrapper(
        env, 
        drift_conf=config['env'] 
    )
    
    if log_dir:
        env = Monitor(env, log_dir, allow_early_resets=True)
    return env

def main():
    parser = argparse.ArgumentParser(description="Train Script for Non-Stationary RL")
    parser.add_argument("--config", type=str, default="configs/cartpole_adaptive.yaml", help="Path to config")
    parser.add_argument("--exp_name", type=str, default=None, help="Override run name for easier filtering")
    args = parser.parse_args()

    # 1. Load Config
    cfg = load_config(args.config)
    
    if args.exp_name:
        run_name = args.exp_name
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        drift_type = cfg['env']['drift_type']
        mode = "Adaptive" if cfg['adaptive']['enabled'] else "Baseline"
        run_name = f"{cfg['env_id']}_{drift_type}_{mode}_{timestamp}"

    log_path = os.path.join(cfg['paths']['log_dir'], run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(cfg['paths']['model_dir'], exist_ok=True)

    print(f"--- Training Start: {run_name} ---")
    print(f"--- Config: {cfg['env']} ---")

    # 2. Setup WandB
    wandb.init(
        project=cfg.get('wandb', {}).get('project', "CartPole_Drift"),
        tags=cfg.get('wandb', {}).get('tags', []),
        mode=cfg.get('wandb', {}).get('mode', "online"),
        name=run_name,
        config=cfg,
        sync_tensorboard=True, 
        monitor_gym=True,
        save_code=True,
        dir=log_path
    )

    # 3. Setup Env
    env = make_env(cfg, log_path)

    # 4. Setup Model
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=cfg['train']['learning_rate'],
        n_steps=cfg['train']['n_steps'],
        batch_size=cfg['train']['batch_size'],
        gamma=cfg['train']['gamma'],
        verbose=1,
        tensorboard_log=cfg['paths']['log_dir'] 
    )

    # 5. Setup Callbacks
    callbacks = []

    # > WandB Callback (System metrics, model checkpoint)
    callbacks.append(
        WandbCallback(
            gradient_save_freq=1000,
            model_save_path=os.path.join(cfg['paths']['model_dir'], f"wandb_{run_name}"),
            verbose=2,
        )
    )

    if cfg['adaptive']['enabled']:
        drift_callback = DriftAdaptiveCallback(verbose=1)
        
        drift_callback.scale_factor = cfg['adaptive'].get('scale_factor', 0.1)
        
        callbacks.append(drift_callback)
        print(f">>> Adaptive Mode ENABLED. Scale Factor: {drift_callback.scale_factor}")

    # 6. Train
    try:
        model.learn(
            total_timesteps=cfg['train']['total_timesteps'], 
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("Training interrupted manually...")
    finally:
        wandb.finish()

    # 7. Save Model & Config
    final_save_path = os.path.join(cfg['paths']['model_dir'], run_name)
    model.save(final_save_path)
    print(f"Model saved locally to: {final_save_path}.zip")
    
    with open(os.path.join(log_path, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f)

if __name__ == "__main__":
    main()