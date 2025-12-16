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
    # Wrap Non-Stationary
    env = NonStationaryCartPoleWrapper(
        env, 
        drift_type=config['env']['drift_type'], 
        change_period=config['env']['change_period']
    )
    if log_dir:
        env = Monitor(env, log_dir, allow_early_resets=True)
    return env

def main():
    parser = argparse.ArgumentParser(description="Insert config path")
    parser.add_argument("--config", type=str, default="configs/cartpole_adaptive.yaml", help="Path to the config file")
    args = parser.parse_args()

    # 1. Load Config
    config_path = args.config
    cfg = load_config(config_path)
    
    # Unique run name
    run_name = f"{cfg['env_id']}_{cfg['env']['drift_type']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if cfg['adaptive']['enabled']:
        run_name += "_Adaptive"
    else:
        run_name += "_Baseline"

    # Create folders
    log_path = os.path.join(cfg['paths']['log_dir'], run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(cfg['paths']['model_dir'], exist_ok=True)

    print(f"--- Training Start: {run_name} ---")

    # ======================================================
    # >>> SETUP WANDB (ONLINE LOGGING) <<<
    # ======================================================
    # sync_tensorboard=True: Auto sync log from SB3 TensorBoard (OFFLINE) to WandB (ONLINE)
    # monitor_gym=True: Auto record videos
    # save_code=True: Save the train.py code to the server for future reference
    wandb.init(
    # Lấy tên project từ file yaml, nếu không có thì fallback về string mặc định
    project=cfg.get('wandb', {}).get('project', "CartPole_Default"),
    
    # Lấy tags từ file yaml
    tags=cfg.get('wandb', {}).get('tags', []),
    
    # Mode online/offline từ yaml
    mode=cfg.get('wandb', {}).get('mode', "online"),
    
    name=run_name,
    config=cfg,
    sync_tensorboard=True, 
    monitor_gym=True,
    save_code=True,
    dir=log_path
    )

    # 2. Setup Env
    env = make_env(cfg, log_path)

    # 3. Setup Model
    # tensorboard_log=... : Đây là chỗ SB3 ghi log OFFLINE
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

    # 4. Setup Callback List
    callbacks = []

    # >>> CALLBACK 1: WandB (Log model, gradient,...)
    callbacks.append(
        WandbCallback(
            gradient_save_freq=1000,
            model_save_path=os.path.join(cfg['paths']['model_dir'], f"wandb_{run_name}"),
            verbose=2,
        )
    )

    # >>> CALLBACK 2: Custom Drift Logic
    if cfg['adaptive']['enabled']:
        drift_callback = DriftAdaptiveCallback(
            check_freq=cfg['adaptive']['check_freq'],
            verbose=1
        )
        drift_callback.adaptive_lr = cfg['adaptive']['adaptive_lr']
        drift_callback.adaptation_steps = cfg['adaptive']['adaptation_steps']
        callbacks.append(drift_callback)

    # 5. Train
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
        # Đóng WandB sạch sẽ kể cả khi lỗi
        wandb.finish()

    # 6. Save Model Local
    save_path = os.path.join(cfg['paths']['model_dir'], run_name)
    model.save(save_path)
    print(f"Model saved locally to: {save_path}.zip")
    
    # 7. Save Config
    with open(os.path.join(log_path, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f)

if __name__ == "__main__":
    main()