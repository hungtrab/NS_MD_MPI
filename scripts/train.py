import sys
import os
import yaml
import datetime
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# TRPO from sb3-contrib (optional, will fallback gracefully)
try:
    from sb3_contrib import TRPO
    TRPO_AVAILABLE = True
except ImportError:
    TRPO_AVAILABLE = False
    print("Warning: sb3-contrib not installed. TRPO unavailable. Install with: pip install sb3-contrib")

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.envs import make_nonstationary_env, get_wrapper_for_env
from src.callbacks.drift_callback import DriftAdaptiveCallback

# Algorithm registry
ALGORITHM_REGISTRY = {
    'PPO': PPO,
    'SAC': SAC,
}
if TRPO_AVAILABLE:
    ALGORITHM_REGISTRY['TRPO'] = TRPO


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(config, log_dir=None, seed=None):
    """
    Create a non-stationary environment based on config.
    
    Automatically selects the appropriate wrapper based on env_id.
    Supports: CartPole, MountainCar, FrozenLake, MiniGrid, HalfCheetah
    
    Args:
        config: Configuration dictionary
        log_dir: Optional log directory for Monitor wrapper
        seed: Random seed for reproducibility
    """
    env_id = config['env_id']
    
    # Build drift configuration from YAML
    drift_conf = {
        'parameter': config['env'].get('parameter', 'gravity'),
        'drift_type': config['env'].get('drift_type', 'static'),
        'magnitude': config['env'].get('magnitude', 0.0),
        'period': config['env'].get('period', 10000),
        'sigma': config['env'].get('sigma', 0.1),
        'bounds': config['env'].get('bounds', [0.0, 20.0]),
    }
    
    # Get additional env kwargs if specified
    env_kwargs = config.get('env_kwargs', {})
    
    # Create non-stationary environment using factory
    try:
        env = make_nonstationary_env(env_id, drift_conf, seed=seed, **env_kwargs)
        # env = gym.make(env_id, **env_kwargs)
    except ValueError as e:
        print(f"Warning: {e}")
        print(f"Falling back to base environment without drift wrapper")
        env = gym.make(env_id, **env_kwargs)
    
    if log_dir:
        env = Monitor(env, log_dir, allow_early_resets=True)
    
    return env

def get_algorithm_class(algo_name: str):
    """Get algorithm class by name."""
    algo_upper = algo_name.upper()
    if algo_upper not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: {available}")
    return ALGORITHM_REGISTRY[algo_upper]


def main():
    parser = argparse.ArgumentParser(description="Train RL agent with drift-adaptive mechanisms")
    parser.add_argument("--config", type=str, default="configs/cartpole_adaptive.yaml", help="Path to the config file")
    parser.add_argument("--exp_name", type=str, default=None, help="Override run name for easier filtering")
    parser.add_argument("--algo", type=str, default=None, help="Override algorithm (PPO, SAC, TRPO)")
    args = parser.parse_args()

    # 1. Load Config
    config_path = args.config
    cfg = load_config(config_path)
    
    # Determine algorithm (CLI override > config > default)
    algo_name = args.algo or cfg.get('train', {}).get('algorithm', 'PPO')
    algo_name = algo_name.upper()
    
    # Validate algorithm
    if algo_name not in ALGORITHM_REGISTRY:
        print(f"Error: Algorithm '{algo_name}' not available.")
        print(f"Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")
        if algo_name == 'TRPO' and not TRPO_AVAILABLE:
            print("Install sb3-contrib: pip install sb3-contrib")
        return
    
    # Unique run name
    if args.exp_name:
        run_name = args.exp_name
    else:
        run_name = f"{cfg['env_id']}_{algo_name}_{cfg['env']['drift_type']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
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

    # 3. Setup Model with Algorithm Factory
    # tensorboard_log=... : Đây là chỗ SB3 ghi log OFFLINE
    AlgoClass = get_algorithm_class(algo_name)
    
    # Build algorithm-specific kwargs
    model_kwargs = {
        'policy': "MlpPolicy",
        'env': env,
        'learning_rate': cfg['train']['learning_rate'],
        'gamma': cfg['train']['gamma'],
        'verbose': 1,
        'tensorboard_log': cfg['paths']['log_dir'],
    }
    
    # Algorithm-specific parameters
    if algo_name in ['PPO', 'TRPO']:
        # On-policy algorithms use n_steps and batch_size
        model_kwargs['n_steps'] = cfg['train'].get('n_steps', 2048)
        model_kwargs['batch_size'] = cfg['train'].get('batch_size', 64)
    elif algo_name == 'SAC':
        # SAC is off-policy, uses buffer_size and batch_size differently
        model_kwargs['buffer_size'] = cfg['train'].get('buffer_size', 100000)
        model_kwargs['batch_size'] = cfg['train'].get('batch_size', 256)
        model_kwargs['learning_starts'] = cfg['train'].get('learning_starts', 1000)
        model_kwargs['tau'] = cfg['train'].get('tau', 0.005)
    
    print(f">>> Initializing {algo_name} with kwargs: {list(model_kwargs.keys())}")
    model = AlgoClass(**model_kwargs)

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

    # >>> CALLBACK 2: Custom Drift Logic (Algorithm-Specific)
    if cfg['adaptive']['enabled']:
        adaptive_cfg = cfg['adaptive']
        drift_callback = DriftAdaptiveCallback(
            # Environment parameter tracking
            target_param=cfg['env'].get('parameter', 'gravity'),
            base_value=9.8,  # Will be auto-detected from env
            
            # Learning rate adaptation (all algorithms)
            scale_factor=adaptive_cfg.get('scale_factor', 0.1),
            min_lr_multiplier=adaptive_cfg.get('min_lr_multiplier', 0.5),
            max_lr_multiplier=adaptive_cfg.get('max_lr_multiplier', 3.0),
            
            # PPO-specific: clip range adaptation
            adapt_clip_range=adaptive_cfg.get('adapt_clip_range', True),
            base_clip_range=adaptive_cfg.get('base_clip_range', 0.2),
            min_clip_range=adaptive_cfg.get('min_clip_range', 0.05),
            max_clip_range=adaptive_cfg.get('max_clip_range', 0.4),
            
            # Entropy adaptation (PPO/SAC)
            adapt_entropy=adaptive_cfg.get('adapt_entropy', True),
            base_ent_coef=adaptive_cfg.get('base_ent_coef', 0.0),  # 0 = auto-detect
            min_ent_coef=adaptive_cfg.get('min_ent_coef', 0.0),
            max_ent_coef=adaptive_cfg.get('max_ent_coef', 0.1),
            
            # TRPO-specific: target KL adaptation
            adapt_target_kl=adaptive_cfg.get('adapt_target_kl', True),
            base_target_kl=adaptive_cfg.get('base_target_kl', 0.01),
            min_target_kl=adaptive_cfg.get('min_target_kl', 0.001),
            max_target_kl=adaptive_cfg.get('max_target_kl', 0.05),
            
            # Logging
            log_freq=adaptive_cfg.get('log_freq', 100),
            verbose=1
        )
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