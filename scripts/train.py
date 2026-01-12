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
from src.callbacks.nsmdmpi_callback import NSMDMPICallback

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


def make_env(config, log_dir=None, seed=None, num_envs=1):
    """
    Create a non-stationary environment based on config.
    
    Automatically selects the appropriate wrapper based on env_id.
    Supports: CartPole, MountainCar, FrozenLake, MiniGrid, HalfCheetah
    
    Args:
        config: Configuration dictionary
        log_dir: Optional log directory for Monitor wrapper
        seed: Random seed for reproducibility
        num_envs: Number of parallel environments (only 1 supported for non-Procgen)
    """
    env_id = config['env_id']
    
    # Parse drift configuration
    # Check if vanilla (no drift) or non-stationary
    if 'env' not in config or config['env'] is None:
        drift_conf = None  # Vanilla baseline
    elif isinstance(config['env'], list):
        # Multi-parameter: env is already a list of drift configs
        drift_conf = config['env']
    else:
        # Single-parameter: convert dict to standard drift config format
        drift_conf = {
            'parameter': config['env'].get('parameter', 'gravity'),
            'drift_type': config['env'].get('drift_type', 'static'),
            'magnitude': config['env'].get('magnitude', 0.0),
            'period': config['env'].get('period', 1000),
            'sigma': config['env'].get('sigma', 0.1),
            'bounds': config['env'].get('bounds', None),
            'base_value': config['env'].get('base_value', None),
        }
    
    # Get additional env kwargs if specified
    env_kwargs = config.get('env_kwargs', {})
    if 'procgen' in env_id.lower():
        try:
            import numpy as np
            
            # Check NumPy version compatibility
            numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
            if numpy_version >= (2, 0):
                print("=" * 70)
                print("ERROR: Procgen is incompatible with NumPy 2.0+")
                print("=" * 70)
                print(f"Current NumPy version: {np.__version__}")
                print("\nOptions:")
                print("  1. Downgrade NumPy: pip install 'numpy<2.0'")
                print("  2. Use other environments: CartPole, MountainCar, FrozenLake, etc.")
                print("=" * 70)
                raise RuntimeError("NumPy version incompatibility with Procgen")
            
            from procgen import ProcgenEnv
            from stable_baselines3.common.vec_env import VecMonitor
            import gymnasium as gym
            from gymnasium import spaces as gym_spaces
            import gym as old_gym
            from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
            import numpy as np
            
            # Wrapper to convert gym spaces to gymnasium spaces
            class GymToGymnasiumWrapper(VecEnvWrapper):
                """Convert old gym spaces to gymnasium spaces for SB3 compatibility."""
                
                def __init__(self, venv):
                    super().__init__(venv)
                    
                    # Convert observation space from gym to gymnasium
                    old_obs_space = venv.observation_space
                    
                    # Handle Dict observation spaces (like Procgen's Dict with 'rgb' key)
                    if isinstance(old_obs_space, old_gym.spaces.Dict):
                        # Extract the 'rgb' observation if present
                        if 'rgb' in old_obs_space.spaces:
                            rgb_space = old_obs_space.spaces['rgb']
                            self.observation_space = gym_spaces.Box(
                                low=rgb_space.low,
                                high=rgb_space.high,
                                shape=rgb_space.shape,
                                dtype=rgb_space.dtype
                            )
                            self._extract_rgb = True
                        else:
                            raise ValueError(f"Dict observation space doesn't contain 'rgb' key: {old_obs_space}")
                    elif isinstance(old_obs_space, old_gym.spaces.Box):
                        self.observation_space = gym_spaces.Box(
                            low=old_obs_space.low,
                            high=old_obs_space.high,
                            shape=old_obs_space.shape,
                            dtype=old_obs_space.dtype
                        )
                        self._extract_rgb = False
                    elif isinstance(old_obs_space, old_gym.spaces.Discrete):
                        self.observation_space = gym_spaces.Discrete(old_obs_space.n)
                        self._extract_rgb = False
                    else:
                        # For other spaces, try to use them directly
                        self.observation_space = old_obs_space
                        self._extract_rgb = False
                    
                    # Convert action space
                    old_act_space = venv.action_space
                    if isinstance(old_act_space, old_gym.spaces.Discrete):
                        self.action_space = gym_spaces.Discrete(old_act_space.n)
                    elif isinstance(old_act_space, old_gym.spaces.Box):
                        self.action_space = gym_spaces.Box(
                            low=old_act_space.low,
                            high=old_act_space.high,
                            shape=old_act_space.shape,
                            dtype=old_act_space.dtype
                        )
                    else:
                        self.action_space = old_act_space
                
                def reset(self):
                    obs = self.venv.reset()
                    # Extract 'rgb' from Dict observations if needed
                    if self._extract_rgb and isinstance(obs, dict):
                        obs = obs['rgb']
                    return obs
                
                def step_async(self, actions):
                    self.venv.step_async(actions)
                
                def step_wait(self):
                    obs, rewards, dones, infos = self.venv.step_wait()
                    # Extract 'rgb' from Dict observations if needed
                    if self._extract_rgb and isinstance(obs, dict):
                        obs = obs['rgb']
                    return obs, rewards, dones, infos
                
                def __getstate__(self):
                    """Support for pickle serialization - exclude unpicklable venv."""
                    state = self.__dict__.copy()
                    # Remove the vectorized environment which contains thread locks
                    if 'venv' in state:
                        del state['venv']
                    return state
                
                def __setstate__(self, state):
                    """Support for pickle deserialization."""
                    self.__dict__.update(state)
                    # Note: venv will need to be recreated if loaded from pickle
            # Procgen requires special configuration
            env_name = env_id.split('-')[1] if '-' in env_id else env_id.replace('procgen', '')  # Extract game name
            
            # Get Procgen-specific config from env section
            distribution_mode = config['env'].get('distribution_mode', 'easy')
            num_levels = config['env'].get('num_levels', 500)
            use_backgrounds = config['env'].get('use_backgrounds', True)
            num_envs = config.get('num_envs', 1)
            
            print(f">>> [Procgen] Creating {env_name} environment")
            print(f"    - distribution_mode: {distribution_mode}")
            print(f"    - num_levels: {num_levels}")
            print(f"    - num_envs: {num_envs}")
            
            env = ProcgenEnv(
                num_envs=num_envs,
                env_name=env_name,
                distribution_mode=distribution_mode,
                use_backgrounds=use_backgrounds,
                restrict_themes=False,
                start_level=0,
                num_levels=num_levels,
            )
            
            # Wrap with SB3-compatible wrappers
            # First, add monitor for logging
            if log_dir:
                env = VecMonitor(env, log_dir)
            
            # Convert gym spaces to gymnasium spaces for SB3 compatibility
            env = GymToGymnasiumWrapper(env)
            
            print(f">>> [Procgen] Environment created successfully (No physical drift injection)")
            print(f"    - Observation space: {env.observation_space}")
            print(f"    - Action space: {env.action_space}")
            return env
            
        except ImportError as e:
            print(f"Error: procgen package not installed. Install with: pip install procgen")
            print(f"Details: {e}")
            raise
        except Exception as e:
            print(f"Error creating Procgen environment: {e}")
            raise

    # Create non-stationary environment using factory (or vanilla if drift_conf is None)
    try:
        if drift_conf is None:
            # Vanilla baseline - no drift wrapper
            env = gym.make(env_id, **env_kwargs)
        else:
            # Non-stationary with drift
            env = make_nonstationary_env(env_id, drift_conf, seed=seed, **env_kwargs)
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
    parser.add_argument("--num_envs", type=int, default=0, help="Number of parallel envs (0 = auto select)")
    args = parser.parse_args()

    # 1. Load Config
    config_path = args.config
    cfg = load_config(config_path)
    
    # Determine algorithm (CLI override > config > default)
    algo_name = args.algo or cfg.get('train', {}).get('algorithm', 'PPO')
    algo_name = algo_name.upper()
    
    # Determine num_envs
    if args.num_envs > 0:
        num_envs = args.num_envs
    else:
        # Default: 4 for PPO/TRPO, 1 for SAC
        num_envs = 4 if algo_name in ['PPO', 'TRPO'] else 1
    
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
        # Generate run name with timestamp
        # Check if vanilla (no drift) or non-stationary
        if 'env' not in cfg or cfg['env'] is None:
            # Vanilla baseline
            run_name = f"{cfg['env_id']}_{algo_name}_vanilla_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        elif isinstance(cfg['env'], list):
            # Multi-parameter: use first param's drift type or 'multi'  
            drift_type = cfg['env'][0].get('drift_type', 'multi') if cfg['env'] else 'multi'
            run_name = f"{cfg['env_id']}_{algo_name}_multi-param_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        else:
            # Single-parameter
            drift_type = cfg['env'].get('drift_type', 'static')
            run_name = f"{cfg['env_id']}_{algo_name}_{drift_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if cfg['adaptive']['enabled']:
            run_name += "_Adaptive"
        elif cfg['nsmdmpi']['enabled']:
            run_name += "_NSMDMPI"
        else:
            run_name += "_Baseline"

    # Create folders
    log_path = os.path.join(cfg['paths']['log_dir'], run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(cfg['paths']['model_dir'], exist_ok=True)

    print(f"--- Training Start: {run_name} ---")
    print(f"--- Num Envs: {num_envs} ---")

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
    env = make_env(cfg, log_path, num_envs=num_envs)

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

    # >>> CALLBACK 2: NS-MD-MPI or Adaptive Drift Logic
    # Check if NS-MD-MPI is enabled (Algorithm 1 from paper)
    if cfg.get('nsmdmpi', {}).get('enabled', False):
        nsmdmpi_cfg = cfg['nsmdmpi']
        nsmdmpi_callback = NSMDMPICallback(
            # Variation budgets (V_R, V_P, V_π*)
            V_R=nsmdmpi_cfg.get('V_R', 10.0),
            V_P=nsmdmpi_cfg.get('V_P', 10.0),
            V_pi_star=nsmdmpi_cfg.get('V_pi_star', 5.0),
            auto_estimate_budgets=nsmdmpi_cfg.get('auto_estimate_budgets', False),
            budget_scale_factor=nsmdmpi_cfg.get('budget_scale_factor', 1.5),
            
            # Trust region (κ_t)
            kappa_base=nsmdmpi_cfg.get('kappa_base', 0.2),
            kappa_min=nsmdmpi_cfg.get('kappa_min', 0.05),
            kappa_max=nsmdmpi_cfg.get('kappa_max', 0.4),
            kappa_adaptive=nsmdmpi_cfg.get('kappa_adaptive', True),
            trust_region_sensitivity=nsmdmpi_cfg.get('trust_region_sensitivity', 5.0),
            
            # Regularization (λ_t)
            lambda_base=nsmdmpi_cfg.get('lambda_base', 1.0),
            lambda_min=nsmdmpi_cfg.get('lambda_min', 0.1),
            lambda_max=nsmdmpi_cfg.get('lambda_max', 10.0),
            lambda_adaptive=nsmdmpi_cfg.get('lambda_adaptive', True),
            regularization_sensitivity=nsmdmpi_cfg.get('regularization_sensitivity', 2.0),
            
            # Drift estimation
            drift_weights=tuple(nsmdmpi_cfg.get('drift_weights', {}).values()) or (1.0, 1.0, 0.5),
            drift_window_size=nsmdmpi_cfg.get('drift_window_size', 1000),
            drift_min_samples=nsmdmpi_cfg.get('drift_min_samples', 100),
            
            # Entropy adaptation
            adapt_entropy=nsmdmpi_cfg.get('adapt_entropy', True),
            base_ent_coef=nsmdmpi_cfg.get('base_ent_coef', 0.0),
            min_ent_coef=nsmdmpi_cfg.get('min_ent_coef', 0.0),
            max_ent_coef=nsmdmpi_cfg.get('max_ent_coef', 0.1),
            
            # Logging
            log_freq=nsmdmpi_cfg.get('log_freq', 100),
            save_budget_history=nsmdmpi_cfg.get('save_budget_history', True),
            budget_save_dir=nsmdmpi_cfg.get('budget_save_dir', 'budgets/'),
            verbose=nsmdmpi_cfg.get('verbose', 1),
        )
        callbacks.append(nsmdmpi_callback)
        run_name += "_NSMDMPI"
        
    # Otherwise, check for baseline adaptive (heuristic method)
    elif cfg.get('adaptive', {}).get('enabled', False):
        adaptive_cfg = cfg['adaptive']
        drift_callback = DriftAdaptiveCallback(
            # Environment parameter tracking
            # Handle both single-param (dict) and multi-param (list)
            target_param=(cfg['env'][0].get('parameter', 'gravity') 
                          if isinstance(cfg['env'], list) 
                          else cfg['env'].get('parameter', 'gravity')),
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
        run_name += "_Adaptive"

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