#!/usr/bin/env python3
"""
Generate config files for experiments.

Structure per environment:
- Moderate: 3 params × 4 drift types × 2 (baseline/nsmdmpi) = 24 configs
- Extreme: 3 params × 2 drift types × 2 = 12 configs  
- Multi: 3 versions × 2 = 6 configs
Total: 42 configs per environment
"""

import os
import yaml

# Parameters per category
MODERATE_PARAMS = ['friction', 'damping', 'gravity']
MODERATE_DRIFTS = ['sine', 'random_walk', 'linear', 'jump']

EXTREME_PARAMS = ['friction', 'mass_scale', 'gravity']
EXTREME_DRIFTS = ['random_walk', 'jump']

# Environment-specific settings
ENV_CONFIGS = {
    'HalfCheetah-v4': {
        'default_params': {
            'friction': {'base_value': 1.0, 'bounds': [0.5, 1.5]},
            'damping': {'base_value': 1.0, 'bounds': [0.5, 2.0]},
            'mass_scale': {'base_value': 1.0, 'bounds': [0.5, 2.0]},
            'gravity': {'base_value': -9.81, 'bounds': [-11.0, -8.0]},
        },
        'moderate': {'magnitude': 0.3, 'period': 100000, 'total_timesteps': 2000000},
        'extreme': {'magnitude': 0.6, 'period': 50000, 'total_timesteps': 2000000},
    },
    'Hopper-v4': {
        'default_params': {
            'friction': {'base_value': 1.0, 'bounds': [0.5, 1.5]},
            'damping': {'base_value': 1.0, 'bounds': [0.5, 2.0]},
            'mass_scale': {'base_value': 1.0, 'bounds': [0.5, 2.0]},
            'gravity': {'base_value': -9.81, 'bounds': [-11.0, -8.0]},
        },
        'moderate': {'magnitude': 0.3, 'period': 100000, 'total_timesteps': 2000000},
        'extreme': {'magnitude': 0.6, 'period': 50000, 'total_timesteps': 2000000},
    },
    'Walker2d-v4': {
        'default_params': {
            'friction': {'base_value': 1.0, 'bounds': [0.5, 1.5]},
            'damping': {'base_value': 1.0, 'bounds': [0.5, 2.0]},
            'mass_scale': {'base_value': 1.0, 'bounds': [0.5, 2.0]},
            'gravity': {'base_value': -9.81, 'bounds': [-11.0, -8.0]},
        },
        'moderate': {'magnitude': 0.3, 'period': 100000, 'total_timesteps': 2000000},
        'extreme': {'magnitude': 0.6, 'period': 50000, 'total_timesteps': 2000000},
    },
}

# NS-MDMPI default settings
NSMDMPI_CONFIG = {
    'enabled': True,
    'V_R': 200.0,
    'V_P': 300.0,
    'V_pi_star': 1200.0,
    'kappa_base': 0.2,
    'kappa_min': 0.1,
    'kappa_max': 0.4,
    'lambda_base': 1.0,
    'kappa_adaptive': True,
    'lambda_adaptive': True,
    'trust_region_sensitivity': 2.0,
    'regularization_sensitivity': 1.0,
    'max_ent_coef': 0.1,
    'drift_window_size': 1000,
    'log_freq': 50,
    'save_budget_history': True,
    'budget_save_dir': 'budgets/',
    'verbose': 1,
}


def generate_config(env_id, category, param, drift_type, is_nsmdmpi, algorithm='PPO'):
    """Generate a single config file."""
    env_cfg = ENV_CONFIGS[env_id]
    cat_cfg = env_cfg[category]
    param_cfg = env_cfg['default_params'][param]
    
    env_name = env_id.split('-')[0].lower()
    method = 'nsmdmpi' if is_nsmdmpi else 'baseline'
    
    config = {
        'env_id': env_id,
        'env': {
            'parameter': param,
            'drift_type': drift_type,
            'magnitude': cat_cfg['magnitude'],
            'period': cat_cfg['period'],
            'base_value': param_cfg['base_value'],
            'bounds': param_cfg['bounds'],
        },
        'wandb': {
            'project': f'NS_MDMPI_{env_name.title()}_{category.title()}',
            'tags': [env_name, category, algorithm.lower(), param, drift_type, method],
            'mode': 'online',
        },
        'train': {
            'algorithm': algorithm,
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'total_timesteps': cat_cfg['total_timesteps'],
            'seed': 42,
        },
        'paths': {
            'log_dir': 'logs/',
            'model_dir': 'models/',
            'video_dir': 'videos/',
        },
    }
    
    if is_nsmdmpi:
        config['nsmdmpi'] = NSMDMPI_CONFIG.copy()
    
    return config


def generate_multi_config(env_id, num_params, is_nsmdmpi, algorithm='PPO'):
    """Generate multi-parameter config."""
    env_cfg = ENV_CONFIGS[env_id]
    env_name = env_id.split('-')[0].lower()
    method = 'nsmdmpi' if is_nsmdmpi else 'baseline'
    
    # Select params based on num_params
    params = ['friction', 'damping', 'mass_scale', 'gravity'][:num_params]
    
    env_list = []
    for param in params:
        param_cfg = env_cfg['default_params'][param]
        env_list.append({
            'parameter': param,
            'drift_type': 'random_walk',
            'magnitude': 0.3,
            'period': 100000,
            'sigma': 0.1,
            'base_value': param_cfg['base_value'],
            'bounds': param_cfg['bounds'],
        })
    
    config = {
        'env_id': env_id,
        'env': env_list,
        'wandb': {
            'project': f'NS_MDMPI_{env_name.title()}_Multi',
            'tags': [env_name, 'multi', algorithm.lower(), f'{num_params}param', method],
            'mode': 'online',
        },
        'train': {
            'algorithm': algorithm,
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'total_timesteps': 2000000,
            'seed': 42,
        },
        'paths': {
            'log_dir': 'logs/',
            'model_dir': 'models/',
            'video_dir': 'videos/',
        },
    }
    
    if is_nsmdmpi:
        config['nsmdmpi'] = NSMDMPI_CONFIG.copy()
    
    return config


def save_config(config, filepath):
    """Save config to YAML file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Created: {filepath}")


def main():
    base_dir = 'configs'
    algorithm = 'PPO'  # Can be changed to TRPO, SAC
    
    for env_id in ENV_CONFIGS.keys():
        env_name = env_id.split('-')[0].lower()
        
        # Moderate configs
        for param in MODERATE_PARAMS:
            for drift in MODERATE_DRIFTS:
                for is_nsmdmpi in [False, True]:
                    method = 'nsmdmpi' if is_nsmdmpi else 'baseline'
                    filename = f"{env_name}_{param}_{drift}_{method}_{algorithm.lower()}.yaml"
                    filepath = os.path.join(base_dir, algorithm, 'moderate', filename)
                    
                    config = generate_config(env_id, 'moderate', param, drift, is_nsmdmpi, algorithm)
                    save_config(config, filepath)
        
        # Extreme configs
        for param in EXTREME_PARAMS:
            for drift in EXTREME_DRIFTS:
                for is_nsmdmpi in [False, True]:
                    method = 'nsmdmpi' if is_nsmdmpi else 'baseline'
                    filename = f"{env_name}_{param}_{drift}_{method}_{algorithm.lower()}.yaml"
                    filepath = os.path.join(base_dir, algorithm, 'extreme', filename)
                    
                    config = generate_config(env_id, 'extreme', param, drift, is_nsmdmpi, algorithm)
                    save_config(config, filepath)
        
        # Multi configs
        for num_params in [2, 3, 4]:
            for is_nsmdmpi in [False, True]:
                method = 'nsmdmpi' if is_nsmdmpi else 'baseline'
                filename = f"{env_name}_{num_params}param_randomwalk_{method}_{algorithm.lower()}.yaml"
                filepath = os.path.join(base_dir, algorithm, 'multi', filename)
                
                config = generate_multi_config(env_id, num_params, is_nsmdmpi, algorithm)
                save_config(config, filepath)
    
    print("\n=== Config Generation Complete ===")
    print(f"Generated configs for: {list(ENV_CONFIGS.keys())}")


if __name__ == '__main__':
    main()
