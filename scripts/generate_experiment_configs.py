#!/usr/bin/env python3
"""Generate 40 experiment config files (10 per env × 4 envs)"""

import os
import yaml

# Config template
BASE_CONFIG = {
    'wandb': {
        'project': 'att_19_{env}',
        'tags': [],
        'mode': 'online'
    },
    'train': {
        'algorithm': 'PPO',
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'gamma': 0.99,
        'total_timesteps': 2000000,
        'seed': 42
    },
    'paths': {
        'log_dir': 'logs/',
        'model_dir': 'models/',
        'video_dir': 'videos/'
    },
    'nsmdmpi': {
        'enabled': True,
        'V_R': 100.0,
        'V_P': 200.0,
        'V_pi_star': 1000.0,
        'kappa_base': 0.2,
        'kappa_min': 0.15,
        'kappa_max': 0.4,
        'lambda_base': 1.0,
        'kappa_adaptive': True,
        'lambda_adaptive': True,
        'trust_region_sensitivity': 1.0,
        'regularization_sensitivity': 1.0,
        'max_ent_coef': 0.1,
        'drift_window_size': 1000,
        'log_freq': 50,
        'save_budget_history': True,
        'budget_save_dir': 'budgets/',
        'verbose': 1
    }
}

# Environment configs with param mappings
ENV_CONFIGS = {
    'halfcheetah': {
        'env_id': 'HalfCheetah-v4',
        'param1': ('friction', [0.5, 1.5], 1.0),
        'param2': ('damping', [0.5, 1.5], 1.0),
        'param3': ('gravity', [-15.0, -5.0], -9.81),
        'nsmdmpi': {'V_R': 100.0, 'V_P': 200.0, 'V_pi_star': 1200.0}
    },
    'hopper': {
        'env_id': 'Hopper-v4',
        'param1': ('friction', [0.5, 1.5], 1.0),
        'param2': ('mass_scale', [0.7, 1.3], 1.0),
        'param3': ('gravity', [-15.0, -5.0], -9.81),
        'nsmdmpi': {'V_R': 80.0, 'V_P': 150.0, 'V_pi_star': 1000.0}
    },
    'walker2d': {
        'env_id': 'Walker2d-v4',
        'param1': ('friction', [0.5, 1.5], 1.0),
        'param2': ('mass_scale', [0.7, 1.3], 1.0),
        'param3': ('gravity', [-15.0, -5.0], -9.81),
        'nsmdmpi': {'V_R': 80.0, 'V_P': 150.0, 'V_pi_star': 1000.0}
    },
    'lunarlander': {
        'env_id': 'LunarLander-v3',
        'param1': ('gravity', [-15.0, -5.0], -10.0),
        'param2': ('wind_power', [10.0, 25.0], 15.0),
        'param3': ('main_engine_power', [10.0, 18.0], 13.0),
        'nsmdmpi': {'V_R': 15.0, 'V_P': 30.0, 'V_pi_star': 100.0}
    }
}

# 10 config scenarios
SCENARIOS = [
    ('C01', 'param1', 'sine'),
    ('C02', 'param1', 'linear'),
    ('C03', 'param1', 'random_walk'),
    ('C04', 'param1', 'jump'),
    ('C05', 'param2', 'sine'),
    ('C06', 'param2', 'linear'),
    ('C07', 'param2', 'random_walk'),
    ('C08', 'param3', 'sine'),
    ('C09', 'param3', 'jump'),
    ('C10', 'multi', 'sine'),  # param1 + param2
]

def create_config(env_name, scenario_id, param_key, drift_type):
    """Create config dict for env/scenario combination"""
    env_cfg = ENV_CONFIGS[env_name]
    
    config = {
        'env_id': env_cfg['env_id'],
        'env': {
            'drift_type': drift_type,
            'magnitude': 0.3,
            'period': 100000,
        },
        'wandb': {
            'project': f'att_19_{env_name.capitalize()}',
            'tags': [env_name, scenario_id, drift_type, 'nsmdmpi'],
            'mode': 'online'
        },
        'train': BASE_CONFIG['train'].copy(),
        'paths': BASE_CONFIG['paths'].copy(),
        'nsmdmpi': {**BASE_CONFIG['nsmdmpi'], **env_cfg['nsmdmpi']}
    }
    
    if param_key == 'multi':
        # Multi-param: param1 + param2
        p1_name, p1_bounds, p1_base = env_cfg['param1']
        p2_name, p2_bounds, p2_base = env_cfg['param2']
        config['env']['parameters'] = [
            {'parameter': p1_name, 'base_value': p1_base, 'bounds': p1_bounds},
            {'parameter': p2_name, 'base_value': p2_base, 'bounds': p2_bounds}
        ]
    else:
        # Single param
        param_name, bounds, base_value = env_cfg[param_key]
        config['env']['parameter'] = param_name
        config['env']['base_value'] = base_value
        config['env']['bounds'] = bounds
    
    return config

def main():
    base_dir = 'configs/experiments'
    os.makedirs(base_dir, exist_ok=True)
    
    count = 0
    for env_name in ENV_CONFIGS.keys():
        env_dir = os.path.join(base_dir, env_name)
        os.makedirs(env_dir, exist_ok=True)
        
        for scenario_id, param_key, drift_type in SCENARIOS:
            config = create_config(env_name, scenario_id, param_key, drift_type)
            
            # Filename
            if param_key == 'multi':
                param_str = 'multi'
            else:
                param_str = config['env'].get('parameter', 'multi')
            
            filename = f"{scenario_id}_{param_str}_{drift_type}.yaml"
            filepath = os.path.join(env_dir, filename)
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            count += 1
            print(f"Created: {filepath}")
    
    print(f"\nTotal: {count} config files created")

if __name__ == '__main__':
    main()
