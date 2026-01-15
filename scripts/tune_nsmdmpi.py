#!/usr/bin/env python3
"""
Hyperparameter tuning for NS-MDMPI.
Generates configs and runs grid search.
"""

import os
import yaml
import itertools
import subprocess
import argparse

# Grid search space
GRID = {
    'trust_region_sensitivity': [0.5, 1.0, 1.5],
    'V_pi_star': [1000, 1500, 2000],
    'kappa_min': [0.1, 0.15, 0.2],
}

# Base config template
BASE_CONFIG = {
    'env_id': 'HalfCheetah-v4',
    'env': {
        'parameter': 'friction',
        'drift_type': 'random_walk',
        'magnitude': 0.3,
        'period': 100000,
        'base_value': 1.0,
        'bounds': [0.5, 1.5],
    },
    'wandb': {
        'project': 'NS_MDMPI_Tuning',
        'mode': 'online',
    },
    'train': {
        'algorithm': 'PPO',
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'gamma': 0.99,
        'total_timesteps': 500000,  # Shorter for tuning
        'seed': 42,
    },
    'paths': {
        'log_dir': 'logs/',
        'model_dir': 'models/',
        'video_dir': 'videos/',
    },
    'nsmdmpi': {
        'enabled': True,
        'V_R': 100.0,
        'V_P': 150.0,
        'V_pi_star': 600.0,
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
        'verbose': 1,
    },
}


def generate_configs():
    """Generate all config combinations."""
    configs = []
    
    keys = list(GRID.keys())
    values = list(GRID.values())
    
    for combo in itertools.product(*values):
        config = yaml.safe_load(yaml.dump(BASE_CONFIG))  # Deep copy
        
        # Apply hyperparams
        param_str = []
        for i, key in enumerate(keys):
            config['nsmdmpi'][key] = combo[i]
            param_str.append(f"{key}={combo[i]}")
        
        # Update tags
        config['wandb']['tags'] = ['tuning', 'halfcheetah'] + param_str
        
        # Generate filename
        filename = f"tune_{'_'.join(str(v) for v in combo)}.yaml"
        configs.append((filename, config))
    
    return configs


def save_configs(configs, output_dir='configs/tuning'):
    """Save configs to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    paths = []
    for filename, config in configs:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        paths.append(filepath)
        print(f"Created: {filepath}")
    
    return paths


def run_experiments(config_paths, parallel=1):
    """Run experiments."""
    print(f"\n=== Running {len(config_paths)} experiments ===")
    
    for i, path in enumerate(config_paths):
        print(f"\n[{i+1}/{len(config_paths)}] Running: {path}")
        cmd = f"python scripts/train.py --config {path}"
        
        if parallel > 1:
            # Background execution
            subprocess.Popen(cmd, shell=True)
        else:
            # Sequential
            subprocess.run(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate configs only')
    parser.add_argument('--run', action='store_true', help='Run all tuning experiments')
    parser.add_argument('--parallel', type=int, default=1, help='Parallel runs')
    args = parser.parse_args()
    
    configs = generate_configs()
    print(f"Total combinations: {len(configs)}")
    
    if args.generate or args.run:
        paths = save_configs(configs)
        
        if args.run:
            run_experiments(paths, args.parallel)
    else:
        # Just print combinations
        for filename, _ in configs:
            print(f"  {filename}")
        print(f"\nUse --generate to create configs, --run to execute")


if __name__ == '__main__':
    main()
