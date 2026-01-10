#!/usr/bin/env python
"""
Script to generate TRPO versions of PPO configs.
"""
import yaml
import os
import glob

# Find all PPO baseline configs
ppo_configs = glob.glob("configs/tune_env/CartPole_*_ppo.yaml") + \
              glob.glob("configs/tune_env/CartPole_*_baseline.yaml")

print(f"Found {len(ppo_configs)} PPO configs to convert to TRPO")

for ppo_config_path in ppo_configs:
    # Load PPO config
    with open(ppo_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Change algorithm to TRPO
    config['train']['algorithm'] = 'TRPO'
    
    # TRPO-specific hyperparameters
    if 'target_kl' not in config['train']:
        config['train']['target_kl'] = 0.01  # TRPO default
    
    # Generate TRPO filename
    trpo_config_path = ppo_config_path.replace('_ppo.yaml', '_trpo.yaml').replace('_baseline.yaml', '_trpo.yaml')
    
    # Write TRPO config
    with open(trpo_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created: {trpo_config_path}")

print(f"\n✅ Generated {len(ppo_configs)} TRPO configs!")
