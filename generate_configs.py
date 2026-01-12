#!/usr/bin/env python3
"""
Generate comprehensive config files for NS-MDMPI experiments
Creates configs for PPO/SAC/TRPO × {vanilla, moderate, extreme, multi} × environments
"""

import os
import yaml
from pathlib import Path

# Config templates
ENVS = {
    "hopper": {
        "env_id": "Hopper-v4",
        "params": ["friction", "mass_scale", "damping"],
        "base_values": {"friction": 0.9, "mass_scale": 1.0, "damping": 1.0},
        "timesteps": 1000000
    },
    "half cheetah": {
        "env_id": "HalfCheetah-v4",
        "params": ["friction", "damping", "mass_scale"],
        "base_values": {"friction": 0.9, "damping": 1.0, "mass_scale": 1.0},
        "timesteps": 1000000
    },
    "lunarlander": {
        "env_id": "LunarLander-v3",
        "params": ["gravity", "wind_power"],
        "base_values": {"gravity": -10.0, "wind_power": 0.0},
        "timesteps": 500000
    }
}

DRIFT_CONFIGS = {
    "moderate": {
        "sine": {
            "magnitude_factor": 0.3,
            "period": 10000,
            "bounds_factor": 0.5
        },
        "linear": {
            "magnitude_factor": 0.3,
            "period": 15000,
            "bounds_factor": 0.4
        },
        "jump": {
            "magnitude_factor": 0.4,
            "period": 8000,
            "bounds_factor": 0.5
        }
    },
    "extreme": {
        "random_walk": {
            "sigma": 0.3,
            "period": 1000,
            "bounds_factor": 1.5
        },
        "jump": {
            "magnitude_factor": 0.8,
            "period": 3000,
            "bounds_factor": 1.0
        }
    }
}

def generate_single_param_config(env_name, param, drift_type, difficulty, algorithm, enable_nsmdmpi=False):
    """Generate single parameter drift config"""
    env_info = ENVS[env_name]
    drift_info = DRIFT_CONFIGS[difficulty].get(drift_type)
    
    if not drift_info:
        return None
    
    base_value = env_info["base_values"].get(param)
    
    # Build env section
    env_section = {
        "parameter": param,
        "drift_type": drift_type
    }
    
    if drift_type == "random_walk":
        env_section["sigma"] = drift_info["sigma"]
        env_section["period"] = drift_info["period"]
    else:
        env_section["magnitude"] = drift_info["magnitude_factor"] * abs(base_value) if base_value else drift_info["magnitude_factor"]
        env_section["period"] = drift_info["period"]
    
    if base_value is not None:
        env_section["base_value"] = base_value
    
    # Calculate bounds
    if base_value:
        spread = abs(base_value) * drift_info["bounds_factor"]
        if param == "gravity":  # Special case for LunarLander
            env_section["bounds"] = [max(-11.9, base_value - spread), min(-1.0, base_value + spread)]
        else:
            env_section["bounds"] = [max(0.1, base_value - spread), base_value + spread]
    
    # Project name based on nsmdmpi
    project_suffix = "NSMDMPI" if enable_nsmdmpi else "Baseline"
    
    # Full config
    config = {
        "env_id": env_info["env_id"],
        "env": env_section,
        "wandb": {
            "project": f"att_3_{difficulty.capitalize()}_{project_suffix}",
            "tags": [env_name, difficulty, algorithm.lower(), param, drift_type, project_suffix.lower()],
            "mode": "online"
        },
        "train": {
            "algorithm": algorithm,
            "learning_rate": 0.0003 if algorithm != "SAC" else 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "total_timesteps": env_info["timesteps"],
            "seed": 42
        },
        "nsmdmpi": {
            "enabled": enable_nsmdmpi
        },
        "paths": {
            "log_dir": "logs/",
            "model_dir": "models/",
            "video_dir": "videos/"
        }
    }
    
    # Add NS-MDMPI parameters if enabled
    if enable_nsmdmpi:
        config["nsmdmpi"].update({
            "V_R": 10.0,
            "V_P": 10.0,
            "V_pi_star": 5.0,
            "kappa_base": 0.01,
            "lambda_base": 0.01,
            "kappa_adaptive": True,
            "lambda_adaptive": True,
            "sensitivity": 0.1
        })
    
    if algorithm == "TRPO":
        config["train"]["target_kl"] = 0.01
    
    return config

def generate_multi_param_config(env_name, params, algorithm, enable_nsmdmpi=False):
    """Generate multi-parameter drift config"""
    env_info = ENVS[env_name]
    
    env_section = []
    for i, param in enumerate(params):
        base_value = env_info["base_values"].get(param)
        drift_types = ["sine", "linear", "jump"]
        drift_type = drift_types[i % len(drift_types)]
        
        param_config = {
            "parameter": param,
            "drift_type": drift_type,
            "magnitude": 0.3 * abs(base_value) if base_value else 0.3,
            "period": 8000 + i * 2000
        }
        
        if base_value is not None:
            param_config["base_value"] = base_value
            spread = abs(base_value) * 0.4
            if param == "gravity":
                param_config["bounds"] = [max(-11.9, base_value - spread), min(-1.0, base_value + spread)]
            else:
                param_config["bounds"] = [max(0.1, base_value - spread), base_value + spread]
        
        env_section.append(param_config)
    
    project_suffix = "NSMDMPI" if enable_nsmdmpi else "Baseline"
    
    config = {
        "env_id": env_info["env_id"],
        "env": env_section,
        "wandb": {
            "project": f"att_3_Multi_{project_suffix}",
            "tags": [env_name, "multi-param", algorithm.lower(), "+".join(params), project_suffix.lower()],
            "mode": "online"
        },
        "train": {
            "algorithm": algorithm,
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "total_timesteps": env_info["timesteps"],
            "seed": 42
        },
        "nsmdmpi": {
            "enabled": enable_nsmdmpi
        },
        "paths": {
            "log_dir": "logs/",
            "model_dir": "models/",
            "video_dir": "videos/"
        }
    }
    
    if enable_nsmdmpi:
        config["nsmdmpi"].update({
            "V_R": 10.0,
            "V_P": 10.0,
            "V_pi_star": 5.0,
            "kappa_base": 0.01,
            "lambda_base": 0.01,
            "kappa_adaptive": True,
            "lambda_adaptive": True,
            "sensitivity": 0.1
        })
    
    if algorithm == "TRPO":
        config["train"]["target_kl"] = 0.01
    
    return config

def generate_vanilla_config(env_name, algorithm):
    """Generate vanilla (no drift) config"""
    env_info = ENVS[env_name]
    
    config = {
        "env_id": env_info["env_id"],
        "wandb": {
            "project": "att_3_Vanilla_Baselines",
            "tags": [env_name, "vanilla", algorithm.lower(), "baseline"],
            "mode": "online"
        },
        "train": {
            "algorithm": algorithm,
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "total_timesteps": env_info["timesteps"],
            "seed": 42
        },
        "nsmdmpi": {"enabled": False},
        "paths": {
            "log_dir": "logs/",
            "model_dir": "models/",
            "video_dir": "videos/"
        }
    }
    
    if algorithm == "TRPO":
        config["train"]["target_kl"] = 0.01
    
    return config

def main():
    base_dir = Path("configs")
    algorithms = ["PPO", "SAC", "TRPO"]
    
    count = 0
    
    for algo in algorithms:
        print(f"\n=== Generating {algo} configs ===")
        
        # Vanilla configs (only baseline, no nsmdmpi)
        for env_name in ENVS.keys():
            config = generate_vanilla_config(env_name, algo)
            filename = f"{env_name}_vanilla_{algo.lower()}.yaml"
            filepath = base_dir / algo / "vanilla" / filename
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"✅ Created {filepath}")
            count += 1
        
        # Moderate configs (both baseline and nsmdmpi)
        for enable_nsmdmpi in [False, True]:
            suffix = "nsmdmpi" if enable_nsmdmpi else "baseline"
            for env_name, env_info in ENVS.items():
                for param in env_info["params"][:2]:  # First 2 params per env
                    for drift_type in ["sine", "linear", "jump"]:
                        config = generate_single_param_config(env_name, param, drift_type, "moderate", algo, enable_nsmdmpi)
                        if config:
                            filename = f"{env_name}_{param}_{drift_type}_{suffix}_{algo.lower()}.yaml"
                            filepath = base_dir / algo / "moderate" / filename
                            
                            with open(filepath, 'w') as f:
                                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                            print(f"✅ Created {filepath}")
                            count += 1
        
        # Extreme configs (both baseline and nsmdmpi)
        for enable_nsmdmpi in [False, True]:
            suffix = "nsmdmpi" if enable_nsmdmpi else "baseline"
            for env_name, env_info in ENVS.items():
                param = env_info["params"][0]  # Use first param
                for drift_type in ["random_walk", "jump"]:
                    config = generate_single_param_config(env_name, param, drift_type, "extreme", algo, enable_nsmdmpi)
                    if config:
                        filename = f"{env_name}_{param}_{drift_type}_{suffix}_{algo.lower()}.yaml"
                        filepath = base_dir / algo / "extreme" / filename
                        
                        with open(filepath, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        print(f"✅ Created {filepath}")
                        count += 1
        
        # Multi configs (both baseline and nsmdmpi)
        for enable_nsmdmpi in [False, True]:
            suffix = "nsmdmpi" if enable_nsmdmpi else "baseline"
            for env_name, env_info in ENVS.items():
                if len(env_info["params"]) >= 2:
                    params = env_info["params"][:2]
                    config = generate_multi_param_config(env_name, params, algo, enable_nsmdmpi)
                    filename = f"{env_name}_{'_'.join(params)}_{suffix}_{algo.lower()}.yaml"
                    filepath = base_dir / algo / "multi" / filename
                    
                    with open(filepath, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    print(f"✅ Created {filepath}")
                    count += 1
    
    print(f"\n🎉 Total configs created: {count}")
    print(f"   - Vanilla: {len(algorithms) * len(ENVS)} (baseline only)")
    print(f"   - Moderate: {len(algorithms) * sum(min(2, len(env['params'])) * 3 for env in ENVS.values()) * 2} (baseline + nsmdmpi)")
    print(f"   - Extreme: {len(algorithms) * len(ENVS) * 2 * 2} (baseline + nsmdmpi)")
    print(f"   - Multi: {len(algorithms) * len(ENVS) * 2} (baseline + nsmdmpi)")

if __name__ == "__main__":
    main()
