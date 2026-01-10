"""
Non-Stationary Environment Wrappers.

Implements Gym wrappers that introduce dynamics drift to standard environments,
converting them into Non-Stationary MDPs for research purposes.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, List, Union

from .drift_generator import DriftGenerator, DriftConfig, create_drift_generator


class NonStationaryCartPoleWrapper(gym.Wrapper):
    """
    A wrapper that converts the standard CartPole into a Non-Stationary MDP.
    It introduces drift to physical parameters (gravity, mass, length) over time.
    
    Theoretical Basis:
        - Simulates Dynamics Drift (Δ_P) as described in Eq. 11
        - Implements specific drift patterns: Linear, Sinusoidal, Jumps, Random Walk
    
    Usage:
        drift_conf = {
            'parameter': 'gravity',
            'drift_type': 'sine',
            'magnitude': 5.0,
            'period': 10000,
        }
        env = gym.make('CartPole-v1')
        env = NonStationaryCartPoleWrapper(env, drift_conf)
    """
    
    # Valid parameters that can be drifted in CartPole
    VALID_PARAMS = ['gravity', 'masscart', 'masspole', 'length']
    
    def __init__(
        self, 
        env: gym.Env, 
        drift_conf: Union[Dict[str, Any], List[Dict[str, Any]]],
        seed: Optional[int] = None
    ):
        """
        Initialize the non-stationary wrapper.
        
        Args:
            env: The base CartPole environment
            drift_conf: Drift configuration dict or list of dicts for multiple params
            seed: Random seed for reproducibility
        """
        super().__init__(env)
        
        # Handle single config or multiple configs
        if isinstance(drift_conf, dict):
            drift_conf = [drift_conf]
        
        self.drift_configs = drift_conf
        self.step_counter = 0
        self.total_steps = 0  # Tracks total steps across episodes
        
        # Store original parameters for reference
        self.original_params = {
            'gravity': self.unwrapped.gravity,
            'masscart': self.unwrapped.masscart,
            'masspole': self.unwrapped.masspole,
            'length': self.unwrapped.length
        }
        
        # Create drift generators for each parameter
        self.drift_generators: Dict[str, DriftGenerator] = {}
        
        for conf in self.drift_configs:
            param = conf.get('parameter', 'gravity')
            if param not in self.VALID_PARAMS:
                raise ValueError(f"Invalid parameter '{param}'. Must be one of {self.VALID_PARAMS}")
            
            # Set base value from environment if not specified
            if 'base_value' not in conf:
                conf['base_value'] = self.original_params[param]
            
            self.drift_generators[param] = create_drift_generator(conf, seed=seed)
        
        # Store target param for backwards compatibility
        self.target_param = list(self.drift_generators.keys())[0]
        
        # For logging
        self._current_drift_info = {}
        
        print(f">>> [Wrapper] Initialized Non-Stationary CartPole")
        for param, gen in self.drift_generators.items():
            print(f"    - {param}: {gen.config.drift_type} (magnitude={gen.config.magnitude}, period={gen.config.period})")

    def step(self, action):
        """Execute one step with drifting physics."""
        # 1. Update physics engine before the step to simulate continuous drift
        self._update_physics()
        
        # 2. Execute action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 3. Log current physical parameters to info dict (for Oracle/Callback access)
        info['drift/step'] = self.total_steps
        info['drift/params'] = {}
        
        for param, gen in self.drift_generators.items():
            current_val = getattr(self.unwrapped, param)
            base_val = self.original_params[param]
            info['drift/params'][param] = {
                'current': current_val,
                'base': base_val,
                'delta': current_val - base_val,
            }
        
        # Backwards compatible: single param access
        if len(self.drift_generators) == 1:
            param = list(self.drift_generators.keys())[0]
            info['drift/current_value'] = getattr(self.unwrapped, param)
            info['drift/parameter'] = param
        
        self.step_counter += 1
        self.total_steps += 1
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment."""
        self.step_counter = 0
        # Note: We don't reset total_steps to maintain drift continuity across episodes
        
        # Update physics to current drift state
        self._update_physics()
        
        return self.env.reset(**kwargs)

    def _update_physics(self):
        """Update all drifting parameters based on current timestep."""
        for param, gen in self.drift_generators.items():
            new_value = gen.get_value(self.total_steps)
            self._set_env_param(param, new_value)

    def _set_env_param(self, param_name: str, value: float):
        """
        Updates the environment parameter and recalculates dependent variables.
        Crucial for maintaining physics consistency in CartPole.
        """
        setattr(self.unwrapped, param_name, value)
        
        # Recalculate derived physical quantities
        if param_name in ['masscart', 'masspole']:
            self.unwrapped.total_mass = self.unwrapped.masspole + self.unwrapped.masscart
        
        if param_name in ['length', 'masspole']:
            self.unwrapped.polemass_length = self.unwrapped.masspole * self.unwrapped.length
    
    def get_drift_info(self) -> Dict[str, Any]:
        """Get current drift information for all parameters."""
        info = {
            'total_steps': self.total_steps,
            'episode_steps': self.step_counter,
            'parameters': {}
        }
        
        for param, gen in self.drift_generators.items():
            info['parameters'][param] = gen.get_drift_info(self.total_steps)
        
        return info
    
    def set_total_steps(self, steps: int):
        """
        Manually set the total step counter.
        Useful for evaluation at specific drift points.
        """
        self.total_steps = steps
        self._update_physics()


class NonStationaryLunarLanderWrapper(gym.Wrapper):
    """
    Non-Stationary wrapper for LunarLander-v2 environment.
    
    Since LunarLander parameters (gravity, wind_power, turbulence_power) can only
    be set during __init__, this wrapper implements PER-EPISODE drift:
    - Parameters are updated on each reset() call
    - Creates non-stationary MDP across episodes
    - Safer than trying to modify Box2D physics during episodes
    
    Supported Parameters:
        - gravity: float (-12.0 to 0.0) - gravitational force
        - wind_power: float (0.0 to 20.0) - wind strength  
        - turbulence_power: float (0.0 to 2.0) - turbulence/noise level
    
    Usage:
        drift_conf = {
            'parameter': 'gravity',
            'drift_type': 'jump',
            'magnitude': 5.0,
            'period': 10,  # episodes, not steps
        }
        env = gym.make('LunarLander-v2')
        env = NonStationaryLunarLanderWrapper(env, drift_conf)
    """
    
    VALID_PARAMS = ['gravity', 'wind_power', 'turbulence_power']
    
    def __init__(
        self,
        env: gym.Env,
        drift_conf: Union[Dict[str, Any], List[Dict[str, Any]]],
        seed: Optional[int] = None
    ):
        """
        Initialize the non-stationary LunarLander wrapper.
        
        Args:
            env: Base LunarLander-v2 environment
            drift_conf: Drift configuration(s)
            seed: Random seed for reproducibility
        """
        super().__init__(env)
        
        # Handle single config or multiple configs
        if isinstance(drift_conf, dict):
            drift_conf = [drift_conf]
        
        self.drift_configs = drift_conf
        self.episode_count = 0
        self.step_counter = 0
        
        # Store original parameters from the environment
        self.original_params = {
            'gravity': self.unwrapped.gravity,
            'wind_power': self.unwrapped.wind_power,
            'turbulence_power': self.unwrapped.turbulence_power,
        }
        
        # Create drift generators (use episode_count as "timestep")
        self.drift_generators: Dict[str, DriftGenerator] = {}
        
        for conf in self.drift_configs:
            param = conf.get('parameter', 'gravity')
            if param not in self.VALID_PARAMS:
                raise ValueError(f"Invalid parameter '{param}'. Must be one of {self.VALID_PARAMS}")
            
            # Set base value from environment if not specified
            if 'base_value' not in conf:
                conf['base_value'] = self.original_params[param]
            
            self.drift_generators[param] = create_drift_generator(conf, seed=seed)
        
        # Store env spec for recreating environment
        self.env_id = env.spec.id if hasattr(env, 'spec') and env.spec else 'LunarLander-v3'
        
        # Current parameter values
        self.current_params = self.original_params.copy()
        
        print(f">>> [Wrapper] Initialized Non-Stationary LunarLander (PER-EPISODE drift)")
        print(f"    - Environment: {self.env_id}")
        for param, gen in self.drift_generators.items():
            print(f"    - {param}: {gen.config.drift_type} (magnitude={gen.config.magnitude}, period={gen.config.period} episodes)")
    
    def reset(self, **kwargs):
        """
        Reset environment with updated parameters based on episode count.
        
        This is where drift happens - we compute new parameter values
        and recreate the environment with those values.
        """
        # Compute new parameter values based on current episode
        for param, gen in self.drift_generators.items():
            # Use episode_count as "timestep" for drift generator
            new_value = gen.get_value(self.episode_count)
            self.current_params[param] = new_value
        
        # Recreate environment with new parameters
        # Get env_kwargs from current env
        env_kwargs = {
            'render_mode': self.unwrapped.render_mode,
            'continuous': self.unwrapped.continuous,
            'gravity': self.current_params['gravity'],
            'enable_wind': self.unwrapped.enable_wind,
            'wind_power': self.current_params['wind_power'],
            'turbulence_power': self.current_params['turbulence_power'],
        }
        
        # Destroy current env and create new one
        self.env.close()
        self.env = gym.make(self.env_id, **env_kwargs)
        
        # Reset counters
        self.step_counter = 0
        self.episode_count += 1
        
        # Reset the new environment
        obs, info = self.env.reset(**kwargs)
        
        # Add drift info to info dict
        info['drift/episode'] = self.episode_count
        info['drift/params'] = {}
        
        for param in self.VALID_PARAMS:
            current_val = self.current_params[param]
            base_val = self.original_params[param]
            info['drift/params'][param] = {
                'current': current_val,
                'base': base_val,
                'delta': current_val - base_val,
            }
        
        return obs, info
    
    def step(self, action):
        """Execute one step."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add drift info to every step
        info['drift/episode'] = self.episode_count
        info['drift/step'] = self.step_counter
        info['drift/params'] = {}
        
        for param in self.VALID_PARAMS:
            current_val = self.current_params[param]
            base_val = self.original_params[param]
            info['drift/params'][param] = {
                'current': current_val,
                'base': base_val,
                'delta': current_val - base_val,
            }
        
        self.step_counter += 1
        
        return obs, reward, terminated, truncated, info
    
    def get_drift_info(self) -> Dict[str, Any]:
        """Get current drift information."""
        info = {
            'episode_count': self.episode_count,
            'step_count': self.step_counter,
            'parameters': {}
        }
        
        for param, gen in self.drift_generators.items():
            info['parameters'][param] = {
                'config': gen.get_drift_info(self.episode_count),
                'current': self.current_params[param],
                'base': self.original_params[param],
            }
        
        return info


class NonStationaryWrapper(gym.Wrapper):
    """
    Generic non-stationary wrapper for any Gym environment.
    
    This wrapper can drift arbitrary environment attributes.
    Use with caution - not all attributes can be safely modified.
    """
    
    def __init__(
        self,
        env: gym.Env,
        drift_conf: Union[Dict[str, Any], List[Dict[str, Any]]],
        seed: Optional[int] = None
    ):
        super().__init__(env)
        
        if isinstance(drift_conf, dict):
            drift_conf = [drift_conf]
        
        self.drift_configs = drift_conf
        self.step_counter = 0
        self.total_steps = 0
        
        # Store original values
        self.original_params = {}
        self.drift_generators: Dict[str, DriftGenerator] = {}
        
        for conf in self.drift_configs:
            param = conf.get('parameter')
            if param is None:
                raise ValueError("Each drift config must specify a 'parameter'")
            
            # Try to get original value from unwrapped env
            try:
                original_val = getattr(self.unwrapped, param)
                self.original_params[param] = original_val
                
                if 'base_value' not in conf:
                    conf['base_value'] = float(original_val)
                    
            except AttributeError:
                raise ValueError(f"Environment does not have attribute '{param}'")
            
            self.drift_generators[param] = create_drift_generator(conf, seed=seed)
        
        print(f">>> [Wrapper] Initialized Non-Stationary Environment")
        for param, gen in self.drift_generators.items():
            print(f"    - {param}: {gen.config.drift_type}")

    def step(self, action):
        self._update_physics()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        info['drift/step'] = self.total_steps
        info['drift/params'] = {}
        
        for param, gen in self.drift_generators.items():
            current_val = getattr(self.unwrapped, param)
            info['drift/params'][param] = {
                'current': current_val,
                'base': self.original_params[param],
            }
        
        self.step_counter += 1
        self.total_steps += 1
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_counter = 0
        self._update_physics()
        return self.env.reset(**kwargs)

    def _update_physics(self):
        for param, gen in self.drift_generators.items():
            new_value = gen.get_value(self.total_steps)
            setattr(self.unwrapped, param, new_value)