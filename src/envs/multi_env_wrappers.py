"""
Non-Stationary Wrappers for Multiple Gymnasium Environments.

Implements wrappers for:
- CartPole: Gravity, mass, length modifications
- MountainCar: Gravity, force modifications
- FrozenLake: Slippery probability, hole locations
- MiniGrid: Goal position, obstacle density
- HalfCheetah (MuJoCo): Friction, damping, mass

Each wrapper introduces dynamics drift following the patterns defined in drift_generator.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, List, Union, Tuple
import copy

from .drift_generator import DriftGenerator, DriftConfig, create_drift_generator


# =============================================================================
# CARTPOLE NON-STATIONARY WRAPPER
# =============================================================================

class NonStationaryCartPoleWrapper(gym.Wrapper):
    """
    Non-Stationary CartPole environment.
    
    Driftable Parameters:
        - gravity: Gravitational acceleration (default=9.8). Higher = harder to balance
        - masscart: Cart mass (default=1.0). Higher = more inertia
        - masspole: Pole mass (default=0.1). Higher = harder to balance
        - length: Pole half-length (default=0.5). Longer = easier
    
    The pole must be balanced on the cart. Changing physics parameters
    affects the difficulty of the balancing task.
    
    Usage:
        drift_conf = {
            'parameter': 'gravity',
            'drift_type': 'sine',
            'magnitude': 5.0,
            'period': 10000,
        }
        env = NonStationaryCartPoleWrapper(gym.make('CartPole-v1'), drift_conf)
    """
    
    VALID_PARAMS = ['gravity', 'masscart', 'masspole', 'length']
    
    # Default values for CartPole-v1
    DEFAULT_VALUES = {
        'gravity': 9.8,
        'masscart': 1.0,
        'masspole': 0.1,
        'length': 0.5,
    }
    
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
        
        # Store original parameters
        self.original_params = {
            'gravity': self.unwrapped.gravity,
            'masscart': self.unwrapped.masscart,
            'masspole': self.unwrapped.masspole,
            'length': self.unwrapped.length
        }
        
        self.drift_generators: Dict[str, DriftGenerator] = {}
        
        for conf in self.drift_configs:
            param = conf.get('parameter', 'gravity')
            if param not in self.VALID_PARAMS:
                raise ValueError(f"Invalid parameter '{param}'. Must be one of {self.VALID_PARAMS}")
            
            if 'base_value' not in conf:
                conf['base_value'] = self.original_params[param]
            
            self.drift_generators[param] = create_drift_generator(conf, seed=seed)
        
        self.target_param = list(self.drift_generators.keys())[0]
        self._current_drift_info = {}
        
        print(f">>> [Wrapper] Initialized Non-Stationary CartPole")
        for param, gen in self.drift_generators.items():
            print(f"    - {param}: {gen.config.drift_type} (magnitude={gen.config.magnitude}, period={gen.config.period})")

    def step(self, action):
        self._update_physics()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
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
        
        if len(self.drift_generators) == 1:
            param = list(self.drift_generators.keys())[0]
            info['drift/current_value'] = getattr(self.unwrapped, param)
            info['drift/parameter'] = param
        
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
            self._set_env_param(param, new_value)

    def _set_env_param(self, param_name: str, value: float):
        """Update parameter and recalculate dependent physics."""
        setattr(self.unwrapped, param_name, value)
        
        # Recalculate derived quantities
        if param_name in ['masscart', 'masspole']:
            self.unwrapped.total_mass = self.unwrapped.masspole + self.unwrapped.masscart
        
        if param_name in ['length', 'masspole']:
            self.unwrapped.polemass_length = self.unwrapped.masspole * self.unwrapped.length
    
    def get_drift_info(self) -> Dict[str, Any]:
        info = {
            'total_steps': self.total_steps,
            'episode_steps': self.step_counter,
            'parameters': {}
        }
        for param, gen in self.drift_generators.items():
            info['parameters'][param] = gen.get_drift_info(self.total_steps)
        return info
    
    def set_total_steps(self, steps: int):
        self.total_steps = steps
        self._update_physics()


# =============================================================================
# MOUNTAINCAR NON-STATIONARY WRAPPER
# =============================================================================

class NonStationaryMountainCarWrapper(gym.Wrapper):
    """
    Non-Stationary MountainCar environment.
    
    Driftable Parameters:
        - gravity: Gravitational force (default=0.0025)
        - force: Engine force (default=0.001)
        - goal_position: Goal x-position (default=0.5)
        - goal_velocity: Required velocity at goal (default=0)
    
    The car must reach the goal on a hill. Changing gravity or force
    affects the difficulty of building momentum.
    
    Usage:
        drift_conf = {
            'parameter': 'gravity',
            'drift_type': 'sine',
            'magnitude': 0.001,
            'period': 10000,
        }
        env = NonStationaryMountainCarWrapper(gym.make('MountainCar-v0'), drift_conf)
    """
    
    VALID_PARAMS = ['gravity', 'force', 'goal_position', 'goal_velocity']
    
    # Default values for MountainCar-v0
    DEFAULT_VALUES = {
        'gravity': 0.0025,
        'force': 0.001,
        'goal_position': 0.5,
        'goal_velocity': 0.0,
    }
    
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
        
        # Store original parameters
        self.original_params = {}
        self.drift_generators: Dict[str, DriftGenerator] = {}
        
        for conf in self.drift_configs:
            param = conf.get('parameter', 'gravity')
            if param not in self.VALID_PARAMS:
                raise ValueError(f"Invalid parameter '{param}'. Must be one of {self.VALID_PARAMS}")
            
            # Get original value
            if hasattr(self.unwrapped, param):
                self.original_params[param] = getattr(self.unwrapped, param)
            else:
                self.original_params[param] = self.DEFAULT_VALUES.get(param, 0.0)
            
            if 'base_value' not in conf:
                conf['base_value'] = self.original_params[param]
            
            self.drift_generators[param] = create_drift_generator(conf, seed=seed)
        
        self.target_param = list(self.drift_generators.keys())[0]
        
        print(f">>> [Wrapper] Initialized Non-Stationary MountainCar")
        for param, gen in self.drift_generators.items():
            print(f"    - {param}: {gen.config.drift_type} (magnitude={gen.config.magnitude})")

    def step(self, action):
        self._update_physics()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        info['drift/step'] = self.total_steps
        info['drift/params'] = {}
        
        for param, gen in self.drift_generators.items():
            current_val = getattr(self.unwrapped, param, self.original_params[param])
            info['drift/params'][param] = {
                'current': current_val,
                'base': self.original_params[param],
                'delta': current_val - self.original_params[param],
            }
        
        if len(self.drift_generators) == 1:
            param = self.target_param
            info['drift/current_value'] = getattr(self.unwrapped, param, self.original_params[param])
            info['drift/parameter'] = param
        
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
            if hasattr(self.unwrapped, param):
                setattr(self.unwrapped, param, new_value)

    def get_drift_info(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'episode_steps': self.step_counter,
            'parameters': {
                param: gen.get_drift_info(self.total_steps)
                for param, gen in self.drift_generators.items()
            }
        }

    def set_total_steps(self, steps: int):
        self.total_steps = steps
        self._update_physics()


# =============================================================================
# FROZENLAKE NON-STATIONARY WRAPPER
# =============================================================================

class NonStationaryFrozenLakeWrapper(gym.Wrapper):
    """
    Non-Stationary FrozenLake environment.
    
    Driftable Parameters:
        - slip_prob: Probability of slipping (default depends on is_slippery)
        - reward_scale: Multiplier for rewards
    
    Note: FrozenLake uses a pre-computed transition matrix, so we modify
    the effective slip probability by intercepting actions.
    
    Usage:
        drift_conf = {
            'parameter': 'slip_prob',
            'drift_type': 'linear',
            'magnitude': 0.3,
            'period': 5000,
        }
        env = NonStationaryFrozenLakeWrapper(gym.make('FrozenLake-v1'), drift_conf)
    """
    
    VALID_PARAMS = ['slip_prob', 'reward_scale']
    
    DEFAULT_VALUES = {
        'slip_prob': 0.0,  # Will be set based on is_slippery
        'reward_scale': 1.0,
    }
    
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
        self.rng = np.random.default_rng(seed)
        
        # Determine base slip probability
        # In FrozenLake, is_slippery=True means 1/3 chance of intended direction
        self.base_slip_prob = 2/3 if getattr(self.unwrapped, 'is_slippery', True) else 0.0
        
        self.original_params = {
            'slip_prob': self.base_slip_prob,
            'reward_scale': 1.0,
        }
        
        self.drift_generators: Dict[str, DriftGenerator] = {}
        self.current_slip_prob = self.base_slip_prob
        self.current_reward_scale = 1.0
        
        for conf in self.drift_configs:
            param = conf.get('parameter', 'slip_prob')
            if param not in self.VALID_PARAMS:
                raise ValueError(f"Invalid parameter '{param}'. Must be one of {self.VALID_PARAMS}")
            
            if 'base_value' not in conf:
                conf['base_value'] = self.original_params[param]
            
            self.drift_generators[param] = create_drift_generator(conf, seed=seed)
        
        self.target_param = list(self.drift_generators.keys())[0]
        
        print(f">>> [Wrapper] Initialized Non-Stationary FrozenLake")
        for param, gen in self.drift_generators.items():
            print(f"    - {param}: {gen.config.drift_type} (base={self.original_params[param]:.2f})")

    def step(self, action):
        # Update drift values
        self._update_params()
        
        # Apply slip probability by potentially modifying action
        if 'slip_prob' in self.drift_generators:
            action = self._apply_slip(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward scaling
        if 'reward_scale' in self.drift_generators:
            reward = reward * self.current_reward_scale
        
        # Log drift info
        info['drift/step'] = self.total_steps
        info['drift/params'] = {}
        
        if 'slip_prob' in self.drift_generators:
            info['drift/params']['slip_prob'] = {
                'current': self.current_slip_prob,
                'base': self.original_params['slip_prob'],
            }
            info['drift/current_value'] = self.current_slip_prob
        
        if 'reward_scale' in self.drift_generators:
            info['drift/params']['reward_scale'] = {
                'current': self.current_reward_scale,
                'base': 1.0,
            }
        
        self.step_counter += 1
        self.total_steps += 1
        
        return obs, reward, terminated, truncated, info

    def _apply_slip(self, action: int) -> int:
        """Apply slip probability to action."""
        if self.rng.random() < self.current_slip_prob:
            # Slip: choose random perpendicular direction
            if action in [0, 2]:  # Left/Right -> Up/Down
                return self.rng.choice([1, 3])
            else:  # Up/Down -> Left/Right
                return self.rng.choice([0, 2])
        return action

    def reset(self, **kwargs):
        self.step_counter = 0
        self._update_params()
        return self.env.reset(**kwargs)

    def _update_params(self):
        for param, gen in self.drift_generators.items():
            new_value = gen.get_value(self.total_steps)
            if param == 'slip_prob':
                self.current_slip_prob = np.clip(new_value, 0.0, 1.0)
            elif param == 'reward_scale':
                self.current_reward_scale = max(new_value, 0.0)

    def get_drift_info(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'parameters': {
                param: gen.get_drift_info(self.total_steps)
                for param, gen in self.drift_generators.items()
            }
        }

    def set_total_steps(self, steps: int):
        self.total_steps = steps
        self._update_params()


# =============================================================================
# MINIGRID NON-STATIONARY WRAPPER
# =============================================================================

class NonStationaryMiniGridWrapper(gym.Wrapper):
    """
    Non-Stationary MiniGrid environment.
    
    Driftable Parameters:
        - reward_scale: Multiplier for step rewards
        - max_steps: Maximum steps before truncation
        - agent_view_size: Agent's field of view (if supported)
    
    Note: MiniGrid environments have procedural generation, so we focus on
    reward and step limit modifications rather than layout changes.
    
    Usage:
        drift_conf = {
            'parameter': 'reward_scale',
            'drift_type': 'sine',
            'magnitude': 0.5,
            'period': 10000,
        }
        env = NonStationaryMiniGridWrapper(gym.make('MiniGrid-Empty-8x8-v0'), drift_conf)
    """
    
    VALID_PARAMS = ['reward_scale', 'max_steps', 'step_penalty']
    
    DEFAULT_VALUES = {
        'reward_scale': 1.0,
        'max_steps': 100,
        'step_penalty': 0.0,
    }
    
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
        
        # Get original max_steps from env
        original_max_steps = getattr(self.unwrapped, 'max_steps', 100)
        
        self.original_params = {
            'reward_scale': 1.0,
            'max_steps': original_max_steps,
            'step_penalty': 0.0,
        }
        
        self.current_params = self.original_params.copy()
        self.drift_generators: Dict[str, DriftGenerator] = {}
        
        for conf in self.drift_configs:
            param = conf.get('parameter', 'reward_scale')
            if param not in self.VALID_PARAMS:
                raise ValueError(f"Invalid parameter '{param}'. Must be one of {self.VALID_PARAMS}")
            
            if 'base_value' not in conf:
                conf['base_value'] = self.original_params[param]
            
            self.drift_generators[param] = create_drift_generator(conf, seed=seed)
        
        self.target_param = list(self.drift_generators.keys())[0]
        
        print(f">>> [Wrapper] Initialized Non-Stationary MiniGrid")
        for param, gen in self.drift_generators.items():
            print(f"    - {param}: {gen.config.drift_type}")

    def step(self, action):
        self._update_params()
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward modifications
        reward = reward * self.current_params['reward_scale']
        reward -= self.current_params['step_penalty']
        
        # Check custom max_steps
        if 'max_steps' in self.drift_generators:
            if self.step_counter >= int(self.current_params['max_steps']):
                truncated = True
        
        # Log drift info
        info['drift/step'] = self.total_steps
        info['drift/params'] = {
            param: {'current': self.current_params[param], 'base': self.original_params[param]}
            for param in self.drift_generators.keys()
        }
        
        if len(self.drift_generators) == 1:
            info['drift/current_value'] = self.current_params[self.target_param]
            info['drift/parameter'] = self.target_param
        
        self.step_counter += 1
        self.total_steps += 1
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_counter = 0
        self._update_params()
        return self.env.reset(**kwargs)

    def _update_params(self):
        for param, gen in self.drift_generators.items():
            self.current_params[param] = gen.get_value(self.total_steps)

    def get_drift_info(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'parameters': {
                param: gen.get_drift_info(self.total_steps)
                for param, gen in self.drift_generators.items()
            }
        }

    def set_total_steps(self, steps: int):
        self.total_steps = steps
        self._update_params()


# =============================================================================
# HALFCHEETAH (MUJOCO) NON-STATIONARY WRAPPER
# =============================================================================

class NonStationaryHalfCheetahWrapper(gym.Wrapper):
    """
    Non-Stationary HalfCheetah (MuJoCo) environment.
    
    Driftable Parameters:
        - friction: Ground friction coefficient
        - damping: Joint damping (affects all joints uniformly)
        - mass_scale: Scales the mass of all bodies
        - gravity: Gravitational acceleration (z-component)
    
    Requires: pip install mujoco gymnasium[mujoco]
    
    Usage:
        drift_conf = {
            'parameter': 'friction',
            'drift_type': 'jump',
            'magnitude': 0.5,
            'period': 50000,
        }
        env = NonStationaryHalfCheetahWrapper(gym.make('HalfCheetah-v4'), drift_conf)
    """
    
    VALID_PARAMS = ['friction', 'damping', 'mass_scale', 'gravity']
    
    DEFAULT_VALUES = {
        'friction': 0.4,      # Default MuJoCo floor friction
        'damping': 1.0,       # Multiplier for joint damping
        'mass_scale': 1.0,    # Multiplier for body masses
        'gravity': -9.81,     # Gravity (negative = downward)
    }
    
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
        
        # Store original MuJoCo model parameters
        self.original_params = {}
        self._store_original_params()
        
        self.drift_generators: Dict[str, DriftGenerator] = {}
        
        for conf in self.drift_configs:
            param = conf.get('parameter', 'friction')
            if param not in self.VALID_PARAMS:
                raise ValueError(f"Invalid parameter '{param}'. Must be one of {self.VALID_PARAMS}")
            
            if 'base_value' not in conf:
                conf['base_value'] = self.original_params.get(param, self.DEFAULT_VALUES[param])
            
            self.drift_generators[param] = create_drift_generator(conf, seed=seed)
        
        self.target_param = list(self.drift_generators.keys())[0]
        
        print(f">>> [Wrapper] Initialized Non-Stationary HalfCheetah")
        for param, gen in self.drift_generators.items():
            print(f"    - {param}: {gen.config.drift_type} (base={gen.config.base_value:.3f})")

    def _store_original_params(self):
        """Store original MuJoCo model parameters."""
        try:
            model = self.unwrapped.model
            
            # Friction (floor geom, usually index 0)
            if hasattr(model, 'geom_friction'):
                self.original_params['friction'] = float(model.geom_friction[0, 0])
            else:
                self.original_params['friction'] = self.DEFAULT_VALUES['friction']
            
            # Damping (store original values for all joints)
            if hasattr(model, 'dof_damping'):
                self.original_damping = model.dof_damping.copy()
                self.original_params['damping'] = 1.0
            
            # Mass (store original values for all bodies)
            if hasattr(model, 'body_mass'):
                self.original_mass = model.body_mass.copy()
                self.original_params['mass_scale'] = 1.0
            
            # Gravity
            if hasattr(model, 'opt') and hasattr(model.opt, 'gravity'):
                self.original_params['gravity'] = float(model.opt.gravity[2])
            else:
                self.original_params['gravity'] = self.DEFAULT_VALUES['gravity']
                
        except Exception as e:
            print(f"Warning: Could not access MuJoCo model parameters: {e}")
            self.original_params = self.DEFAULT_VALUES.copy()

    def step(self, action):
        self._update_physics()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        info['drift/step'] = self.total_steps
        info['drift/params'] = {}
        
        for param, gen in self.drift_generators.items():
            current_val = self._get_current_param(param)
            info['drift/params'][param] = {
                'current': current_val,
                'base': self.original_params.get(param, self.DEFAULT_VALUES[param]),
            }
        
        if len(self.drift_generators) == 1:
            info['drift/current_value'] = self._get_current_param(self.target_param)
            info['drift/parameter'] = self.target_param
        
        self.step_counter += 1
        self.total_steps += 1
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_counter = 0
        result = self.env.reset(**kwargs)
        self._update_physics()
        return result

    def _update_physics(self):
        """Update MuJoCo model parameters based on drift."""
        try:
            model = self.unwrapped.model
            
            for param, gen in self.drift_generators.items():
                new_value = gen.get_value(self.total_steps)
                
                if param == 'friction':
                    # Update floor friction (geom 0)
                    if hasattr(model, 'geom_friction'):
                        model.geom_friction[0, 0] = new_value
                        model.geom_friction[0, 1] = new_value * 0.005  # torsional
                        model.geom_friction[0, 2] = new_value * 0.0001  # rolling
                
                elif param == 'damping':
                    # Scale all joint damping
                    if hasattr(model, 'dof_damping') and hasattr(self, 'original_damping'):
                        model.dof_damping[:] = self.original_damping * new_value
                
                elif param == 'mass_scale':
                    # Scale all body masses
                    if hasattr(model, 'body_mass') and hasattr(self, 'original_mass'):
                        model.body_mass[:] = self.original_mass * new_value
                
                elif param == 'gravity':
                    # Update gravity
                    if hasattr(model, 'opt') and hasattr(model.opt, 'gravity'):
                        model.opt.gravity[2] = new_value
                        
        except Exception as e:
            pass  # Silently fail if MuJoCo access fails

    def _get_current_param(self, param: str) -> float:
        """Get current value of a parameter from MuJoCo model."""
        try:
            model = self.unwrapped.model
            
            if param == 'friction':
                return float(model.geom_friction[0, 0])
            elif param == 'damping':
                if hasattr(self, 'original_damping'):
                    return float(model.dof_damping[0] / self.original_damping[0])
                return 1.0
            elif param == 'mass_scale':
                if hasattr(self, 'original_mass'):
                    return float(model.body_mass[1] / self.original_mass[1])
                return 1.0
            elif param == 'gravity':
                return float(model.opt.gravity[2])
        except:
            pass
        
        return self.DEFAULT_VALUES.get(param, 0.0)

    def get_drift_info(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'episode_steps': self.step_counter,
            'parameters': {
                param: gen.get_drift_info(self.total_steps)
                for param, gen in self.drift_generators.items()
            }
        }

    def set_total_steps(self, steps: int):
        self.total_steps = steps
        self._update_physics()


# =============================================================================
# REGISTRY AND FACTORY
# =============================================================================

# Environment to wrapper mapping
WRAPPER_REGISTRY = {
    'CartPole-v0': 'NonStationaryCartPoleWrapper',
    'CartPole-v1': 'NonStationaryCartPoleWrapper',
    'MountainCar-v0': 'NonStationaryMountainCarWrapper',
    'MountainCarContinuous-v0': 'NonStationaryMountainCarWrapper',
    'FrozenLake-v1': 'NonStationaryFrozenLakeWrapper',
    'FrozenLake8x8-v1': 'NonStationaryFrozenLakeWrapper',
    'HalfCheetah-v4': 'NonStationaryHalfCheetahWrapper',
    'HalfCheetah-v5': 'NonStationaryHalfCheetahWrapper',
}

# Default driftable parameters per environment
DEFAULT_DRIFT_PARAMS = {
    'CartPole': ['gravity', 'masscart', 'masspole', 'length'],
    'MountainCar': ['gravity', 'force', 'goal_position'],
    'FrozenLake': ['slip_prob', 'reward_scale'],
    'MiniGrid': ['reward_scale', 'max_steps', 'step_penalty'],
    'HalfCheetah': ['friction', 'damping', 'mass_scale', 'gravity'],
}


def get_wrapper_for_env(env_id: str):
    """Get the appropriate wrapper class for an environment ID."""
    # Build complete registry with all wrappers
    all_wrappers = {
        'NonStationaryCartPoleWrapper': NonStationaryCartPoleWrapper,
        'NonStationaryMountainCarWrapper': NonStationaryMountainCarWrapper,
        'NonStationaryFrozenLakeWrapper': NonStationaryFrozenLakeWrapper,
        'NonStationaryHalfCheetahWrapper': NonStationaryHalfCheetahWrapper,
        'NonStationaryMiniGridWrapper': NonStationaryMiniGridWrapper,
    }
    
    # Direct match
    if env_id in WRAPPER_REGISTRY:
        wrapper_name = WRAPPER_REGISTRY[env_id]
        return all_wrappers[wrapper_name]
    
    # Partial match (e.g., 'MiniGrid-Empty-8x8-v0' -> MiniGrid)
    for key_prefix in ['CartPole', 'MountainCar', 'FrozenLake', 'HalfCheetah', 'MiniGrid']:
        if key_prefix in env_id:
            if key_prefix == 'CartPole':
                return NonStationaryCartPoleWrapper
            elif key_prefix == 'MountainCar':
                return NonStationaryMountainCarWrapper
            elif key_prefix == 'FrozenLake':
                return NonStationaryFrozenLakeWrapper
            elif key_prefix == 'HalfCheetah':
                return NonStationaryHalfCheetahWrapper
            elif key_prefix == 'MiniGrid':
                return NonStationaryMiniGridWrapper
    
    raise ValueError(f"No wrapper available for environment: {env_id}")


def make_nonstationary_env(
    env_id: str,
    drift_conf: Union[Dict[str, Any], List[Dict[str, Any]]],
    seed: Optional[int] = None,
    **env_kwargs
) -> gym.Env:
    """
    Factory function to create a non-stationary environment.
    
    Args:
        env_id: Gymnasium environment ID
        drift_conf: Drift configuration
        seed: Random seed
        **env_kwargs: Additional arguments for gym.make()
        
    Returns:
        Wrapped non-stationary environment
        
    Example:
        env = make_nonstationary_env(
            'CartPole-v1',
            {'parameter': 'gravity', 'drift_type': 'sine', 'magnitude': 5.0, 'period': 10000}
        )
    """
    base_env = gym.make(env_id, **env_kwargs)
    wrapper_class = get_wrapper_for_env(env_id)
    return wrapper_class(base_env, drift_conf, seed=seed)
