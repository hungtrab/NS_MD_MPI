"""
Non-Stationary Environment Module.

Provides wrappers and utilities for creating non-stationary MDPs.

Supported Environments:
- CartPole-v1: gravity, masscart, masspole, length
- MountainCar-v0: gravity, force, goal_position
- FrozenLake-v1: slip_prob, reward_scale
- MiniGrid-*: reward_scale, max_steps, step_penalty
- HalfCheetah-v4: friction, damping, mass_scale, gravity
"""

from .drift_generator import (
    DriftGenerator,
    DriftConfig,
    create_drift_generator,
)

from .wrappers import (
    NonStationaryWrapper,
)

from .multi_env_wrappers import (
    NonStationaryCartPoleWrapper,
    NonStationaryMountainCarWrapper,
    NonStationaryFrozenLakeWrapper,
    NonStationaryMiniGridWrapper,
    NonStationaryHalfCheetahWrapper,
    make_nonstationary_env,
    get_wrapper_for_env,
    WRAPPER_REGISTRY,
    DEFAULT_DRIFT_PARAMS,
)

__all__ = [
    # Drift Generator
    'DriftGenerator',
    'DriftConfig',
    'create_drift_generator',
    # CartPole
    'NonStationaryCartPoleWrapper',
    'NonStationaryWrapper',
    # Multi-environment
    'NonStationaryMountainCarWrapper',
    'NonStationaryFrozenLakeWrapper',
    'NonStationaryMiniGridWrapper',
    'NonStationaryHalfCheetahWrapper',
    # Factory
    'make_nonstationary_env',
    'get_wrapper_for_env',
    'WRAPPER_REGISTRY',
    'DEFAULT_DRIFT_PARAMS',
]