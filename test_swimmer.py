#!/usr/bin/env python3
"""
Quick test to debug Swimmer wrapper issue
"""
import sys
sys.path.insert(0, '.')

import gymnasium as gym
from src.envs.multi_env_wrappers import make_nonstationary_env

print("=" * 60)
print("Testing Swimmer-v4 Wrapper")
print("=" * 60)

drift_config = {
    'parameter': 'friction',
    'drift_type': 'sine',
    'magnitude': 0.25,
    'period': 10000,
    'base_value': 1.0,
    'bounds': [0.5, 1.5]
}

try:
    print("\n1. Creating base Swimmer-v4...")
    base_env = gym.make('Swimmer-v4')
    print(f"✓ Base env created: {base_env}")
    print(f"  Obs space: {base_env.observation_space}")
    print(f"  Action space: {base_env.action_space}")
    
    # Check model structure
    print("\n2. Checking MuJoCo model structure...")
    print(f"  njnt (joints): {base_env.unwrapped.model.njnt}")
    print(f"  nbody (bodies): {base_env.unwrapped.model.nbody}")
    print(f"  ngeom (geoms): {base_env.unwrapped.model.ngeom}")
    
    base_env.close()
    
    print("\n3. Creating wrapped Swimmer with drift...")
    env = make_nonstationary_env('Swimmer-v4', drift_config, seed=42)
    print(f"✓ Wrapped env created: {env}")
    
    print("\n4. Testing reset...")
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Obs shape: {obs.shape}")
    
    print("\n5. Testing step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful")
    print(f"  Reward: {reward}")
    
    env.close()
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
