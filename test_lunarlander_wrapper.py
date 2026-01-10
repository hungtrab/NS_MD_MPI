#!/usr/bin/env python
"""
Verification test for LunarLander wrapper.
Tests that wrapper can be created and parameters drift correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.envs.wrappers import NonStationaryLunarLanderWrapper
import gymnasium as gym

print("=" * 60)
print("LunarLander Wrapper Verification Test")
print("=" * 60)

# Test gravity jump drift
env = gym.make('LunarLander-v3')
drift_conf = {
    'parameter': 'gravity',
    'drift_type': 'jump',
    'magnitude': 8.0,    # Base=-10.0, jump to -2.0 (low gravity like Moon)
    'period': 2  # Jump every 2 episodes for testing
}

print("\n1. Creating wrapped environment...")
wrapped = NonStationaryLunarLanderWrapper(env, drift_conf, seed=42)

print("\n2. Testing drift across episodes...")
gravities = []

for episode in range(5):
    obs, info = wrapped.reset()
    g = wrapped.current_params['gravity']
    gravities.append(g)
    print(f"   Episode {episode}: gravity = {g:.2f}")
    
    # Run a few steps
    for _ in range(10):
        action = wrapped.action_space.sample()
        obs, reward, term, trunc, info = wrapped.step(action)
        if term or trunc:
            break

print("\n3. Verifying drift pattern...")
# Episode 0, 2, 4 should have base gravity (-10.0)
# Episode 1, 3 should have jumped gravity (-2.0 = -10.0 + 8.0)
expected_base = -10.0
expected_jump = -2.0

assert abs(gravities[0] - expected_base) < 0.1, f"Episode 0 wrong: {gravities[0]}"
assert abs(gravities[1] - expected_jump) < 0.1, f"Episode 1 wrong: {gravities[1]}"
assert abs(gravities[2] - expected_base) < 0.1, f"Episode 2 wrong: {gravities[2]}"
assert abs(gravities[3] - expected_jump) < 0.1, f"Episode 3 wrong: {gravities[3]}"

print("   ✅ Drift pattern correct!")

print("\n4. Testing drift info...")
drift_info = wrapped.get_drift_info()
print(f"   Episode count: {drift_info['episode_count']}")
print(f"   Parameters tracked: {list(drift_info['parameters'].keys())}")

print("\n=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
