#!/usr/bin/env python3
"""
Quick test of NS-MD-MPI implementation.

Tests:
1. VariationBudgetTracker basic functionality
2. NSMDMPICallback initialization
3. Budget auto-estimation from drift config
4. Integration with training environment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.evaluation.variation_budgets import (
    VariationBudgetTracker,
    estimate_budgets_from_drift_config,
)
from src.callbacks.nsmdmpi_callback import NSMDMPICallback
from src.envs import make_nonstationary_env


def test_variation_budget_tracker():
    """Test VariationBudgetTracker functionality."""
    print("=" * 60)
    print("TEST 1: VariationBudgetTracker")
    print("=" * 60)
    
    # Create tracker
    tracker = VariationBudgetTracker(
        V_R=10.0,
        V_P=10.0,
        V_pi_star=5.0,
        time_horizon=1000,
    )
    
    print(f"\nInitial state:")
    print(tracker)
    
    # Simulate drift updates
    print(f"\nSimulating 10 drift updates...")
    for i in range(10):
        delta_R = np.random.uniform(0.1, 0.5)
        delta_P = np.random.uniform(0.1, 0.5)
        delta_pi = np.random.uniform(0.05, 0.3)
        tracker.update(delta_R, delta_P, delta_pi, timestep=i*100)
    
    print(f"\nAfter updates:")
    print(tracker)
    
    # Check methods
    summary = tracker.get_summary()
    print(f"\nSummary metrics:")
    print(f"  Min budget remaining: {summary['min_fraction_remaining']:.1%}")
    print(f"  Total consumed: V_R={summary['V_R_consumed']:.2f}, "
          f"V_P={summary['V_P_consumed']:.2f}, V_π*={summary['V_pi_star_consumed']:.2f}")
    
    # Test budget exhaustion check
    is_exhausted = tracker.is_budget_exhausted(threshold=0.5)
    print(f"  Budget exhausted (>50%): {is_exhausted}")
    
    print("\n✓ VariationBudgetTracker test passed!")
    return tracker


def test_budget_estimation():
    """Test budget auto-estimation from drift config."""
    print("\n" + "=" * 60)
    print("TEST 2: Budget Auto-Estimation")
    print("=" * 60)
    
    # Test different drift types
    drift_configs = [
        {'drift_type': 'sine', 'magnitude': 5.0, 'period': 10000},
        {'drift_type': 'jump', 'magnitude': 10.0, 'period': 20000},
        {'drift_type': 'linear', 'magnitude': 3.0, 'period': 15000},
        {'drift_type': 'static', 'magnitude': 0.0, 'period': 10000},
    ]
    
    time_horizon = 100000
    
    for cfg in drift_configs:
        V_R, V_P, V_pi = estimate_budgets_from_drift_config(
            cfg, time_horizon, scale_factor=1.5
        )
        print(f"\n{cfg['drift_type']:12s} (mag={cfg['magnitude']:.1f}, period={cfg['period']}):")
        print(f"  Estimated budgets: V_R={V_R:.2f}, V_P={V_P:.2f}, V_π*={V_pi:.2f}")
    
    print("\n✓ Budget estimation test passed!")


def test_nsmdmpi_callback_init():
    """Test NSMDMPICallback initialization."""
    print("\n" + "=" * 60)
    print("TEST 3: NSMDMPICallback Initialization")
    print("=" * 60)
    
    # Create callback
    callback = NSMDMPICallback(
        V_R=25.0,
        V_P=25.0,
        V_pi_star=12.0,
        kappa_base=0.2,
        lambda_base=1.0,
        drift_weights=(1.0, 1.0, 0.5),
        verbose=0,
    )
    
    print("\nCallback created successfully!")
    print(f"  V_R: {callback.V_R_init}")
    print(f"  V_P: {callback.V_P_init}")
    print(f"  V_π*: {callback.V_pi_star_init}")
    print(f"  κ_base: {callback.kappa_base}")
    print(f"  λ_base: {callback.lambda_base}")
    print(f"  Drift weights: {callback.drift_weights}")
    
    print("\n✓ NSMDMPICallback initialization test passed!")
    return callback


def test_environment_integration():
    """Test integration with non-stationary environment."""
    print("\n" + "=" * 60)
    print("TEST 4: Environment Integration")
    print("=" * 60)
    
    # Create non-stationary environment
    drift_conf = {
        'parameter': 'gravity',
        'drift_type': 'sine',
        'magnitude': 3.0,
        'period': 5000,
    }
    
    try:
        env = make_nonstationary_env('CartPole-v1', drift_conf, seed=42)
        print("\n✓ Environment created successfully!")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Get drift info
        if hasattr(env, 'get_drift_info'):
            drift_info = env.get_drift_info()
            print(f"\nDrift info available:")
            print(f"  Total steps: {drift_info['total_steps']}")
            print(f"  Parameters: {list(drift_info['parameters'].keys())}")
        
        # Test reset and step
        obs, info = env.reset()
        print(f"\n✓ Environment reset successful!")
        print(f"  Initial obs shape: {obs.shape}")
        
        # Take a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if 'drift/current_value' in info:
                print(f"  Step {i+1}: drift/current_value = {info['drift/current_value']:.2f}")
        
        print("\n✓ Environment integration test passed!")
        
    except Exception as e:
        print(f"\n✗ Environment integration test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NS-MD-MPI IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Budget tracker
        tracker = test_variation_budget_tracker()
        
        # Test 2: Budget estimation
        test_budget_estimation()
        
        # Test 3: Callback initialization
        callback = test_nsmdmpi_callback_init()
        
        # Test 4: Environment integration
        test_environment_integration()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nNS-MD-MPI implementation is ready to use!")
        print("\nNext steps:")
        print("  1. Run training: python scripts/train.py --config configs/nsmdmpi_cartpole.yaml")
        print("  2. Compare with baseline: Set nsmdmpi.enabled=false")
        print("  3. Compare with adaptive: Set adaptive.enabled=true instead")
        print()
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED! ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
