"""
Test Script for Non-Stationary Environment Wrappers.

Takes random actions and renders the environment to screen,
displaying drift information in real-time.

Usage:
    python scripts/test_env.py                                    # Default: CartPole
    python scripts/test_env.py --env MountainCar-v0               # MountainCar
    python scripts/test_env.py --env CartPole-v1 --drift sine     # With sine drift
    python scripts/test_env.py --episodes 3 --max-steps 500       # Custom episodes
"""

import sys
import os
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.envs import make_nonstationary_env, WRAPPER_REGISTRY


def test_environment(
    env_id: str = "CartPole-v1",
    drift_type: str = "sine",
    parameter: str = None,
    magnitude: float = None,
    period: int = 500,
    n_episodes: int = 3,
    max_steps: int = 500,
    delay: float = 0.02,
):
    """
    Test a non-stationary environment with random actions.
    
    Args:
        env_id: Gymnasium environment ID
        drift_type: Type of drift (static, jump, linear, sine, random_walk)
        parameter: Parameter to drift (auto-detected if None)
        magnitude: Drift magnitude (auto-set if None)
        period: Drift period in steps
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        delay: Delay between frames (seconds)
    """
    # Auto-detect parameter and magnitude based on environment
    if parameter is None:
        if "CartPole" in env_id:
            parameter = "gravity"
            magnitude = magnitude or 5.0
        elif "MountainCar" in env_id:
            parameter = "gravity"
            magnitude = magnitude or 0.001
        elif "FrozenLake" in env_id:
            parameter = "slip_prob"
            magnitude = magnitude or 0.3
        elif "HalfCheetah" in env_id:
            parameter = "friction"
            magnitude = magnitude or 0.5
        else:
            parameter = "gravity"
            magnitude = magnitude or 1.0
    
    if magnitude is None:
        magnitude = 1.0
    
    # Build drift configuration
    drift_conf = {
        'parameter': parameter,
        'drift_type': drift_type,
        'magnitude': magnitude,
        'period': period,
        'sigma': magnitude * 0.1,  # For random walk
        'bounds': [0.0, magnitude * 10],
    }
    
    print("=" * 60)
    print("  Non-Stationary Environment Test")
    print("=" * 60)
    print(f"  Environment:  {env_id}")
    print(f"  Parameter:    {parameter}")
    print(f"  Drift Type:   {drift_type}")
    print(f"  Magnitude:    {magnitude}")
    print(f"  Period:       {period} steps")
    print(f"  Episodes:     {n_episodes}")
    print(f"  Max Steps:    {max_steps}")
    print("=" * 60)
    print()
    
    # Create environment with human rendering
    try:
        env = make_nonstationary_env(env_id, drift_conf, render_mode="human")
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nSupported environments: {list(WRAPPER_REGISTRY.keys())}")
        return
    
    total_rewards = []
    
    for episode in range(n_episodes):
        print(f"\n>>> Episode {episode + 1}/{n_episodes}")
        print("-" * 40)
        
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Take random action
            action = env.action_space.sample()
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Print drift info every 50 steps
            if step % 50 == 0:
                drift_info = info.get('drift/params', {})
                if drift_info:
                    for param, data in drift_info.items():
                        current = data.get('current', 0)
                        base = data.get('base', 0)
                        delta = data.get('delta', 0)
                        print(f"  Step {step:4d}: {param} = {current:.4f} (base={base:.4f}, Δ={delta:+.4f})")
                else:
                    print(f"  Step {step:4d}: reward = {reward:.2f}")
            
            # Render delay
            time.sleep(delay)
        
        total_rewards.append(episode_reward)
        print(f"\n  Episode finished: {step} steps, reward = {episode_reward:.1f}")
        
        if done:
            print("  Reason: Terminal state reached")
        elif truncated:
            print("  Reason: Episode truncated (time limit)")
        else:
            print("  Reason: Max steps reached")
    
    env.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
    print(f"  Episodes:       {n_episodes}")
    print(f"  Avg Reward:     {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"  Min Reward:     {min(total_rewards):.2f}")
    print(f"  Max Reward:     {max(total_rewards):.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test non-stationary environment with random actions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_env.py --env CartPole-v1 --drift sine
  python scripts/test_env.py --env MountainCar-v0 --drift jump --period 20
  python scripts/test_env.py --env CartPole-v1 --param gravity --mag 8.0
  python scripts/test_env.py --episodes 5 --delay 0.05
        """
    )
    
    parser.add_argument(
        "--env", type=str, default="CartPole-v1",
        help="Gymnasium environment ID (default: CartPole-v1)"
    )
    parser.add_argument(
        "--drift", type=str, default="sine",
        choices=["static", "jump", "linear", "sine", "random_walk"],
        help="Drift type (default: sine)"
    )
    parser.add_argument(
        "--param", type=str, default=None,
        help="Parameter to drift (auto-detected if not specified)"
    )
    parser.add_argument(
        "--mag", type=float, default=None,
        help="Drift magnitude (auto-set if not specified)"
    )
    parser.add_argument(
        "--period", type=int, default=500,
        help="Drift period in steps (default: 500)"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to run (default: 3)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Maximum steps per episode (default: 500)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.02,
        help="Delay between frames in seconds (default: 0.02)"
    )
    parser.add_argument(
        "--list-envs", action="store_true",
        help="List supported environments and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_envs:
        print("\nSupported Non-Stationary Environments:")
        print("-" * 40)
        for env_pattern, wrapper_class in WRAPPER_REGISTRY.items():
            print(f"  {env_pattern}: {wrapper_class.__name__}")
        print()
        return
    
    test_environment(
        env_id=args.env,
        drift_type=args.drift,
        parameter=args.param,
        magnitude=args.mag,
        period=args.period,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
