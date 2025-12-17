"""
Video Recording Script for Non-Stationary MDP Experiments.

Records agent behavior in the drifting environment for visualization.
Supports multiple environments (CartPole, MountainCar, FrozenLake, MiniGrid, HalfCheetah)
Supports multiple algorithms (PPO, SAC, TRPO)
"""

import sys
import os
import yaml
import argparse
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO, SAC
from gymnasium.wrappers import RecordVideo

# TRPO from sb3-contrib (optional)
try:
    from sb3_contrib import TRPO
    TRPO_AVAILABLE = True
except ImportError:
    TRPO_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.envs import make_nonstationary_env, get_wrapper_for_env


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_drift_conf(cfg: dict) -> dict:
    """Build drift configuration from YAML config."""
    return {
        'parameter': cfg['env'].get('parameter', 'gravity'),
        'drift_type': cfg['env'].get('drift_type', 'static'),
        'magnitude': cfg['env'].get('magnitude', 0.0),
        'period': cfg['env'].get('period', 10000),
        'sigma': cfg['env'].get('sigma', 0.1),
        'bounds': cfg['env'].get('bounds', [0.0, 20.0]),
    }


def load_model(model_path: str, algo: str = "auto"):
    """
    Load a trained model with automatic algorithm detection.
    
    Args:
        model_path: Path to saved model
        algo: Algorithm name ("PPO", "SAC", "TRPO", or "auto" for auto-detect)
        
    Returns:
        Loaded model
    """
    algo = algo.upper()
    
    # Auto-detect from model path
    if algo == "AUTO":
        path_upper = model_path.upper()
        if "SAC" in path_upper:
            algo = "SAC"
        elif "TRPO" in path_upper:
            algo = "TRPO"
        else:
            algo = "PPO"  # Default
    
    print(f"Loading {algo} model from: {model_path}")
    
    if algo == "PPO":
        return PPO.load(model_path)
    elif algo == "SAC":
        return SAC.load(model_path)
    elif algo == "TRPO":
        if not TRPO_AVAILABLE:
            raise ImportError("TRPO requires sb3-contrib. Install with: pip install sb3-contrib")
        return TRPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Supported: PPO, SAC, TRPO")


def record_video(
    model_path: str,
    config_path: str,
    algo: str = "auto",
    output_dir: str = "videos/",
    n_episodes: int = 1,
    video_length: int = 1000,
):
    """
    Record video of agent in non-stationary environment.
    
    Args:
        model_path: Path to saved model
        config_path: Path to config YAML
        algo: Algorithm name ("PPO", "SAC", "TRPO", or "auto")
        output_dir: Directory to save videos
        n_episodes: Number of episodes to record
        video_length: Maximum steps per video
    """
    cfg = load_config(config_path)
    drift_conf = build_drift_conf(cfg)
    
    # Create output directory
    video_folder = Path(output_dir) / Path(model_path).stem
    video_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. Create environment with render_mode="rgb_array" for video
    # Use factory with render_mode
    env = make_nonstationary_env(cfg['env_id'], drift_conf, render_mode="rgb_array")
    
    # 2. Wrap RecordVideo
    env = RecordVideo(
        env, 
        video_folder=str(video_folder), 
        name_prefix="eval_video",
        episode_trigger=lambda x: True,  # Record all episodes
    )
    
    # 4. Load Model with algorithm detection
    model = load_model(model_path, algo=algo)
    
    # 5. Run episodes
    for ep in range(n_episodes):
        print(f"\nRecording Episode {ep + 1}/{n_episodes}...")
        
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not (done or truncated) and step < video_length:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Log drift info periodically
            if step % 100 == 0:
                drift_info = info.get('drift/params', {})
                for param, data in drift_info.items():
                    print(f"  Step {step}: {param} = {data['current']:.2f} (Δ = {data['delta']:.2f})")
        
        print(f"  Episode finished: {step} steps, reward = {total_reward:.1f}")
    
    env.close()
    print(f"\nVideos saved to: {video_folder}")


def main():
    parser = argparse.ArgumentParser(description="Record agent videos")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--config", type=str, default="configs/cartpole_adaptive.yaml", help="Path to config")
    parser.add_argument("--algo", type=str, default="auto", help="Algorithm (PPO, SAC, TRPO, or auto)")
    parser.add_argument("--output", type=str, default="videos/", help="Output directory")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--length", type=int, default=1000, help="Max steps per episode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    record_video(
        model_path=args.model,
        config_path=args.config,
        algo=args.algo,
        output_dir=args.output,
        n_episodes=args.episodes,
        video_length=args.length,
    )


if __name__ == "__main__":
    main()