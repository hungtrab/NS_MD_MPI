"""
Video Recording Script for Non-Stationary MDP Experiments.

Records agent behavior in the drifting environment for visualization.
"""

import sys
import os
import yaml
import argparse
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.envs.wrappers import NonStationaryCartPoleWrapper


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


def record_video(
    model_path: str,
    config_path: str,
    output_dir: str = "videos/",
    n_episodes: int = 1,
    video_length: int = 1000,
):
    """
    Record video of agent in non-stationary environment.
    
    Args:
        model_path: Path to saved model
        config_path: Path to config YAML
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
    env = gym.make(cfg['env_id'], render_mode="rgb_array")
    
    # 2. Wrap Non-Stationary (must wrap before RecordVideo)
    env = NonStationaryCartPoleWrapper(env, drift_conf)
    
    # 3. Wrap RecordVideo
    env = RecordVideo(
        env, 
        video_folder=str(video_folder), 
        name_prefix="eval_video",
        episode_trigger=lambda x: True,  # Record all episodes
    )
    
    # 4. Load Model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
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
        output_dir=args.output,
        n_episodes=args.episodes,
        video_length=args.length,
    )


if __name__ == "__main__":
    main()