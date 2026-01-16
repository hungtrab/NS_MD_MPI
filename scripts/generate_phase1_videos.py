#!/usr/bin/env python3
"""
Generate videos of trained agents playing LunarLander.
Creates side-by-side comparison of Phase 1 experiments.
"""

import gymnasium as gym
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
import numpy as np

def record_video(model_path, env_id, output_path, n_episodes=3, max_steps=1000):
    """Record video of agent playing."""
    
    print(f"\n{'='*60}")
    print(f"Recording: {Path(model_path).stem}")
    print(f"{'='*60}")
    
    # Create environment with render mode
    env = gym.make(env_id, render_mode="rgb_array")
    
    # Load trained model
    try:
        model = PPO.load(model_path, env=env)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Record frames
    all_frames = []
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        frames = []
        
        for step in range(max_steps):
            # Render frame
            frame = env.render()
            frames.append(frame)
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                break
        
        episode_rewards.append(ep_reward)
        all_frames.extend(frames)
        print(f"  Episode {ep+1}: reward = {ep_reward:.2f}, steps = {len(frames)}")
    
    env.close()
    
    # Save video 
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try cv2 first (more reliable for mp4)
        try:
            import cv2
            
            height, width = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
            
            for frame in all_frames:
                # Convert RGB to BGR for cv2
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            
            print(f"\n✅ Video saved: {output_path}")
            print(f"   Average reward: {np.mean(episode_rewards):.2f}")
            print(f"   Total frames: {len(all_frames)}")
            return output_path
            
        except ImportError:
            # Fallback to imageio GIF
            import imageio
            gif_path = output_path.replace('.mp4', '.gif')
            imageio.mimsave(gif_path, all_frames[::3], duration=100)  # Skip frames for smaller GIF
            
            print(f"\n✅ GIF saved: {gif_path}")
            print(f"   Average reward: {np.mean(episode_rewards):.2f}")
            return gif_path
        
    except Exception as e:
        print(f"❌ Failed to save video: {e}")
        return None


def main():
    # Phase 1 models (timestamp 132731-132733)
    # Note: NSMDMPI saved as .pt params only, not full SB3 model
    models = {
        'vanilla': {
            'path': 'models/LunarLander-v3_PPO_vanilla_20260114-132731_Baseline.zip',
            'env': 'LunarLander-v3',
            'name': '1_vanilla_no_drift'
        },
        'baseline': {
            'path': 'models/LunarLander-v3_PPO_sine_20260114-132733_Baseline.zip',
            'env': 'LunarLander-v3',
            'name': '2_moderate_baseline'
        },
        # NSMDMPI model was saved as .pt params, cannot load as SB3 model
        # To fix: update callback to also save full model.zip
    }
    
    print("="*60)
    print("  GENERATING PHASE 1 VIDEOS")
    print("="*60)
    
    # Create videos directory
    video_dir = Path('videos/phase1')
    video_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    
    for name, config in models.items():
        model_path = config['path']
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found: {model_path}")
            continue
        
        output_path = str(video_dir / f"{config['name']}.mp4")
        
        result = record_video(
            model_path=model_path,
            env_id=config['env'],
            output_path=output_path,
            n_episodes=3,
            max_steps=1000
        )
        
        if result:
            generated.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n✅ Generated {len(generated)} videos:")
    for vid in generated:
        print(f"   - {vid}")
    
    print(f"\n📁 Videos saved to: {video_dir}/")
    print("\nTo view:")
    print(f"   vlc {video_dir}/*.mp4")


if __name__ == "__main__":
    main()
