import sys
import os
import yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.envs.wrappers import NonStationaryCartPoleWrapper

def main():
    # Đường dẫn file config và model (Bạn sửa lại tên model tương ứng sau khi train xong)
    config_path = "configs/cartpole_adaptive.yaml"
    model_path = "models/CartPole-v1_jump_YYYYMMDD-HHMMSS_Adaptive.zip" # <--- SỬA TÊN FILE NÀY

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Tạo môi trường giống hệt lúc train
    env = gym.make(cfg['env_id'])
    env = NonStationaryCartPoleWrapper(env, drift_type=cfg['env']['drift_type'])
    
    # Load Model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)

    # Đánh giá
    print("Evaluating...")
    # deterministic=True: Dùng hành động tốt nhất (không random)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)

    print(f"--- Result ---")
    print(f"Drift Type: {cfg['env']['drift_type']}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()