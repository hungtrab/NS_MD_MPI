import sys
import os
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.envs.wrappers import NonStationaryCartPoleWrapper

def main():
    # Config
    config_path = "configs/cartpole_adaptive.yaml"
    model_path = "models/CartPole-v1_jump_YYYYMMDD-HHMMSS_Adaptive.zip" # <--- SỬA TÊN FILE NÀY
    video_folder = "videos/"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. Tạo môi trường với render_mode="rgb_array" để quay video
    env = gym.make(cfg['env_id'], render_mode="rgb_array")
    
    # 2. Wrap Non-Stationary (Quan trọng: Phải wrap trước RecordVideo)
    env = NonStationaryCartPoleWrapper(env, drift_type=cfg['env']['drift_type'])
    
    # 3. Wrap RecordVideo
    # episode_trigger: lambda x: True nghĩa là quay mọi episode (ở đây ta chỉ chạy 1)
    env = RecordVideo(env, video_folder=video_folder, name_prefix="eval_video", disable_logger=True)

    # 4. Load Model
    model = PPO.load(model_path)

    # 5. Run 1 Episode
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step = 0
    
    print("Recording video...")
    while not (done or truncated):
        # Action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # In thông tin vật lý để debug xem drift có xảy ra không
        if step % 100 == 0:
            gravity = env.unwrapped.gravity
            print(f"Step {step}: Gravity = {gravity:.2f}")

    env.close()
    print(f"Done! Video saved to {video_folder}")
    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    os.makedirs("videos", exist_ok=True)
    main()