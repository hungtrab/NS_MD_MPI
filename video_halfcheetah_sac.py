from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from stable_baselines3 import SAC

ENV_ID = "HalfCheetah-v4"
model = SAC.load("HalfCheetah-v4_sac_model.zip") 


video_folder = "./videos/halfcheetah_eval/"
eval_env = gym.make(ENV_ID, render_mode="rgb_array")

eval_env = RecordVideo(
    eval_env, 
    video_folder=video_folder, 
    episode_trigger=lambda x: x % 10 == 0
)

obs, info = eval_env.reset()
n_steps = 1000
episode_count = 0

print(f"Start capturing to: {video_folder}")

while episode_count < 30:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    if terminated or truncated:
        obs, info = eval_env.reset()
        episode_count += 1

eval_env.close()
print("Video captured") 