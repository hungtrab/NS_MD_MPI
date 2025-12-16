import gymnasium as gym
from stable_baselines3 import SAC

ENV_ID = "HalfCheetah-v4"
model = SAC.load("HalfCheetah-v4_sac_model.zip")

env = gym.make(ENV_ID)  # no render_mode
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()