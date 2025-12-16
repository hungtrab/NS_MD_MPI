import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "HalfCheetah-v4" 

env = gym.make(ENV_ID)
env = DummyVecEnv([lambda: env])

model = SAC("MlpPolicy", env, verbose=1, 
            learning_rate=0.0003,
            gamma=0.99,
            buffer_size=1000000,
            ent_coef='auto',
            seed=42)

model.learn(total_timesteps=1_000_000)

model.save(f"{ENV_ID}_sac_model")
print(f"SAC model for {ENV_ID} saved.")