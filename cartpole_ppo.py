import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "CartPole-v1"

env = gym.make(ENV_ID)
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1, 
            learning_rate=0.0003,
            gamma=0.99,            
            n_steps=2048,          
            seed=42)

model.learn(total_timesteps=100_000)

model.save(f"{ENV_ID}_ppo_model")
print(f"Saved PPO model for {ENV_ID}.")