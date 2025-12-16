import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "CartPole-v1" 
env = gym.make(ENV_ID)
env = DummyVecEnv([lambda: env])

model = PPO.load("CartPole-v1_ppo_model.zip", env=env)


mean_reward, std_reward = evaluate_policy(
    model, 
    model.get_env(), 
    n_eval_episodes=20, 
    render=False, 
    deterministic=True
)

print(f"Expected reward after 20 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

