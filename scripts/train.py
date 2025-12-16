# scripts/train.py
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from src.envs.wrappers import DriftCartPoleWrapper
from src.callbacks.drift_callback import DriftAdaptiveCallback

# 1. Load Config
with open("configs/cartpole_adaptive.yaml") as f:
    config = yaml.safe_load(f)

# 2. Setup Env
env = gym.make(config["env_id"])
env = DriftCartPoleWrapper(env, drift_type=config["drift_type"])

# 3. Setup Model
model = PPO("MlpPolicy", env, **config["algo"], verbose=1)

# 4. Setup Callback
callback = DriftAdaptiveCallback() if config["adaptive"]["enabled"] else None

# 5. Train
model.learn(total_timesteps=config["total_timesteps"], callback=callback)

# 6. Save
model.save("models/ppo_adaptive_cartpole")