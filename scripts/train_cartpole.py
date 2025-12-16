import sys
import os
# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.envs.wrappers import NonStationaryCartPoleWrapper
from src.callbacks.drift_callback import DriftAdaptiveCallback

def make_env(drift_type="jump"):
    env = gym.make("CartPole-v1")
    env = NonStationaryCartPoleWrapper(env, drift_type=drift_type, change_period=5000)
    env = Monitor(env)
    return env

def train_experiment():
    TOTAL_TIMESTEPS = 60000
    DRIFT_TYPE = "jump"

    print(f"--- Starting CartPole Non-Stationary Experiment ({DRIFT_TYPE}) ---")

    # 1. Train Baseline Agent (Stationary - No adaptation mechanism)
    print("1. Training Baseline Agent...")
    env_baseline = make_env(DRIFT_TYPE)
    model_baseline = PPO("MlpPolicy", env_baseline, verbose=0, learning_rate=0.0003)
    model_baseline.learn(total_timesteps=TOTAL_TIMESTEPS)
    model_baseline.save("models/ppo_cartpole_baseline")
    print("Baseline trained.")

    # 2. Train Adaptive Agent (Ours - With Drift Detection Callback)
    print("2. Training Adaptive Agent...")
    env_adaptive = make_env(DRIFT_TYPE)
    model_adaptive = PPO("MlpPolicy", env_adaptive, verbose=0, learning_rate=0.0003)
    
    # Callback will automatically adjust LR when environment changes
    drift_callback = DriftAdaptiveCallback(check_freq=500) 
    
    model_adaptive.learn(total_timesteps=TOTAL_TIMESTEPS, callback=drift_callback)
    model_adaptive.save("models/ppo_cartpole_adaptive")
    print("Adaptive Agent trained.")

    print("Done! Check logs or run eval script to compare.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_experiment()