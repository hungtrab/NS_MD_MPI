import gymnasium as gym
import numpy as np

class NonStationaryCartPoleWrapper(gym.Wrapper):
    def __init__(self, env, drift_type="jump", change_period=2000):
        super().__init__(env)
        self.drift_type = drift_type
        self.change_period = change_period
        self.step_cnt = 0
        # base parameters
        self.base_gravity = 9.8
        self.base_masscart = 1.0
        self.base_masspole = 0.1

    def step(self, action):
        self.step_cnt += 1
        
        # --- Logic Drift ---
        if self.drift_type == "jump":
            # Jump: Dramatic changes at intervals
            # Example: Gravity jumps from 9.8 to 15.0 then back
            factor = 1.5 if (self.step_cnt // self.change_period) % 2 == 1 else 1.0
            new_gravity = self.base_gravity * factor
            
            self.env.unwrapped.gravity = new_gravity

        elif self.drift_type == "sine":
            # Sine: Continuous sinusoidal changes (Simulating seasonal/cyclical changes)
            phase = self.step_cnt / 1000
            # Gravity oscillates from 9.8 +/- 50%
            self.env.unwrapped.gravity = self.base_gravity * (1.0 + 0.5 * np.sin(phase))

        elif self.drift_type == "ramp":
            # Ramp: Increasing difficulty over time (Simulating equipment wear)
            # Cart mass increases gradually over time
            drift_factor = 1.0 + (self.step_cnt / 20000.0)
            self.env.unwrapped.masscart = self.base_masscart * drift_factor
            # Need to update total_mass because CartPole uses it for calculations
            self.env.unwrapped.total_mass = self.env.unwrapped.masspole + self.env.unwrapped.masscart

        return super().step(action)

class NonStationaryMountainCarWrapper(gym.Wrapper):
    def __init__(self, env, drift_type="decay"):
        super().__init__(env)
        self.drift_type = drift_type
        self.step_cnt = 0
        self.base_force = 0.001

    def step(self, action):
        self.step_cnt += 1
        
        if self.drift_type == "decay":
            # Power Decay: Engine power decreases over time (Ramp down)
            # Decreases to a minimum of 50% power after 100k steps
            decay = max(0.5, 1.0 - (self.step_cnt / 100000.0))
            self.env.unwrapped.force = self.base_force * decay
            
        elif self.drift_type == "fluctuate":
            # Engine power fluctuates (Random Walk/Noise)
            noise = np.random.normal(0, 0.0001)
            self.env.unwrapped.force = np.clip(self.base_force + noise, 0.0005, 0.0015)

        return super().step(action)