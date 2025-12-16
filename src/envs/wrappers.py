import numpy as np
import gymnasium as gym
import math

class NonStationaryCartPoleWrapper(gym.Wrapper):
    """
    A wrapper that converts the standard CartPole into a Non-Stationary MDP.
    It introduces drift to physical parameters (gravity, mass, length) over time.
    
    Theoretical Basis:
        - Simulates Dynamics Drift (Delta_P) as described in Eq. 11[cite: 240].
        - Implements specific drift patterns: Linear, Sinusoidal, and Jumps[cite: 770, 771, 774].
    """
    def __init__(self, env, drift_conf):
        super().__init__(env)
        self.drift_conf = drift_conf
        self.step_counter = 0
        self.global_step_counter = 0
        
        # Store original parameters for reference
        self.original_params = {
            'gravity': self.unwrapped.gravity,
            'masscart': self.unwrapped.masscart,
            'masspole': self.unwrapped.masspole,
            'length': self.unwrapped.length
        }
        
        # Identify the target parameter to drift (e.g., 'gravity')
        self.target_param = drift_conf.get('parameter', 'gravity')
        self.base_value = self.original_params[self.target_param]
        
        print(f">>> [Wrapper] Initialized Non-Stationary Env. Target: {self.target_param} | Type: {drift_conf['drift_type']}")

    def step(self, action):
        # 1. Update physics engine before the step to simulate continuous drift
        self._update_physics()
        
        # 2. Execute action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 3. Log current physical parameter to info dict (for Oracle/Callback access)
        # This acts as the "Drift Proxy" mentioned in Sec 7.2 [cite: 631]
        current_val = getattr(self.unwrapped, self.target_param)
        info['drift/current_value'] = current_val
        info['drift/step'] = self.step_counter
        self.global_step_counter += 1
        self.step_counter += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_counter = 0
        # Reset parameter to base value to ensure consistent episodes during training
        self._set_env_param(self.target_param, self.base_value)
        return self.env.reset(**kwargs)

    def _update_physics(self):
        """
        Calculates the new parameter value based on the configured drift pattern.
        Corresponds to the "Drift patterns" in Sec 8.1.
        """
        t = self.global_step_counter
        mode = self.drift_conf['drift_type']
        
        # Drift configuration
        magnitude = self.drift_conf.get('magnitude', 0.0) 
        period = self.drift_conf.get('period', 1000)
        
        new_value = self.base_value

        # --- DRIFT PATTERNS ---
        if mode == 'linear':
            # Linear ramps (slow drift) [cite: 771]
            rate = magnitude / period 
            new_value = self.base_value + (rate * t)

        elif mode == 'sine':
            # Sinusoidal drift (periodic) [cite: 774]
            # Formula: base + mag * sin(2 * pi * t / period)
            new_value = self.base_value + magnitude * math.sin(2 * math.pi * t / period)

        elif mode == 'jump':
            # Piecewise-constant jumps [cite: 770]
            # Simulates abrupt system faults or regime shifts
            if t > period:
                new_value = self.base_value + magnitude
        
        # --- APPLY TO ENV ---
        self._set_env_param(self.target_param, new_value)

    def _set_env_param(self, param_name, value):
        """
        Updates the environment parameter and recalculates dependent variables.
        Crucial for maintaining physics consistency in CartPole.
        """
        setattr(self.unwrapped, param_name, value)
        
        # Recalculate derived physical quantities
        # total_mass = masspole + masscart
        if param_name in ['masscart', 'masspole']:
            self.unwrapped.total_mass = self.unwrapped.masspole + self.unwrapped.masscart
        
        # polemass_length = masspole * length
        if param_name in ['length', 'masspole']:
            self.unwrapped.polemass_length = self.unwrapped.masspole * self.unwrapped.length