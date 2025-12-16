import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class DriftAdaptiveCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.last_mean_reward = -float('inf')
        self.base_lr = 0.0003  # Default
        self.adaptive_lr = 0.001 # High lr for adaptation
        self.adaptation_steps = 0 # Countdown steps to keep high LR
        self.rewards_buffer = [] # Buffer to calculate average reward

    def _on_step(self) -> bool:
        # Get reward of the current step (if any)
        # infos usually contain 'episode' info if episode ends
        for info in self.locals['infos']:
            if 'episode' in info:
                self.rewards_buffer.append(info['episode']['r'])

        # Periodic check
        if self.n_calls % self.check_freq == 0:
            if len(self.rewards_buffer) > 0:
                current_mean_reward = np.mean(self.rewards_buffer[-20:]) # Mean of last 20 episodes
                
                # Simple Drift Detection Logic:
                # If reward drops significantly (> 30%) compared to last check -> Drift detected
                if self.last_mean_reward > -float('inf'):
                    drop_percentage = (self.last_mean_reward - current_mean_reward) / abs(self.last_mean_reward) if self.last_mean_reward != 0 else 0
                    
                    if drop_percentage > 0.3: # Drift detection threshold (tunable)
                        if self.verbose > 0:
                            print(f"\n[Drift Detected] Step {self.num_timesteps}: Reward dropped from {self.last_mean_reward:.2f} to {current_mean_reward:.2f}")
                            print(f"--> Activating Adaptive Mode: Increasing LR to {self.adaptive_lr}")
                        
                        # Activate adaptive mode for the next 3000 steps
                        self.adaptation_steps = 3000 

                self.last_mean_reward = current_mean_reward
                self.rewards_buffer = [] # Reset buffer

        # --- Hyperparameter Adjustment Mechanism ---
        # Directly access PPO's Optimizer to modify Learning Rate
        optimizer = self.model.policy.optimizer
        
        if self.adaptation_steps > 0:
            # In adaptation phase -> Use high LR
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.adaptive_lr
            self.adaptation_steps -= 1
        else:
            # Normal phase -> Use low LR (stable)
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.base_lr

        return True