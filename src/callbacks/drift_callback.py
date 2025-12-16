import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class DriftAdaptiveCallback(BaseCallback):
    """
    Implements Drift-Adaptive Scheduling for PPO.
    
    Theoretical Basis:
        - Adjusts the learning step (trust region/temperature) based on environmental drift.
        - Follows the "follow the drift" heuristic: large steps when MDP moves, small steps when stable[cite: 619, 620].
        - Corresponds to Algorithm 1 in the paper[cite: 654].
    """
    def __init__(self, verbose=0):
        super(DriftAdaptiveCallback, self).__init__(verbose)
        self.current_lr = 0.0
        self.base_lr = 0.0 # Will be set on training start

    def _on_training_start(self) -> None:
        """
        Retrieve the initial learning rate from the optimizer.
        """
        # Assuming only one param group for PPO
        self.base_lr = self.model.policy.optimizer.param_groups[0]["lr"]
        if self.verbose > 0:
            print(f">>> [Callback] Adaptive Training Started. Base LR: {self.base_lr}")

    def _on_step(self) -> bool:
        """
        Executed at every step. 
        1. Detects current drift level (Oracle/Proxy).
        2. Adjusts Hyperparameters (Learning Rate).
        3. Logs to WandB.
        """
        # 1. Retrieve Drift Information from Environment Wrapper
        # vector_env.get_attr returns a list (one for each env)
        current_phys_values = self.training_env.get_attr("gravity") # Assuming gravity is the target
        
        # Heuristic: Calculate drift magnitude relative to base gravity (9.8)
        # In a real scenario, this would use the Commutator Proxy [cite: 631]
        current_val = current_phys_values[0] if current_phys_values else 9.8
        drift_magnitude = abs(current_val - 9.8)
        
        # 2. Adaptive Logic (Schedule)
        # "Large steps when the MDP moves" [cite: 619]
        # We scale LR proportionally to the drift magnitude.
        # Scale factor 0.0001 is a hyperparameter (c1 in Eq 30) [cite: 659]
        adaptation_factor = 1.0 + (drift_magnitude * 0.1) 
        new_lr = self.base_lr * adaptation_factor
        
        # Update Optimizer
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = new_lr
            
        self.current_lr = new_lr

        # 3. Logging
        # Tracks (M3) Policy movement and (M4) Commutators [cite: 821, 822]
        if self.n_calls % 100 == 0:  # Log every 100 steps to reduce overhead
            self.logger.record("adaptive/learning_rate", new_lr)
            self.logger.record("adaptive/drift_magnitude", drift_magnitude)
            self.logger.record("env/current_gravity", current_val)
            
            # If WandB is active, these are automatically synced
        
        return True