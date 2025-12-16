import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class DriftAdaptiveCallback(BaseCallback):
    """
    Implements Drift-Adaptive Scheduling for PPO.
    Fixed version: Updates the schedule instead of the optimizer directly.
    """
    def __init__(self, base_gravity=9.8, verbose=0):
        super(DriftAdaptiveCallback, self).__init__(verbose)
        self.base_gravity = base_gravity
        self.current_lr = 0.0
        self.base_lr = 0.0 

    def _on_training_start(self) -> None:
        # Lấy LR ban đầu
        self.base_lr = self.model.learning_rate
        if self.verbose > 0:
            print(f">>> [Callback] Adaptive Training Started. Base LR: {self.base_lr}")

    def _on_step(self) -> bool:
        # Tối ưu: Chỉ check drift mỗi 100 steps hoặc cuối mỗi rollout để đỡ lag
        # Nhưng nếu muốn check từng step (theo lý thuyết) thì chấp nhận chậm
        
        # 1. Retrieve Drift Information
        # Lưu ý: get_attr vẫn chậm, nhưng nếu env đơn thì ko sao.
        current_phys_values = self.training_env.get_attr("gravity")
        current_val = current_phys_values[0] if current_phys_values else self.base_gravity
        
        drift_magnitude = abs(current_val - self.base_gravity)
        
        # 2. Adaptive Logic
        adaptation_factor = 1.0 + (drift_magnitude * 0.1) 
        new_lr = self.base_lr * adaptation_factor
        self.current_lr = new_lr

        # --- FIX QUAN TRỌNG NHẤT ---
        # Thay vì update optimizer, ta update cái hàm schedule.
        # Khi PPO gọi _update_learning_rate, nó sẽ dùng hàm này để lấy LR mới.
        self.model.lr_schedule = lambda _: new_lr
        # ---------------------------

        # 3. Logging
        if self.n_calls % 100 == 0:
            self.logger.record("adaptive/learning_rate", new_lr)
            self.logger.record("adaptive/drift_magnitude", drift_magnitude)
            self.logger.record("env/current_gravity", current_val)
        
        return True