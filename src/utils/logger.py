import os
import wandb
from torch.utils.tensorboard import SummaryWriter

class UnifiedLogger:
    def __init__(self, config, run_name, use_wandb=True, use_tb=True):
        self.use_wandb = use_wandb
        self.use_tb = use_tb
        
        # Setup WandB (Online)
        if self.use_wandb:
            # Nhớ login wandb trong terminal trước: wandb login
            wandb.init(
                project=config.get("project_name", "my_project"),
                name=run_name,
                config=config,
                reinit=True
            )
            print(">>> WandB Initialized.")

        # Setup TensorBoard (Offline)
        if self.use_tb:
            log_dir = os.path.join("runs", run_name)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f">>> TensorBoard Initialized at {log_dir}")

    def log_metrics(self, metrics_dict, step):
        """
        metrics_dict: Dictionary chứa key-value (vd: {'loss': 0.5, 'acc': 0.9})
        step: Global step hoặc epoch
        """
        # Log to WandB
        if self.use_wandb:
            wandb.log(metrics_dict, step=step)

        # Log to TensorBoard
        if self.use_tb:
            for key, value in metrics_dict.items():
                # TensorBoard cần add từng scalar một
                self.writer.add_scalar(key, value, step)

    def close(self):
        if self.use_wandb:
            wandb.finish()
        if self.use_tb:
            self.writer.close()
        print(">>> Logging closed.")