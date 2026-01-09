import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import yaml
import subprocess
import os
import sys
import pathlib
import pandas as pd

def check_dependencies():
    """Check if required packages are installed in the current environment."""
    required_packages = [
        'stable_baselines3',
        'gymnasium',
        'wandb',
        'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("\n" + "="*70)
        print("ERROR: Missing required packages in current environment!")
        print("="*70)
        print(f"Missing: {', '.join(missing_packages)}")
        print("\nPlease install them with:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("="*70 + "\n")
        sys.exit(1)
    
    print(f"✓ All required dependencies found in: {sys.executable}\n")

# Function để chạy train.py với config đã chỉnh sửa
def run_experiment(config_path, project_root):
    # Gọi lệnh train.py với đường dẫn tuyệt đối
    train_script = project_root / "scripts" / "train.py"
    cmd = [sys.executable, str(train_script), "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    
    if result.returncode != 0:
        print(f"Training failed with error:\n{result.stderr}")
    
    return result.returncode

def get_mean_reward_last_10_percent(log_dir):
    """
    Đọc file monitor.csv và tính trung bình phần thưởng của 10% thời gian cuối cùng.
    Đây là metrics quan trọng cho NS-MD-MPI.
    """
    # Tìm file monitor.csv trong thư mục log (đệ quy tìm vì SB3 có thể tạo thư mục con theo timestamp)
    monitor_file = None
    for root, dirs, files in os.walk(log_dir):
        if "monitor.csv" in files:
            monitor_file = os.path.join(root, "monitor.csv")
            break
    
    if monitor_file is None:
        print(f"Không tìm thấy monitor.csv trong {log_dir}")
        return -1000.0
    
    try:
        # Đọc CSV: Column r là reward, t là timestep, l là length
        # Skip comment line (first line starts with #)
        df = pd.read_csv(monitor_file, comment='#')
        
        if df.empty or len(df) < 10:
            print(f"Not enough episodes: {len(df) if not df.empty else 0}")
            return -1000.0
        
        # Lọc ra 10% phần cuối
        n_episodes = len(df)
        last_10_percent_idx = int(n_episodes * 0.9)
        
        # Tính trung bình reward của các episode cuối
        final_rewards = df['r'].iloc[last_10_percent_idx:]
        mean_reward = final_rewards.mean()
        
        print(f" -> Calc Score: {mean_reward:.2f} (from {len(final_rewards)} episodes, total: {n_episodes})")
        return mean_reward
        
    except Exception as e:
        print(f"Lỗi khi đọc log: {e}")
        print(f"Monitor file: {monitor_file}")
        if monitor_file and os.path.exists(monitor_file):
            print(f"File exists, first few lines:")
            with open(monitor_file, 'r') as f:
                print(f.read(500))
        return -1000.0

# Objective Function cho Optuna
def objective(trial):
    # 1. Load config cơ bản (ví dụ MountainCar Jump)
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    base_config = project_root / "configs" / "MountainCar_nsmdmpi_force_jump_ppo.yaml"
    
    if not base_config.exists():
        raise FileNotFoundError(f"Config file not found: {base_config}")
    
    with open(base_config, 'r') as f:
        cfg = yaml.safe_load(f)

    # ==========================================================
    # TỐNG CÁC THAM SỐ CẦN TUNE
    # ==========================================================

    # --- A. Tune các tham số Kiểm soát phản hồi (Control) ---
    
    # Trust Region Sensitivity (C1): Tăng khi drift mạnh
    # Khoảng tìm kiếm: Nhỏ (bị trễ) -> Rất lớn (quá nhạy)
    cfg['nsmdmpi']['trust_region_sensitivity'] = trial.suggest_float('trust_region_sensitivity', 0.1, 10.0, log=True)

    # EMA Tau (Coefficient làm mooth): 
    # Khoảng: 0.01 (quá chậm, tin quá khứ) -> 0.9 (tin ngay lập tức, phản ứng nhanh)
    cfg['nsmdmpi']['ema_tau'] = trial.suggest_float('ema_tau', 0.01, 0.9)
    
    # Kappa Min/Max (Khoảng dao động của Trust Region)
    cfg['nsmdmpi']['kappa_min'] = trial.suggest_float('kappa_min', 0.01, 0.2)
    cfg['nsmdmpi']['kappa_max'] = trial.suggest_float('kappa_max', 0.2, 0.6)

    # Regularization Sensitivity (Beta)
    cfg['nsmdmpi']['regularization_sensitivity'] = trial.suggest_float('regularization_sensitivity', 0.1, 5.0)

    # --- B. Tune các tham số Ngân sách (Scale Factors) ---
    
    # Thay vì tune V_P trực tiếp, ta tune hệ số nhân
    # V_P = Theoretical_V_P * scale_factor
    cfg['nsmdmpi']['budget_scale_factor'] = trial.suggest_float('budget_scale_factor', 0.5, 3.0)

    # ==========================================================
    # CẤU HÌNH OPTUNA PRUNING (Tắt thử nghiệm tệ sớm)
    # ==========================================================
    # Chú ý: Đây là giả lập logic pruning. Trong thực tế, bạn cần viết một wrapper
    # để báo cáo score trung gian lại cho Optuna.
    # Ở đây ta chạy trọn vẹn mỗi trial.

    # 2. Ghi ra file config tạm thời và thiết lập log directory riêng cho trial
    trial_dir = project_root / "tune_results" / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    temp_config_path = trial_dir / "config.yaml"
    
    # Đặt log directory riêng cho trial này
    trial_log_dir = trial_dir / "logs"
    cfg['paths']['log_dir'] = str(trial_log_dir)
    
    # Disable wandb during hyperparameter tuning to avoid pickling errors and save time
    # We only need the performance metrics, not model saving/logging
    if 'wandb' not in cfg:
        cfg['wandb'] = {}
    cfg['wandb']['enabled'] = False
    
    # Disable final model saving during tuning to avoid pickling errors
    if 'paths' not in cfg:
        cfg['paths'] = {}
    cfg['paths']['save_model'] = False
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(cfg, f)

    # CHÚ Ý QUAN TRỌNG: Bạn nên giảm total_timesteps trong config gốc xuống 
    # ví dụ: 200000 thay vì 5000000 để Optuna chạy thử nhanh hơn.
    # Sau khi tune xong, mới dùng tham số tốt nhất để chạy full thời gian.

    # 3. Chạy huấn luyện
    try:
        # Process này sẽ BLOCK (chặn) dòng code cho đến khi train.py xong
        returncode = run_experiment(str(temp_config_path), project_root)
        if returncode != 0:
            print(f"Trial {trial.number} failed with return code {returncode}")
            return -1000.0
    except Exception as e:
        print(f"Trial {trial.number} crashed: {e}")
        return -1000.0

    # 4. Lấy kết quả thật
    # Lấy thư mục log đã cấu hình trong config
    log_dir = pathlib.Path(cfg['paths']['log_dir'])
    
    # Kiểm tra xem log directory có tồn tại không
    if not log_dir.exists() or not any(log_dir.iterdir()):
        print(f"Log directory empty or not found: {log_dir}")
        return -1000.0
    
    # Cần tìm đúng thư mục con vì SB3 tạo thêm thư mục theo thời gian (YYYY-MM-DD...)
    # Ta lấy thư mục gần nhất được sửa đổi trong log_dir
    try:
        subdirs = [d for d in log_dir.iterdir() if d.is_dir()]
        if not subdirs:
            recent_log_dir = log_dir
        else:
            recent_log_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    except Exception as e:
        print(f"Error finding recent log directory: {e}")
        return -1000.0

    score = get_mean_reward_last_10_percent(str(recent_log_dir))
    return score

if __name__ == "__main__":
    # Check dependencies first
    check_dependencies()
    
    # Tạo thư mục tune_results
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    tune_results_dir = project_root / "tune_results"
    tune_results_dir.mkdir(exist_ok=True)
    
    # Tạo study
    study = optuna.create_study(
        study_name="nsmdmpi_mountaincar_jump",
        direction="maximize", # Tối đa hóa Reward (hoặc minimize Dynamic Regret)
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1000)
    )

    # Chạy 50 trials
    study.optimize(objective, n_trials=50)

    print("\nBest hyperparameters:")
    print(study.best_params)
    print("Best Value:", study.best_value)
    
    # Lưu kết quả
    results_file = tune_results_dir / "best_hyperparameters.yaml"
    with open(results_file, 'w') as f:
        yaml.dump({
            'best_params': study.best_params,
            'best_value': float(study.best_value),
            'n_trials': len(study.trials)
        }, f)
    print(f"\nResults saved to {results_file}")