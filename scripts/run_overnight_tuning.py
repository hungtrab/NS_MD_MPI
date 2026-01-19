#!/usr/bin/env python3
"""
Overnight Tuning Script
Run multiple WandB sweeps in parallel for Hopper and HalfCheetah
"""

import subprocess
import multiprocessing
import os
import sys
import time
from datetime import datetime

# Configuration
SWEEPS = [
    # Hopper NS-MDMPI
    {"env": "Hopper-v4", "type": "moderate", "count": 20, "name": "Hopper_NSMDMPI_1"},
    {"env": "Hopper-v4", "type": "moderate", "count": 20, "name": "Hopper_NSMDMPI_2"},
    
    # HalfCheetah NS-MDMPI
    {"env": "HalfCheetah-v4", "type": "moderate", "count": 20, "name": "HalfCheetah_NSMDMPI_1"},
    {"env": "HalfCheetah-v4", "type": "moderate", "count": 20, "name": "HalfCheetah_NSMDMPI_2"},
]

def run_sweep(config):
    """Run a single sweep"""
    name = config["name"]
    env = config["env"]
    sweep_type = config["type"]
    count = config["count"]
    
    log_dir = "logs/tuning"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{name}.log"
    
    cmd = [
        "python", "scripts/wandb_sweep_agent.py",
        "--create-sweep",
        "--type", sweep_type,
        "--env", env,
        "--count", str(count)
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {name}...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")
    
    with open(log_file, "w") as f:
        process = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/.."
        )
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished {name} (exit code: {process.returncode})")
    return process.returncode

def main():
    print("=" * 60)
    print(f"OVERNIGHT TUNING - Started at {datetime.now()}")
    print("=" * 60)
    print(f"\nRunning {len(SWEEPS)} sweeps in parallel...")
    print()
    
    # Run all sweeps in parallel
    with multiprocessing.Pool(processes=len(SWEEPS)) as pool:
        results = pool.map(run_sweep, SWEEPS)
    
    print()
    print("=" * 60)
    print(f"All tuning completed at {datetime.now()}")
    print("=" * 60)
    print("\nResults:")
    for sweep, result in zip(SWEEPS, results):
        status = "✅ Success" if result == 0 else f"❌ Failed (code {result})"
        print(f"  {sweep['name']}: {status}")
    
    print("\nCheck WandB dashboard:")
    print("  https://wandb.ai/hungtrab-hanoi-university-of-science-and-technology")

if __name__ == "__main__":
    main()
