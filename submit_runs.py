#!/usr/bin/env python3
"""
Submission script for running multiple model training jobs across available GPUs.

This script detects available GPUs and distributes multiple training runs across them,
each with a different random seed. All runs are executed in parallel.

Usage:
    python submit_runs.py --runs-per-gpu 4
    python submit_runs.py --runs-per-gpu 10 --base-seed 100
"""

import torch
import subprocess
import argparse
import sys
from pathlib import Path
import time
from datetime import datetime


def get_available_gpus():
    """
    Detect available CUDA GPUs.
    
    Returns:
        List of GPU device strings (e.g., ['cuda:0', 'cuda:1'])
    """
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Will use CPU.")
        return ['cpu']
    
    num_gpus = torch.cuda.device_count()
    gpus = [f'cuda:{i}' for i in range(num_gpus)]
    
    # Check GPU availability (some GPUs might be in use)
    available_gpus = []
    for gpu in gpus:
        try:
            # Try to create a tensor on this GPU to check availability
            test_tensor = torch.zeros(1, device=gpu)
            del test_tensor
            available_gpus.append(gpu)
        except RuntimeError as e:
            print(f"Warning: {gpu} is not available: {e}")
    
    return available_gpus if available_gpus else ['cpu']


def run_training_job(seed, device, run_id, timestamp=None, script_path='run_model.py', log_dir='logs'):
    """
    Run a single training job with specified parameters.
    
    Args:
        seed: Random seed for this run
        device: Device to run on (e.g., 'cuda:0')
        run_id: Identifier for this run
        timestamp: Timestamp string to use for all runs in this batch
        script_path: Path to the run_model.py script
        log_dir: Directory to store log files
    
    Returns:
        tuple of (subprocess.Popen object, log file handle)
    """
    cmd = [
        sys.executable,
        str(script_path),
        '--seed', str(seed),
        '--device', device,
        '--run-id', str(run_id)
    ]
    if timestamp is not None:
        cmd.extend(['--timestamp', timestamp])
    
    print(f"Starting run: seed={seed}, device={device}, run_id={run_id}")
    print(f"Command: {' '.join(cmd)}")
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Open log file for this run (redirect both stdout and stderr to avoid pipe deadlock)
    log_file = open(log_path / f'{timestamp}_{run_id}.log', 'w')
    
    # Run in background (non-blocking) with output to file
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    
    return process, log_file


def submit_runs(runs_per_gpu=1, base_seed=0, script_path='run_model.py'):
    """
    Submit multiple training runs across available GPUs.
    
    Args:
        runs_per_gpu: Number of runs to execute per GPU
        base_seed: Base seed value (each run gets base_seed + run_index)
        script_path: Path to the run_model.py script
    """
    # Generate a single timestamp for all runs in this batch
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Using timestamp for all runs: {timestamp}")
    
    # Detect available GPUs
    gpus = get_available_gpus()
    print(f"Found {len(gpus)} available device(s): {gpus}")
    
    if not gpus:
        print("Error: No available devices found.")
        return
    
    # Calculate total number of runs
    total_runs = len(gpus) * runs_per_gpu
    print(f"Submitting {total_runs} runs ({runs_per_gpu} per GPU)")
    
    # Generate run configurations
    runs = []
    run_index = 0
    for gpu_idx, gpu in enumerate(gpus):
        for run_idx in range(runs_per_gpu):
            seed = base_seed + run_index
            run_id = f"gpu{gpu_idx}_run{run_idx}"
            runs.append({
                'seed': seed,
                'device': gpu,
                'run_id': run_id,
                'run_index': run_index
            })
            run_index += 1
    
    # Submit runs in parallel
    processes = []
    print("\nRunning jobs in parallel...")
    for run_config in runs:
        process, log_file = run_training_job(
            seed=run_config['seed'],
            device=run_config['device'],
            run_id=run_config['run_id'],
            timestamp=timestamp,
            script_path=script_path
        )
        processes.append((process, run_config, log_file))
    
    # Wait for all processes to complete
    print(f"\nWaiting for {len(processes)} jobs to complete...")
    print(f"Logs are being written to logs/{timestamp}_*.log")
    for process, run_config, log_file in processes:
        process.wait()
        log_file.close()
        if process.returncode != 0:
            print(f"Error in run {run_config['run_id']} (exit code {process.returncode}). Check logs/{timestamp}_{run_config['run_id']}.log")
        else:
            print(f"Completed run {run_config['run_id']} successfully")
    
    print("\nAll runs completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Submit multiple model training runs across available GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Run 4 jobs per GPU with base seed 0
    python submit_runs.py --runs-per-gpu 4
    
    # Run 10 jobs per GPU with custom seed
    python submit_runs.py --runs-per-gpu 10 --base-seed 100
                """
    )
    
    parser.add_argument(
        '--runs-per-gpu',
        type=int,
        default=1,
        help='Number of runs to execute per GPU (default: 1)'
    )
    
    parser.add_argument(
        '--base-seed',
        type=int,
        default=0,
        help='Base seed value. Each run gets base_seed + run_index (default: 0)'
    )
    
    parser.add_argument(
        '--script-path',
        type=str,
        default='run_model.py',
        help='Path to run_model.py script (default: run_model.py)'
    )
    
    args = parser.parse_args()
    
    # Validate script path
    script_path = Path(args.script_path)
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Submit runs
    submit_runs(
        runs_per_gpu=args.runs_per_gpu,
        base_seed=args.base_seed,
        script_path=str(script_path)
    )


if __name__ == '__main__':
    main()
