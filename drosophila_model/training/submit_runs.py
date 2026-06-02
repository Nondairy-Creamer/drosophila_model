#!/usr/bin/env python3
"""
Submission script for running multiple model training jobs across available GPUs.

This script detects available GPUs and distributes multiple training runs across them,
each with a different random seed. All runs are executed in parallel.

Usage:
    python submit_runs.py --runs-per-gpu 4
    python submit_runs.py --runs-per-gpu 10 --base-seed 100
    python submit_runs.py --runs-per-gpu 4 --tag my_experiment
"""

import torch
import subprocess
import argparse
import sys
import os
import signal
from pathlib import Path
import time
from datetime import datetime

# Global list to track running processes for cleanup
_running_processes = []


def cleanup_processes(signum=None, frame=None):
    """Clean up all running child processes on exit or signal."""
    for process, _, log_file in _running_processes:
        if process.poll() is None:  # Process still running
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        try:
            log_file.close()
        except:
            pass
    if signum is not None:
        sys.exit(1)


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
    if num_gpus == 0:
        print("Warning: No CUDA GPUs detected. Will use CPU.")
        return ['cpu']
    
    # Don't create CUDA tensors here -- the parent process only dispatches
    # jobs and never needs GPU access.  Initialising CUDA contexts would
    # waste ~1.5 GB per GPU and can interfere with child processes after fork.
    return [f'cuda:{i}' for i in range(num_gpus)]


def run_training_job(seed, device, run_id, timestamp=None, script_path=None, log_dir='logs', model_path=None):
    """
    Run a single training job with specified parameters.
    
    Args:
        seed: Random seed for this run
        device: Device to run on (e.g., 'cuda:0')
        run_id: Identifier for this run
        timestamp: Timestamp string to use for all runs in this batch
        script_path: Optional path to a run_model.py script. If None, runs the
            installed module ``drosophila_model.training.run_model``.
        log_dir: Directory to store log files
        model_path: Optional path to existing model folder to load
    
    Returns:
        tuple of (subprocess.Popen object, log file handle)
    """
    if script_path is None:
        cmd = [sys.executable, '-u', '-m', 'drosophila_model.training.run_model']
    else:
        cmd = [sys.executable, '-u', str(script_path)]
    cmd.extend([
        '--seed', str(seed),
        '--device', device,
        '--run-id', str(run_id),
    ])
    if timestamp is not None:
        cmd.extend(['--timestamp', timestamp])
    if model_path is not None:
        cmd.extend(['--model-folder', model_path])
    
    print(f"Starting run: seed={seed}, device={device}, run_id={run_id}")
    print(f"Command: {' '.join(cmd)}")
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Open log file for this run (redirect both stdout and stderr to avoid pipe deadlock)
    # Use line buffering so output appears immediately
    log_file = open(log_path / f'{timestamp}_{run_id}.log', 'w', buffering=1)
    
    # Set environment variable to ensure unbuffered output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    # Run in background (non-blocking) with output to file
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env
    )
    
    return process, log_file


def submit_runs(runs_per_gpu=1, base_seed=0, script_path=None, model_path=None, tag=None):
    """
    Submit multiple training runs across available GPUs.
    
    Args:
        runs_per_gpu: Number of runs to execute per GPU
        base_seed: Base seed value (each run gets base_seed + run_index)
        script_path: Optional path to a run_model.py script. If None, runs the
            installed module ``drosophila_model.training.run_model``.
        model_path: Optional path to existing model folder to load
        tag: Optional tag to use instead of the auto-generated timestamp
    """
    timestamp = tag if tag is not None else datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Using tag for all runs: {timestamp}")
    
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
    
    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup_processes)
    signal.signal(signal.SIGTERM, cleanup_processes)
    
    # Submit runs in parallel
    global _running_processes
    _running_processes = []
    print("\nRunning jobs in parallel...")
    for run_config in runs:
        process, log_file = run_training_job(
            seed=run_config['seed'],
            device=run_config['device'],
            run_id=run_config['run_id'],
            timestamp=timestamp,
            script_path=script_path,
            model_path=model_path
        )
        _running_processes.append((process, run_config, log_file))
    
    # Wait for all processes using polling so children are reaped as they finish
    # (sequential process.wait() leaves zombies when an earlier child outlasts later ones)
    print(f"\nWaiting for {len(_running_processes)} jobs to complete...")
    print(f"Logs are being written to logs/{timestamp}_*.log")
    remaining = list(_running_processes)
    while remaining:
        still_running = []
        for process, run_config, log_file in remaining:
            ret = process.poll()
            if ret is not None:  # process has finished
                log_file.close()
                if ret != 0:
                    print(f"Error in run {run_config['run_id']} (exit code {ret}). Check logs/{timestamp}_{run_config['run_id']}.log")
                else:
                    print(f"Completed run {run_config['run_id']} successfully")
            else:
                still_running.append((process, run_config, log_file))
        remaining = still_running
        if remaining:
            time.sleep(5)
    
    _running_processes = []  # Clear after completion
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
        default=None,
        help='Optional path to a run_model.py script. If omitted, runs the '
             'installed module drosophila_model.training.run_model.'
    )
    
    parser.add_argument(
        '--model-folder',
        type=str,
        default=None,
        help='Path to existing model folder to load (e.g., models/best_model)'
    )
    
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help='Tag to use instead of the auto-generated timestamp for naming runs'
    )
    
    parser.add_argument(
        '--foreground', '--fg',
        action='store_true',
        default=False,
        help='Run in the foreground instead of detaching (default: run in background).'
    )
    
    args = parser.parse_args()
    
    # By default, re-launch ourselves detached with output to log_submit_runs.log
    # (replicates: nohup python -u submit_runs.py ... > log_submit_runs.log 2>&1 &)
    if not args.foreground and os.environ.get('_SUBMIT_RUNS_FG') != '1':
        env = os.environ.copy()
        env['_SUBMIT_RUNS_FG'] = '1'
        log_file = open('log_submit_runs.log', 'w')
        proc = subprocess.Popen(
            [sys.executable, '-u'] + sys.argv + ['--foreground'],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach from terminal (POSIX setsid)
            env=env,
        )
        print(f"Running in background (PID {proc.pid}). Output: log_submit_runs.log")
        print(f"To follow: tail -f log_submit_runs.log")
        print(f"To stop:   kill {proc.pid}")
        sys.exit(0)
    
    # Validate script path if explicitly provided; otherwise we invoke the
    # installed module drosophila_model.training.run_model.
    script_path = None
    if args.script_path is not None:
        script_path = Path(args.script_path)
        if not script_path.exists():
            print(f"Error: Script not found at {script_path}")
            sys.exit(1)
        script_path = str(script_path)
    
    # Validate model path if provided
    model_path = None
    if args.model_folder is not None:
        model_path = Path(args.model_folder)
        model_file = model_path / 'model.pt'
        if not model_file.exists():
            print(f"Error: Model file not found at {model_file}")
            sys.exit(1)
        model_path = str(model_path)
    
    # Submit runs
    submit_runs(
        runs_per_gpu=args.runs_per_gpu,
        base_seed=args.base_seed,
        script_path=script_path,
        model_path=model_path,
        tag=args.tag
    )


if __name__ == '__main__':
    main()
