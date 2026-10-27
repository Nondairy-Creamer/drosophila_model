#!/usr/bin/env python3
"""Stop every launch_runs launcher and training process (POSIX).

Usage (from the repo root):

    python -m drosophila_model.training.stop_runs

Matches both the module form (``-m drosophila_model.training.launch_runs`` /
``...training.train``) and the script form (``launch_runs.py`` /
``training/train.py``; a bare ``train.py`` is too common a name to match).
launch_runs launchers are signalled first, so their handler can stop their
children; any training process still alive is then signalled directly.
"""
import argparse
import os
import signal
import subprocess
import sys

PATTERNS = (
    r'drosophila_model\.training\.launch_runs|launch_runs\.py',
    r'drosophila_model\.training\.train( |$)|training/train\.py',
)


def matching_pids(pattern):
    """PIDs whose command line matches ``pattern`` (excluding this process)."""
    result = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True)
    own = {os.getpid(), os.getppid()}
    return [int(p) for p in result.stdout.split() if int(p) not in own]


def main():
    argparse.ArgumentParser(
        description='Send SIGTERM to every launch_runs launcher and training '
                    'process (POSIX only).'
    ).parse_args()
    try:
        for pattern in PATTERNS:
            for pid in matching_pids(pattern):
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f'Sent SIGTERM to {pid}')
                except ProcessLookupError:
                    pass
                except PermissionError:
                    print(f'No permission to signal {pid}')
    except FileNotFoundError:
        sys.exit('pgrep not found: stop_runs only works on POSIX systems.')


if __name__ == '__main__':
    main()
