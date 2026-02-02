#!/usr/bin/env python3
import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repo tests via pytest.")
    parser.add_argument(
        "--path",
        default="tests",
        help="Test path to run (default: tests)",
    )
    args = parser.parse_args()

    return subprocess.call([sys.executable, "-m", "pytest", args.path])


if __name__ == "__main__":
    raise SystemExit(main())
