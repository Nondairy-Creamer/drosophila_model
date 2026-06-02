"""
Find and save the steady state for a trained model.

Usage:
    python analysis/find_steady_state.py <model_path>
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports

import torch
from drosophila_model.model_definition import neural_models as nm

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find steady state for a trained model and save it."
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt file",
    )
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_pt = model_path / "model.pt" if model_path.is_dir() else model_path

    print(f"Loading model from {model_pt} ...")
    model = nm.BaseRnn.load(model_path)

    model.find_steady_state(max_iter=25)

    print(f"Saving model to {model_pt} ...")
    model.save(model_pt)
    print("Done.")


if __name__ == "__main__":
    main()
