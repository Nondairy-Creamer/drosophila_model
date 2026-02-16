"""
Find and save the steady state for a trained model.

Usage:
    python analysis/find_steady_state.py <model_path>
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import data_utilities as du
import neural_models as nm
from analysis_utilities import resolve_model_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find steady state for a trained model and save it."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model folder or model.pt file",
    )
    args = parser.parse_args()

    model_pt, model_folder = resolve_model_paths(Path(args.model_path))
    local_config = du.load_config()
    device = torch.device(local_config["model"]["device"])

    print(f"Loading model from {model_pt} ...")
    model = nm.NeuralModel.load(model_pt, device=device)

    model.find_steady_state(max_iter=25)

    print(f"Saving model to {model_pt} ...")
    model.save(model_pt)
    print("Done.")


if __name__ == "__main__":
    main()
