"""
Find and save the steady state for a trained model.

Overwrites the model's model.pt in place (no backup is kept).

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.find_steady_state models/<model>/
"""
import argparse
from pathlib import Path

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

    if not model.find_steady_state(max_iter=25):
        # Dynamics blew up: the steady state was left unchanged, so don't
        # rewrite model.pt (and don't report success).
        raise SystemExit(f"No stable steady state found; {model_pt} left unchanged.")

    print(f"Saving model to {model_pt} ...")
    model.save(model_pt)
    print("Done.")


if __name__ == "__main__":
    main()
