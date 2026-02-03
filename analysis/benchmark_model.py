"""
Benchmark optic flow predictions against stimuli.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import training_utilities as tu

from analysis_utilities import run_model_on_stimuli


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model optic flow predictions.")
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="models/best_model",
        help="Path to model folder or model.pt (default: models/best_model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (default: cpu)",
    )
    args = parser.parse_args()

    responses = run_model_on_stimuli(args.model_path, device=args.device)
    for stim_name, response in responses.items():
        cosine_similarity = (1.0 - tu.cos_loss(response["optic_flow"], response["pred_flow"])).item()
        print(f"{stim_name}: cosine similarity = {cosine_similarity:.4f}")


if __name__ == "__main__":
    main()
