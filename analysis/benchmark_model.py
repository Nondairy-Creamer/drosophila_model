"""
Benchmark optic flow predictions against stimuli.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
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
    parser.add_argument(
        "--num-stim",
        type=int,
        default=None,
        help="Max number of stimuli per type (default: use all)",
    )
    args = parser.parse_args()

    responses = run_model_on_stimuli(
        args.model_path, device=args.device, max_stimuli=args.num_stim,
    )

    for stim_name, response in responses.items():
        true_flow = response["optic_flow"]
        pred_flow = response["pred_flow"]
        num_stim = true_flow.shape[0]

        per_stim_sims: list[float] = []
        for i in range(num_stim):
            sim = (1.0 - tu.cos_loss(true_flow[i], pred_flow[i])).item()
            per_stim_sims.append(sim)

        mean_sim = sum(per_stim_sims) / len(per_stim_sims)
        sims_tensor = torch.tensor(per_stim_sims)
        std_sim = sims_tensor.std().item()

        print(f"\n{stim_name} ({num_stim} stimuli):")
        for i, sim in enumerate(per_stim_sims):
            print(f"  stim {i:3d}: {sim:.4f}")
        print(f"  mean = {mean_sim:.4f}  std = {std_sim:.4f}")


if __name__ == "__main__":
    main()
