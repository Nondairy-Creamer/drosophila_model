"""
Evaluate a model's MSE and cosine loss on the stimulus defined in its config.

Usage:
    python analysis/get_model_score.py <model_path> [--stim-type <type>]
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import data_utilities as du
import neural_models as nm
import stimulus_utilities as su
from loss_functions import mse_loss, cos_loss
from analysis_utilities import predict_optic_flow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MSE and cosine loss for a trained model."
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt",
    )
    parser.add_argument(
        "--stim-type",
        type=str,
        default=None,
        help="Stimulus type override (default: use value from model config)",
    )
    args = parser.parse_args()

    # --- Load model and config ---
    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    local_config = du.load_config()
    model = nm.NeuralModel.load(model_path)

    stimulus_type = args.stim_type if args.stim_type else local_config["stimulus"]["stimulus_type"]
    column_phi_theta = model.column_phi_theta

    # --- Create stimulus ---
    print(f"Generating stimulus (type: {stimulus_type})...")
    stimulus, optic_flow, _ = su.get_stimulus_one_eye(
        stimulus_type,
        local_config,
        column_phi_theta,
    )

    if local_config['training']['batch_size'] is not None:
        print('Using batch size', local_config['training']['batch_size'])
        stimulus = stimulus[:local_config['training']['batch_size'], :, :]
        optic_flow = optic_flow[:local_config['training']['batch_size'], :, :]

    print(f"  Stimulus shape: {stimulus.shape}")
    print(f"  Optic flow shape: {optic_flow.shape}")

    # --- Run model ---
    print("Running model...")
    pred_flow = predict_optic_flow(model, stimulus)

    # --- Compute losses ---
    mse = mse_loss(optic_flow, pred_flow)
    cosine = cos_loss(optic_flow, pred_flow)

    print(f"\nResults ({stimulus_type}):")
    print(f"  MSE loss:    {mse.item():.6f}")
    print(f"  Cosine similarity: {1 - cosine.item():.6f}")


if __name__ == "__main__":
    main()
