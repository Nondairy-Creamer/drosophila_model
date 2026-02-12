"""
Create a movie comparing true vs predicted optic flow.
2x2 layout:
  - Top left: Stimulus intensity (scatter plot)
  - Top right: True optic flow (quiver plot)
  - Bottom left: Blank (reserved)
  - Bottom right: Predicted optic flow (quiver plot)
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

import data_utilities as du
import neural_models as nm

from analysis_utilities import (
    resolve_model_paths,
    make_flow_comparison_movie,
)


def _extract_optic_flow(target_loaded: object) -> torch.Tensor:
    if torch.is_tensor(target_loaded):
        return target_loaded

    if isinstance(target_loaded, dict):
        for key in ('decoder', 'optic_flow', 'flow'):
            if key in target_loaded:
                value = target_loaded[key]
                if torch.is_tensor(value):
                    return value
        raise ValueError(
            "Target dict does not contain a tensor under keys: decoder, optic_flow, flow."
        )

    raise TypeError("Unsupported target format. Expected tensor or dict of tensors.")


def _extract_pred_flow(pred_output: object) -> torch.Tensor:
    if torch.is_tensor(pred_output):
        return pred_output

    if isinstance(pred_output, tuple):
        pred_output = pred_output[0]

    if isinstance(pred_output, dict):
        if 'decoder' in pred_output:
            return pred_output['decoder']
        raise ValueError("Predict output dict missing 'decoder' key.")

    raise TypeError("Unsupported predict output format.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make optic flow movie for a model.")
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="models/best_model",
        help="Path to model folder or model.pt",
    )
    parser.add_argument("--stim-index", type=int, default=5, help="Stimulus index to visualize")
    args = parser.parse_args()

    model_file, model_folder = resolve_model_paths(Path(args.model_path))
    config = du.load_config(model_folder / 'config.toml')
    device = config['model'].get('device', 'cpu')
    print(f"Loading model on {device}...")
    model = nm.NeuralModel.load(model_file, device=device)
    device = model.device
    dtype = model.dtype
    frame_rate = max(1, int((1 / model.dt) / 2))

    print("Loading inputs and targets...")
    cell_inputs = torch.load(model_folder / 'cell_inputs.pt', map_location=device)
    cell_inputs = cell_inputs.to(device=device, dtype=dtype)
    target_loaded = torch.load(model_folder / 'target.pt', map_location=device)
    true_flow = _extract_optic_flow(target_loaded).to(device=device, dtype=dtype)

    print("Running model prediction...")
    with torch.no_grad():
        pred_output = model.predict(cell_inputs)
    pred_flow = _extract_pred_flow(pred_output).to(device=device, dtype=dtype)

    from training_utilities import cos_loss
    print("Cosine similarity", 1 - cos_loss(true_flow, pred_flow))

    # Convert to numpy for plotting
    stimulus = cell_inputs.detach().cpu().numpy()
    true_flow_np = true_flow.detach().cpu().numpy()
    pred_flow_np = pred_flow.detach().cpu().numpy()

    print(f"Stimulus shape: {stimulus.shape}")
    print(f"True flow shape: {true_flow_np.shape}")
    print(f"Pred flow shape: {pred_flow_np.shape}")

    # Use model column locations projected to 3D (radius=1)
    column_xyz = model.get_column_xyz()
    cell_x = column_xyz[:, 0]
    cell_y = column_xyz[:, 1]
    cell_z = column_xyz[:, 2]

    stim_index = args.stim_index
    print(f"Creating movie for stimulus {stim_index} with {stimulus.shape[1]} frames...")

    output_folder = model_folder / 'optic_flow_movie'
    out_path = output_folder / f'optic_flow_comparison_stim_{stim_index}.mp4'

    make_flow_comparison_movie(
        stimulus[stim_index],
        true_flow_np[stim_index],
        pred_flow_np[stim_index],
        cell_x,
        cell_y,
        cell_z,
        out_path,
        frame_rate,
    )


if __name__ == "__main__":
    main()
