"""
Create movies comparing true vs predicted optic flow for each stimulus type.
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

import numpy as np

from analysis_utilities import (
    resolve_model_folder,
    run_model_on_stimuli,
    make_flow_comparison_movie,
)
from loss_functions import cos_loss


def _make_movie_for_stim(
    stim_name: str,
    response: dict,
    output_folder: Path,
    stim_index: int,
) -> None:
    """Extract arrays from a response dict and delegate to the shared movie maker."""
    stimulus = response["stimulus"].detach().cpu().numpy()
    true_flow = response["optic_flow"].detach().cpu().numpy()
    pred_flow = response["pred_flow"].detach().cpu().numpy()
    column_xyz = response["column_xyz"].detach().cpu().numpy()
    dt = float(response["dt"].item())

    num_stimuli = stimulus.shape[0]
    if num_stimuli == 0:
        print(f"Skipping {stim_name}: no stimuli available.")
        return
    if stim_index >= num_stimuli:
        stim_index = num_stimuli - 1

    frame_rate = max(1, int((1 / dt) / 2))
    cell_x = column_xyz[:, 0]
    cell_z = column_xyz[:, 2]

    out_path = output_folder / f"optic_flow_{stim_name}_stim_{stim_index}.mp4"

    make_flow_comparison_movie(
        stimulus[stim_index],
        true_flow[stim_index],
        pred_flow[stim_index],
        cell_x,
        cell_z,
        out_path,
        frame_rate,
        stim_name=stim_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Make optic flow movies for all stimuli.")
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="models/best_model",
        help="Path to model folder or model.pt (default: models/best_model)",
    )
    parser.add_argument(
        "--stim-index",
        type=int,
        default=5,
        help="Stimulus index to visualize per stimulus type",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Optional output folder (default: <model_folder>/optic_flow_movie_stimuli)",
    )
    args = parser.parse_args()

    stimulus_specs = [
        {
            "name": "sine_right_4hz",
            "stimulus_type": "directional_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "stim_time": 2.0,
                },
            },
            "stimulus_kwargs": {
                "sine_direction": "right",
                "sine_temporal_frequency": 4.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "sine_left_4hz",
            "stimulus_type": "directional_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "stim_time": 2.0,
                },
            },
            "stimulus_kwargs": {
                "sine_direction": "left",
                "sine_temporal_frequency": 4.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "sine_up_4hz",
            "stimulus_type": "directional_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "stim_time": 2.0,
                },
            },
            "stimulus_kwargs": {
                "sine_direction": "up",
                "sine_temporal_frequency": 4.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "sine_down_4hz",
            "stimulus_type": "directional_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "stim_time": 2.0,
                },
            },
            "stimulus_kwargs": {
                "sine_direction": "down",
                "sine_temporal_frequency": 4.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "random_sine_4s_warmup",
            "stimulus_type": "random_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "seed": 100,
                },
            },
            "stimulus_kwargs": {
                "random_sine_duration_s": 4.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "sine_const_rand",
            "stimulus_type": "sine_const_rand",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "seed": 100,
                },
            },
            "stimulus_kwargs": {
                "const_rand_const_time": 1.0,
                "const_rand_const_tf": 4.0,
                "const_rand_const_first": True,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "sintel_max",
            "stimulus_type": "sintel",
            "stimulus_kwargs": {
                "use_full_length": True,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "flow360_max",
            "stimulus_type": "flow_360",
            "stimulus_kwargs": {
                "use_full_length": True,
            },
        },
    ]

    responses = run_model_on_stimuli(args.model_path, stimulus_specs=stimulus_specs)
    model_folder = resolve_model_folder(Path(args.model_path))
    output_folder = Path(args.output_folder) if args.output_folder else model_folder / "optic_flow_movie_stimuli"

    for stim_name, response in responses.items():
        _make_movie_for_stim(stim_name, response, output_folder, args.stim_index)

    print("Cosine similarity (1 - cos_loss) per stimulus:")
    for stim_name, response in responses.items():
        cosine_sim = 1.0 - cos_loss(response["optic_flow"], response["pred_flow"])
        print(f"  {stim_name}: {cosine_sim.item():.6f}")


if __name__ == "__main__":
    main()
