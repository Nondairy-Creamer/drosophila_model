"""
Create movies comparing true vs predicted optic flow for each stimulus type.
2x2 layout:
  - Top left: Stimulus intensity (scatter plot)
  - Top right: True optic flow (quiver plot)
  - Bottom left: Filtered stimulus (after input conv)
  - Bottom right: Predicted optic flow (quiver plot)
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from analysis_utilities import (
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

    filtered_stimulus = None
    if "filtered_stimulus" in response:
        filtered_stimulus = response["filtered_stimulus"].detach().cpu().numpy()

    num_stimuli = stimulus.shape[0]
    if num_stimuli == 0:
        print(f"Skipping {stim_name}: no stimuli available.")
        return
    if stim_index >= num_stimuli:
        stim_index = num_stimuli - 1

    frame_rate = max(1, int((1 / dt) / 2))
    cell_x = column_xyz[:, 0]
    cell_y = column_xyz[:, 1]
    cell_z = column_xyz[:, 2]

    out_path = output_folder / f"optic_flow_{stim_name}_stim_{stim_index}.mp4"

    filt_stim = filtered_stimulus[stim_index] if filtered_stimulus is not None else None

    make_flow_comparison_movie(
        stimulus[stim_index],
        true_flow[stim_index],
        pred_flow[stim_index],
        cell_x,
        cell_y,
        cell_z,
        out_path,
        frame_rate,
        stim_name=stim_name,
        filtered_stimulus=filt_stim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Make optic flow movies for all stimuli.")
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt",
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
        # {
        #     "name": "sine_right_4hz",
        #     "stimulus_type": "directional_sine",
        #     "column_info": "column_phi_theta",
        #     "config_overrides": {
        #         "stimulus": {
        #             "stim_time": 10.0,
        #         },
        #     },
        #     "stimulus_kwargs": {
        #         "sine_direction": "right",
        #         "sine_temporal_frequency": 4.0,
        #         "pre_stim_time_override": 1.0,
        #     },
        # },
        # {
        #     "name": "sine_left_4hz",
        #     "stimulus_type": "directional_sine",
        #     "column_info": "column_phi_theta",
        #     "config_overrides": {
        #         "stimulus": {
        #             "stim_time": 10.0,
        #         },
        #     },
        #     "stimulus_kwargs": {
        #         "sine_direction": "left",
        #         "sine_temporal_frequency": 4.0,
        #         "pre_stim_time_override": 2.0,
        #     },
        # },
        # {
        #     "name": "sine_up_4hz",
        #     "stimulus_type": "directional_sine",
        #     "column_info": "column_phi_theta",
        #     "config_overrides": {
        #         "stimulus": {
        #             "stim_time": 10.0,
        #         },
        #     },
        #     "stimulus_kwargs": {
        #         "sine_direction": "up",
        #         "sine_temporal_frequency": 4.0,
        #         "pre_stim_time_override": 2.0,
        #     },
        # },
        # {
        #     "name": "sine_down_4hz",
        #     "stimulus_type": "directional_sine",
        #     "column_info": "column_phi_theta",
        #     "config_overrides": {
        #         "stimulus": {
        #             "stim_time": 10.0,
        #         },
        #     },
        #     "stimulus_kwargs": {
        #         "sine_direction": "down",
        #         "sine_temporal_frequency": 4.0,
        #         "pre_stim_time_override": 2.0,
        #     },
        # },
        # {
        #     "name": "random_sine_4s_warmup",
        #     "stimulus_type": "random_sine",
        #     "column_info": "column_phi_theta",
        #     "config_overrides": {
        #         "stimulus": {
        #             "seed": 100,
        #         },
        #     },
        #     "stimulus_kwargs": {
        #         "random_sine_duration_s": 4.0,
        #         "pre_stim_time_override": 2.0,
        #     },
        # },
        {
            "name": "random_pano_4s_warmup",
            "stimulus_type": "random_pano",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "seed": 100,
                    "stim_time": 10.0,
                    "num_repeats": 1,
                },
            },
            "stimulus_kwargs": {
                "pre_stim_time_override": 2.0,
            },
        },
        # {
        #     "name": "sintel_max",
        #     "stimulus_type": "sintel",
        #     "stimulus_kwargs": {
        #         "use_full_length": True,
        #         "pre_stim_time_override": 2.0,
        #     },
        # },
        # {
        #     "name": "flow360_max",
        #     "stimulus_type": "flow_360",
        #     "stimulus_kwargs": {
        #         "use_full_length": True,
        #     },
        # },
    ]


    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    output_folder = Path(args.output_folder) if args.output_folder else model_folder / "optic_flow_movie_stimuli"

    cosine_sims: dict[str, float] = {}

    # Process one stimulus at a time to avoid holding all responses in memory.
    for spec in stimulus_specs:
        stim_name = spec["name"]
        print(f"\n--- Processing {stim_name} ---")
        responses = run_model_on_stimuli(
            args.model_folder, stimulus_specs=[spec]
        )
        response = responses[stim_name]

        _make_movie_for_stim(stim_name, response, output_folder, args.stim_index)

        cosine_sim = 1.0 - cos_loss(response["optic_flow"], response["pred_flow"])
        cosine_sims[stim_name] = cosine_sim.item()

        # Free memory before the next stimulus.
        del responses, response

    print("\nCosine similarity (1 - cos_loss) per stimulus:")
    for stim_name, sim in cosine_sims.items():
        print(f"  {stim_name}: {sim:.6f}")


if __name__ == "__main__":
    main()
