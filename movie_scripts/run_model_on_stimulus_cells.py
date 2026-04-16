"""
Run a trained model on a freshly generated one-eye stimulus and visualize
cell responses in a diamond layout with the stimulus in the center.

Given a set of 4 cell types, produces a movie with:
  - Center: stimulus intensity (small scatter)
  - Diamond positions: cell responses (right, left, up, down)

Uses the same model loading and stimulus generation as run_model_on_stimulus.py.

Usage:
    python movie_scripts/run_model_on_stimulus_cells.py models/my_model/
    python movie_scripts/run_model_on_stimulus_cells.py models/my_model/ --cell-types T5a T5b T5c T5d
    python movie_scripts/run_model_on_stimulus_cells.py models/my_model/ --stim-type random_sine --config path/to/config.toml
"""
import argparse
import sys
from pathlib import Path

_repo_root = str(Path(__file__).parent.parent)
sys.path.insert(0, _repo_root)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(_repo_root) / "analysis"))

import numpy as np
import torch

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su
from analysis_utilities import interp_nan_column

from movie_utilities import (
    make_scatter_diamond_sequence_factory,
    make_movie,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model on a stimulus and visualize cell responses in diamond layout."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.toml (default: uses workspace config.toml)")
    parser.add_argument("--cell-types", type=str, nargs=4, default=['T4a', 'T4b', 'T4c', 'T4d'],
                        help="Four cell types to plot in diamond (right, left, up, down)")
    parser.add_argument("--stim-type", type=str, default="random_sine",
                        help="Stimulus type passed to get_stimulus_one_eye")
    parser.add_argument("--sine-direction", type=str, nargs="+", default=None,
                        choices=["right", "left", "up", "down"],
                        help="Direction(s) for directional_sine stimulus type (default: all four)")
    parser.add_argument("--input-neurons", type=str, nargs="+", default=['L1', 'L2', 'L3'],
                        help="Input neuron types for model.predict")
    parser.add_argument("--num-stim", type=int, default=4,
                        help="Number of stimuli to use per direction")
    parser.add_argument("--color-percentile", type=float, default=99,
                        help="Percentile for normalizing cell responses")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model_folder>/movie)")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    config = du.load_config(args.config) if args.config else du.load_config()
    model = nm.NeuralModel.load(model_path)

    device = model.device
    dtype = model.dtype
    model_dt = model.dt
    column_phi_theta = model.column_phi_theta

    frame_rate = int(1 / model_dt)

    output_dir = Path(args.output_dir) if args.output_dir else model_folder / "movie"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stim_type == 'directional_sine':
        directions = args.sine_direction or ['right', 'left', 'up', 'down']
        stim_parts = []
        for d in directions:
            print(f"Generating stimulus (type={args.stim_type}, direction={d})...")
            s, _, _ = su.get_stimulus_one_eye(
                args.stim_type, config, column_phi_theta, sine_direction=d)
            stim_parts.append(s[:args.num_stim])
        stimulus = torch.cat(stim_parts, dim=0).to(device=device, dtype=dtype)
    else:
        print(f"Generating stimulus (type={args.stim_type})...")
        stimulus, _, _ = su.get_stimulus_one_eye(
            args.stim_type, config, column_phi_theta)
        stimulus = stimulus[:args.num_stim].to(device=device, dtype=dtype)

    output_neurons = list(set(args.input_neurons + args.cell_types))

    print(f"Running model prediction ({stimulus.shape[0]} stimuli, "
          f"{stimulus.shape[1]} frames each)...")
    with torch.no_grad():
        pred = model.predict(
            stimulus,
            input_neurons=args.input_neurons,
            output_neurons=output_neurons,
        )['outputs']['firing_rate']

    cell_phi = model.column_phi_theta[:, 0]
    cell_theta = model.column_phi_theta[:, 1]
    cell_x = -np.sin(cell_theta) * np.cos(cell_phi)
    cell_z = np.cos(cell_theta)

    seq_data = {}
    for mct in args.cell_types:
        cell_resp = pred[mct].detach().cpu().numpy()
        cell_resp /= np.nanpercentile(cell_resp, args.color_percentile)
        cell_resp = interp_nan_column(cell_resp, cell_phi, cell_theta)
        seq_data[mct] = cell_resp

    seq_data['stimulus'] = stimulus.detach().cpu().numpy()

    cell_label = "_".join(args.cell_types)
    print(f"Creating diamond sequence movie for {cell_label}...")
    init_scatter_diamond = make_scatter_diamond_sequence_factory(
        seq_data, cell_x, cell_z, scale=1.8, size_scale=0.6)
    make_movie(init_scatter_diamond, ['sequence'], output_dir, frame_rate,
               save_name=cell_label.lower())

    print("Done!")


if __name__ == "__main__":
    main()
