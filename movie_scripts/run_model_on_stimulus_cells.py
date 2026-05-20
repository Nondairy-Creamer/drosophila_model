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


def _select_stim(stim, stim_index, num_stim):
    if stim_index is None:
        return stim[:num_stim]
    bad = [i for i in stim_index if i < 0 or i >= stim.shape[0]]
    if bad:
        raise IndexError(
            f"--stim-index {bad} out of range for stimulus with {stim.shape[0]} entries"
        )
    return stim[stim_index]


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
                        help="Number of stimuli to use per direction (ignored if --stim-index is set)")
    parser.add_argument("--stim-index", type=int, nargs="+", default=None,
                        help="Specific stimulus indices to render (overrides --num-stim). "
                             "For directional_sine, indices apply within each direction.")
    parser.add_argument("--color-percentile", type=float, default=99,
                        help="Percentile for normalizing cell responses")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model_folder>/movie)")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    config = du.load_config(args.config) if args.config else du.load_config()
    model = nm.BaseRnn.load(model_path)

    device = model.device
    dtype = model.dtype
    model_dt = model.dt
    column_phi_theta = model.column_phi_theta

    # Force stimulus generation to use the model's dt and device so the resampled
    # stimulus rate matches the movie playback rate, and device matches the model
    # (config can specify a cuda device that doesn't exist on this machine).
    cfg_dt = float(config['model']['dt'])
    if abs(cfg_dt - model_dt) > 1e-9:
        print(f"Overriding config dt {cfg_dt} -> model dt {model_dt} for stimulus generation")
        config['model']['dt'] = model_dt
    cfg_device = str(config['model'].get('device', ''))
    if cfg_device != str(device):
        print(f"Overriding config device {cfg_device!r} -> model device {str(device)!r}")
        config['model']['device'] = str(device)

    frame_rate = int(round(1 / model_dt))
    print(f"Movie frame rate: {frame_rate} fps (model_dt={model_dt})")

    output_dir = Path(args.output_dir) if args.output_dir else model_folder / "movie"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stim_type == 'directional_sine':
        directions = args.sine_direction or ['right', 'left', 'up', 'down']
        stim_parts, mask_parts = [], []
        for d in directions:
            print(f"Generating stimulus (type={args.stim_type}, direction={d})...")
            s, _, m = su.get_stimulus_one_eye(
                args.stim_type, config, column_phi_theta, sine_direction=d)
            stim_parts.append(_select_stim(s, args.stim_index, args.num_stim))
            mask_parts.append(_select_stim(m, args.stim_index, args.num_stim))
        stimulus = torch.cat(stim_parts, dim=0).to(device=device, dtype=dtype)
        train_mask = torch.cat(mask_parts, dim=0)
    else:
        print(f"Generating stimulus (type={args.stim_type})...")
        stimulus, _, train_mask = su.get_stimulus_one_eye(
            args.stim_type, config, column_phi_theta)
        stimulus = _select_stim(stimulus, args.stim_index, args.num_stim).to(
            device=device, dtype=dtype)
        train_mask = _select_stim(train_mask, args.stim_index, args.num_stim)

    t5_types = ['T5a', 'T5b', 'T5c', 'T5d']
    output_neurons = list(set(args.input_neurons + args.cell_types + t5_types))

    print(f"Running model prediction ({stimulus.shape[0]} stimuli, "
          f"{stimulus.shape[1]} frames each)...")
    with torch.no_grad():
        # First pass: throw-away run used only to obtain a warmed-up state
        # (final voltages + input-conv filter state) at the end of the stimulus.
        warmup = model.predict(
            stimulus,
            input_neurons=args.input_neurons,
            output_neurons=output_neurons,
        )
        init_v = warmup['model_state']['v']
        init_filter_state = warmup['model_state']['filter_state']
        del warmup

        # Second pass: re-run with the warmed-up state as the initial condition;
        # this is what we render in the movie.
        print("Re-running with warmed-up initial state for the rendered movie...")
        pred = model.predict(
            stimulus,
            input_neurons=args.input_neurons,
            output_neurons=output_neurons,
            init_state=init_v,
            init_filter_state=init_filter_state,
        )['outputs']['firing_rate']

    cell_phi = model.column_phi_theta[:, 0]
    cell_theta = model.column_phi_theta[:, 1]
    cell_x = -np.sin(cell_theta) * np.cos(cell_phi)
    cell_z = np.cos(cell_theta)

    stim_np = stimulus.detach().cpu().numpy()
    # Mask out-of-FOV columns to NaN so the hex-mosaic factory drops them
    # (and per-frame normalization below uses only in-FOV columns).
    train_mask_np = train_mask.detach().cpu().numpy().astype(bool)
    stim_np = np.where(train_mask_np, stim_np, np.nan)
    # Per-frame min-max normalize so each frame spans the full data range
    # (the diamond scatter cmap uses Normalize(0, 1), so we map to [0, 1]).
    frame_min = np.nanmin(stim_np, axis=-1, keepdims=True)
    frame_max = np.nanmax(stim_np, axis=-1, keepdims=True)
    frame_range = frame_max - frame_min
    safe_range = np.where(frame_range > 0, frame_range, 1.0)
    stim_np = (stim_np - frame_min) / safe_range
    diamond_scale = 1.8
    diamond_offset = 2.0 * diamond_scale + 0.4  # leave a small gap between center stim and panels

    sum_types = [(t4, t5) for t4, t5 in zip(args.cell_types, t5_types)]
    sum_labels = [f"{t4}+{t5}" for t4, t5 in sum_types]
    sum_save_name = "t4_plus_t5"

    groups = [
        (args.cell_types, args.cell_types, "_".join(args.cell_types).lower()),
        (t5_types, t5_types, "_".join(t5_types).lower()),
        (sum_labels, sum_types, sum_save_name),
    ]

    for label_list, source_list, save_name in groups:
        seq_data = {}
        for label, src in zip(label_list, source_list):
            if isinstance(src, tuple):
                cell_resp = sum(pred[s].detach().cpu().numpy() for s in src)
            else:
                cell_resp = pred[src].detach().cpu().numpy()
            cell_resp /= np.nanpercentile(cell_resp, args.color_percentile)
            cell_resp = interp_nan_column(cell_resp, cell_phi, cell_theta)
            seq_data[label] = cell_resp
        seq_data['stimulus'] = stim_np

        print(f"Creating diamond sequence movie for {' '.join(label_list)}...")
        init_scatter_diamond = make_scatter_diamond_sequence_factory(
            seq_data, cell_x, cell_z,
            offset=diamond_offset, scale=diamond_scale, size_scale=0.6,
            stim_cmap='gray', stim_scale_frac=1.0, stim_size_frac=1.0,
            stim_p_q=model.column_p_q)
        make_movie(init_scatter_diamond, ['sequence'], output_dir, frame_rate,
                   save_name=save_name)

    print("Done!")


if __name__ == "__main__":
    main()
