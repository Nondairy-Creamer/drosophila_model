"""
Create a movie showing raw stimulus vs filtered stimulus (after apply_input_conv).
1x2 layout:
  - Left: Raw stimulus intensity
  - Right: Filtered stimulus (after apply_input_conv)

Same as make_flow_movie.py but generates stimulus from a spec rather than
loading cell_inputs.pt.
"""
import argparse
import sys
from copy import deepcopy
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su

from analysis_utilities import (
    resolve_model_paths,
    fig_to_rgb,
    # predict_optic_flow,
    # make_flow_comparison_movie,
)
# from loss_functions import cos_loss


def make_input_comparison_movie(
    stimulus: np.ndarray,
    filtered_stimulus: np.ndarray,
    cell_x: np.ndarray,
    cell_z: np.ndarray,
    output_path: Path,
    frame_rate: int,
    stim_name: str = "",
) -> None:
    """Create a 1x2 movie: raw stimulus (left) vs filtered stimulus (right)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors
    import imageio as iio_v2

    num_time = stimulus.shape[0]

    stim_norm = mcolors.Normalize(
        vmin=np.nanmin(stimulus), vmax=np.nanmax(stimulus)
    )
    filt_norm = mcolors.Normalize(
        vmin=np.nanmin(filtered_stimulus), vmax=np.nanmax(filtered_stimulus)
    )

    fig, (ax_raw, ax_filt) = plt.subplots(1, 2, figsize=(12.8, 5.12), dpi=100)

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())
    for ax in (ax_raw, ax_filt):
        ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
        ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax.set_aspect("equal")
        ax.set_xlabel("x (anterior ← → posterior)")
        ax.set_ylabel("z (ventral ↔ dorsal)")

    gray_cmap = plt.cm.gray
    sc_raw = ax_raw.scatter(
        cell_x, cell_z,
        c=gray_cmap(stim_norm(stimulus[0])),
        s=90, edgecolors="none",
    )
    title_label = f"Raw Stimulus ({stim_name})" if stim_name else "Raw Stimulus"
    ax_raw.set_title(title_label)
    fig.colorbar(sc_raw, ax=ax_raw, label="intensity")

    sc_filt = ax_filt.scatter(
        cell_x, cell_z,
        c=gray_cmap(filt_norm(filtered_stimulus[0])),
        s=90, edgecolors="none",
    )
    ax_filt.set_title("Filtered Stimulus")
    fig.colorbar(sc_filt, ax=ax_filt, label="intensity")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing movie to {output_path}...")

    with iio_v2.get_writer(
        str(output_path), codec="libx264", fps=frame_rate
    ) as writer:
        for t in range(num_time):
            sc_raw.set_facecolors(gray_cmap(stim_norm(stimulus[t])))
            sc_filt.set_facecolors(gray_cmap(filt_norm(filtered_stimulus[t])))

            frame_title = (
                f"Raw Stimulus ({stim_name}) frame {t+1}/{num_time}"
                if stim_name
                else f"Raw Stimulus (frame {t+1}/{num_time})"
            )
            ax_raw.set_title(frame_title)
            writer.append_data(fig_to_rgb(fig))

            if (t + 1) % 20 == 0:
                print(f"  Frame {t+1}/{num_time}")

    plt.close(fig)
    print(f"Done! Movie saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make input comparison movie for a custom stimulus."
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt",
    )
    parser.add_argument(
        "--stim-index", type=int, default=5,
        help="Stimulus index to visualize",
    )
    parser.add_argument(
        "--output-folder", type=str, default=None,
        help="Optional output folder (default: <model_folder>/optic_flow_movie_stimuli)",
    )
    args = parser.parse_args()

    stimulus_spec = {
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
    }

    # --- Load model once ---
    model_pt, model_folder = resolve_model_paths(Path(args.model_folder))
    config = du.load_config(model_folder / "config.toml")
    model = nm.NeuralModel.load(model_pt, device="cpu")

    output_folder = (
        Path(args.output_folder) if args.output_folder
        else model_folder / "optic_flow_movie_stimuli"
    )

    # --- Generate stimulus ---
    stim_name = stimulus_spec["name"]
    stim_config = deepcopy(config)
    stim_config["model"]["device"] = str(model.device)
    stim_config["stimulus"]["pre_stim_time"] = 0.0
    overrides = stimulus_spec.get("config_overrides")
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(stim_config.get(key), dict):
                stim_config[key] = {**stim_config[key], **value}
            else:
                stim_config[key] = value

    column_info = stimulus_spec.get("column_info")
    if column_info == "column_phi_theta":
        column_info = model.column_phi_theta

    print(f"Generating stimulus '{stim_name}'...")
    stimulus_kwargs = stimulus_spec.get("stimulus_kwargs", {})
    stimulus_all, optic_flow_all, _ = su.get_stimulus_one_eye(
        stimulus_spec["stimulus_type"],
        stim_config,
        column_info,
        **stimulus_kwargs,
    )
    if stimulus_all.shape[0] > 1:
        stimulus_all = stimulus_all[1:]
        optic_flow_all = optic_flow_all[1:]

    stim_index = args.stim_index
    num_stimuli = stimulus_all.shape[0]
    if num_stimuli == 0:
        print(f"No stimuli available for '{stim_name}'. Exiting.")
        return
    if stim_index >= num_stimuli:
        stim_index = num_stimuli - 1
        print(f"Requested stim_index {args.stim_index} exceeds available "
              f"({num_stimuli}), using {stim_index}.")

    # Select only the single stimulus we need
    stim_input = stimulus_all[stim_index : stim_index + 1]

    # --- Compute filtered stimulus (single stimulus only) ---
    if not model.use_input_conv:
        print("Model has no input conv — nothing to compare. Exiting.")
        return

    print(f"Computing filtered stimulus for index {stim_index}...")
    with torch.no_grad():
        stim_on_device = stim_input.to(device=model.device, dtype=model.dtype)
        filter_state = None
        filtered_frames = []
        for t in range(stim_on_device.shape[1]):
            y_t, filter_state = model.apply_input_conv(
                stim_on_device[:, t, :], filter_state
            )
            filtered_frames.append(y_t[0])
        filtered_stim_np = torch.stack(filtered_frames).detach().cpu().numpy()

    # --- Convert to numpy ---
    stimulus_np = stim_input[0].detach().cpu().numpy()

    column_xyz = model.get_column_xyz()
    cell_x = column_xyz[:, 0]
    cell_z = column_xyz[:, 2]

    frame_rate = max(1, int((1 / model.dt) / 2))
    out_path = output_folder / f"input_comparison_{stim_name}_stim_{stim_index}.mp4"

    print(f"Creating movie ({stimulus_np.shape[0]} frames) -> {out_path}")

    make_input_comparison_movie(
        stimulus_np,
        filtered_stim_np,
        cell_x,
        cell_z,
        out_path,
        frame_rate,
        stim_name=stim_name,
    )

    # # --- Optic flow comparison (commented out for now) ---
    # true_flow_single = optic_flow_all[stim_index]
    # print(f"Running model prediction for stimulus {stim_index}...")
    # pred_flow = predict_optic_flow(model, stim_input)[0]
    # cosine_sim = 1.0 - cos_loss(true_flow_single, pred_flow)
    # print(f"Cosine similarity: {cosine_sim.item():.6f}")
    #
    # true_flow_np = true_flow_single.detach().cpu().numpy()
    # pred_flow_np = pred_flow.detach().cpu().numpy()
    # cell_y = column_xyz[:, 1]
    # flow_out_path = output_folder / f"optic_flow_{stim_name}_stim_{stim_index}.mp4"
    # make_flow_comparison_movie(
    #     stimulus_np, true_flow_np, pred_flow_np,
    #     cell_x, cell_y, cell_z,
    #     flow_out_path, frame_rate,
    #     stim_name=stim_name,
    #     filtered_stimulus=filtered_stim_np,
    # )


if __name__ == "__main__":
    main()
