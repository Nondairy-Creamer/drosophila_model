"""
Create a movie showing predicted optic flow for both eyes.

Loads the actual training inputs (cell_inputs.pt / target.pt) from a model
folder, splits into right and left eye, runs the model, and renders:

  Row 0: Left eye stimulus       |  Right eye stimulus
  Row 1: Left eye filtered       |  Right eye filtered
  Row 2: Left eye predicted flow |  Right eye predicted flow

Usage:
    python analysis/make_flow_movie_two_eyes.py models/20260226_190611_gpu8_run0/
    python analysis/make_flow_movie_two_eyes.py models/best_model --stim-index 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

import data_utilities as du
import neural_models as nm

from analysis_utilities import (
    resolve_model_paths,
    fig_to_rgb,
    angular_flow_to_cartesian_xz,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make two-eye optic flow movie for a model.")
    parser.add_argument(
        "model_path", type=str, nargs="?", default="models/best_model",
        help="Path to model folder or model.pt",
    )
    parser.add_argument("--stim-index", type=int, default=5,
                        help="Stimulus index to visualize")
    parser.add_argument("--fps", type=int, default=None,
                        help="Output frame rate (default: half the model rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to render")
    args = parser.parse_args()

    model_file, model_folder = resolve_model_paths(Path(args.model_path))
    local_config = du.load_config()
    device = local_config['model'].get('device', 'cpu')
    print(f"Loading model from {model_file} on {device}...")
    model = nm.NeuralModel.load(model_file, device=device)
    device = model.device
    dtype = model.dtype
    frame_rate = args.fps or max(1, int((1 / model.dt) / 2))

    stim_index = args.stim_index

    print("Loading inputs and targets...")
    cell_inputs_all = torch.load(model_folder / 'cell_inputs.pt', map_location=device)
    cell_inputs_all = cell_inputs_all.to(device=device, dtype=dtype)

    print(f"  cell_inputs shape: {cell_inputs_all.shape}")
    print(f"  Using stimulus index {stim_index}")

    stim_input = cell_inputs_all[stim_index:stim_index + 1]  # (1, T, 2*N)
    N = stim_input.shape[2] // 2
    right_input = stim_input[:, :, :N]   # (1, T, N)
    left_input = stim_input[:, :, N:]    # (1, T, N)

    num_frames = stim_input.shape[1]
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
        right_input = right_input[:, :num_frames]
        left_input = left_input[:, :num_frames]

    print(f"Running model prediction ({num_frames} frames, N={N} columns per eye)...")
    with torch.no_grad():
        both = torch.cat([right_input, left_input], dim=0)  # (2, T, N)
        outputs = model.predict(both)[0]
        decoder_both = outputs['decoder']  # (2, T, N, 2)
        pred_flow_right = decoder_both[0].cpu().numpy()
        pred_flow_left = decoder_both[1].cpu().numpy()

    # Filtered stimulus
    filtered_right_np = None
    filtered_left_np = None
    with torch.no_grad():
        if model.use_input_conv:
            filtered_right_np = _compute_filtered(model, right_input)
            filtered_left_np = _compute_filtered(model, left_input)

    stimulus_right = right_input[0].cpu().numpy()  # (T, N)
    stimulus_left = left_input[0].cpu().numpy()

    column_xyz = model.get_column_xyz()
    cell_x = column_xyz[:, 0]
    cell_y = column_xyz[:, 1]
    cell_z = column_xyz[:, 2]

    output_folder = model_folder / 'optic_flow_movie'
    out_path = output_folder / f'two_eye_flow_stim_{stim_index}.mp4'

    print(f"Creating movie -> {out_path}")
    _make_two_eye_flow_movie(
        stimulus_left=stimulus_left[:num_frames],
        stimulus_right=stimulus_right[:num_frames],
        pred_flow_left=pred_flow_left[:num_frames],
        pred_flow_right=pred_flow_right[:num_frames],
        cell_x=cell_x,
        cell_y=cell_y,
        cell_z=cell_z,
        output_path=out_path,
        frame_rate=frame_rate,
        filtered_left=filtered_left_np[:num_frames] if filtered_left_np is not None else None,
        filtered_right=filtered_right_np[:num_frames] if filtered_right_np is not None else None,
    )


def _compute_filtered(model, input_batch):
    """Run input conv step-by-step, return (T, N) numpy array."""
    filter_state = None
    frames = []
    for t in range(input_batch.shape[1]):
        y_t, filter_state = model.apply_input_conv(input_batch[:, t, :], filter_state)
        frames.append(y_t[0])
    return torch.stack(frames, dim=0).cpu().numpy()


def _make_two_eye_flow_movie(
    stimulus_left: np.ndarray,
    stimulus_right: np.ndarray,
    pred_flow_left: np.ndarray,
    pred_flow_right: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_z: np.ndarray,
    output_path: Path,
    frame_rate: int,
    filtered_left: np.ndarray | None = None,
    filtered_right: np.ndarray | None = None,
) -> None:
    """Render a two-eye flow movie.

    Parameters
    ----------
    stimulus_left, stimulus_right : (T, N)
    pred_flow_left, pred_flow_right : (T, N, 2) angular flow (d_right, d_up)
    cell_x, cell_y, cell_z : (N,)
    filtered_left, filtered_right : (T, N) or None
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors
    import imageio as iio

    has_filtered = filtered_left is not None
    num_rows = 2 + (1 if has_filtered else 0)
    num_frames = stimulus_left.shape[0]

    # --- Stimulus colour scale ---
    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    # --- Filtered colour scale ---
    filt_norm = None
    if has_filtered:
        filt_min = min(np.nanmin(filtered_left), np.nanmin(filtered_right))
        filt_max = max(np.nanmax(filtered_left), np.nanmax(filtered_right))
        filt_norm = mcolors.Normalize(vmin=filt_min, vmax=filt_max)

    # --- Flow projection & colour scale ---
    flow_vx_r, flow_vz_r = angular_flow_to_cartesian_xz(
        pred_flow_right[:, :, 0], pred_flow_right[:, :, 1], cell_x, cell_y, cell_z,
    )
    flow_vx_l, flow_vz_l = angular_flow_to_cartesian_xz(
        pred_flow_left[:, :, 0], pred_flow_left[:, :, 1], cell_x, cell_y, cell_z,
    )
    mag_r = np.hypot(flow_vx_r, flow_vz_r)
    mag_l = np.hypot(flow_vx_l, flow_vz_l)

    all_mag = np.concatenate([mag_r.ravel(), mag_l.ravel()])
    finite_mag = all_mag[np.isfinite(all_mag)]
    max_mag = np.percentile(finite_mag, 99) if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0
    flow_norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    flow_vx_r = np.nan_to_num(flow_vx_r, nan=0.0)
    flow_vz_r = np.nan_to_num(flow_vz_r, nan=0.0)
    flow_vx_l = np.nan_to_num(flow_vx_l, nan=0.0)
    flow_vz_l = np.nan_to_num(flow_vz_l, nan=0.0)
    mag_r = np.nan_to_num(mag_r, nan=0.0)
    mag_l = np.nan_to_num(mag_l, nan=0.0)

    # --- Figure setup ---
    fig, axes = plt.subplots(num_rows, 2, figsize=(12.8, 5.12 * num_rows), dpi=100)

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    def _setup_eye_axes(ax_left, ax_right):
        for ax, flip in [(ax_left, False), (ax_right, True)]:
            if flip:
                ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
            else:
                ax.set_xlim(cell_x.min() - x_margin, cell_x.max() + x_margin)
            ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
            ax.set_aspect("equal")

    cur_row = 0

    # Row 0: raw stimulus
    ax_stim_l, ax_stim_r = axes[cur_row, 0], axes[cur_row, 1]
    _setup_eye_axes(ax_stim_l, ax_stim_r)
    sc_stim_l = ax_stim_l.scatter(cell_x, cell_z, c=stimulus_left[0], cmap="gray",
                                   norm=stim_norm, s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(cell_x, cell_z, c=stimulus_right[0], cmap="gray",
                                   norm=stim_norm, s=90, edgecolors="none")
    ax_stim_l.set_title("Left Eye (stimulus)")
    ax_stim_r.set_title("Right Eye (stimulus)")
    fig.colorbar(sc_stim_l, ax=ax_stim_l, label="intensity")
    fig.colorbar(sc_stim_r, ax=ax_stim_r, label="intensity")
    cur_row += 1

    # Row 1 (optional): filtered stimulus
    sc_filt_l = sc_filt_r = None
    if has_filtered:
        ax_filt_l, ax_filt_r = axes[cur_row, 0], axes[cur_row, 1]
        _setup_eye_axes(ax_filt_l, ax_filt_r)
        sc_filt_l = ax_filt_l.scatter(cell_x, cell_z, c=filtered_left[0], cmap="gray",
                                       norm=filt_norm, s=90, edgecolors="none")
        sc_filt_r = ax_filt_r.scatter(cell_x, cell_z, c=filtered_right[0], cmap="gray",
                                       norm=filt_norm, s=90, edgecolors="none")
        ax_filt_l.set_title("Left Eye (filtered)")
        ax_filt_r.set_title("Right Eye (filtered)")
        fig.colorbar(sc_filt_l, ax=ax_filt_l, label="intensity")
        fig.colorbar(sc_filt_r, ax=ax_filt_r, label="intensity")
        cur_row += 1

    # Last row: predicted flow
    ax_flow_l, ax_flow_r = axes[cur_row, 0], axes[cur_row, 1]
    _setup_eye_axes(ax_flow_l, ax_flow_r)
    quiver_kw = dict(
        cmap="viridis", norm=flow_norm, angles="xy",
        scale_units="xy", scale=1 / arrow_scale, width=0.003,
    )
    q_left = ax_flow_l.quiver(cell_x, cell_z, flow_vx_l[0], flow_vz_l[0], mag_l[0], **quiver_kw)
    q_right = ax_flow_r.quiver(cell_x, cell_z, flow_vx_r[0], flow_vz_r[0], mag_r[0], **quiver_kw)
    ax_flow_l.set_title("Left Eye (predicted flow)")
    ax_flow_r.set_title("Right Eye (predicted flow)")
    fig.colorbar(q_left, ax=ax_flow_l, label="magnitude")
    fig.colorbar(q_right, ax=ax_flow_r, label="magnitude")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {num_frames} frames to {output_path}...")

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_frames):
            sc_stim_l.set_array(stimulus_left[t])
            sc_stim_r.set_array(stimulus_right[t])

            if sc_filt_l is not None:
                sc_filt_l.set_array(filtered_left[t])
                sc_filt_r.set_array(filtered_right[t])

            q_left.set_UVC(flow_vx_l[t], flow_vz_l[t], mag_l[t])
            q_right.set_UVC(flow_vx_r[t], flow_vz_r[t], mag_r[t])

            title = f"frame {t + 1}/{num_frames}"
            axes[0, 0].set_title(f"Left Eye (stimulus) {title}")
            axes[0, 1].set_title(f"Right Eye (stimulus) {title}")

            writer.append_data(fig_to_rgb(fig))
            if (t + 1) % 50 == 0:
                print(f"  Frame {t + 1}/{num_frames}")

    plt.close(fig)
    print(f"Done! Movie saved to {output_path}")


if __name__ == "__main__":
    main()
