"""
Run a trained model on saved cell_inputs.pt and visualize predicted optic flow.

Auto-detects one-eye vs two-eye based on input width relative to the model's
column count (N columns = one-eye, 2N = two-eye).

One-eye layout (2x2):
  Top left:     Stimulus intensity
  Top right:    True optic flow
  Bottom left:  Filtered stimulus (after input conv, if available)
  Bottom right: Predicted optic flow

Two-eye layout (Nx2):
  Row 0: Left stimulus        | Right stimulus
  Row 1: Left filtered        | Right filtered  (optional)
  Row 2: Left predicted flow  | Right predicted flow

Usage:
    python movie_scripts/run_model_on_cell_inputs.py models/my_model/
    python movie_scripts/run_model_on_cell_inputs.py models/my_model/ --stim-index 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

import neural_models as nm

from movie_utilities import (
    masked_rgba,
    setup_eye_axes,
    prepare_flow_quiver,
    write_movie,
    compute_filtered,
    fig_to_rgb,
    get_flow_key,
    make_flow_comparison_movie,
    flow_direction_colors,
    update_direction_quiver,
    add_direction_colorwheel,
)


def _extract_optic_flow(target_loaded):
    """Pull the flow tensor from the saved target file (tensor or dict)."""
    if torch.is_tensor(target_loaded):
        return target_loaded
    if isinstance(target_loaded, dict):
        for key in ('decoder', 'optic_flow', 'flow'):
            if key in target_loaded and torch.is_tensor(target_loaded[key]):
                return target_loaded[key]
        raise ValueError("Target dict missing flow tensor under keys: decoder, optic_flow, flow.")
    raise TypeError("Unsupported target format.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model on cell_inputs.pt and visualize optic flow.")
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--stim-index", type=int, default=5,
                        help="Stimulus index to visualize")
    parser.add_argument("--fps", type=int, default=None,
                        help="Output frame rate (default: half the model rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to render")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    print(f"Loading model from {model_path}...")
    model = nm.NeuralModel.load(model_path)
    device = model.device
    dtype = model.dtype
    frame_rate = args.fps or max(1, int((1 / model.dt) / 2))

    N = model.column_phi_theta.shape[0]
    column_xyz = model.get_column_xyz()
    cell_x, cell_y, cell_z = column_xyz[:, 0], column_xyz[:, 1], column_xyz[:, 2]

    flow_key = get_flow_key(model)
    if flow_key is None:
        print("WARNING: No flow readout found. Using first readout.")
        flow_key = next(iter(model._readout_modules.keys()))

    stim_index = args.stim_index

    print("Loading inputs...")
    cell_inputs_data = torch.load(model_folder / 'cell_inputs.pt', map_location=device)
    if isinstance(cell_inputs_data, dict):
        cell_inputs_all = cell_inputs_data['cell_inputs'].to(device=device, dtype=dtype)
        train_mask_all = cell_inputs_data.get('train_mask', None)
    else:
        cell_inputs_all = cell_inputs_data.to(device=device, dtype=dtype)
        train_mask_all = None

    input_width = cell_inputs_all.shape[2]
    two_eyes = (input_width == 2 * N)
    print(f"  cell_inputs shape: {cell_inputs_all.shape} — {'two-eye' if two_eyes else 'one-eye'} detected")
    print(f"  Using stimulus index {stim_index}")

    output_folder = model_folder / 'optic_flow_movie'

    if two_eyes:
        _run_two_eyes(model, cell_inputs_all, train_mask_all, stim_index, N,
                      cell_x, cell_y, cell_z, flow_key, frame_rate,
                      args.max_frames, output_folder, device, dtype)
    else:
        _run_one_eye(model, cell_inputs_all, train_mask_all, stim_index,
                     cell_x, cell_y, cell_z, flow_key, frame_rate,
                     args.max_frames, output_folder, model_folder, device, dtype)


def _run_one_eye(model, cell_inputs_all, train_mask_all, stim_index,
                 cell_x, cell_y, cell_z, flow_key, frame_rate,
                 max_frames, output_folder, model_folder, device, dtype):
    """One-eye path: use make_flow_comparison_movie from analysis_utilities."""

    stim_input = cell_inputs_all[stim_index:stim_index + 1]

    # Load true flow if available
    target_path = model_folder / 'target.pt'
    true_flow_np = None
    if target_path.exists():
        target_loaded = torch.load(target_path, map_location=device)
        true_flow_all = _extract_optic_flow(target_loaded).to(device=device, dtype=dtype)
        true_flow_np = true_flow_all[stim_index].detach().cpu().numpy()

    print(f"Running model prediction for stimulus {stim_index}...")
    with torch.no_grad():
        pred_output = model.predict(stim_input)
        pred_flow = pred_output["outputs"][flow_key][0].cpu().numpy()

    # Filtered stimulus
    filtered_stim_np = None
    with torch.no_grad():
        if model.use_input_conv:
            filtered_stim_np = compute_filtered(model, stim_input)

    stimulus = stim_input[0].detach().cpu().numpy()

    if max_frames is not None:
        stimulus = stimulus[:max_frames]
        pred_flow = pred_flow[:max_frames]
        if true_flow_np is not None:
            true_flow_np = true_flow_np[:max_frames]
        if filtered_stim_np is not None:
            filtered_stim_np = filtered_stim_np[:max_frames]

    if true_flow_np is not None:
        from loss_functions import cos_loss
        true_flow_t = torch.tensor(true_flow_np)
        pred_flow_t = torch.tensor(pred_flow)
        print("Cosine similarity", 1 - cos_loss(true_flow_t, pred_flow_t))

    out_path = output_folder / f'optic_flow_comparison_stim_{stim_index}.mp4'
    print(f"Creating movie -> {out_path}")

    make_flow_comparison_movie(
        stimulus,
        true_flow_np if true_flow_np is not None else pred_flow,
        pred_flow if true_flow_np is not None else None,
        cell_x, cell_y, cell_z,
        out_path, frame_rate,
        filtered_stimulus=filtered_stim_np,
    )


def _run_two_eyes(model, cell_inputs_all, train_mask_all, stim_index, N,
                  cell_x, cell_y, cell_z, flow_key, frame_rate,
                  max_frames, output_folder, device, dtype):
    """Two-eye path: split input, predict, render two-eye movie."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    stim_input = cell_inputs_all[stim_index:stim_index + 1]
    right_input = stim_input[:, :, :N]
    left_input = stim_input[:, :, N:]

    if train_mask_all is not None:
        stim_mask = train_mask_all[stim_index].cpu().numpy()
        mask_right = stim_mask[:, :N]
        mask_left = stim_mask[:, N:]
    else:
        mask_right = None
        mask_left = None

    num_frames = stim_input.shape[1]
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
        right_input = right_input[:, :num_frames]
        left_input = left_input[:, :num_frames]
        if mask_right is not None:
            mask_right = mask_right[:num_frames]
            mask_left = mask_left[:num_frames]

    print(f"Running model prediction ({num_frames} frames, N={N} columns per eye)...")
    with torch.no_grad():
        right_result, left_result = model.predict_two_eyes(right_input, left_input)
        pred_flow_right = right_result['outputs'][flow_key][0].cpu().numpy()
        pred_flow_left = left_result['outputs'][flow_key][0].cpu().numpy()

    filtered_right_np = None
    filtered_left_np = None
    with torch.no_grad():
        if model.use_input_conv:
            filtered_right_np = compute_filtered(model, right_input)
            filtered_left_np = compute_filtered(model, left_input)

    stimulus_right = right_input[0].cpu().numpy()
    stimulus_left = left_input[0].cpu().numpy()

    out_path = output_folder / f'two_eyes_flow_stim_{stim_index}.mp4'
    print(f"Creating movie -> {out_path}")

    has_filtered = filtered_left_np is not None
    num_rows = 2 + (1 if has_filtered else 0)

    gray_cmap = plt.cm.gray
    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    filt_norm = None
    if has_filtered:
        filt_min = min(np.nanmin(filtered_left_np), np.nanmin(filtered_right_np))
        filt_max = max(np.nanmax(filtered_left_np), np.nanmax(filtered_right_np))
        filt_norm = mcolors.Normalize(vmin=filt_min, vmax=filt_max)

    flow_vx_r, flow_vz_r, mag_r, _, _ = prepare_flow_quiver(pred_flow_right[:num_frames], cell_x, cell_y, cell_z)
    flow_vx_l, flow_vz_l, mag_l, _, _ = prepare_flow_quiver(pred_flow_left[:num_frames], cell_x, cell_y, cell_z)

    all_vx = np.concatenate([flow_vx_r.ravel(), flow_vx_l.ravel()])
    all_vz = np.concatenate([flow_vz_r.ravel(), flow_vz_l.ravel()])
    _, max_mag = flow_direction_colors(all_vx, all_vz)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    fig, axes = plt.subplots(num_rows, 2, figsize=(12.8, 5.12 * num_rows), dpi=100)

    cur_row = 0
    mask_l_0 = mask_left[0] if mask_left is not None else None
    mask_r_0 = mask_right[0] if mask_right is not None else None

    ax_stim_l, ax_stim_r = axes[cur_row, 0], axes[cur_row, 1]
    setup_eye_axes(ax_stim_l, ax_stim_r, cell_x, cell_z)
    sc_stim_l = ax_stim_l.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_left[0], mask_l_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_right[0], mask_r_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    ax_stim_l.set_title("Left Eye (stimulus)")
    ax_stim_r.set_title("Right Eye (stimulus)")
    fig.colorbar(sc_stim_l, ax=ax_stim_l, label="intensity")
    fig.colorbar(sc_stim_r, ax=ax_stim_r, label="intensity")
    cur_row += 1

    sc_filt_l = sc_filt_r = None
    if has_filtered:
        ax_filt_l, ax_filt_r = axes[cur_row, 0], axes[cur_row, 1]
        setup_eye_axes(ax_filt_l, ax_filt_r, cell_x, cell_z)
        sc_filt_l = ax_filt_l.scatter(cell_x, cell_z,
            c=masked_rgba(filtered_left_np[0], mask_l_0, gray_cmap, filt_norm), s=90, edgecolors="none")
        sc_filt_r = ax_filt_r.scatter(cell_x, cell_z,
            c=masked_rgba(filtered_right_np[0], mask_r_0, gray_cmap, filt_norm), s=90, edgecolors="none")
        ax_filt_l.set_title("Left Eye (filtered)")
        ax_filt_r.set_title("Right Eye (filtered)")
        fig.colorbar(sc_filt_l, ax=ax_filt_l, label="intensity")
        fig.colorbar(sc_filt_r, ax=ax_filt_r, label="intensity")
        cur_row += 1

    ax_flow_l, ax_flow_r = axes[cur_row, 0], axes[cur_row, 1]
    setup_eye_axes(ax_flow_l, ax_flow_r, cell_x, cell_z)
    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    colors_l0, _ = flow_direction_colors(flow_vx_l[0], flow_vz_l[0], max_mag)
    colors_r0, _ = flow_direction_colors(flow_vx_r[0], flow_vz_r[0], max_mag)
    q_left = ax_flow_l.quiver(cell_x, cell_z, flow_vx_l[0], flow_vz_l[0], **quiver_kw)
    q_left.set_facecolor(colors_l0)
    q_left.set_edgecolor(colors_l0)
    q_right = ax_flow_r.quiver(cell_x, cell_z, flow_vx_r[0], flow_vz_r[0], **quiver_kw)
    q_right.set_facecolor(colors_r0)
    q_right.set_edgecolor(colors_r0)
    ax_flow_l.set_title("Left Eye (predicted flow)")
    ax_flow_r.set_title("Right Eye (predicted flow)")
    add_direction_colorwheel(fig)

    plt.tight_layout()

    def update(t):
        mask_l_t = mask_left[t] if mask_left is not None else None
        mask_r_t = mask_right[t] if mask_right is not None else None
        sc_stim_l.set_facecolors(masked_rgba(stimulus_left[t], mask_l_t, gray_cmap, stim_norm))
        sc_stim_r.set_facecolors(masked_rgba(stimulus_right[t], mask_r_t, gray_cmap, stim_norm))
        if sc_filt_l is not None:
            sc_filt_l.set_facecolors(masked_rgba(filtered_left_np[t], mask_l_t, gray_cmap, filt_norm))
            sc_filt_r.set_facecolors(masked_rgba(filtered_right_np[t], mask_r_t, gray_cmap, filt_norm))
        update_direction_quiver(q_left, flow_vx_l[t], flow_vz_l[t], max_mag)
        update_direction_quiver(q_right, flow_vx_r[t], flow_vz_r[t], max_mag)
        title = f"frame {t + 1}/{num_frames}"
        axes[0, 0].set_title(f"Left Eye (stimulus) {title}")
        axes[0, 1].set_title(f"Right Eye (stimulus) {title}")

    write_movie(fig, update, num_frames, out_path, frame_rate)


if __name__ == "__main__":
    main()
