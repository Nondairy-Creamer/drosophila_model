"""
Run a trained model on the saved cell_inputs.pt (the exact inputs the model was
trained on) and visualize predicted optic flow.

Auto-detects one-eye vs two-eye based on input width relative to the model's
column count (N columns = one-eye, 2N = two-eye).

One-eye layout (2x2):
  Top left:     Stimulus intensity
  Top right:    True optic flow
  Bottom left:  Filtered stimulus (after input conv, if available)
  Bottom right: Predicted optic flow

Two-eye layout (2x2):
  Row 0: Left stimulus        | Right stimulus
  Row 1: Left predicted flow  | Right predicted flow

To run a model on a freshly generated stimulus (random pano/sine, edges,
flyview, drone, ...) — including the fly-view ego-motion comparison — use
``run_model_on_stimulus.py`` instead.

Usage:
    python movie_scripts/run_model_on_cell_inputs.py models/my_model/
    python movie_scripts/run_model_on_cell_inputs.py models/my_model/ --stim-index 5
"""
import argparse
import sys
from pathlib import Path


import torch
import numpy as np

from drosophila_model.model_definition import neural_models as nm

from drosophila_model.analysis.movie_scripts.movie_utilities import (
    masked_rgba,
    write_movie,
    compute_filtered,
    fig_to_rgb,
    get_flow_key,
    make_flow_comparison_movie,
    prepare_eye_quiver,
    setup_eye_axis,
    init_eye_quiver,
    update_eye_quiver,
    add_eye_colorwheel,
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
                        help="Output frame rate (default: the model's base rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to render")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)
    device = model.device
    dtype = model.dtype
    frame_rate = args.fps or max(1, round(1 / model.dt))

    N = model.regions['optic_lobe'].column_phi_theta.shape[0]
    column_xyz = model.regions['optic_lobe'].get_column_xyz()
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

    output_folder = model_folder / 'figures' / 'run_model_on_cell_inputs'

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
        pred_flow = pred_output["outputs"]["optic_lobe"][flow_key][0].cpu().numpy()

    # Filtered stimulus
    filtered_stim_np = None
    with torch.no_grad():
        if model.regions['optic_lobe'].use_input_conv:
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
        from drosophila_model.training.loss_functions import cos_loss
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
        eye="right",
    )


def _run_two_eyes(model, cell_inputs_all, train_mask_all, stim_index, N,
                  cell_x, cell_y, cell_z, flow_key, frame_rate,
                  max_frames, output_folder, device, dtype):
    """Two-eye path: split the saved input, predict, and render a 2x2 movie
    (left/right stimulus over left/right predicted flow)."""
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
        right_result, left_result = model.regions['optic_lobe'].predict_two_eyes(
            right_input, left_input)
        pred_flow_right = right_result['outputs']['optic_lobe'][flow_key][0].cpu().numpy()
        pred_flow_left = left_result['outputs']['optic_lobe'][flow_key][0].cpu().numpy()

    stimulus_right = right_input[0].cpu().numpy()
    stimulus_left = left_input[0].cpu().numpy()

    out_path = output_folder / f'two_eyes_flow_stim_{stim_index}.mp4'
    print(f"Creating movie -> {out_path}")

    gray_cmap = plt.cm.gray
    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    # predict_two_eyes already flipped the predicted left flow into the shared frame.
    pred_prepared = _prepare_eye_flow(
        pred_flow_right[:num_frames], pred_flow_left[:num_frames],
        cell_x, cell_y, cell_z)
    max_mag = _flow_max_mag([pred_prepared])
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.24), dpi=100,
                             gridspec_kw={"height_ratios": [3, 3]})

    mask_l_0 = mask_left[0] if mask_left is not None else None
    mask_r_0 = mask_right[0] if mask_right is not None else None

    ax_stim_l, ax_stim_r = axes[0, 0], axes[0, 1]
    setup_eye_axis(ax_stim_l, cell_x, cell_z, "left")
    setup_eye_axis(ax_stim_r, cell_x, cell_z, "right")
    sc_stim_l = ax_stim_l.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_left[0], mask_l_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_right[0], mask_r_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    ax_stim_l.set_title("Left Eye (stimulus)")
    ax_stim_r.set_title("Right Eye (stimulus)")
    fig.colorbar(sc_stim_l, ax=ax_stim_l, label="intensity")
    fig.colorbar(sc_stim_r, ax=ax_stim_r, label="intensity")

    pred_flow_row = _init_flow_row(
        axes[1, 0], axes[1, 1], pred_prepared,
        cell_x, cell_z, max_mag, arrow_scale, "predicted flow")
    add_eye_colorwheel(fig)

    plt.tight_layout()

    def update(t):
        mask_l_t = mask_left[t] if mask_left is not None else None
        mask_r_t = mask_right[t] if mask_right is not None else None
        sc_stim_l.set_facecolors(masked_rgba(stimulus_left[t], mask_l_t, gray_cmap, stim_norm))
        sc_stim_r.set_facecolors(masked_rgba(stimulus_right[t], mask_r_t, gray_cmap, stim_norm))
        _update_flow_row(pred_flow_row, t, max_mag)
        title = f"frame {t + 1}/{num_frames}"
        ax_stim_l.set_title(f"Left Eye (stimulus) {title}")
        ax_stim_r.set_title(f"Right Eye (stimulus) {title}")

    write_movie(fig, update, num_frames, out_path, frame_rate)


def _prepare_eye_flow(flow_right, flow_left, cell_x, cell_y, cell_z):
    """Project per-eye (T, N, 2) flow into eye-view quiver components."""
    vx_r, vz_r, _, _, _ = prepare_eye_quiver(flow_right, cell_x, cell_y, cell_z, "right")
    vx_l, vz_l, _, _, _ = prepare_eye_quiver(flow_left, cell_x, cell_y, cell_z, "left")
    return {"vx_r": vx_r, "vz_r": vz_r, "vx_l": vx_l, "vz_l": vz_l}


def _flow_max_mag(prepared_list):
    """99th-percentile arrow magnitude across one or more prepared flow rows."""
    xs, zs = [], []
    for p in prepared_list:
        xs += [p["vx_r"].ravel(), p["vx_l"].ravel()]
        zs += [p["vz_r"].ravel(), p["vz_l"].ravel()]
    all_mag = np.hypot(np.concatenate(xs), np.concatenate(zs))
    finite_mag = all_mag[np.isfinite(all_mag)]
    return (float(np.percentile(finite_mag, 99))
            if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)


def _init_flow_row(ax_l, ax_r, prepared, cell_x, cell_z, max_mag, arrow_scale, label):
    """Set up a left/right eye quiver row; returns handles for per-frame updates."""
    setup_eye_axis(ax_l, cell_x, cell_z, "left")
    setup_eye_axis(ax_r, cell_x, cell_z, "right")
    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    q_l = init_eye_quiver(ax_l, cell_x, cell_z,
                          prepared["vx_l"][0], prepared["vz_l"][0], max_mag, "left", **quiver_kw)
    q_r = init_eye_quiver(ax_r, cell_x, cell_z,
                          prepared["vx_r"][0], prepared["vz_r"][0], max_mag, "right", **quiver_kw)
    ax_l.set_title(f"Left Eye ({label})")
    ax_r.set_title(f"Right Eye ({label})")
    return {"q_l": q_l, "q_r": q_r, "prepared": prepared}


def _update_flow_row(row, t, max_mag):
    """Advance a flow row (from :func:`_init_flow_row`) to frame *t*."""
    p = row["prepared"]
    update_eye_quiver(row["q_l"], p["vx_l"][t], p["vz_l"][t], max_mag, "left")
    update_eye_quiver(row["q_r"], p["vx_r"][t], p["vz_r"][t], max_mag, "right")


if __name__ == "__main__":
    main()
