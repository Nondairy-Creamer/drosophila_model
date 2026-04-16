"""
Run a trained model on a freshly generated stimulus and visualize flow.

Supports both one-eye and two-eye stimuli via the --two-eyes flag.

One-eye layout (3 rows):
  Row 0: Stimulus intensity (scatter)
  Row 1: True optic flow (quiver)
  Row 2: Predicted optic flow (quiver)

Two-eye layout (3x2):
  Row 0: Left stimulus        | Right stimulus
  Row 1: Left true flow       | Right true flow
  Row 2: Left predicted flow  | Right predicted flow

When --ego-motion is set (two-eye only), uses estimate_ego_motion_from_image
and adds rotation/translation rate trailing-window plots:
  Row 0: Left stimulus        | Right stimulus
  Row 1: Left predicted flow  | Right predicted flow
  Row 2: Rotation rate        | Translation rate

Usage:
    python movie_scripts/run_model_on_stimulus.py models/my_model/ --stim-type random_sine
    python movie_scripts/run_model_on_stimulus.py models/my_model/ --two-eyes --stim-type random_pano
    python movie_scripts/run_model_on_stimulus.py models/my_model/ --two-eyes --ego-motion --stim-type random_pano
    python movie_scripts/run_model_on_stimulus.py models/my_model/ --two-eyes --ego-motion --translation
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su

from movie_utilities import (
    masked_rgba,
    setup_eye_axes,
    prepare_flow_quiver,
    write_movie,
    ylim_from_data,
    subtract_rotation_flow,
    fig_to_rgb,
    get_flow_key,
    flow_direction_colors,
    update_direction_quiver,
    add_direction_colorwheel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model on a freshly generated stimulus and visualize flow."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.toml (default: uses workspace config.toml)")
    parser.add_argument("--two-eyes", action="store_true",
                        help="Use two-eye stimulus and predict_two_eyes")
    parser.add_argument("--stim-type", type=str, default=None,
                        help="Stimulus type (default: random_sine for 1-eye, random_pano for 2-eye)")
    parser.add_argument("--sine-direction", type=str, nargs="+", default=None,
                        choices=["right", "left", "up", "down"],
                        help="Direction(s) for directional_sine (default: all four)")
    parser.add_argument("--stim-index", type=int, nargs="+", default=[0],
                        help="Stimulus indices to visualize (space-separated)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output movie frame rate (default: half the model rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to render")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model_folder>/stimulus_flow_movie)")
    # Ego-motion features (from run_model_on_pano)
    parser.add_argument("--ego-motion", action="store_true",
                        help="Use estimate_ego_motion_from_image (two-eye only) and show rate plots")
    parser.add_argument("--translation", action="store_true",
                        help="Subtract rotation-induced flow (requires --ego-motion)")
    parser.add_argument("--save-figures", action="store_true",
                        help="Save rate and integrated pose PNG figures (requires --ego-motion)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for pano stimulus generation")
    args = parser.parse_args()

    if args.stim_type is None:
        args.stim_type = "random_pano" if args.two_eyes else "random_sine"

    if args.ego_motion and not args.two_eyes:
        parser.error("--ego-motion requires --two-eyes")

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    config = du.load_config(args.config) if args.config else du.load_config()
    model = nm.NeuralModel.load(model_path)

    device = model.device
    dtype = model.dtype
    model_dt = model.dt
    column_phi_theta = model.column_phi_theta
    N = column_phi_theta.shape[0]

    frame_rate = args.fps or max(1, int((1 / model_dt) / 2))

    column_xyz = model.get_column_xyz()
    cell_x, cell_y, cell_z = column_xyz[:, 0], column_xyz[:, 1], column_xyz[:, 2]

    flow_key = get_flow_key(model)
    if flow_key is None:
        print("WARNING: No flow readout found in model. Using first readout.")
        flow_key = next(iter(model._readout_modules.keys()))

    output_dir = Path(args.output_dir) if args.output_dir else model_folder / "stimulus_flow_movie"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.two_eyes and args.ego_motion and args.stim_type == "random_pano":
        _run_ego_motion_pano(args, model, config, model_folder, output_dir,
                             device, dtype, model_dt, column_phi_theta, N,
                             cell_x, cell_y, cell_z, frame_rate, flow_key)
    elif args.two_eyes:
        _run_two_eyes(args, model, config, output_dir,
                      device, dtype, model_dt, column_phi_theta, N,
                      cell_x, cell_y, cell_z, frame_rate, flow_key)
    else:
        _run_one_eye(args, model, config, output_dir,
                     device, dtype, model_dt, column_phi_theta,
                     cell_x, cell_y, cell_z, frame_rate, flow_key)


# ---------------------------------------------------------------------------
# One-eye path
# ---------------------------------------------------------------------------

def _run_one_eye(args, model, config, output_dir,
                 device, dtype, model_dt, column_phi_theta,
                 cell_x, cell_y, cell_z, frame_rate, flow_key):
    if args.stim_type == 'directional_sine':
        directions = args.sine_direction or ['right', 'left', 'up', 'down']
        stim_parts, flow_parts, mask_parts = [], [], []
        for d in directions:
            print(f"Generating stimulus (type={args.stim_type}, direction={d})...")
            s, f, m = su.get_stimulus_one_eye(
                args.stim_type, config, column_phi_theta, sine_direction=d)
            stim_parts.append(s)
            flow_parts.append(f)
            mask_parts.append(m)

        n_per_dir = stim_parts[0].shape[0]

        for stim_index in args.stim_index:
            if stim_index >= n_per_dir:
                print(f"Skipping stim {stim_index}: only {n_per_dir} examples per direction")
                continue

            batch_stim = torch.stack(
                [sp[stim_index] for sp in stim_parts]).to(device=device, dtype=dtype)
            batch_true = np.stack(
                [fp[stim_index].cpu().numpy() for fp in flow_parts])
            batch_mask = np.stack(
                [mp[stim_index].cpu().numpy() for mp in mask_parts])

            num_frames = batch_stim.shape[1]
            if args.max_frames is not None:
                num_frames = min(num_frames, args.max_frames)
                batch_stim = batch_stim[:, :num_frames]
                batch_true = batch_true[:, :num_frames]
                batch_mask = batch_mask[:, :num_frames]

            print(f"\nRunning model on stimulus {stim_index}, "
                  f"{len(directions)} directions ({num_frames} frames)...")
            with torch.no_grad():
                result = model.predict(batch_stim)
                batch_pred = result["outputs"][flow_key].cpu().numpy()[:, :num_frames]

            out_path = output_dir / f"directional_sine_stim_{stim_index}.mp4"
            print(f"  Creating movie -> {out_path}")
            _make_one_eye_directional_movie(
                stimuli=batch_stim.cpu().numpy(),
                true_flows=batch_true,
                pred_flows=batch_pred,
                directions=directions,
                cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
                output_path=out_path, frame_rate=frame_rate,
                masks=batch_mask,
            )
        return

    print(f"Generating stimulus (type={args.stim_type})...")
    stimulus, optic_flow, train_mask = su.get_stimulus_one_eye(
        args.stim_type, config, column_phi_theta,
    )
    stimulus_np = stimulus.cpu().numpy()
    optic_flow_np = optic_flow.cpu().numpy()
    train_mask_np = train_mask.cpu().numpy()
    n_examples = stimulus_np.shape[0]
    print(f"  Generated {n_examples} stimulus examples, shape {stimulus_np.shape}")

    for stim_index in args.stim_index:
        if stim_index >= n_examples:
            print(f"Skipping stim {stim_index}: only {n_examples} examples available")
            continue

        stim_input = stimulus[stim_index:stim_index + 1].to(device=device, dtype=dtype)
        true_flow = optic_flow_np[stim_index]
        mask = train_mask_np[stim_index]

        num_frames = stim_input.shape[1]
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)
            stim_input = stim_input[:, :num_frames]
            true_flow = true_flow[:num_frames]
            mask = mask[:num_frames]

        print(f"\nRunning model on stimulus {stim_index} ({num_frames} frames)...")
        with torch.no_grad():
            result = model.predict(stim_input)
            pred_flow = result["outputs"][flow_key][0].cpu().numpy()

        pred_flow = pred_flow[:num_frames]
        stim_np = stimulus_np[stim_index, :num_frames]

        out_path = output_dir / f"{args.stim_type}_stim_{stim_index}.mp4"
        print(f"  Creating movie -> {out_path}")
        _make_one_eye_flow_movie(
            stimulus=stim_np, true_flow=true_flow, pred_flow=pred_flow,
            cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
            output_path=out_path, frame_rate=frame_rate,
            mask=mask, stim_type=args.stim_type,
        )


# ---------------------------------------------------------------------------
# Two-eye path
# ---------------------------------------------------------------------------

def _run_two_eyes(args, model, config, output_dir,
                  device, dtype, model_dt, column_phi_theta, N,
                  cell_x, cell_y, cell_z, frame_rate, flow_key):
    print(f"Generating two-eye stimulus (type={args.stim_type})...")
    stimulus, optic_flow, observer_rotation, observer_translation, train_mask = (
        su.get_stimulus_two_eyes(args.stim_type, config, column_phi_theta)
    )
    stimulus_np = stimulus.cpu().numpy()
    optic_flow_np = optic_flow.cpu().numpy()
    observer_rotation_np = observer_rotation.cpu().numpy()
    train_mask_np = train_mask.cpu().numpy()
    n_examples = stimulus_np.shape[0]
    print(f"  Generated {n_examples} stimulus examples, shape {stimulus_np.shape}")

    pre_stim_steps = int(config["stimulus"].get("pre_stim_time", 0) / model_dt)
    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
    phi_right_t = torch.tensor(phi, dtype=torch.float32)
    theta_right_t = torch.tensor(theta, dtype=torch.float32)
    phi_left_t = torch.tensor(-phi, dtype=torch.float32)

    for stim_index in args.stim_index:
        if stim_index >= n_examples:
            print(f"Skipping stim {stim_index}: only {n_examples} examples available")
            continue

        stim_right = stimulus_np[stim_index, :, :N]
        stim_left = stimulus_np[stim_index, :, N:]
        flow_right = optic_flow_np[stim_index, :, :N, :]
        flow_left = optic_flow_np[stim_index, :, N:, :]
        mask_right = train_mask_np[stim_index, :, :N]
        mask_left = train_mask_np[stim_index, :, N:]

        if np.all(np.isnan(flow_right[pre_stim_steps:])):
            print(f"  Stim {stim_index}: ground-truth flow unavailable — "
                  "synthesizing from odometry (Z=1)")
            from analysis.analysis_utilities import generate_optic_flow
            observer_translation_np = observer_translation.cpu().numpy()
            omega_t = torch.tensor(
                observer_rotation_np[stim_index] * np.pi / 180.0, dtype=torch.float32)
            trans_t = torch.tensor(
                observer_translation_np[stim_index], dtype=torch.float32)

            pred_ur_r, pred_uu_r = generate_optic_flow(phi_right_t, theta_right_t, omega_t, trans_t, 1.0)
            pred_ur_l, pred_uu_l = generate_optic_flow(phi_left_t, theta_right_t, omega_t, trans_t, 1.0)
            flow_right = np.stack([pred_ur_r.numpy(), pred_uu_r.numpy()], axis=-1) * (180.0 / np.pi)
            flow_left = np.stack([pred_ur_l.numpy(), pred_uu_l.numpy()], axis=-1) * (180.0 / np.pi)
            flow_label = "Flow (from odometry, Z=1)"
        else:
            flow_label = "True Flow"

        num_frames = stim_right.shape[0]
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)
            stim_right = stim_right[:num_frames]
            stim_left = stim_left[:num_frames]
            flow_right = flow_right[:num_frames]
            flow_left = flow_left[:num_frames]
            mask_right = mask_right[:num_frames]
            mask_left = mask_left[:num_frames]

        right_input = torch.tensor(stim_right, device=device, dtype=dtype).unsqueeze(0)
        left_input = torch.tensor(stim_left, device=device, dtype=dtype).unsqueeze(0)

        print(f"\nRunning model on stimulus {stim_index} ({num_frames} frames, N={N})...")
        with torch.no_grad():
            right_result, left_result = model.predict_two_eyes(right_input, left_input)

        pred_flow_right = right_result["outputs"][flow_key][0].cpu().numpy()[:num_frames]
        pred_flow_left = left_result["outputs"][flow_key][0].cpu().numpy()[:num_frames]

        out_path = output_dir / f"{args.stim_type}_stim_{stim_index}.mp4"
        print(f"  Creating movie -> {out_path}")
        _make_two_eye_flow_movie(
            stimulus_left=stim_left, stimulus_right=stim_right,
            true_flow_left=flow_left, true_flow_right=flow_right,
            pred_flow_left=pred_flow_left, pred_flow_right=pred_flow_right,
            cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
            output_path=out_path, frame_rate=frame_rate,
            mask_left=mask_left, mask_right=mask_right,
            stim_type=args.stim_type, flow_label=flow_label,
        )


# ---------------------------------------------------------------------------
# Ego-motion pano path (from run_model_on_pano.py)
# ---------------------------------------------------------------------------

def _run_ego_motion_pano(args, model, config, model_folder, output_dir,
                         device, dtype, model_dt, column_phi_theta, N,
                         cell_x, cell_y, cell_z, frame_rate, flow_key):
    import neural_models as nm

    both_phi = np.concatenate([column_phi_theta[:, 0], -column_phi_theta[:, 0]])
    both_theta = np.concatenate([column_phi_theta[:, 1], column_phi_theta[:, 1]])

    stim_time = config['stimulus']['stim_time']
    velocity_amp = config['stimulus']['velocity_amp']
    velocity_time_scale = config['stimulus']['velocity_time_scale']
    pre_stim_time = config['stimulus'].get('pre_stim_time', 0.0)

    print(f"Generating random pano stimulus "
          f"(stim_time={stim_time}s, velocity_amp={velocity_amp}, "
          f"velocity_time_scale={velocity_time_scale}, seed={args.seed})...")
    stimulus_both, optic_flow_both, rotation_vectors, oob_mask = su.get_random_velocity_pano(
        both_phi, both_theta,
        velocity_amp=velocity_amp,
        velocity_time_scale=velocity_time_scale,
        num_repeats=1, dt=model_dt, stim_time=stim_time, seed=args.seed,
    )

    stim = stimulus_both[0]
    rot_vectors = rotation_vectors[0]
    oob = oob_mask[0]

    true_rot_rate = rot_vectors * (180.0 / np.pi)
    true_trans_rate = np.zeros_like(true_rot_rate)

    stimulus_right_np = stim[:, :N]
    stimulus_left_np = stim[:, N:]
    oob_right = oob[:, :N]
    oob_left = oob[:, N:]

    flow_both = optic_flow_both[0]
    true_mag_right = np.linalg.norm(flow_both[:, :N, :], axis=-1)
    true_mag_left = np.linalg.norm(flow_both[:, N:, :], axis=-1)

    pre_stim_frames = int(round(pre_stim_time / model_dt))
    if pre_stim_frames > 0:
        pre_right = np.full((pre_stim_frames, N), 0.5)
        pre_left = np.full((pre_stim_frames, N), 0.5)
        stimulus_right_np = np.concatenate([pre_right, stimulus_right_np], axis=0)
        stimulus_left_np = np.concatenate([pre_left, stimulus_left_np], axis=0)
        pre_rot = np.zeros((pre_stim_frames, 3))
        true_rot_rate = np.concatenate([pre_rot, true_rot_rate], axis=0)
        true_trans_rate = np.concatenate([pre_rot, true_trans_rate], axis=0)
        pre_oob = np.zeros((pre_stim_frames, N), dtype=bool)
        oob_right = np.concatenate([pre_oob, oob_right], axis=0)
        oob_left = np.concatenate([pre_oob, oob_left], axis=0)
        pre_mag = np.zeros((pre_stim_frames, N))
        true_mag_right = np.concatenate([pre_mag, true_mag_right], axis=0)
        true_mag_left = np.concatenate([pre_mag, true_mag_left], axis=0)

    num_frames = stimulus_right_np.shape[0]
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
        stimulus_right_np = stimulus_right_np[:num_frames]
        stimulus_left_np = stimulus_left_np[:num_frames]
        true_rot_rate = true_rot_rate[:num_frames]
        true_trans_rate = true_trans_rate[:num_frames]
        oob_right = oob_right[:num_frames]
        oob_left = oob_left[:num_frames]
        true_mag_right = true_mag_right[:num_frames]
        true_mag_left = true_mag_left[:num_frames]

    right_input = torch.tensor(stimulus_right_np, device=device, dtype=dtype).unsqueeze(0)
    left_input = torch.tensor(stimulus_left_np, device=device, dtype=dtype).unsqueeze(0)

    oob_always_right = np.all(oob_right, axis=0)
    oob_always_left = np.all(oob_left, axis=0)
    oob_columns = torch.tensor(
        np.concatenate([oob_always_right, oob_always_left]), device=device)

    print(f"Running model on {num_frames} frames, N={N}...")
    with torch.no_grad():
        pred_rot, pred_trans, right_res, left_res = model.estimate_ego_motion_from_image(
            right_input, left_input, oob_columns=oob_columns)

    decoder_key = next(k for k, m in model._readout_modules.items() if m.predicts_flow)
    pred_flow_right = right_res["outputs"][decoder_key][0].cpu().numpy()
    pred_flow_left = left_res["outputs"][decoder_key][0].cpu().numpy()
    pred_rot_np = pred_rot[0].cpu().numpy()
    pred_trans_np = pred_trans[0].cpu().numpy()

    if args.translation:
        pred_flow_right, pred_flow_left = subtract_rotation_flow(
            pred_flow_right, pred_flow_left, pred_rot_np, column_phi_theta)

    if args.save_figures:
        _save_rate_figure(true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_np,
                          model_dt, 0, output_dir / "pano_rate.png")
        _save_integrated_figure(true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_np,
                                model_dt, 0, output_dir / "pano_integrated.png")

    def _cos_sim_flat(a, b):
        a_flat, b_flat = a.ravel(), b.ravel()
        return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)

    rot_cos = _cos_sim_flat(true_rot_rate[pre_stim_frames:], pred_rot_np[pre_stim_frames:])
    print(f"  Rotation cosine similarity: {rot_cos:.4f}")

    in_fov_right = ~oob_right
    in_fov_left = ~oob_left

    flow_label = "translation flow" if args.translation else "predicted flow"
    out_path = output_dir / "pano_model.mp4"
    print(f"  Creating movie -> {out_path}")
    _make_ego_motion_movie(
        stimulus_left=stimulus_left_np, stimulus_right=stimulus_right_np,
        pred_flow_left=pred_flow_left[:num_frames],
        pred_flow_right=pred_flow_right[:num_frames],
        true_rot=true_rot_rate, pred_rot=pred_rot_np,
        true_trans=true_trans_rate, pred_trans=pred_trans_np,
        cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
        in_fov_left=in_fov_left, in_fov_right=in_fov_right,
        output_path=out_path, frame_rate=frame_rate,
        model_dt=model_dt, flow_label=flow_label,
    )

    has_mag_readout = 'MotionMagReadout' in model._readout_modules
    if has_mag_readout and args.save_figures:
        pred_mag_right = right_res['outputs']['MotionMagReadout'][0].cpu().numpy()
        pred_mag_left = left_res['outputs']['MotionMagReadout'][0].cpu().numpy()
        _save_magnitude_figure(
            true_mag_right, true_mag_left,
            pred_mag_right, pred_mag_left,
            model_dt, output_dir / "pano_magnitude.png")


# ---------------------------------------------------------------------------
# Movie rendering functions
# ---------------------------------------------------------------------------

def _make_one_eye_flow_movie(
    stimulus, true_flow, pred_flow, cell_x, cell_y, cell_z,
    output_path, frame_rate, mask=None, stim_type="",
):
    """Render a 3-row movie: stimulus, true flow, predicted flow."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    num_frames = stimulus.shape[0]
    gray_cmap = plt.cm.gray
    stim_norm = mcolors.Normalize(vmin=np.nanmin(stimulus), vmax=np.nanmax(stimulus))

    true_vx, true_vz, true_mag, _, _ = prepare_flow_quiver(true_flow, cell_x, cell_y, cell_z)
    pred_vx, pred_vz, pred_mag, _, _ = prepare_flow_quiver(pred_flow, cell_x, cell_y, cell_z)

    all_vx = np.concatenate([true_vx.ravel(), pred_vx.ravel()])
    all_vz = np.concatenate([true_vz.ravel(), pred_vz.ravel()])
    _, max_mag = flow_direction_colors(all_vx, all_vz)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    fig, axes = plt.subplots(3, 1, figsize=(8, 16), dpi=100)
    ax_stim, ax_true, ax_pred = axes

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())
    # Match analysis_utilities.make_stimulus_flow_movie / two-eye left panel:
    # posterior (large x) on the left of the figure so facets face like other movies.
    for ax in axes:
        ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
        ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax.set_aspect("equal")

    mask_0 = mask[0] if mask is not None else None
    sc_stim = ax_stim.scatter(
        cell_x, cell_z, c=masked_rgba(stimulus[0], mask_0, gray_cmap, stim_norm),
        s=90, edgecolors="none")
    ax_stim.set_title(f"Stimulus ({stim_type})")

    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    true_colors_0, _ = flow_direction_colors(true_vx[0], true_vz[0], max_mag)
    q_true = ax_true.quiver(cell_x, cell_z, true_vx[0], true_vz[0], **quiver_kw)
    q_true.set_facecolor(true_colors_0)
    q_true.set_edgecolor(true_colors_0)
    ax_true.set_title("True Flow")

    pred_colors_0, _ = flow_direction_colors(pred_vx[0], pred_vz[0], max_mag)
    q_pred = ax_pred.quiver(cell_x, cell_z, pred_vx[0], pred_vz[0], **quiver_kw)
    q_pred.set_facecolor(pred_colors_0)
    q_pred.set_edgecolor(pred_colors_0)
    ax_pred.set_title("Predicted Flow")
    add_direction_colorwheel(fig)
    plt.tight_layout()

    def update(t):
        mask_t = mask[t] if mask is not None else None
        sc_stim.set_facecolors(masked_rgba(stimulus[t], mask_t, gray_cmap, stim_norm))
        update_direction_quiver(q_true, true_vx[t], true_vz[t], max_mag)
        update_direction_quiver(q_pred, pred_vx[t], pred_vz[t], max_mag)
        ax_stim.set_title(f"Stimulus ({stim_type}) — frame {t + 1}/{num_frames}")

    write_movie(fig, update, num_frames, output_path, frame_rate)


def _make_one_eye_directional_movie(
    stimuli, true_flows, pred_flows, directions,
    cell_x, cell_y, cell_z,
    output_path, frame_rate, masks=None,
):
    """Render a 3×D grid movie: stimulus / true flow / pred flow per direction."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    n_dirs = len(directions)
    num_frames = stimuli.shape[1]
    gray_cmap = plt.cm.gray
    stim_norm = mcolors.Normalize(vmin=np.nanmin(stimuli), vmax=np.nanmax(stimuli))

    true_quivers = []
    pred_quivers = []
    for d_idx in range(n_dirs):
        tvx, tvz, _, _, _ = prepare_flow_quiver(true_flows[d_idx], cell_x, cell_y, cell_z)
        pvx, pvz, _, _, _ = prepare_flow_quiver(pred_flows[d_idx], cell_x, cell_y, cell_z)
        true_quivers.append((tvx, tvz))
        pred_quivers.append((pvx, pvz))

    all_vx = np.concatenate([q[0].ravel() for q in true_quivers + pred_quivers])
    all_vz = np.concatenate([q[1].ravel() for q in true_quivers + pred_quivers])
    _, max_mag = flow_direction_colors(all_vx, all_vz)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    fig, axes = plt.subplots(3, n_dirs, figsize=(4 * n_dirs, 12), dpi=100,
                             squeeze=False)

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())
    dot_size = max(20, int(90 * 2 / n_dirs))
    for ax in axes.flat:
        ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
        ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax.set_aspect("equal")

    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    sc_stims = []
    q_trues = []
    q_preds = []

    for d_idx, d in enumerate(directions):
        mask_0 = masks[d_idx, 0] if masks is not None else None
        sc = axes[0, d_idx].scatter(cell_x, cell_z,
            c=masked_rgba(stimuli[d_idx, 0], mask_0, gray_cmap, stim_norm),
            s=dot_size, edgecolors="none")
        sc_stims.append(sc)
        axes[0, d_idx].set_title(f"Stimulus ({d})")

        tvx, tvz = true_quivers[d_idx]
        colors_0, _ = flow_direction_colors(tvx[0], tvz[0], max_mag)
        qt = axes[1, d_idx].quiver(cell_x, cell_z, tvx[0], tvz[0], **quiver_kw)
        qt.set_facecolor(colors_0)
        qt.set_edgecolor(colors_0)
        q_trues.append(qt)
        axes[1, d_idx].set_title(f"True Flow ({d})")

        pvx, pvz = pred_quivers[d_idx]
        colors_0, _ = flow_direction_colors(pvx[0], pvz[0], max_mag)
        qp = axes[2, d_idx].quiver(cell_x, cell_z, pvx[0], pvz[0], **quiver_kw)
        qp.set_facecolor(colors_0)
        qp.set_edgecolor(colors_0)
        q_preds.append(qp)
        axes[2, d_idx].set_title(f"Predicted Flow ({d})")

    add_direction_colorwheel(fig)
    plt.tight_layout()

    def update(t):
        for d_idx in range(n_dirs):
            mask_t = masks[d_idx, t] if masks is not None else None
            sc_stims[d_idx].set_facecolors(
                masked_rgba(stimuli[d_idx, t], mask_t, gray_cmap, stim_norm))
            tvx, tvz = true_quivers[d_idx]
            update_direction_quiver(q_trues[d_idx], tvx[t], tvz[t], max_mag)
            pvx, pvz = pred_quivers[d_idx]
            update_direction_quiver(q_preds[d_idx], pvx[t], pvz[t], max_mag)
        axes[0, 0].set_title(
            f"Stimulus ({directions[0]}) — frame {t + 1}/{num_frames}")

    write_movie(fig, update, num_frames, output_path, frame_rate)


def _make_two_eye_flow_movie(
    stimulus_left, stimulus_right,
    true_flow_left, true_flow_right,
    pred_flow_left, pred_flow_right,
    cell_x, cell_y, cell_z,
    output_path, frame_rate,
    mask_left=None, mask_right=None,
    stim_type="", flow_label="True Flow",
):
    """Render a 3x2 movie: stimulus / true flow / predicted flow for both eyes."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    num_frames = stimulus_left.shape[0]
    gray_cmap = plt.cm.gray
    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    cell_y_left = -cell_y

    true_vx_r, true_vz_r, true_mag_r, _, _ = prepare_flow_quiver(true_flow_right, cell_x, cell_y, cell_z)
    true_vx_l, true_vz_l, true_mag_l, _, _ = prepare_flow_quiver(true_flow_left, cell_x, cell_y_left, cell_z)
    pred_vx_r, pred_vz_r, pred_mag_r, _, _ = prepare_flow_quiver(pred_flow_right, cell_x, cell_y, cell_z)
    pred_vx_l, pred_vz_l, pred_mag_l, _, _ = prepare_flow_quiver(pred_flow_left, cell_x, cell_y_left, cell_z)

    all_vx = np.concatenate([true_vx_r.ravel(), true_vx_l.ravel(),
                             pred_vx_r.ravel(), pred_vx_l.ravel()])
    all_vz = np.concatenate([true_vz_r.ravel(), true_vz_l.ravel(),
                             pred_vz_r.ravel(), pred_vz_l.ravel()])
    _, max_mag = flow_direction_colors(all_vx, all_vz)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 16), dpi=100)
    ax_stim_l, ax_stim_r = axes[0]
    ax_true_l, ax_true_r = axes[1]
    ax_pred_l, ax_pred_r = axes[2]

    setup_eye_axes(ax_stim_l, ax_stim_r, cell_x, cell_z)
    setup_eye_axes(ax_true_l, ax_true_r, cell_x, cell_z)
    setup_eye_axes(ax_pred_l, ax_pred_r, cell_x, cell_z)

    mask_l_0 = mask_left[0] if mask_left is not None else None
    mask_r_0 = mask_right[0] if mask_right is not None else None

    sc_stim_l = ax_stim_l.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_left[0], mask_l_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_right[0], mask_r_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    ax_stim_l.set_title(f"Left Eye ({stim_type})")
    ax_stim_r.set_title(f"Right Eye ({stim_type})")

    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)

    def _init_quiver(ax, vx, vz):
        c, _ = flow_direction_colors(vx, vz, max_mag)
        q = ax.quiver(cell_x, cell_z, vx, vz, **quiver_kw)
        q.set_facecolor(c)
        q.set_edgecolor(c)
        return q

    q_true_l = _init_quiver(ax_true_l, true_vx_l[0], true_vz_l[0])
    q_true_r = _init_quiver(ax_true_r, true_vx_r[0], true_vz_r[0])
    ax_true_l.set_title(f"Left Eye — {flow_label}")
    ax_true_r.set_title(f"Right Eye — {flow_label}")

    q_pred_l = _init_quiver(ax_pred_l, pred_vx_l[0], pred_vz_l[0])
    q_pred_r = _init_quiver(ax_pred_r, pred_vx_r[0], pred_vz_r[0])
    ax_pred_l.set_title("Left Eye — Predicted Flow")
    ax_pred_r.set_title("Right Eye — Predicted Flow")

    add_direction_colorwheel(fig)
    plt.tight_layout()

    def update(t):
        mask_l_t = mask_left[t] if mask_left is not None else None
        mask_r_t = mask_right[t] if mask_right is not None else None
        sc_stim_l.set_facecolors(masked_rgba(stimulus_left[t], mask_l_t, gray_cmap, stim_norm))
        sc_stim_r.set_facecolors(masked_rgba(stimulus_right[t], mask_r_t, gray_cmap, stim_norm))
        update_direction_quiver(q_true_l, true_vx_l[t], true_vz_l[t], max_mag)
        update_direction_quiver(q_true_r, true_vx_r[t], true_vz_r[t], max_mag)
        update_direction_quiver(q_pred_l, pred_vx_l[t], pred_vz_l[t], max_mag)
        update_direction_quiver(q_pred_r, pred_vx_r[t], pred_vz_r[t], max_mag)
        ax_stim_l.set_title(f"Left Eye ({stim_type}) — frame {t + 1}/{num_frames}")
        ax_stim_r.set_title(f"Right Eye ({stim_type}) — frame {t + 1}/{num_frames}")

    write_movie(fig, update, num_frames, output_path, frame_rate)


def _make_ego_motion_movie(
    stimulus_left, stimulus_right,
    pred_flow_left, pred_flow_right,
    true_rot, pred_rot, true_trans, pred_trans,
    cell_x, cell_y, cell_z,
    in_fov_left, in_fov_right,
    output_path, frame_rate, model_dt,
    trail_sec=5.0, flow_label="predicted flow",
):
    """Movie: stimulus + predicted flow + rotation/translation rate plots."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    num_frames = stimulus_left.shape[0]
    trail_frames = int(round(trail_sec / model_dt))
    times = np.arange(num_frames) * model_dt

    gray_cmap = plt.cm.gray
    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    flow_vx_r, flow_vz_r, mag_r, _, _ = prepare_flow_quiver(pred_flow_right, cell_x, cell_y, cell_z)
    flow_vx_l, flow_vz_l, mag_l, _, _ = prepare_flow_quiver(pred_flow_left, cell_x, cell_y, cell_z)

    all_vx = np.concatenate([flow_vx_r.ravel(), flow_vx_l.ravel()])
    all_vz = np.concatenate([flow_vz_r.ravel(), flow_vz_l.ravel()])
    _, max_mag = flow_direction_colors(all_vx, all_vz)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    from movie_utilities import get_fov

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.8), dpi=100,
                             gridspec_kw={"height_ratios": [3, 3, 2]})
    ax_stim_l, ax_stim_r = axes[0]
    ax_flow_l, ax_flow_r = axes[1]
    ax_rot, ax_trans = axes[2]

    setup_eye_axes(ax_stim_l, ax_stim_r, cell_x, cell_z)
    sc_stim_l = ax_stim_l.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_left[0], get_fov(in_fov_left, 0), gray_cmap, stim_norm),
        s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_right[0], get_fov(in_fov_right, 0), gray_cmap, stim_norm),
        s=90, edgecolors="none")
    ax_stim_l.set_title("Left Eye (stimulus)")
    ax_stim_r.set_title("Right Eye (stimulus)")

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
    ax_flow_l.set_title(f"Left Eye ({flow_label})")
    ax_flow_r.set_title(f"Right Eye ({flow_label})")
    add_direction_colorwheel(fig)

    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    rot_true_lines, rot_pred_lines = [], []
    for label, color in zip(rot_labels, colors):
        lt, = ax_rot.plot([], [], color=color, linewidth=1.5, label=f"true {label}")
        lp, = ax_rot.plot([], [], color=color, linewidth=1.5, linestyle="--", label=f"pred {label}")
        rot_true_lines.append(lt)
        rot_pred_lines.append(lp)
    ax_rot.legend(loc="upper left", fontsize=7, ncol=2)
    ax_rot.set_ylabel("deg/s")
    ax_rot.set_xlabel("time (s)")
    ax_rot.set_title("Rotation Rate")
    all_rot = np.concatenate([true_rot, pred_rot], axis=0)
    ax_rot.set_ylim(*ylim_from_data(all_rot, min_margin=1.0))

    trans_true_lines, trans_pred_lines = [], []
    for label, color in zip(trans_labels, colors):
        lt, = ax_trans.plot([], [], color=color, linewidth=1.5, label=f"true {label}")
        lp, = ax_trans.plot([], [], color=color, linewidth=1.5, linestyle="--", label=f"pred {label}")
        trans_true_lines.append(lt)
        trans_pred_lines.append(lp)
    ax_trans.legend(loc="upper left", fontsize=7, ncol=2)
    ax_trans.set_ylabel("velocity")
    ax_trans.set_xlabel("time (s)")
    ax_trans.set_title("Translation Rate")
    all_trans = np.concatenate([true_trans, pred_trans], axis=0)
    ax_trans.set_ylim(*ylim_from_data(all_trans, min_margin=0.01))

    plt.tight_layout()

    def update(t):
        sc_stim_l.set_facecolors(masked_rgba(stimulus_left[t], get_fov(in_fov_left, t), gray_cmap, stim_norm))
        sc_stim_r.set_facecolors(masked_rgba(stimulus_right[t], get_fov(in_fov_right, t), gray_cmap, stim_norm))
        update_direction_quiver(q_left, flow_vx_l[t], flow_vz_l[t], max_mag)
        update_direction_quiver(q_right, flow_vx_r[t], flow_vz_r[t], max_mag)

        time_s = t * model_dt
        ax_stim_l.set_title(f"Left Eye (t={time_s:.2f}s)")
        ax_stim_r.set_title(f"Right Eye (t={time_s:.2f}s)")

        t_start = max(0, t - trail_frames + 1)
        t_slice = slice(t_start, t + 1)
        t_times = times[t_slice]

        for i in range(3):
            rot_true_lines[i].set_data(t_times, true_rot[t_slice, i])
            rot_pred_lines[i].set_data(t_times, pred_rot[t_slice, i])
        ax_rot.set_xlim(t_times[0], t_times[0] + trail_sec)

        for i in range(3):
            trans_true_lines[i].set_data(t_times, true_trans[t_slice, i])
            trans_pred_lines[i].set_data(t_times, pred_trans[t_slice, i])
        ax_trans.set_xlim(t_times[0], t_times[0] + trail_sec)

    write_movie(fig, update, num_frames, output_path, frame_rate, log_every=100)


# ---------------------------------------------------------------------------
# Static figure helpers (from run_model_on_pano)
# ---------------------------------------------------------------------------

def _save_rate_figure(true_rot, pred_rot, true_trans, pred_trans,
                      model_dt, stim_idx, output_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import neural_models as nm

    true_rot = np.asarray(true_rot)
    pred_rot = np.asarray(pred_rot)
    true_trans = np.asarray(true_trans)
    pred_trans = np.asarray(pred_trans)

    T = true_rot.shape[0]
    times = np.arange(T) * model_dt
    rot_labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    trans_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(times, true_rot[:, i], linewidth=1.2, label="Ground truth")
        ax.plot(times, pred_rot[:, i], linewidth=1.2, linestyle="--", label="Model")
        ax.set_ylabel(f"{rot_labels[i]} (deg/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Rotational Velocity")
        ax = axes[i, 1]
        ax.plot(times, true_trans[:, i], linewidth=1.2, label="Ground truth")
        ax.plot(times, pred_trans[:, i], linewidth=1.2, linestyle="--", label="Model")
        ax.set_ylabel(f"{trans_labels[i]} (m/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Translational Velocity")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Stimulus {stim_idx} — Velocity", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved rate figure -> {output_path}")


def _save_integrated_figure(true_rot, pred_rot, true_trans, pred_trans,
                            model_dt, stim_idx, output_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import neural_models as nm

    true_rot = np.asarray(true_rot)
    pred_rot = np.asarray(pred_rot)
    true_trans = np.asarray(true_trans)
    pred_trans = np.asarray(pred_trans)

    true_rot_t = torch.tensor(true_rot, dtype=torch.float64)
    pred_rot_t = torch.tensor(pred_rot, dtype=torch.float64)
    true_trans_t = torch.tensor(true_trans, dtype=torch.float64)
    pred_trans_t = torch.tensor(pred_trans, dtype=torch.float64)

    true_rot_int = nm.integrate_angular_velocity(true_rot_t, model_dt).numpy()
    pred_rot_int = nm.integrate_angular_velocity(pred_rot_t, model_dt).numpy()
    true_pos = nm.integrate_translational_velocity(true_trans_t, true_rot_t, model_dt).numpy()
    pred_pos = nm.integrate_translational_velocity(pred_trans_t, pred_rot_t, model_dt).numpy()

    T = true_rot.shape[0]
    times = np.arange(T) * model_dt
    rot_labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    trans_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(times, true_rot_int[:, i], linewidth=1.2, label="Ground truth")
        ax.plot(times, pred_rot_int[:, i], linewidth=1.2, linestyle="--", label="Model")
        ax.set_ylabel(f"{rot_labels[i]} (deg)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Integrated Rotation")
        ax = axes[i, 1]
        ax.plot(times, true_pos[:, i], linewidth=1.2, label="Ground truth")
        ax.plot(times, pred_pos[:, i], linewidth=1.2, linestyle="--", label="Model")
        ax.set_ylabel(f"{trans_labels[i]} (m)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Integrated Translation")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Stimulus {stim_idx} — Integrated Pose", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved integrated figure -> {output_path}")


def _save_magnitude_figure(true_mag_right, true_mag_left,
                           pred_mag_right, pred_mag_left,
                           model_dt, output_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    T = true_mag_right.shape[0]
    times = np.arange(T) * model_dt
    true_mag_mean = (np.nanmean(true_mag_right, axis=1) + np.nanmean(true_mag_left, axis=1)) / 2
    pred_mag_mean = (np.nanmean(pred_mag_right, axis=1) + np.nanmean(pred_mag_left, axis=1)) / 2

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, true_mag_mean, color="#e41a1c", linewidth=1.5, label="True magnitude (mean)")
    ax.plot(times, pred_mag_mean, color="#377eb8", linewidth=1.5, linestyle="--", label="Predicted magnitude (mean)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motion magnitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Mean Motion Magnitude: True vs Predicted", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved magnitude figure -> {output_path}")


if __name__ == "__main__":
    main()
