"""
Run a trained model on a freshly generated random pano and visualize predicted flow + odometry.

Generates a single random panorama stimulus using the model's column layout
and config parameters (stim_time, velocity_amp, etc.), runs the model, and renders:

  Row 0: Left eye stimulus       |  Right eye stimulus
  Row 1: Left eye predicted flow |  Right eye predicted flow
  Row 2: True vs predicted rotation | True vs predicted translation

Also saves static rate and integrated-pose figures (same as run_model_on_drone.py).

Usage:
    python analysis/run_model_on_pano.py models/my_model/
    python analysis/run_model_on_pano.py models/my_model/ --config path/to/config.toml
    python analysis/run_model_on_pano.py models/my_model/ --translation
    python analysis/run_model_on_pano.py models/my_model/ --seed 42
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su

from analysis_utilities import (
    fig_to_rgb,
    angular_flow_to_cartesian_xz,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model on a fresh random pano and visualize flow + odometry."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.toml for stimulus parameters "
                             "(default: <model_folder>/config.toml)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for stimulus generation (default: random)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output movie frame rate (default: half the model rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to render")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model_folder>/pano_flow_movie)")
    parser.add_argument("--translation", action="store_true",
                        help="Plot rotation-adjusted (translation-only) flow instead of full flow")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    # Load config for stimulus parameters (defaults to workspace config.toml)
    if args.config:
        print(f"Loading config from {args.config}...")
        config = du.load_config(args.config)
    else:
        print("Loading config from config.toml...")
        config = du.load_config()

    print(f"Loading model from {model_path}...")
    model = nm.NeuralModel.load(model_path)
    model.eval()

    model_dt = model.dt
    device = model.device
    dtype = model.dtype

    column_phi_theta = model.column_phi_theta
    N = column_phi_theta.shape[0]
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
        num_repeats=1,
        dt=model_dt,
        stim_time=stim_time,
        seed=args.seed,
    )

    # Take the first example (num_repeats=1 still produces doubled for symmetrization)
    stim = stimulus_both[0]        # (T, 2N)
    rot_vectors = rotation_vectors[0]  # (T, 3) rad/s
    oob = oob_mask[0]              # (T, 2N)

    true_rot_rate = rot_vectors * (180.0 / np.pi)  # rad/s -> deg/s
    true_trans_rate = np.zeros_like(true_rot_rate)  # pure rotation, no translation

    # Split into right / left eye
    stimulus_right_np = stim[:, :N]
    stimulus_left_np = stim[:, N:]
    oob_right = oob[:, :N]   # (T, N) — time-varying
    oob_left = oob[:, N:]    # (T, N)

    # Prepend pre-stimulus interval (middle gray, stationary, no oob)
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

    num_frames = stimulus_right_np.shape[0]
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
        stimulus_right_np = stimulus_right_np[:num_frames]
        stimulus_left_np = stimulus_left_np[:num_frames]
        true_rot_rate = true_rot_rate[:num_frames]
        true_trans_rate = true_trans_rate[:num_frames]
        oob_right = oob_right[:num_frames]
        oob_left = oob_left[:num_frames]

    right_input = torch.tensor(stimulus_right_np, device=device, dtype=dtype).unsqueeze(0)
    left_input = torch.tensor(stimulus_left_np, device=device, dtype=dtype).unsqueeze(0)

    # Match training: column is oob only if it's NEVER valid (oob at every frame)
    oob_always_right = np.all(oob_right, axis=0)
    oob_always_left = np.all(oob_left, axis=0)
    oob_columns = torch.tensor(
        np.concatenate([oob_always_right, oob_always_left]), device=device,
    )

    print(f"Running model on {num_frames} frames, N={N}...")
    with torch.no_grad():
        pred_rot, pred_trans, right_res, left_res = model.estimate_ego_motion_from_image(
            right_input, left_input, oob_columns=oob_columns,
        )

    pred_flow_right = right_res["outputs"]["decoder"][0].cpu().numpy()
    pred_flow_left = left_res["outputs"]["decoder"][0].cpu().numpy()
    pred_rot_np = pred_rot[0].cpu().numpy()
    pred_trans_np = pred_trans[0].cpu().numpy()

    if args.translation:
        pred_flow_right, pred_flow_left = _subtract_rotation_flow(
            pred_flow_right, pred_flow_left,
            pred_rot_np, column_phi_theta,
        )

    column_xyz = model.get_column_xyz()
    cell_x, cell_y, cell_z = column_xyz[:, 0], column_xyz[:, 1], column_xyz[:, 2]

    output_dir = Path(args.output_dir) if args.output_dir else model_folder / "pano_flow_movie"
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_rate_figure(
        true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_np,
        model_dt, 0,
        output_dir / "pano_rate.png",
    )
    _save_integrated_figure(
        true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_np,
        model_dt, 0,
        output_dir / "pano_integrated.png",
    )

    def _cos_sim_flat(a, b):
        a_flat, b_flat = a.ravel(), b.ravel()
        return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)

    # Skip pre-stim frames (zero rotation) for similarity metric
    rot_cos = _cos_sim_flat(true_rot_rate[pre_stim_frames:], pred_rot_np[pre_stim_frames:])
    print(f"  Rotation cosine similarity: {rot_cos:.4f}")

    in_fov_right = ~oob_right  # (T, N)
    in_fov_left = ~oob_left    # (T, N)

    fps = args.fps if args.fps is not None else max(1, int((1 / model_dt) / 2))
    out_path = output_dir / "pano_model.mp4"

    flow_label = "translation flow" if args.translation else "predicted flow"
    print(f"  Creating movie -> {out_path}")
    _make_movie(
        stimulus_left=stimulus_left_np,
        stimulus_right=stimulus_right_np,
        pred_flow_left=pred_flow_left[:num_frames],
        pred_flow_right=pred_flow_right[:num_frames],
        true_rot=true_rot_rate,
        pred_rot=pred_rot_np,
        true_trans=true_trans_rate,
        pred_trans=pred_trans_np,
        cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
        in_fov_left=in_fov_left, in_fov_right=in_fov_right,
        output_path=out_path,
        frame_rate=fps,
        model_dt=model_dt,
        flow_label=flow_label,
    )


def _save_rate_figure(true_rot, pred_rot, true_trans, pred_trans,
                      model_dt, stim_idx, output_path):
    """Save a 3x2 figure comparing ground-truth vs model velocity estimates."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

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
    """Save a 3x2 figure comparing integrated ground-truth vs model pose."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

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
    true_pos = nm.integrate_translational_velocity(
        true_trans_t, true_rot_t, model_dt).numpy()
    pred_pos = nm.integrate_translational_velocity(
        pred_trans_t, pred_rot_t, model_dt).numpy()

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


def _subtract_rotation_flow(pred_flow_right, pred_flow_left, omega, column_phi_theta):
    """Subtract rotation-induced flow, returning translation-only angular flow."""
    phi_r = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
    phi_l = -phi_r

    def _rotation_angular_flow(phi, theta, omega):
        x = np.stack([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)], axis=-1)
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
        e_theta = np.stack([np.cos(theta) * np.cos(phi),
                            np.cos(theta) * np.sin(phi),
                            -np.sin(theta)], axis=-1)
        u_rot = np.cross(x[None, :, :], omega[:, None, :])
        d_right_rot = -np.einsum('tnj,nj->tn', u_rot, e_phi)
        d_up_rot = -np.einsum('tnj,nj->tn', u_rot, e_theta)
        return d_right_rot, d_up_rot

    dr_r, du_r = _rotation_angular_flow(phi_r, theta, omega)
    dr_l, du_l = _rotation_angular_flow(phi_l, theta, omega)

    out_right = pred_flow_right.copy()
    out_right[:, :, 0] -= dr_r
    out_right[:, :, 1] -= du_r

    out_left = pred_flow_left.copy()
    out_left[:, :, 0] -= dr_l
    out_left[:, :, 1] -= du_l

    return out_right, out_left


def _make_movie(
    stimulus_left, stimulus_right,
    pred_flow_left, pred_flow_right,
    true_rot, pred_rot,
    true_trans, pred_trans,
    cell_x, cell_y, cell_z,
    in_fov_left, in_fov_right,
    output_path, frame_rate, model_dt,
    trail_sec=5.0,
    flow_label="predicted flow",
):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors
    import imageio as iio

    num_frames = stimulus_left.shape[0]
    trail_frames = int(round(trail_sec / model_dt))
    times = np.arange(num_frames) * model_dt

    _MASK_COLOR = np.array([1.0, 1.0, 0.0, 1.0])
    gray_cmap = plt.cm.gray

    def _masked_rgba(values, mask, norm):
        rgba = gray_cmap(norm(values))
        if mask is not None:
            rgba[~mask] = _MASK_COLOR
        return rgba

    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

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

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.8), dpi=100,
                             gridspec_kw={"height_ratios": [3, 3, 2]})
    ax_stim_l, ax_stim_r = axes[0]
    ax_flow_l, ax_flow_r = axes[1]
    ax_rot, ax_trans = axes[2]

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

    # in_fov can be (N,) static or (T, N) time-varying
    def _get_fov(in_fov, t):
        if in_fov.ndim == 2:
            return in_fov[t]
        return in_fov

    _setup_eye_axes(ax_stim_l, ax_stim_r)
    sc_stim_l = ax_stim_l.scatter(
        cell_x, cell_z, c=_masked_rgba(stimulus_left[0], _get_fov(in_fov_left, 0), stim_norm),
        s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(
        cell_x, cell_z, c=_masked_rgba(stimulus_right[0], _get_fov(in_fov_right, 0), stim_norm),
        s=90, edgecolors="none")
    ax_stim_l.set_title("Left Eye (stimulus)")
    ax_stim_r.set_title("Right Eye (stimulus)")

    _setup_eye_axes(ax_flow_l, ax_flow_r)
    quiver_kw = dict(
        cmap="viridis", norm=flow_norm, angles="xy",
        scale_units="xy", scale=1 / arrow_scale, width=0.003,
    )
    q_left = ax_flow_l.quiver(cell_x, cell_z, flow_vx_l[0], flow_vz_l[0], mag_l[0], **quiver_kw)
    q_right = ax_flow_r.quiver(cell_x, cell_z, flow_vx_r[0], flow_vz_r[0], mag_r[0], **quiver_kw)
    ax_flow_l.set_title(f"Left Eye ({flow_label})")
    ax_flow_r.set_title(f"Right Eye ({flow_label})")

    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    def _ylim(data, margin_frac=0.1, min_margin=0.01):
        lo, hi = np.nanmin(data), np.nanmax(data)
        margin = max(margin_frac * (hi - lo), min_margin)
        return lo - margin, hi + margin

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
    ax_rot.set_ylim(*_ylim(all_rot, min_margin=1.0))

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
    ax_trans.set_ylim(*_ylim(all_trans, min_margin=0.01))

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {num_frames} frames at {frame_rate} fps...")

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_frames):
            sc_stim_l.set_facecolors(_masked_rgba(stimulus_left[t], _get_fov(in_fov_left, t), stim_norm))
            sc_stim_r.set_facecolors(_masked_rgba(stimulus_right[t], _get_fov(in_fov_right, t), stim_norm))

            q_left.set_UVC(flow_vx_l[t], flow_vz_l[t], mag_l[t])
            q_right.set_UVC(flow_vx_r[t], flow_vz_r[t], mag_r[t])

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

            writer.append_data(fig_to_rgb(fig))
            if (t + 1) % 100 == 0:
                print(f"    Frame {t+1}/{num_frames}")

    plt.close(fig)
    print(f"  Done! -> {output_path}")


if __name__ == "__main__":
    main()
