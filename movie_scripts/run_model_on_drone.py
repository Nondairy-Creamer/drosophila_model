"""
Run a trained model on drone data and visualize predicted flow + odometry.

Loads drone_data.pt, runs the model on the sampled stimulus, and renders:

  Row 0: Left eye stimulus       |  Right eye stimulus
  Row 1: Left eye predicted flow |  Right eye predicted flow
  Row 2: True vs predicted rotation | True vs predicted translation

Usage:
    python analysis/run_model_on_drone.py models/my_model/
    python analysis/run_model_on_drone.py models/my_model/ --stim-index 1
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

from analysis_utilities import (
    fig_to_rgb,
    angular_flow_to_cartesian_xz,
    flow_direction_colors,
    update_direction_quiver,
    add_direction_colorwheel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model on drone data and visualize flow + odometry."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--stim-index", type=int, nargs="+", default=[0],
                        help="Stimulus indices to visualize (default: 0)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output movie frame rate (default: native video rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to render")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model_folder>/drone_flow_movie)")
    parser.add_argument("--translation", action="store_true",
                        help="Plot rotation-adjusted (translation-only) flow instead of full flow")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    local_config = du.load_config()

    print(f"Loading model from {model_path}...")
    model = nm.NeuralModel.load(model_path)
    model.eval()

    model_dt = model.dt
    connectome_folder = Path(local_config["data"]["connectome_folder"])
    drone_data_path = connectome_folder / "drone_data.pt"
    drone_video_dir = connectome_folder / "drone_videos"

    print("Loading drone clips...")
    drone_clips = su.get_drone_stimulus(model_dt, data_path=str(drone_data_path))
    print(f"Found {len(drone_clips)} clips")

    column_xyz = model.get_column_xyz()
    cell_x, cell_y, cell_z = column_xyz[:, 0], column_xyz[:, 1], column_xyz[:, 2]

    output_dir = Path(args.output_dir) if args.output_dir else model_folder / "drone_flow_movie"
    output_dir.mkdir(parents=True, exist_ok=True)

    for clip_idx in args.stim_index:
        if clip_idx >= len(drone_clips):
            print(f"Skipping clip {clip_idx}: only {len(drone_clips)} clips available")
            continue

        clip = drone_clips[clip_idx]
        clip_id = clip["clip_id"]
        video_dt = clip["video_dt"]
        N = clip["right"].shape[1]

        num_frames = clip["right"].shape[0]
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)

        right_input = torch.tensor(
            clip["right"][:num_frames], device=model.device, dtype=model.dtype
        ).unsqueeze(0)
        left_input = torch.tensor(
            clip["left"][:num_frames], device=model.device, dtype=model.dtype
        ).unsqueeze(0)

        in_fov_right = clip["in_fov_right"].numpy()
        in_fov_left = clip["in_fov_left"].numpy()
        oob = torch.tensor(
            np.concatenate([~in_fov_right, ~in_fov_left]), device=model.device
        )

        print(f"Clip {clip_idx} (id={clip_id}): running model on {num_frames} frames, N={N}...")
        with torch.no_grad():
            pred_rot, pred_trans, right_res, left_res = model.estimate_ego_motion_from_image(
                right_input, left_input, oob_columns=oob,
            )

        decoder_key = next(k for k, m in model._readout_modules.items() if m.predicts_flow)
        pred_flow_right = right_res["outputs"][decoder_key][0].cpu().numpy()
        pred_flow_left = left_res["outputs"][decoder_key][0].cpu().numpy()
        pred_rot_np = pred_rot[0].cpu().numpy()
        pred_trans_np = pred_trans[0].cpu().numpy()

        t4_keys = ["T4a", "T4b", "T4c", "T4d"]
        t5_keys = ["T5a", "T5b", "T5c", "T5d"]
        fr_right = right_res["outputs"].get("firing_rate", {})
        fr_left = left_res["outputs"].get("firing_rate", {})
        t4_right = {k: fr_right[k][0].cpu().numpy() for k in t4_keys if k in fr_right}
        t4_left = {k: fr_left[k][0].cpu().numpy() for k in t4_keys if k in fr_left}
        t5_right = {k: fr_right[k][0].cpu().numpy() for k in t5_keys if k in fr_right}
        t5_left = {k: fr_left[k][0].cpu().numpy() for k in t5_keys if k in fr_left}

        if args.translation:
            pred_flow_right, pred_flow_left = _subtract_rotation_flow(
                pred_flow_right, pred_flow_left,
                pred_rot_np, model.column_phi_theta,
            )

        true_rot_rate = clip["rotation_rate"][:num_frames]
        true_trans_rate = clip["translation_rate"][:num_frames]

        # Rescale predicted translation to match ground truth magnitude
        trans_scale = np.std(true_trans_rate) / (np.std(pred_trans_np) + 1e-12)
        pred_trans_scaled = pred_trans_np * trans_scale

        _save_rate_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_scaled,
            model_dt, clip_idx, clip_id,
            output_dir / f"drone_rate_clip_{clip_idx}.png",
        )
        _save_integrated_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_scaled,
            model_dt, clip_idx, clip_id,
            output_dir / f"drone_integrated_clip_{clip_idx}.png",
        )

        # Load raw stereo videos (TEMP: copied from drone_analysis/process_drone.py).
        # Number of raw video frames corresponding to num_frames model steps.
        num_video_frames = int(np.ceil(num_frames * model_dt / video_dt)) + 1
        raw_left_path = drone_video_dir / f"left_{clip_id}.mp4"
        raw_right_path = drone_video_dir / f"right_{clip_id}.mp4"
        raw_left = _load_video_grayscale(raw_left_path, max_frames=num_video_frames)
        raw_right = _load_video_grayscale(raw_right_path, max_frames=num_video_frames)
        print(f"  Loaded raw videos: left {raw_left.shape}, right {raw_right.shape}")

        dump_path = output_dir / f"drone_model_data_clip_{clip_idx}.pt"
        dump = {
            "clip_idx": clip_idx,
            "clip_id": clip_id,
            "model_dt": model_dt,
            "video_dt": video_dt,
            "num_frames": num_frames,
            "raw_left_video": raw_left,
            "raw_right_video": raw_right,
            "left_movie": clip["left"][:num_frames],
            "right_movie": clip["right"][:num_frames],
            "left_model_input": left_input[0].cpu().numpy(),
            "right_model_input": right_input[0].cpu().numpy(),
            "in_fov_left": in_fov_left,
            "in_fov_right": in_fov_right,
            "pred_rotation_rate": pred_rot_np,
            "pred_translation_rate": pred_trans_np,
            "pred_translation_rate_scaled": pred_trans_scaled,
            "true_rotation_rate": true_rot_rate,
            "true_translation_rate": true_trans_rate,
            "t4_activity_left": t4_left,
            "t4_activity_right": t4_right,
            "t5_activity_left": t5_left,
            "t5_activity_right": t5_right,
        }
        torch.save(dump, dump_path)
        print(f"  Saved model data dict -> {dump_path}")

        # Normalize translation to unit std (direction only, scale is arbitrary)
        true_std = np.std(true_trans_rate)
        pred_std = np.std(pred_trans_np)
        if true_std > 1e-12:
            true_trans_rate = true_trans_rate / true_std
        if pred_std > 1e-12:
            pred_trans_np = pred_trans_np / pred_std

        # Cosine similarity over all time points and dimensions (flattened)
        def _cos_sim_flat(a, b):
            a_flat, b_flat = a.ravel(), b.ravel()
            return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)

        rot_cos = _cos_sim_flat(true_rot_rate, pred_rot_np)
        trans_cos = _cos_sim_flat(true_trans_rate, pred_trans_np)
        print(f"  Rotation  cosine similarity: {rot_cos:.4f}")
        print(f"  Translation cosine similarity: {trans_cos:.4f}")

        fps = args.fps if args.fps is not None else round(1.0 / video_dt)
        step = max(1, int(round(video_dt / model_dt)))
        out_path = output_dir / f"drone_model_clip_{clip_idx}.mp4"

        flow_label = "translation flow" if args.translation else "predicted flow"
        print(f"  Creating movie -> {out_path}")
        _make_movie(
            stimulus_left=clip["left"][:num_frames],
            stimulus_right=clip["right"][:num_frames],
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
            step=step,
            model_dt=model_dt,
            flow_label=flow_label,
        )


def _load_video_grayscale(path, max_frames=None):
    """Load video as grayscale uint8 frames. Copied from drone_analysis/process_drone.py."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise FileNotFoundError(f"No frames read from video: {path}")
    return np.stack(frames)


def _save_rate_figure(true_rot, pred_rot, true_trans, pred_trans,
                      model_dt, clip_idx, clip_id, output_path):
    """Save a 3x2 figure comparing ground-truth vs model velocity estimates.

    Left column: rotational velocity (roll, pitch, yaw).
    Right column: translational velocity (x, y, z).
    """
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
    fig.suptitle(f"Clip {clip_idx} (id={clip_id}) — Velocity", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved rate figure → {output_path}")


def _save_integrated_figure(true_rot, pred_rot, true_trans, pred_trans,
                            model_dt, clip_idx, clip_id, output_path):
    """Save a 3x2 figure comparing integrated ground-truth vs model pose.

    Left column: integrated rotation (cumulative angle in degrees).
    Right column: integrated translation (cumulative position in metres).
    Uses SO(3)-aware integration from neural_models.
    """
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
    fig.suptitle(f"Clip {clip_idx} (id={clip_id}) — Integrated Pose", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved integrated figure → {output_path}")


def _subtract_rotation_flow(pred_flow_right, pred_flow_left, omega, column_phi_theta):
    """Subtract rotation-induced flow, returning translation-only angular flow.

    Uses the same decomposition as NeuralModel.estimate_ego_motion_from_flow:
        u_rot = x × omega
        d_right_rot = -(u_rot · e_phi),  d_up_rot = -(u_rot · e_theta)

    Args:
        pred_flow_right: (T, N, 2) predicted angular flow for right eye
        pred_flow_left:  (T, N, 2) predicted angular flow for left eye
        omega: (T, 3) estimated rotation [roll, pitch, yaw]
        column_phi_theta: (N, 2) column positions [phi, theta] for the right eye

    Returns:
        trans_flow_right, trans_flow_left: same shapes, rotation subtracted
    """
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
    output_path, frame_rate, step, model_dt,
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
    frame_indices = list(range(0, num_frames, step))

    _MASK_COLOR = np.array([1.0, 1.0, 0.0, 1.0])
    gray_cmap = plt.cm.gray

    def _masked_rgba(values, mask, norm):
        rgba = gray_cmap(norm(values))
        if mask is not None:
            rgba[~mask] = _MASK_COLOR
        return rgba

    # --- Stimulus colour scale ---
    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    # --- Flow projection & colour scale ---
    flow_vx_r, flow_vz_r = angular_flow_to_cartesian_xz(
        pred_flow_right[:, :, 0], pred_flow_right[:, :, 1], cell_x, cell_y, cell_z,
    )
    flow_vx_l, flow_vz_l = angular_flow_to_cartesian_xz(
        pred_flow_left[:, :, 0], pred_flow_left[:, :, 1], cell_x, cell_y, cell_z,
    )
    mag_r = np.hypot(flow_vx_r, flow_vz_r)
    mag_l = np.hypot(flow_vx_l, flow_vz_l)

    all_vx = np.concatenate([flow_vx_r.ravel(), flow_vx_l.ravel()])
    all_vz = np.concatenate([flow_vz_r.ravel(), flow_vz_l.ravel()])
    _, max_mag = flow_direction_colors(
        np.nan_to_num(all_vx, nan=0.0), np.nan_to_num(all_vz, nan=0.0))
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    flow_vx_r = np.nan_to_num(flow_vx_r, nan=0.0)
    flow_vz_r = np.nan_to_num(flow_vz_r, nan=0.0)
    flow_vx_l = np.nan_to_num(flow_vx_l, nan=0.0)
    flow_vz_l = np.nan_to_num(flow_vz_l, nan=0.0)
    mag_r = np.nan_to_num(mag_r, nan=0.0)
    mag_l = np.nan_to_num(mag_l, nan=0.0)

    # --- Figure setup: 3 rows x 2 cols ---
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

    # Row 0: stimulus
    _setup_eye_axes(ax_stim_l, ax_stim_r)
    sc_stim_l = ax_stim_l.scatter(
        cell_x, cell_z, c=_masked_rgba(stimulus_left[0], in_fov_left, stim_norm),
        s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(
        cell_x, cell_z, c=_masked_rgba(stimulus_right[0], in_fov_right, stim_norm),
        s=90, edgecolors="none")
    ax_stim_l.set_title("Left Eye (stimulus)")
    ax_stim_r.set_title("Right Eye (stimulus)")

    # Row 1: predicted flow
    _setup_eye_axes(ax_flow_l, ax_flow_r)
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

    # Row 2: rotation and translation (true vs predicted)
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
    print(f"  Writing {len(frame_indices)} frames at {frame_rate} fps...")

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in frame_indices:
            sc_stim_l.set_facecolors(_masked_rgba(stimulus_left[t], in_fov_left, stim_norm))
            sc_stim_r.set_facecolors(_masked_rgba(stimulus_right[t], in_fov_right, stim_norm))

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

            writer.append_data(fig_to_rgb(fig))
            if (t + 1) % 100 == 0:
                print(f"    Frame {t+1}/{num_frames}")

    plt.close(fig)
    print(f"  Done! -> {output_path}")


if __name__ == "__main__":
    main()
