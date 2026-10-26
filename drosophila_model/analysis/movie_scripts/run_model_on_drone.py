"""
Run a trained model on drone data and visualize predicted flow + odometry.

Loads the ommatidium-sampled ``small_drone`` clips from the baked cache
(``<data_folder>/stimulus/baked/small_drone.pt``, built by
``process_raw_data small_drone``), runs the model's two-eye ego-motion
estimate on each requested clip, and renders:

  Row 0: Left eye stimulus       |  Right eye stimulus
  Row 1: Left eye predicted flow |  Right eye predicted flow
  (Row 2 with --depth: left/right estimated nearness)
  Last row: True vs predicted rotation | True vs predicted translation

plus static rate / integrated-pose figures per clip.

Usage (from the repo root):
    python -m drosophila_model.analysis.movie_scripts.run_model_on_drone models/my_model/
    python -m drosophila_model.analysis.movie_scripts.run_model_on_drone models/my_model/ --stim-index 1
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling import drone_utilities as du
from drosophila_model.data_handling import stimulus_generators as su
from drosophila_model.analysis.model_analysis.analysis_utilities import (
    cos_sim, save_integrated_figure, save_rate_figure,
)

from drosophila_model.analysis.movie_scripts.movie_utilities import (
    masked_rgba,
    write_movie,
    ylim_from_data,
    subtract_rotation_flow,
    get_flow_key,
    prepare_eye_quiver,
    setup_eye_axis,
    init_eye_quiver,
    update_eye_quiver,
    add_eye_colorwheel,
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
                        help="Output directory (default: <model_folder>/figures/run_model_on_drone)")
    parser.add_argument("--translation", action="store_true",
                        help="Plot rotation-adjusted (translation-only) flow instead of full flow")
    parser.add_argument("--depth", action="store_true",
                        help="Add a row showing the estimated per-column nearness "
                             "(rho ~ 1/depth) heatmap")
    parser.add_argument("--save-data", action="store_true",
                        help="Also save each clip's model inputs, ego-motion estimates and "
                             "T4/T5 activity to "
                             "<model_folder>/analysis_data/run_model_on_drone/")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)

    model_dt = model.dt
    column_phi_theta = model.regions['optic_lobe'].column_phi_theta

    flow_key = get_flow_key(model)
    if flow_key is None:
        raise SystemExit("Model has no flow readout (predicts_flow=True); "
                         "cannot estimate ego-motion.")

    print("Loading drone clips...")
    drone_clips = du.get_drone_stimulus(
        model_dt, dataset="small_drone",
        sample_phi=column_phi_theta[:, 0], sample_theta=column_phi_theta[:, 1])
    print(f"Found {len(drone_clips)} clips")
    # Solve ego-motion with the rig geometry registered for this dataset,
    # as training does.
    camera_offset = su.camera_offset_for_stimulus("small_drone")

    column_xyz = model.regions['optic_lobe'].get_column_xyz()
    cell_x, cell_y, cell_z = column_xyz[:, 0], column_xyz[:, 1], column_xyz[:, 2]

    output_dir = (Path(args.output_dir) if args.output_dir
                  else model_folder / "figures" / "run_model_on_drone")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = model_folder / "analysis_data" / "run_model_on_drone"

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
            pred_rot, pred_trans, right_res, left_res = model.regions['optic_lobe'].estimate_ego_motion_from_image(
                right_input, left_input, oob_columns=oob, camera_offset=camera_offset,
            )

        pred_flow_right = right_res["outputs"]["optic_lobe"][flow_key][0].cpu().numpy()
        pred_flow_left = left_res["outputs"]["optic_lobe"][flow_key][0].cpu().numpy()
        pred_rot_np = pred_rot[0].cpu().numpy()
        pred_trans_np = pred_trans[0].cpu().numpy()

        nearness_right = None
        nearness_left = None
        if args.depth:
            nearness_right, nearness_left = _estimate_depth_per_column(
                model, right_res, left_res, flow_key, oob, num_frames, N,
            )
            nearness_right, nearness_left = _mask_pole_columns(
                nearness_right, nearness_left, pred_trans_np, column_phi_theta,
            )

        if args.translation:
            pred_flow_right, pred_flow_left = subtract_rotation_flow(
                pred_flow_right, pred_flow_left, pred_rot_np, column_phi_theta,
            )

        true_rot_rate = clip["rotation_rate"][:num_frames]
        true_trans_rate = clip["translation_rate"][:num_frames]

        # Rescale predicted translation to match ground truth magnitude
        trans_scale = np.std(true_trans_rate) / (np.std(pred_trans_np) + 1e-12)
        pred_trans_scaled = pred_trans_np * trans_scale

        title = f"Clip {clip_idx} (id={clip_id})"
        save_rate_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_scaled,
            model_dt, title, output_dir / f"drone_rate_clip_{clip_idx}.png",
        )
        save_integrated_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_scaled,
            model_dt, title, output_dir / f"drone_integrated_clip_{clip_idx}.png",
        )

        if args.save_data:
            _save_clip_data(
                data_dir / f"drone_model_data_clip_{clip_idx}.pt",
                right_res, left_res,
                clip_idx=clip_idx, clip_id=clip_id, model_dt=model_dt,
                video_dt=video_dt, num_frames=num_frames,
                left_movie=clip["left"][:num_frames],
                right_movie=clip["right"][:num_frames],
                left_model_input=left_input[0].cpu().numpy(),
                right_model_input=right_input[0].cpu().numpy(),
                in_fov_left=in_fov_left, in_fov_right=in_fov_right,
                pred_rotation_rate=pred_rot_np,
                pred_translation_rate=pred_trans_np,
                pred_translation_rate_scaled=pred_trans_scaled,
                true_rotation_rate=true_rot_rate,
                true_translation_rate=true_trans_rate,
            )

        # Normalize translation to unit std (direction only, scale is arbitrary)
        true_std = np.std(true_trans_rate)
        pred_std = np.std(pred_trans_np)
        if true_std > 1e-12:
            true_trans_rate = true_trans_rate / true_std
        if pred_std > 1e-12:
            pred_trans_np = pred_trans_np / pred_std

        # Cosine similarity over all time points and dimensions (flattened)
        rot_cos = cos_sim(true_rot_rate, pred_rot_np)
        trans_cos = cos_sim(true_trans_rate, pred_trans_np)
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
            nearness_left=nearness_left,
            nearness_right=nearness_right,
        )


def _save_clip_data(path, right_res, left_res, **fields):
    """Save one clip's inputs, estimates and T4/T5 firing rates (``--save-data``).

    ``fields`` are stored as-is; T4/T5 activity per eye is added as
    ``t4_activity_left`` / ``t4_activity_right`` / ``t5_activity_*`` dicts of
    ``(T, columns)`` arrays keyed by cell type.
    """
    t4_keys = ["T4a", "T4b", "T4c", "T4d"]
    t5_keys = ["T5a", "T5b", "T5c", "T5d"]
    fr_right = right_res["outputs"]["optic_lobe"].get("firing_rate", {})
    fr_left = left_res["outputs"]["optic_lobe"].get("firing_rate", {})
    dump = dict(fields)
    for name, keys in (("t4", t4_keys), ("t5", t5_keys)):
        dump[f"{name}_activity_left"] = {
            k: fr_left[k][0].cpu().numpy() for k in keys if k in fr_left}
        dump[f"{name}_activity_right"] = {
            k: fr_right[k][0].cpu().numpy() for k in keys if k in fr_right}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, path)
    print(f"  Saved model data dict -> {path}")


def _estimate_depth_per_column(model, right_res, left_res, decoder_key,
                               oob, num_frames, N):
    """Run the ego-motion solver to recover per-column ρ_i = ‖T‖/Z_i (nearness),
    and return it split per eye.

    Columns excluded by the valid mask (NaN flow or out-of-FOV) are NaN.

    Returns:
        nearness_right, nearness_left: (T, N) numpy arrays of ρ_i ∝ 1/Z_i.
    """
    optic_flow_right = right_res["outputs"]["optic_lobe"][decoder_key]   # (1, T, N, 2)
    optic_flow_left = left_res["outputs"]["optic_lobe"][decoder_key]
    optic_flow = torch.cat((optic_flow_right, optic_flow_left), dim=2)  # (1, T, 2N, 2)

    phi = model.regions['optic_lobe'].column_phi_theta_two_eyes[0].to(dtype=optic_flow.dtype)
    theta = model.regions['optic_lobe'].column_phi_theta_two_eyes[1].to(dtype=optic_flow.dtype)

    valid_mask = ~torch.isnan(optic_flow[0, 0, :, 0]) & ~torch.isnan(optic_flow[0, 0, :, 1])
    if oob is not None:
        valid_mask = valid_mask & ~oob

    d_right = optic_flow[0, :num_frames, :, 0][:, valid_mask]
    d_up = optic_flow[0, :num_frames, :, 1][:, valid_mask]

    with torch.no_grad():
        _, _, rho_valid = model.regions['optic_lobe'].estimate_ego_motion_from_flow(
            phi[valid_mask], theta[valid_mask], d_right, d_up,
        )

    rho_full = torch.full(
        (num_frames, 2 * N), float("nan"),
        device=rho_valid.device, dtype=rho_valid.dtype,
    )
    rho_full[:, valid_mask] = rho_valid
    rho_np = rho_full.cpu().numpy()
    return rho_np[:, :N], rho_np[:, N:]


def _mask_pole_columns(nearness_right, nearness_left, pred_trans,
                       column_phi_theta, threshold=0.1):
    """NaN-out columns within the focus of expansion/contraction cone.

    Mirrors the per-frame mask used inside ``ego_motion._estimate_translation_from_flow``
    when aggregating ρ into a scalar: drops columns where
    ``‖Π_i t̂‖² = 1 - (x_i · t̂)² ≤ threshold``.

    Args:
        nearness_right, nearness_left: (T, N) numpy arrays of ρ per eye.
        pred_trans: (T, 3) predicted translation; only its direction is used.
        column_phi_theta: (N, 2) right-eye column [phi, theta].
        threshold: same default (0.1) as the algorithm's internal mask.
    """
    phi_r = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
    phi_l = -phi_r

    def _viewdir(phi):
        return np.stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)], axis=-1)
    x_r = _viewdir(phi_r)
    x_l = _viewdir(phi_l)

    norm = np.linalg.norm(pred_trans, axis=-1, keepdims=True)
    t_hat = pred_trans / np.where(norm > 1e-12, norm, 1.0)

    cos_r = np.einsum('ni,ti->tn', x_r, t_hat)
    cos_l = np.einsum('ni,ti->tn', x_l, t_hat)
    valid_r = (1.0 - cos_r ** 2) > threshold
    valid_l = (1.0 - cos_l ** 2) > threshold

    return (np.where(valid_r, nearness_right, np.nan),
            np.where(valid_l, nearness_left, np.nan))


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
    nearness_left=None, nearness_right=None,
):
    """Movie of every ``step``-th model frame: stimulus, flow, optional
    nearness, and trailing true-vs-predicted rotation/translation rates."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    num_frames = stimulus_left.shape[0]
    trail_frames = int(round(trail_sec / model_dt))
    times = np.arange(num_frames) * model_dt
    frame_indices = list(range(0, num_frames, step))

    gray_cmap = plt.cm.gray
    show_nearness = nearness_left is not None and nearness_right is not None

    def _nearness_rgba(values, mask, norm):
        # Columns without a finite nearness are painted like out-of-FOV ones.
        return masked_rgba(values, mask & np.isfinite(values), gray_cmap, norm)

    def _nearness_norm_for_frame(t):
        """Per-frame Normalize shared by both eyes, computed from finite,
        in-FOV ρ values across both eyes."""
        if not show_nearness:
            return None
        vL, vR = nearness_left[t], nearness_right[t]
        mL = in_fov_left & np.isfinite(vL)
        mR = in_fov_right & np.isfinite(vR)
        parts = []
        if mL.any():
            parts.append(vL[mL])
        if mR.any():
            parts.append(vR[mR])
        if parts:
            v = np.concatenate(parts)
            vmin, vmax = float(v.min()), float(v.max())
        else:
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- Stimulus colour scale ---
    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    # --- Flow projection & colour scale ---
    # Per-eye projection (cell_y mirroring for the left eye) is encapsulated in
    # the eye_view helpers; see analysis_utilities.py for the convention.
    flow_vx_r, flow_vz_r, _, _, _ = prepare_eye_quiver(
        pred_flow_right, cell_x, cell_y, cell_z, "right")
    flow_vx_l, flow_vz_l, _, _, _ = prepare_eye_quiver(
        pred_flow_left, cell_x, cell_y, cell_z, "left")

    all_mag = np.hypot(
        np.concatenate([flow_vx_r.ravel(), flow_vx_l.ravel()]),
        np.concatenate([flow_vz_r.ravel(), flow_vz_l.ravel()]),
    )
    finite_mag = all_mag[np.isfinite(all_mag)]
    max_mag = (float(np.percentile(finite_mag, 99))
               if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    # --- Figure setup: 3 (or 4 with nearness) rows x 2 cols ---
    if show_nearness:
        fig, axes = plt.subplots(4, 2, figsize=(12.8, 16.0), dpi=100,
                                 gridspec_kw={"height_ratios": [3, 3, 3, 2]})
        ax_stim_l, ax_stim_r = axes[0]
        ax_flow_l, ax_flow_r = axes[1]
        ax_near_l, ax_near_r = axes[2]
        ax_rot, ax_trans = axes[3]
    else:
        fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.8), dpi=100,
                                 gridspec_kw={"height_ratios": [3, 3, 2]})
        ax_stim_l, ax_stim_r = axes[0]
        ax_flow_l, ax_flow_r = axes[1]
        ax_rot, ax_trans = axes[2]

    setup_eye_axis(ax_stim_l, cell_x, cell_z, "left")
    setup_eye_axis(ax_stim_r, cell_x, cell_z, "right")
    sc_stim_l = ax_stim_l.scatter(
        cell_x, cell_z, c=masked_rgba(stimulus_left[0], in_fov_left, gray_cmap, stim_norm),
        s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(
        cell_x, cell_z, c=masked_rgba(stimulus_right[0], in_fov_right, gray_cmap, stim_norm),
        s=90, edgecolors="none")
    ax_stim_l.set_title("Left Eye (stimulus)")
    ax_stim_r.set_title("Right Eye (stimulus)")

    setup_eye_axis(ax_flow_l, cell_x, cell_z, "left")
    setup_eye_axis(ax_flow_r, cell_x, cell_z, "right")
    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    q_left = init_eye_quiver(
        ax_flow_l, cell_x, cell_z,
        flow_vx_l[0], flow_vz_l[0], max_mag, "left", **quiver_kw)
    q_right = init_eye_quiver(
        ax_flow_r, cell_x, cell_z,
        flow_vx_r[0], flow_vz_r[0], max_mag, "right", **quiver_kw)
    ax_flow_l.set_title(f"Left Eye ({flow_label})")
    ax_flow_r.set_title(f"Right Eye ({flow_label})")

    if show_nearness:
        setup_eye_axis(ax_near_l, cell_x, cell_z, "left")
        setup_eye_axis(ax_near_r, cell_x, cell_z, "right")
        near_norm0 = _nearness_norm_for_frame(0)
        sc_near_l = ax_near_l.scatter(
            cell_x, cell_z, c=_nearness_rgba(nearness_left[0], in_fov_left, near_norm0),
            s=90, edgecolors="none")
        sc_near_r = ax_near_r.scatter(
            cell_x, cell_z, c=_nearness_rgba(nearness_right[0], in_fov_right, near_norm0),
            s=90, edgecolors="none")
        ax_near_l.set_title("Left Eye (nearness ρ ∝ 1/Z)")
        ax_near_r.set_title("Right Eye (nearness ρ ∝ 1/Z)")
        near_sm = plt.cm.ScalarMappable(cmap=gray_cmap, norm=near_norm0)
        near_sm.set_array([])
        fig.colorbar(
            near_sm, ax=ax_near_r, location="right",
            shrink=0.85, pad=0.02, label="ρ ∝ 1/Z",
        )

    # Last row: rotation and translation (true vs predicted)
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
    # After tight_layout: the fixed-position polar wheel is not a subplot.
    add_eye_colorwheel(fig)

    def update(k):
        t = frame_indices[k]
        sc_stim_l.set_facecolors(masked_rgba(stimulus_left[t], in_fov_left, gray_cmap, stim_norm))
        sc_stim_r.set_facecolors(masked_rgba(stimulus_right[t], in_fov_right, gray_cmap, stim_norm))

        update_eye_quiver(q_left, flow_vx_l[t], flow_vz_l[t], max_mag, "left")
        update_eye_quiver(q_right, flow_vx_r[t], flow_vz_r[t], max_mag, "right")

        if show_nearness:
            near_norm = _nearness_norm_for_frame(t)
            sc_near_l.set_facecolors(_nearness_rgba(nearness_left[t], in_fov_left, near_norm))
            sc_near_r.set_facecolors(_nearness_rgba(nearness_right[t], in_fov_right, near_norm))
            near_sm.set_clim(vmin=near_norm.vmin, vmax=near_norm.vmax)
            near_sm.changed()

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

    write_movie(fig, update, len(frame_indices), output_path, frame_rate, log_every=100)


if __name__ == "__main__":
    main()
