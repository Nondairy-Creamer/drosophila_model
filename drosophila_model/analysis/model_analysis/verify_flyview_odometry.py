"""
Validate the fly-view odometry reconstruction.

``bake_fly_view_dataset`` (in ``data_handling/stimulus_generators.py``) solves
per-frame ego-motion (rotation deg/s, translation units/s) from the rendered
optic flow + per-column depth via
``ego_motion.estimate_ego_motion_known_depth``. We compare that reconstruction
against the ground-truth odometry stored in the dataset (``rotation_rate`` deg/s
and ``translation_rate`` units/s, both body frame). Datasets without those
fields fall back to deriving body-frame velocities from the renderer trajectory
(``positions`` + ``roll_pitch_yaw``) using the repo's
``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` convention.

For each clip we overlay reconstruction vs the stored odometry (3 rotation +
3 translation axes) and print per-axis Pearson correlation. A diagonal
least-squares gain (recon ~ gain * truth, per axis) is reported in aggregate so
any axis-dependent scale / sign mismatch is explicit. Both recon and truth are
deg/s (rotation) and units/s (translation), so a clean match is corr ~1, gain ~1.

Figures -> ``figures/verify_flyview_odometry/`` (general analysis, not tied to a
trained model).

Usage:
    python -m drosophila_model.analysis.model_analysis.verify_flyview_odometry
    python -m drosophila_model.analysis.model_analysis.verify_flyview_odometry --clips 0 5 10
    python -m drosophila_model.analysis.model_analysis.verify_flyview_odometry --negate-flow
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from drosophila_model.config import load_config
from drosophila_model.model_definition.ego_motion import (
    load_viewing_directions,
    estimate_ego_motion_known_depth,
)

AXIS_LABELS = ["x (roll)", "y (pitch)", "z (yaw)"]
RAD2DEG = 180.0 / np.pi


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _log_vee(R):
    """Rotation-vector (axis * angle) of a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-9:
        return 0.5 * w
    return angle / (2.0 * np.sin(angle)) * w


def reconstruct_odometry(flow, depth, phi, theta, fps, flow_flip, swap_flow=False):
    """Solve (rotation deg/s, translation units/s) from flow + depth.

    Mirrors ``bake_fly_view_dataset``: flow is rad/frame, optionally sign-flipped
    (the bake default), scaled to rad/s, and fed with per-column depth into the
    closed-form known-depth solver. ``swap_flow`` exchanges the d_right / d_up
    components before solving (diagnostic for a basis swap).
    """
    bad = np.isnan(flow).any(axis=-1) | (np.hypot(flow[..., 0], flow[..., 1]) > 1.0)
    sign = -1.0 if flow_flip else 1.0
    flow_rad_s = np.nan_to_num(flow, nan=0.0) * fps * sign
    d_right = flow_rad_s[..., 0]
    d_up = flow_rad_s[..., 1]
    if swap_flow:
        d_right, d_up = d_up, d_right
    omega, trans = estimate_ego_motion_known_depth(
        phi, theta,
        torch.from_numpy(d_right),
        torch.from_numpy(d_up),
        torch.from_numpy(depth),
        valid=torch.from_numpy(~bad),
    )
    return omega.numpy() * RAD2DEG, trans.numpy()


def trajectory_odometry(positions, rpy_deg, fps):
    """Body-frame angular velocity (deg/s) and translation (units/s) from the
    renderer trajectory using the repo's ``R = Rz Ry Rx`` body->world convention.

    Fallback for datasets that don't ship ``rotation_rate`` / ``translation_rate``.
    """
    rpy = np.deg2rad(rpy_deg)
    n = len(rpy)
    R = [_Rz(rpy[t, 2]) @ _Ry(rpy[t, 1]) @ _Rx(rpy[t, 0]) for t in range(n)]
    omega = np.zeros((n, 3))
    trans = np.zeros((n, 3))
    for t in range(n - 1):
        omega[t] = _log_vee(R[t].T @ R[t + 1]) * fps * RAD2DEG
        trans[t] = R[t].T @ ((positions[t + 1] - positions[t]) * fps)
    if n > 1:
        omega[-1] = omega[-2]
        trans[-1] = trans[-2]
    return omega, trans


def file_odometry(d, idx, fps):
    """Ground-truth odometry stored in the dataset (``rotation_rate`` deg/s,
    ``translation_rate`` units/s, both body frame). Falls back to deriving it
    from the trajectory if those fields are absent.
    """
    if "rotation_rate" in d and "translation_rate" in d:
        rot = np.asarray(d["rotation_rate"][idx], dtype=np.float64)
        trans = np.asarray(d["translation_rate"][idx], dtype=np.float64)
        return rot, trans, "file (truth)"
    positions = np.asarray(d["positions"][idx], dtype=np.float64)
    rpy_deg = np.asarray(d["roll_pitch_yaw"][idx], dtype=np.float64)
    rot, trans = trajectory_odometry(positions, rpy_deg, fps)
    return rot, trans, "trajectory (truth)"


def _corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def plot_clip(clip_idx, recon_rot, recon_trans, true_rot, true_trans, dt, out_dir,
              tag="", truth_label="truth"):
    n = recon_rot.shape[0]
    time = np.arange(n) * dt
    sl = slice(1, -1) if n > 2 else slice(None)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(time, true_rot[:, i], label=truth_label, linewidth=1.2)
        ax.plot(time, recon_rot[:, i], label="reconstructed", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(f"Rotation {AXIS_LABELS[i]} (deg/s)")
        r = _corr(recon_rot[sl, i], true_rot[sl, i])
        ax.set_title(f"r = {r:+.3f}", fontsize=9, loc="right")
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Rotation (deg/s)", loc="left")

        ax = axes[i, 1]
        ax.plot(time, true_trans[:, i], label=truth_label, linewidth=1.2)
        ax.plot(time, recon_trans[:, i], label="reconstructed", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(f"Translation {AXIS_LABELS[i]} (units/s)")
        r = _corr(recon_trans[sl, i], true_trans[sl, i])
        ax.set_title(f"r = {r:+.3f}", fontsize=9, loc="right")
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Translation (units/s)", loc="left")

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    title = f"Fly-view odometry: flow reconstruction vs {truth_label} — clip {clip_idx}"
    if tag:
        title += f" [{tag}]"
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"flyview_odometry_clip_{clip_idx}{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate fly-view odometry reconstruction.")
    parser.add_argument("--clips", type=int, nargs="+", default=None,
                        help="Clip indices to plot (default: 0 5 10 15).")
    parser.add_argument("--negate-flow", action="store_true",
                        help="Negate both flow components before solving (diagnostic; "
                             "the bake no longer inverts).")
    parser.add_argument("--swap-flow", action="store_true",
                        help="Swap the d_right / d_up flow components before solving.")
    parser.add_argument("--config-path", type=str, default=None)
    args = parser.parse_args()
    flow_flip = args.negate_flow

    import pickle
    config = load_config(args.config_path or "config.toml")
    data_folder = Path(config["data"]["data_folder"])
    pkl_path = data_folder / "stimulus" / "raw" / "fib_forest_fly_view_dataset_fwd.pkl"
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    fps = float(d["fps"])
    dt = 1.0 / fps
    phi, theta = load_viewing_directions()
    n_clips = len(d["responses"])
    clips = args.clips if args.clips is not None else [c for c in (0, 5, 10, 15) if c < n_clips]

    out_dir = Path(__file__).resolve().parents[3] / "figures" / "verify_flyview_odometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"fps={fps:.4f}, flow_flip={flow_flip}, {n_clips} clips total -> {out_dir}")

    # Aggregate per-axis gain (recon ~ gain * truth) across the plotted clips.
    rot_recon_all, rot_true_all, tr_recon_all, tr_true_all = [], [], [], []

    truth_label = "truth"
    for idx in clips:
        flow = np.asarray(d["forward_flows"][idx], dtype=np.float64)
        depth = np.asarray(d["depths"][idx], dtype=np.float64)
        if flow.shape[0] < 3:
            print(f"  Clip {idx}: too short, skipping.")
            continue

        recon_rot, recon_trans = reconstruct_odometry(
            flow, depth, phi, theta, fps, flow_flip, swap_flow=args.swap_flow)
        true_rot, true_trans, truth_label = file_odometry(d, idx, fps)
        plot_clip(idx, recon_rot, recon_trans, true_rot, true_trans, dt, out_dir,
                  tag="swap" if args.swap_flow else "", truth_label=truth_label)

        sl = slice(1, -1)
        rot_recon_all.append(recon_rot[sl])
        rot_true_all.append(true_rot[sl])
        tr_recon_all.append(recon_trans[sl])
        tr_true_all.append(true_trans[sl])

    if rot_recon_all:
        rr = np.concatenate(rot_recon_all)
        rt = np.concatenate(rot_true_all)
        tr = np.concatenate(tr_recon_all)
        tt = np.concatenate(tr_true_all)
        print("\nAggregate over plotted clips:")
        for i in range(3):
            gain = np.polyfit(rt[:, i], rr[:, i], 1)[0]
            print(f"  rotation {AXIS_LABELS[i]:<10} corr={_corr(rr[:, i], rt[:, i]):+.3f}  "
                  f"gain(recon/truth)={gain:+.3f}")
        for i in range(3):
            gain = np.polyfit(tt[:, i], tr[:, i], 1)[0]
            print(f"  translation {AXIS_LABELS[i]:<7} corr={_corr(tr[:, i], tt[:, i]):+.3f}  "
                  f"gain(recon/truth)={gain:+.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
