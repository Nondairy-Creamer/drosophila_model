"""Compare three drone pose estimation methods.

1. CSV: Old per-camera AprilTag solvePnP estimates (averaged L+R)
2. Trig: Trigonometric estimates from tag angular positions + optic flow
3. Batch opt: New joint optimization estimates from drone_pose.py

Usage:
    python drone_analysis/compare_pose.py
    python drone_analysis/compare_pose.py --clips 32
"""

import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
import argparse

from drosophila_model.utilities import data_utilities as du
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.spatial.transform import Rotation

from drosophila_model.analysis.drone_analysis.process_drone import extract_odometry
from drosophila_model.utilities.ego_motion import (
    integrate_angular_velocity, integrate_translational_velocity,
)


PHI_RIGHT_DEG = (-30, -150)
PHI_LEFT_DEG = (150, 30)
THETA_DEG = (45, 135)
BOX_FILTER_SIZE = 4


# ── Utilities (copied from process_drone, will be removed there) ─────────

def pixels_to_angles(u_px, v_px, image_width, image_height, phi_lim, theta_lim):
    """Convert pixel coordinates to spherical angles."""
    phi = phi_lim[0] + (u_px / (image_width - 1)) * (phi_lim[1] - phi_lim[0])
    theta = theta_lim[0] + (v_px / (image_height - 1)) * (theta_lim[1] - theta_lim[0])
    return phi, theta


def _rotation_angular_rates(phi_arr, theta_arr, omega):
    """Rotation-induced angular rates at per-timestep target positions."""
    st, ct = np.sin(theta_arr), np.cos(theta_arr)
    sp, cp = np.sin(phi_arr), np.cos(phi_arr)
    x = np.column_stack([st * cp, st * sp, ct])
    e_phi = np.column_stack([-sp, cp, np.zeros_like(phi_arr)])
    e_theta = np.column_stack([ct * cp, ct * sp, -st])
    u_rot = np.cross(x, omega)
    dtheta_rot = np.sum(u_rot * e_theta, axis=1)
    dphi_rot = np.sum(u_rot * e_phi, axis=1) / st
    return dphi_rot, dtheta_rot


# ── Data loaders ─────────────────────────────────────────────────────────

def load_csv_pose(data_dir: Path, clip_id: int) -> dict | None:
    """Load old per-camera AprilTag pose estimates from CSV.

    Reads apriltag_pose_log_{clip_id}_both.csv, averages left/right
    camera estimates, and zero-references to the first frame.

    Returns dict with time_s, position (T, 3), rotation_deg (T, 3).
    """
    csv_path = data_dir / f'apriltag_pose_log_{clip_id}_both.csv'
    if not csv_path.exists():
        print(f'  No CSV found: {csv_path}')
        return None

    pose = pd.read_csv(csv_path)
    time_s = pose['t_video_s'].values

    position = np.column_stack([
        0.5 * (pose['R_x_used'].values + pose['L_x_used'].values),
        0.5 * (pose['R_y_used'].values + pose['L_y_used'].values),
        0.5 * (pose['R_z_used'].values + pose['L_z_used'].values),
    ])

    rotation_deg = np.column_stack([
        0.5 * (pose['R_roll_used_deg'].values + pose['L_roll_used_deg'].values),
        0.5 * (pose['R_pitch_used_deg'].values + pose['L_pitch_used_deg'].values),
        0.5 * (pose['R_yaw_used_deg'].values + pose['L_yaw_used_deg'].values),
    ])

    position -= position[0]
    rotation_deg -= rotation_deg[0]

    return {'time_s': time_s, 'position': position, 'rotation_deg': rotation_deg}


def compute_trig_estimate(data_dir: Path, clip_id: int,
                          image_size: tuple[int, int]) -> dict | None:
    """Compute trigonometric pose estimate from tag angular positions.

    Uses optic-flow rotation estimation + trigonometric translation from
    rotation-corrected tag angles.  Integrates rates to cumulative
    position and rotation (starting at zero).

    Returns dict with time_s, position (T, 3), rotation_deg (T, 3).
    """
    csv_path = data_dir / f'apriltag_pose_log_{clip_id}_both.csv'
    if not csv_path.exists():
        print(f'  No CSV for trig estimate: {csv_path}')
        return None

    pose = pd.read_csv(csv_path)
    video_dt = float(pose['t_video_s'].iloc[1] - pose['t_video_s'].iloc[0])
    time_s = pose['t_video_s'].values

    image_height, image_width = image_size
    phi_right_lim = np.array(PHI_RIGHT_DEG) * np.pi / 180
    phi_left_lim = np.array(PHI_LEFT_DEG) * np.pi / 180
    theta_lim = np.array(THETA_DEG) * np.pi / 180

    L_phi, L_theta = pixels_to_angles(
        pose['L_u_used_px'].values, pose['L_v_used_px'].values,
        image_width, image_height, phi_left_lim, theta_lim)
    R_phi, R_theta = pixels_to_angles(
        pose['R_u_used_px'].values, pose['R_v_used_px'].values,
        image_width, image_height, phi_right_lim, theta_lim)

    L_phi = uniform_filter1d(L_phi, size=BOX_FILTER_SIZE)
    L_theta = uniform_filter1d(L_theta, size=BOX_FILTER_SIZE)
    R_phi = uniform_filter1d(R_phi, size=BOX_FILTER_SIZE)
    R_theta = uniform_filter1d(R_theta, size=BOX_FILTER_SIZE)

    dL_phi = np.gradient(L_phi, video_dt)
    dL_theta = np.gradient(L_theta, video_dt)
    dR_phi = np.gradient(R_phi, video_dt)
    dR_theta = np.gradient(R_theta, video_dt)

    d_right_L = -dL_phi * np.sin(L_theta)
    d_up_L = -dL_theta
    d_right_R = -dR_phi * np.sin(R_theta)
    d_up_R = -dR_theta

    phi = torch.tensor([L_phi.mean(), R_phi.mean()], dtype=torch.float32)
    theta = torch.tensor([L_theta.mean(), R_theta.mean()], dtype=torch.float32)
    d_right = torch.tensor(
        np.stack([d_right_L, d_right_R], axis=1), dtype=torch.float32)
    d_up = torch.tensor(
        np.stack([d_up_L, d_up_R], axis=1), dtype=torch.float32)

    R_y = uniform_filter1d(
        pose['R_y_used'].values.astype(np.float64), size=BOX_FILTER_SIZE)
    L_y = uniform_filter1d(
        pose['L_y_used'].values.astype(np.float64), size=BOX_FILTER_SIZE)
    R_x = uniform_filter1d(
        pose['R_x_used'].values.astype(np.float64), size=BOX_FILTER_SIZE)
    L_x = uniform_filter1d(
        pose['L_x_used'].values.astype(np.float64), size=BOX_FILTER_SIZE)
    dR_y = np.gradient(R_y, video_dt)
    dL_y = np.gradient(L_y, video_dt)
    T_y_known = torch.tensor(0.5 * (dR_y + dL_y), dtype=torch.float32)

    R_pitch_rad = uniform_filter1d(
        pose['R_pitch_used_deg'].values.astype(np.float64),
        size=BOX_FILTER_SIZE) * (np.pi / 180)
    L_pitch_rad = uniform_filter1d(
        pose['L_pitch_used_deg'].values.astype(np.float64),
        size=BOX_FILTER_SIZE) * (np.pi / 180)
    dR_pitch = np.gradient(R_pitch_rad, video_dt)
    dL_pitch = np.gradient(L_pitch_rad, video_dt)
    omega_y_known = torch.tensor(0.5 * (dR_pitch + dL_pitch), dtype=torch.float32)

    rotation, translation, _ = extract_odometry(
        phi, theta, d_right, d_up, omega_y_known, T_y_known)

    translation[:, 0] *= -1
    translation[:, 2] *= -1

    rotation_rad = rotation.numpy().copy()
    rotation_deg_s = rotation.numpy() * (180.0 / np.pi)

    # Subtract flow-estimated rotation from angular positions
    dR_phi_rot, dR_theta_rot = _rotation_angular_rates(
        R_phi, R_theta, rotation_rad)
    dL_phi_rot, dL_theta_rot = _rotation_angular_rates(
        L_phi, L_theta, rotation_rad)
    R_theta_corr = R_theta - cumulative_trapezoid(
        dR_theta_rot, dx=video_dt, initial=0)
    R_phi_corr = R_phi - cumulative_trapezoid(
        dR_phi_rot, dx=video_dt, initial=0)
    L_theta_corr = L_theta - cumulative_trapezoid(
        dL_theta_rot, dx=video_dt, initial=0)
    L_phi_corr = L_phi - cumulative_trapezoid(
        dL_phi_rot, dx=video_dt, initial=0)

    # Trig translation rates: dZ from polar angle, dX from azimuthal angle
    trig_dZ_R = (np.abs(R_y)
                 * np.gradient(1.0 / np.tan(np.pi - R_theta_corr), video_dt))
    trig_dX_R = (R_x
                 * np.gradient(1.0 / np.tan(np.pi - np.abs(R_phi_corr)), video_dt))
    trig_dZ_L = (np.abs(L_y)
                 * np.gradient(1.0 / np.tan(np.pi - L_theta_corr), video_dt))
    trig_dX_L = (L_x
                 * np.gradient(1.0 / np.tan(np.pi - np.abs(L_phi_corr)), video_dt))
    trig_trans_right = np.column_stack([trig_dX_R, dR_y, trig_dZ_R])
    trig_trans_left = np.column_stack([trig_dX_L, dL_y, trig_dZ_L])

    gauss_sigma = 0.15 / video_dt
    trig_trans_right = gaussian_filter1d(
        trig_trans_right, sigma=gauss_sigma, axis=0, truncate=2.0)
    trig_trans_left = gaussian_filter1d(
        trig_trans_left, sigma=gauss_sigma, axis=0, truncate=2.0)

    trans_rate = 0.5 * (trig_trans_right + trig_trans_left)

    # Integrate rates → cumulative position / rotation (start at zero)
    position = np.column_stack([
        cumulative_trapezoid(trans_rate[:, i], dx=video_dt, initial=0)
        for i in range(3)
    ])
    rotation_deg = np.column_stack([
        cumulative_trapezoid(rotation_deg_s[:, i], dx=video_dt, initial=0)
        for i in range(3)
    ])

    return {'time_s': time_s, 'position': position, 'rotation_deg': rotation_deg}


def load_batch_opt(pose_dir: Path, clip_id: int) -> dict | None:
    """Load batch-optimised pose estimate from .npz file.

    Integrates the stored velocity signals using SO(3)-aware integration
    from neural_models to obtain position and rotation.

    Returns dict with fps, position (T, 3), rotation_deg (T, 3).
    """
    npz_path = pose_dir / f'pose_{clip_id}.npz'
    if not npz_path.exists():
        print(f'  No batch-opt .npz found: {npz_path}')
        return None

    data = np.load(npz_path)
    fps = float(data['fps'])
    dt = 1.0 / fps
    n = data['rotation_deg_s'].shape[0]
    time_s = np.arange(n) / fps

    omega_t = torch.tensor(data['rotation_deg_s'], dtype=torch.float64)
    vel_t = torch.tensor(data['velocity_m_s'], dtype=torch.float64)

    rotvec_deg = integrate_angular_velocity(omega_t, dt).numpy()
    rotation_deg = Rotation.from_rotvec(np.deg2rad(rotvec_deg)).as_euler(
        'xyz', degrees=True)
    position = integrate_translational_velocity(vel_t, omega_t, dt).numpy()

    return {
        'time_s': time_s,
        'position': position,
        'rotation_deg': rotation_deg,
    }


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_pose_comparison(clip_id: int,
                         csv_data: dict | None,
                         trig_data: dict | None,
                         batch_data: dict | None,
                         plot_dir: Path):
    """Plot up to three pose estimates on a 3x2 grid.

    Left column: translation X, Y, Z.  Right column: roll, pitch, yaw.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sources = []
    if csv_data is not None:
        sources.append(('CSV (solvePnP)', csv_data, '-', 0.7))
    if trig_data is not None:
        sources.append(('Trig (flow + angles)', trig_data, '--', 0.8))
    if batch_data is not None:
        sources.append(('Batch opt', batch_data, '-', 0.9))

    if not sources:
        print(f'  No data to plot for clip {clip_id}')
        return

    pos_labels = ['X (forward) (m)', 'Y (left) (m)', 'Z (up) (m)']
    rot_labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')

    for label, data, ls, alpha in sources:
        t = data['time_s']
        pos = data['position']
        rot = data['rotation_deg']
        for i in range(3):
            axes[i, 0].plot(t, pos[:, i], label=label,
                            linestyle=ls, linewidth=1.2, alpha=alpha)
            axes[i, 1].plot(t, rot[:, i], label=label,
                            linestyle=ls, linewidth=1.2, alpha=alpha)

    # Shared Y limits across X/Y/Z within each column
    pos_ylims = [axes[i, 0].get_ylim() for i in range(3)]
    pos_lo = min(lo for lo, _ in pos_ylims)
    pos_hi = max(hi for _, hi in pos_ylims)
    rot_ylims = [axes[i, 1].get_ylim() for i in range(3)]
    rot_lo = min(lo for lo, _ in rot_ylims)
    rot_hi = max(hi for _, hi in rot_ylims)

    for i in range(3):
        axes[i, 0].set_ylim(pos_lo, pos_hi)
        axes[i, 0].set_ylabel(pos_labels[i])
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylim(rot_lo, rot_hi)
        axes[i, 1].set_ylabel(rot_labels[i])
        axes[i, 1].grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8, loc='upper right')
    axes[0, 1].legend(fontsize=8, loc='upper right')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[0, 0].set_title('Translation')
    axes[0, 1].set_title('Rotation')
    fig.suptitle(f'Pose Comparison — Clip {clip_id}', fontsize=14)
    plt.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / f'pose_comparison_{clip_id}.png'
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f'  Saved plot → {save_path}')


# ── Helpers ──────────────────────────────────────────────────────────────

def get_image_size(data_dir: Path, clip_id: int) -> tuple[int, int]:
    """Read one frame from right_{clip_id}.mp4 to get (height, width)."""
    cap = cv2.VideoCapture(str(data_dir / f'right_{clip_id}.mp4'))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'Cannot read frame from right_{clip_id}.mp4')
    return frame.shape[0], frame.shape[1]


def find_clip_ids(data_dir: Path) -> list[int]:
    """Find clip IDs that have an apriltag_pose_log CSV."""
    ids = set()
    for p in data_dir.glob('apriltag_pose_log_*_both.csv'):
        parts = p.stem.split('_')
        try:
            ids.add(int(parts[-2]))
        except (ValueError, IndexError):
            continue
    return sorted(ids)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare CSV, trig, and batch-opt pose estimates')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory with videos and CSV files '
                             '(default: <data_folder>/stimulus/drone_videos)')
    parser.add_argument('--pose-dir', type=str, default=None,
                        help='Directory with batch-opt .npz files '
                             '(default: <data_folder>/stimulus/drone_poses)')
    parser.add_argument('--plot-dir', type=str,
                        default='figures/compare_pose',
                        help='Output directory for comparison plots')
    parser.add_argument('--clips', type=int, nargs='+', default=None,
                        help='Process only these clip IDs (default: all)')
    args = parser.parse_args()

    data_folder = Path(du.load_config()['data']['data_folder'])
    data_dir = Path(args.data_dir) if args.data_dir else data_folder / 'stimulus' / 'drone_videos'
    pose_dir = Path(args.pose_dir) if args.pose_dir else data_folder / 'stimulus' / 'drone_poses'
    plot_dir = Path(args.plot_dir)

    all_ids = find_clip_ids(data_dir)
    if not all_ids:
        print(f'No apriltag_pose_log CSVs found in {data_dir}')
        return
    clip_ids = args.clips if args.clips else all_ids
    clip_ids = [c for c in clip_ids if c in all_ids]
    print(f'Found {len(all_ids)} clips, comparing {len(clip_ids)}: {clip_ids}')

    image_size = get_image_size(data_dir, clip_ids[0])
    print(f'Image size: {image_size[1]}x{image_size[0]}')

    for clip_id in clip_ids:
        print(f'\nClip {clip_id}')
        csv_data = load_csv_pose(data_dir, clip_id)
        trig_data = compute_trig_estimate(data_dir, clip_id, image_size)
        batch_data = load_batch_opt(pose_dir, clip_id)
        plot_pose_comparison(clip_id, csv_data, trig_data, batch_data, plot_dir)


if __name__ == '__main__':
    main()
