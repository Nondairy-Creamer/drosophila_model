"""Process drone stereo video data into preprocessed tensors.

Loads each pair of left/right stereo videos, loads batch-optimised pose
estimates from drone_pose.py (running estimation if needed), samples
frames at ommatidium locations, and saves all clips as a single .pt file.

Usage:
    python drone_analysis/process_drone.py
    python drone_analysis/process_drone.py --data-dir ../connectome_data/drone_videos
"""

import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path
import argparse
import sys
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stimulus_utilities import build_ommatidium_sampling_matrix

BOX_FILTER_SIZE = 4


def load_video_grayscale(path):
    """Load video as grayscale uint8 frames."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames)


def pixels_to_angles(u_px, v_px, image_width, image_height, phi_lim, theta_lim):
    """Convert pixel coordinates to spherical angles.

    Args:
        u_px: horizontal pixel positions (array or scalar)
        v_px: vertical pixel positions (array or scalar)
        image_width, image_height: frame dimensions in pixels
        phi_lim: (start, end) azimuthal angle limits in radians
        theta_lim: (start, end) polar angle limits in radians

    Returns:
        phi, theta: angles in radians (same shape as input)
    """
    phi = phi_lim[0] + (u_px / (image_width - 1)) * (phi_lim[1] - phi_lim[0])
    theta = theta_lim[0] + (v_px / (image_height - 1)) * (theta_lim[1] - theta_lim[0])
    return phi, theta


def _rotation_angular_rates(phi_arr, theta_arr, omega):
    """Compute rotation-induced angular rates at per-timestep target positions.

    Args:
        phi_arr: (T,) azimuthal angle of target at each frame
        theta_arr: (T,) polar angle of target at each frame
        omega: (T, 3) rotation [roll, pitch, yaw] in rad/s

    Returns:
        dphi_rot, dtheta_rot: (T,) rotation-induced angular rates
    """
    st, ct = np.sin(theta_arr), np.cos(theta_arr)
    sp, cp = np.sin(phi_arr), np.cos(phi_arr)
    x = np.column_stack([st * cp, st * sp, ct])
    e_phi = np.column_stack([-sp, cp, np.zeros_like(phi_arr)])
    e_theta = np.column_stack([ct * cp, ct * sp, -st])
    u_rot = np.cross(x, omega)
    dtheta_rot = np.sum(u_rot * e_theta, axis=1)
    dphi_rot = np.sum(u_rot * e_phi, axis=1) / st
    return dphi_rot, dtheta_rot


def plot_translation(flow_trans, saved_trans, video_dt, clip_id, save_path):
    """Plot flow-based translation with the final saved (trig avg) overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = flow_trans.shape[0]
    times = np.arange(T) * video_dt
    dim_labels = ['X (forward)', 'Y (lateral)', 'Z (vertical)']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, dim_labels)):
        ax.plot(times, flow_trans[:, i],
                label='Flow', linewidth=1.5, color='#1f77b4')
        ax.plot(times, saved_trans[:, i],
                label='Saved (trig avg)', linewidth=1.5, linestyle='--', color='#ff7f0e')
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(f'Translation — Clip {clip_id}')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f'  Saved translation plot: {save_path}')


def plot_rotation(tag_rot, saved_rot, video_dt, clip_id, save_path):
    """Plot tag-based rotation rates with the saved (flow-estimated) overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = tag_rot.shape[0]
    times = np.arange(T) * video_dt
    dim_labels = ['Roll', 'Pitch', 'Yaw']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, dim_labels)):
        ax.plot(times, tag_rot[:, i],
                label='Tag (avg L+R)', linewidth=1.5, color='#1f77b4')
        ax.plot(times, saved_rot[:, i],
                label='Saved (flow)', linewidth=1.5, linestyle='--', color='#ff7f0e')
        ax.set_ylabel(f'{label} (deg/s)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(f'Rotation — Clip {clip_id}')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f'  Saved rotation plot: {save_path}')


def plot_translation_comparison(flow_trans, trig_right, trig_left,
                                video_dt, clip_id, save_path):
    """Compare flow-based and trigonometric translation estimates.

    Creates 3 vertically stacked subplots (X, Y, Z), each showing
    flow-based, right-eye trig, and left-eye trig translation traces.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = flow_trans.shape[0]
    times = np.arange(T) * video_dt
    dim_labels = ['X (forward)', 'Y (lateral)', 'Z (vertical)']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, dim_labels)):
        ax.plot(times, flow_trans[:, i],
                label='Flow', linewidth=1.5, color='#1f77b4')
        ax.plot(times, trig_right[:, i],
                label='Trig (right eye)', linewidth=1.5, linestyle='--', color='#ff7f0e')
        ax.plot(times, trig_left[:, i],
                label='Trig (left eye)', linewidth=1.5, linestyle=':', color='#2ca02c')
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(f'Translation Estimates — Clip {clip_id}')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f'  Saved translation comparison plot: {save_path}')


def plot_translation_comparison_avg(flow_trans, trig_avg,
                                    video_dt, clip_id, save_path):
    """Compare flow-based translation with averaged trig estimate.

    Creates 3 vertically stacked subplots (X, Y, Z), each showing
    the flow-based and the left/right averaged trig translation traces.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = flow_trans.shape[0]
    times = np.arange(T) * video_dt
    dim_labels = ['X (forward)', 'Y (lateral)', 'Z (vertical)']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, dim_labels)):
        ax.plot(times, flow_trans[:, i],
                label='Flow', linewidth=1.5, color='#1f77b4')
        ax.plot(times, trig_avg[:, i],
                label='Trig (avg L+R)', linewidth=1.5, linestyle='--', color='#ff7f0e')
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(f'Translation Estimates (averaged eyes) — Clip {clip_id}')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f'  Saved averaged translation comparison plot: {save_path}')


def _load_or_estimate_pose(clip_id, data_dir, pose_dir):
    """Load batch-optimised pose from .npz, running estimation if needed."""
    npz_path = pose_dir / f'pose_{clip_id}.npz'
    if npz_path.exists():
        print(f'  Pose data found: {npz_path}')
        return np.load(npz_path)

    print(f'  No pose data at {npz_path}, running estimation...')
    from drone_pose import process_clip as estimate_clip, save_clip_result, get_image_size
    from estimate_pose import EquirectRig

    image_size = get_image_size(data_dir, clip_id)
    rig = EquirectRig.create_default(image_size)
    result = estimate_clip(
        clip_id, data_dir, rig,
        tag_size=0.16, tag_family='tag36h11',
        smoothness=0.1, world_to_m=1.0,
    )
    if result is None:
        return None

    pose_dir.mkdir(parents=True, exist_ok=True)
    save_clip_result(result, pose_dir)
    return np.load(npz_path)


def process_clip(clip_id, data_dir, pose_dir, monocular=False, default_fps=30.0):
    """Process a single drone clip: load videos and batch-opt pose.

    When ``monocular`` is True, loads only ``right_{clip_id}.mp4`` and skips
    the left video and pose entirely; ``video_dt`` is derived from the video
    file (falls back to ``default_fps`` if the container does not report fps),
    and the returned dict omits ``left``/rotation/translation keys.

    Returns a dict with video tensors and (in stereo mode) rotation/translation
    rates and locations, or ``None`` on failure.
    """
    data_dir = Path(data_dir)
    right = load_video_grayscale(data_dir / f'right_{clip_id}.mp4')

    if monocular:
        cap = cv2.VideoCapture(str(data_dir / f'right_{clip_id}.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
        cap.release()
        video_dt = 1.0 / float(fps)
        print(f'  Clip {clip_id} (monocular): {len(right)} frames ({len(right) * video_dt:.1f}s)')
        return {
            'right': torch.from_numpy(right),
            'video_dt': video_dt,
            'clip_id': clip_id,
        }

    pose_dir = Path(pose_dir)
    left = load_video_grayscale(data_dir / f'left_{clip_id}.mp4')

    pose_data = _load_or_estimate_pose(clip_id, data_dir, pose_dir)
    if pose_data is None:
        return None

    fps = float(pose_data['fps'])
    video_dt = 1.0 / fps

    rotation = torch.from_numpy(pose_data['rotation_deg_s'].astype(np.float32))
    translation = torch.from_numpy(pose_data['velocity_m_s'].astype(np.float32))
    rotation_location = torch.from_numpy(
        pose_data['rotation_deg'].astype(np.float32))
    translation_location = torch.from_numpy(
        pose_data['position_m'].astype(np.float32))

    min_len = min(len(left), len(right), len(rotation))
    left = left[:min_len]
    right = right[:min_len]
    rotation = rotation[:min_len]
    translation = translation[:min_len]
    rotation_location = rotation_location[:min_len]
    translation_location = translation_location[:min_len]

    print(f'  Clip {clip_id}: {min_len} frames ({min_len * video_dt:.1f}s)')

    return {
        'left': torch.from_numpy(left),
        'right': torch.from_numpy(right),
        'rotation': rotation,
        'translation': translation,
        'rotation_location': rotation_location,
        'translation_location': translation_location,
        'video_dt': video_dt,
        'clip_id': clip_id,
    }


def sample_clip(clip, W_right, in_fov_right, W_left=None, in_fov_left=None):
    """Spatially sample a raw clip at ommatidium locations.

    Returns a new dict with 'right' as (T, N) float32 in [0, 1] plus
    video_dt/clip_id. If ``W_left`` and ``in_fov_left`` are provided, also
    samples the left eye and includes rotation/translation fields when
    present on the input clip.
    """
    T = clip['right'].shape[0]

    right_flat = clip['right'].numpy().reshape(T, -1).T.astype(np.float32)
    sampled_right = np.asarray(W_right.dot(right_flat)).T / 255.0
    sampled_right[:, ~in_fov_right] = 0.5

    result = {
        'right': torch.from_numpy(sampled_right),
        'video_dt': clip['video_dt'],
        'clip_id': clip['clip_id'],
        'in_fov_right': torch.from_numpy(in_fov_right.copy()),
    }

    if W_left is not None and in_fov_left is not None:
        left_flat = clip['left'].numpy().reshape(T, -1).T.astype(np.float32)
        sampled_left = np.asarray(W_left.dot(left_flat)).T / 255.0
        sampled_left[:, ~in_fov_left] = 0.5
        result['left'] = torch.from_numpy(sampled_left)
        result['in_fov_left'] = torch.from_numpy(in_fov_left.copy())

    for k in ('rotation', 'translation', 'rotation_location', 'translation_location'):
        if k in clip:
            result[k] = clip[k]
    return result


def extract_odometry(phi, theta, d_right, d_up, omega_y_known, T_y_known):
    """Estimate ego-motion from optic flow with known pitch rotation and lateral translation.

    Solves for roll and yaw rotation (omega_x, omega_z) and forward/vertical
    translation (T_x, T_z) while enforcing known pitch rate (omega_y) and
    lateral translation rate (T_y).

    Coordinate system:
        - +X: Forward (phi=0, theta=pi/2)
        - +Y: Left (phi=pi/2, theta=pi/2)
        - +Z: Up (theta=0)

    Flow sign convention:
        - Positive d_right: rightward motion on retina
        - Positive d_up: upward motion on retina

    Args:
        phi: (N,) azimuthal angles of viewing directions
        theta: (N,) polar angles of viewing directions
        d_right: (N,) or (T, N) angular flow in phi direction
        d_up: (N,) or (T, N) angular flow in theta direction
        omega_y_known: scalar or (T,) known rotation rate about Y axis (pitch)
        T_y_known: scalar or (T,) known translation rate in Y (lateral)

    Returns:
        omega_star: (3,) or (T, 3) rotation vector [roll, pitch, yaw]
        T_star: (3,) or (T, 3) translation vector [T_x, T_y, T_z]
        Z_star: (N,) or (T, N) estimated depths
    """

    def cross_matrix(a):
        ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
        O = torch.zeros_like(ax)
        return torch.stack([
            torch.stack([   O, -az,  ay], dim=-1),
            torch.stack([  az,   O, -ax], dim=-1),
            torch.stack([ -ay,  ax,   O], dim=-1)
        ], dim=-2)

    unbatched = d_right.dim() == 1
    if unbatched:
        d_right = d_right.unsqueeze(0)
        d_up = d_up.unsqueeze(0)

    T_len, N = d_right.shape
    dev = d_right.device
    dt = d_right.dtype

    if not isinstance(omega_y_known, torch.Tensor):
        omega_y_known = torch.tensor(omega_y_known, device=dev, dtype=dt)
    if omega_y_known.dim() == 0:
        omega_y_known = omega_y_known.expand(T_len)

    if not isinstance(T_y_known, torch.Tensor):
        T_y_known = torch.tensor(T_y_known, device=dev, dtype=dt)
    if T_y_known.dim() == 0:
        T_y_known = T_y_known.expand(T_len)

    # 3D viewing directions (N, 3)
    x = torch.stack((
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ), dim=1)

    # Basis vectors on the sphere (N, 3)
    e_theta = torch.stack((
        torch.cos(theta) * torch.cos(phi),
        torch.cos(theta) * torch.sin(phi),
        -torch.sin(theta)
    ), dim=1)

    e_phi = torch.stack((
        -torch.sin(phi),
        torch.cos(phi),
        torch.zeros_like(phi)
    ), dim=1)

    # Reconstruct 3D flow from angular components
    u = -d_right[..., None] * e_phi - d_up[..., None] * e_theta  # (T_len, N, 3)

    # === Rotation: solve for omega_x, omega_z with known omega_y ===
    A = cross_matrix(x)                        # (N, 3, 3)
    A_flat = A.reshape(3 * N, 3)               # (3N, 3)
    u_flat = u.reshape(T_len, 3 * N).T         # (3N, T_len)

    # Subtract known omega_y contribution
    omega_y_contrib = A_flat[:, 1:2] @ omega_y_known.unsqueeze(0)  # (3N, T_len)
    u_adjusted = u_flat - omega_y_contrib

    # Solve reduced 2-variable system for [omega_x, omega_z]
    A_reduced = A_flat[:, [0, 2]]              # (3N, 2)
    AtA = A_reduced.T @ A_reduced              # (2, 2)
    Atb = A_reduced.T @ u_adjusted             # (2, T_len)
    omega_xz = torch.linalg.solve(AtA, Atb).T  # (T_len, 2)

    omega_star = torch.zeros(T_len, 3, device=dev, dtype=dt)
    omega_star[:, 0] = omega_xz[:, 0]
    omega_star[:, 1] = omega_y_known
    omega_star[:, 2] = omega_xz[:, 1]

    # === De-rotated flow ===
    v = u - torch.einsum('nij,tj->tni', A, omega_star)  # (T_len, N, 3)

    # === Translation: SVD for [Tx, Tz] direction, T_y known ===
    translation_magnitude = torch.sqrt((v ** 2).mean(dim=(-2, -1)))  # (T_len,)

    # Projection matrices: Pi_i = I - x_i x_i^T
    outer = x.unsqueeze(2) @ x.unsqueeze(1)   # (N, 3, 3)
    I_mat = torch.eye(3, device=dev, dtype=dt).expand(N, 3, 3)
    P = I_mat - outer                          # (N, 3, 3)

    # C = [v_i]_x @ Pi_i  (eliminates depth)
    vx = cross_matrix(v)                       # (T_len, N, 3, 3)
    C_blocks = vx @ P                          # (T_len, N, 3, 3)
    C = C_blocks.reshape(T_len, 3 * N, 3)     # (T_len, 3N, 3)

    trans_eps = 1e-6
    has_translation = translation_magnitude > trans_eps

    T_star = torch.zeros(T_len, 3, device=dev, dtype=dt)
    T_star[:, 1] = T_y_known

    if has_translation.any():
        C_valid = C[has_translation]

        # SVD on the unknown-component columns to find [Tx, Tz] direction
        C_xz = C_valid[:, :, [0, 2]]                         # (T_v, 3N, 2)
        _, _, Vh = torch.linalg.svd(C_xz, full_matrices=False)
        T_dir_xz = Vh[:, -1]                                 # (T_v, 2)

        # Embed into full 3D for sign disambiguation
        T_dir = torch.zeros(T_dir_xz.shape[0], 3, device=dev, dtype=dt)
        T_dir[:, 0] = T_dir_xz[:, 0]
        T_dir[:, 2] = T_dir_xz[:, 1]

        # Sign disambiguation: v · w > 0 ⟹ positive depths
        w_i = torch.einsum('nij,tj->tni', P, T_dir)          # (T_v, N, 3)
        v_valid = v[has_translation]
        vdw = (v_valid * w_i).sum(dim=-1)                     # (T_v, N)
        wnorm2 = (w_i * w_i).sum(dim=-1)                     # (T_v, N)
        weighted_sign = (vdw * wnorm2).sum(dim=-1)            # (T_v,)
        sign_flip = torch.where(weighted_sign < 0,
                                torch.tensor(-1.0, device=dev, dtype=dt),
                                torch.tensor(1.0, device=dev, dtype=dt))
        T_dir = T_dir * sign_flip[:, None]

        # Scale by translation magnitude
        T_star[has_translation] = T_dir * translation_magnitude[has_translation, None]
        T_star[:, 1] = T_y_known

    # === Depth estimation ===
    w = torch.einsum('nij,tj->tni', P, T_star)  # (T_len, N, 3)
    v_dot_w = (v * w).sum(dim=-1)                # (T_len, N)
    w_norm_sq = (w * w).sum(dim=-1)              # (T_len, N)
    Z_star = w_norm_sq / (v_dot_w + 1e-12)      # (T_len, N)

    if unbatched:
        omega_star = omega_star.squeeze(0)
        T_star = T_star.squeeze(0)
        Z_star = Z_star.squeeze(0)

    return omega_star, T_star, Z_star


def test_odometry():
    """Debug test for extract_odometry with synthetic noiseless flow."""
    print('=== extract_odometry debug test ===')
    torch.manual_seed(42)

    phi = torch.tensor([-90.0, 90.0]) * (np.pi / 180)
    theta = torch.tensor([90.0, 90.0]) * (np.pi / 180)
    N_pts = len(phi)
    T_samples = 10

    omega_true = torch.randn(T_samples, 3) * 10
    T_true = torch.randn(T_samples, 3) * 10
    T_true[:, 1] = torch.randn(T_samples) * 10
    Z_true = torch.rand(T_samples, N_pts) * 9 + 1

    # Viewing directions and tangent-plane basis vectors
    x = torch.stack((
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ), dim=1)

    e_theta = torch.stack((
        torch.cos(theta) * torch.cos(phi),
        torch.cos(theta) * torch.sin(phi),
        -torch.sin(theta),
    ), dim=1)

    e_phi = torch.stack((
        -torch.sin(phi),
        torch.cos(phi),
        torch.zeros_like(phi),
    ), dim=1)

    # Forward model: u = x × omega + (1/Z) * (I - x xᵀ) @ T
    outer = x.unsqueeze(2) @ x.unsqueeze(1)
    P = torch.eye(3).expand(N_pts, 3, 3) - outer

    u_rot = torch.cross(
        x.unsqueeze(0).expand(T_samples, -1, -1),
        omega_true.unsqueeze(1).expand(-1, N_pts, -1),
        dim=-1,
    )
    u_trans = torch.einsum('nij,tj->tni', P, T_true) / Z_true.unsqueeze(-1)
    u = u_rot + u_trans

    d_right = -(u * e_phi.unsqueeze(0)).sum(dim=-1)
    d_up = -(u * e_theta.unsqueeze(0)).sum(dim=-1)

    omega_pred, T_pred, Z_pred = extract_odometry(
        phi, theta, d_right, d_up,
        omega_y_known=omega_true[:, 1],
        T_y_known=T_true[:, 1],
    )

    def cosine_sim(a, b):
        return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)

    rot_cos = cosine_sim(omega_pred, omega_true)
    trans_cos = cosine_sim(T_pred, T_true)

    print(f'  Rotation  cos-sim per sample: {rot_cos.tolist()}')
    print(f'  Translation cos-sim per sample: {trans_cos.tolist()}')
    print(f'  Rotation  cos-sim  mean={rot_cos.mean():.6f}  min={rot_cos.min():.6f}')
    print(f'  Translation cos-sim  mean={trans_cos.mean():.6f}  min={trans_cos.min():.6f}')
    print('=== end debug test ===\n')


def main():
    parser = argparse.ArgumentParser(description='Process drone stereo videos into tensors')
    parser.add_argument('--data-dir', type=str, default='../connectome_data/drone_videos',
                        help='Directory containing left_N.mp4, right_N.mp4')
    parser.add_argument('--pose-dir', type=str, default='../connectome_data/drone_poses',
                        help='Directory with batch-opt pose .npz files from drone_pose.py')
    parser.add_argument('--output-dir', type=str, default='../connectome_data',
                        help='Output directory for the .pt file')
    parser.add_argument('--column-info', type=str, default='../connectome_data/optic_lobes_column_info.parquet',
                        help='Path to column_info parquet with phi/theta')
    parser.add_argument('--filter-std', type=float, default=2.5,
                        help='Gaussian filter std in degrees for ommatidium sampling')
    parser.add_argument('--monocular', action='store_true',
                        help='Process right_*.mp4 only (no left video, no pose). '
                             'Output is saved as drone_data_monocular.pt.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pose_dir = Path(args.pose_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Camera field-of-view limits (degrees)
    phi_right_deg = (-30, -150)
    phi_left_deg = (150, 30)
    theta_deg = (45, 135)

    phi_right_lim = np.array(phi_right_deg) * np.pi / 180
    phi_left_lim = np.array(phi_left_deg) * np.pi / 180
    theta_lim = np.array(theta_deg) * np.pi / 180

    # Load ommatidium positions
    column_info_df = pd.read_parquet(args.column_info)
    sample_phi = column_info_df['phi'].to_numpy()
    sample_theta = column_info_df['theta'].to_numpy()
    print(f'Loaded {len(sample_phi)} ommatidium positions from {args.column_info}')

    # Discover clips from video files
    right_ids = set()
    for p in data_dir.glob('right_*.mp4'):
        try:
            right_ids.add(int(p.stem.split('_')[1]))
        except (ValueError, IndexError):
            continue
    if args.monocular:
        clip_ids = sorted(right_ids)
    else:
        left_ids = set()
        for p in data_dir.glob('left_*.mp4'):
            try:
                left_ids.add(int(p.stem.split('_')[1]))
            except (ValueError, IndexError):
                continue
        clip_ids = sorted(left_ids & right_ids)
    print(f'Found {len(clip_ids)} clips: {clip_ids}')

    raw_clips = []
    for clip_id in clip_ids:
        print(f'Loading clip {clip_id}...')
        clip = process_clip(clip_id, data_dir, pose_dir, monocular=args.monocular)
        if clip is not None:
            raw_clips.append(clip)

    if not raw_clips:
        print('No valid clips found.')
        return

    # Build sampling matrices (all clips share the same resolution)
    _, image_height, image_width = raw_clips[0]['right'].shape

    print(f'Building sampling matrices for {image_height}x{image_width} frames...')
    W_right = build_ommatidium_sampling_matrix(
        (image_height, image_width), phi_right_lim, theta_lim,
        sample_phi, sample_theta, filter_std=args.filter_std,
    )
    in_fov_right = np.asarray(W_right.sum(axis=1)).ravel() > 0

    if args.monocular:
        W_left, in_fov_left = None, None
    else:
        W_left = build_ommatidium_sampling_matrix(
            (image_height, image_width), phi_left_lim, theta_lim,
            -sample_phi, sample_theta, filter_std=args.filter_std,
        )
        in_fov_left = np.asarray(W_left.sum(axis=1)).ravel() > 0

    clips = []
    for raw_clip in raw_clips:
        print(f'Sampling clip {raw_clip["clip_id"]}...')
        clips.append(sample_clip(raw_clip, W_right, in_fov_right, W_left, in_fov_left))

    out_name = 'drone_data_monocular.pt' if args.monocular else 'drone_data.pt'
    out_path = output_dir / out_name
    torch.save(clips, out_path)
    size_mb = out_path.stat().st_size / 1e6

    print(f'\n{"="*60}')
    print(f'Saved {len(clips)} clips to {out_path} ({size_mb:.0f} MB)')
    N = clips[0]['right'].shape[1]
    print(f'{"Clip":>6} {"Frames":>8} {"Ommatidia":>10} {"Duration":>10}')
    for clip in clips:
        T = clip['right'].shape[0]
        dt = clip['video_dt']
        print(f'{clip["clip_id"]:>6} {T:>8} {N:>10} {T*dt:>9.1f}s')


if __name__ == '__main__':
    # test_odometry()
    main()
