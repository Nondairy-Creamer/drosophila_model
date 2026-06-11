"""Process drone video datasets into preprocessed tensors.

Two datasets are supported, each producing a list of ommatidium-sampled clips:

* ``small_drone`` — equirectangular stereo rig in the ``drone_videos`` folder.
  Left/right videos plus batch-optimised pose from ``drone_pose.py`` (run on
  demand); pixels map to angles through a linear FOV grid.
* ``rpi_drone`` — real two-camera (dome) rig at
  ``<data_folder>/stimulus/rpi_drone``, one ``recording_*`` folder per clip.
  One video per eye, sampled at the fly ommatidia through the per-eye dome
  calibration in ``rpi_drone/dome_calib`` (``dome_right.csv`` / ``dome_left.csv``,
  columns x, y, theta_deg, phi_deg), rotated into the body frame. Ground truth is
  the angular velocity (``gyro_x/y/z`` in ``imu.csv``) and linear velocity /
  position (``vx/vy/vz_mps``, ``x/y/z_m`` in ``estimate.csv``), aligned to video
  frames by timestamp.

``main`` calls a dedicated function per dataset and saves their outputs to a
single ``drone_data.pt`` keyed by dataset name:
``{'small_drone': [...], 'rpi_drone': [...]}``.

Usage:
    python drone_analysis/process_drone.py
    python drone_analysis/process_drone.py --data-dir <data_folder>/stimulus/drone_videos
"""

import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path
import argparse
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from drosophila_model.utilities.stimulus_utilities import (
    build_ommatidium_sampling_matrix,
    build_calibrated_sampling_matrix,
)
from drosophila_model.config import load_config

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
    from drosophila_model.analysis.drone_analysis.drone_pose import process_clip as estimate_clip, save_clip_result, get_image_size
    from drosophila_model.analysis.drone_analysis.estimate_pose import EquirectRig

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


def process_small_drone_dataset(data_dir, pose_dir, sample_phi, sample_theta,
                                filter_std=2.5, monocular=False):
    """Process the equirectangular ``small_drone`` dataset into sampled clips.

    Reads the ``drone_videos`` folder: discovers ``left_N.mp4`` / ``right_N.mp4``
    pairs, loads (or estimates) pose,
    builds linear-FOV ommatidium sampling matrices, and returns a list of
    sampled clip dicts. In ``monocular`` mode only the right videos/pose-free
    path is used.

    Args:
        data_dir: directory containing the stereo videos.
        pose_dir: directory of batch-opt pose ``.npz`` files from drone_pose.py.
        sample_phi, sample_theta: (N,) ommatidium angles (radians).
        filter_std: Gaussian filter std in degrees.
        monocular: process only right videos (no left, no pose).

    Returns:
        List of sampled clip dicts (empty if no clips are found).
    """
    data_dir = Path(data_dir)

    # Camera field-of-view limits (degrees) for the equirectangular rig.
    phi_right_lim = np.array((-30, -150)) * np.pi / 180
    phi_left_lim = np.array((150, 30)) * np.pi / 180
    theta_lim = np.array((45, 135)) * np.pi / 180

    right_ids = set()
    for p in data_dir.glob('right_*.mp4'):
        try:
            right_ids.add(int(p.stem.split('_')[1]))
        except (ValueError, IndexError):
            continue
    if monocular:
        clip_ids = sorted(right_ids)
    else:
        left_ids = set()
        for p in data_dir.glob('left_*.mp4'):
            try:
                left_ids.add(int(p.stem.split('_')[1]))
            except (ValueError, IndexError):
                continue
        clip_ids = sorted(left_ids & right_ids)
    print(f'[small_drone] Found {len(clip_ids)} clips: {clip_ids}')

    raw_clips = []
    for clip_id in clip_ids:
        print(f'[small_drone] Loading clip {clip_id}...')
        clip = process_clip(clip_id, data_dir, pose_dir, monocular=monocular)
        if clip is not None:
            raw_clips.append(clip)

    if not raw_clips:
        print('[small_drone] No valid clips found.')
        return []

    _, image_height, image_width = raw_clips[0]['right'].shape
    print(f'[small_drone] Building sampling matrices for {image_height}x{image_width} frames...')
    W_right = build_ommatidium_sampling_matrix(
        (image_height, image_width), phi_right_lim, theta_lim,
        sample_phi, sample_theta, filter_std=filter_std,
    )
    in_fov_right = np.asarray(W_right.sum(axis=1)).ravel() > 0

    if monocular:
        W_left, in_fov_left = None, None
    else:
        W_left = build_ommatidium_sampling_matrix(
            (image_height, image_width), phi_left_lim, theta_lim,
            -sample_phi, sample_theta, filter_std=filter_std,
        )
        in_fov_left = np.asarray(W_left.sum(axis=1)).ravel() > 0

    clips = []
    for raw_clip in raw_clips:
        print(f'[small_drone] Sampling clip {raw_clip["clip_id"]}...')
        clips.append(sample_clip(raw_clip, W_right, in_fov_right, W_left, in_fov_left))
    return clips


def _rot_y(deg):
    """Rotation matrix about the +y axis by ``deg`` degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(deg):
    """Rotation matrix about the +z axis by ``deg`` degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_dome_calibration(csv_path, phi_offset_deg):
    """Load a dome-calibration CSV and rotate it into the body frame.

    The cleaned calibration CSVs (``dome_right.csv`` / ``dome_left.csv``) give,
    per labelled pixel: ``x`` (horizontal, +right), ``y`` (vertical, +down),
    ``theta_deg`` (polar angle from the image-center pole = optical axis), and
    ``phi_deg`` (azimuth around that pole). These describe directions in the
    camera frame where the optical axis is the north pole (+z).

    The camera is then placed in the body frame (+x forward, +y left, +z up):
    the optical axis is rotated +90 deg in theta (from straight up to the
    horizon) and by ``phi_offset_deg`` in phi. Per the rig geometry the right
    eye uses -80 deg (faces right, -y, with a 10 deg inward/forward tilt) and
    the left eye uses +80 deg (faces left, +y, same tilt). After this the angles
    are in the standard convention used everywhere else.

    Returns ``(u, v, phi, theta)``: pixel coords and body-frame angles (radians).
    """
    df = pd.read_csv(csv_path)
    u = df['x'].to_numpy(dtype=float)
    v = df['y'].to_numpy(dtype=float)
    theta_cam = np.radians(df['theta_deg'].to_numpy(dtype=float))  # polar from pole
    phi_cam = np.radians(df['phi_deg'].to_numpy(dtype=float))      # azimuth around pole

    d_cam = np.column_stack([
        np.sin(theta_cam) * np.cos(phi_cam),
        np.sin(theta_cam) * np.sin(phi_cam),
        np.cos(theta_cam),
    ])
    rotation = _rot_z(phi_offset_deg) @ _rot_y(90.0)
    d_body = d_cam @ rotation.T

    phi = np.arctan2(d_body[:, 1], d_body[:, 0])
    theta = np.arccos(np.clip(d_body[:, 2], -1.0, 1.0))
    return u, v, phi, theta


def load_rpi_ground_truth(recording_dir, gyro_scale=1.0):
    """Load rpi_drone ground truth aligned to video frames by timestamp.

    Frame timestamps come from ``stereo_frames.csv`` (one row per video frame).
    Angular velocity (``gyro_x/y/z`` in ``imu.csv``) and position / linear
    velocity (``x/y/z_m``, ``vx/vy/vz_mps`` in ``estimate.csv``) are linearly
    interpolated onto those frame times using each file's ``time_s`` column, so
    differing sample rates between the IMU, the state estimate, and the video
    are handled correctly.

    Axes are taken to already be in the body frame (+x forward, +y left,
    +z up) used by the model; adjust here if the rig differs.

    Args:
        recording_dir: directory with imu.csv, estimate.csv, stereo_frames.csv.
        gyro_scale: multiplier converting raw gyro units to deg/s (the
            ``rotation`` rate convention used by the rest of the pipeline).
            The raw IMU units are unknown; defaults to 1.0 (treated as deg/s).

    Returns:
        rotation_rate: (T, 3) angular velocity in deg/s.
        translation_rate: (T, 3) linear velocity in m/s.
        positions: (T, 3) position in m.
        frame_times: (T,) video frame times in seconds.
    """
    recording_dir = Path(recording_dir)
    frames = pd.read_csv(recording_dir / 'stereo_frames.csv')
    frames = frames.sort_values('video_frame_index')
    frame_times = frames['time_s'].to_numpy(dtype=float)

    imu = pd.read_csv(recording_dir / 'imu.csv')
    est = pd.read_csv(recording_dir / 'estimate.csv')

    def interp_cols(df, cols):
        t = df['time_s'].to_numpy(dtype=float)
        return np.column_stack([
            np.interp(frame_times, t, df[c].to_numpy(dtype=float)) for c in cols
        ])

    rotation_rate = interp_cols(imu, ['gyro_x', 'gyro_y', 'gyro_z']) * gyro_scale
    translation_rate = interp_cols(est, ['vx_mps', 'vy_mps', 'vz_mps'])
    positions = interp_cols(est, ['x_m', 'y_m', 'z_m'])
    return rotation_rate, translation_rate, positions, frame_times


def process_rpi_drone_dataset(data_dir, calib_dir, sample_phi, sample_theta,
                              filter_std=2.5, gyro_scale=1.0):
    """Process the ``rpi_drone`` two-camera dome dataset into sampled clips.

    Each ``recording_*`` subfolder of ``data_dir`` is one clip with ``right.mp4``
    / ``left.mp4`` plus ``imu.csv`` / ``estimate.csv`` / ``stereo_frames.csv``.
    Fly ommatidia are mapped into each video through the per-eye dome
    calibration in ``calib_dir`` (``dome_right.csv`` / ``dome_left.csv``), which
    is rotated into the body frame by :func:`load_dome_calibration`.

    The right eye is sampled at the right-eye columns ``(sample_phi,
    sample_theta)`` and the left eye at their sagittal mirror ``(-sample_phi,
    sample_theta)``.

    Returns:
        List of sampled clip dicts (one per recording).
    """
    data_dir = Path(data_dir)
    calib_dir = Path(calib_dir)

    recording_dirs = sorted(p for p in data_dir.glob('recording_*') if p.is_dir())
    if not recording_dirs:
        print(f'[rpi_drone] No recording_* folders found in {data_dir}.')
        return []
    print(f'[rpi_drone] Found {len(recording_dirs)} recordings.')

    # Right/left calibration rotated into the body frame (rig geometry: right
    # eye faces -y with 10 deg inward tilt, left eye faces +y).
    cu_r, cv_r, cphi_r, cth_r = load_dome_calibration(calib_dir / 'dome_right.csv', -80.0)
    cu_l, cv_l, cphi_l, cth_l = load_dome_calibration(calib_dir / 'dome_left.csv', 80.0)

    sampler_cache = {}

    def get_samplers(image_height, image_width):
        key = (image_height, image_width)
        if key not in sampler_cache:
            print(f'[rpi_drone] Building calibrated sampling matrices for '
                  f'{image_height}x{image_width} frames...')
            W_right, in_fov_right = build_calibrated_sampling_matrix(
                key, cu_r, cv_r, cphi_r, cth_r,
                sample_phi, sample_theta, filter_std=filter_std,
            )
            W_left, in_fov_left = build_calibrated_sampling_matrix(
                key, cu_l, cv_l, cphi_l, cth_l,
                -sample_phi, sample_theta, filter_std=filter_std,
            )
            sampler_cache[key] = (W_right, in_fov_right, W_left, in_fov_left)
        return sampler_cache[key]

    clips = []
    for rec in recording_dirs:
        print(f'[rpi_drone] Loading {rec.name}...')
        right = load_video_grayscale(rec / 'right.mp4')
        left = load_video_grayscale(rec / 'left.mp4')

        cap = cv2.VideoCapture(str(rec / 'right.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        video_dt = 1.0 / float(fps)

        rotation_rate, translation_rate, positions, _ = load_rpi_ground_truth(
            rec, gyro_scale=gyro_scale,
        )

        n_frames = min(len(right), len(left), len(rotation_rate))
        right = right[:n_frames]
        left = left[:n_frames]
        rotation_rate = rotation_rate[:n_frames]
        translation_rate = translation_rate[:n_frames]
        positions = positions[:n_frames]

        raw_clip = {
            'right': torch.from_numpy(right),
            'left': torch.from_numpy(left),
            'rotation': torch.from_numpy(rotation_rate.astype(np.float32)),
            'translation': torch.from_numpy(translation_rate.astype(np.float32)),
            'translation_location': torch.from_numpy(positions.astype(np.float32)),
            'video_dt': video_dt,
            'clip_id': rec.name,
        }
        print(f'[rpi_drone] {rec.name}: {n_frames} frames ({n_frames * video_dt:.1f}s)')

        W_right, in_fov_right, W_left, in_fov_left = get_samplers(
            right.shape[1], right.shape[2])
        clips.append(sample_clip(raw_clip, W_right, in_fov_right, W_left, in_fov_left))
    return clips


def _print_dataset_summary(name, clips):
    """Print a per-clip frame/ommatidium/duration summary for one dataset."""
    if not clips:
        print(f'{name}: no clips')
        return
    N = clips[0]['right'].shape[1]
    print(f'{name}: {len(clips)} clips')
    print(f'{"Clip":>6} {"Frames":>8} {"Ommatidia":>10} {"Duration":>10}')
    for clip in clips:
        T = clip['right'].shape[0]
        dt = clip['video_dt']
        print(f'{clip["clip_id"]:>6} {T:>8} {N:>10} {T*dt:>9.1f}s')


def main():
    parser = argparse.ArgumentParser(description='Process drone video datasets into tensors')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='drone_videos directory with left_N.mp4, right_N.mp4 '
                             '(default: <data_folder>/stimulus/drone_videos)')
    parser.add_argument('--pose-dir', type=str, default=None,
                        help='Directory with batch-opt pose .npz files from drone_pose.py '
                             '(default: <data_folder>/stimulus/drone_poses)')
    parser.add_argument('--rpi-dir', type=str, default=None,
                        help='rpi_drone directory of recording_* folders '
                             '(default: <data_folder>/stimulus/rpi_drone)')
    parser.add_argument('--calib-dir', type=str, default=None,
                        help='Dome calibration directory with dome_right.csv, dome_left.csv '
                             '(default: <rpi-dir>/dome_calib)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for the .pt file '
                             '(default: <data_folder>/stimulus)')
    parser.add_argument('--column-info', type=str, default=None,
                        help='Path to optic-lobe column extras parquet with phi/theta '
                             '(default: <data_folder>/connectome_data/baked/optic_lobe_columns.parquet)')
    parser.add_argument('--filter-std', type=float, default=2.5,
                        help='Gaussian filter std in degrees for ommatidium sampling')
    parser.add_argument('--monocular', action='store_true',
                        help='Process drone_videos right_*.mp4 only (no left video, no pose). '
                             'Output is saved as drone_data_monocular.pt.')
    args = parser.parse_args()

    data_folder = Path(load_config()['data']['data_folder'])
    drone_videos_dir = Path(args.data_dir) if args.data_dir else data_folder / 'stimulus' / 'drone_videos'
    pose_dir = Path(args.pose_dir) if args.pose_dir else data_folder / 'stimulus' / 'drone_poses'
    rpi_dir = Path(args.rpi_dir) if args.rpi_dir else data_folder / 'stimulus' / 'rpi_drone'
    calib_dir = Path(args.calib_dir) if args.calib_dir else rpi_dir / 'dome_calib'
    output_dir = Path(args.output_dir) if args.output_dir else data_folder / 'stimulus'
    column_info_path = (
        Path(args.column_info) if args.column_info
        else data_folder / 'connectome_data' / 'baked' / 'optic_lobe_columns.parquet'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    column_info_df = pd.read_parquet(column_info_path)
    sample_phi = column_info_df['phi'].to_numpy()
    sample_theta = column_info_df['theta'].to_numpy()
    print(f'Loaded {len(sample_phi)} ommatidium positions from {column_info_path}')

    # Monocular mode keeps its own single-dataset (list) output for back-compat.
    if args.monocular:
        clips = process_small_drone_dataset(
            drone_videos_dir, pose_dir, sample_phi, sample_theta,
            filter_std=args.filter_std, monocular=True,
        )
        if not clips:
            print('No valid clips found.')
            return
        out_path = output_dir / 'drone_data_monocular.pt'
        torch.save(clips, out_path)
        print(f'\n{"="*60}')
        print(f'Saved to {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)')
        _print_dataset_summary('small_drone (monocular)', clips)
        return

    datasets = {}
    datasets['small_drone'] = process_small_drone_dataset(
        drone_videos_dir, pose_dir, sample_phi, sample_theta,
        filter_std=args.filter_std,
    )

    if rpi_dir.exists():
        try:
            datasets['rpi_drone'] = process_rpi_drone_dataset(
                rpi_dir, calib_dir, sample_phi, sample_theta,
                filter_std=args.filter_std,
            )
        except Exception as exc:  # keep small_drone output even if rpi fails
            print(f'[rpi_drone] Skipping: {exc}')
    else:
        print(f'[rpi_drone] Directory not found at {rpi_dir}, skipping.')

    if not any(datasets.values()):
        print('No valid clips found in any dataset.')
        return

    out_path = output_dir / 'drone_data.pt'
    torch.save(datasets, out_path)
    size_mb = out_path.stat().st_size / 1e6

    print(f'\n{"="*60}')
    print(f'Saved datasets {sorted(datasets)} to {out_path} ({size_mb:.0f} MB)')
    for name, clips in datasets.items():
        _print_dataset_summary(name, clips)


if __name__ == '__main__':
    # test_odometry()
    main()
