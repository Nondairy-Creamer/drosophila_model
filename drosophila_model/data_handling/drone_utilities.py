"""Helper functions for turning raw drone-rig recordings into stimuli.

This module holds all the drone-specific machinery: video loading, pose
loading/estimation, ommatidium sampling, ego-motion solving, dome calibration,
and the per-dataset processors. The thin per-type generators in
``drosophila_model/data_handling/stimulus_generators.py`` (``small_drone`` and
``rpi_drone``) import :func:`get_drone_stimulus` and
:func:`drone_clips_to_two_eye` from here, and analysis scripts import the
lower-level helpers directly.

Two rigs are supported:

* ``small_drone`` — equirectangular stereo rig in the ``drone_videos`` folder.
  Left/right videos plus batch-optimised pose from ``drone_pose.py`` (run on
  demand); pixels map to angles through a linear FOV grid.
* ``rpi_drone`` — real two-camera (dome) rig at
  ``<data_folder>/stimulus/raw/rpi_drone``, one ``recording_*`` folder per clip.
  One video per eye, sampled at the fly ommatidia through the per-eye dome
  calibration in ``rpi_drone/dome_calib`` (``dome_right.csv`` / ``dome_left.csv``).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drosophila_model.data_handling.sampling import (
    build_ommatidium_sampling_matrix,
    build_calibrated_sampling_matrix,
    interpolate_to_model_dt,
)


def load_video_grayscale(path):
    """Load video as (T, H, W) uint8 grayscale frames."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {path}')
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames)


def get_image_size(data_dir, clip_id):
    """Read one frame from ``right_{clip_id}.mp4`` to get ``(height, width)``."""
    import cv2
    cap = cv2.VideoCapture(str(Path(data_dir) / f'right_{clip_id}.mp4'))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'Cannot read frame from right_{clip_id}.mp4')
    return frame.shape[0], frame.shape[1]


def _load_or_estimate_pose(clip_id, data_dir, pose_dir):
    """Load batch-optimised pose from .npz, running estimation if needed."""
    npz_path = pose_dir / f'pose_{clip_id}.npz'
    if npz_path.exists():
        print(f'  Pose data found: {npz_path}')
        return np.load(npz_path)

    print(f'  No pose data at {npz_path}, running estimation...')
    from drosophila_model.data_handling.drone_pose import process_clip as estimate_clip, save_clip_result
    from drosophila_model.data_handling.estimate_pose import EquirectRig

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
    import cv2
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

    A final 180 deg roll about the optical axis is applied: the calibration
    azimuth is measured from image-up, and rotating the pole down to the horizon
    via ``Ry(90)`` would otherwise carry image-up to the ground (the sampled eye
    view comes out rotated 180 deg). The roll keeps scene-up aligned with body
    +z.

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
    rotation = _rot_z(phi_offset_deg) @ _rot_y(90.0) @ _rot_z(180.0)
    d_body = d_cam @ rotation.T

    phi = np.arctan2(d_body[:, 1], d_body[:, 0])
    theta = np.arccos(np.clip(d_body[:, 2], -1.0, 1.0))
    return u, v, phi, theta


def load_rpi_ground_truth(recording_dir):
    """Load rpi_drone ground truth aligned to video frames by timestamp.

    Frame timestamps come from ``stereo_frames.csv`` (one row per video frame).
    Angular velocity (``gyro_x/y/z_dps``), linear velocity
    (``vx/vy/vz_meterpersecond``) and position (``x/y/z_meter``) all come from
    ``matt_data.csv`` and are linearly interpolated onto those frame times using
    its ``time_s`` column, so differing sample rates between the logged ground
    truth and the video are handled correctly.

    The gyro is already in deg/s (matching the ``small_drone`` dataset, which
    stores ``rotation_deg_s``). Axes map to the body frame (+x forward roll,
    +y left pitch, +z up yaw); the gyro z-axis is inverted relative to the
    estimator's +z-up yaw, so it is negated.

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

    data = pd.read_csv(recording_dir / 'matt_data.csv')

    def interp_cols(df, cols):
        t = df['time_s'].to_numpy(dtype=float)
        return np.column_stack([
            np.interp(frame_times, t, df[c].to_numpy(dtype=float)) for c in cols
        ])

    rotation_rate = interp_cols(data, ['gyro_x_dps', 'gyro_y_dps', 'gyro_z_dps'])
    rotation_rate[:, 2] *= -1.0  # gyro z is inverted vs +z-up yaw convention
    translation_rate = interp_cols(
        data, ['vx_meterpersecond', 'vy_meterpersecond', 'vz_meterpersecond'])
    positions = interp_cols(data, ['x_meter', 'y_meter', 'z_meter'])
    return rotation_rate, translation_rate, positions, frame_times


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


def process_rpi_drone_dataset(data_dir, calib_dir, sample_phi, sample_theta,
                              filter_std=2.5):
    """Process the ``rpi_drone`` two-camera dome dataset into sampled clips.

    Each ``recording_*`` subfolder of ``data_dir`` is one clip with ``right.mp4``
    / ``left.mp4`` plus ``matt_data.csv`` / ``stereo_frames.csv``.
    Fly ommatidia are mapped into each video through the per-eye dome
    calibration in ``calib_dir`` (``dome_right.csv`` / ``dome_left.csv``), which
    is rotated into the body frame by :func:`load_dome_calibration`.

    The right eye is sampled at the right-eye columns ``(sample_phi,
    sample_theta)`` and the left eye at their sagittal mirror ``(-sample_phi,
    sample_theta)``.

    Returns:
        List of sampled clip dicts (one per recording).
    """
    import cv2
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

        rotation_rate, translation_rate, positions, _ = load_rpi_ground_truth(rec)

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


def get_drone_stimulus(model_dt, dataset='small_drone', *, data_folder=None,
                       sample_phi=None, sample_theta=None,
                       frame_start=0, frame_end=None):
    """
    Load cached ommatidium-sampled clips for ``dataset`` and temporally
    interpolate them to ``model_dt``.

    Reads the cache ``<data_folder>/stimulus/raw/<dataset>.pt`` produced by
    ``process_raw_data.py`` (the expensive raw-video decode + ommatidium spatial
    sampling is done there, once). The cache holds clips at the native video
    frame rate; this function does the cheap temporal interpolation to
    ``model_dt``. This is the on-the-fly path used by analysis scripts; training
    consumes the fixed datasets baked by ``create_training_data.py``.

    Rates are computed at the native video frame rate and then interpolated
    to model_dt, avoiding step-function artifacts from differentiating
    interpolated positions.

    Args:
        model_dt: model time step for temporal interpolation (seconds).
        dataset: ``'small_drone'`` or ``'rpi_drone'``.
        data_folder: data root; defaults to ``config['data']['data_folder']``.
        sample_phi, sample_theta: (N,) right-eye ommatidium angles (radians).
            Optional; when given they are validated against the columns baked
            into the cache (raises on mismatch).
        frame_start: first model time step to include (after interpolation).
        frame_end: last model time step to include (None = end of clip).

    Returns:
        List of dicts, one per clip, each with keys:
            'right': (time_i, N) array, right-eye stimulus in [0, 1].
            'left': (time_i, N) array, left-eye stimulus in [0, 1].
            'rotation': (time_i, 3) array, rotation position in degrees.
            'translation': (time_i, 3) array, translation position.
            'rotation_rate': (time_i, 3) array, rotation rate in deg/s.
            'translation_rate': (time_i, 3) array, translation rate.
            'video_dt': original video time step (seconds).
            'clip_id': clip identifier.
            'in_fov_right' / 'in_fov_left': (N,) bool column masks.
    """
    if data_folder is None:
        from drosophila_model.config import load_config
        data_folder = load_config()['data']['data_folder']
    data_folder = Path(data_folder)

    if dataset not in ('small_drone', 'rpi_drone'):
        raise ValueError(
            f"Unknown drone dataset '{dataset}'; expected 'small_drone' or 'rpi_drone'.")

    cache_path = data_folder / 'stimulus' / 'raw' / f'{dataset}.pt'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Drone cache not found: {cache_path}. Build it first with:\n"
            f"  python drosophila_model/data_handling/process_raw_data.py {dataset}"
        )
    cached = torch.load(cache_path, weights_only=False)
    clips = cached['clips']

    # The cache bakes in the ommatidium columns used for spatial sampling; if the
    # caller requested specific columns, ensure they match what was baked.
    if sample_phi is not None and sample_theta is not None:
        cached_phi = np.asarray(cached['sample_phi'])
        cached_theta = np.asarray(cached['sample_theta'])
        if (cached_phi.shape != np.asarray(sample_phi).shape
                or not np.allclose(cached_phi, sample_phi)
                or not np.allclose(cached_theta, sample_theta)):
            raise ValueError(
                f"Requested ommatidium columns do not match those baked into "
                f"{cache_path}. Re-run:\n"
                f"  python drosophila_model/data_handling/process_raw_data.py {dataset}"
            )

    results = []

    for clip in clips:
        video_dt = clip['video_dt']
        T = clip['right'].shape[0]

        sampled_right = clip['right'].numpy()
        sampled_left = clip['left'].numpy()

        sampled_right = interpolate_to_model_dt(sampled_right[None], video_dt, model_dt)[0]
        sampled_left = interpolate_to_model_dt(sampled_left[None], video_dt, model_dt)[0]

        # Stored rotation/translation are already rates from extract_odometry
        rot_rate_native = clip['rotation'].numpy()
        trans_rate_native = clip['translation'].numpy()

        # Integrate rates to get cumulative positions
        from drosophila_model.model_definition.ego_motion import integrate_angular_velocity, integrate_translational_velocity
        omega_t = torch.tensor(rot_rate_native, dtype=torch.float64)
        rotation = integrate_angular_velocity(omega_t, video_dt).numpy()
        translation = integrate_translational_velocity(
            torch.tensor(trans_rate_native, dtype=torch.float64),
            omega_t, video_dt,
        ).numpy()

        # Interpolate positions and rates to model_dt.
        video_times = np.arange(T) * video_dt
        num_model_steps = int(round((T - 1) * video_dt / model_dt)) + 1
        model_times = np.linspace(video_times[0], video_times[-1], num_model_steps)

        rotation_interp = np.column_stack([
            np.interp(model_times, video_times, rotation[:, i]) for i in range(3)
        ])
        translation_interp = np.column_stack([
            np.interp(model_times, video_times, translation[:, i]) for i in range(3)
        ])
        rot_rate_interp = np.column_stack([
            np.interp(model_times, video_times, rot_rate_native[:, i]) for i in range(3)
        ])
        trans_rate_interp = np.column_stack([
            np.interp(model_times, video_times, trans_rate_native[:, i]) for i in range(3)
        ])

        if 'in_fov_right' not in clip:
            raise KeyError(
                f"Clip {clip['clip_id']} missing 'in_fov_right'/'in_fov_left'."
            )
        result = {
            'right': sampled_right[frame_start:frame_end],
            'left': sampled_left[frame_start:frame_end],
            'rotation': rotation_interp[frame_start:frame_end],
            'translation': translation_interp[frame_start:frame_end],
            'rotation_rate': rot_rate_interp[frame_start:frame_end],
            'translation_rate': trans_rate_interp[frame_start:frame_end],
            'video_dt': video_dt,
            'clip_id': clip['clip_id'],
            'in_fov_right': clip['in_fov_right'],
            'in_fov_left': clip['in_fov_left'],
        }
        results.append(result)

    return results


def drone_clips_to_two_eye(drone_clips, stim_time, model_dt, device, dtype):
    """Segment preprocessed drone clips into a two-eye stimulus tensor.

    Shared by every two-eye drone loader (``small_drone``, ``rpi_drone``):
    each clip is chopped into non-overlapping
    ``stim_time`` segments, right/left eyes concatenated along the column axis,
    and the ground-truth rotation/translation rates carried alongside.

    Returns ``(stimulus, optic_flow, observer_rotation, observer_translation,
    train_mask)``; optic_flow is all-NaN (real flow is recovered downstream).
    """
    in_fov = torch.cat([
        drone_clips[0]['in_fov_right'],
        drone_clips[0]['in_fov_left'],
    ], dim=0)  # (2*N,) True = in field of view

    seg_len = int(round(stim_time / model_dt))
    max_clip_len = max(clip['right'].shape[0] for clip in drone_clips)
    if seg_len > max_clip_len:
        warnings.warn(
            f"stim_time={stim_time}s ({seg_len} steps) exceeds longest "
            f"drone clip ({max_clip_len} steps = {max_clip_len * model_dt:.1f}s). "
            f"Capping to {max_clip_len * model_dt:.1f}s."
        )
        seg_len = max_clip_len

    seg_rights, seg_lefts, seg_rots, seg_trans = [], [], [], []
    n_segments = 0
    for clip in drone_clips:
        T_clip = clip['right'].shape[0]
        num_segments = T_clip // seg_len
        if num_segments == 0:
            continue
        usable = num_segments * seg_len
        n_segments += num_segments
        seg_rights.append(clip['right'][:usable].reshape(num_segments, seg_len, -1))
        seg_lefts.append(clip['left'][:usable].reshape(num_segments, seg_len, -1))
        seg_rots.append(clip['rotation_rate'][:usable].reshape(num_segments, seg_len, 3))
        seg_trans.append(clip['translation_rate'][:usable].reshape(num_segments, seg_len, 3))

    stim_right = np.concatenate(seg_rights, axis=0)
    stim_left = np.concatenate(seg_lefts, axis=0)
    true_rotation = np.concatenate(seg_rots, axis=0)
    true_translation = np.concatenate(seg_trans, axis=0)

    stimulus = torch.tensor(
        np.concatenate([stim_right, stim_left], axis=-1), device=device, dtype=dtype,
    )
    optic_flow = torch.full((*stimulus.shape, 2), float('nan'), device=device, dtype=dtype)
    observer_rotation = torch.tensor(true_rotation, device=device, dtype=dtype)
    observer_translation = torch.tensor(true_translation, device=device, dtype=dtype)
    train_mask = in_fov.unsqueeze(0).unsqueeze(0).expand(
        n_segments, seg_len, -1).to(device=device)

    return stimulus, optic_flow, observer_rotation, observer_translation, train_mask
