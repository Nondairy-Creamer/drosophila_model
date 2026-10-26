"""Batch-process all drone clips through AprilTag pose estimation.

Discovers left_N.mp4 / right_N.mp4 pairs in the data directory, estimates
the drone body pose trajectory for each clip using dual-camera AprilTag
observations, and saves:
    - rotation location (roll, pitch, yaw in degrees) and rate (deg/s)
    - translation location (x, y, z in metres) and rate (m/s)

Fixes one tag as the world origin and jointly estimates the other tag's
pose along with the drone trajectory (:func:`estimate_pose.run_pipeline_auto`).
Only requires the tag side length (default 16 cm). Uses the equirectangular
small_drone rig by default, or a pinhole rig from ``--calibration``.

Usage:
    python -m drosophila_model.data_handling.drone_pose
    python -m drosophila_model.data_handling.drone_pose --clips 32
    python -m drosophila_model.data_handling.drone_pose --data-dir /path/to/videos --tag-size 0.16
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import median_filter
from pathlib import Path
import argparse

from drosophila_model.config import get_data_folder, load_config

from drosophila_model.data_handling.estimate_pose import (
    DEFAULT_SMOOTHNESS, DEFAULT_TAG_FAMILY, DEFAULT_TAG_SIZE,
    DualCameraRig, EquirectRig, run_pipeline_auto,
)
from drosophila_model.data_handling.drone_utilities import get_image_size


# ── Clip discovery ───────────────────────────────────────────────────────

def find_clip_ids(data_dir: Path) -> list[int]:
    """Find clip IDs that have both left_N.mp4 and right_N.mp4."""
    left_ids = set()
    right_ids = set()
    for p in data_dir.glob('left_*.mp4'):
        try:
            left_ids.add(int(p.stem.split('_')[1]))
        except (ValueError, IndexError):
            continue
    for p in data_dir.glob('right_*.mp4'):
        try:
            right_ids.add(int(p.stem.split('_')[1]))
        except (ValueError, IndexError):
            continue
    both = sorted(left_ids & right_ids)
    if left_ids - right_ids:
        print(f'  Warning: left-only clips (no right): {sorted(left_ids - right_ids)}')
    if right_ids - left_ids:
        print(f'  Warning: right-only clips (no left): {sorted(right_ids - left_ids)}')
    return both


# ── Pose → locations & rates ─────────────────────────────────────────────

def compute_angular_velocity(rvecs: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity from absolute rotation vectors.

    Computes omega via relative rotations between consecutive frames
    (avoids gimbal lock issues with Euler-angle differentiation).

    Args:
        rvecs: (T, 3) rotation vectors (axis-angle, radians)
        dt: time step in seconds

    Returns:
        (T, 3) angular velocity [roll, pitch, yaw] in rad/s (body frame)
    """
    rotations = Rotation.from_rotvec(rvecs)
    n = len(rvecs)
    if n < 2:
        return np.zeros_like(rvecs)

    dR = rotations[:-1].inv() * rotations[1:]
    omega_fwd = dR.as_rotvec() / dt

    omega = np.zeros((n, 3))
    omega[0] = omega_fwd[0]
    omega[-1] = omega_fwd[-1]
    if n > 2:
        omega[1:-1] = 0.5 * (omega_fwd[:-1] + omega_fwd[1:])

    return omega


def compute_all(
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    dt: float,
    world_to_m: float = 1.0,
    median_kernel: int = 5,
) -> dict:
    """Compute locations and rates from raw pose estimates.

    Raw rvecs/tvecs are in the world frame (anchor tag's local frame).
    Velocities are differentiated and expressed in the instantaneous body
    frame, then median-filtered when *median_kernel* > 0.  Positions and
    rotations are derived directly from the raw poses (use
    ``drosophila_model.model_definition.ego_motion.integrate_angular_velocity`` /
    ``integrate_translational_velocity`` to reintegrate from the filtered
    velocities when needed).

    Args:
        rvecs: (T, 3) rotation vectors (radians), world frame
        tvecs: (T, 3) positions in world units, world frame
        dt: time step in seconds
        world_to_m: scale factor from world (tag-size) units to metres
        median_kernel: odd integer for median filter width (0 to disable)

    Returns:
        dict with rotation_deg, rotation_deg_s, position_m, velocity_m_s
    """
    R_0 = Rotation.from_rotvec(rvecs[0])
    t_0 = tvecs[0]

    all_rots = Rotation.from_rotvec(rvecs)
    delta_rots = R_0.inv() * all_rots
    # Extrinsic x-y-z Euler angles (= intrinsic Z-Y-X yaw-pitch-roll),
    # returned as [roll, pitch, yaw] in degrees.
    rotation_deg = delta_rots.as_euler('xyz', degrees=True)

    omega_rad = compute_angular_velocity(rvecs, dt)
    rotation_deg_s = omega_rad * (180.0 / np.pi)

    R_0_mat = R_0.as_matrix()
    delta_t = (tvecs - t_0) * world_to_m
    position_m = (R_0_mat.T @ delta_t.T).T

    vel_world = np.gradient(tvecs, dt, axis=0) * world_to_m
    velocity_m_s = all_rots.inv().apply(vel_world)

    if median_kernel > 0:
        rotation_deg_s = median_filter(rotation_deg_s, size=(median_kernel, 1))
        velocity_m_s = median_filter(velocity_m_s, size=(median_kernel, 1))

    return {
        'rotation_deg': rotation_deg,
        'rotation_deg_s': rotation_deg_s,
        'position_m': position_m,
        'velocity_m_s': velocity_m_s,
    }


# ── Per-clip processing ──────────────────────────────────────────────────

def process_clip(
    clip_id: int,
    data_dir: Path,
    rig: EquirectRig | DualCameraRig,
    tag_size: float,
    tag_family: str,
    smoothness: float,
    world_to_m: float,
    median_kernel: int = 5,
) -> dict | None:
    """Run pose estimation on one clip and compute locations + rates.

    Uses joint tag + trajectory estimation (``run_pipeline_auto``).

    Returns:
        dict with rotation/translation locations and rates, raw poses, etc.
        or None on failure.
    """
    left_path = data_dir / f'left_{clip_id}.mp4'
    right_path = data_dir / f'right_{clip_id}.mp4'

    print(f'\n{"="*60}')
    print(f'Clip {clip_id}')
    print(f'{"="*60}')

    try:
        trajectory = run_pipeline_auto(
            left_video_path=str(left_path),
            right_video_path=str(right_path),
            rig=rig,
            tag_size=tag_size,
            tag_family=tag_family,
            smoothness_weight=smoothness,
        )
    except Exception as e:
        print(f'  FAILED: {e}')
        return None

    fps = trajectory['fps']
    dt = 1.0 / fps

    derived = compute_all(
        trajectory['rvecs'], trajectory['tvecs'], dt,
        world_to_m=world_to_m, median_kernel=median_kernel)

    return {
        'clip_id': clip_id,
        'rotation_deg': derived['rotation_deg'],
        'rotation_deg_s': derived['rotation_deg_s'],
        'position_m': derived['position_m'],
        'velocity_m_s': derived['velocity_m_s'],
        'rvecs': trajectory['rvecs'],
        'tvecs': trajectory['tvecs'],
        'fps': fps,
        'n_frames': trajectory['n_frames'],
        'mean_reprojection_px': trajectory['mean_reprojection_px'],
    }


# ── I/O helpers ──────────────────────────────────────────────────────────

def save_clip_result(result: dict, output_dir: Path):
    """Save one clip's pose data to pose_{clip_id}.npz."""
    clip_id = result['clip_id']
    path = output_dir / f'pose_{clip_id}.npz'
    np.savez(
        path,
        rotation_deg=result['rotation_deg'],
        rotation_deg_s=result['rotation_deg_s'],
        position_m=result['position_m'],
        velocity_m_s=result['velocity_m_s'],
        rvecs=result['rvecs'],
        tvecs=result['tvecs'],
        fps=result['fps'],
    )
    print(f'  Saved -> {path}')


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Estimate drone pose for all clips in a directory')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory with left_N.mp4 / right_N.mp4 pairs '
                             '(default: <data_folder>/stimulus/raw/drone_videos)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for pose .npz files '
                             '(default: <data_folder>/stimulus/raw/drone_poses)')
    parser.add_argument('--tag-size', type=float, default=DEFAULT_TAG_SIZE,
                        help=f'AprilTag side length in metres (default: {DEFAULT_TAG_SIZE})')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Pinhole camera rig calibration JSON (see '
                             'estimate_pose.DualCameraRig.load). If omitted, '
                             'uses the equirectangular small_drone rig.')
    parser.add_argument('--tag-family', default=DEFAULT_TAG_FAMILY,
                        help=f'AprilTag family (default: {DEFAULT_TAG_FAMILY})')
    parser.add_argument('--smoothness', type=float, default=DEFAULT_SMOOTHNESS,
                        help=f'Temporal smoothness weight (default: {DEFAULT_SMOOTHNESS})')
    parser.add_argument('--world-to-m', type=float, default=1.0,
                        help='Scale from --tag-size units to metres '
                             '(default: 1.0, i.e. --tag-size is in metres)')
    parser.add_argument('--median-kernel', type=int, default=5,
                        help='Median filter width for velocity smoothing '
                             '(odd int, 0 to disable; default: 5)')
    parser.add_argument('--clips', type=int, nargs='+', default=None,
                        help='Process only these clip IDs (default: all)')
    args = parser.parse_args()

    # The data folder is only needed for the default input/output locations.
    raw_dir = (None if args.data_dir and args.output_dir
               else get_data_folder(load_config()) / 'stimulus' / 'raw')
    data_dir = Path(args.data_dir) if args.data_dir else raw_dir / 'drone_videos'
    output_dir = Path(args.output_dir) if args.output_dir else raw_dir / 'drone_poses'

    # Discover clips
    all_ids = find_clip_ids(data_dir)
    if not all_ids:
        print(f'No left_N.mp4 / right_N.mp4 pairs found in {data_dir}')
        return
    if args.clips:
        missing = sorted(set(args.clips) - set(all_ids))
        if missing:
            print(f'  Warning: requested clips not found: {missing}')
        clip_ids = [c for c in args.clips if c in all_ids]
    else:
        clip_ids = all_ids
    if not clip_ids:
        print(f'None of the requested clips {args.clips} exist in {data_dir}')
        return
    print(f'Found {len(all_ids)} clip pairs, processing {len(clip_ids)}: {clip_ids}')
    print(f'Tag size={args.tag_size:.3f} m, joint tag + trajectory optimisation')

    # Camera rig
    if args.calibration:
        rig = DualCameraRig.load(args.calibration)
        print(f'Loaded pinhole calibration from {args.calibration}')
    else:
        image_size = get_image_size(data_dir, clip_ids[0])
        rig = EquirectRig.create_default(image_size)
        print(f'Using equirectangular rig '
              f'(image {image_size[1]}x{image_size[0]})')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process clips
    results = []
    for clip_id in clip_ids:
        result = process_clip(
            clip_id, data_dir, rig,
            tag_size=args.tag_size,
            tag_family=args.tag_family,
            smoothness=args.smoothness,
            world_to_m=args.world_to_m,
            median_kernel=args.median_kernel,
        )
        if result is None:
            continue

        save_clip_result(result, output_dir)
        results.append(result)

    # Summary
    if results:
        print(f'\n{"="*60}')
        print(f'Processed {len(results)}/{len(clip_ids)} clips')
        print(f'{"Clip":>6} {"Frames":>8} {"Duration":>10} '
              f'{"Reproj (px)":>12} {"w RMS (deg/s)":>12} {"v RMS (m/s)":>12}')
        for r in results:
            dur = r['n_frames'] / r['fps']
            omega_rms = np.sqrt((r['rotation_deg_s'] ** 2).mean())
            v_rms = np.sqrt((r['velocity_m_s'] ** 2).mean())
            print(f'{r["clip_id"]:>6} {r["n_frames"]:>8} {dur:>9.1f}s '
                  f'{r["mean_reprojection_px"]:>12.2f} {omega_rms:>12.1f} {v_rms:>12.2f}')
        print(f'Output directory: {output_dir}')


if __name__ == '__main__':
    main()
