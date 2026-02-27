"""Process copter stereo video data into preprocessed tensors.

Loads each pair of left/right stereo videos and corresponding pose log CSV,
converts frames to grayscale, applies coordinate transformation, trims
leading and trailing zero-pose periods, samples at ommatidium locations using column_info,
and saves all clips as a single .pt file containing a list of dicts.

Usage:
    python process_copter.py
    python process_copter.py --data-dir ../connectome_data/copter_videos --output-dir ../connectome_data
"""

import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path
import argparse
from scipy.ndimage import gaussian_filter1d

from stimulus_utilities import build_ommatidium_sampling_matrix

SMOOTH_TIME_S = 0.2


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


def process_clip(clip_id, data_dir):
    """Process a single stereo clip: load videos + pose, trim zero-pose prefix and suffix.

    Returns a dict with tensors or None if the clip has no valid pose data.
    """
    data_dir = Path(data_dir)

    left = load_video_grayscale(data_dir / f'left_{clip_id}.mp4')
    right = load_video_grayscale(data_dir / f'right_{clip_id}.mp4')

    pose = pd.read_csv(data_dir / f'apriltag_pose_log_{clip_id}.csv')
    video_dt = float(pose['t_video_s'].iloc[1] - pose['t_video_s'].iloc[0])

    # CSV axes: +x forward, +y up, +z right (left-handed)
    # Model axes: +x forward, +y left, +z up (right-handed)
    # Mapping: model_x = csv_x, model_y = csv_z, model_z = csv_y
    # CSV pitch_used_deg → roll (about x/fwd), CSV yaw_used_deg → pitch (about y/left), CSV roll_used_deg → yaw (about z/up)
    rotation = pose[['pitch_used_deg', 'yaw_used_deg', 'roll_used_deg']].to_numpy(dtype=np.float32)
    rotation[:, 0] *= -1
    rotation[:, 2] *= -1

    translation = pose[['x_used', 'z_used', 'y_used']].to_numpy(dtype=np.float32)

    # Align to shortest (video and CSV may differ by a frame)
    min_len = min(len(left), len(right), len(pose))
    left = left[:min_len]
    right = right[:min_len]
    rotation = rotation[:min_len]
    translation = translation[:min_len]

    # Find first frame with non-zero pose
    pose_mag = np.abs(rotation).sum(axis=1) + np.abs(translation).sum(axis=1)
    nonzero = np.where(pose_mag > 0)[0]

    if len(nonzero) == 0:
        print(f'  Clip {clip_id}: all-zero pose, skipping')
        return None

    start = nonzero[0]
    end = nonzero[-1] + 1
    trimmed_start = start
    trimmed_end = min_len - end
    remaining = end - start
    print(f'  Clip {clip_id}: trimmed {trimmed_start} leading + {trimmed_end} trailing zero-pose frames, '
          f'{remaining} frames remaining ({remaining * video_dt:.1f}s)')

    left = left[start:end]
    right = right[start:end]
    rotation = rotation[start:end]
    translation = translation[start:end]

    smooth_sigma = SMOOTH_TIME_S / video_dt
    rotation = gaussian_filter1d(rotation, sigma=smooth_sigma, axis=0).astype(np.float32)
    translation = gaussian_filter1d(translation, sigma=smooth_sigma, axis=0).astype(np.float32)

    return {
        'left': torch.from_numpy(left),
        'right': torch.from_numpy(right),
        'rotation': torch.from_numpy(rotation),
        'translation': torch.from_numpy(translation),
        'video_dt': video_dt,
        'clip_id': clip_id,
    }


def sample_clip(clip, W_right, W_left, in_fov_right, in_fov_left):
    """Spatially sample a raw clip at ommatidium locations.

    Returns a new dict with 'right' and 'left' as (T, N) float32 arrays
    in [0, 1], plus rotation/translation/video_dt/clip_id unchanged.
    """
    T = clip['right'].shape[0]

    right_flat = clip['right'].numpy().reshape(T, -1).T.astype(np.float32)
    sampled_right = np.asarray(W_right.dot(right_flat)).T / 255.0

    left_flat = clip['left'].numpy().reshape(T, -1).T.astype(np.float32)
    sampled_left = np.asarray(W_left.dot(left_flat)).T / 255.0

    sampled_right[:, ~in_fov_right] = 0.5
    sampled_left[:, ~in_fov_left] = 0.5

    return {
        'right': torch.from_numpy(sampled_right),
        'left': torch.from_numpy(sampled_left),
        'rotation': clip['rotation'],
        'translation': clip['translation'],
        'video_dt': clip['video_dt'],
        'clip_id': clip['clip_id'],
        'in_fov_right': torch.from_numpy(in_fov_right.copy()),
        'in_fov_left': torch.from_numpy(in_fov_left.copy()),
    }


def main():
    parser = argparse.ArgumentParser(description='Process copter stereo videos into tensors')
    parser.add_argument('--data-dir', type=str, default='../connectome_data/copter_videos',
                        help='Directory containing left_N.mp4, right_N.mp4, apriltag_pose_log_N.csv')
    parser.add_argument('--output-dir', type=str, default='../connectome_data',
                        help='Output directory for processed copter_data.pt')
    parser.add_argument('--column-info', type=str, default='../connectome_data/optic_lobes_column_info.parquet',
                        help='Path to column_info parquet with phi/theta')
    parser.add_argument('--filter-std', type=float, default=2.5,
                        help='Gaussian filter std in degrees for ommatidium sampling')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Camera field-of-view limits (degrees)
    phi_right_deg = (-30, -150)
    phi_left_deg = (150, 30)
    theta_deg = (45, 135)

    # Load ommatidium positions
    column_info_df = pd.read_parquet(args.column_info)
    sample_phi = column_info_df['phi'].to_numpy()
    sample_theta = column_info_df['theta'].to_numpy()
    print(f'Loaded {len(sample_phi)} ommatidium positions from {args.column_info}')

    clip_ids = sorted(
        int(p.stem.split('_')[-1])
        for p in data_dir.glob('apriltag_pose_log_*.csv')
    )
    print(f'Found {len(clip_ids)} clips: {clip_ids}')

    # Load all raw clips first so we can build sampling matrices from image size
    raw_clips = []
    for clip_id in clip_ids:
        print(f'Loading clip {clip_id}...')
        clip = process_clip(clip_id, data_dir)
        if clip is not None:
            raw_clips.append(clip)

    if not raw_clips:
        print('No valid clips found.')
        return

    # Build sampling matrices (all clips share the same resolution)
    _, image_height, image_width = raw_clips[0]['right'].shape
    phi_right_lim = np.array(phi_right_deg) * np.pi / 180
    phi_left_lim = np.array(phi_left_deg) * np.pi / 180
    theta_lim = np.array(theta_deg) * np.pi / 180

    print(f'Building sampling matrices for {image_height}x{image_width} frames...')
    W_right = build_ommatidium_sampling_matrix(
        (image_height, image_width), phi_right_lim, theta_lim,
        sample_phi, sample_theta, filter_std=args.filter_std,
    )
    W_left = build_ommatidium_sampling_matrix(
        (image_height, image_width), phi_left_lim, theta_lim,
        -sample_phi, sample_theta, filter_std=args.filter_std,
    )
    in_fov_right = np.asarray(W_right.sum(axis=1)).ravel() > 0
    in_fov_left = np.asarray(W_left.sum(axis=1)).ravel() > 0

    # Sample each clip
    clips = []
    for raw_clip in raw_clips:
        print(f'Sampling clip {raw_clip["clip_id"]}...')
        clips.append(sample_clip(raw_clip, W_right, W_left, in_fov_right, in_fov_left))

    out_path = output_dir / 'copter_data.pt'
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
    main()
