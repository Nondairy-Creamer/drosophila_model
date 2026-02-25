"""Process copter stereo video data into preprocessed tensors.

Loads each pair of left/right stereo videos and corresponding pose log CSV,
converts frames to grayscale float [0, 1], applies coordinate transformation,
trims the initial zero-pose period, and saves all clips as a single .pt file
containing a list of dicts.

Usage:
    python process_copter.py
    python process_copter.py --data-dir ../connectome_data/copter_videos --output-dir ../connectome_data/copter_processed
"""

import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path
import argparse


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
    """Process a single stereo clip: load videos + pose, trim zero-pose prefix.

    Returns a dict with tensors or None if the clip has no valid pose data.
    """
    data_dir = Path(data_dir)

    left = load_video_grayscale(data_dir / f'left_{clip_id}.mp4')
    right = load_video_grayscale(data_dir / f'right_{clip_id}.mp4')

    pose = pd.read_csv(data_dir / f'apriltag_pose_log_{clip_id}.csv')
    video_dt = float(pose['t_video_s'].iloc[1] - pose['t_video_s'].iloc[0])

    # CSV axes: +x forward, +y up, +z right
    # Model axes: +x forward, +y left, +z up
    rotation = pose[['roll_used_deg', 'yaw_used_deg', 'pitch_used_deg']].to_numpy(dtype=np.float32)
    rotation[:, 1] *= -1

    translation = pose[['x_used', 'z_used', 'y_used']].to_numpy(dtype=np.float32)
    translation[:, 1] *= -1

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
    remaining = min_len - start
    print(f'  Clip {clip_id}: trimmed {start} zero-pose frames, {remaining} frames remaining '
          f'({remaining * video_dt:.1f}s)')

    return {
        'left': torch.from_numpy(left[start:]),
        'right': torch.from_numpy(right[start:]),
        'rotation': torch.from_numpy(rotation[start:]),
        'translation': torch.from_numpy(translation[start:]),
        'video_dt': video_dt,
        'clip_id': clip_id,
    }


def main():
    parser = argparse.ArgumentParser(description='Process copter stereo videos into tensors')
    parser.add_argument('--data-dir', type=str, default='../connectome_data/copter_videos',
                        help='Directory containing left_N.mp4, right_N.mp4, apriltag_pose_log_N.csv')
    parser.add_argument('--output-dir', type=str, default='../connectome_data',
                        help='Output directory for processed copter_data.pt')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_ids = sorted(
        int(p.stem.split('_')[-1])
        for p in data_dir.glob('apriltag_pose_log_*.csv')
    )
    print(f'Found {len(clip_ids)} clips: {clip_ids}')

    clips = []
    for clip_id in clip_ids:
        print(f'Processing clip {clip_id}...')
        clip = process_clip(clip_id, data_dir)
        if clip is None:
            continue
        clips.append(clip)

    out_path = output_dir / 'copter_data.pt'
    torch.save(clips, out_path)
    size_mb = out_path.stat().st_size / 1e6

    print(f'\n{"="*60}')
    print(f'Saved {len(clips)} clips to {out_path} ({size_mb:.0f} MB)')
    print(f'{"Clip":>6} {"Frames":>8} {"Resolution":>12} {"Duration":>10}')
    for clip in clips:
        T, H, W = clip['left'].shape
        dt = clip['video_dt']
        print(f'{clip["clip_id"]:>6} {T:>8} {H}x{W:>12} {T*dt:>9.1f}s')


if __name__ == '__main__':
    main()
