"""Fix old-format apriltag pose CSVs to match model coordinate conventions.

Applies the coordinate flips from process_drone.py directly to the CSV:
  - Negate left-eye yaw values (L_yaw_*_deg)
  - Swap column names: z <-> y (position), yaw <-> pitch (rotation)

Usage:
    python fix_old_format.py path/to/apriltag_pose_log_*_both.csv
    python fix_old_format.py ../connectome_data/drone_videos/apriltag_pose_log_*_both.csv
"""

import argparse
import pandas as pd
from pathlib import Path


NEGATE_COLUMNS = [
    'L_yaw_meas_deg', 'L_yaw_used_deg',
    'L_z_meas', 'L_z_used',
]

RENAME_MAP = {}
for side in ('L', 'R'):
    for suffix in ('meas', 'used'):
        RENAME_MAP[f'{side}_y_{suffix}'] = f'{side}_z_{suffix}'
        RENAME_MAP[f'{side}_z_{suffix}'] = f'{side}_y_{suffix}'
        RENAME_MAP[f'{side}_yaw_{suffix}_deg'] = f'{side}_pitch_{suffix}_deg'
        RENAME_MAP[f'{side}_pitch_{suffix}_deg'] = f'{side}_yaw_{suffix}_deg'


def fix_csv(path):
    df = pd.read_csv(path)

    for col in NEGATE_COLUMNS:
        if col not in df.columns:
            print(f'  WARNING: column {col!r} not found in {path.name}, skipping')
            continue
        df[col] = -df[col]

    new_columns = [RENAME_MAP.get(c, c) for c in df.columns]
    df.columns = new_columns

    df.to_csv(path, index=False)
    print(f'  Fixed {path.name} ({len(df)} rows)')


VALID_CLIP_IDS = {'32', '33'}


def main():
    parser = argparse.ArgumentParser(description='Fix old-format apriltag pose CSVs')
    parser.add_argument('files', nargs='+', type=Path, help='CSV files to fix in-place')
    args = parser.parse_args()

    for path in args.files:
        if not path.exists():
            print(f'  Skipping {path}: not found')
            continue
        clip_id = path.stem.split('_')[-2]
        if clip_id not in VALID_CLIP_IDS:
            print(f'  Skipping {path.name}: clip {clip_id} not in {VALID_CLIP_IDS}')
            continue
        fix_csv(path)


if __name__ == '__main__':
    main()
