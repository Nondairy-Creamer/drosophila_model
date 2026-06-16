"""Overlay detected AprilTag locations on the rpi_drone camera videos.

Runs the dome-calibrated tag extractor (:mod:`data_handling.extract_tags`),
then renders a movie of the raw left/right camera frames with each detection
drawn on top: the tag outline, a dot at its centre, and a label with the tag
id, the spherical-PnP range, and the body-frame bearing (phi, theta).

Because the dome lens is treated as a spherical observer, the range and
bearing come from the angular calibration + spherical PnP, not a pinhole model.

Usage:
    python analysis/movie_scripts/plot_tag_location.py
    python analysis/movie_scripts/plot_tag_location.py \\
        --recording recording_20260611_234117 --max-frames 400
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from drosophila_model.config import load_config
from drosophila_model.data_handling.extract_tags import (
    DEFAULT_TAG_SIZE,
    compute_board_trajectory,
    extract_tags_from_recording,
)
from drosophila_model.analysis.movie_scripts.movie_utilities import write_movie

# World/body frame (coordinate_conventions.md): +x fwd, +y left, +z up;
# rotations roll(x) / pitch(y) / yaw(z), CCW positive.
_ROT_LABELS = ['roll', 'pitch', 'yaw']
_TRANS_LABELS = ['x', 'y', 'z']
_AXIS_COLORS = ['#e41a1c', '#377eb8', '#4daf4a']


def drone_egomotion_from_board(traj):
    """Drone egomotion from the board pose trajectory, in the body convention.

    The board is static, so its solved pose ``T_body_board(t)`` encodes the
    drone's motion. We express that motion in the drone's own body frame at the
    first tracked frame (which follows coordinate_conventions.md: +x forward,
    +y left, +z up). Concretely, with ``T_body0_body(t) = T_body0_board @
    T_body_board(t)^-1``, the translation/rotation are read off relative to the
    start. This makes the axes physically meaningful to the drone — e.g.
    approaching the left-mounted tag shows up as +y (left), not in z.

    Returns:
        (euler_deg, pos_rel_m): (N, 3) arrays of [roll, pitch, yaw] degrees and
        [x, y, z] metres, both relative to the first tracked frame.
    """
    R = Rotation.from_rotvec(traj['rvec'])   # R_body_board(t)
    t = traj['tvec']                         # (N, 3)
    R0, t0 = R[0], t[0]

    # Drone centre in the board frame, then mapped into the start body frame.
    center_board = -R.inv().apply(t)         # = -R(t)^T t(t)
    pos_rel = R0.apply(center_board) + t0    # body0 frame; zero at t=0

    # Drone orientation relative to start: R_body0_body(t) = R0 @ R(t)^T.
    R_rel = R0 * R.inv()
    euler_deg = R_rel.as_euler('xyz', degrees=True)
    return euler_deg, pos_rel

# Stable per-tag colours (cycled if more tag ids appear).
_TAG_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
]


def _tag_color(tag_id):
    return _TAG_PALETTE[tag_id % len(_TAG_PALETTE)]


class FrameGrabber:
    """Sequential (monotonic) random-ish frame reader for a video."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f'Cannot open video: {path}')
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._pos = -1

    def read_at(self, idx):
        """Grayscale frame at *idx* (idx must be non-decreasing across calls)."""
        while self._pos < idx:
            if not self.cap.grab():
                return None
            self._pos += 1
        ok, frame = self.cap.retrieve()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def release(self):
        self.cap.release()


def _margin_lim(data, frac=0.1, min_margin=1e-3):
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    margin = max(frac * (hi - lo), min_margin)
    return lo - margin, hi + margin


def _setup_egomotion_axes(ax_rot, ax_trans, traj, trail_seconds):
    """Set up scrolling egomotion panels; return a state dict for updates.

    Lines start empty and are filled in over a trailing time window as the
    movie progresses (mirroring visualize_drone_data.py), with fixed y-limits
    from the full trajectory so the scale stays stable.
    """
    euler_deg, pos_rel = drone_egomotion_from_board(traj)
    times = traj['time_s'].astype(float)
    if not np.isfinite(times).all():
        times = traj['frame_idx'].astype(float)

    rot_lines, trans_lines = [], []
    for lab, col in zip(_ROT_LABELS, _AXIS_COLORS):
        ln, = ax_rot.plot([], [], color=col, label=lab, linewidth=1.3)
        rot_lines.append(ln)
    for lab, col in zip(_TRANS_LABELS, _AXIS_COLORS):
        ln, = ax_trans.plot([], [], color=col, label=lab, linewidth=1.3)
        trans_lines.append(ln)

    ax_rot.set_ylim(*_margin_lim(euler_deg, min_margin=1.0))
    ax_rot.set_ylabel('rotation (deg)')
    ax_rot.set_xlabel('time (s)')
    ax_rot.set_title('Drone rotation (world frame, rel. to start)')
    ax_rot.legend(loc='upper left', fontsize=8, ncol=3)
    ax_rot.grid(True, alpha=0.3)

    ax_trans.set_ylim(*_margin_lim(pos_rel, min_margin=0.05))
    ax_trans.set_ylabel('translation (m)')
    ax_trans.set_xlabel('time (s)')
    ax_trans.set_title('Drone translation (world frame, rel. to start)')
    ax_trans.legend(loc='upper left', fontsize=8, ncol=3)
    ax_trans.grid(True, alpha=0.3)

    return {
        'times': times, 'euler': euler_deg, 'pos': pos_rel,
        'frames': traj['frame_idx'], 'trail': trail_seconds,
        'rot_lines': rot_lines, 'trans_lines': trans_lines,
        'ax_rot': ax_rot, 'ax_trans': ax_trans,
    }


def _update_egomotion(state, idx):
    """Advance the scrolling egomotion window to video frame *idx*."""
    frames, times = state['frames'], state['times']
    pos = int(np.searchsorted(frames, idx, side='right'))
    if pos == 0:
        for ln in state['rot_lines'] + state['trans_lines']:
            ln.set_data([], [])
        return
    t_now = times[pos - 1]
    t0 = t_now - state['trail']
    start = int(np.searchsorted(times, t0, side='left'))
    tw = times[start:pos]
    for i, ln in enumerate(state['rot_lines']):
        ln.set_data(tw, state['euler'][start:pos, i])
    for i, ln in enumerate(state['trans_lines']):
        ln.set_data(tw, state['pos'][start:pos, i])
    state['ax_rot'].set_xlim(t0, t_now)
    state['ax_trans'].set_xlim(t0, t_now)


def make_tag_location_movie(recording_dir, records, output_path,
                            cameras=('right', 'left'), max_frames=None,
                            step=1, fps=None, traj=None, trail_seconds=6.0):
    """Render the per-camera video with AprilTag detections overlaid.

    If ``traj`` (a board trajectory from ``compute_board_trajectory``) is
    given, two panels are added showing the drone's estimated rotation and
    translation (world frame, per coordinate_conventions.md), drawn as a
    scrolling trailing window (``trail_seconds`` wide) that progresses with
    the movie, like visualize_drone_data.py.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    recording_dir = Path(recording_dir)

    by_cam_frame = defaultdict(list)
    for r in records:
        by_cam_frame[(r['camera'], r['frame_idx'])].append(r)

    # Display order: left camera on the left, right camera on the right.
    display_order = ['left', 'right']
    ordered = [c for c in display_order if c in cameras]
    ordered += [c for c in cameras if c not in display_order]

    grabbers, cam_names = [], []
    for cam in ordered:
        video_path = recording_dir / f'{cam}.mp4'
        if not video_path.exists():
            print(f'  [{cam}] no video at {video_path}, skipping')
            continue
        grabbers.append(FrameGrabber(video_path))
        cam_names.append(cam)
    if not grabbers:
        raise FileNotFoundError(f'No camera videos found in {recording_dir}')

    n_total = min(g.n_frames for g in grabbers)
    frame_indices = list(range(0, n_total, step))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if fps is None:
        native = grabbers[0].fps
        fps = (native / step) if native and native > 0 else 20.0

    first = [g.read_at(frame_indices[0]) for g in grabbers]
    h, w = first[0].shape

    show_ego = traj is not None
    n_cams = len(grabbers)
    if show_ego:
        fig = plt.figure(figsize=(6.5 * n_cams, 9), dpi=110)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2])
        gs_top = gs[0].subgridspec(1, n_cams)
        gs_bot = gs[1].subgridspec(1, 2)
        axes = [fig.add_subplot(gs_top[0, i]) for i in range(n_cams)]
        ax_rot = fig.add_subplot(gs_bot[0, 0])
        ax_trans = fig.add_subplot(gs_bot[0, 1])
        ego_state = _setup_egomotion_axes(ax_rot, ax_trans, traj, trail_seconds)
    else:
        fig, axes = plt.subplots(1, n_cams, figsize=(6.5 * n_cams, 5.2), dpi=110)
        axes = list(np.atleast_1d(axes))

    images, overlay_artists = [], []
    for ax, cam, frame in zip(axes, cam_names, first):
        im = ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{cam} camera')
        images.append(im)
        overlay_artists.append([])

    def update(t):
        idx = frame_indices[t]
        for ax_i, (ax, cam, grabber, im) in enumerate(
                zip(axes, cam_names, grabbers, images)):
            frame = grabber.read_at(idx)
            if frame is not None:
                im.set_data(frame)

            for artist in overlay_artists[ax_i]:
                artist.remove()
            overlay_artists[ax_i] = []

            dets = by_cam_frame.get((cam, idx), [])
            for det in dets:
                color = _tag_color(det['tag_id'])
                corners = np.vstack([det['corners_uv'], det['corners_uv'][0]])
                line, = ax.plot(corners[:, 0], corners[:, 1],
                                color=color, linewidth=1.5)
                cu, cv = det['center_uv']
                dot = ax.scatter([cu], [cv], s=40, c=color,
                                 edgecolors='white', linewidths=0.8, zorder=5)
                rng = det['range_m']
                rng_txt = f'{rng:.2f}m' if np.isfinite(rng) else 'n/a'
                label = ax.text(
                    cu + 8, cv - 8,
                    f"id {det['tag_id']}  {rng_txt}\n"
                    f"phi {det['center_phi_deg']:.0f} "
                    f"theta {det['center_theta_deg']:.0f}",
                    color=color, fontsize=7, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='black',
                              alpha=0.5, edgecolor='none'))
                overlay_artists[ax_i].extend([line, dot, label])

            time_s = dets[0]['time_s'] if dets else np.nan
            time_txt = f', t={time_s:.2f}s' if np.isfinite(time_s) else ''
            ax.set_title(f'{cam} camera (frame {idx}{time_txt}) '
                         f'- {len(dets)} tag(s)')

        if show_ego:
            _update_egomotion(ego_state, idx)

    fig.tight_layout()
    write_movie(fig, update, len(frame_indices), output_path, fps)
    for g in grabbers:
        g.release()


def main():
    parser = argparse.ArgumentParser(
        description='Overlay detected AprilTag locations on rpi_drone videos')
    parser.add_argument('--rpi-dir', type=str, default=None,
                        help='rpi_drone directory of recording_* folders '
                             '(default: <data_folder>/stimulus/raw/rpi_drone)')
    parser.add_argument('--calib-dir', type=str, default=None,
                        help='Dome calibration directory (default: <rpi-dir>/dome_calib)')
    parser.add_argument('--recording', type=str, default=None,
                        help='recording_* folder to render (default: first found)')
    parser.add_argument('--cameras', type=str, nargs='+', default=['right', 'left'],
                        choices=['right', 'left'], help='Cameras to show')
    parser.add_argument('--output-dir', type=str, default='figures/plot_tag_location',
                        help='Output directory for the movie')
    parser.add_argument('--tag-family', type=str, default='tag36h11',
                        help='AprilTag family (default: tag36h11)')
    parser.add_argument('--tag-size', type=float, default=DEFAULT_TAG_SIZE,
                        help=f'Tag side length in metres (default: {DEFAULT_TAG_SIZE})')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Limit number of rendered frames (default: all)')
    parser.add_argument('--step', type=int, default=1,
                        help='Render every Nth video frame (default: 1)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output frame rate (default: native / step)')
    parser.add_argument('--no-egomotion', action='store_true',
                        help='Do not add the drone rotation/translation panels')
    parser.add_argument('--trail-seconds', type=float, default=6.0,
                        help='Width of the scrolling egomotion window (default: 6s)')
    args = parser.parse_args()

    data_folder = Path(load_config()['data']['data_folder'])
    rpi_dir = Path(args.rpi_dir) if args.rpi_dir else data_folder / 'stimulus' / 'raw' / 'rpi_drone'
    calib_dir = Path(args.calib_dir) if args.calib_dir else rpi_dir / 'dome_calib'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.recording:
        rec = rpi_dir / args.recording
    else:
        candidates = sorted(p for p in rpi_dir.glob('recording_*') if p.is_dir())
        if not candidates:
            print(f'No recording_* folders found in {rpi_dir}.')
            return
        rec = candidates[0]

    print(f'Extracting tags from {rec.name}...')
    records = extract_tags_from_recording(
        rec, calib_dir, tag_family=args.tag_family, tag_size=args.tag_size)

    traj = None
    if not args.no_egomotion:
        print('Computing board trajectory + drone egomotion...')
        traj, _, _ = compute_board_trajectory(records, args.tag_size)
        if traj is None:
            print('  Board trajectory unavailable; rendering without egomotion.')

    out_path = output_dir / f'{rec.name}_tag_location.mp4'
    print(f'Rendering movie -> {out_path}')
    make_tag_location_movie(
        rec, records, out_path, cameras=tuple(args.cameras),
        max_frames=args.max_frames, step=args.step, fps=args.fps, traj=traj,
        trail_seconds=args.trail_seconds)


if __name__ == '__main__':
    main()
