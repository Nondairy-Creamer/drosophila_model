"""Extract AprilTag bearings and poses from the ``rpi_drone`` dome-camera videos.

The dome lens is *not* a pinhole camera, so we never form a flat
intrinsics/PnP projection. Instead we treat each camera as a spherical
observer: the per-eye dome calibration (``dome_right.csv`` / ``dome_left.csv``,
loaded and rotated into the body frame by
:func:`drone_utilities.load_dome_calibration`) gives, for a scattered set of
pixels, the viewing direction on the unit sphere. We interpolate that map so
any detected tag pixel becomes a unit direction in the body frame
(+x forward, +y left, +z up).

For every detection this gives:

- the bearing (``phi``, ``theta``) of the tag centre and corners;
- the full 6-DOF tag pose (range + orientation) from the four corner rays and
  the known tag size, by spherical PnP (:func:`solve_tag_pose_spherical`).

When several tags are mounted on one rigid board, :func:`autocalibrate_board`
recovers the board layout from co-detections and
:func:`compute_board_trajectory` solves one board pose per frame (filling
undetected frames by interpolation). :mod:`plot_tag_location` renders these as
a movie, including the drone egomotion derived from the board trajectory.

The CLI prints a per-recording detection summary; pass ``--output-dir`` to
also save the detections and board trajectory as CSV/NPZ.

Usage:
    python -m drosophila_model.data_handling.extract_tags
    python -m drosophila_model.data_handling.extract_tags --recording recording_<timestamp>
    python -m drosophila_model.data_handling.extract_tags --output-dir <dir>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from collections import Counter, defaultdict

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling.drone_utilities import (
    load_dome_calibration,
)
from drosophila_model.data_handling.estimate_pose import (
    DEFAULT_TAG_FAMILY,
    DEFAULT_TAG_SIZE,
    detect_tags_in_frame,
    _tag_local_corners,
)

# Per-eye azimuth offset placing the dome optical axis in the body frame,
# matching drone_utilities.process_rpi_drone_dataset (right faces -y, left +y).
CAMERA_CALIB = {
    'right': ('dome_right.csv', -80.0),
    'left': ('dome_left.csv', 80.0),
}

# Detector defaults. quad_decimate=1.0 (no input downsampling) roughly doubles
# the rate of full-board detections on the fisheye dome footage vs the library
# default of 2.0; CLAHE / extra blur were measured to hurt, so they are off.
DEFAULT_DETECTOR_PARAMS = dict(
    quad_decimate=1.0,
    quad_sigma=0.0,
    decode_sharpening=0.25,
    nthreads=4,
)


def make_detector(tag_family=DEFAULT_TAG_FAMILY, **params):
    """Build a pupil_apriltags Detector with our tuned defaults."""
    from pupil_apriltags import Detector

    cfg = {**DEFAULT_DETECTOR_PARAMS, **params}
    return Detector(families=tag_family, **cfg)


def _angles_to_unit(phi, theta):
    """Spherical angles (radians) -> (..., 3) unit direction vectors."""
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


class DomeProjector:
    """Maps image pixels to body-frame viewing directions via dome calibration.

    Built from the scattered ``(u, v) <-> (phi, theta)`` calibration points.
    Pixel -> direction is interpolated on the 3D unit sphere (interpolating
    unit vectors then renormalising), which avoids azimuth-wrap artefacts and
    keeps the spherical-observer interpretation. Pixels outside the convex hull
    of the calibration points are out of field-of-view and return NaN.
    """

    def __init__(self, calib_u, calib_v, calib_phi, calib_theta):
        from scipy.interpolate import LinearNDInterpolator

        pixels = np.column_stack([np.asarray(calib_u, float),
                                  np.asarray(calib_v, float)])
        dirs = _angles_to_unit(np.asarray(calib_phi, float),
                               np.asarray(calib_theta, float))
        self._interp = LinearNDInterpolator(pixels, dirs)

    @classmethod
    def from_dome_csv(cls, csv_path, phi_offset_deg):
        u, v, phi, theta = load_dome_calibration(csv_path, phi_offset_deg)
        return cls(u, v, phi, theta)

    def pixels_to_directions(self, uv):
        """(..., 2) pixel coords -> (..., 3) unit directions (NaN if outside)."""
        uv = np.asarray(uv, float)
        dirs = self._interp(uv.reshape(-1, 2))
        norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
        with np.errstate(invalid='ignore'):
            dirs = dirs / norm
        return dirs.reshape(uv.shape[:-1] + (3,))

    def pixels_to_angles(self, uv):
        """(..., 2) pixel coords -> (phi, theta) in radians (NaN if outside)."""
        d = self.pixels_to_directions(uv)
        phi = np.arctan2(d[..., 1], d[..., 0])
        theta = np.arccos(np.clip(d[..., 2], -1.0, 1.0))
        return phi, theta


def solve_tag_pose_spherical(corner_dirs, tag_size):
    """Full 6-DOF tag pose from 4 corner rays (spherical PnP, no pinhole model).

    Given the four tag-corner viewing directions in the body frame (unit
    vectors from the dome calibration) and the tag's real side length, solve
    for the rigid transform ``T_body_tag`` by minimising the *angular*
    reprojection error between predicted and measured corner directions. This
    needs no focal length / intrinsics and, unlike the simple
    ``d = x / tan(theta)`` estimate, recovers range *and* orientation, so it is
    correct for tilted (non-fronto-parallel) tags.

    Args:
        corner_dirs: (4, 3) unit directions of the tag corners, in
            pupil_apriltags order (BL, BR, TR, TL). NaNs (out-of-FOV corner)
            yield ``None``.
        tag_size: tag side length in metres.

    Returns:
        dict with ``rvec`` / ``tvec`` (T_body_tag, tag-frame -> body-frame),
        ``position`` (tag centre in body frame, == tvec), ``range_m`` (distance
        from the eye to the tag centre), ``normal`` (tag face normal in body
        frame), and ``reproj_err_deg`` (mean angular residual). ``None`` if the
        solve is degenerate.
    """
    corner_dirs = np.asarray(corner_dirs, float)
    if corner_dirs.shape != (4, 3) or not np.isfinite(corner_dirs).all():
        return None

    local = _tag_local_corners(tag_size)  # (4, 3): BL, BR, TR, TL; +Z out of face
    # Renormalise the measured rays (interpolation can perturb the norm).
    dirs = corner_dirs / np.linalg.norm(corner_dirs, axis=1, keepdims=True)

    centroid = dirs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-9:
        return None
    centroid /= norm

    # Range init from the diagonal subtense (simple x/tan(theta) geometry).
    diag_angle = np.arccos(np.clip(dirs[0] @ dirs[2], -1.0, 1.0))
    if diag_angle > np.radians(0.25):
        dist0 = (tag_size * np.sqrt(2) / 2.0) / np.tan(diag_angle / 2.0)
    else:
        dist0 = 2.0

    # Orientation init: tag face normal (+Z) points back toward the eye
    # (-centroid), tag up (+Y) roughly aligns with body up (+Z).
    R_init, _ = Rotation.align_vectors(
        [-centroid, [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        weights=[1.0, 0.5],
    )
    rvec_init = R_init.as_rotvec()
    tvec_init = dist0 * centroid

    def residuals(params):
        R = Rotation.from_rotvec(params[:3]).as_matrix()
        pred = (R @ local.T).T + params[3:6]
        pred /= np.linalg.norm(pred, axis=1, keepdims=True)
        return (pred - dirs).ravel()

    result = least_squares(
        residuals, np.concatenate([rvec_init, tvec_init]),
        method='lm', max_nfev=200,
    )

    rvec = result.x[:3]
    tvec = result.x[3:6]
    R = Rotation.from_rotvec(rvec).as_matrix()
    pred = (R @ local.T).T + tvec
    pred /= np.linalg.norm(pred, axis=1, keepdims=True)
    ang_err = np.degrees(np.arccos(np.clip(np.sum(pred * dirs, axis=1), -1.0, 1.0)))

    return {
        'rvec': rvec,
        'tvec': tvec,
        'position': tvec,
        'range_m': float(np.linalg.norm(tvec)),
        'normal': R @ np.array([0.0, 0.0, 1.0]),
        'reproj_err_deg': float(ang_err.mean()),
    }


def _kabsch(P, Q):
    """Rigid transform (R, t) mapping points P (N,3) onto Q (N,3): Q ~ R P + t."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


def autocalibrate_board(records, tag_size, min_codetections=5):
    """Estimate each tag's corners in a common board frame from the detections.

    Uses the per-detection single-tag poses (which share a deterministic,
    flip-consistent initialisation, so they are stable frame-to-frame): an
    anchor tag (the most-detected one) defines the board frame, and every other
    tag's corners are expressed in that frame as the median over all frames
    where it co-occurs with the anchor. The board origin is then recentred to
    the mean of all corners. No prior knowledge of the layout is required.

    Returns:
        (board_corners, anchor_id) where board_corners maps tag_id -> (4, 3)
        corner positions in the board frame, or ({}, None) if it can't be built.
    """
    valid = [r for r in records if np.isfinite(r['tag_tvec']).all()]
    if not valid:
        return {}, None

    counts = Counter(r['tag_id'] for r in valid)
    anchor_id = counts.most_common(1)[0][0]

    by_frame = defaultdict(dict)
    for r in valid:
        by_frame[(r['camera'], r['frame_idx'])][r['tag_id']] = r

    local = _tag_local_corners(tag_size)

    def corners_body(rec):
        R = Rotation.from_rotvec(rec['tag_rvec']).as_matrix()
        return (R @ local.T).T + rec['tag_tvec']

    accum = defaultdict(list)
    for dets in by_frame.values():
        if anchor_id not in dets:
            continue
        a = dets[anchor_id]
        R_ba = Rotation.from_rotvec(a['tag_rvec']).as_matrix()
        t_ba = a['tag_tvec']
        for tag_id, rec in dets.items():
            corners_anchor = (R_ba.T @ (corners_body(rec) - t_ba).T).T
            accum[tag_id].append(corners_anchor)

    board_corners = {}
    for tag_id, frames in accum.items():
        if tag_id != anchor_id and len(frames) < min_codetections:
            continue
        board_corners[tag_id] = np.median(np.stack(frames), axis=0)

    if not board_corners:
        return {}, None

    origin = np.mean(np.concatenate(list(board_corners.values())), axis=0)
    for tag_id in board_corners:
        board_corners[tag_id] = board_corners[tag_id] - origin
    return board_corners, anchor_id


def solve_board_pose_spherical(frame_dets, board_corners, init_rt=None):
    """Joint spherical PnP for a rigid multi-tag board from one frame.

    Pools the corner rays of every detected tag that is in the board model and
    solves a single board pose ``T_body_board`` by minimising the total angular
    reprojection error. Works with any subset (>=1) of the tags, and the extra
    spread across tags resolves the single-tag planar flip ambiguity.

    Args:
        frame_dets: detection dicts for one frame (any cameras).
        board_corners: tag_id -> (4, 3) corners in the board frame.
        init_rt: optional (rvec, tvec) initial guess (e.g. previous frame).

    Returns:
        dict with rvec, tvec (T_body_board), range_m (board centre distance),
        n_tags, reproj_err_deg; or None if no usable tags.
    """
    P, D, ranges = [], [], []
    for r in frame_dets:
        tid = r['tag_id']
        if tid not in board_corners:
            continue
        cd = np.asarray(r['corner_dirs'], float)
        if not np.isfinite(cd).all():
            continue
        cd = cd / np.linalg.norm(cd, axis=1, keepdims=True)
        P.append(board_corners[tid])
        D.append(cd)
        ranges.append(np.full(4, r['range_m'] if np.isfinite(r['range_m']) else 2.0))
    if not P:
        return None

    P = np.vstack(P)
    D = np.vstack(D)
    n_tags = len(D) // 4

    if init_rt is not None:
        rvec0, tvec0 = init_rt
    else:
        Q = D * np.concatenate(ranges)[:, None]
        R0, t0 = _kabsch(P, Q)
        rvec0 = Rotation.from_matrix(R0).as_rotvec()
        tvec0 = t0

    def residuals(params):
        R = Rotation.from_rotvec(params[:3]).as_matrix()
        pred = (R @ P.T).T + params[3:6]
        pred /= np.linalg.norm(pred, axis=1, keepdims=True)
        return (pred - D).ravel()

    result = least_squares(residuals, np.concatenate([rvec0, tvec0]),
                           method='lm', max_nfev=200)
    rvec, tvec = result.x[:3], result.x[3:6]
    R = Rotation.from_rotvec(rvec).as_matrix()
    pred = (R @ P.T).T + tvec
    pred /= np.linalg.norm(pred, axis=1, keepdims=True)
    ang_err = np.degrees(np.arccos(np.clip(np.sum(pred * D, axis=1), -1.0, 1.0)))

    return {
        'rvec': rvec,
        'tvec': tvec,
        'range_m': float(np.linalg.norm(tvec)),
        'n_tags': n_tags,
        'reproj_err_deg': float(ang_err.mean()),
    }


def compute_board_trajectory(records, tag_size, n_frames=None, interpolate=True,
                             min_codetections=5):
    """Per-frame rigid board pose, auto-calibrating the layout and filling gaps.

    Solves one board pose per frame that has >=1 detected board tag, then (if
    ``interpolate``) fills frames with no detection by SLERP/linear
    interpolation between the nearest solved frames. This is the practical
    mitigation for the many frames where the detector misses some or all tags.

    Returns:
        (traj, board_corners, anchor_id) where traj is a dict of per-frame
        arrays over [first_solved, last_solved]: frame_idx, time_s, rvec (.,3),
        tvec (.,3), range_m, phi_deg, theta_deg (board-centre bearing), n_tags,
        reproj_err_deg, measured (bool), interpolated (bool). traj is None if
        the board could not be calibrated.
    """
    board_corners, anchor_id = autocalibrate_board(
        records, tag_size, min_codetections=min_codetections)
    if not board_corners:
        print('  Board auto-calibration failed (not enough co-detections).')
        return None, {}, None
    print(f'  Board calibrated: anchor tag {anchor_id}, '
          f'tags {sorted(board_corners)}')

    by_frame = defaultdict(list)
    for r in records:
        by_frame[r['frame_idx']].append(r)
    time_by_frame = {r['frame_idx']: r['time_s'] for r in records}

    if n_frames is None:
        n_frames = max(by_frame) + 1

    rvecs = np.full((n_frames, 3), np.nan)
    tvecs = np.full((n_frames, 3), np.nan)
    n_tags = np.zeros(n_frames, dtype=int)
    reproj = np.full(n_frames, np.nan)
    measured = np.zeros(n_frames, dtype=bool)

    prev_rt = None
    for fi in range(n_frames):
        dets = by_frame.get(fi)
        if not dets:
            continue
        pose = solve_board_pose_spherical(dets, board_corners, init_rt=prev_rt)
        if pose is None:
            continue
        rvecs[fi] = pose['rvec']
        tvecs[fi] = pose['tvec']
        n_tags[fi] = pose['n_tags']
        reproj[fi] = pose['reproj_err_deg']
        measured[fi] = True
        prev_rt = (pose['rvec'], pose['tvec'])

    solved = np.where(measured)[0]
    if solved.size == 0:
        return None, board_corners, anchor_id
    lo, hi = solved[0], solved[-1]
    interpolated = np.zeros(n_frames, dtype=bool)

    if interpolate and solved.size >= 2:
        key_rots = Rotation.from_rotvec(rvecs[solved])
        slerp = Slerp(solved.astype(float), key_rots)
        gap = np.arange(lo, hi + 1)
        fill = gap[~measured[gap]]
        if fill.size:
            rvecs[fill] = slerp(fill.astype(float)).as_rotvec()
            for d in range(3):
                tvecs[fill, d] = np.interp(fill, solved, tvecs[solved, d])
            interpolated[fill] = True

    idx = np.arange(lo, hi + 1)
    keep = measured[idx] | interpolated[idx]
    idx = idx[keep]

    tvec_sel = tvecs[idx]
    dir_norm = tvec_sel / np.linalg.norm(tvec_sel, axis=1, keepdims=True)

    # Interpolate timestamps onto every kept frame (gap-filled frames have no
    # detection and therefore no recorded time of their own).
    known_f = np.array(sorted(time_by_frame))
    known_t = np.array([time_by_frame[f] for f in known_f])
    finite = np.isfinite(known_t)
    if finite.sum() >= 2:
        time_s = np.interp(idx, known_f[finite], known_t[finite])
    else:
        time_s = np.array([time_by_frame.get(int(i), np.nan) for i in idx])

    traj = {
        'frame_idx': idx,
        'time_s': time_s,
        'rvec': rvecs[idx],
        'tvec': tvec_sel,
        'range_m': np.linalg.norm(tvec_sel, axis=1),
        'phi_deg': np.degrees(np.arctan2(dir_norm[:, 1], dir_norm[:, 0])),
        'theta_deg': np.degrees(np.arccos(np.clip(dir_norm[:, 2], -1.0, 1.0))),
        'n_tags': n_tags[idx],
        'reproj_err_deg': reproj[idx],
        'measured': measured[idx],
        'interpolated': interpolated[idx],
    }
    n_meas = int(measured[idx].sum())
    n_interp = int(interpolated[idx].sum())
    print(f'  Board trajectory: {len(idx)} frames '
          f'({n_meas} measured, {n_interp} interpolated) '
          f'over frames {lo}-{hi}')
    return traj, board_corners, anchor_id


def _frame_times(recording_dir):
    """Video-frame times (s) from stereo_frames.csv, or None if unavailable."""
    path = Path(recording_dir) / 'stereo_frames.csv'
    if not path.exists():
        return None
    frames = pd.read_csv(path).sort_values('video_frame_index')
    return frames['time_s'].to_numpy(dtype=float)


def extract_tags_from_video(video_path, projector, camera_name,
                            tag_family=DEFAULT_TAG_FAMILY, frame_times=None,
                            tag_size=DEFAULT_TAG_SIZE, detector_params=None):
    """Detect AprilTags in one video and project them to body-frame bearings.

    Streams frames (no full-video load) and, for every detection, records the
    centre and four corner pixels together with their interpolated body-frame
    directions, (phi, theta) bearings, and the spherical-PnP pose (range +
    orientation) when all four corners are in the calibrated FOV.

    Returns:
        list[dict], one per detection, with keys: camera, frame_idx, time_s,
        tag_id, center_uv (2,), corners_uv (4, 2), center_dir (3,),
        corner_dirs (4, 3), center_phi_deg, center_theta_deg, in_fov,
        range_m, reproj_err_deg, tag_rvec (3,), tag_tvec (3,).
    """
    detector = make_detector(tag_family, **(detector_params or {}))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    records = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for det in detect_tags_in_frame(gray, camera_name, frame_idx, detector):
            corners_uv = det.corners                      # (4, 2)
            center_uv = corners_uv.mean(axis=0)           # (2,)
            corner_dirs = projector.pixels_to_directions(corners_uv)
            center_dir = projector.pixels_to_directions(center_uv)
            center_phi, center_theta = projector.pixels_to_angles(center_uv)
            in_fov = bool(np.isfinite(center_dir).all())
            t = (float(frame_times[frame_idx])
                 if frame_times is not None and frame_idx < len(frame_times)
                 else np.nan)

            pose = solve_tag_pose_spherical(corner_dirs, tag_size)
            records.append({
                'camera': camera_name,
                'frame_idx': frame_idx,
                'time_s': t,
                'tag_id': int(det.tag_id),
                'center_uv': center_uv,
                'corners_uv': corners_uv,
                'center_dir': center_dir,
                'corner_dirs': corner_dirs,
                'center_phi_deg': float(np.degrees(center_phi)),
                'center_theta_deg': float(np.degrees(center_theta)),
                'in_fov': in_fov,
                'range_m': pose['range_m'] if pose else np.nan,
                'reproj_err_deg': pose['reproj_err_deg'] if pose else np.nan,
                'tag_rvec': pose['rvec'] if pose else np.full(3, np.nan),
                'tag_tvec': pose['tvec'] if pose else np.full(3, np.nan),
            })
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f'  [{camera_name}] {frame_idx} frames, '
                  f'{len(records)} detections so far')

    cap.release()
    print(f'  [{camera_name}] done: {frame_idx} frames, {len(records)} detections')
    return records


def extract_tags_from_recording(recording_dir, calib_dir, tag_family=DEFAULT_TAG_FAMILY,
                                tag_size=DEFAULT_TAG_SIZE, detector_params=None):
    """Extract tag bearings + pose from both cameras of one recording folder."""
    recording_dir = Path(recording_dir)
    calib_dir = Path(calib_dir)
    frame_times = _frame_times(recording_dir)

    records = []
    for camera_name, (calib_file, phi_offset) in CAMERA_CALIB.items():
        video_path = recording_dir / f'{camera_name}.mp4'
        if not video_path.exists():
            print(f'  [{camera_name}] no video at {video_path}, skipping')
            continue
        projector = DomeProjector.from_dome_csv(calib_dir / calib_file, phi_offset)
        print(f'  [{camera_name}] detecting tags in {video_path.name}...')
        records.extend(extract_tags_from_video(
            video_path, projector, camera_name,
            tag_family=tag_family, frame_times=frame_times, tag_size=tag_size,
            detector_params=detector_params))
    return records


def save_records(records, out_dir, recording_name):
    """Save detections as a flat CSV (centre bearings) and a full .npz."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f'{recording_name}_tags.csv'
    npz_path = out_dir / f'{recording_name}_tags.npz'

    df = pd.DataFrame([{
        'camera': r['camera'],
        'frame_idx': r['frame_idx'],
        'time_s': r['time_s'],
        'tag_id': r['tag_id'],
        'u': r['center_uv'][0],
        'v': r['center_uv'][1],
        'phi_deg': r['center_phi_deg'],
        'theta_deg': r['center_theta_deg'],
        'dir_x': r['center_dir'][0],
        'dir_y': r['center_dir'][1],
        'dir_z': r['center_dir'][2],
        'range_m': r['range_m'],
        'reproj_err_deg': r['reproj_err_deg'],
        'in_fov': r['in_fov'],
    } for r in records])
    df.to_csv(csv_path, index=False)

    if records:
        np.savez(
            npz_path,
            camera=np.array([r['camera'] for r in records]),
            frame_idx=np.array([r['frame_idx'] for r in records]),
            time_s=np.array([r['time_s'] for r in records]),
            tag_id=np.array([r['tag_id'] for r in records]),
            center_uv=np.stack([r['center_uv'] for r in records]),
            corners_uv=np.stack([r['corners_uv'] for r in records]),
            center_dir=np.stack([r['center_dir'] for r in records]),
            corner_dirs=np.stack([r['corner_dirs'] for r in records]),
            range_m=np.array([r['range_m'] for r in records]),
            reproj_err_deg=np.array([r['reproj_err_deg'] for r in records]),
            tag_rvec=np.stack([r['tag_rvec'] for r in records]),
            tag_tvec=np.stack([r['tag_tvec'] for r in records]),
            in_fov=np.array([r['in_fov'] for r in records]),
        )
    print(f'  Saved {len(records)} detections -> {csv_path.name}, {npz_path.name}')
    return csv_path, npz_path


def save_board_trajectory(traj, out_dir, recording_name):
    """Save the per-frame board pose trajectory as CSV + NPZ."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'{recording_name}_board.csv'
    npz_path = out_dir / f'{recording_name}_board.npz'

    df = pd.DataFrame({
        'frame_idx': traj['frame_idx'],
        'time_s': traj['time_s'],
        'range_m': traj['range_m'],
        'phi_deg': traj['phi_deg'],
        'theta_deg': traj['theta_deg'],
        'n_tags': traj['n_tags'],
        'reproj_err_deg': traj['reproj_err_deg'],
        'measured': traj['measured'],
        'interpolated': traj['interpolated'],
    })
    df.to_csv(csv_path, index=False)
    np.savez(npz_path, **traj)
    print(f'  Saved board trajectory -> {csv_path.name}, {npz_path.name}')


def _print_summary(records):
    """Per-camera / per-tag detection summary."""
    if not records:
        print('  No tags detected.')
        return
    df = pd.DataFrame(records)
    print(f'  {len(df)} detections, {df["in_fov"].sum()} inside calibrated FOV')
    for cam, cam_df in df.groupby('camera'):
        ids = cam_df.groupby('tag_id').size().to_dict()
        rng = cam_df['range_m'].to_numpy()
        rng = rng[np.isfinite(rng)]
        rng_str = (f'range {rng.min():.2f}-{rng.max():.2f}m (median {np.median(rng):.2f})'
                   if rng.size else 'range n/a')
        print(f'  [{cam}] {len(cam_df)} detections, tag ids/counts: {ids}, {rng_str}')


def main():
    parser = argparse.ArgumentParser(
        description='Extract AprilTag bearings, tag poses and the board '
                    'trajectory from rpi_drone dome videos')
    parser.add_argument('--rpi-dir', type=str, default=None,
                        help='rpi_drone directory of recording_* folders '
                             '(default: <data_folder>/stimulus/raw/rpi_drone)')
    parser.add_argument('--calib-dir', type=str, default=None,
                        help='Dome calibration directory with dome_right.csv, '
                             'dome_left.csv (default: <rpi-dir>/dome_calib)')
    parser.add_argument('--recording', type=str, default=None,
                        help='Process only this recording_* folder (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='If given, save per-recording detection and board '
                             'trajectory CSV/NPZ files here (default: print '
                             'the summary only)')
    parser.add_argument('--tag-family', type=str, default=DEFAULT_TAG_FAMILY,
                        help=f'AprilTag family (default: {DEFAULT_TAG_FAMILY})')
    parser.add_argument('--tag-size', type=float, default=DEFAULT_TAG_SIZE,
                        help='Real AprilTag side length in metres '
                             f'(default: {DEFAULT_TAG_SIZE})')
    parser.add_argument('--quad-decimate', type=float,
                        default=DEFAULT_DETECTOR_PARAMS['quad_decimate'],
                        help='Detector input decimation; 1.0 = no downsampling '
                             '(default: 1.0, best for small/distant tags)')
    parser.add_argument('--quad-sigma', type=float,
                        default=DEFAULT_DETECTOR_PARAMS['quad_sigma'],
                        help='Gaussian blur applied before detection (default: 0.0)')
    parser.add_argument('--decode-sharpening', type=float,
                        default=DEFAULT_DETECTOR_PARAMS['decode_sharpening'],
                        help='Detector decode sharpening (default: 0.25)')
    parser.add_argument('--no-board', action='store_true',
                        help='Skip the multi-tag board pose solve / trajectory')
    parser.add_argument('--no-interpolate', action='store_true',
                        help='Do not fill undetected frames in the board trajectory')
    args = parser.parse_args()

    detector_params = dict(
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        decode_sharpening=args.decode_sharpening,
    )

    if args.rpi_dir:
        rpi_dir = Path(args.rpi_dir)
    else:
        rpi_dir = get_data_folder(load_config()) / 'stimulus' / 'raw' / 'rpi_drone'
    calib_dir = Path(args.calib_dir) if args.calib_dir else rpi_dir / 'dome_calib'
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.recording:
        recording_dirs = [rpi_dir / args.recording]
        if not recording_dirs[0].is_dir():
            print(f'Recording folder not found: {recording_dirs[0]}')
            return
    else:
        recording_dirs = sorted(p for p in rpi_dir.glob('recording_*') if p.is_dir())
    if not recording_dirs:
        print(f'No recording_* folders found in {rpi_dir}.')
        return

    for rec in recording_dirs:
        print(f'\n=== {rec.name} ===')
        records = extract_tags_from_recording(
            rec, calib_dir, tag_family=args.tag_family, tag_size=args.tag_size,
            detector_params=detector_params)
        _print_summary(records)
        if output_dir is not None:
            save_records(records, output_dir, rec.name)

        if not args.no_board and records:
            traj, _, _ = compute_board_trajectory(
                records, args.tag_size, interpolate=not args.no_interpolate)
            if traj is not None and output_dir is not None:
                save_board_trajectory(traj, output_dir, rec.name)


if __name__ == '__main__':
    main()
