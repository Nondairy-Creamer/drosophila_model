"""Estimate drone pose trajectory from dual-camera AprilTag observations.

Uses batch optimization (reprojection error minimization) over all frames
from two rigidly-mounted cameras to recover the drone body pose at each
timestep.  More accurate than per-frame PnP because it jointly optimizes
across both cameras and can incorporate temporal smoothness.

Pipeline:
    1. Detect AprilTags in both camera streams
    2. Load calibration (intrinsics per camera, body↔camera extrinsics)
    3. Initialize poses via per-frame PnP
    4. Refine with batch least-squares (minimize reprojection error)

Transform conventions (T_A_B means "transforms points from B to A"):
    T_world_body  – drone pose:   p_world = T_world_body @ p_body
    T_body_cam    – camera mount: p_body  = T_body_cam  @ p_cam

Dependencies:
    pip install numpy scipy opencv-python pupil-apriltags

Usage:
    python drone_analysis/estimate_pose.py \\
        --left-video left_0.mp4 --right-video right_0.mp4 \\
        --calibration calibration.json --tag-map tags.json \\
        --output trajectory.npz
"""

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy import sparse
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json


# ── Pose helpers ─────────────────────────────────────────────────────────

def pose_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Rotation vector + translation → 4×4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    T[:3, 3] = tvec
    return T


def matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4×4 homogeneous matrix → (rvec, tvec)."""
    return Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3].copy()


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class CameraCalibration:
    """Intrinsic + extrinsic calibration for one camera.

    Attributes:
        camera_matrix: (3, 3) intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
        dist_coeffs: distortion coefficients (OpenCV ordering)
        T_body_cam: (4, 4) camera-frame → body-frame transform
    """
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    T_body_cam: np.ndarray

    @property
    def T_cam_body(self) -> np.ndarray:
        """Inverse: body-frame → camera-frame."""
        return np.linalg.inv(self.T_body_cam)


@dataclass
class DualCameraRig:
    """Calibration for a two-camera drone setup."""
    cameras: dict[int, CameraCalibration]

    def save(self, path: str):
        data = {}
        for cam_id, cal in self.cameras.items():
            data[str(cam_id)] = {
                'camera_matrix': cal.camera_matrix.tolist(),
                'dist_coeffs': cal.dist_coeffs.tolist(),
                'T_body_cam': cal.T_body_cam.tolist(),
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'DualCameraRig':
        with open(path) as f:
            data = json.load(f)
        cameras = {}
        for cam_id_str, d in data.items():
            cameras[int(cam_id_str)] = CameraCalibration(
                camera_matrix=np.array(d['camera_matrix']),
                dist_coeffs=np.array(d['dist_coeffs']),
                T_body_cam=np.array(d['T_body_cam']),
            )
        return cls(cameras=cameras)

    @classmethod
    def create_from_fov(cls, image_size: tuple[int, int],
                        hfov_deg: float = 120.0,
                        vfov_deg: float = 90.0,
                        baseline_m: float = 0.05) -> 'DualCameraRig':
        """Create rig for two side-facing cameras from FOV specs.

        Conventions (see coordinate_conventions.md):
            Body frame:  +X forward, +Y left, +Z up
            Camera frame (OpenCV): +X right, +Y down, +Z forward

        Camera 0 ("right"): optical axis along body −Y  (phi = −90°)
            FOV spans phi ∈ [−30°, −150°], theta ∈ [45°, 135°]
        Camera 1 ("left"):  optical axis along body +Y  (phi = +90°)
            FOV spans phi ∈ [+30°, +150°], theta ∈ [45°, 135°]

        Cameras are offset ±baseline/2 along body Y (back-to-back).

        Args:
            image_size: (height, width) in pixels
            hfov_deg: horizontal field of view (default 120°)
            vfov_deg: vertical field of view (default 90°)
            baseline_m: lens-to-lens distance in metres (default 0.05)
        """
        h, w = image_size
        fx = (w / 2) / np.tan(np.radians(hfov_deg / 2))
        fy = (h / 2) / np.tan(np.radians(vfov_deg / 2))
        K = np.array([[fx, 0, (w - 1) / 2],
                       [0, fy, (h - 1) / 2],
                       [0,  0,  1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        half_b = baseline_m / 2

        # Right camera: facing body −Y, offset toward −Y
        #   cam_Z (forward) → body −Y
        #   cam_Y (down)    → body −Z
        #   cam_X (right)   → body −X
        T_body_cam0 = np.array([[-1, 0,  0, 0],
                                 [ 0, 0, -1, -half_b],
                                 [ 0,-1,  0, 0],
                                 [ 0, 0,  0, 1]], dtype=np.float64)

        # Left camera: facing body +Y, offset toward +Y
        #   cam_Z (forward) → body +Y
        #   cam_Y (down)    → body −Z
        #   cam_X (right)   → body +X
        T_body_cam1 = np.array([[ 1, 0, 0, 0],
                                 [ 0, 0, 1, half_b],
                                 [ 0,-1, 0, 0],
                                 [ 0, 0, 0, 1]], dtype=np.float64)

        return cls(cameras={
            0: CameraCalibration(K.copy(), dist.copy(), T_body_cam0),
            1: CameraCalibration(K.copy(), dist.copy(), T_body_cam1),
        })


@dataclass
class TagDetection:
    """A single AprilTag detection in one frame."""
    tag_id: int
    corners: np.ndarray    # (4, 2) pixel coords, pupil-apriltags ordering
    camera_id: int
    frame_idx: int


@dataclass
class TagMap:
    """Known 3D positions of AprilTag corners in the world frame."""
    tags: dict[int, np.ndarray] = field(default_factory=dict)

    def add_tag(self, tag_id: int, center: np.ndarray, size: float,
                R_world_tag: np.ndarray | None = None):
        """Add a planar tag.

        Corner order (matching pupil-apriltags): BL, BR, TR, TL when viewed
        from the front.  Tag-local frame: X right, Y up, Z out-of-face.

        Args:
            tag_id: unique identifier
            center: (3,) world position of tag centre
            size: side length in world units (metres)
            R_world_tag: (3, 3) tag→world rotation (default: identity)
        """
        if R_world_tag is None:
            R_world_tag = np.eye(3)
        half = size / 2.0
        corners_local = np.array([[-half, -half, 0],
                                   [ half, -half, 0],
                                   [ half,  half, 0],
                                   [-half,  half, 0]])
        self.tags[tag_id] = (R_world_tag @ corners_local.T).T + center

    def add_tag_corners(self, tag_id: int, corners_3d: np.ndarray):
        """Add a tag with explicit 3D corner positions (4, 3)."""
        self.tags[tag_id] = np.asarray(corners_3d, dtype=np.float64)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({str(k): v.tolist() for k, v in self.tags.items()}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TagMap':
        with open(path) as f:
            data = json.load(f)
        tm = cls()
        for k, v in data.items():
            tm.tags[int(k)] = np.array(v, dtype=np.float64)
        return tm


# ── Equirectangular camera model ──────────────────────────────────────────

@dataclass
class EquirectCamera:
    """Equirectangular camera defined by angular FOV and body-frame offset.

    Projection: pixels map linearly to (phi, theta) in body-frame spherical
    coordinates.  Body frame: +X forward, +Y left, +Z up.

    phi   = atan2(Y, X)                 azimuth, 0=forward, +90=left
    theta = atan2(sqrt(X²+Y²), Z)      polar,   0=up, 90=horiz, 180=down
    """
    phi_range_deg: tuple[float, float]     # (start, end) azimuth for u=0..W-1
    theta_range_deg: tuple[float, float]   # (start, end) polar for v=0..H-1
    image_size: tuple[int, int]            # (H, W) in pixels
    offset_body: np.ndarray                # (3,) camera position in body frame


@dataclass
class EquirectRig:
    """Two equirectangular cameras on a drone."""
    cameras: dict[int, EquirectCamera]

    @classmethod
    def create_default(cls, image_size: tuple[int, int],
                       baseline_m: float = 0.05) -> 'EquirectRig':
        """Standard dual side-facing camera rig.

        Camera 0 ("right"): phi in [-30, -150], theta in [45, 135]
        Camera 1 ("left"):  phi in [150, 30],   theta in [45, 135]
        Offset ±baseline/2 along body Y.
        """
        half_b = baseline_m / 2
        return cls(cameras={
            0: EquirectCamera(
                phi_range_deg=(-30.0, -150.0),
                theta_range_deg=(45.0, 135.0),
                image_size=image_size,
                offset_body=np.array([0.0, -half_b, 0.0]),
            ),
            1: EquirectCamera(
                phi_range_deg=(150.0, 30.0),
                theta_range_deg=(45.0, 135.0),
                image_size=image_size,
                offset_body=np.array([0.0, half_b, 0.0]),
            ),
        })


# ── Detection ────────────────────────────────────────────────────────────

def load_video_grayscale(path: str) -> np.ndarray:
    """Load a video as (T, H, W) uint8 grayscale."""
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


def detect_tags_in_frame(frame_gray: np.ndarray, camera_id: int,
                         frame_idx: int, detector) -> list[TagDetection]:
    """Detect AprilTags in a single grayscale frame."""
    results = detector.detect(frame_gray)
    return [
        TagDetection(
            tag_id=r.tag_id,
            corners=np.array(r.corners, dtype=np.float64),
            camera_id=camera_id,
            frame_idx=frame_idx,
        )
        for r in results
    ]


def detect_all_tags(videos: dict[int, np.ndarray],
                    tag_family: str = 'tag36h11',
                    nthreads: int = 4) -> list[TagDetection]:
    """Run AprilTag detection on all cameras and frames.

    Args:
        videos: {camera_id: (T, H, W) uint8 grayscale}
        tag_family: AprilTag family string
        nthreads: detector thread count

    Returns:
        All detections, sorted by (frame_idx, camera_id).
    """
    from pupil_apriltags import Detector

    detector = Detector(families=tag_family, nthreads=nthreads)
    all_detections: list[TagDetection] = []

    for cam_id in sorted(videos.keys()):
        frames = videos[cam_id]
        print(f'  Camera {cam_id}: detecting in {len(frames)} frames...')
        for i, frame in enumerate(frames):
            all_detections.extend(
                detect_tags_in_frame(frame, cam_id, i, detector))
            if (i + 1) % 500 == 0:
                print(f'    {i + 1}/{len(frames)} '
                      f'({len(all_detections)} detections so far)')
        n = sum(1 for d in all_detections if d.camera_id == cam_id)
        print(f'    → {n} detections')

    all_detections.sort(key=lambda d: (d.frame_idx, d.camera_id))
    print(f'  Total: {len(all_detections)} detections')
    return all_detections


# ── Per-frame PnP (initialization) ───────────────────────────────────────

def _pnp_single_camera(
    cam_dets: list[TagDetection],
    tag_map: TagMap,
    calib: CameraCalibration,
) -> np.ndarray | None:
    """Solve PnP for one camera's detections → T_world_body (4×4) or None."""
    obj_pts, img_pts = [], []
    for det in cam_dets:
        if det.tag_id not in tag_map.tags:
            continue
        obj_pts.append(tag_map.tags[det.tag_id])
        img_pts.append(det.corners)
    if not obj_pts:
        return None
    obj_pts = np.vstack(obj_pts)
    img_pts = np.vstack(img_pts)
    if len(obj_pts) < 4:
        return None

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts,
        calib.camera_matrix, calib.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    # solvePnP returns T_cam_world: p_cam = R @ p_world + t
    T_cam_world = pose_to_matrix(rvec.ravel(), tvec.ravel())
    T_world_cam = np.linalg.inv(T_cam_world)
    # T_world_body = T_world_cam @ T_cam_body
    return T_world_cam @ calib.T_cam_body


def _pnp_multi_camera(
    frame_dets: list[TagDetection],
    tag_map: TagMap,
    rig: DualCameraRig,
) -> np.ndarray | None:
    """Best per-frame body pose from whichever camera has the most points."""
    best_pose = None
    best_n = 0
    for cam_id, calib in rig.cameras.items():
        cam_dets = [d for d in frame_dets if d.camera_id == cam_id]
        if not cam_dets:
            continue
        n_pts = sum(
            4 for d in cam_dets if d.tag_id in tag_map.tags)
        pose = _pnp_single_camera(cam_dets, tag_map, calib)
        if pose is not None and n_pts > best_n:
            best_pose = pose
            best_n = n_pts
    return best_pose


def initialize_poses(
    detections: list[TagDetection],
    tag_map: TagMap,
    rig: DualCameraRig,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame PnP → initial rvecs/tvecs; missing frames interpolated.

    Returns:
        rvecs: (n_frames, 3)
        tvecs: (n_frames, 3)
    """
    by_frame: dict[int, list[TagDetection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame_idx, []).append(d)

    rvecs = np.zeros((n_frames, 3))
    tvecs = np.zeros((n_frames, 3))
    valid = np.zeros(n_frames, dtype=bool)

    for fi in range(n_frames):
        dets = by_frame.get(fi, [])
        if not dets:
            continue
        pose = _pnp_multi_camera(dets, tag_map, rig)
        if pose is not None:
            rvecs[fi], tvecs[fi] = matrix_to_pose(pose)
            valid[fi] = True

    n_valid = int(valid.sum())
    print(f'  PnP solved {n_valid}/{n_frames} frames')

    if n_valid == 0:
        print('  WARNING: no frames solved — check tag map / calibration')
        return rvecs, tvecs

    if n_valid < n_frames:
        vi = np.where(valid)[0]
        for dim in range(3):
            rvecs[:, dim] = np.interp(np.arange(n_frames), vi, rvecs[vi, dim])
            tvecs[:, dim] = np.interp(np.arange(n_frames), vi, tvecs[vi, dim])
        print(f'  Interpolated {n_frames - n_valid} missing frames')

    return rvecs, tvecs


# ── Batch optimisation ───────────────────────────────────────────────────

def _project_tag_corners(
    rvec_body: np.ndarray,
    tvec_body: np.ndarray,
    corners_world: np.ndarray,
    calib: CameraCalibration,
) -> np.ndarray:
    """Project (4, 3) world-frame tag corners → (4, 2) pixels."""
    T_world_body = pose_to_matrix(rvec_body, tvec_body)
    T_cam_world = calib.T_cam_body @ np.linalg.inv(T_world_body)

    rvec_cam = Rotation.from_matrix(T_cam_world[:3, :3]).as_rotvec()
    tvec_cam = T_cam_world[:3, 3]

    projected, _ = cv2.projectPoints(
        corners_world, rvec_cam, tvec_cam,
        calib.camera_matrix, calib.dist_coeffs,
    )
    return projected.reshape(-1, 2)


def _reprojection_residuals(
    params: np.ndarray,
    detections: list[TagDetection],
    tag_map: TagMap,
    rig: DualCameraRig,
    n_frames: int,
    smoothness_weight: float,
) -> np.ndarray:
    """Residuals = [reprojection errors … , smoothness penalties …]."""
    poses = params.reshape(n_frames, 6)
    residuals: list[np.ndarray] = []

    for det in detections:
        rvec = poses[det.frame_idx, :3]
        tvec = poses[det.frame_idx, 3:]
        projected = _project_tag_corners(
            rvec, tvec, tag_map.tags[det.tag_id], rig.cameras[det.camera_id])
        residuals.append((projected - det.corners).ravel())  # 8 values

    if smoothness_weight > 0 and n_frames > 1:
        residuals.append((np.diff(poses, axis=0) * smoothness_weight).ravel())

    return np.concatenate(residuals)


def _build_sparsity(
    detections: list[TagDetection],
    n_frames: int,
    smoothness_weight: float,
) -> sparse.lil_matrix:
    """Jacobian sparsity: each detection depends only on its frame's 6 dof."""
    n_reproj = len(detections) * 8
    n_smooth = (n_frames - 1) * 6 if smoothness_weight > 0 else 0
    n_params = n_frames * 6

    A = sparse.lil_matrix((n_reproj + n_smooth, n_params), dtype=int)

    for i, det in enumerate(detections):
        r0 = i * 8
        c0 = det.frame_idx * 6
        A[r0:r0 + 8, c0:c0 + 6] = 1

    if smoothness_weight > 0:
        for t in range(n_frames - 1):
            r0 = n_reproj + t * 6
            c0 = t * 6
            A[r0:r0 + 6, c0:c0 + 12] = 1

    return A


def estimate_trajectory(
    detections: list[TagDetection],
    tag_map: TagMap,
    rig: DualCameraRig,
    n_frames: int,
    smoothness_weight: float = 0.1,
    init_rvecs: np.ndarray | None = None,
    init_tvecs: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Batch-optimise the full drone trajectory.

    Args:
        detections: tag detections (all cameras / frames)
        tag_map: known tag world positions
        rig: camera calibration
        n_frames: number of frames in the trajectory
        smoothness_weight: temporal smoothness regularisation (0 = off)
        init_rvecs, init_tvecs: (n_frames, 3) initial estimates (PnP).
            Computed automatically if not supplied.
        verbose: print solver progress

    Returns:
        dict with 'rvecs', 'tvecs', 'poses' (4×4), 'cost', etc.
    """
    valid_dets = [d for d in detections if d.tag_id in tag_map.tags]
    if not valid_dets:
        raise ValueError('No detections match known tags')

    print(f'  Optimising {n_frames} poses from {len(valid_dets)} detections '
          f'(smoothness={smoothness_weight:.3g})')

    if init_rvecs is None or init_tvecs is None:
        r, t = initialize_poses(valid_dets, tag_map, rig, n_frames)
        if init_rvecs is None:
            init_rvecs = r
        if init_tvecs is None:
            init_tvecs = t

    x0 = np.hstack([init_rvecs, init_tvecs]).ravel()
    sparsity = _build_sparsity(valid_dets, n_frames, smoothness_weight)

    result = least_squares(
        _reprojection_residuals, x0,
        args=(valid_dets, tag_map, rig, n_frames, smoothness_weight),
        jac_sparsity=sparsity,
        method='trf',
        x_scale='jac',
        ftol=1e-8, xtol=1e-8,
        max_nfev=500,
        verbose=2 if verbose else 0,
    )

    poses_flat = result.x.reshape(n_frames, 6)
    rvecs = poses_flat[:, :3]
    tvecs = poses_flat[:, 3:]
    T_world_body = np.array([
        pose_to_matrix(rvecs[i], tvecs[i]) for i in range(n_frames)])

    mean_reproj = np.sqrt(2 * result.cost / max(len(valid_dets) * 8, 1))
    print(f'  Done — cost={result.cost:.4f}  mean_reproj={mean_reproj:.2f}px  '
          f'nfev={result.nfev}  status={result.status}')

    return {
        'rvecs': rvecs,
        'tvecs': tvecs,
        'poses': T_world_body,
        'cost': result.cost,
        'mean_reprojection_px': mean_reproj,
        'n_detections': len(valid_dets),
        'optimizer_result': result,
    }


# ── Auto-tag batch optimisation (equirectangular) ────────────────────────
# Solves for the relative pose of a second tag jointly with the trajectory.
# Uses equirectangular projection matching the actual camera geometry.
# Only the tag size is needed — no world-frame tag positions required.

def _tag_local_corners(tag_size: float) -> np.ndarray:
    """Tag corners in the tag's local frame (4, 3).

    Order: BL, BR, TR, TL (matching pupil_apriltags).
    """
    half = tag_size / 2.0
    return np.array([[-half, -half, 0],
                      [ half, -half, 0],
                      [ half,  half, 0],
                      [-half,  half, 0]], dtype=np.float64)


def _project_equirect(
    rvec_body: np.ndarray,
    tvec_body: np.ndarray,
    corners_world: np.ndarray,
    cam: EquirectCamera,
) -> np.ndarray:
    """Project (N, 3) world-frame points → (N, 2) equirectangular pixels.

    Transforms to body frame, subtracts camera offset, then maps
    spherical (phi, theta) to pixels linearly.
    """
    R = Rotation.from_rotvec(rvec_body).as_matrix()
    p_body = (R.T @ (corners_world - tvec_body).T).T
    p_rel = p_body - cam.offset_body

    phi_deg = np.degrees(np.arctan2(p_rel[:, 1], p_rel[:, 0]))
    r_horiz = np.sqrt(p_rel[:, 0]**2 + p_rel[:, 1]**2)
    theta_deg = np.degrees(np.arctan2(r_horiz, p_rel[:, 2]))

    H, W = cam.image_size
    phi_s, phi_e = cam.phi_range_deg
    theta_s, theta_e = cam.theta_range_deg
    u = (phi_deg - phi_s) / (phi_e - phi_s) * (W - 1)
    v = (theta_deg - theta_s) / (theta_e - theta_s) * (H - 1)

    return np.column_stack([u, v])


def _pnp_equirect(
    corners_2d: np.ndarray,
    tag_size: float,
    cam: EquirectCamera,
) -> np.ndarray | None:
    """Solve for body pose from one tag's pixel corners (equirectangular).

    The tag's local frame is treated as the world origin.
    Uses Levenberg-Marquardt on equirectangular reprojection residuals.

    Returns:
        T_world_body (4×4) or None on failure.
    """
    corners_3d = _tag_local_corners(tag_size)
    H, W = cam.image_size
    phi_s, phi_e = cam.phi_range_deg
    theta_s, theta_e = cam.theta_range_deg

    phis_rad = np.radians(
        phi_s + (phi_e - phi_s) * corners_2d[:, 0] / (W - 1))
    thetas_rad = np.radians(
        theta_s + (theta_e - theta_s) * corners_2d[:, 1] / (H - 1))
    phi_c, theta_c = phis_rad.mean(), thetas_rad.mean()

    d_body = np.array([np.cos(phi_c) * np.sin(theta_c),
                       np.sin(phi_c) * np.sin(theta_c),
                       np.cos(theta_c)])
    d_body /= np.linalg.norm(d_body)

    ang = max(phis_rad.max() - phis_rad.min(),
              thetas_rad.max() - thetas_rad.min())
    dist = tag_size / (2 * np.tan(ang / 2)) if ang > np.radians(0.5) else 2.0

    # Tag face normal [0,0,1] should appear as -d_body in body frame;
    # tag up [0,1,0] should roughly align with body up [0,0,1].
    R_body_world, _ = Rotation.align_vectors(
        [-d_body, [0, 0, 1]],
        [[0, 0, 1], [0, 1, 0]],
        weights=[1.0, 0.5],
    )
    R_wb = R_body_world.inv().as_matrix()
    rvec_init = Rotation.from_matrix(R_wb).as_rotvec()
    tvec_init = -R_wb @ (cam.offset_body + dist * d_body)

    def residuals(params):
        return (_project_equirect(params[:3], params[3:], corners_3d, cam)
                - corners_2d).ravel()

    result = least_squares(residuals,
                           np.concatenate([rvec_init, tvec_init]),
                           method='lm', max_nfev=200)
    if result.cost > 500.0:
        return None
    return pose_to_matrix(result.x[:3], result.x[3:6])


def _init_body_poses_anchor(
    detections: list[TagDetection],
    rig: EquirectRig,
    tag_size: float,
    anchor_id: int,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize body poses using anchor-tag detections (equirectangular)."""
    anchor_dets = [d for d in detections if d.tag_id == anchor_id]
    by_frame: dict[int, list[TagDetection]] = {}
    for d in anchor_dets:
        by_frame.setdefault(d.frame_idx, []).append(d)

    rvecs = np.zeros((n_frames, 3))
    tvecs = np.zeros((n_frames, 3))
    valid = np.zeros(n_frames, dtype=bool)

    for fi in range(n_frames):
        dets = by_frame.get(fi, [])
        for det in dets:
            cam = rig.cameras[det.camera_id]
            T_world_body = _pnp_equirect(det.corners, tag_size, cam)
            if T_world_body is None:
                continue
            rvecs[fi], tvecs[fi] = matrix_to_pose(T_world_body)
            valid[fi] = True
            break

    n_valid = int(valid.sum())
    print(f'  Body pose init (anchor tag {anchor_id}): {n_valid}/{n_frames}')

    if n_valid > 0 and n_valid < n_frames:
        vi = np.where(valid)[0]
        for dim in range(3):
            rvecs[:, dim] = np.interp(np.arange(n_frames), vi, rvecs[vi, dim])
            tvecs[:, dim] = np.interp(np.arange(n_frames), vi, tvecs[vi, dim])
        print(f'  Interpolated {n_frames - n_valid} missing frames')
    elif n_valid == 0:
        print('  WARNING: no frames solved from anchor tag')

    return rvecs, tvecs


def _init_float_tag_pose(
    detections: list[TagDetection],
    rig: EquirectRig,
    tag_size: float,
    float_id: int,
    body_rvecs: np.ndarray,
    body_tvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the float tag's pose relative to the anchor (equirectangular).

    For each float-tag detection, runs PnP in the float tag's local frame
    to get T_float_body, then chains with the anchor-based body pose:
    T_world_float = T_world_body @ inv(T_float_body).

    Returns:
        (rvec, tvec) of T_world_float — median across sampled frames.
    """
    float_dets = [d for d in detections if d.tag_id == float_id]
    step = max(1, len(float_dets) // 200)
    sample = float_dets[::step]

    float_rvecs, float_tvecs = [], []
    for det in sample:
        cam = rig.cameras[det.camera_id]
        T_float_body = _pnp_equirect(det.corners, tag_size, cam)
        if T_float_body is None:
            continue
        T_world_body = pose_to_matrix(
            body_rvecs[det.frame_idx], body_tvecs[det.frame_idx])
        T_world_float = T_world_body @ np.linalg.inv(T_float_body)
        r, t = matrix_to_pose(T_world_float)
        float_rvecs.append(r)
        float_tvecs.append(t)

    if not float_rvecs:
        print(f'  WARNING: no float tag pose estimates for tag {float_id}')
        return np.zeros(3), np.zeros(3)

    rvec_init = np.median(float_rvecs, axis=0)
    tvec_init = np.median(float_tvecs, axis=0)
    print(f'  Float tag {float_id} init: pos=({tvec_init[0]:.3f}, '
          f'{tvec_init[1]:.3f}, {tvec_init[2]:.3f}) from '
          f'{len(float_rvecs)}/{len(sample)} samples')
    return rvec_init, tvec_init


def _reprojection_residuals_auto(
    params: np.ndarray,
    detections: list[TagDetection],
    anchor_corners: np.ndarray,
    local_corners: np.ndarray,
    float_tag_id: int,
    rig: EquirectRig,
    n_frames: int,
    smoothness_weight: float,
) -> np.ndarray:
    """Residuals with float-tag pose as unknowns (equirectangular).

    params: [float_rvec(3), float_tvec(3), body_pose_0(6), …, body_pose_N(6)]
    """
    float_rvec = params[:3]
    float_tvec = params[3:6]
    body_params = params[6:].reshape(n_frames, 6)

    T_world_float = pose_to_matrix(float_rvec, float_tvec)
    float_corners = (
        (T_world_float[:3, :3] @ local_corners.T).T + T_world_float[:3, 3])

    residuals: list[np.ndarray] = []
    for det in detections:
        corners_3d = (float_corners if det.tag_id == float_tag_id
                      else anchor_corners)
        projected = _project_equirect(
            body_params[det.frame_idx, :3], body_params[det.frame_idx, 3:],
            corners_3d, rig.cameras[det.camera_id])
        residuals.append((projected - det.corners).ravel())

    if smoothness_weight > 0 and n_frames > 1:
        residuals.append(
            (np.diff(body_params, axis=0) * smoothness_weight).ravel())

    return np.concatenate(residuals)


def _build_sparsity_auto(
    detections: list[TagDetection],
    float_tag_id: int,
    n_frames: int,
    smoothness_weight: float,
) -> sparse.lil_matrix:
    """Jacobian sparsity with float-tag pose in first 6 columns."""
    n_reproj = len(detections) * 8
    n_smooth = (n_frames - 1) * 6 if smoothness_weight > 0 else 0
    n_params = 6 + n_frames * 6

    A = sparse.lil_matrix((n_reproj + n_smooth, n_params), dtype=int)

    for i, det in enumerate(detections):
        r0 = i * 8
        c_body = 6 + det.frame_idx * 6
        A[r0:r0 + 8, c_body:c_body + 6] = 1
        if det.tag_id == float_tag_id:
            A[r0:r0 + 8, 0:6] = 1

    if smoothness_weight > 0:
        for t in range(n_frames - 1):
            r0 = n_reproj + t * 6
            c0 = 6 + t * 6
            A[r0:r0 + 6, c0:c0 + 12] = 1

    return A


def estimate_trajectory_auto(
    detections: list[TagDetection],
    rig: EquirectRig,
    tag_size: float,
    n_frames: int,
    smoothness_weight: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Batch-optimise trajectory, auto-estimating relative tag positions.

    Uses equirectangular projection. Fixes one tag as the world origin
    and solves for the other tag's pose (6 DOF) jointly with the drone
    trajectory. Only requires the tag side length.
    """
    tag_ids = sorted(set(d.tag_id for d in detections))
    if len(tag_ids) != 2:
        raise ValueError(
            f'Expected exactly 2 tag IDs, got {tag_ids}')

    counts = {tid: sum(1 for d in detections if d.tag_id == tid)
              for tid in tag_ids}
    anchor_id = max(counts, key=counts.get)
    float_id = [t for t in tag_ids if t != anchor_id][0]
    print(f'  Anchor tag: {anchor_id} ({counts[anchor_id]} det), '
          f'float tag: {float_id} ({counts[float_id]} det)')

    local_corners = _tag_local_corners(tag_size)
    anchor_corners = local_corners.copy()

    body_rvecs, body_tvecs = _init_body_poses_anchor(
        detections, rig, tag_size, anchor_id, n_frames)
    float_rvec, float_tvec = _init_float_tag_pose(
        detections, rig, tag_size, float_id, body_rvecs, body_tvecs)

    x0 = np.concatenate([
        float_rvec, float_tvec,
        np.hstack([body_rvecs, body_tvecs]).ravel(),
    ])

    valid_dets = [d for d in detections if d.tag_id in (anchor_id, float_id)]
    n_det = len(valid_dets)
    print(f'  Optimising {n_frames} poses + tag transform '
          f'from {n_det} detections (smoothness={smoothness_weight:.3g})')

    sparsity = _build_sparsity_auto(
        valid_dets, float_id, n_frames, smoothness_weight)

    result = least_squares(
        _reprojection_residuals_auto, x0,
        args=(valid_dets, anchor_corners, local_corners, float_id,
              rig, n_frames, smoothness_weight),
        jac_sparsity=sparsity,
        method='trf',
        x_scale='jac',
        ftol=1e-8, xtol=1e-8,
        max_nfev=1000,
        verbose=2 if verbose else 0,
    )

    float_rvec_opt = result.x[:3]
    float_tvec_opt = result.x[3:6]
    body_params = result.x[6:].reshape(n_frames, 6)
    rvecs = body_params[:, :3]
    tvecs = body_params[:, 3:]

    T_world_body = np.array([
        pose_to_matrix(rvecs[i], tvecs[i]) for i in range(n_frames)])
    T_world_float = pose_to_matrix(float_rvec_opt, float_tvec_opt)

    mean_reproj = np.sqrt(2 * result.cost / max(n_det * 8, 1))
    print(f'  Done — cost={result.cost:.4f}  mean_reproj={mean_reproj:.2f}px  '
          f'nfev={result.nfev}  status={result.status}')
    print(f'  Float tag estimated pos: ({float_tvec_opt[0]:.3f}, '
          f'{float_tvec_opt[1]:.3f}, {float_tvec_opt[2]:.3f})')

    return {
        'rvecs': rvecs,
        'tvecs': tvecs,
        'poses': T_world_body,
        'cost': result.cost,
        'mean_reprojection_px': mean_reproj,
        'n_detections': n_det,
        'optimizer_result': result,
        'anchor_tag_id': anchor_id,
        'float_tag_id': float_id,
        'T_world_float_tag': T_world_float,
    }


def run_pipeline_auto(
    left_video_path: str,
    right_video_path: str,
    rig: EquirectRig,
    tag_size: float = 0.16,
    tag_family: str = 'tag36h11',
    smoothness_weight: float = 0.1,
) -> dict:
    """Full pipeline with auto tag-position estimation (equirectangular).

    Only requires the tag side length — no TagMap needed.
    """
    print('Step 1/3  Loading videos...')
    videos = {
        0: load_video_grayscale(right_video_path),
        1: load_video_grayscale(left_video_path),
    }
    n_frames = min(len(v) for v in videos.values())
    for cam_id in videos:
        videos[cam_id] = videos[cam_id][:n_frames]
    print(f'  {n_frames} frames per camera')

    print('Step 2/3  Detecting AprilTags...')
    detections = detect_all_tags(videos, tag_family=tag_family)
    detected_ids = sorted(set(d.tag_id for d in detections))
    print(f'  Detected tag IDs: {detected_ids}')

    print('Step 3/3  Initialising + batch optimisation...')
    trajectory = estimate_trajectory_auto(
        detections, rig, tag_size, n_frames,
        smoothness_weight=smoothness_weight,
    )

    cap = cv2.VideoCapture(str(right_video_path))
    trajectory['fps'] = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    trajectory['n_frames'] = n_frames

    return trajectory


# ── Full pipeline (known tag positions) ──────────────────────────────────

def run_pipeline(
    left_video_path: str,
    right_video_path: str,
    tag_map: TagMap,
    rig: DualCameraRig,
    tag_family: str = 'tag36h11',
    smoothness_weight: float = 0.1,
    output_path: str | None = None,
) -> dict:
    """Load videos → detect → initialise → batch-optimise → (save).

    Args:
        left_video_path: left camera video
        right_video_path: right camera video
        tag_map: known tag world positions
        rig: dual-camera calibration
        tag_family: AprilTag family
        smoothness_weight: temporal smoothness weight
        output_path: save trajectory to .npz if provided

    Returns:
        trajectory dict from estimate_trajectory, plus 'fps' and 'n_frames'
    """
    print('Step 1/4  Loading videos...')
    videos = {
        0: load_video_grayscale(right_video_path),
        1: load_video_grayscale(left_video_path),
    }
    n_frames = min(len(v) for v in videos.values())
    for cam_id in videos:
        videos[cam_id] = videos[cam_id][:n_frames]
    print(f'  {n_frames} frames per camera')

    print('Step 2/4  Detecting AprilTags...')
    detections = detect_all_tags(videos, tag_family=tag_family)

    detected_ids = sorted(set(d.tag_id for d in detections))
    known_ids = sorted(tag_map.tags.keys())
    matched = sorted(set(detected_ids) & set(known_ids))
    print(f'  Detected tag IDs: {detected_ids}')
    print(f'  Known tag IDs:    {known_ids}')
    print(f'  Matched:          {matched}')

    print('Step 3/4  Initialising poses (per-frame PnP)...')
    valid_dets = [d for d in detections if d.tag_id in tag_map.tags]
    init_rvecs, init_tvecs = initialize_poses(
        valid_dets, tag_map, rig, n_frames)

    print('Step 4/4  Batch optimisation...')
    trajectory = estimate_trajectory(
        detections, tag_map, rig, n_frames,
        smoothness_weight=smoothness_weight,
        init_rvecs=init_rvecs, init_tvecs=init_tvecs,
    )

    cap = cv2.VideoCapture(str(right_video_path))
    trajectory['fps'] = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    trajectory['n_frames'] = n_frames

    if output_path:
        np.savez(
            output_path,
            rvecs=trajectory['rvecs'],
            tvecs=trajectory['tvecs'],
            poses=trajectory['poses'],
            fps=trajectory['fps'],
        )
        print(f'  Saved trajectory → {output_path}')

    return trajectory


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Estimate drone pose from dual-camera AprilTag observations')
    parser.add_argument('--left-video', required=True,
                        help='Left camera video file')
    parser.add_argument('--right-video', required=True,
                        help='Right camera video file')
    parser.add_argument('--calibration', required=True,
                        help='Camera rig calibration JSON')
    parser.add_argument('--tag-map', required=True,
                        help='Tag world positions JSON')
    parser.add_argument('--tag-family', default='tag36h11',
                        help='AprilTag family (default: tag36h11)')
    parser.add_argument('--smoothness', type=float, default=0.1,
                        help='Temporal smoothness weight (default: 0.1)')
    parser.add_argument('--output', default='trajectory.npz',
                        help='Output .npz file (default: trajectory.npz)')
    args = parser.parse_args()

    rig = DualCameraRig.load(args.calibration)
    tag_map = TagMap.load(args.tag_map)

    run_pipeline(
        left_video_path=args.left_video,
        right_video_path=args.right_video,
        tag_map=tag_map,
        rig=rig,
        tag_family=args.tag_family,
        smoothness_weight=args.smoothness,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
