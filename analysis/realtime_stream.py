from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from stereo_stream_runtime import (
    DEFAULT_PHI_LEFT_DEG,
    DEFAULT_PHI_RIGHT_DEG,
    DEFAULT_THETA_DEG,
    StereoOpticFlowEstimator,
    get_nominal_dt,
    open_source,
    read_source_frame,
)


DEFAULT_APRILTAG_MODULE_PATH = "/home/daewon/stereo_vis_control/auto_control/apriltag_pose.py"


def candidate_site_packages(module_path: Path) -> list[Path]:
    project_root = module_path.parent.parent
    candidates = [
        project_root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        project_root / "data_collection" / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    return [path for path in candidates if path.exists()]


def ensure_apriltag_dependencies(module_path: str) -> None:
    try:
        import pupil_apriltags  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    for site_packages in candidate_site_packages(Path(module_path).expanduser().resolve()):
        site_packages_str = str(site_packages)
        if site_packages_str not in sys.path:
            sys.path.insert(0, site_packages_str)

    import pupil_apriltags  # noqa: F401


def load_module_from_path(module_name: str, module_path: str) -> ModuleType:
    path = Path(module_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")

    ensure_apriltag_dependencies(str(path))

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_age_s(value: float | None, fallback: float = 1e9) -> float:
    if value is None:
        return fallback
    try:
        value_f = float(value)
    except Exception:
        return fallback
    if not np.isfinite(value_f):
        return fallback
    return max(0.0, value_f)


def build_display_state(apr: ModuleType, state: dict[str, Any], pose_source: str) -> dict[str, Any]:
    if hasattr(apr, "build_display_state"):
        return apr.build_display_state(state, pose_source)

    if pose_source == "filtered" and bool(state.get("filtered", False)):
        out = dict(state)
        out["pos_m"] = [float(v) for v in state.get("filtered_pos_m", state["pos_m"])]
        out["rpy_deg"] = [float(v) for v in state.get("filtered_rpy_deg", state["rpy_deg"])]
        out["display_source"] = "filtered"
        return out

    out = dict(state)
    out["display_source"] = "raw"
    return out


class LatestStereoFrame:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._seq = 0
        self._packet: tuple[int, int, float, np.ndarray, np.ndarray] | None = None
        self._closed = False
        self._error: BaseException | None = None

    def update(
        self,
        source_frame_index: int,
        capture_time_s: float,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
    ) -> None:
        with self._cond:
            self._seq += 1
            self._packet = (self._seq, source_frame_index, capture_time_s, left_frame, right_frame)
            self._cond.notify_all()

    def close(self, error: BaseException | None = None) -> None:
        with self._cond:
            self._closed = True
            self._error = error
            self._cond.notify_all()

    def wait_for_next(
        self,
        last_seq: int,
        timeout_s: float | None = None,
    ) -> tuple[int, int, float, np.ndarray, np.ndarray] | None:
        deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
        with self._cond:
            while True:
                if self._packet is not None and self._seq != last_seq:
                    return self._packet
                if self._closed:
                    if self._error is not None:
                        raise self._error
                    return None
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._cond.wait(remaining)


def stereo_capture_loop(
    left_cap: cv2.VideoCapture,
    right_cap: cv2.VideoCapture,
    loop_videos: bool,
    left_mode: str,
    right_mode: str,
    latest_frame: LatestStereoFrame,
) -> None:
    source_frame_index = 0
    try:
        while True:
            capture_t = time.monotonic()
            left_ok, left_frame = read_source_frame(left_cap, loop_videos, left_mode)
            right_ok, right_frame = read_source_frame(right_cap, loop_videos, right_mode)
            if not left_ok or left_frame is None:
                latest_frame.close(EOFError("Left source ended or failed"))
                return
            if not right_ok or right_frame is None:
                latest_frame.close(EOFError("Right source ended or failed"))
                return
            latest_frame.update(source_frame_index, capture_t, left_frame, right_frame)
            source_frame_index += 1
    except BaseException as exc:
        latest_frame.close(exc)


def configure_low_latency_capture(cap: cv2.VideoCapture) -> None:
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


class AprilTagDirectTracker:
    def __init__(
        self,
        apriltag_pose: ModuleType,
        args: argparse.Namespace,
        sample_left_frame: np.ndarray,
        sample_right_frame: np.ndarray,
    ) -> None:
        self.apr = apriltag_pose
        self.args = args

        k_l, d_l = self.apr.load_opencv_pinhole_yaml(args.left_yaml)
        k_r, d_r = self.apr.load_opencv_pinhole_yaml(args.right_yaml)

        h_l, w_l = sample_left_frame.shape[:2]
        h_r, w_r = sample_right_frame.shape[:2]
        self.map1_l, self.map2_l, self.new_k_l = self.apr.make_rectify_maps(
            w_l, h_l, k_l, d_l, alpha=args.rect_alpha
        )
        self.map1_r, self.map2_r, self.new_k_r = self.apr.make_rectify_maps(
            w_r, h_r, k_r, d_r, alpha=args.rect_alpha
        )

        self.cam_params_l = self.apr.camera_params_from_K(self.new_k_l)
        self.cam_params_r = self.apr.camera_params_from_K(self.new_k_r)
        self.detector = self.apr.Detector(
            families=args.tag_family,
            nthreads=args.threads,
            quad_decimate=args.decimate,
            quad_sigma=args.blur,
            refine_edges=args.refine_edges,
            decode_sharpening=0.25,
            debug=False,
        )

        self.last_used_pose_l = None
        self.last_used_r_l = None
        self.last_quat_l = None
        self.last_used_pose_r = None
        self.last_used_r_r = None
        self.last_quat_r = None

        self.kf_left = self.apr.PoseKalmanFilter()
        self.kf_right = self.apr.PoseKalmanFilter()
        self.last_kf_t_l = None
        self.last_kf_t_r = None
        self.last_fresh_t_l = None
        self.last_fresh_t_r = None

    def step(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        wall_time_s: float,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float]]:
        t0 = time.perf_counter()
        rect_l = cv2.remap(left_frame, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(right_frame, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        t1 = time.perf_counter()

        t_left0 = time.perf_counter()
        vis_l, state_l, st_l = self.apr.estimate_pose_one(
            rect_l,
            self.detector,
            self.cam_params_l,
            self.args.pose_tag_size_left,
            self.args.tag_id_left,
            self.new_k_l,
            self.args.max_rot_jump_deg,
            self.last_used_pose_l,
            self.last_used_r_l,
            self.last_quat_l,
        )
        self.last_used_pose_l, self.last_used_r_l, self.last_quat_l = st_l
        t_left1 = time.perf_counter()

        t_right0 = time.perf_counter()
        vis_r, state_r, st_r = self.apr.estimate_pose_one(
            rect_r,
            self.detector,
            self.cam_params_r,
            self.args.pose_tag_size_right,
            self.args.tag_id_right,
            self.new_k_r,
            self.args.max_rot_jump_deg,
            self.last_used_pose_r,
            self.last_used_r_r,
            self.last_quat_r,
        )
        self.last_used_pose_r, self.last_used_r_r, self.last_quat_r = st_r
        t_right1 = time.perf_counter()

        state_l = self.apr.remap_state_axes(state_l, "LEFT", enabled=(not self.args.disable_axis_remap))
        state_r = self.apr.remap_state_axes(state_r, "RIGHT", enabled=(not self.args.disable_axis_remap))

        if state_l["fresh_pose"]:
            self.last_fresh_t_l = wall_time_s
        if state_r["fresh_pose"]:
            self.last_fresh_t_r = wall_time_s

        state_l["age_s"] = finite_age_s(
            None if self.last_fresh_t_l is None else wall_time_s - self.last_fresh_t_l
        )
        state_r["age_s"] = finite_age_s(
            None if self.last_fresh_t_r is None else wall_time_s - self.last_fresh_t_r
        )
        state_l["fresh"] = bool(state_l["age_s"] <= self.args.max_cam_age_s)
        state_r["fresh"] = bool(state_r["age_s"] <= self.args.max_cam_age_s)

        dt_l = 0.0 if self.last_kf_t_l is None else max(1e-6, wall_time_s - self.last_kf_t_l)
        dt_r = 0.0 if self.last_kf_t_r is None else max(1e-6, wall_time_s - self.last_kf_t_r)
        self.kf_left.predict(dt_l)
        self.kf_right.predict(dt_r)
        self.last_kf_t_l = wall_time_s
        self.last_kf_t_r = wall_time_s

        if state_l["detected"]:
            self.kf_left.update(state_l["pos_m"], state_l["rpy_deg"])
        if state_r["detected"]:
            self.kf_right.update(state_r["pos_m"], state_r["rpy_deg"])

        filtered_pos_l, filtered_rpy_l = self.kf_left.get_pose()
        filtered_pos_r, filtered_rpy_r = self.kf_right.get_pose()
        state_l["filtered_pos_m"] = filtered_pos_l
        state_l["filtered_rpy_deg"] = filtered_rpy_l
        state_l["filtered"] = bool(self.kf_left.initialized)
        state_r["filtered_pos_m"] = filtered_pos_r
        state_r["filtered_rpy_deg"] = filtered_rpy_r
        state_r["filtered"] = bool(self.kf_right.initialized)

        filtered_fusion = self.apr.fuse_states(
            self.apr.build_filtered_state(state_l),
            self.apr.build_filtered_state(state_r),
            self.args.max_cam_age_s,
            self.args.max_fuse_pos_delta_m,
            self.args.max_fuse_yaw_delta_deg,
        )
        t2 = time.perf_counter()

        timings_ms = {
            "rectify_ms": 1000.0 * (t1 - t0),
            "apriltag_left_ms": 1000.0 * (t_left1 - t_left0),
            "apriltag_right_ms": 1000.0 * (t_right1 - t_right0),
            "kalman_fusion_ms": 1000.0 * (t2 - t_right1),
            "apriltag_total_ms": 1000.0 * (t2 - t0),
        }

        return state_l, state_r, filtered_fusion, {
            "left_vis": vis_l,
            "right_vis": vis_r,
            "left_rect": rect_l,
            "right_rect": rect_r,
        }, timings_ms


def overlay_debug(
    vis_l: np.ndarray,
    vis_r: np.ndarray,
    display_l: dict[str, Any],
    display_r: dict[str, Any],
    state_l: dict[str, Any],
    state_r: dict[str, Any],
    display_fusion: dict[str, Any],
    display_fusion_mode: str,
    ego_rotation_np: np.ndarray,
    ego_translation_np: np.ndarray,
) -> None:
    left_text = (
        f"L {display_l.get('display_source', 'raw')} det={int(state_l['detected'])} fresh={int(state_l['fresh'])} "
        f"id={state_l['tag_id']} "
        f"xyz=({display_l['pos_m'][0]:+.3f},{display_l['pos_m'][1]:+.3f},{display_l['pos_m'][2]:+.3f})"
    )
    right_text = (
        f"R {display_r.get('display_source', 'raw')} det={int(state_r['detected'])} fresh={int(state_r['fresh'])} "
        f"id={state_r['tag_id']} "
        f"xyz=({display_r['pos_m'][0]:+.3f},{display_r['pos_m'][1]:+.3f},{display_r['pos_m'][2]:+.3f})"
    )
    ego_text = (
        f"ego rot=({ego_rotation_np[0]:+.3f},{ego_rotation_np[1]:+.3f},{ego_rotation_np[2]:+.3f}) "
        f"trans=({ego_translation_np[0]:+.3f},{ego_translation_np[1]:+.3f},{ego_translation_np[2]:+.3f})"
    )

    cv2.putText(vis_l, left_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
    cv2.putText(vis_r, right_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
    cv2.putText(vis_l, ego_text, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 2)
    cv2.putText(vis_r, ego_text, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 2)

    if display_fusion.get("available", False):
        fusion_text = (
            f"FUSED {display_fusion_mode} src={display_fusion.get('source', 'none')} "
            f"xyz=({display_fusion['pos_m'][0]:+.3f},{display_fusion['pos_m'][1]:+.3f},{display_fusion['pos_m'][2]:+.3f}) "
            f"yaw={display_fusion['rpy_deg'][2]:+.1f}"
        )
        color = (0, 255, 0)
    else:
        fusion_text = "FUSED src=none"
        color = (0, 0, 255)

    cv2.putText(vis_l, fusion_text, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)
    cv2.putText(vis_r, fusion_text, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)


def build_state_message(
    state_l: dict[str, Any],
    state_r: dict[str, Any],
    filtered_fusion: dict[str, Any],
    ego_rotation_np: np.ndarray,
    ego_translation_np: np.ndarray,
    frame_index: int,
    source_time_s: float,
    dt_s: float,
) -> dict[str, Any]:
    left_pub_pos = state_l.get("filtered_pos_m", state_l["pos_m"]) if state_l.get("filtered", False) else state_l["pos_m"]
    left_pub_rpy = state_l.get("filtered_rpy_deg", state_l["rpy_deg"]) if state_l.get("filtered", False) else state_l["rpy_deg"]
    right_pub_pos = state_r.get("filtered_pos_m", state_r["pos_m"]) if state_r.get("filtered", False) else state_r["pos_m"]
    right_pub_rpy = state_r.get("filtered_rpy_deg", state_r["rpy_deg"]) if state_r.get("filtered", False) else state_r["rpy_deg"]

    msg: dict[str, Any] = {
        "t": float(time.time()),
        "source_frame_idx": int(frame_index),
        "source_timestamp_monotonic_s": float(source_time_s),
        "dt_s": float(dt_s),
        "ego_rotation_x": float(ego_rotation_np[0]),
        "ego_rotation_y": float(ego_rotation_np[1]),
        "ego_rotation_z": float(ego_rotation_np[2]),
        "ego_translation_x": float(ego_translation_np[0]),
        "ego_translation_y": float(ego_translation_np[1]),
        "ego_translation_z": float(ego_translation_np[2]),
        "states": {
            "left": {
                "tag_id": int(state_l["tag_id"]),
                "detected": int(state_l["detected"]),
                "used_hold": int(state_l["used_hold"]),
                "fresh": bool(state_l["fresh"]),
                "age_s": float(state_l["age_s"]),
                "pose_err": float(state_l["pose_err"]),
                "pos_m": [float(v) for v in left_pub_pos],
                "rpy_deg": [float(v) for v in left_pub_rpy],
                "center_px": [float(v) for v in state_l["center_px"]],
                "filtered": bool(state_l.get("filtered", False)),
            },
            "right": {
                "tag_id": int(state_r["tag_id"]),
                "detected": int(state_r["detected"]),
                "used_hold": int(state_r["used_hold"]),
                "fresh": bool(state_r["fresh"]),
                "age_s": float(state_r["age_s"]),
                "pose_err": float(state_r["pose_err"]),
                "pos_m": [float(v) for v in right_pub_pos],
                "rpy_deg": [float(v) for v in right_pub_rpy],
                "center_px": [float(v) for v in state_r["center_px"]],
                "filtered": bool(state_r.get("filtered", False)),
            },
        },
        "fusion": {
            "available": bool(filtered_fusion.get("available", False)),
            "source": str(filtered_fusion.get("source", "none")),
            "contributors": [str(v) for v in filtered_fusion.get("contributors", [])],
            "pos_m": [float(v) for v in filtered_fusion.get("pos_m", [0.0, 0.0, 0.0])],
            "rpy_deg": [float(v) for v in filtered_fusion.get("rpy_deg", [0.0, 0.0, 0.0])],
            "pos_delta_m": float(filtered_fusion.get("pos_delta_m", 0.0)),
            "yaw_delta_deg": float(filtered_fusion.get("yaw_delta_deg", 0.0)),
        },
    }

    if filtered_fusion.get("available", False):
        msg["pos_m"] = [float(v) for v in filtered_fusion["pos_m"]]
        msg["rpy_deg"] = [float(v) for v in filtered_fusion["rpy_deg"]]
        msg["state_source"] = str(filtered_fusion.get("source", "none"))

    return msg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open stereo videos/cameras directly, estimate optic flow and AprilTag pose in one loop, "
            "then publish filtered AprilTag pose plus ego-motion over UDP."
        )
    )
    parser.add_argument("--left-src", type=str, default="0", help="Left camera index or video path.")
    parser.add_argument("--right-src", type=str, default="1", help="Right camera index or video path.")
    parser.add_argument(
        "--source-mode",
        choices=("auto", "camera", "video"),
        default="auto",
        help="Interpret the sources as cameras, videos, or infer automatically.",
    )
    parser.add_argument("--loop-videos", action="store_true", help="Loop video files when they end.")
    parser.add_argument("--camera-width", type=int, default=None, help="Requested camera width.")
    parser.add_argument("--camera-height", type=int, default=None, help="Requested camera height.")
    parser.add_argument("--camera-fps", type=float, default=None, help="Requested camera FPS.")
    parser.add_argument(
        "--fixed-dt",
        type=float,
        default=None,
        help="Override frame dt in seconds. By default videos use FPS metadata and cameras use wall-clock time.",
    )
    parser.add_argument("--model", type=str, default="models/best_model/model.pt", help="Path to model.pt.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for optic flow model.")
    parser.add_argument(
        "--no-fast-model",
        action="store_true",
        help=(
            "Skip the CurrentModelFast inference wrapper and use the "
            "loaded CurrentModel directly. Default is to wrap, which "
            "applies fused per-neuron parameters, F.linear decoder, and "
            "the scatter sparse kernel on Linux ARM."
        ),
    )
    parser.add_argument(
        "--sparse-impl",
        choices=("auto", "csr", "coo", "scatter"),
        default="auto",
        help=(
            "CurrentModelFast synapse-matmul backend. 'auto' picks "
            "'scatter' on Linux ARM (Raspberry Pi / Jetson / Graviton) "
            "and 'csr' everywhere else. Override only for benchmarking."
        ),
    )
    parser.add_argument(
        "--fov-right-deg",
        type=float,
        nargs=2,
        default=DEFAULT_PHI_RIGHT_DEG,
        metavar=("START", "END"),
        help="Right camera phi limits in degrees.",
    )
    parser.add_argument(
        "--fov-left-deg",
        type=float,
        nargs=2,
        default=DEFAULT_PHI_LEFT_DEG,
        metavar=("START", "END"),
        help="Left camera phi limits in degrees.",
    )
    parser.add_argument(
        "--theta-deg",
        type=float,
        nargs=2,
        default=DEFAULT_THETA_DEG,
        metavar=("START", "END"),
        help="Shared theta limits in degrees.",
    )
    parser.add_argument("--filter-std", type=float, default=2.5, help="Gaussian pre-sampling blur std in degrees.")
    parser.add_argument("--pub-ip", type=str, default="127.0.0.1", help="UDP destination IP for controller state.")
    parser.add_argument("--pub-port", type=int, default=5005, help="UDP destination port for controller state.")
    parser.add_argument("--pub-hz", type=float, default=0.0, help="Publish rate limit. 0 publishes every frame.")
    parser.add_argument("--status-every", type=int, default=30, help="Print publish status every N processed frames.")
    parser.add_argument(
        "--profile-times",
        action="store_true",
        help="Print average per-stage processing times in the status output.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum frames to process.")
    parser.add_argument("--display", action="store_true", help="Show a local preview while processing.")
    parser.add_argument(
        "--low-latency-capture",
        action="store_true",
        help="Continuously drain live sources on a background thread and process only the newest stereo pair.",
    )
    parser.add_argument(
        "--apriltag-module-path",
        type=str,
        default=DEFAULT_APRILTAG_MODULE_PATH,
        help="Path to the existing apriltag_pose.py implementation to reuse.",
    )
    parser.add_argument("--left-yaml", type=str, default="pinhole_calib_left.yaml")
    parser.add_argument("--right-yaml", type=str, default="pinhole_calib_right.yaml")
    parser.add_argument("--rect-alpha", type=float, default=0.0)
    parser.add_argument("--tag-family", type=str, default="tag36h11")
    parser.add_argument("--tag-id-left", type=int, default=None)
    parser.add_argument("--tag-id-right", type=int, default=None)
    parser.add_argument("--pose-tag-size-left", type=float, default=0.15)
    parser.add_argument("--pose-tag-size-right", type=float, default=0.15)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--decimate", type=float, default=1.0)
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--refine-edges", action="store_true")
    parser.add_argument("--max-rot-jump-deg", type=float, default=45.0)
    parser.add_argument("--max-cam-age-s", type=float, default=0.25)
    parser.add_argument("--max-fuse-pos-delta-m", type=float, default=0.6)
    parser.add_argument("--max-fuse-yaw-delta-deg", type=float, default=45.0)
    parser.add_argument(
        "--draw-pose-source",
        choices=("raw", "filtered"),
        default="raw",
        help="Pose source used for projected axes in the preview window.",
    )
    parser.add_argument("--disable-axis-remap", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    apr = load_module_from_path("apriltag_pose_direct", args.apriltag_module_path)
    estimator = StereoOpticFlowEstimator(
        model_path=args.model,
        device=args.device,
        phi_right_deg=tuple(args.fov_right_deg),
        phi_left_deg=tuple(args.fov_left_deg),
        theta_deg=tuple(args.theta_deg),
        filter_std_deg=args.filter_std,
        use_fast_model=(not args.no_fast_model),
        sparse_impl=args.sparse_impl,
    )

    left_cap = None
    right_cap = None
    left_cap, left_mode, left_label = open_source(
        args.left_src,
        args.source_mode,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )
    right_cap, right_mode, right_label = open_source(
        args.right_src,
        args.source_mode,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )

    udp_sock = None
    latest_frame = None
    capture_thread = None
    try:
        dt_default = 1.0 / float(args.camera_fps) if args.camera_fps else float(estimator.model.dt)
        nominal_dt = get_nominal_dt(
            left_cap,
            right_cap,
            left_mode,
            right_mode,
            args.fixed_dt,
            dt_default,
        )

        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setblocking(False)
        pub_addr = (args.pub_ip, args.pub_port)
        pub_period = (1.0 / args.pub_hz) if args.pub_hz and args.pub_hz > 0.0 else 0.0
        last_pub_t = 0.0
        use_low_latency_capture = bool(args.low_latency_capture or left_mode == "camera" or right_mode == "camera")

        if use_low_latency_capture:
            configure_low_latency_capture(left_cap)
            configure_low_latency_capture(right_cap)
            latest_frame = LatestStereoFrame()
            capture_thread = threading.Thread(
                target=stereo_capture_loop,
                args=(left_cap, right_cap, args.loop_videos, left_mode, right_mode, latest_frame),
                daemon=True,
            )
            capture_thread.start()

        print(
            f"Processing stereo sources directly from L={left_label} ({left_mode}) "
            f"R={right_label} ({right_mode})"
        )
        print(f"Publishing filtered state UDP to udp://{args.pub_ip}:{args.pub_port}")
        if use_low_latency_capture:
            print("Low-latency capture enabled: processing only the newest stereo pair.")

        tracker = None
        frame_index = 0
        last_capture_t = None
        first_pub_t = None
        published_frames = 0
        last_packet_seq = 0
        dropped_pairs = 0
        timing_accum = {
            "optic_flow_ms": 0.0,
            "rectify_ms": 0.0,
            "apriltag_left_ms": 0.0,
            "apriltag_right_ms": 0.0,
            "kalman_fusion_ms": 0.0,
            "apriltag_total_ms": 0.0,
            "publish_ms": 0.0,
            "loop_ms": 0.0,
        }
        timing_count = 0

        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            loop_t0 = time.perf_counter()

            if use_low_latency_capture:
                packet = latest_frame.wait_for_next(last_packet_seq, timeout_s=1.0)
                if packet is None:
                    print("Source stream ended or timed out; stopping.")
                    break
                packet_seq, source_frame_index, capture_t, left_frame, right_frame = packet
                if packet_seq > (last_packet_seq + 1):
                    dropped_pairs += packet_seq - last_packet_seq - 1
                last_packet_seq = packet_seq
            else:
                capture_t = time.monotonic()
                left_ok, left_frame = read_source_frame(left_cap, args.loop_videos, left_mode)
                right_ok, right_frame = read_source_frame(right_cap, args.loop_videos, right_mode)
                if not left_ok or left_frame is None:
                    print("Left source ended or failed; stopping.")
                    break
                if not right_ok or right_frame is None:
                    print("Right source ended or failed; stopping.")
                    break
                source_frame_index = frame_index

            if tracker is None:
                tracker = AprilTagDirectTracker(apr, args, left_frame, right_frame)
                print(
                    "AprilTag tracker initialized "
                    f"with left cam_params={tracker.cam_params_l} right cam_params={tracker.cam_params_r}"
                )

            if nominal_dt is not None:
                dt_s = float(nominal_dt)
            elif last_capture_t is None:
                dt_s = dt_default
            else:
                dt_s = max(1e-4, capture_t - last_capture_t)
            last_capture_t = capture_t

            wall_time_s = time.time()
            flow_t0 = time.perf_counter()
            (
                ego_rotation_np,
                ego_translation_np,
                _right_avg_flow_np,
                _left_avg_flow_np,
            ) = estimator.step(left_frame, right_frame)
            flow_t1 = time.perf_counter()
            state_l, state_r, filtered_fusion, extras, april_timings_ms = tracker.step(left_frame, right_frame, wall_time_s)
            msg = build_state_message(
                state_l,
                state_r,
                filtered_fusion,
                ego_rotation_np,
                ego_translation_np,
                source_frame_index,
                capture_t,
                dt_s,
            )

            now = time.time()
            pub_t0 = time.perf_counter()
            if pub_period == 0.0 or (now - last_pub_t) >= pub_period:
                payload = (json.dumps(msg) + "\n").encode("utf-8")
                try:
                    udp_sock.sendto(payload, pub_addr)
                    if first_pub_t is None:
                        first_pub_t = time.monotonic()
                    published_frames += 1
                except BlockingIOError:
                    pass
                last_pub_t = now
            pub_t1 = time.perf_counter()

            timing_accum["optic_flow_ms"] += 1000.0 * (flow_t1 - flow_t0)
            timing_accum["rectify_ms"] += april_timings_ms["rectify_ms"]
            timing_accum["apriltag_left_ms"] += april_timings_ms["apriltag_left_ms"]
            timing_accum["apriltag_right_ms"] += april_timings_ms["apriltag_right_ms"]
            timing_accum["kalman_fusion_ms"] += april_timings_ms["kalman_fusion_ms"]
            timing_accum["apriltag_total_ms"] += april_timings_ms["apriltag_total_ms"]
            timing_accum["publish_ms"] += 1000.0 * (pub_t1 - pub_t0)
            timing_accum["loop_ms"] += 1000.0 * (time.perf_counter() - loop_t0)
            timing_count += 1

            if args.status_every > 0 and frame_index % args.status_every == 0:
                if published_frames >= 2 and first_pub_t is not None:
                    elapsed_s = max(1e-6, time.monotonic() - first_pub_t)
                    publish_hz = (published_frames - 1) / elapsed_s
                else:
                    publish_hz = 0.0
                status = (
                    f"frame {frame_index:05d} "
                    f"src={source_frame_index:05d} "
                    f"dt={dt_s:.4f}s "
                    f"publish_rate={publish_hz:.2f} Hz "
                    f"dropped={dropped_pairs} "
                    f"fusion_available={int(bool(filtered_fusion.get('available', False)))} "
                    f"fusion_source={filtered_fusion.get('source', 'none')}"
                )
                if args.profile_times and timing_count > 0:
                    avg = {key: value / timing_count for key, value in timing_accum.items()}
                    status += (
                        " | avg_ms "
                        f"flow={avg['optic_flow_ms']:.1f} "
                        f"rect={avg['rectify_ms']:.1f} "
                        f"tagL={avg['apriltag_left_ms']:.1f} "
                        f"tagR={avg['apriltag_right_ms']:.1f} "
                        f"kf={avg['kalman_fusion_ms']:.1f} "
                        f"tag_total={avg['apriltag_total_ms']:.1f} "
                        f"pub={avg['publish_ms']:.2f} "
                        f"loop={avg['loop_ms']:.1f}"
                    )
                    timing_accum = {key: 0.0 for key in timing_accum}
                    timing_count = 0
                print(status)

            if args.display:
                vis_l = extras["left_vis"].copy()
                vis_r = extras["right_vis"].copy()
                display_l = build_display_state(apr, state_l, args.draw_pose_source)
                display_r = build_display_state(apr, state_r, args.draw_pose_source)
                if args.draw_pose_source == "filtered" and filtered_fusion.get("available", False):
                    display_fusion = filtered_fusion
                    display_fusion_mode = "filtered"
                else:
                    display_fusion = apr.fuse_states(
                        state_l,
                        state_r,
                        args.max_cam_age_s,
                        args.max_fuse_pos_delta_m,
                        args.max_fuse_yaw_delta_deg,
                    )
                    display_fusion_mode = "raw"

                if hasattr(apr, "draw_state_axes"):
                    apr.draw_state_axes(vis_l, tracker.new_k_l, display_l, "LEFT", args.pose_tag_size_left)
                    apr.draw_state_axes(vis_r, tracker.new_k_r, display_r, "RIGHT", args.pose_tag_size_right)

                overlay_debug(
                    vis_l,
                    vis_r,
                    display_l,
                    display_r,
                    state_l,
                    state_r,
                    display_fusion,
                    display_fusion_mode,
                    ego_rotation_np,
                    ego_translation_np,
                )
                canvas = cv2.hconcat([vis_l, vis_r])
                cv2.imshow("direct_stereo_optic_flow_apriltag", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1

    finally:
        if udp_sock is not None:
            udp_sock.close()
        if left_cap is not None:
            left_cap.release()
        if right_cap is not None:
            right_cap.release()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())