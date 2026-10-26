#!/usr/bin/env python3
"""Stream optic-flow ego-motion estimates to the drone controller.

Reads a left and a right camera (or video), estimates translation from the
right eye only with the trained model (rotation is not estimated, so the
rotation and roll/pitch/yaw fields are always zero), integrates it into a
position and streams controller-compatible state over UDP. The left source is
still required but only feeds the ``--display`` preview and the video frame
rate.

The network advances one model step (``model.dt``) per frame, so the frame
interval should match the dt the model was trained at; a warning is printed
when it does not.

Usage:
    python -m drosophila_model.analysis.model_analysis.stream_optic_flow --model models/<model>/ --left-src 0 --right-src 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PHI_RIGHT_DEG = (-30.0, -150.0)
DEFAULT_PHI_LEFT_DEG = (150.0, 30.0)
DEFAULT_THETA_DEG = (45.0, 135.0)
# Relative frame-interval vs model.dt mismatch above which a warning is printed.
DT_MISMATCH_TOLERANCE = 0.1
# Number of wall-clock frame intervals whose median is checked against model.dt.
NUM_DT_CHECK_FRAMES = 10


def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.exists():
        return path
    raise FileNotFoundError(f"File not found: {path_str}")


def warn_if_dt_mismatch(frame_dt: float, model_dt: float, source: str) -> None:
    """Warn when frames arrive at a different interval than the model's dt.

    The network takes one ``model_dt`` step per frame, so the image motion per
    step is ``frame_dt / model_dt`` times what it saw in training.
    """
    if abs(frame_dt - model_dt) > DT_MISMATCH_TOLERANCE * model_dt:
        print(
            f"Warning: frame interval {frame_dt:.4f}s ({source}) differs from the "
            f"model dt {model_dt:.4f}s; the model sees {frame_dt / model_dt:.2f}x the "
            "image motion per step it was trained on.",
            file=sys.stderr,
        )


def robust_read(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    ok, frame = cap.read()
    if ok and frame is not None:
        return True, frame

    if cap.grab():
        ok, frame = cap.retrieve()
        if ok and frame is not None:
            return True, frame

    return False, None


def infer_source_mode(src: str, source_mode: str) -> str:
    if source_mode != "auto":
        return source_mode
    if src.isdigit():
        return "camera"
    if Path(src).exists():
        return "video"
    return "camera"


def open_source(
    src: str,
    source_mode: str,
    camera_width: int | None,
    camera_height: int | None,
    camera_fps: float | None,
) -> tuple[cv2.VideoCapture, str, str]:
    resolved_mode = infer_source_mode(src, source_mode)

    if resolved_mode == "video":
        path = resolve_existing_path(src)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {path}")
        return cap, resolved_mode, str(path)

    cap_src: Any = int(src) if str(src).isdigit() else src
    cap = cv2.VideoCapture(cap_src, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {src}")

    time.sleep(0.3)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if camera_width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_width))
    if camera_height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_height))
    if camera_fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(camera_fps))
    return cap, resolved_mode, str(src)


def read_source_frame(
    cap: cv2.VideoCapture,
    loop_videos: bool,
    source_mode: str,
) -> tuple[bool, np.ndarray | None]:
    ok, frame = robust_read(cap)
    if ok:
        return True, frame

    if loop_videos and source_mode == "video":
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return robust_read(cap)

    return False, None


def sample_image(image_gray: np.ndarray, W, in_fov: np.ndarray) -> torch.Tensor:
    flat = image_gray.ravel().astype(np.float32)
    sampled = np.asarray(W.dot(flat)).ravel() / 255.0
    sampled[~in_fov] = 0.5
    return torch.from_numpy(sampled).unsqueeze(0).unsqueeze(0)


def deg_pair_to_rad(values: tuple[float, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * np.pi / 180.0

def wrap_deg_180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = [float(a) for a in v]
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64) + skew(rotvec)

    axis = rotvec / angle
    K = skew(axis)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * K
        + (1.0 - math.cos(angle)) * (K @ K)
    )


def rotation_matrix_to_rpy_deg(R: np.ndarray) -> np.ndarray:
    sy = float(-R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cp = math.cos(pitch)

    if abs(cp) > 1e-8:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        yaw = 0.0

    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def render_debug_view(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    message: dict[str, Any],
) -> np.ndarray:
    left_vis = left_frame if left_frame.ndim == 3 else cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
    right_vis = right_frame if right_frame.ndim == 3 else cv2.cvtColor(right_frame, cv2.COLOR_GRAY2BGR)
    canvas = np.hstack([left_vis.copy(), right_vis.copy()])

    pos = message["pos_m"]
    rpy = message["rpy_deg"]
    rot = message["ego_rotation_deg_s"]
    trans = message["ego_translation_rate"]

    lines = [
        f"frame={message['frame_index']} dt={message['dt_s']:.4f}s",
        f"pos = [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]",
        f"rpy = [{rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f}] deg",
        f"rot = [{rot[0]:+.1f}, {rot[1]:+.1f}, {rot[2]:+.1f}] deg/s",
        f"vel = [{trans[0]:+.3f}, {trans[1]:+.3f}, {trans[2]:+.3f}]",
        "keys: q quit, r reset",
    ]

    for idx, text in enumerate(lines):
        y = 30 + idx * 28
        cv2.putText(
            canvas,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def build_message(
    frame_index: int,
    dt_s: float,
    pos_m: np.ndarray,
    rpy_deg: np.ndarray,
    ego_rotation_rad_s: np.ndarray,
    ego_translation_rate: np.ndarray,
    used_hold: int,
) -> dict[str, Any]:
    rot_deg_s = np.degrees(ego_rotation_rad_s)
    cam_state = {
        "source": "optic_flow",
        "pos_m": [float(v) for v in pos_m],
        "rpy_deg": [float(wrap_deg_180(v)) for v in rpy_deg],
        "age_s": 0.0,
        "pose_err": 0.0 if not used_hold else 1e9,
        "detected": int(not used_hold),
        "used_hold": int(used_hold),
    }

    return {
        "frame_index": int(frame_index),
        "timestamp_monotonic_s": float(time.monotonic()),
        "dt_s": float(dt_s),
        "state_source": "optic_flow",
        "pos_m": [float(v) for v in pos_m],
        "rpy_deg": [float(wrap_deg_180(v)) for v in rpy_deg],
        "ego_rotation_rad_s": [float(v) for v in ego_rotation_rad_s],
        "ego_rotation_deg_s": [float(v) for v in rot_deg_s],
        "ego_translation_rate": [float(v) for v in ego_translation_rate],
        "pose_err": 0.0 if not used_hold else 1e9,
        "detected": int(not used_hold),
        "used_hold": int(used_hold),
        "states": {"right": cam_state},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read left/right images from cameras or videos, estimate translation from the "
            "right eye with the trained optic-flow model, and stream controller-compatible "
            "state over UDP."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a trained model folder or model.pt file.",
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
    parser.add_argument(
        "--camera-fps", type=float, default=None,
        help="Requested camera FPS (should be 1 / model dt; the model takes one step per frame).",
    )
    parser.add_argument(
        "--fixed-dt",
        type=float,
        default=None,
        help="Override frame dt in seconds (used for pose integration; the model still takes "
             "one model-dt step per frame). By default videos use FPS metadata and cameras "
             "use wall-clock time.",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the model translation estimate before integration.",
    )
    parser.add_argument(
        "--integration-frame",
        choices=("world", "body"),
        default="world",
        help="Integrate translation in the world frame or directly in the body frame "
             "(identical while rotation is not estimated).",
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
        help="Left camera phi limits in degrees (unused: ego-motion is estimated from the "
             "right eye only).",
    )
    parser.add_argument(
        "--theta-deg",
        type=float,
        nargs=2,
        default=DEFAULT_THETA_DEG,
        metavar=("START", "END"),
        help="Shared theta limits in degrees.",
    )
    parser.add_argument(
        "--filter-std",
        type=float,
        default=2.5,
        help="Gaussian pre-sampling blur std in degrees.",
    )
    parser.add_argument("--udp-ip", type=str, default="127.0.0.1", help="Controller UDP host.")
    parser.add_argument("--udp-port", type=int, default=5005, help="Controller UDP port.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum frames to process.")
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional CSV path for timestamped pose logging.",
    )
    parser.add_argument("--display", action="store_true", help="Show a live preview window.")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace video playback using the source FPS so downstream consumers receive real-time-like updates.",
    )
    parser.add_argument("--print-hz", type=float, default=5.0, help="Console status print rate.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    global cv2, np, torch, nm, build_ommatidium_sampling_matrix
    sock = None
    log_file = None
    log_writer = None
    try:
        import cv2  # type: ignore[no-redef]
        import numpy as np  # type: ignore[no-redef]
        import torch  # type: ignore[no-redef]

        from drosophila_model.model_definition import neural_models as nm # type: ignore[no-redef]
        from drosophila_model.data_handling.sampling import build_ommatidium_sampling_matrix # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        print(
            "Missing runtime dependency while starting stream_optic_flow: "
            f"{exc}. Activate the project environment and try again.",
            file=sys.stderr,
        )
        return 1

    model_path = resolve_existing_path(args.model)
    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)
    model.eval()

    if args.log_path is not None:
        log_path = Path(args.log_path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "wall_time_s",
            "monotonic_time_s",
            "frame_index",
            "dt_s",
            "x_m",
            "y_m",
            "z_m",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        ])
        log_file.flush()
        print(f"Logging pose stream to {log_path}")

    phi_right_lim = deg_pair_to_rad(tuple(args.fov_right_deg))
    theta_lim = deg_pair_to_rad(tuple(args.theta_deg))

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

    try:
        left_ok, first_left = read_source_frame(left_cap, args.loop_videos, left_mode)
        right_ok, first_right = read_source_frame(right_cap, args.loop_videos, right_mode)
        if not left_ok or first_left is None:
            raise RuntimeError(f"Failed to read first frame from left source: {left_label}")
        if not right_ok or first_right is None:
            raise RuntimeError(f"Failed to read first frame from right source: {right_label}")

        first_right_gray = cv2.cvtColor(first_right, cv2.COLOR_BGR2GRAY) if first_right.ndim == 3 else first_right

        sample_phi_right = np.asarray(model.regions['optic_lobe'].column_phi_theta[:, 0], dtype=np.float64)
        sample_theta = np.asarray(model.regions['optic_lobe'].column_phi_theta[:, 1], dtype=np.float64)

        # Ego-motion is estimated from the right eye only, so only the right
        # image is sampled onto the model's columns.
        print(f"Building ommatidium sampler: right={first_right_gray.shape[::-1]}")
        W_right = build_ommatidium_sampling_matrix(
            first_right_gray.shape,
            phi_right_lim,
            theta_lim,
            sample_phi_right,
            sample_theta,
            filter_std=args.filter_std,
        )
        in_fov_right = np.asarray(W_right.sum(axis=1)).ravel() > 0
        # Right-eye only: oob mask is sized for a single eye's columns.
        oob_columns = torch.tensor(~in_fov_right, device=model.device)

        nominal_dt = None
        if args.fixed_dt is not None:
            nominal_dt = float(args.fixed_dt)
        elif left_mode == "video" and right_mode == "video":
            fps_values = [left_cap.get(cv2.CAP_PROP_FPS), right_cap.get(cv2.CAP_PROP_FPS)]
            fps_values = [fps for fps in fps_values if fps and fps > 1e-6]
            if fps_values:
                nominal_dt = 1.0 / min(fps_values)
        if nominal_dt is not None:
            warn_if_dt_mismatch(nominal_dt, model.dt,
                                "--fixed-dt" if args.fixed_dt is not None else "video FPS")
        measured_dts = []

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        frame_index = 0
        state_right = None
        filter_state_right = None
        cached_mappings = None

        R_world_from_body = np.eye(3, dtype=np.float64)
        pos_world = np.zeros(3, dtype=np.float64)
        last_capture_t = None
        next_deadline = time.monotonic()
        last_print_t = 0.0

        current_left = first_left
        current_right = first_right

        print(
            f"Streaming optic flow from L={left_label} ({left_mode}) "
            f"R={right_label} ({right_mode}) to {args.udp_ip}:{args.udp_port}"
        )

        while True:
            capture_t = time.monotonic()
            if frame_index == 0:
                left_frame = current_left
                right_frame = current_right
            else:
                left_ok, left_frame = read_source_frame(left_cap, args.loop_videos, left_mode)
                right_ok, right_frame = read_source_frame(right_cap, args.loop_videos, right_mode)
                if not left_ok or left_frame is None:
                    print("Left source ended or failed; stopping.")
                    break
                if not right_ok or right_frame is None:
                    print("Right source ended or failed; stopping.")
                    break

            if args.max_frames is not None and frame_index >= args.max_frames:
                break

            if nominal_dt is not None:
                dt_s = nominal_dt
            elif last_capture_t is None:
                dt_s = float(model.dt)
            else:
                dt_s = max(1e-4, capture_t - last_capture_t)
                if len(measured_dts) < NUM_DT_CHECK_FRAMES:
                    measured_dts.append(dt_s)
                    if len(measured_dts) == NUM_DT_CHECK_FRAMES:
                        warn_if_dt_mismatch(float(np.median(measured_dts)), model.dt,
                                            "measured wall-clock")
            last_capture_t = capture_t

            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY) if right_frame.ndim == 3 else right_frame

            right_stim = sample_image(right_gray, W_right, in_fov_right).to(
                device=model.device, dtype=model.dtype
            )

            with torch.no_grad():
                ego_rotation, ego_translation, right_result, _ = (
                    model.regions['optic_lobe'].estimate_ego_motion_from_image(
                        right_eye_inputs=right_stim,
                        initial_state_right=state_right,
                        initial_filter_state_right=filter_state_right,
                        oob_columns=oob_columns,
                        cached_mappings=cached_mappings,
                        eyes=('right',),
                        terms=('trans',),
                    )
                )

            if cached_mappings is None:
                cached_mappings = right_result["mappings"]

            state_right = right_result["model_state"]["v"]
            filter_state_right = right_result["model_state"]["filter_state"]

            # `eyes=('right',)` runs only the right eye; `terms=('trans',)` skips
            # rotation estimation, so ego_rotation is all-zeros by construction.
            # The solver returns rotation in the flow's units (deg/s); it is
            # integrated below as rad/s.
            omega_body = np.radians(
                ego_rotation[0, 0].detach().cpu().numpy().astype(np.float64))
            translation_body_raw = ego_translation[0, 0].detach().cpu().numpy().astype(np.float64)

            used_hold = 0
            if not (np.all(np.isfinite(omega_body)) and np.all(np.isfinite(translation_body_raw))):
                omega_body = np.zeros(3, dtype=np.float64)
                translation_body_raw = np.zeros(3, dtype=np.float64)
                used_hold = 1
            else:
                delta_R = rotation_matrix_from_rotvec(omega_body * dt_s)
                R_world_from_body = R_world_from_body @ delta_R
                translation_body = float(args.translation_scale) * translation_body_raw
                if args.integration_frame == "world":
                    translation_step = R_world_from_body @ translation_body
                else:
                    translation_step = translation_body
                pos_world = pos_world + translation_step * dt_s

            rpy_deg = rotation_matrix_to_rpy_deg(R_world_from_body)
            message = build_message(
                frame_index=frame_index,
                dt_s=dt_s,
                pos_m=pos_world,
                rpy_deg=rpy_deg,
                ego_rotation_rad_s=omega_body,
                ego_translation_rate=float(args.translation_scale) * translation_body_raw,
                used_hold=used_hold,
            )
            payload = (json.dumps(message) + "\n").encode("utf-8")
            sock.sendto(payload, (args.udp_ip, args.udp_port))

            if log_writer is not None and log_file is not None:
                wall_time_s = time.time()
                monotonic_time_s = message["timestamp_monotonic_s"]
                pos_vals = message["pos_m"]
                rpy_vals = message["rpy_deg"]
                log_writer.writerow([
                    f"{wall_time_s:.6f}",
                    f"{monotonic_time_s:.6f}",
                    int(frame_index),
                    f"{dt_s:.6f}",
                    f"{pos_vals[0]:.9f}",
                    f"{pos_vals[1]:.9f}",
                    f"{pos_vals[2]:.9f}",
                    f"{rpy_vals[0]:.9f}",
                    f"{rpy_vals[1]:.9f}",
                    f"{rpy_vals[2]:.9f}",
                ])
                log_file.flush()

            now = time.monotonic()
            print_period = 1.0 / max(args.print_hz, 1e-6)
            if now - last_print_t >= print_period:
                p = message["pos_m"]
                r = message["rpy_deg"]
                print(
                    f"[{frame_index:06d}] dt={dt_s:.4f}s "
                    f"pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
                    f"rpy=({r[0]:+.1f},{r[1]:+.1f},{r[2]:+.1f}) "
                    f"hold={used_hold}"
                )
                last_print_t = now

            if args.display:
                canvas = render_debug_view(left_frame, right_frame, message)
                cv2.imshow("stream_optic_flow", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    R_world_from_body = np.eye(3, dtype=np.float64)
                    pos_world = np.zeros(3, dtype=np.float64)
                    print("State reset to zero.")

            frame_index += 1

            if args.realtime and nominal_dt is not None:
                next_deadline += nominal_dt
                sleep_s = next_deadline - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    next_deadline = time.monotonic()

    finally:
        if sock is not None:
            sock.close()
        if log_file is not None:
            log_file.close()
        left_cap.release()
        right_cap.release()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
