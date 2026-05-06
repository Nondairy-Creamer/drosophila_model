from __future__ import annotations

import json
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

import neural_models as nm
from stimulus_utilities import build_ommatidium_sampling_matrix


DEFAULT_PHI_RIGHT_DEG = (-30.0, -150.0)
DEFAULT_PHI_LEFT_DEG = (150.0, 30.0)
DEFAULT_THETA_DEG = (45.0, 135.0)
HEADER_STRUCT = struct.Struct("!4sIII")
HEADER_MAGIC = b"SFS1"


def resolve_existing_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    here = Path(__file__).resolve().parent
    for candidate in (here / path, here.parent / path):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"File not found: {path}")


def robust_read(cap) -> tuple[bool, np.ndarray | None]:
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
    if src.startswith("picam:"):
        return "picamera2"
    if src.isdigit():
        return "camera"
    if Path(src).exists():
        return "video"
    return "camera"


class _Picamera2Capture:
    """``cv2.VideoCapture``-compatible adapter around picamera2.

    On Raspberry Pi OS Bookworm, CSI cameras don't enumerate as V4L2
    devices in a way OpenCV likes; the supported library is
    ``picamera2`` (which sits on top of ``libcamera``). This adapter
    exposes just enough of the ``cv2.VideoCapture`` API to let the
    rest of this module — ``read_source_frame``, ``stereo_capture_loop``,
    ``configure_low_latency_capture``, ``get_nominal_dt`` — keep
    working without per-backend branching.

    Frames are returned in BGR order (cv2-compatible) by using
    picamera2's quirky ``"RGB888"`` format string, which is documented
    to yield BGR ndarrays for OpenCV interop.

    ``set()`` is a no-op (size/fps are configured at construction
    time); ``get()`` answers FPS / width / height for the dt-inference
    paths and returns ``0`` for other props.

    Install on Pi (no uv): ::

        sudo apt install python3-picamera2 python3-libcamera
        python3 -m venv --system-site-packages .venv
        source .venv/bin/activate
        pip install -e .
    """

    def __init__(
        self,
        picamera2_cls,
        camera_num: int,
        width: int | None,
        height: int | None,
        fps: float | None,
    ) -> None:
        cam = picamera2_cls(camera_num=int(camera_num))
        size = (int(width or 640), int(height or 480))
        cfg = cam.create_video_configuration(
            main={"size": size, "format": "RGB888"},
        )
        if fps is not None and fps > 0.0:
            us = int(round(1_000_000.0 / float(fps)))
            controls = dict(cfg.get("controls", {}))
            controls["FrameDurationLimits"] = (us, us)
            cfg["controls"] = controls
        cam.configure(cfg)
        cam.start()
        self._cam = cam
        self._opened = True
        self._size = size
        self._fps = float(fps) if fps else 0.0
        self._camera_num = int(camera_num)

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened:
            return False, None
        try:
            arr = self._cam.capture_array("main")
        except Exception:
            return False, None
        if arr is None:
            return False, None
        return True, arr

    def grab(self) -> bool:
        return self._opened

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        return self.read()

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._size[1])
        return 0.0

    def set(self, prop: int, value: float) -> bool:  # noqa: ARG002 - cv2 API parity
        return False

    def release(self) -> None:
        if not self._opened:
            return
        try:
            self._cam.stop()
        except Exception:
            pass
        try:
            self._cam.close()
        except Exception:
            pass
        self._opened = False


def _open_picamera2_capture(
    camera_num: int,
    width: int | None,
    height: int | None,
    fps: float | None,
) -> _Picamera2Capture:
    try:
        # Lazy-imported because picamera2 is Pi-only and not in the
        # general dev environment. The type: ignore silences pyright's
        # "could not be resolved" on hosts without it installed.
        from picamera2 import Picamera2  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "picamera2 is required for source-mode 'picamera2' but is "
            "not importable. On Raspberry Pi OS Bookworm: "
            "`sudo apt install python3-picamera2 python3-libcamera`, "
            "then create your venv with system-site-packages access: "
            "`python3 -m venv --system-site-packages .venv`. "
            "(uv users: `uv venv --system-site-packages`.)"
        ) from exc
    return _Picamera2Capture(Picamera2, camera_num, width, height, fps)


def _parse_picam_camera_num(src: str) -> int:
    """Extract camera index from either a bare integer string or 'picam:N'."""
    s = str(src).strip()
    if s.startswith("picam:"):
        s = s.split(":", 1)[1]
    if not s.isdigit():
        raise ValueError(
            f"picamera2 source must be an integer or 'picam:N', got {src!r}"
        )
    return int(s)


def open_source(
    src: str,
    source_mode: str,
    camera_width: int | None,
    camera_height: int | None,
    camera_fps: float | None,
):
    """Open a stereo source. Returns ``(cap, mode, label)``.

    ``cap`` is either a real ``cv2.VideoCapture`` (for ``video`` /
    ``camera`` modes) or a :class:`_Picamera2Capture` adapter (for
    ``picamera2`` mode). Both expose the same surface that
    :func:`robust_read`, :func:`read_source_frame`, and
    :func:`stereo_capture_loop` need.
    """
    resolved_mode = infer_source_mode(src, source_mode)

    if resolved_mode == "video":
        path = resolve_existing_path(src)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {path}")
        return cap, resolved_mode, str(path)

    if resolved_mode == "picamera2":
        camera_num = _parse_picam_camera_num(src)
        cap = _open_picamera2_capture(
            camera_num, camera_width, camera_height, camera_fps,
        )
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open picamera2 source: picam:{camera_num}")
        return cap, resolved_mode, f"picam:{camera_num}"

    cap_src = int(src) if src.isdigit() else src
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
    cap,
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


def get_nominal_dt(
    left_cap,
    right_cap,
    left_mode: str,
    right_mode: str,
    fixed_dt: float | None,
    default_dt: float,
) -> float | None:
    if fixed_dt is not None:
        return float(fixed_dt)

    if left_mode == "video" and right_mode == "video":
        fps_values = [left_cap.get(cv2.CAP_PROP_FPS), right_cap.get(cv2.CAP_PROP_FPS)]
        fps_values = [fps for fps in fps_values if fps and fps > 1e-6]
        if fps_values:
            return 1.0 / min(fps_values)

    return default_dt if default_dt > 0.0 else None


def sample_image(image_gray: np.ndarray, sampler, in_fov: np.ndarray) -> torch.Tensor:
    flat = image_gray.astype(np.float32).ravel()
    sampled = np.asarray(sampler.dot(flat)).ravel() / 255.0
    sampled[~in_fov] = 0.5
    return torch.from_numpy(sampled).unsqueeze(0).unsqueeze(0)


def deg_pair_to_rad(values: tuple[float, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * np.pi / 180.0


def grayscale_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def encode_frame(frame: np.ndarray, jpeg_quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return encoded.tobytes()


def decode_frame(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode JPEG frame")
    return frame


def send_stereo_packet(
    sock: socket.socket,
    header: dict,
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    jpeg_quality: int = 90,
) -> None:
    header_bytes = json.dumps(header).encode("utf-8")
    left_bytes = encode_frame(left_frame, jpeg_quality)
    right_bytes = encode_frame(right_frame, jpeg_quality)
    sock.sendall(HEADER_STRUCT.pack(
        HEADER_MAGIC, len(header_bytes), len(left_bytes), len(right_bytes)
    ))
    sock.sendall(header_bytes)
    sock.sendall(left_bytes)
    sock.sendall(right_bytes)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("Socket closed while receiving frame packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_stereo_packet(sock: socket.socket) -> tuple[dict, np.ndarray, np.ndarray]:
    prefix = _recv_exact(sock, HEADER_STRUCT.size)
    magic, header_len, left_len, right_len = HEADER_STRUCT.unpack(prefix)
    if magic != HEADER_MAGIC:
        raise RuntimeError(f"Unexpected frame stream magic {magic!r}")
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    left_frame = decode_frame(_recv_exact(sock, left_len))
    right_frame = decode_frame(_recv_exact(sock, right_len))
    return header, left_frame, right_frame


def create_server_socket(host: str, port: int, backlog: int = 1) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    return sock


def connect_to_stream(host: str, port: int, timeout_s: float = 10.0) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    sock.connect((host, port))
    sock.settimeout(None)
    return sock


def make_pose_log_path(prefix: str, logs_dir: Path | None = None) -> Path:
    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"{prefix}_{timestamp}.csv"


class PoseIntegrator:
    def __init__(self) -> None:
        self.rotation_world_from_body = np.eye(3, dtype=np.float64)
        self.translation_world = np.zeros(3, dtype=np.float64)

    def step(
        self,
        rot_rate_deg_s: np.ndarray,
        trans_rate_body: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rot_step_rad = np.deg2rad(np.asarray(rot_rate_deg_s, dtype=np.float64)) * float(dt_s)
        delta_R, _ = cv2.Rodrigues(rot_step_rad)
        self.rotation_world_from_body = delta_R @ self.rotation_world_from_body
        trans_world = self.rotation_world_from_body @ np.asarray(trans_rate_body, dtype=np.float64)
        self.translation_world = self.translation_world + trans_world * float(dt_s)
        rot_pos_rad, _ = cv2.Rodrigues(self.rotation_world_from_body)
        rot_pos_deg = np.rad2deg(rot_pos_rad.reshape(3))
        return rot_pos_deg, self.translation_world.copy()


class StereoOpticFlowEstimator:
    def __init__(
        self,
        model_path: str | Path,
        device: str = 'cuda:0',
        phi_right_deg: tuple[float, float] = DEFAULT_PHI_RIGHT_DEG,
        phi_left_deg: tuple[float, float] = DEFAULT_PHI_LEFT_DEG,
        theta_deg: tuple[float, float] = DEFAULT_THETA_DEG,
        filter_std_deg: float = 2.5,
        use_fast_model: bool = True,
        sparse_impl: str = 'auto',
        pipelined: bool = False,
        queue_depth: int = 2,
        input_neurons: tuple[str, ...] = ('L1', 'L2', 'L3'),
        verbose: bool = True,
    ) -> None:
        """Streaming stereo optic-flow / ego-motion estimator.

        Args:
            model_path: Path to the trained ``model.pt`` (resolved against
                cwd, this file's dir, and its parent).
            device: Torch device for the model.
            phi_right_deg / phi_left_deg / theta_deg: Per-camera FOV
                limits in degrees, used to build the sampling matrices
                that map raw pixels to hex-grid columns.
            filter_std_deg: Gaussian pre-sampling blur (deg) for the
                ommatidium sampler.
            use_fast_model: When ``True`` and the loaded model is a plain
                :class:`CurrentModel`, wrap it in
                :class:`CurrentModelFast` for inference (precomputed
                fused parameters, BLAS-friendly decoder, scatter sparse
                kernel on Linux ARM, etc). ``False`` keeps the loaded
                model as-is — useful for A/B comparisons against the
                slower path. Models that aren't ``CurrentModel`` (e.g.
                ``HandTunedModel``, ``ConductanceModel``) are always
                left as-is regardless of this flag.
            sparse_impl: Forwarded to ``CurrentModelFast.from_trained``
                when wrapping. ``'auto'`` picks the best per-host
                backend (``'csr'`` on x86 / Apple Silicon, ``'scatter'``
                on Linux ARM / Raspberry Pi).
            pipelined: When ``True``, route ``step()`` through a
                fully asynchronous three-thread pipeline:

                * **Main thread** (``step()``): camera I/O, image
                  sampling, mailbox push, latest-result drain. Never
                  blocks on the model.
                * **Stage-1 worker**: recurrent dynamics. Always
                  processes the freshest frame in its single-slot
                  mailbox; older frames that haven't been picked up
                  are silently overwritten ("drop oldest").
                * **Stage-2 worker** (inside
                  :class:`PipelinedEgoMotionEstimator`): flow readout
                  + ego-motion solve. Runs in parallel with stage 1 of
                  the next frame.

                Each thread runs as soon as its inputs are available,
                so a slow stage 2 never holds up stage 1 (and vice
                versa). ``step()`` returns the most recently completed
                ego-motion result; if the model is too slow to finish
                a frame within the inter-frame interval, the returned
                estimate is based on a slightly older frame, but the
                main loop's cadence is unaffected. Per-eye mean flow
                is not surfaced in this mode (downstream callers in
                the realtime loop ignore it) and is returned as zeros.
            queue_depth: Forwarded to ``PipelinedEgoMotionEstimator``
                (the firing-rates handoff between stage 1 and stage 2).
                Default ``2`` absorbs minor stage-2 jitter without
                back-pressuring stage 1.
            input_neurons: Stimulus injection cell types passed to the
                pipelined estimator's ``reset(input_neurons=...)`` on
                the first frame.  Matches the default the model uses
                in the serial path.
            verbose: Print the chosen backend at construction time.
        """
        self.model_path = resolve_existing_path(model_path)
        loaded = nm.NeuralModel.load(self.model_path, device=device)
        loaded.eval()

        self._wrapped_fast = False
        is_plain_current = (
            isinstance(loaded, nm.CurrentModel)
            and not isinstance(loaded, nm.CurrentModelFast)
        )
        if use_fast_model and is_plain_current:
            self.model = nm.CurrentModelFast.from_trained(
                loaded, sparse_impl=sparse_impl,
            )
            self.model.eval()
            self._wrapped_fast = True
            del loaded
            if verbose:
                print(
                    f"[StereoOpticFlowEstimator] wrapped model in "
                    f"CurrentModelFast(sparse_impl={self.model.sparse_impl!r})"
                )
        else:
            self.model = loaded
            if verbose:
                why = (
                    "(opted out via use_fast_model=False)"
                    if (not use_fast_model and is_plain_current)
                    else f"(model class {type(loaded).__name__} not eligible)"
                )
                print(
                    f"[StereoOpticFlowEstimator] using loaded model "
                    f"{type(loaded).__name__} as-is {why}"
                )

        self.phi_right_deg = tuple(phi_right_deg)
        self.phi_left_deg = tuple(phi_left_deg)
        self.theta_deg = tuple(theta_deg)
        self.filter_std_deg = float(filter_std_deg)

        self._input_neurons = tuple(input_neurons)
        self._pipelined_mode = bool(pipelined)
        self._queue_depth = int(queue_depth)
        self._pipelined_est: nm.PipelinedEgoMotionEstimator | None = None
        self._pipelined_started = False

        # Streaming-mode plumbing: stage 1 runs on its own worker thread
        # so step() never blocks on the model. The main thread shoves
        # each freshly sampled (right, left) pair into a single-slot
        # mailbox (drop-oldest); the stage-1 worker pops it, runs the
        # recurrent step, and hands firing rates to the existing stage-2
        # worker inside PipelinedEgoMotionEstimator. Completed
        # ego-motion results are drained out of the stage-2 output queue
        # by the main thread and cached as ``_latest_result``.
        self._stage1_in: queue.Queue | None = None
        self._stage1_thread: threading.Thread | None = None
        self._stage1_stop_evt: threading.Event | None = None
        self._latest_result_lock = threading.Lock()
        self._latest_result: tuple[torch.Tensor, torch.Tensor] | None = None
        self._n_results_seen = 0

        if self._pipelined_mode:
            try:
                self._pipelined_est = nm.PipelinedEgoMotionEstimator(
                    self.model, queue_depth=self._queue_depth,
                )
            except (AttributeError, ValueError) as exc:
                # Model has no flow readout (e.g. an older config) —
                # fall back to serial rather than crashing the runtime.
                if verbose:
                    print(
                        f"[StereoOpticFlowEstimator] pipelined mode "
                        f"unavailable ({exc!r}); falling back to serial."
                    )
                self._pipelined_mode = False
                self._pipelined_est = None
            else:
                if verbose:
                    print(
                        f"[StereoOpticFlowEstimator] pipelined=True "
                        f"(queue_depth={self._queue_depth}, async stage-1+stage-2 "
                        f"workers; step() is non-blocking, returns latest result)"
                    )

        # Recurrent state owned by the serial path. The pipelined estimator
        # owns its own copy of these so the four ``initial_*`` slots stay
        # ``None`` when ``self._pipelined_mode`` is True.
        self.initial_state_right = None
        self.initial_state_left = None
        self.initial_filter_state_right = None
        self.initial_filter_state_left = None
        self.cached_mappings = None
        self._initialized = False
        self._W_right = None
        self._W_left = None
        self._in_fov_right = None
        self._in_fov_left = None
        self._oob_columns = None

    def initialize(self, left_frame: np.ndarray, right_frame: np.ndarray) -> None:
        left_gray = grayscale_frame(left_frame)
        right_gray = grayscale_frame(right_frame)

        sample_phi_right = np.asarray(self.model.column_phi_theta[:, 0], dtype=np.float64)
        sample_theta = np.asarray(self.model.column_phi_theta[:, 1], dtype=np.float64)
        sample_phi_left = -sample_phi_right

        self._W_right = build_ommatidium_sampling_matrix(
            right_gray.shape,
            deg_pair_to_rad(self.phi_right_deg),
            deg_pair_to_rad(self.theta_deg),
            sample_phi_right,
            sample_theta,
            filter_std=self.filter_std_deg,
        )
        self._W_left = build_ommatidium_sampling_matrix(
            left_gray.shape,
            deg_pair_to_rad(self.phi_left_deg),
            deg_pair_to_rad(self.theta_deg),
            sample_phi_left,
            sample_theta,
            filter_std=self.filter_std_deg,
        )

        self._in_fov_right = np.asarray(self._W_right.sum(axis=1)).ravel() > 0
        self._in_fov_left = np.asarray(self._W_left.sum(axis=1)).ravel() > 0
        self._oob_columns = torch.tensor(
            np.concatenate([~self._in_fov_right, ~self._in_fov_left]),
            device=self.model.device,
        )

        if self._pipelined_mode and self._pipelined_est is not None:
            self._pipelined_est.start()
            self._pipelined_est.reset(
                S=1,
                input_neurons=list(self._input_neurons),
                oob_columns=self._oob_columns,
            )
            self._pipelined_started = True

            # Single-slot drop-oldest mailbox between main thread and
            # stage-1 worker. ``maxsize=1`` means the worker always
            # picks up the freshest input; older frames that haven't
            # been picked up are silently overwritten.
            self._stage1_in = queue.Queue(maxsize=1)
            self._stage1_stop_evt = threading.Event()
            with self._latest_result_lock:
                self._latest_result = None
            self._n_results_seen = 0
            self._stage1_thread = threading.Thread(
                target=self._stage1_loop,
                name='stereo-stage1',
                daemon=True,
            )
            self._stage1_thread.start()

        self._initialized = True

    def step(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process one stereo pair, return ego-motion + per-eye mean flow.

        In the default serial mode, the returned values are for the
        frame just submitted.  In ``pipelined=True`` mode, the very
        first call still returns its own result (paying full latency
        once), and every subsequent call returns the *previous* frame's
        ego-motion while stage 1 of the current frame runs inline.
        Per-eye mean flow is zero-filled in the pipelined path because
        the threaded worker doesn't surface it.

        Args:
            left_frame: BGR or grayscale uint8 image, left camera.
            right_frame: BGR or grayscale uint8 image, right camera.

        Returns:
            ``(ego_rotation_np, ego_translation_np,
              right_avg_flow_np, left_avg_flow_np)``.  Rotation /
            translation are ``(3,)`` float64 arrays; per-eye mean flow
            is ``(2,)`` float64 (``[d_right, d_up]``) in the serial
            path and zeros in the pipelined path.
        """
        if not self._initialized:
            self.initialize(left_frame, right_frame)

        left_gray = grayscale_frame(left_frame)
        right_gray = grayscale_frame(right_frame)

        visual_stimulus_right = sample_image(right_gray, self._W_right, self._in_fov_right).to(
            device=self.model.device, dtype=self.model.dtype
        )
        visual_stimulus_left = sample_image(left_gray, self._W_left, self._in_fov_left).to(
            device=self.model.device, dtype=self.model.dtype
        )

        if self._pipelined_mode and self._pipelined_est is not None:
            return self._step_pipelined(visual_stimulus_right, visual_stimulus_left)
        return self._step_serial(visual_stimulus_right, visual_stimulus_left)

    def _step_serial(
        self,
        visual_stimulus_right: torch.Tensor,
        visual_stimulus_left: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            ego_rotation, ego_translation, right_res, left_res = self.model.estimate_ego_motion_from_image(
                visual_stimulus_right,
                visual_stimulus_left,
                initial_state_right=self.initial_state_right,
                initial_state_left=self.initial_state_left,
                initial_filter_state_right=self.initial_filter_state_right,
                initial_filter_state_left=self.initial_filter_state_left,
                oob_columns=self._oob_columns,
                cached_mappings=self.cached_mappings,
            )

        if self.cached_mappings is None:
            self.cached_mappings = right_res["mappings"]

        self.initial_state_right = right_res["model_state"]["v"]
        self.initial_state_left = left_res["model_state"]["v"]
        self.initial_filter_state_right = right_res["model_state"]["filter_state"]
        self.initial_filter_state_left = left_res["model_state"]["filter_state"]

        flow_key = next(
            (
                key
                for key, readout in self.model._readout_modules.items()
                if getattr(readout, "predicts_flow", False) and key in right_res["outputs"]
            ),
            None,
        )
        if flow_key is None:
            raise ValueError("No flow readout configured; cannot compute per-eye average flow.")

        ego_rotation_np = ego_rotation[0, 0].detach().cpu().numpy().astype(np.float64)
        ego_translation_np = ego_translation[0, 0].detach().cpu().numpy().astype(np.float64)
        right_avg_flow_np = torch.nanmean(right_res["outputs"][flow_key][0, 0], dim=0)
        left_avg_flow_np = torch.nanmean(left_res["outputs"][flow_key][0, 0], dim=0)
        right_avg_flow_np = right_avg_flow_np.detach().cpu().numpy().astype(np.float64)
        left_avg_flow_np = left_avg_flow_np.detach().cpu().numpy().astype(np.float64)
        return ego_rotation_np, ego_translation_np, right_avg_flow_np, left_avg_flow_np

    def _step_pipelined(
        self,
        visual_stimulus_right: torch.Tensor,
        visual_stimulus_left: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Async streaming step: non-blocking, returns the latest finished result.

        Drops the freshly sampled stereo pair into the stage-1
        worker's single-slot mailbox (overwriting any input the worker
        hasn't grabbed yet), drains all completed ego-motion results
        out of the stage-2 output queue into ``_latest_result``, and
        returns whatever result was most recently produced.

        Each thread runs as soon as its inputs are ready:

        * **Main thread** (this method): camera I/O, image sampling,
          mailbox push, result drain, return. Wall-clock dominated by
          ``cv2`` operations; the model contributes nothing here.
        * **Stage-1 worker** (``_stage1_loop``): pulls the latest
          mailbox entry, runs ``input_conv`` → forward → ``firing_rate``,
          hands firing rates to stage 2.
        * **Stage-2 worker** (inside :class:`PipelinedEgoMotionEstimator`):
          pulls firing rates, runs flow readout + ego-motion solve.

        If stage 1 is mid-step when a new frame arrives, the new frame
        sits in the mailbox and is picked up the moment stage 1
        finishes — no synchronization with stage 2 is required because
        firing rates flow through a separate queue.

        Returns:
            ``(rot, trans, zeros2, zeros2)``. The per-eye mean flow
            slots are zero-filled because the threaded path doesn't
            surface them (and downstream callers ignore them). When
            no result has finished yet (the very first few ms after
            startup), zeros are returned.
        """
        assert self._stage1_in is not None
        assert self._pipelined_est is not None

        # Mailbox push: drop the previous unread entry so stage 1 always
        # sees the newest inputs. Single-producer (this thread) so the
        # try/except dance is race-free.
        item = (visual_stimulus_right, visual_stimulus_left)
        try:
            self._stage1_in.put_nowait(item)
        except queue.Full:
            try:
                self._stage1_in.get_nowait()
            except queue.Empty:
                pass
            try:
                self._stage1_in.put_nowait(item)
            except queue.Full:
                pass

        # Drain everything stage 2 has finished since last step(); keep
        # only the most recent (rot, trans). Non-blocking.
        while True:
            try:
                rot, trans = self._pipelined_est.get(timeout=0.0)
            except queue.Empty:
                break
            with self._latest_result_lock:
                self._latest_result = (rot, trans)
                self._n_results_seen += 1

        zeros2 = np.zeros(2, dtype=np.float64)
        with self._latest_result_lock:
            latest = self._latest_result
        if latest is None:
            return (
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                zeros2,
                zeros2,
            )
        rot, trans = latest
        rot_np = rot[0, 0].detach().cpu().numpy().astype(np.float64)
        trans_np = trans[0, 0].detach().cpu().numpy().astype(np.float64)
        return rot_np, trans_np, zeros2, zeros2

    def _stage1_loop(self) -> None:
        """Worker: pop the latest mailbox entry and submit it to stage 2.

        Sleeping on ``self._stage1_in.get(timeout=...)`` lets us still
        notice ``_stage1_stop_evt`` if the camera stops producing.
        """
        assert self._stage1_in is not None
        assert self._stage1_stop_evt is not None
        assert self._pipelined_est is not None
        try:
            while not self._stage1_stop_evt.is_set():
                try:
                    item = self._stage1_in.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:  # shutdown sentinel
                    return
                visual_right, visual_left = item
                # PipelinedEgoMotionEstimator.submit() runs stage 1
                # inline (on this thread) and hands fr to stage 2.
                # Backpressure from stage 2 is absorbed by the queue
                # depth in the underlying estimator.
                self._pipelined_est.submit(visual_right, visual_left)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[StereoOpticFlowEstimator] stage-1 worker died: {exc!r}")
            raise

    def close(self) -> None:
        """Stop both workers and drain queues (idempotent)."""
        # Tell stage 1 to exit and unblock its mailbox wait.
        if self._stage1_stop_evt is not None:
            self._stage1_stop_evt.set()
        if self._stage1_in is not None:
            try:
                self._stage1_in.put_nowait(None)
            except queue.Full:
                try:
                    self._stage1_in.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._stage1_in.put_nowait(None)
                except queue.Full:
                    pass
        if self._stage1_thread is not None:
            self._stage1_thread.join(timeout=2.0)
            self._stage1_thread = None
        # Now stop the underlying stage-2 worker.
        if self._pipelined_est is not None:
            self._pipelined_est.stop()
            self._pipelined_started = False

    def __enter__(self) -> "StereoOpticFlowEstimator":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def save_integrated_pose_figure(
    rot_history: np.ndarray,
    trans_history: np.ndarray,
    dt_history_s: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    integrator = PoseIntegrator()
    rot_pos = np.zeros_like(rot_history, dtype=np.float64)
    trans_pos = np.zeros_like(trans_history, dtype=np.float64)

    for index, dt_s in enumerate(dt_history_s):
        rot_pos[index], trans_pos[index] = integrator.step(
            rot_history[index], trans_history[index], float(dt_s)
        )

    times = np.concatenate([[0.0], np.cumsum(dt_history_s[:-1])])
    rot_labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    trans_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    for axis_index in range(3):
        ax_left = axes[axis_index, 0]
        ax_left.plot(times, rot_pos[:, axis_index], linewidth=1.2)
        ax_left.set_ylabel(f"{rot_labels[axis_index]} (deg)")
        ax_left.grid(True, alpha=0.3)
        if axis_index == 0:
            ax_left.set_title("Integrated Rotation")

        ax_right = axes[axis_index, 1]
        ax_right.plot(times, trans_pos[:, axis_index], linewidth=1.2)
        ax_right.set_ylabel(f"{trans_labels[axis_index]} (m)")
        ax_right.grid(True, alpha=0.3)
        if axis_index == 0:
            ax_right.set_title("Integrated Translation")

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.suptitle("Predicted Integrated Pose (streaming)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
