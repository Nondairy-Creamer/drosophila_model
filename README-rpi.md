# Real-time streaming on a Raspberry Pi 5

End-to-end setup notes for running the optic flow / ego-motion pipeline live on a Raspberry Pi 5 with two CSI cameras, publishing ego-motion estimates over UDP at 30 Hz.

This is a separate doc from the main `README.md` because the Pi side has its own dependency story (apt-installed libcamera + system-site-packages venv, no CUDA, no uv) and a few one-time gotchas that are easy to lose track of.

## Verified configuration

| Component | Tested with |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1 |
| OS | Raspberry Pi OS Trixie (Debian 13), Python 3.11 |
| Cameras | 2 × Raspberry Pi HQ Camera (Sony IMX477), one on each CSI port |
| `libcamera` | `0.5.2+rpt20250903-1` (see [Troubleshooting](#troubleshooting-libcameralibpisp-abi-break) — newer 0.7 builds are currently broken) |
| `libpisp` | `1.4.0-1` |
| Python torch | CPU-only, default PyPI aarch64 wheel |

Ballpark steady-state cost per frame at 30 Hz, 640 × 480, no AprilTag:

| Stage | Where it runs | Wall-clock per frame |
|---|---|---|
| Camera capture + cv2 grayscale + ommatidium sampling | Main thread (in `step()`) | ~20 ms |
| Recurrent dynamics (stage 1) | Stage-1 worker thread | ~25 ms |
| Decoder + ego-motion solve (stage 2) | Stage-2 worker thread | ~7 ms |
| **Main-loop wall-clock** | | **~33 ms (= 30 Hz)** |

Because stages 1 and 2 run on dedicated threads, the main loop's per-frame cost is bounded by image sampling, not by the model. Slowing the model further still keeps the main loop at 30 Hz; the only effect is that the published ego-motion is based on a slightly older frame.

## Hardware setup

1. Plug each Pi camera ribbon into one of the Pi 5's two CSI ports (label-side towards the SD card slot on both the Pi and the camera; the connectors are slightly different on Pi 5 — use a Pi 5 ribbon).
2. Boot once and run `sudo raspi-config` → `Interface Options` → enable cameras if you haven't already.
3. Confirm both cameras are detected at the kernel level:

   ```bash
   ls -l /dev/video*           # should show several entries
   v4l2-ctl --list-devices     # both rp1-cfe entries should appear
   ```

If only one camera shows up, check the `dtoverlay=imx477` lines in `/boot/firmware/config.txt` — Pi 5 needs an explicit overlay for each camera (`,cam0` / `,cam1`).

## System packages (apt)

```bash
sudo apt update
sudo apt install python3-picamera2 python3-libcamera \
                 python3-venv python3-dev build-essential \
                 libopenblas-dev
```

`python3-picamera2` and `python3-libcamera` ship the libcamera C++ library + Python bindings via apt only — they cannot be installed via pip because they wrap C bindings to libcamera/libpisp. That's why the venv (next step) needs `--system-site-packages`.

If `import libcamera` fails with `undefined symbol: ...compute_optimal_stride...`, your system has the upstream Trixie libcamera/libpisp ABI break described below — jump to [Troubleshooting](#troubleshooting-libcameralibpisp-abi-break) before the venv step.

## Python environment (plain venv, no uv)

uv on the Pi is more friction than it's worth (libcamera/libpisp need to live in the venv as system-site-packages, which is not uv's happy path). Use a plain venv:

```bash
cd ~/code/drosophila_model
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rpi.txt
```

`requirements-rpi.txt` is the slim CPU-only dep list — `torch` (CPU aarch64 wheels from default PyPI), `numpy`, `scipy`, `h5py`, `opencv-python`. None of the analysis / training / dev deps in the main `requirements.txt`. None of the CUDA index-URL pinning.

Sanity check that the venv sees both pip-installed and apt-installed packages:

```bash
python -c "import torch, numpy, scipy, cv2, h5py; print('pip side OK')"
python -c "from picamera2 import Picamera2; print('apt side OK:', Picamera2.global_camera_info())"
```

The second line should print a list with two camera dicts (one per CSI port). If it prints `[]`, the kernel saw the cameras but libcamera didn't — usually a `dtoverlay` issue in `/boot/firmware/config.txt`.

## Run the streamer

```bash
cd ~/code/drosophila_model
source .venv/bin/activate

python analysis/realtime_stream.py \
  --model models/model_with_inputs/model.pt \
  --pipelined --no-apriltag \
  --source-mode picamera2 --left-src 0 --right-src 1 \
  --device cpu \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --status-every 30 --profile-times
```

Flag-by-flag:

| Flag | What it does |
|---|---|
| `--pipelined` | Use the 3-thread async pipeline (main / stage-1 / stage-2). Required to hit 30 Hz on this hardware. |
| `--no-apriltag` | Skip AprilTag pose tracking + Kalman fusion. Disable until you actually need tag-based ground truth. (Also auto-disabled if `apriltag_pose.py` is not on disk.) |
| `--source-mode picamera2` | Use the picamera2 / libcamera CSI backend. Without this, the script tries `cv2.VideoCapture` on `/dev/video0` etc., which doesn't work for Pi CSI cameras. |
| `--left-src 0 --right-src 1` | Camera indices in `Picamera2.global_camera_info()` order. You can also pass `--left-src picam:0 --right-src picam:1` and skip `--source-mode`. |
| `--device cpu` | No CUDA on Pi. |
| `--camera-width / --camera-height / --camera-fps` | Limits picamera2's output size. The IMX477 is 12 MP native; 640 × 480 is plenty for the ommatidium sampler and avoids burning 99 % of pixels. |
| `--status-every 30` | Print one status line every 30 frames (~1 s at 30 Hz). |
| `--profile-times` | Include the per-stage `flow=`, `pub=`, `loop=` ms breakdown in the status line. |

What you should see once it stabilizes (≈1 s after start):

```text
frame 00030 src=00030 dt=0.0333s publish_rate=30.0 Hz dropped=0 ... | avg_ms flow=20.5 pub=0.22 loop=33.2
frame 00060 src=00060 dt=0.0333s publish_rate=30.0 Hz dropped=0 ... | avg_ms flow=21.1 pub=0.22 loop=33.4
```

`loop` ≈ `1000 / camera_fps` and `dropped == 0` is the success signal. The first frame will show `flow=1500ms` or so — that's the one-time init cost (sparse sampling matrix build + first model warmup) and is normal.

## What's actually publishing

UDP packets go to `127.0.0.1:5005` by default (override with `--pub-ip` / `--pub-port`). Each packet is a newline-terminated JSON object with at least:

```json
{
  "t": 1738358400.123,
  "source_frame_idx": 90,
  "dt_s": 0.0333,
  "ego_rotation_x": -0.04, "ego_rotation_y": 0.12, "ego_rotation_z": -0.01,
  "ego_translation_x": 0.001, "ego_translation_y": 0.0, "ego_translation_z": -0.02,
  "states": { "left": {...}, "right": {...} },
  "fusion": { "available": false, "source": "none", ... }
}
```

When AprilTag is disabled, the `states.{left,right}` and `fusion` blocks are present but zero-filled and flagged `available=false`. Downstream consumers don't need to special-case the no-tag mode.

## Architecture overview

The async pipeline is implemented in two files:

| File | Role |
|---|---|
| `neural_models.py` :: `CurrentModelFast` | Inference-optimized subclass of `CurrentModel` — precomputes fused per-neuron parameters, BLAS-friendly decoder, picks the right sparse-mm backend at runtime (`scatter` on Linux ARM, `csr` elsewhere). |
| `neural_models.py` :: `PipelinedEgoMotionEstimator` | Stage-2 worker thread (decoder + ego-motion solve). Owns recurrent state. |
| `analysis/stereo_stream_runtime.py` :: `StereoOpticFlowEstimator` | Owns the model + sampling matrices. In `--pipelined` mode it adds a stage-1 worker thread and a single-slot drop-oldest mailbox between the main loop and stage 1. |
| `analysis/realtime_stream.py` | Main loop: read cameras → call `estimator.step()` → publish UDP. Holds 30 Hz cadence regardless of model timing. |

The mailbox between main and stage 1 has `maxsize=1` and is drop-oldest, so if the model temporarily falls behind (e.g. an apt cron job nicks 20 ms of CPU), the stage-1 worker simply skips ahead to the freshest frame the moment it's free again. Same idea on the stage-1 → stage-2 handoff but with `queue_depth=2` to absorb minor stage-2 jitter.

## Troubleshooting: libcamera/libpisp ABI break

**Symptom**

```text
ImportError: /lib/aarch64-linux-gnu/libcamera.so.0.7: undefined symbol: _ZN7libpisp22compute_optimal_strideER24pisp_image_format_config
```

`rpicam-hello` fails identically. This is an upstream packaging bug on Trixie: `libcamera 0.7.0+rpt20260205-1` (Feb 2026) calls `libpisp::compute_optimal_stride`, but the highest published `libpisp1` is `1.4.0-1`, which doesn't export that symbol yet. There is currently no version of `libpisp1` in any apt repo that matches the Feb 2026 libcamera build.

**Fix: roll libcamera back to the last working build (Sept 2025, `0.5.2+rpt20250903-1`).**

```bash
cd /tmp
V=0.5.2+rpt20250903-1
for pkg in libcamera0.5 libcamera-ipa libcamera-tools libcamera-dev libcamera-v4l2 python3-libcamera; do
  wget "https://archive.raspberrypi.com/debian/pool/main/libc/libcamera/${pkg}_${V}_arm64.deb"
done
sudo apt install ./libcamera0.5_${V}_arm64.deb \
                 ./libcamera-ipa_${V}_arm64.deb \
                 ./libcamera-tools_${V}_arm64.deb \
                 ./libcamera-dev_${V}_arm64.deb \
                 ./libcamera-v4l2_${V}_arm64.deb \
                 ./python3-libcamera_${V}_arm64.deb
```

Apt will offer to remove `libcamera0.7` and `rpicam-apps` (they hard-depend on libcamera 0.7). Say yes — `rpicam-apps` is just CLI tools (`rpicam-hello`, etc.); the streaming pipeline doesn't use them.

**Pin everything so the next `apt upgrade` doesn't undo it:**

```bash
sudo apt-mark hold \
  libcamera0.5 libcamera-ipa libcamera-tools libcamera-dev libcamera-v4l2 \
  python3-libcamera python3-picamera2 \
  libpisp1 libpisp-common
apt-mark showhold
```

Verify:

```bash
python -c "from picamera2 import Picamera2; print(Picamera2.global_camera_info())"
```

This should now print two camera dicts. Once a `libpisp` ≥ 1.5 with the missing symbol is published in the rpi archive, `apt-mark unhold` the libcamera packages and run `sudo apt full-upgrade` to move forward again.

## Other troubleshooting

**Cameras enumerate as `[]`** — kernel sees them but libcamera doesn't. Check `/boot/firmware/config.txt` for `dtoverlay=imx477,cam0` and `dtoverlay=imx477,cam1` (replace `imx477` with your sensor name if you have a different module). Reboot.

**`ModuleNotFoundError: No module named 'picamera2'`** — venv was created without `--system-site-packages`. Recreate it:

```bash
deactivate; rm -rf .venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-rpi.txt
```

**`flow` ms is much higher than 20** — image sampling dominates. Check that `--camera-width` / `--camera-height` are set; without them picamera2 may negotiate a larger sensor mode and grayscale conversion gets expensive.

**`dropped > 0` in the status line** — the capture thread is producing faster than the main loop is consuming. Either lower `--camera-fps` or enable `--low-latency-capture` (drains the camera queue in a background thread; only delivers the freshest stereo pair to the main loop).

**`publish_rate` ≪ `camera_fps`** — UDP socket is blocking somewhere or the consumer is far away on a slow link. The script uses non-blocking UDP and silently drops on `BlockingIOError`, so a low publish rate usually means the main loop is missing its budget. Inspect `loop=` ms in the profile output.

**One camera's image is upside down** — the `Picamera2.global_camera_info()` output shows a `Rotation` field per sensor; physical orientation in the rig dictates which one is "up". The streaming code does *not* auto-rotate. If needed, rotate by 180° before passing to `estimator.step()` (cheap with `cv2.rotate(frame, cv2.ROTATE_180)` in the capture loop) and update the FOV / phi flags so the hex-grid sampler stays consistent.
