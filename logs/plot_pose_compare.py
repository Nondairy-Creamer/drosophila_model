#!/usr/bin/env python3
"""Plot optic-flow pose against AprilTag raw and filtered pose logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


POSE_COLUMNS = (
    ("x_m", "x (m)"),
    ("y_m", "y (m)"),
    ("z_m", "z (m)"),
    ("roll_deg", "roll (deg)"),
    ("pitch_deg", "pitch (deg)"),
    ("yaw_deg", "yaw (deg)"),
)

SERIES_STYLES = {
    "optic_flow": {"color": "#1f77b4", "linewidth": 1.6},
    "apriltag_raw": {"color": "#d62728", "linewidth": 1.2, "alpha": 0.9},
    "apriltag_filtered": {"color": "#2ca02c", "linewidth": 1.6},
}


def find_default_optic_flow_log(log_dir: Path) -> Path:
    matches = sorted(log_dir.glob("optic_flow_pose*.csv"))
    if not matches:
        raise FileNotFoundError(f"No optic-flow CSV found in {log_dir}")
    return matches[-1]


def load_pose_csv(path: Path, require_available: bool) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    wall_time_s = np.asarray([float(row["wall_time_s"]) for row in rows], dtype=np.float64)
    frame_index = np.asarray([int(row["frame_index"]) for row in rows], dtype=np.int64)
    dt_s = np.asarray([float(row["dt_s"]) for row in rows], dtype=np.float64)

    data = {
        "wall_time_s": wall_time_s,
        "frame_index": frame_index,
        "dt_s": dt_s,
    }
    for col, _label in POSE_COLUMNS:
        values = np.asarray([float(row[col]) for row in rows], dtype=np.float64)
        data[col] = values

    if require_available and "available" in rows[0]:
        available = np.asarray([int(row["available"]) != 0 for row in rows], dtype=bool)
        for col, _label in POSE_COLUMNS:
            masked = data[col].copy()
            masked[~available] = np.nan
            data[col] = masked
        data["available"] = available

    return data


def estimate_duration_s(data: dict[str, np.ndarray]) -> float:
    positive_dt = data["dt_s"][data["dt_s"] > 0]
    if positive_dt.size > 0:
        nominal_dt = float(np.median(positive_dt))
    else:
        nominal_dt = 1.0
    max_frame = max(float(data["frame_index"][-1]), 1.0)
    return nominal_dt * max_frame


def build_time_axis(
    data: dict[str, np.ndarray],
    time_mode: str,
    reference_duration_s: float,
) -> tuple[np.ndarray, str]:
    if time_mode == "wall":
        return data["wall_time_s"] - data["wall_time_s"][0], "wall time since start (s)"
    if time_mode == "dt":
        dt = data["dt_s"].copy()
        if dt.size > 0:
            dt[0] = 0.0
        return np.cumsum(dt), "cumulative logged dt (s)"
    if time_mode == "frame":
        return data["frame_index"].astype(np.float64), "frame index"

    max_frame = max(float(data["frame_index"][-1]), 1.0)
    aligned_time = data["frame_index"].astype(np.float64) * (reference_duration_s / max_frame)
    return aligned_time, "aligned video time (s)"


def robust_amplitude(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    centered = finite - np.nanmedian(finite)
    amp = float(np.nanpercentile(np.abs(centered), 95))
    return amp if amp > 1e-12 else 1.0


def scale_optic_flow_series(
    optic_flow: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    scale_mode: str,
) -> dict[str, float]:
    scale_factors: dict[str, float] = {}

    if scale_mode == "none":
        for col, _label in POSE_COLUMNS:
            scale_factors[col] = 1.0
        return scale_factors

    if scale_mode == "global":
        optic_amp = np.mean([robust_amplitude(optic_flow[col]) for col, _label in POSE_COLUMNS])
        ref_amp = np.mean([robust_amplitude(reference[col]) for col, _label in POSE_COLUMNS])
        global_scale = ref_amp / max(optic_amp, 1e-12)
        for col, _label in POSE_COLUMNS:
            optic_flow[col] = optic_flow[col] * global_scale
            scale_factors[col] = global_scale
        return scale_factors

    for col, _label in POSE_COLUMNS:
        optic_amp = robust_amplitude(optic_flow[col])
        ref_amp = robust_amplitude(reference[col])
        scale = ref_amp / max(optic_amp, 1e-12)
        optic_flow[col] = optic_flow[col] * scale
        scale_factors[col] = scale
    return scale_factors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot optic-flow pose against AprilTag raw and filtered CSV logs."
    )
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--optic-flow",
        type=str,
        default=None,
        help="Optic-flow pose CSV. Defaults to the latest optic_flow_pose*.csv in this folder.",
    )
    parser.add_argument(
        "--apriltag-raw",
        type=str,
        default=str(default_dir / "apriltag_pose_compare_raw.csv"),
        help="AprilTag raw pose CSV.",
    )
    parser.add_argument(
        "--apriltag-filtered",
        type=str,
        default=str(default_dir / "apriltag_pose_compare_filtered.csv"),
        help="AprilTag filtered pose CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_dir / "pose_compare.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Optic Flow vs AprilTag Pose",
        help="Figure title.",
    )
    parser.add_argument(
        "--time-mode",
        choices=("aligned", "wall", "dt", "frame"),
        default="aligned",
        help=(
            "How to build the x-axis. 'aligned' rescales each log by frame index to a shared "
            "reference duration and is the best choice when all logs come from the same video."
        ),
    )
    parser.add_argument(
        "--reference-series",
        choices=("optic_flow", "apriltag_raw", "apriltag_filtered"),
        default="apriltag_filtered",
        help="Reference series used to set the shared duration for aligned time mode.",
    )
    parser.add_argument(
        "--reference-duration-s",
        type=float,
        default=None,
        help="Optional explicit duration for aligned time mode. Overrides the reference-series estimate.",
    )
    parser.add_argument(
        "--optic-flow-scale",
        choices=("per-axis", "global", "none"),
        default="per-axis",
        help=(
            "Scale the optic-flow pose before plotting so its magnitude better matches the "
            "AprilTag reference."
        ),
    )
    parser.add_argument(
        "--optic-flow-scale-reference",
        choices=("apriltag_raw", "apriltag_filtered"),
        default="apriltag_filtered",
        help="AprilTag series used as the magnitude reference for optic-flow scaling.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in addition to saving it.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg" if not args.show else matplotlib.get_backend())
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as exc:
        parser.error(f"Missing plotting dependency: {exc}")

    script_dir = Path(__file__).resolve().parent
    optic_flow_path = Path(args.optic_flow) if args.optic_flow else find_default_optic_flow_log(script_dir)
    raw_path = Path(args.apriltag_raw)
    filtered_path = Path(args.apriltag_filtered)
    output_path = Path(args.output)

    optic_flow = load_pose_csv(optic_flow_path, require_available=False)
    apriltag_raw = load_pose_csv(raw_path, require_available=True)
    apriltag_filtered = load_pose_csv(filtered_path, require_available=True)

    scale_reference = apriltag_filtered if args.optic_flow_scale_reference == "apriltag_filtered" else apriltag_raw
    scale_factors = scale_optic_flow_series(optic_flow, scale_reference, args.optic_flow_scale)

    series = {
        "optic_flow": optic_flow,
        "apriltag_raw": apriltag_raw,
        "apriltag_filtered": apriltag_filtered,
    }
    reference_duration_s = (
        float(args.reference_duration_s)
        if args.reference_duration_s is not None
        else estimate_duration_s(series[args.reference_series])
    )
    x_label = None
    for data in series.values():
        plot_time_s, x_label = build_time_axis(data, args.time_mode, reference_duration_s)
        data["plot_time_s"] = plot_time_s

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
    axes = axes.ravel()

    plot_series = [
        ("optic_flow", "Optic flow", optic_flow),
        ("apriltag_raw", "AprilTag raw", apriltag_raw),
        ("apriltag_filtered", "AprilTag filtered", apriltag_filtered),
    ]

    for ax, (col, label) in zip(axes, POSE_COLUMNS):
        for key, legend_label, data in plot_series:
            style = SERIES_STYLES[key]
            ax.plot(data["plot_time_s"], data[col], label=legend_label, **style)
        ax.set_title(label)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best")
    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    print(f"  optic-flow      : {optic_flow_path}")
    print(f"  apriltag raw    : {raw_path}")
    print(f"  apriltag filtered: {filtered_path}")
    print(f"  time mode       : {args.time_mode}")
    print(
        f"  optic-flow scale: {args.optic_flow_scale} "
        f"(reference={args.optic_flow_scale_reference})"
    )
    if args.time_mode == "aligned":
        print(
            f"  aligned duration: {reference_duration_s:.3f}s "
            f"(reference={args.reference_series})"
        )
    for col, _label in POSE_COLUMNS:
        print(f"  scale[{col}]    : {scale_factors[col]:.6f}")

    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
