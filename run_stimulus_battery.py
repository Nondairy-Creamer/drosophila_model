"""
Run a full battery of stimuli through a model and produce figures + hex videos.

For each stimulus type, runs the corresponding view_*.py script with hex
animations enabled, then uses ffmpeg to tile the hex videos (stimulus | T4a | T5a)
into stacked composite movies.

Usage:
    python run_stimulus_battery.py models/hand_tuned/model.pt
    python run_stimulus_battery.py models/hand_tuned_offset_omega_1_5_offset_0_05/offset_omega_1_5_offset_0_05.pt
    python run_stimulus_battery.py models/hand_tuned/model.pt --stimuli sine edge
    python run_stimulus_battery.py models/hand_tuned/model.pt --skip-existing
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ALL_STIMULI = ["natural_scene", "sine", "edge", "impulse"]


def _run(cmd, dry_run=False):
    print(f"  $ {cmd}")
    if dry_run:
        return
    ret = os.system(cmd)
    if ret != 0:
        print(f"  WARNING: command exited with code {ret}")


def _ffmpeg_hstack(inputs, output, dry_run=False):
    """Tile 2-3 videos horizontally."""
    n = len(inputs)
    if any(not Path(f).exists() for f in inputs):
        missing = [f for f in inputs if not Path(f).exists()]
        print(f"  Skipping hstack — missing: {missing}")
        return
    inp = " ".join(f'-i {f}' for f in inputs)
    filt = "".join(f"[{i}:v]" for i in range(n)) + f"hstack=inputs={n}[v]"
    cmd = f'ffmpeg -y {inp} -filter_complex "{filt}" -map "[v]" {output}'
    _run(cmd, dry_run)


def _ffmpeg_vstack(inputs, output, dry_run=False):
    """Tile videos vertically."""
    n = len(inputs)
    if any(not Path(f).exists() for f in inputs):
        missing = [f for f in inputs if not Path(f).exists()]
        print(f"  Skipping vstack — missing: {missing}")
        return
    inp = " ".join(f'-i {f}' for f in inputs)
    filt = "".join(f"[{i}:v]" for i in range(n)) + f"vstack=inputs={n}[v]"
    cmd = f'ffmpeg -y {inp} -filter_complex "{filt}" -map "[v]" -c:v libx264 -crf 23 {output}'
    _run(cmd, dry_run)


def run_natural_scene(model_path, save_dir, dry_run=False, skip_existing=False):
    """Natural panorama: right + left directions, hex stim | T4a | T5a."""
    marker = save_dir / "natural_pano_response_right.png"
    if skip_existing and marker.exists():
        print("  natural_scene: already exists, skipping.")
        return

    directions = ["right", "left"]
    for d in directions:
        cmd = (
            f"{sys.executable} analysis/view_natural_scene.py"
            f" --hex --hex-cell-type T4a T5a"
            f" --highpass-tau 1 --stim-time 10 --pano-index 30"
            f" --direction {d}"
            f" --save {save_dir}/natural_pano_response_{d}.png"
            f" --model {model_path}"
        )
        _run(cmd, dry_run)

    per_dir_mp4s = []
    for d in directions:
        stim_mp4 = f"{save_dir}/hex_stim_pano30_{d}_30dps.mp4"
        t4a_mp4 = f"{save_dir}/hex_T4a_pano30_{d}_30dps.mp4"
        t5a_mp4 = f"{save_dir}/hex_T5a_pano30_{d}_30dps.mp4"
        row_mp4 = f"{save_dir}/natural_pano_{d}.mp4"
        _ffmpeg_hstack([stim_mp4, t4a_mp4, t5a_mp4], row_mp4, dry_run)
        per_dir_mp4s.append(row_mp4)

    _ffmpeg_vstack(per_dir_mp4s,
                   f"{save_dir}/natural_pano_stacked.mp4", dry_run)


def run_sine(model_path, save_dir, dry_run=False, skip_existing=False):
    """Sine grating: right + left, hex stim | T4a | T5a."""
    wl, tf = 30, 1.0
    marker = save_dir / f"sine_responses_lambda{wl:.0f}_tf{tf:.1f}.png"
    if skip_existing and marker.exists():
        print("  sine: already exists, skipping.")
        return

    directions = ["right", "left"]
    for d in directions:
        cmd = (
            f"{sys.executable} analysis/view_sine_stimulus.py"
            f" --wavelength {wl} --temporal-frequency {tf}"
            f" --hex --hex-cell-type T4a T5a"
            f" --stim-time 10 --zero-center"
            f" --direction {d}"
            f" --save-dir {save_dir}"
            f" --model {model_path}"
        )
        _run(cmd, dry_run)

    tag = f"lambda{wl:.0f}_tf{tf:.1f}"
    per_dir_mp4s = []
    for d in directions:
        stim_mp4 = f"{save_dir}/sine_hex_stim_{d}_{tag}.mp4"
        t4a_mp4 = f"{save_dir}/sine_hex_T4a_{d}_{tag}.mp4"
        t5a_mp4 = f"{save_dir}/sine_hex_T5a_{d}_{tag}.mp4"
        row_mp4 = f"{save_dir}/sine_{d}_{tag}.mp4"
        _ffmpeg_hstack([stim_mp4, t4a_mp4, t5a_mp4], row_mp4, dry_run)
        per_dir_mp4s.append(row_mp4)

    _ffmpeg_vstack(per_dir_mp4s,
                   f"{save_dir}/sine_stacked_{tag}.mp4", dry_run)


def run_edge(model_path, save_dir, dry_run=False, skip_existing=False):
    """Moving edges: ON right, ON left, OFF right, OFF left."""
    marker = save_dir / "edge_responses_30dps.png"
    if skip_existing and marker.exists():
        print("  edge: already exists, skipping.")
        return

    conditions = ["ON right", "ON left", "OFF right", "OFF left"]
    for cond in conditions:
        cmd = (
            f'{sys.executable} analysis/view_edge_stimulus.py'
            f' --hex --hex-cell-type T4a'
            f' --condition "{cond}"'
            f' --stim-time 3 --zero-center'
            f' --save-dir {save_dir}'
            f' --model {model_path}'
        )
        _run(cmd, dry_run)

    per_cond_mp4s = []
    for cond in conditions:
        safe = cond.lower().replace(" ", "_")
        stim_mp4 = f"{save_dir}/edge_hex_stim_{safe}_30dps.mp4"
        t4a_mp4 = f"{save_dir}/edge_hex_T4a_{safe}_30dps.mp4"
        row_mp4 = f"{save_dir}/edge_{safe}.mp4"
        _ffmpeg_hstack([stim_mp4, t4a_mp4], row_mp4, dry_run)
        per_cond_mp4s.append(row_mp4)

    if len(per_cond_mp4s) >= 2:
        _ffmpeg_vstack(per_cond_mp4s,
                       f"{save_dir}/edge_stacked.mp4", dry_run)


def run_impulse(model_path, save_dir, dry_run=False, skip_existing=False):
    """Full-field impulse: increment + decrement, hex for L1 and T4a."""
    marker = save_dir / "impulse_responses_5ms.png"
    if skip_existing and marker.exists():
        print("  impulse: already exists, skipping.")
        return

    cmd = (
        f"{sys.executable} analysis/view_impulse_stimulus.py"
        f" --hex --hex-cell-type L1 T4a"
        f" --post-stim-time 2"
        f" --save-dir {save_dir}"
        f" --model {model_path}"
    )
    _run(cmd, dry_run)

    per_cond_mp4s = []
    for cond in ["increment", "decrement"]:
        l1_mp4 = f"{save_dir}/impulse_hex_L1_{cond}_5ms.mp4"
        t4a_mp4 = f"{save_dir}/impulse_hex_T4a_{cond}_5ms.mp4"
        row_mp4 = f"{save_dir}/impulse_{cond}.mp4"
        _ffmpeg_hstack([l1_mp4, t4a_mp4], row_mp4, dry_run)
        per_cond_mp4s.append(row_mp4)

    _ffmpeg_vstack(per_cond_mp4s,
                   f"{save_dir}/impulse_stacked.mp4", dry_run)


STIMULUS_RUNNERS = {
    "natural_scene": run_natural_scene,
    "sine": run_sine,
    "edge": run_edge,
    "impulse": run_impulse,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run a battery of stimuli through a model."
    )
    parser.add_argument(
        "model_path", type=str,
        help="Path to the model .pt file.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Output directory (default: same directory as the model file).",
    )
    parser.add_argument(
        "--stimuli", type=str, nargs="+", default=ALL_STIMULI,
        choices=ALL_STIMULI,
        help="Which stimulus types to run (default: all).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a stimulus if its marker figure already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    save_dir = Path(args.save_dir) if args.save_dir else model_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:    {model_path}")
    print(f"Save dir: {save_dir}")
    print(f"Stimuli:  {args.stimuli}")
    print()

    for stim_name in args.stimuli:
        print(f"{'='*60}")
        print(f"  {stim_name}")
        print(f"{'='*60}")
        runner = STIMULUS_RUNNERS[stim_name]
        runner(model_path, save_dir,
               dry_run=args.dry_run, skip_existing=args.skip_existing)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
