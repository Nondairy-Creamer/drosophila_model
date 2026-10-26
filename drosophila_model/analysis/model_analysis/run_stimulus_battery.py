"""
Run a full battery of stimuli through a model and produce figures + hex videos.

For each stimulus type, runs the corresponding view_*.py script with hex
animations enabled, then uses ffmpeg to tile the hex videos (stimulus | T4a | T5a)
into stacked composite movies.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.run_stimulus_battery models/<run>
    python -m drosophila_model.analysis.model_analysis.run_stimulus_battery models/<run>/model.pt
    python -m drosophila_model.analysis.model_analysis.run_stimulus_battery models/<run> --stimuli sine edge
    python -m drosophila_model.analysis.model_analysis.run_stimulus_battery models/<run> --skip-existing
"""
import argparse
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ALL_STIMULI = ["natural_scene", "sine", "edge", "impulse"]


def _view_cmd(name, *args):
    """Command that runs ``view_<name>`` as a module with *args*."""
    module = f"drosophila_model.analysis.model_analysis.view_{name}"
    return [sys.executable, "-m", module, *(str(a) for a in args)]


def _run(cmd, dry_run=False):
    """Run *cmd* (an argument list); return True if it succeeded."""
    # Flush so this log stays in order with the child's output when piped.
    print(f"  $ {shlex.join(cmd)}", flush=True)
    if dry_run:
        return True
    try:
        ret = subprocess.run(cmd).returncode
    except FileNotFoundError:
        print(f"  ERROR: '{cmd[0]}' not found.")
        return False
    if ret != 0:
        print(f"  ERROR: command exited with code {ret}")
        return False
    return True


def _ffmpeg_exe():
    """System ffmpeg if on PATH, else the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    return exe


def _ffmpeg_stack(inputs, output, how, dry_run=False):
    """Tile videos side by side (``how='hstack'``) or vertically (``'vstack'``)."""
    if not dry_run:
        missing = [str(f) for f in inputs if not Path(f).exists()]
        if missing:
            print(f"  Skipping {how} - missing: {missing}")
            return False
    n = len(inputs)
    # Only ffmpeg errors are printed; its banner and progress would bury the log.
    cmd = [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"]
    for f in inputs:
        cmd += ["-i", str(f)]
    filt = "".join(f"[{i}:v]" for i in range(n)) + f"{how}=inputs={n}[v]"
    cmd += ["-filter_complex", filt, "-map", "[v]"]
    if how == "vstack":
        cmd += ["-c:v", "libx264", "-crf", "23"]
    cmd.append(str(output))
    return _run(cmd, dry_run)


def run_natural_scene(model_path, save_dir, dry_run=False, skip_existing=False):
    """Natural panorama: right + left directions, hex stim | T4a | T5a."""
    marker = save_dir / "natural_pano_response_right.png"
    if skip_existing and marker.exists():
        print("  natural_scene: already exists, skipping.")
        return True

    directions = ["right", "left"]
    ok = True
    for d in directions:
        ok &= _run(_view_cmd(
            "natural_scene",
            "--hex", "--hex-cell-type", "T4a", "T5a",
            "--highpass-tau", 1, "--stim-time", 10, "--pano-index", 30,
            "--direction", d,
            "--save", save_dir / f"natural_pano_response_{d}.png",
            "--model", model_path,
        ), dry_run)

    per_dir_mp4s = []
    for d in directions:
        row_mp4 = save_dir / f"natural_pano_{d}.mp4"
        ok &= _ffmpeg_stack(
            [save_dir / f"hex_{name}_pano30_{d}_30dps.mp4"
             for name in ("stim", "T4a", "T5a")],
            row_mp4, "hstack", dry_run)
        per_dir_mp4s.append(row_mp4)

    ok &= _ffmpeg_stack(per_dir_mp4s, save_dir / "natural_pano_stacked.mp4",
                        "vstack", dry_run)
    return ok


def run_sine(model_path, save_dir, dry_run=False, skip_existing=False):
    """Sine grating: right + left, hex stim | T4a | T5a."""
    wl, tf = 30, 1.0
    directions = ["right", "left"]
    tag = f"lambda{wl:.0f}_tf{tf:.1f}"
    marker = save_dir / f"sine_responses_{'_'.join(directions)}_{tag}.png"
    if skip_existing and marker.exists():
        print("  sine: already exists, skipping.")
        return True

    # One call for both directions, so the static figures hold both.
    ok = _run(_view_cmd(
        "sine_stimulus",
        "--wavelength", wl, "--temporal-frequency", tf,
        "--hex", "--hex-cell-type", "T4a", "T5a",
        "--stim-time", 10, "--zero-center",
        "--direction", *directions,
        "--save-dir", save_dir,
        "--model", model_path,
    ), dry_run)

    per_dir_mp4s = []
    for d in directions:
        row_mp4 = save_dir / f"sine_{d}_{tag}.mp4"
        ok &= _ffmpeg_stack(
            [save_dir / f"sine_hex_{name}_{d}_{tag}.mp4"
             for name in ("stim", "T4a", "T5a")],
            row_mp4, "hstack", dry_run)
        per_dir_mp4s.append(row_mp4)

    ok &= _ffmpeg_stack(per_dir_mp4s, save_dir / f"sine_stacked_{tag}.mp4",
                        "vstack", dry_run)
    return ok


def run_edge(model_path, save_dir, dry_run=False, skip_existing=False):
    """Moving edges: ON right, ON left, OFF right, OFF left, hex stim | T4a."""
    marker = save_dir / "edge_responses_30dps.png"
    if skip_existing and marker.exists():
        print("  edge: already exists, skipping.")
        return True

    # view_edge_stimulus runs (and with --hex animates) all four conditions.
    ok = _run(_view_cmd(
        "edge_stimulus",
        "--hex", "--hex-cell-type", "T4a",
        "--stim-time", 3, "--zero-center",
        "--save-dir", save_dir,
        "--model", model_path,
    ), dry_run)

    per_cond_mp4s = []
    for cond in ["ON right", "ON left", "OFF right", "OFF left"]:
        safe = cond.lower().replace(" ", "_")
        row_mp4 = save_dir / f"edge_{safe}.mp4"
        ok &= _ffmpeg_stack(
            [save_dir / f"edge_hex_{name}_{safe}_30dps.mp4"
             for name in ("stim", "T4a")],
            row_mp4, "hstack", dry_run)
        per_cond_mp4s.append(row_mp4)

    ok &= _ffmpeg_stack(per_cond_mp4s, save_dir / "edge_stacked.mp4",
                        "vstack", dry_run)
    return ok


_IMPULSE_FIGURE = re.compile(r"impulse_responses_(\d+ms)\.png")


def _impulse_tag(save_dir):
    """Pulse tag (``'<N>ms'``) of the newest impulse figure in *save_dir*.

    view_impulse_stimulus names its outputs after the pulse length, which
    defaults to one model time step, so the tag depends on the model's dt.
    Returns None when no impulse figure exists.
    """
    found = []
    for path in save_dir.glob("impulse_responses_*ms.png"):
        match = _IMPULSE_FIGURE.fullmatch(path.name)
        if match:
            found.append((path.stat().st_mtime, match.group(1)))
    return max(found)[1] if found else None


def run_impulse(model_path, save_dir, dry_run=False, skip_existing=False):
    """Full-field impulse: increment + decrement, hex for L1 and T4a."""
    if skip_existing and _impulse_tag(save_dir) is not None:
        print("  impulse: already exists, skipping.")
        return True

    ok = _run(_view_cmd(
        "impulse_stimulus",
        "--hex", "--hex-cell-type", "L1", "T4a",
        "--post-stim-time", 2,
        "--save-dir", save_dir,
        "--model", model_path,
    ), dry_run)

    tag = "<N>ms" if dry_run else _impulse_tag(save_dir)
    if tag is None:
        print("  Skipping impulse stacking - no impulse figures found.")
        return False

    per_cond_mp4s = []
    for cond in ["increment", "decrement"]:
        row_mp4 = save_dir / f"impulse_{cond}.mp4"
        ok &= _ffmpeg_stack(
            [save_dir / f"impulse_hex_{ct}_{cond}_{tag}.mp4"
             for ct in ("L1", "T4a")],
            row_mp4, "hstack", dry_run)
        per_cond_mp4s.append(row_mp4)

    ok &= _ffmpeg_stack(per_cond_mp4s, save_dir / "impulse_stacked.mp4",
                        "vstack", dry_run)
    return ok


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
        help="Path to the model .pt file or its folder.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Output directory (default: <model_folder>/figures/run_stimulus_battery).",
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

    model_root = model_path.parent if model_path.is_file() else model_path
    save_dir = (Path(args.save_dir) if args.save_dir
                else model_root / "figures" / "run_stimulus_battery")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:    {model_path}")
    print(f"Save dir: {save_dir}")
    print(f"Stimuli:  {args.stimuli}")
    print()

    failed = []
    for stim_name in args.stimuli:
        print(f"{'='*60}")
        print(f"  {stim_name}")
        print(f"{'='*60}")
        runner = STIMULUS_RUNNERS[stim_name]
        if not runner(model_path, save_dir,
                      dry_run=args.dry_run, skip_existing=args.skip_existing):
            failed.append(stim_name)
        print()

    if failed:
        print(f"Finished with errors in: {', '.join(failed)}")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
