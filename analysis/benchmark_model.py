"""
Full model benchmark suite.

Runs all five analyses on a trained model:
  1. edge_sweep             — edge velocity tuning curves
  2. pano_sweep             — panoramic ego-motion estimation (5 stimuli)
  3. drone_sweep            — drone video ego-motion estimation (3 clips)
  4. white_noise_silencing  — STA filter recovery + silencing comparison
  5. sine_wave_silencing    — tuning curves + silencing comparison

All figures are saved under <model_folder>/figures/.

Usage:
    python analysis/benchmark_model.py models/my_model/
    python analysis/benchmark_model.py models/my_model/ --skip-drone
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _run_subprocess(label: str, cmd: list[str]) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"{sep}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {label} exited with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full benchmark suite on a trained model."
    )
    parser.add_argument("model_folder", type=str,
                        help="Path to model folder (containing model.pt)")
    parser.add_argument("--skip-edge", action="store_true",
                        help="Skip edge sweep")
    parser.add_argument("--skip-pano", action="store_true",
                        help="Skip panoramic sweep")
    parser.add_argument("--skip-drone", action="store_true",
                        help="Skip drone sweep")
    parser.add_argument("--skip-white-noise-silencing", action="store_true",
                        help="Skip white noise silencing analysis")
    parser.add_argument("--skip-sine-silencing", action="store_true",
                        help="Skip sine wave silencing analysis")
    parser.add_argument("--num-panos", type=int, default=5,
                        help="Number of pano stimuli (default: 5)")
    parser.add_argument("--num-clips", type=int, default=3,
                        help="Number of drone clips (default: 3)")
    parser.add_argument("--pano-stim-time", type=float, default=5.0,
                        help="Pano stimulus duration in seconds (default: 5.0)")
    args = parser.parse_args()

    analysis_dir = Path(__file__).parent
    model_folder = args.model_folder

    # --- Edge sweep (subprocess — no return value needed) ---
    if not args.skip_edge:
        _run_subprocess("Edge Sweep", [
            sys.executable, str(analysis_dir / "edge_sweep.py"),
            model_folder,
        ])

    # --- Pano sweep (import directly to capture metrics) ---
    pano_results = None
    if not args.skip_pano:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Panoramic Sweep")
        print(f"{sep}\n")
        from analysis.pano_sweep import run_pano_sweep
        pano_results = run_pano_sweep(
            model_folder,
            num_panos=args.num_panos,
            stim_time=args.pano_stim_time,
        )

    # --- White noise silencing (subprocess — includes baseline) ---
    if not args.skip_white_noise_silencing:
        _run_subprocess("White Noise Silencing", [
            sys.executable, str(analysis_dir / "white_noise_silencing.py"),
            model_folder,
        ])

    # --- Sine wave silencing (subprocess) ---
    if not args.skip_sine_silencing:
        _run_subprocess("Sine Wave Silencing", [
            sys.executable, str(analysis_dir / "sine_wave_silencing.py"),
            model_folder,
        ])

    # --- Drone sweep (import directly to capture metrics) ---
    drone_results = None
    if not args.skip_drone:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Drone Sweep")
        print(f"{sep}\n")
        from analysis.drone_sweep import run_drone_sweep
        drone_results = run_drone_sweep(
            model_folder,
            num_clips=args.num_clips,
        )

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  Benchmark Summary")
    print(f"{'='*60}")

    if pano_results is not None:
        print(f"  Pano rotation  median cos sim: {pano_results['median_rot_cos']:.4f}")

    if drone_results is not None:
        print(f"  Drone rotation  median cos sim: {drone_results['median_rot_cos']:.4f}")
        print(f"  Drone translation median cos sim: {drone_results['median_trans_cos']:.4f}")

    print(f"\n  Figures saved to {Path(model_folder) / 'figures'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
