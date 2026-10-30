"""
Full model benchmark suite.

Runs these analyses on a trained model, in this order (each can be skipped):
  1. edge_sweep             — edge velocity tuning curves
  2. pano_sweep             — panoramic ego-motion estimation (--num-panos,
                              default 1)
  3. white_noise_silencing  — STA filter recovery + silencing comparison
  4. sine_wave_silencing    — tuning curves + silencing comparison
  5. performance_sweep      — flow + ego-motion cosine on the held-out
                              test split of random_sine and random_pano

All figures are saved under <model_folder>/figures/ (the white-noise STA data
under <model_folder>/analysis_data/white_noise_silencing/).

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.benchmark_model models/my_model/
    python -m drosophila_model.analysis.model_analysis.benchmark_model models/my_model/ --skip-edge
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_ANALYSIS_PACKAGE = "drosophila_model.analysis.model_analysis"


def _banner(label: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"{sep}\n")


def _run_subprocess(label: str, module: str, *args: str) -> None:
    """Run ``python -m <analysis package>.<module> <args>`` and warn on failure."""
    _banner(label)
    result = subprocess.run(
        [sys.executable, "-m", f"{_ANALYSIS_PACKAGE}.{module}", *args])
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
    parser.add_argument("--skip-white-noise-silencing", action="store_true",
                        help="Skip white noise silencing analysis")
    parser.add_argument("--skip-sine-silencing", action="store_true",
                        help="Skip sine wave silencing analysis")
    parser.add_argument("--skip-performance", action="store_true",
                        help="Skip performance sweep (flow + ego cosine)")
    parser.add_argument("--num-panos", type=int, default=1,
                        help="Number of pano stimuli (default: 1)")
    parser.add_argument("--pano-stim-time", type=float, default=5.0,
                        help="Pano stimulus duration in seconds (default: 5.0)")
    args = parser.parse_args()

    model_folder = args.model_folder

    # --- Edge sweep (subprocess — no return value needed) ---
    if not args.skip_edge:
        _run_subprocess("Edge Sweep", "edge_sweep", model_folder)

    # --- Pano sweep (import directly to capture metrics) ---
    pano_results = None
    if not args.skip_pano:
        _banner("Panoramic Sweep")
        from drosophila_model.analysis.model_analysis.pano_sweep import run_pano_sweep
        pano_results = run_pano_sweep(
            model_folder,
            num_panos=args.num_panos,
            stim_time=args.pano_stim_time,
        )

    # --- White noise silencing (subprocess — includes baseline) ---
    if not args.skip_white_noise_silencing:
        _run_subprocess("White Noise Silencing", "white_noise_silencing", model_folder)

    # --- Sine wave silencing (subprocess) ---
    if not args.skip_sine_silencing:
        _run_subprocess("Sine Wave Silencing", "sine_wave_silencing", model_folder)

    # --- Performance sweep (import directly to capture metrics) ---
    performance_results = None
    if not args.skip_performance:
        _banner("Performance Sweep")
        from drosophila_model.analysis.model_analysis.performance_sweep import run_performance_sweep
        performance_results = run_performance_sweep(model_folder)

    # --- Cell polarity from baseline STA ---
    on_cells = []
    off_cells = []
    sta_data_path = (Path(model_folder) / 'analysis_data' / 'white_noise_silencing'
                     / 'baseline' / 'sta_data.npz')
    if sta_data_path.exists():
        from drosophila_model.analysis.model_analysis.white_noise import sta_polarity
        sta_data = np.load(sta_data_path)
        on_cells, off_cells = sta_polarity(
            {ct: sta_data[f'sta_{ct}'] for ct in sta_data['cell_types']})

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  Benchmark Summary")
    print(f"{'='*60}")

    if on_cells or off_cells:
        print(f"  ON  cells (positive polarity): {', '.join(on_cells)}")
        print(f"  OFF cells (negative polarity): {', '.join(off_cells)}")

    if pano_results is not None:
        print(f"  Pano rotation  median cos sim: {pano_results['median_rot_cos']:.4f}")

    if performance_results is not None:
        flow_cos = performance_results['flow_cos']
        ego_cos = performance_results['ego_cos']
        print("  Performance (cosine similarity):")
        for name in dict.fromkeys(list(flow_cos) + list(ego_cos)):
            parts = []
            if name in flow_cos:
                parts.append(f"flow {flow_cos[name]:.4f}")
            ego = ego_cos.get(name, {})
            if ego.get('rot') is not None:
                parts.append(f"ego rot {ego['rot']:.4f}")
            if ego.get('trans') is not None:
                parts.append(f"ego trans {ego['trans']:.4f}")
            print(f"    {name:<14}{'  '.join(parts)}")

    print(f"\n  Figures saved to {Path(model_folder) / 'figures'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
