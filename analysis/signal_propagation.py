"""
Signal propagation through L1 -> Tm1 -> T5.

Loads an untrained model (``model_init.pt``), presents a single rightward
sine-wave stimulus, and saves XT (column-by-time) plots of the firing rates
for L1, Tm1, and the four T5 subtypes (T5a-d).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import neural_models as nm
from stimulus_utilities import make_lr_sine_wave
from analysis_utilities import prepare_xt


CELL_TYPES = ["L1", "Tm1", "T5a"]


def run_signal_propagation(
    model,
    output_dir,
    wavelength=30.0,
    temporal_frequency=4.0,
    stim_time=1.0,
    pre_stim_time=0.5,
    contrast=1.0,
):
    """Present a rightward sine wave and save XT plots for L1, Tm1, T5."""
    dt = model.dt
    sample_phi = model.column_phi_theta[:, 0]
    sample_theta = model.column_phi_theta[:, 1]

    sine_wave, _ = make_lr_sine_wave(
        sample_phi, sample_theta,
        contrast=contrast,
        wavelength=wavelength,
        temporal_frequency=temporal_frequency,
        dt=dt,
        stim_time=stim_time,
        pre_stim_time=pre_stim_time,
    )

    stim_tensor = torch.tensor(sine_wave[None, :, :], device=model.device, dtype=model.dtype)

    available = [ct for ct in CELL_TYPES if ct in model.unique_cell_types]
    missing = [ct for ct in CELL_TYPES if ct not in model.unique_cell_types]
    if missing:
        print(f"Warning: cell types not present in model, skipping: {missing}")

    print(f"Running sine wave (λ={wavelength}°, tf={temporal_frequency} Hz, "
          f"{sine_wave.shape[0]} time steps) through model...")
    with torch.no_grad():
        pred = model.predict(
            stim_tensor,
            input_neurons=["L1", "L2", "L3"],
            output_neurons=available,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_stim_steps = int(round(pre_stim_time / dt))
    time_axis_s = np.arange(sine_wave.shape[0]) * dt - pre_stim_time

    panels = [("stimulus", sine_wave)] + [
        (ct, pred["outputs"]["firing_rate"][ct][0].cpu().numpy()) for ct in available
    ]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4 * len(panels), 5), squeeze=False,
    )
    for idx, (title, data) in enumerate(panels):
        ax = axes[0, idx]
        sorted_data, extent = prepare_xt(data, sample_phi)
        cmap = "gray" if title == "stimulus" else "viridis"
        im = ax.imshow(
            sorted_data,
            aspect="auto",
            extent=[extent[0], extent[1], time_axis_s[-1], time_axis_s[0]],
            cmap=cmap,
            origin="upper",
        )
        ax.axhline(0, color="white", linewidth=0.5, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("φ (rad)")
        if idx == 0:
            ax.set_ylabel("Time (s)")
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)

    fig.suptitle(
        f"Signal propagation — rightward sine wave "
        f"(λ={wavelength}°, tf={temporal_frequency} Hz)",
        fontsize=13,
    )
    plt.tight_layout()

    save_path = output_dir / (
        f"signal_propagation_lambda{wavelength:.0f}_tf{temporal_frequency:.1f}.png"
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_folder", type=str,
        help="Path to model folder (containing model_init.pt and config.toml).",
    )
    parser.add_argument("--wavelength", type=float, default=30.0,
                        help="Spatial wavelength in degrees (default: 30).")
    parser.add_argument("--temporal-frequency", type=float, default=4.0,
                        help="Temporal frequency in Hz (default: 4).")
    parser.add_argument("--stim-time", type=float, default=1.0,
                        help="Stimulus duration in seconds (default: 1.0).")
    parser.add_argument("--pre-stim-time", type=float, default=0.5,
                        help="Pre-stimulus warmup in seconds (default: 0.5).")
    parser.add_argument("--contrast", type=float, default=1.0,
                        help="Sine-wave contrast (default: 1.0).")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <model_folder>/figures/signal_propagation).",
    )
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    model = nm.NeuralModel.load(model_folder / "model_init.pt")
    model.eval()

    output_dir = (Path(args.output_dir) if args.output_dir
                  else model_folder / "figures" / "signal_propagation")

    run_signal_propagation(
        model,
        output_dir=output_dir,
        wavelength=args.wavelength,
        temporal_frequency=args.temporal_frequency,
        stim_time=args.stim_time,
        pre_stim_time=args.pre_stim_time,
        contrast=args.contrast,
    )


if __name__ == "__main__":
    main()
