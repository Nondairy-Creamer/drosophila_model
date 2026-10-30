"""
Signal propagation from the lamina to T4 / T5.

Loads a model (by default the untrained ``model_init.pt``), presents a single
rightward sine-wave stimulus (static during the pre-stimulus period) and saves:

- ``signal_propagation_<model file>_lambda<wavelength>_tf<frequency>.png``:
  XT (column-by-time) plots of the stimulus and, for ``XT_CELL_TYPES``
  (L1, L2, Mi1, Tm1, T4a, T5a; T4a / T5a prefer this grating's direction on the
  right eye), the voltage relative to firing threshold (top row; blue = below
  threshold, where the rectified rate is zero and passes no gradient) and the
  firing rate (bottom row). Each panel has its own colour scale, so its title
  gives the response amplitude.
- ``response_amplitude_<model file>_lambda<wavelength>_tf<frequency>.png``: for
  the ON chain (L1 -> Mi1 / Tm3 -> T4) and the OFF chain
  (L2 / L3 -> Tm1 / Tm2 / Tm4 / Tm9 -> T5), each cell type's firing-rate
  modulation at the grating's temporal frequency (log scale) and the fraction
  of time it spends above threshold.

``--gain-scale`` (multiplies every cell type's output strength) and
``--threshold-shift`` (adds to every threshold, in mV; conductance models
only) let you preview other initialisations without retraining.

Usage:
    python -m drosophila_model.analysis.model_analysis.signal_propagation models/<model>/
    python -m drosophila_model.analysis.model_analysis.signal_propagation models/<model>/ --gain-scale 3 --threshold-shift -2
"""
import argparse
from pathlib import Path


import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling.stimulus_generators import make_lr_sine_wave


XT_CELL_TYPES = ["L1", "L2", "Mi1", "Tm1", "T4a", "T5a"]
ON_CHAIN = ["L1", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "CT1_M", "T4a", "T4b", "T4c", "T4d"]
OFF_CHAIN = ["L2", "L3", "Tm1", "Tm2", "Tm4", "Tm9", "CT1_L", "T5a", "T5b", "T5c", "T5d"]


def xt_by_azimuth(activity, phi_deg, bin_deg=3.0):
    """Average columns (``activity`` is (time, columns)) into even ``bin_deg`` azimuth bins.

    Returns ``(binned, extent)`` with ``extent = [phi_min, phi_max]`` of the bin edges, so the
    x axis of an XT plot is linear in azimuth (the columns themselves are not evenly spaced).
    Columns with NaNs are dropped; empty bins are NaN.
    """
    valid = np.isfinite(activity).all(axis=0)
    activity, phi_deg = activity[:, valid], phi_deg[valid]
    edges = np.arange(phi_deg.min(), phi_deg.max() + bin_deg, bin_deg)
    which = np.clip(np.digitize(phi_deg, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(which, minlength=len(edges) - 1)
    sums = np.stack([np.bincount(which, weights=row, minlength=len(edges) - 1) for row in activity])
    with np.errstate(invalid="ignore"):
        binned = sums / counts
    return binned, [float(edges[0]), float(edges[-1])]


def modulation_amplitude(x, dt, temporal_frequency):
    """Mean over columns of the amplitude of ``x`` (time, columns) at ``temporal_frequency``."""
    t = np.arange(x.shape[0]) * dt
    carrier = np.exp(-2j * np.pi * temporal_frequency * t)
    return float(np.mean(np.abs((x - x.mean(axis=0)).T @ carrier)) * 2 / x.shape[0])


def run_signal_propagation(
    model,
    output_dir,
    wavelength=30.0,
    temporal_frequency=4.0,
    stim_time=1.0,
    pre_stim_time=0.5,
    contrast=1.0,
    tag="model_init",
):
    """Present a rightward sine wave and save the XT and response-amplitude figures."""
    dt = model.dt
    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]

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

    wanted = list(dict.fromkeys(XT_CELL_TYPES + ON_CHAIN + OFF_CHAIN))
    available = [ct for ct in wanted if ct in model.unique_cell_types]
    missing = [ct for ct in wanted if ct not in model.unique_cell_types]
    if missing:
        print(f"Warning: cell types not present in model, skipping: {missing}")

    print(f"Running sine wave (lambda={wavelength} deg, tf={temporal_frequency} Hz, "
          f"{sine_wave.shape[0]} time steps) through model...")
    with torch.no_grad():
        pred = model.predict(
            stim_tensor,
            input_neurons=["L1", "L2", "L3"],
            output_neurons=available,
        )
    out = pred["outputs"]["optic_lobe"]

    # Voltage relative to each cell type's firing threshold (0 for models without one).
    thresholds = getattr(model, "threshold", None)
    type_index = {ct: i for i, ct in enumerate(model.unique_cell_types)}

    def relative_voltage(ct):
        v = out["voltage"][ct][0].cpu().numpy()
        if thresholds is None:
            return v
        return v - float(thresholds[type_index[ct]])

    firing = {ct: out["firing_rate"][ct][0].cpu().numpy() for ct in available}

    # Response amplitude over the last whole grating cycles, skipping the first cycle after onset.
    pre_idx = int(round(pre_stim_time / dt))
    end_idx = pre_idx + int(round(stim_time / dt))
    num_cycles = int(stim_time * temporal_frequency) - 1
    start_idx = end_idx - int(round(num_cycles / temporal_frequency / dt))
    amplitude, above = {}, {}
    for ct in available:
        x = firing[ct][start_idx:end_idx]
        x = x[:, np.isfinite(x).all(axis=0)]
        amplitude[ct] = modulation_amplitude(x, dt, temporal_frequency)
        above[ct] = float(np.mean(x > 0))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{tag}_lambda{wavelength:.0f}_tf{temporal_frequency:.1f}"

    time_axis_s = np.arange(sine_wave.shape[0]) * dt - pre_stim_time
    xt_types = [ct for ct in XT_CELL_TYPES if ct in available]
    fig, axes = plt.subplots(
        2, len(xt_types) + 1, figsize=(3.3 * (len(xt_types) + 1), 9), squeeze=False,
    )
    for row in range(2):
        panels = [("stimulus", sine_wave)] + [
            (ct, relative_voltage(ct) if row == 0 else firing[ct]) for ct in xt_types
        ]
        for idx, (name, data) in enumerate(panels):
            ax = axes[row, idx]
            if name == "stimulus" and row == 1:
                ax.axis("off")
                continue
            binned, extent = xt_by_azimuth(data, np.degrees(sample_phi))
            if name == "stimulus":
                kwargs = dict(cmap="gray")
                title = "stimulus"
            elif row == 0:
                vmax = float(np.nanmax(np.abs(binned))) or 1.0
                kwargs = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                title = f"{name}: v - threshold (mV)"
            else:
                kwargs = dict(cmap="viridis", vmin=0)
                title = f"{name}: rate, {temporal_frequency:g} Hz amp {amplitude[name]:.3g}"
            im = ax.imshow(
                binned,
                aspect="auto",
                extent=[extent[0], extent[1], time_axis_s[-1], time_axis_s[0]],
                origin="upper",
                **kwargs,
            )
            ax.axhline(0, color="k" if row == 0 and name != "stimulus" else "white",
                       linewidth=0.5, linestyle=":")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("φ (deg)")
            if idx == 0 or (row == 1 and idx == 1):
                ax.set_ylabel("Time (s)")
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)

    fig.suptitle(
        f"{tag}: rightward sine wave (λ={wavelength}°, tf={temporal_frequency} Hz, "
        f"static before t = 0)",
        fontsize=12,
    )
    plt.tight_layout()
    save_path = output_dir / f"signal_propagation_{suffix}.png"
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved {save_path}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    chains = [(ct, "ON") for ct in ON_CHAIN if ct in available] + \
             [(ct, "OFF") for ct in OFF_CHAIN if ct in available]
    x = np.arange(len(chains))
    colours = ["C1" if chain == "ON" else "C0" for _, chain in chains]
    amps = [max(amplitude[ct], 1e-4) for ct, _ in chains]
    axes[0].bar(x, amps, color=colours)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"rate modulation at {temporal_frequency:g} Hz")
    axes[1].bar(x, [100 * above[ct] for ct, _ in chains], color=colours)
    axes[1].set_ylabel("% of time above threshold")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x, [ct for ct, _ in chains], rotation=45)
    axes[0].legend(handles=[Patch(color="C1", label="ON chain: L1 → Mi1 / Tm3 → T4"),
                            Patch(color="C0", label="OFF chain: L2 / L3 → Tm1 / Tm2 / Tm4 / Tm9 → T5")],
                   fontsize=9, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title(f"{tag}: response to the rightward grating, per cell type")
    plt.tight_layout()
    save_path = output_dir / f"response_amplitude_{suffix}.png"
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved {save_path}")

    for ct, chain in chains:
        print(f"  {chain:3s} {ct:6s} amplitude {amplitude[ct]:8.4f}   above threshold {100 * above[ct]:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_folder", type=str,
        help="Path to model folder (containing model_init.pt).",
    )
    parser.add_argument("--model-file", type=str, default="model_init.pt",
                        help="Model file in the folder (default: model_init.pt, the untrained model).")
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
    parser.add_argument("--gain-scale", type=float, default=1.0,
                        help="Multiply every cell type's output strength by this (default: 1).")
    parser.add_argument("--threshold-shift", type=float, default=0.0,
                        help="Add this to every firing threshold, in mV (default: 0).")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <model_folder>/figures/signal_propagation).",
    )
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    model = nm.BaseRnn.load(model_folder / args.model_file)
    model.eval()

    tag = Path(args.model_file).stem
    with torch.no_grad():
        if args.gain_scale != 1.0:
            model.output_strength_log.add_(np.log(args.gain_scale))
            tag += f"_gain{args.gain_scale:g}"
        if args.threshold_shift != 0.0:
            if not hasattr(model, "threshold"):
                raise ValueError("--threshold-shift needs a model with firing thresholds "
                                 "(ConductanceDynamics).")
            model.threshold.add_(args.threshold_shift)
            tag += f"_thr{args.threshold_shift:+g}"

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
        tag=tag,
    )


if __name__ == "__main__":
    main()
