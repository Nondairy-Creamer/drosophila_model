"""
Sine wave temporal frequency sweep analysis.

Generates constant-velocity sine waves (left, right, up, down) at two
wavelengths (30 deg, 60 deg) across a range of temporal frequencies.
Velocity is scaled with wavelength so that temporal frequency is matched
across spatial frequencies. Plots mean firing rate as a function of
temporal frequency for each direction, plus XT plots at 8 Hz (firing rates and,
under ``xt_plots/``, stimulus after the model's input convolution when enabled).
"""
import sys
from pathlib import Path

# Add parent directory to path for imports

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.utilities.stimulus_utilities import make_lr_sine_wave, make_ud_sine_wave
from drosophila_model.analysis.model_analysis.analysis_utilities import input_conv_stimulus_timecourse, prepare_xt


NEURON_GROUPS = [
    ("T4", ["T4a", "T4b", "T4c", "T4d"]),
    ("T5", ["T5a", "T5b", "T5c", "T5d"]),
    ("L", ["L2", "L3", "L4", "L5"]),
    ("C_CT1", ["C2", "C3", "CT1_L", "CT1_M"]),
    ("Tm_A", ["Tm1", "Tm2", "Tm4", "Tm9"]),
    ("Mi_Tm", ["Mi1", "Tm3", "Mi4", "Mi9"]),
    ("T_other", ["T1", "T2", "T3"]),
]


def generate_stimulus(direction, sample_phi, sample_theta, wavelength, velocity, dt, stim_time, pre_stim_time, zero_center=False):
    """Generate a single sine wave stimulus for a given direction and velocity.

    Returns stimulus tensor of shape (1, time, columns) and the number of
    warm-up time steps (pre_stim_time frames).
    """
    temporal_frequency = velocity / wavelength

    direction = direction.lower()
    if direction in ("right", "left"):
        tf = temporal_frequency if direction == "right" else -temporal_frequency
        sine_wave, _ = make_lr_sine_wave(
            sample_phi, sample_theta,
            wavelength=wavelength,
            temporal_frequency=tf,
            dt=dt,
            stim_time=stim_time,
            pre_stim_time=pre_stim_time,
        )
    elif direction in ("up", "down"):
        tf = temporal_frequency if direction == "up" else -temporal_frequency
        sine_wave, _ = make_ud_sine_wave(
            sample_phi, sample_theta,
            wavelength=wavelength,
            temporal_frequency=tf,
            dt=dt,
            stim_time=stim_time,
            pre_stim_time=pre_stim_time,
        )
    else:
        raise ValueError(f"Unknown direction: {direction}")
    if zero_center:
        sine_wave = 2*sine_wave - 1
    # sine_wave shape: (time, columns) — add batch dim
    stimulus = sine_wave[None, :, :]
    return stimulus


def run_sine_wave_sweep(model, silenced_neurons=None, output_dir=None,
                        expressive=True, zero_center=False):
    """Run sine-wave temporal-frequency sweep and return per-cell mean rates.

    Args:
        model: A loaded ``BaseRnn`` (already on the desired device and
            in eval mode).
        silenced_neurons: Optional list of neuron names to silence.
            ``output_strength_log`` is saved and restored automatically.
        output_dir: Path for figures.  If *None*, plotting is skipped.
        expressive: If True, analyse all ``NEURON_GROUPS``; otherwise only
            T4 and T5.

    Returns:
        dict mapping ``cell_type -> direction -> wavelength -> np.ndarray``
        of mean firing rates across temporal frequencies.
    """
    original_output_strength = None
    original_input_strength = None
    if silenced_neurons:
        original_output_strength = model.output_strength_log.data.clone()
        original_input_strength = model.input_strength_log.data.clone()
        model.silence_neurons(silenced_neurons)

    try:
        return _run_sine_wave_sweep_inner(model, output_dir, expressive, zero_center)
    finally:
        if original_output_strength is not None:
            model.output_strength_log.data.copy_(original_output_strength)
            model.input_strength_log.data.copy_(original_input_strength)


def _run_sine_wave_sweep_inner(model, output_dir, expressive, zero_center=False):
    """Core sweep loop (called by :func:`run_sine_wave_sweep`)."""

    dt = model.dt
    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]

    # --- parameters ---
    directions = ["left", "right", "up", "down"]
    wavelengths = [30, 60]  # degrees
    temporal_frequencies = [0.5, 1, 2, 4, 8, 16]  # Hz
    pre_stim_time = 0.5  # warm-up (seconds)
    stim_time = 1.0  # stimulus duration (seconds)

    stim_dt = model.dt

    pre_stim_steps = int(pre_stim_time / dt)
    num_sweep = len(temporal_frequencies)

    # Select neuron groups to plot
    T4_T5_GROUPS = [g for g in NEURON_GROUPS if g[0] in ("T4", "T5")]
    active_groups = NEURON_GROUPS if expressive else T4_T5_GROUPS

    # Collect unique neuron types across active groups present in the model
    available_types = set(model.unique_cell_types)
    all_neurons = []
    for _, types in active_groups:
        for ct in types:
            if ct not in all_neurons and ct in available_types:
                all_neurons.append(ct)

    # --- build all stimuli ---
    stim_list = []
    index_map = []  # (direction, wavelength, sweep_index) per batch entry

    for direction in directions:
        for wavelength in wavelengths:
            for si, tf in enumerate(temporal_frequencies):
                velocity = tf * wavelength
                stim_np = generate_stimulus(
                    direction, sample_phi, sample_theta,
                    wavelength, velocity, stim_dt, stim_time, pre_stim_time,
                    zero_center=zero_center,
                )
                stim_list.append(stim_np)
                index_map.append((direction, wavelength, si))

    all_stim = np.concatenate(stim_list, axis=0)
    all_stim_tensor = torch.tensor(all_stim, device=model.device, dtype=model.dtype)
    print(f"Running {all_stim_tensor.shape[0]} stimuli through the model...")

    with torch.no_grad():
        pred = model.predict(all_stim_tensor, input_neurons=["L1", "L2", "L3"],
                             output_neurons=all_neurons)['outputs']

    # --- extract mean decoder responses (if model has a decoder) ---
    decoder_results = None
    decoder_key = next((k for k, m in model._readout_modules.items() if m.predicts_flow), None)
    if decoder_key is not None:
        decoder_out = pred[decoder_key]
        decoder_stim = decoder_out[:, pre_stim_steps:, :, :]
        mean_responses = torch.nanmean(decoder_stim, dim=(1, 2)).cpu().numpy()

        if mean_responses.shape[-1] == 4:
            mean_responses = np.stack([
                mean_responses[:, 0] - mean_responses[:, 1],
                mean_responses[:, 2] - mean_responses[:, 3],
            ], axis=-1)

        decoder_results = {d: {w: np.zeros((num_sweep, 2)) for w in wavelengths} for d in directions}
        for i, (direction, wavelength, si) in enumerate(index_map):
            decoder_results[direction][wavelength][si, :] = mean_responses[i]
    else:
        print("Model has no decoder — skipping decoder figure.")

    # --- extract per-cell-type mean firing rates ---
    cell_results = {}
    for ct in all_neurons:
        fr = pred["firing_rate"][ct][:, pre_stim_steps:, :]  # (N, stim_steps, columns)
        mean_fr = torch.nanmean(fr, dim=(1, 2)).cpu().numpy()  # (N,)
        cell_results[ct] = {d: {w: np.zeros(num_sweep) for w in wavelengths} for d in directions}
        for i, (direction, wavelength, si) in enumerate(index_map):
            cell_results[ct][direction][wavelength][si] = mean_fr[i]

    # --- plotting (skip when no output_dir) ---
    if output_dir is not None:
        sweep_folder = Path(output_dir)
        sweep_folder.mkdir(parents=True, exist_ok=True)

        x_label = "Temporal frequency (Hz)"
        title_label = "Temporal Frequency"
        sweep_values = temporal_frequencies

        def _format_ax(ax, title, ylabel):
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.legend(fontsize=8)
            ax.set_xscale("log")
            ax.set_xticks(sweep_values)
            ax.set_xticklabels([str(v) for v in sweep_values])

        # --- Decoder figure ---
        if decoder_results is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
            dir_axes = {"right": axes[0, 0], "left": axes[0, 1],
                        "up": axes[1, 0], "down": axes[1, 1]}

            for direction, ax in dir_axes.items():
                for wavelength in wavelengths:
                    data = decoder_results[direction][wavelength]
                    ax.plot(sweep_values, data[:, 0], "o-", label=f"LR (λ={wavelength}°)")
                    ax.plot(sweep_values, data[:, 1], "s--", label=f"UD (λ={wavelength}°)")
                _format_ax(ax, f"Direction: {direction}", "Mean decoder response")

            fig.suptitle(f"Decoder — Sine Wave {title_label} Sweep", fontsize=14)
            plt.tight_layout()
            save_path = sweep_folder / "decoder_sweep.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Saved {save_path}")

        # --- Per-group velocity sweep figures ---
        for group_name, cell_types in active_groups:
            present = [ct for ct in cell_types if ct in cell_results]
            if not present:
                print(f"Skipping group {group_name} — no matching neurons in model.")
                continue

            for wavelength in wavelengths:
                nplots = len(present)
                ncols = min(nplots, 2)
                nrows = (nplots + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                                         sharex=True, squeeze=False)

                for idx, ct in enumerate(present):
                    row, col = divmod(idx, ncols)
                    ax = axes[row, col]
                    for direction in directions:
                        data = cell_results[ct][direction][wavelength]
                        ax.plot(sweep_values, data, "o-", label=direction)
                    _format_ax(ax, ct, "Mean firing rate")

                for idx in range(nplots, nrows * ncols):
                    row, col = divmod(idx, ncols)
                    axes[row, col].axis("off")

                fig.suptitle(f"{group_name} — {title_label} Sweep (λ={wavelength}°)", fontsize=14)
                plt.tight_layout()
                save_path = sweep_folder / f"{group_name.lower()}_lambda{wavelength}.png"
                fig.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"Saved {save_path}")

        # --- XT plots at 8 Hz ---
        xt_folder = sweep_folder / "xt_plots"
        xt_folder.mkdir(parents=True, exist_ok=True)

        tf_8hz = 8
        if tf_8hz not in temporal_frequencies:
            print("8 Hz not in sweep — skipping XT plots.")
        else:
            si_8hz = temporal_frequencies.index(tf_8hz)
            batch_lookup = {}
            for i, (direction, wavelength, si) in enumerate(index_map):
                if si == si_8hz:
                    batch_lookup[(direction, wavelength)] = i

            conv_inputs = input_conv_stimulus_timecourse(model, all_stim_tensor)
            use_conv = getattr(model.regions['optic_lobe'], "use_input_conv", False)
            stim_label = "Input-conv stimulus" if use_conv else "Raw stimulus (no input conv)"

            for wavelength in wavelengths:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                dir_axes = {"right": axes[0, 0], "left": axes[0, 1],
                            "up": axes[1, 0], "down": axes[1, 1]}
                for direction, ax in dir_axes.items():
                    bi = batch_lookup[(direction, wavelength)]
                    stim_np = conv_inputs[bi].detach().cpu().numpy()
                    sorted_data, extent = prepare_xt(stim_np, sample_phi)
                    im = ax.imshow(sorted_data, aspect="auto",
                                   extent=[extent[0], extent[1], extent[3], extent[2]],
                                   cmap="viridis", origin="upper")
                    ax.set_title(direction)
                    ax.set_xlabel("φ (rad)")
                    ax.set_ylabel("Time step")
                    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)
                fig.suptitle(f"{stim_label} — 8 Hz, λ={wavelength}°", fontsize=14)
                plt.tight_layout()
                save_path = xt_folder / f"stimulus_8hz_lambda{wavelength}_input_conv_xt.png"
                fig.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"Saved {save_path}")

            xt_wavelength = wavelengths[0]
            for ct in all_neurons:
                fr_all = pred["firing_rate"][ct]  # (N, time, columns)
                for wavelength in [xt_wavelength]:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    dir_axes = {"right": axes[0, 0], "left": axes[0, 1],
                                "up": axes[1, 0], "down": axes[1, 1]}

                    for direction, ax in dir_axes.items():
                        bi = batch_lookup[(direction, wavelength)]
                        fr_np = fr_all[bi].cpu().numpy()  # (time, columns)
                        sorted_data, extent = prepare_xt(fr_np, sample_phi)
                        im = ax.imshow(sorted_data, aspect="auto",
                                       extent=[extent[0], extent[1], extent[3], extent[2]],
                                       cmap="viridis", origin="upper")
                        ax.set_title(direction)
                        ax.set_xlabel("φ (rad)")
                        ax.set_ylabel("Time step")
                        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)

                    fig.suptitle(f"{ct} — 8 Hz, λ={wavelength}°", fontsize=14)
                    plt.tight_layout()
                    save_path = xt_folder / f"{ct}_lambda{wavelength}_8hz.png"
                    fig.savefig(save_path, dpi=150)
                    plt.close(fig)
                    print(f"Saved {save_path}")

    return cell_results


def main():
    parser = argparse.ArgumentParser(description="Sine wave velocity sweep analysis.")
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder (containing model.pt and config.toml)",
    )
    parser.add_argument(
        "--expressive", action="store_true", default=False,
        help="Plot all neuron groups (default: only T4 and T5)",
    )
    parser.add_argument(
        "--silenced-neurons", type=str, nargs="*", default=None,
        help="Neuron types to silence before analysis",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <model_folder>/figures/sine_sweep)",
    )
    parser.add_argument(
        "--zero-center", action="store_true",
        help="Zero-center the stimulus (map [0, 1] -> [-1, 1]).",
    )
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    model = nm.BaseRnn.load(model_folder / "model.pt")
    model.eval()

    output_dir = (Path(args.output_dir) if args.output_dir
                  else model_folder / "figures" / "sine_sweep")

    cell_results = run_sine_wave_sweep(
        model,
        silenced_neurons=args.silenced_neurons,
        output_dir=output_dir,
        expressive=args.expressive,
        zero_center=args.zero_center,
    )

    print(f"\nDone! Computed sweep for {len(cell_results)} cell types.")


if __name__ == "__main__":
    main()
