"""
Sine wave temporal frequency sweep analysis.

Generates constant-velocity sine waves (left, right, up, down) at two
wavelengths (30 deg, 90 deg) across a range of temporal frequencies.
Velocity is scaled with wavelength so that temporal frequency is matched
across spatial frequencies. Plots mean decoder response (LR and UD
components) as a function of temporal frequency for each direction.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import data_utilities as du
import neural_models as nm
from stimulus_utilities import make_lr_sine_wave, make_ud_sine_wave


def generate_stimulus(direction, sample_phi, sample_theta, wavelength, velocity, dt, stim_time, pre_stim_time):
    """Generate a single sine wave stimulus for a given direction and velocity.

    Returns stimulus tensor of shape (1, time, columns) and the number of
    warm-up time steps (pre_stim_time frames).
    """
    # temporal_frequency = velocity / wavelength  (both in degrees)
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

    # sine_wave shape: (time, columns) — add batch dim
    stimulus = sine_wave[None, :, :]
    return stimulus


def main():
    parser = argparse.ArgumentParser(description="Sine wave velocity sweep analysis.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model folder (containing model.pt and config.toml)",
    )
    args = parser.parse_args()

    model_folder = Path(args.model_path)
    model_config = du.load_config(model_folder / "config.toml")
    computer_config = du.load_config()
    device = torch.device(computer_config["model"]["device"])
    model = nm.NeuralModel.load(model_folder / "model.pt", device=device)

    dt = model_config["model"]["dt"]
    sample_phi = model.column_phi_theta[:, 0]
    sample_theta = model.column_phi_theta[:, 1]

    # --- parameters ---
    plot_by_velocity = False  # True: sweep & plot velocity; False: sweep & plot temporal frequency

    directions = ["left", "right", "up", "down"]
    wavelengths = [30, 60]  # degrees
    temporal_frequencies = [0.5, 1, 2, 4, 8, 16]  # Hz
    velocities = [15, 30, 60, 120, 240, 480]  # deg/s
    pre_stim_time = 0.5  # warm-up (seconds)
    stim_time = 1.0  # stimulus duration (seconds)

    pre_stim_steps = int(pre_stim_time / dt)

    # Build the list of sweep values.  When sweeping by velocity the same
    # velocities are used for every wavelength; when sweeping by temporal
    # frequency the velocity is scaled so that tf is matched across wavelengths.
    sweep_values = velocities if plot_by_velocity else temporal_frequencies
    num_sweep = len(sweep_values)

    # --- build all stimuli and track their (direction, wavelength, sweep_index) indices ---
    stim_list = []
    index_map = []  # (direction, wavelength, sweep_index) per batch entry

    for direction in directions:
        for wavelength in wavelengths:
            for si, sv in enumerate(sweep_values):
                if plot_by_velocity:
                    velocity = sv  # velocity is the sweep variable
                else:
                    velocity = sv * wavelength  # velocity = tf * wavelength
                stim_np = generate_stimulus(
                    direction, sample_phi, sample_theta,
                    wavelength, velocity, dt, stim_time, pre_stim_time,
                )
                stim_list.append(stim_np)
                index_map.append((direction, wavelength, si))

    # Concatenate into one batch: (N, time, columns)
    all_stim = np.concatenate(stim_list, axis=0)
    all_stim_tensor = torch.tensor(all_stim, device=model.device, dtype=model.dtype)
    print(f"Running {all_stim_tensor.shape[0]} stimuli through the model...")

    t4_types = ["T4a", "T4b", "T4c", "T4d"]
    t5_types = ["T5a", "T5b", "T5c", "T5d"]
    other_types = ['T1', 'Tm1', 'Tm2', 'Tm20']
    output_neurons = t4_types + t5_types + other_types

    with torch.no_grad():
        pred = model.predict(all_stim_tensor, input_neurons=["L1", "L2", "L3"],
                             output_neurons=output_neurons)[0]
        decoder_out = pred["decoder"]  # (N, time, columns, 2)

    # --- extract mean responses ---
    # Mean decoder output over the stimulus period (exclude warm-up) and over columns.
    decoder_stim = decoder_out[:, pre_stim_steps:, :, :]  # (N, stim_steps, columns, 2)
    mean_responses = torch.nanmean(decoder_stim, dim=(1, 2)).cpu().numpy()  # (N, 2)

    # Decoder results: results[direction][wavelength] = (num_sweep, 2)
    results = {d: {w: np.zeros((num_sweep, 2)) for w in wavelengths} for d in directions}
    for i, (direction, wavelength, si) in enumerate(index_map):
        results[direction][wavelength][si, :] = mean_responses[i]

    # T4/T5 results: cell_results[cell_type][direction][wavelength] = (num_sweep,)
    cell_results = {}
    for ct in output_neurons:
        fr = pred["firing_rate"][ct][:, pre_stim_steps:, :]  # (N, stim_steps, columns)
        mean_fr = torch.nanmean(fr, dim=(1, 2)).cpu().numpy()  # (N,)
        cell_results[ct] = {d: {w: np.zeros(num_sweep) for w in wavelengths} for d in directions}
        for i, (direction, wavelength, si) in enumerate(index_map):
            cell_results[ct][direction][wavelength][si] = mean_fr[i]

    # --- plotting ---
    fig_folder = model_folder / "figures"
    fig_folder.mkdir(parents=True, exist_ok=True)

    x_label = "Velocity (deg/s)" if plot_by_velocity else "Temporal frequency (Hz)"
    title_label = "Velocity" if plot_by_velocity else "Temporal Frequency"

    def _make_direction_axes(fig):
        axes = fig.subplots(2, 2, sharex=True)
        return {
            "right": axes[0, 0],
            "left": axes[0, 1],
            "up": axes[1, 0],
            "down": axes[1, 1],
        }

    def _format_ax(ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(sweep_values)
        ax.set_xticklabels([str(v) for v in sweep_values])

    # --- Decoder figure (all wavelengths together) ---
    fig = plt.figure(figsize=(12, 10))
    direction_axes = _make_direction_axes(fig)

    for direction, ax in direction_axes.items():
        for wavelength in wavelengths:
            data = results[direction][wavelength]
            ax.plot(sweep_values, data[:, 0], "o-", label=f"LR (λ={wavelength}°)")
            ax.plot(sweep_values, data[:, 1], "s--", label=f"UD (λ={wavelength}°)")
        _format_ax(ax, f"Direction: {direction}", "Mean decoder response")

    fig.suptitle(f"Decoder — Sine Wave {title_label} Sweep", fontsize=14)
    plt.tight_layout()

    save_path = fig_folder / "sine_wave_velocity_sweep.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {save_path}")

    # --- T4 and T5 figures (one per group per wavelength) ---
    # Each subplot is a cell type (a–d); lines are directions.
    for group_name, cell_types in [("T4", t4_types), ("T5", t5_types), ("other", other_types)]:
        for wavelength in wavelengths:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
            cell_axes = {ct: ax for ct, ax in zip(cell_types, axes.flat)}

            for ct, ax in cell_axes.items():
                for direction in directions:
                    data = cell_results[ct][direction][wavelength]
                    ax.plot(sweep_values, data, "o-", label=direction)
                _format_ax(ax, ct, "Mean firing rate")

            fig.suptitle(f"{group_name} — {title_label} Sweep (λ={wavelength}°)", fontsize=14)
            plt.tight_layout()

            save_path = fig_folder / f"sine_wave_velocity_sweep_{group_name.lower()}_lambda{wavelength}.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    main()
