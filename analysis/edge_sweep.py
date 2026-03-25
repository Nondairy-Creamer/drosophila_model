"""
Edge velocity sweep analysis.

Generates light and dark edges sweeping at constant velocity (left, right,
up, down) across a gray background over a range of speeds.  Plots mean
decoder response (LR and UD components) and cell firing rates as a function
of velocity for each direction and edge polarity.
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
from stimulus_utilities import make_lr_edge, make_ud_edge


def generate_stimulus(direction, sample_phi, sample_theta, velocity, edge_polarity, dt, stim_time, pre_stim_time):
    """Generate a single edge stimulus for a given direction, velocity, and polarity.

    Returns stimulus tensor of shape (1, time, columns) and the number of
    warm-up time steps (pre_stim_time frames).
    """
    direction = direction.lower()
    if direction in ("right", "left"):
        vel = velocity if direction == "right" else -velocity
        edge, _ = make_lr_edge(
            sample_phi, sample_theta,
            velocity=vel,
            edge_polarity=edge_polarity,
            dt=dt,
            stim_time=stim_time,
            pre_stim_time=pre_stim_time,
        )
    elif direction in ("up", "down"):
        vel = velocity if direction == "up" else -velocity
        edge, _ = make_ud_edge(
            sample_phi, sample_theta,
            velocity=vel,
            edge_polarity=edge_polarity,
            dt=dt,
            stim_time=stim_time,
            pre_stim_time=pre_stim_time,
        )
    else:
        raise ValueError(f"Unknown direction: {direction}")

    stimulus = edge[None, :, :]
    return stimulus


def main():
    parser = argparse.ArgumentParser(description="Edge velocity sweep analysis.")
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder (containing model.pt and config.toml)",
    )
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    model_config = du.load_config(model_folder / "config.toml")
    model = nm.NeuralModel.load(model_folder / "model.pt")

    dt = model_config["model"]["dt"]
    sample_phi = model.column_phi_theta[:, 0]
    sample_theta = model.column_phi_theta[:, 1]

    # --- parameters ---
    directions = ["left", "right", "up", "down"]
    polarities = ["light", "dark"]
    velocities = [15, 30, 60, 120, 240, 480]  # deg/s
    fov_deg = 180.0  # sweep extent (degrees)
    pre_stim_time = 0.5  # warm-up (seconds)

    num_velocities = len(velocities)
    pre_stim_steps = int(pre_stim_time / dt)

    t4_types = ["T4a", "T4b", "T4c", "T4d"]
    t5_types = ["T5a", "T5b", "T5c", "T5d"]
    other_types = ['T1', 'Tm1', 'Tm2', 'Tm20']
    output_neurons = t4_types + t5_types + other_types

    # Pre-allocate results storage
    results = None
    if model.has_decoder:
        results = {d: {p: np.zeros((num_velocities, 2)) for p in polarities} for d in directions}
    else:
        print("Model has no decoder — skipping decoder figure.")

    cell_results = {ct: {d: {p: np.zeros(num_velocities) for p in polarities}
                         for d in directions} for ct in output_neurons}

    # --- loop over velocities (each has a different stim duration) ---
    for vi, velocity in enumerate(velocities):
        stim_time = fov_deg / velocity  # seconds to cross full FOV
        print(f"Velocity {velocity} deg/s — stim_time = {stim_time:.2f}s")

        stim_list = []
        index_map = []  # (direction, polarity) for this batch
        for direction in directions:
            for polarity in polarities:
                stim_np = generate_stimulus(
                    direction, sample_phi, sample_theta,
                    velocity, polarity, dt, stim_time, pre_stim_time,
                )
                stim_list.append(stim_np)
                index_map.append((direction, polarity))

        batch_stim = np.concatenate(stim_list, axis=0)
        batch_tensor = torch.tensor(batch_stim, device=model.device, dtype=model.dtype)
        print(f"  Batch shape: {batch_tensor.shape}")

        with torch.no_grad():
            pred = model.predict(batch_tensor, input_neurons=["L1", "L2", "L3"],
                                 output_neurons=output_neurons)['outputs']

        # Decoder responses
        if results is not None:
            decoder_out = pred["decoder"][:, pre_stim_steps:, :, :]
            mean_resp = torch.nanmean(decoder_out, dim=(1, 2)).cpu().numpy()
            for i, (direction, polarity) in enumerate(index_map):
                results[direction][polarity][vi, :] = mean_resp[i]

        # Cell firing rates
        for ct in output_neurons:
            fr = pred["firing_rate"][ct][:, pre_stim_steps:, :]
            mean_fr = torch.nanmean(fr, dim=(1, 2)).cpu().numpy()
            for i, (direction, polarity) in enumerate(index_map):
                cell_results[ct][direction][polarity][vi] = mean_fr[i]

    # --- plotting ---
    fig_folder = model_folder / "figures"
    fig_folder.mkdir(parents=True, exist_ok=True)

    x_label = "Velocity (deg/s)"

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
        ax.set_xticks(velocities)
        ax.set_xticklabels([str(v) for v in velocities])

    # --- Decoder figure (both polarities together) ---
    if results is not None:
        fig = plt.figure(figsize=(12, 10))
        direction_axes = _make_direction_axes(fig)

        for direction, ax in direction_axes.items():
            for polarity in polarities:
                data = results[direction][polarity]
                ax.plot(velocities, data[:, 0], "o-", label=f"LR ({polarity})")
                ax.plot(velocities, data[:, 1], "s--", label=f"UD ({polarity})")
            _format_ax(ax, f"Direction: {direction}", "Mean decoder response")

        fig.suptitle("Decoder — Edge Velocity Sweep", fontsize=14)
        plt.tight_layout()

        save_path = fig_folder / "edge_velocity_sweep.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Figure saved to {save_path}")

    # --- T4 / T5 / other figures (one per group per polarity) ---
    # Compute shared y-axis limits per cell type across both polarities.
    cell_ylims = {}
    for ct in output_neurons:
        ymin = min(cell_results[ct][d][p][vi]
                   for d in directions for p in polarities for vi in range(num_velocities))
        ymax = max(cell_results[ct][d][p][vi]
                   for d in directions for p in polarities for vi in range(num_velocities))
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.001
        cell_ylims[ct] = (ymin - margin, ymax + margin)

    for group_name, cell_types in [("T4", t4_types), ("T5", t5_types), ("other", other_types)]:
        for polarity in polarities:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
            cell_axes = {ct: ax for ct, ax in zip(cell_types, axes.flat)}

            for ct, ax in cell_axes.items():
                for direction in directions:
                    data = cell_results[ct][direction][polarity]
                    ax.plot(velocities, data, "o-", label=direction)
                _format_ax(ax, ct, "Mean firing rate")
                ax.set_ylim(cell_ylims[ct])

            fig.suptitle(f"{group_name} — Edge Velocity Sweep ({polarity} edge)", fontsize=14)
            plt.tight_layout()

            save_path = fig_folder / f"edge_velocity_sweep_{group_name.lower()}_{polarity}.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    main()
