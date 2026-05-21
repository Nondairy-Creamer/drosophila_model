"""
Edge velocity sweep analysis.

Generates light and dark edges sweeping at constant velocity (left, right,
up, down) across a gray background over a range of speeds.  Plots mean
decoder response (LR and UD components) and cell firing rates as a function
of velocity for each direction and edge polarity.

Run with ``--expressive`` to match :mod:`sine_wave_sweep` neuron coverage (all
groups below).  Without it, only T4 and T5 subtypes are included in the sweep
and XT outputs.

With ``--expressive``, plots the same neuron groups as :mod:`sine_wave_sweep`
(T4, T5, L, C_CT1, Tm_A, Mi_Tm, T_other) when those types exist in the model.
Also writes space–time (XT) plots at a fixed edge speed (default 240°/s) for
both light and dark edges: one PNG per cell type under ``edge_sweep/xt_plots/``
per polarity, plus stimulus XTs: raw drive before input convolution
(``stimulus_edge_*_{light|dark}_raw_xt.png``), and after input convolution when
the model uses it (``stimulus_edge_*_{light|dark}_input_conv_xt.png``).
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

from drosophila_model import data_utilities as du
from drosophila_model.model_definition.model_types import neural_models as nm
from drosophila_model.stimulus_utilities import make_lr_edge, make_ud_edge
from analysis_utilities import input_conv_stimulus_timecourse, prepare_xt

# Same groups as sine_wave_sweep.NEURON_GROUPS (keep in sync)
NEURON_GROUPS = [
    ("T4", ["T4a", "T4b", "T4c", "T4d"]),
    ("T5", ["T5a", "T5b", "T5c", "T5d"]),
    ("L", ["L2", "L3", "L4", "L5"]),
    ("C_CT1", ["C2", "C3", "CT1_L", "CT1_M"]),
    ("Tm_A", ["Tm1", "Tm2", "Tm4", "Tm9"]),
    ("Mi_Tm", ["Mi1", "Tm3", "Mi4", "Mi9"]),
    ("T_other", ["T1", "T2", "T3"]),
]

# XT plots: match a representative speed from the sweep (or nearest); both polarities
XT_EDGE_VELOCITY_DEG_S = 240.0


def generate_stimulus(direction, sample_phi, sample_theta, velocity, edge_polarity, dt, stim_time, pre_stim_time):
    """Generate a single edge stimulus for a given direction, velocity, and polarity.

    Returns stimulus tensor of shape (1, time, columns) and the number of
    warm-up time steps (pre_stim_time frames).
    """
    direction = direction.lower()
    if direction in ("right", "left"):
        vel = velocity if direction == "right" else -velocity
        edge, _ = make_lr_edge(
            sample_phi,
            sample_theta,
            velocity=vel,
            edge_polarity=edge_polarity,
            dt=dt,
            stim_time=stim_time,
            pre_stim_time=pre_stim_time,
        )
    elif direction in ("up", "down"):
        vel = velocity if direction == "up" else -velocity
        edge, _ = make_ud_edge(
            sample_phi,
            sample_theta,
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


def _single_velocity_batch_index(direction: str, polarity: str, directions, polarities) -> int:
    """Index 0..N-1 within one velocity block (order: nested directions × polarities)."""
    j = 0
    for d in directions:
        for p in polarities:
            if d == direction and p == polarity:
                return j
            j += 1
    raise ValueError(f"Unknown (direction, polarity)=({direction}, {polarity})")


def main():
    parser = argparse.ArgumentParser(description="Edge velocity sweep analysis.")
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder (containing model.pt and config.toml)",
    )
    parser.add_argument(
        "--expressive",
        action="store_true",
        default=False,
        help="Plot all neuron groups and XT plots for every included cell type (default: only T4 and T5).",
    )
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    model_config = du.load_config(model_folder / "config.toml")
    model = nm.BaseRnn.load(model_folder / "model.pt")

    dt = model_config["model"]["dt"]
    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]

    # --- parameters ---
    directions = ["left", "right", "up", "down"]
    polarities = ["light", "dark"]
    velocities = [15, 30, 60, 120, 240, 480, 960, 1920]  # deg/s
    fov_deg = 180.0  # sweep extent (degrees)
    pre_stim_time = 0.5  # warm-up (seconds)

    num_velocities = len(velocities)
    pre_stim_steps = int(pre_stim_time / dt)

    T4_T5_GROUPS = [g for g in NEURON_GROUPS if g[0] in ("T4", "T5")]
    active_groups = NEURON_GROUPS if args.expressive else T4_T5_GROUPS

    available_types = set(model.unique_cell_types)
    all_neurons = []
    for _, types in active_groups:
        for ct in types:
            if ct not in all_neurons and ct in available_types:
                all_neurons.append(ct)

    if not all_neurons:
        print("No output cell types from selected groups — check model.unique_cell_types.")
        return

    output_neurons = all_neurons

    # Pre-allocate results storage
    results = None
    decoder_key = next((k for k, m in model._readout_modules.items() if m.predicts_flow), None)
    if decoder_key is not None:
        results = {d: {p: np.zeros((num_velocities, 2)) for p in polarities} for d in directions}
    else:
        print("Model has no decoder — skipping decoder figure.")

    cell_results = {ct: {d: {p: np.zeros(num_velocities) for p in polarities} for d in directions} for ct in output_neurons}

    # --- loop over velocities (each has a different stim duration) ---
    for vi, velocity in enumerate(velocities):
        stim_time = fov_deg / velocity  # seconds to cross full FOV
        print(f"Velocity {velocity} deg/s — stim_time = {stim_time:.2f}s")

        stim_list = []
        index_map = []  # (direction, polarity) for this batch
        for direction in directions:
            for polarity in polarities:
                stim_np = generate_stimulus(
                    direction,
                    sample_phi,
                    sample_theta,
                    velocity,
                    polarity,
                    dt,
                    stim_time,
                    pre_stim_time,
                )
                stim_list.append(stim_np)
                index_map.append((direction, polarity))

        batch_stim = np.concatenate(stim_list, axis=0)
        batch_tensor = torch.tensor(batch_stim, device=model.device, dtype=model.dtype)
        print(f"  Batch shape: {batch_tensor.shape}")

        with torch.no_grad():
            pred = model.predict(
                batch_tensor,
                input_neurons=["L1", "L2", "L3"],
                output_neurons=output_neurons,
            )["outputs"]

        # Decoder responses (baseline-subtracted)
        if results is not None:
            decoder_baseline = torch.nanmean(pred[decoder_key][:, :pre_stim_steps, :, :], dim=(1, 2))
            decoder_stim = torch.nanmean(pred[decoder_key][:, pre_stim_steps:, :, :], dim=(1, 2))
            mean_resp = (decoder_stim - decoder_baseline).cpu().numpy()
            for i, (direction, polarity) in enumerate(index_map):
                results[direction][polarity][vi, :] = mean_resp[i]

        # Cell firing rates (baseline-subtracted)
        for ct in output_neurons:
            fr = pred["firing_rate"][ct]
            baseline = torch.nanmean(fr[:, :pre_stim_steps, :], dim=(1, 2))
            stim = torch.nanmean(fr[:, pre_stim_steps:, :], dim=(1, 2))
            mean_fr = (stim - baseline).cpu().numpy()
            for i, (direction, polarity) in enumerate(index_map):
                cell_results[ct][direction][polarity][vi] = mean_fr[i]

    # --- plotting ---
    fig_folder = model_folder / "figures" / "edge_sweep"
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

    # --- Per-group velocity sweep figures (same grouping as sine_wave_sweep) ---
    cell_ylims = {}
    for ct in output_neurons:
        ymin = min(
            cell_results[ct][d][p][vi]
            for d in directions
            for p in polarities
            for vi in range(num_velocities)
        )
        ymax = max(
            cell_results[ct][d][p][vi]
            for d in directions
            for p in polarities
            for vi in range(num_velocities)
        )
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.001
        cell_ylims[ct] = (ymin - margin, ymax + margin)

    for group_name, cell_types in active_groups:
        present = [ct for ct in cell_types if ct in cell_results]
        if not present:
            print(f"Skipping group {group_name} — no matching neurons in model.")
            continue

        for polarity in polarities:
            nplots = len(present)
            ncols = min(nplots, 2)
            nrows = (nplots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, squeeze=False)

            for idx, ct in enumerate(present):
                row, col = divmod(idx, ncols)
                ax = axes[row, col]
                for direction in directions:
                    data = cell_results[ct][direction][polarity]
                    ax.plot(velocities, data, "o-", label=direction)
                _format_ax(ax, ct, "Mean firing rate")
                ax.set_ylim(cell_ylims[ct])

            for idx in range(nplots, nrows * ncols):
                row, col = divmod(idx, ncols)
                axes[row, col].axis("off")

            fig.suptitle(f"{group_name} — Edge Velocity Sweep ({polarity} edge)", fontsize=14)
            plt.tight_layout()

            save_path = fig_folder / f"edge_velocity_sweep_{group_name.lower()}_{polarity}.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Figure saved to {save_path}")

    # --- XT plots at fixed edge speed (reuse batch from last velocity loop — re-run one velocity) ---
    xt_v = XT_EDGE_VELOCITY_DEG_S
    if xt_v not in velocities:
        xt_v = min(velocities, key=lambda v: abs(v - XT_EDGE_VELOCITY_DEG_S))
        print(f"XT: {XT_EDGE_VELOCITY_DEG_S:g} deg/s not in sweep — using nearest {xt_v:g} deg/s.")
    stim_time_xt = fov_deg / xt_v

    stim_list_xt = []
    for direction in directions:
        for polarity in polarities:
            stim_np = generate_stimulus(
                direction,
                sample_phi,
                sample_theta,
                xt_v,
                polarity,
                dt,
                stim_time_xt,
                pre_stim_time,
            )
            stim_list_xt.append(stim_np)
    batch_xt = np.concatenate(stim_list_xt, axis=0)
    batch_xt_tensor = torch.tensor(batch_xt, device=model.device, dtype=model.dtype)

    with torch.no_grad():
        pred_xt = model.predict(
            batch_xt_tensor,
            input_neurons=["L1", "L2", "L3"],
            output_neurons=output_neurons,
        )["outputs"]

    conv_xt = input_conv_stimulus_timecourse(model, batch_xt_tensor)
    use_conv = getattr(model.regions['optic_lobe'], "use_input_conv", False)

    xt_folder = fig_folder / "xt_plots"
    xt_folder.mkdir(parents=True, exist_ok=True)

    batch_xt_np = np.asarray(batch_xt)
    conv_xt_np = conv_xt.detach().cpu().numpy()

    def _save_stimulus_xt_figure(data_np: np.ndarray, polarity: str, suptitle: str, path: Path) -> None:
        fig_stim, axes_stim = plt.subplots(2, 2, figsize=(12, 10))
        dir_axes_stim = {"right": axes_stim[0, 0], "left": axes_stim[0, 1],
                         "up": axes_stim[1, 0], "down": axes_stim[1, 1]}
        for direction, ax in dir_axes_stim.items():
            bi = _single_velocity_batch_index(direction, polarity, directions, polarities)
            stim_np = data_np[bi]
            sorted_data, extent = prepare_xt(stim_np, sample_phi)
            im = ax.imshow(
                sorted_data,
                aspect="auto",
                extent=[extent[0], extent[1], extent[3], extent[2]],
                cmap="viridis",
                origin="upper",
            )
            ax.set_title(direction)
            ax.set_xlabel("φ (rad)")
            ax.set_ylabel("Time step")
            fig_stim.colorbar(im, ax=ax, shrink=0.7, pad=0.04)
        fig_stim.suptitle(suptitle, fontsize=14)
        plt.tight_layout()
        fig_stim.savefig(path, dpi=150)
        plt.close(fig_stim)
        print(f"Saved {path}")

    for xt_polarity in polarities:
        _save_stimulus_xt_figure(
            batch_xt_np,
            xt_polarity,
            f"Raw stimulus (before input convolution) — edge {xt_v:g}°/s, {xt_polarity} (XT)",
            xt_folder / f"stimulus_edge_{int(xt_v)}deg_s_{xt_polarity}_raw_xt.png",
        )
        if use_conv:
            _save_stimulus_xt_figure(
                conv_xt_np,
                xt_polarity,
                f"Input-conv stimulus — edge {xt_v:g}°/s, {xt_polarity} (XT)",
                xt_folder / f"stimulus_edge_{int(xt_v)}deg_s_{xt_polarity}_input_conv_xt.png",
            )

        for ct in output_neurons:
            fr_all = pred_xt["firing_rate"][ct]
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            dir_axes = {"right": axes[0, 0], "left": axes[0, 1], "up": axes[1, 0], "down": axes[1, 1]}

            for direction, ax in dir_axes.items():
                bi = _single_velocity_batch_index(direction, xt_polarity, directions, polarities)
                fr_np = fr_all[bi].cpu().numpy()
                sorted_data, extent = prepare_xt(fr_np, sample_phi)
                im = ax.imshow(
                    sorted_data,
                    aspect="auto",
                    extent=[extent[0], extent[1], extent[3], extent[2]],
                    cmap="viridis",
                    origin="upper",
                )
                ax.set_title(direction)
                ax.set_xlabel("φ (rad)")
                ax.set_ylabel("Time step")
                fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)

            fig.suptitle(
                f"{ct} — edge {xt_v:g}°/s, {xt_polarity} edge (XT)",
                fontsize=14,
            )
            plt.tight_layout()
            save_path = xt_folder / f"{ct}_velocity{int(xt_v)}deg_s_{xt_polarity}_xt.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
