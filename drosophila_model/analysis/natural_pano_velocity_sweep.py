"""
Natural panoramic scene velocity sweep analysis.

Rotates filtered panoramic images (same pipeline as ``get_random_velocity_pano``)
about a cardinal axis at constant angular speed: yaw for left/right, pitch for
up/down. By default, velocity = temporal_frequency × λ (deg/s) to match
``sine_wave_sweep.py``; use ``--plot-by-velocity`` to sweep fixed deg/s instead.

Averages model outputs over all panoramas in ``all_panoramic.pt`` (optional
``--max-panos`` subset).
"""
import sys
from pathlib import Path


import argparse

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from drosophila_model.model_definition.model_types import neural_models as nm
from drosophila_model.stimulus_utilities import get_constant_velocity_pano


def _exponential_highpass(block, dt, tau):
    """Subtract a causal exponential lowpass from each column to remove DC.

    First-order IIR lowpass:  y[t] = α·y[t-1] + (1-α)·x[t],  α = exp(-dt/τ)
    Returns:  x[t] - y[t]
    """
    alpha = np.exp(-dt / tau)
    out = np.empty_like(block)
    # block shape: (batch, time, columns)
    lp = block[:, 0, :].copy()
    out[:, 0, :] = 0.0
    for t in range(1, block.shape[1]):
        lp = alpha * lp + (1.0 - alpha) * block[:, t, :]
        out[:, t, :] = block[:, t, :] - lp
    return out


def _central_ommatidium_index(sample_phi: np.ndarray, sample_theta: np.ndarray) -> int:
    """Column closest to forward viewing (phi=0, theta=pi/2) on the unit sphere."""
    phi_ref = 0.0
    theta_ref = np.pi / 2.0
    ref = np.array(
        [
            np.sin(theta_ref) * np.cos(phi_ref),
            np.sin(theta_ref) * np.sin(phi_ref),
            np.cos(theta_ref),
        ]
    )
    xyz = np.stack(
        [
            np.sin(sample_theta) * np.cos(sample_phi),
            np.sin(sample_theta) * np.sin(sample_phi),
            np.cos(sample_theta),
        ],
        axis=1,
    )
    return int(np.argmax(xyz @ ref))


def main():
    parser = argparse.ArgumentParser(
        description="Natural panorama constant-velocity sweep (cardinal rotations)."
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder (containing model.pt and config.toml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for initial panorama orientations.",
    )
    parser.add_argument(
        "--max-panos",
        type=int,
        default=None,
        help="Use only the first K panoramas (default: all).",
    )
    parser.add_argument(
        "--zero-center",
        "--zero_center",
        action="store_true",
        dest="zero_center",
        help="Map stimulus from [0, 1] to [-1, 1] via 2x - 1 (same as sine_wave_sweep --zero_center).",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        default=False,
        dest="log_scale",
        help="Log-compress raw panoramas before filtering (see get_constant_velocity_pano).",
    )
    parser.add_argument(
        "--plot-by-velocity",
        action="store_true",
        dest="plot_by_velocity",
        help="Sweep/plot velocity (deg/s) directly; default is temporal frequency (Hz) like sine_wave_sweep.",
    )
    parser.add_argument(
        "--no-center-trace",
        action="store_true",
        dest="no_center_trace",
        help="Do not overlay dashed traces for the central ommatidium column.",
    )
    parser.add_argument(
        "--highpass-tau",
        type=float,
        default=None,
        dest="highpass_tau",
        help="Time constant (seconds) for causal exponential highpass filter "
             "applied per column before the model. Removes DC offset.",
    )

    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    model = nm.BaseRnn.load(model_folder / "model.pt")

    dt = model.dt
    print(f"dt: {dt}")
    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]
    center_idx = _central_ommatidium_index(
        np.asarray(sample_phi), np.asarray(sample_theta)
    )
    print(
        f"Central ommatidium column index {center_idx} "
        f"(phi={np.degrees(sample_phi[center_idx]):.2f}°, "
        f"theta={np.degrees(sample_theta[center_idx]):.2f}°)"
    )

    plot_by_velocity = args.plot_by_velocity
    directions = ["left", "right", "up", "down"]
    wavelengths = [30, 60]
    temporal_frequencies = [0.5, 1, 2, 4, 8, 16]
    velocities = [15, 30, 60, 120, 240, 480]
    pre_stim_time = 0.5
    stim_time = 1.0

    sweep_values = velocities if plot_by_velocity else temporal_frequencies
    num_sweep = len(sweep_values)

    pano_indices = None
    if args.max_panos is not None:
        pano_indices = np.arange(args.max_panos, dtype=int)

    stim_list = []
    index_map = []
    seed_counter = 0
    per_pano_min = None
    per_pano_max = None

    for direction in directions:
        for wavelength in wavelengths:
            for si, sv in enumerate(sweep_values):
                if plot_by_velocity:
                    velocity = sv
                else:
                    velocity = sv * wavelength
                block = get_constant_velocity_pano(
                    sample_phi,
                    sample_theta,
                    direction,
                    velocity_deg_per_s=velocity,
                    dt=dt,
                    stim_time=stim_time,
                    pre_stim_time=pre_stim_time,
                    seed=args.seed + seed_counter,
                    pano_indices=pano_indices,
                    log_scale=args.log_scale,
                )
                seed_counter += 1
                p_count = block.shape[0]
                if per_pano_min is None:
                    per_pano_min = np.full(p_count, np.inf, dtype=np.float64)
                    per_pano_max = np.full(p_count, -np.inf, dtype=np.float64)
                for p in range(p_count):
                    arr = block[p]
                    per_pano_min[p] = min(per_pano_min[p], float(arr.min()))
                    per_pano_max[p] = max(per_pano_max[p], float(arr.max()))
                    stim_list.append(block[p : p + 1, :, :])
                    index_map.append((direction, wavelength, si))

    all_stim = np.concatenate(stim_list, axis=0)
    if args.zero_center:
        all_stim = 2.0 * all_stim - 1.0
    if args.highpass_tau is not None:
        print(f"Applying exponential highpass filter (τ={args.highpass_tau} s)")
        all_stim = _exponential_highpass(all_stim, dt, args.highpass_tau)

    all_stim_tensor = torch.tensor(all_stim, device=model.device, dtype=model.dtype)
    pre_stim_steps = int(round(pre_stim_time / dt))

    print(
        "Per-panorama luminance over all sweep conditions (time × photoreceptors), "
        "before global tensor concat:"
    )
    id_list = (
        np.arange(len(per_pano_min), dtype=int)
        if pano_indices is None
        else np.asarray(pano_indices, dtype=int)
    )
    for p, src_idx in enumerate(id_list):
        lo, hi = per_pano_min[p], per_pano_max[p]
        line = f"  pano batch[{p}] (source index {src_idx}): [0,1] min={lo:.6g} max={hi:.6g}"
        if args.zero_center:
            line += f"  |  after 2x-1: min={2.0 * lo - 1.0:.6g} max={2.0 * hi - 1.0:.6g}"
        print(line)
    if args.zero_center:
        g_lo, g_hi = float(all_stim.min()), float(all_stim.max())
        print(f"  Full batch after concat + scaling: min={g_lo:.6g} max={g_hi:.6g}")

    print(f"Running {all_stim_tensor.shape[0]} stimuli through the model...")

    t4_types = ["T4a", "T4b", "T4c", "T4d"]
    t5_types = ["T5a", "T5b", "T5c", "T5d"]
    other_types = ["T1", "Tm1", "Tm2", "Tm20"]
    output_neurons = t4_types + t5_types + other_types

    with torch.no_grad():
        pred = model.predict(
            all_stim_tensor,
            input_neurons=["L1", "L2", "L3"],
            output_neurons=output_neurons,
        )["outputs"]

    panos_per_condition = len(index_map) // (len(directions) * len(wavelengths) * num_sweep)
    if panos_per_condition * len(directions) * len(wavelengths) * num_sweep != len(index_map):
        raise RuntimeError("Unexpected index_map length")

    def _mean_over_panos(flat_arr):
        """flat_arr: (num_stimuli, ...) -> (num_conditions, ...) with conditions in index order."""
        out = np.zeros((len(directions), len(wavelengths), num_sweep) + flat_arr.shape[1:])
        counts = np.zeros((len(directions), len(wavelengths), num_sweep), dtype=int)
        dir_idx = {d: i for i, d in enumerate(directions)}
        wl_idx = {w: i for i, w in enumerate(wavelengths)}
        for i, (direction, wavelength, si) in enumerate(index_map):
            di, wi = dir_idx[direction], wl_idx[wavelength]
            out[di, wi, si] += flat_arr[i]
            counts[di, wi, si] += 1
        out /= np.maximum(counts[..., np.newaxis], 1) if out.ndim > 3 else np.maximum(counts, 1)
        return out, counts

    results = None
    results_center = None
    if model.has_decoder:
        decoder_out = pred["decoder"]
        decoder_stim = decoder_out[:, pre_stim_steps:, :, :]
        mean_per_stim = torch.nanmean(decoder_stim, dim=(1, 2)).cpu().numpy()
        agg, _ = _mean_over_panos(mean_per_stim)
        results = {
            directions[di]: {
                wavelengths[wi]: agg[di, wi, :, :].copy()
                for wi in range(len(wavelengths))
            }
            for di in range(len(directions))
        }
        mean_per_stim_c = torch.nanmean(
            decoder_out[:, pre_stim_steps:, center_idx, :], dim=1
        ).cpu().numpy()
        agg_c, _ = _mean_over_panos(mean_per_stim_c)
        results_center = {
            directions[di]: {
                wavelengths[wi]: agg_c[di, wi, :, :].copy()
                for wi in range(len(wavelengths))
            }
            for di in range(len(directions))
        }

    cell_results = {}
    cell_results_center = {}
    for ct in output_neurons:
        fr = pred["firing_rate"][ct][:, pre_stim_steps:, :]
        mean_per_stim = torch.nanmean(fr, dim=(1, 2)).cpu().numpy()
        agg, _ = _mean_over_panos(mean_per_stim)
        cell_results[ct] = {
            directions[di]: {
                wavelengths[wi]: agg[di, wi, :].copy()
                for wi in range(len(wavelengths))
            }
            for di in range(len(directions))
        }
        mean_per_stim_c = torch.nanmean(
            fr[:, :, center_idx], dim=1
        ).cpu().numpy()
        agg_c, _ = _mean_over_panos(mean_per_stim_c)
        cell_results_center[ct] = {
            directions[di]: {
                wavelengths[wi]: agg_c[di, wi, :].copy()
                for wi in range(len(wavelengths))
            }
            for di in range(len(directions))
        }

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

    if results is not None:
        fig = plt.figure(figsize=(12, 10))
        direction_axes = _make_direction_axes(fig)

        for direction, ax in direction_axes.items():
            for wavelength in wavelengths:
                data = results[direction][wavelength]
                ax.plot(sweep_values, data[:, 0], "o-", label=f"LR (λ={wavelength}°)")
                ax.plot(sweep_values, data[:, 1], "s--", label=f"UD (λ={wavelength}°)")
                if not args.no_center_trace and results_center is not None:
                    dc = results_center[direction][wavelength]
                    ax.plot(
                        sweep_values,
                        dc[:, 0],
                        "o:",
                        linewidth=1.5,
                        label=f"LR col{center_idx} (λ={wavelength}°)",
                    )
                    ax.plot(
                        sweep_values,
                        dc[:, 1],
                        "s:",
                        linewidth=1.5,
                        label=f"UD col{center_idx} (λ={wavelength}°)",
                    )
            _format_ax(ax, f"Direction: {direction}", "Mean decoder response")

        fig.suptitle(
            f"Decoder — Natural pano rotation {title_label} sweep "
            f"(n={panos_per_condition} pano(s); solid=mean, dotted=col {center_idx})",
            fontsize=14,
        )
        plt.tight_layout()
        save_path = fig_folder / "natural_pano_velocity_sweep.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Figure saved to {save_path}")
    else:
        print("Model has no decoder — skipping decoder figure.")

    for group_name, cell_types in [
        ("T4", t4_types),
        ("T5", t5_types),
        ("other", other_types),
    ]:
        for wavelength in wavelengths:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
            cell_axes = {ct: ax for ct, ax in zip(cell_types, axes.flat)}
            for ct, ax in cell_axes.items():
                prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                for di, direction in enumerate(directions):
                    c = prop_cycle[di % len(prop_cycle)]
                    data = cell_results[ct][direction][wavelength]
                    ax.plot(
                        sweep_values,
                        data,
                        "o-",
                        color=c,
                        label=f"{direction} (mean)",
                    )
                    if not args.no_center_trace:
                        d_c = cell_results_center[ct][direction][wavelength]
                        ax.plot(
                            sweep_values,
                            d_c,
                            "s--",
                            color=c,
                            alpha=0.85,
                            label=f"{direction} (col {center_idx})",
                        )
                _format_ax(ax, ct, "Mean firing rate")
            fig.suptitle(
                f"{group_name} — Natural pano {title_label} sweep "
                f"(λ={wavelength}°, n={panos_per_condition} pano(s); "
                f"dashed=col {center_idx})",
                fontsize=14,
            )
            plt.tight_layout()
            save_path = (
                fig_folder / f"natural_pano_velocity_sweep_{group_name.lower()}_lambda{wavelength}.png"
            )
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    main()
