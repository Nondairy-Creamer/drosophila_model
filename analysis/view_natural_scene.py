"""
Show a single natural-panorama stimulus through the HandTunedModel and plot
activity traces for each cell-type group over time.

Usage:
    python view_natural_scene.py                           # defaults
    python view_natural_scene.py --direction left --velocity 60
    python view_natural_scene.py --model models/hand_tuned/model.pt
"""
import argparse
from pathlib import Path
import sys

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import neural_models as nm

from stimulus_utilities import get_constant_velocity_pano


# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "models/hand_tuned/model.pt"
DEFAULT_DIRECTION = "right"
DEFAULT_VELOCITY = 30.0      # deg/s
DEFAULT_STIM_TIME = 3.0      # seconds of motion
DEFAULT_PRE_STIM_TIME = 0.5  # static panorama before motion
DEFAULT_PANO_INDEX = 0       # which panorama from all_panoramic.pt

INPUT_NEURONS = ["R1-6"]

def _central_column_index(sample_phi, sample_theta):
    """Column closest to forward-viewing (phi=0, theta=pi/2)."""
    ref = np.array([0.0, 1.0, 0.0])  # sin(pi/2)*cos(0), sin(pi/2)*sin(0), cos(pi/2)
    xyz = np.stack([
        np.sin(sample_theta) * np.cos(sample_phi),
        np.sin(sample_theta) * np.sin(sample_phi),
        np.cos(sample_theta),
    ], axis=1)
    return int(np.argmax(xyz @ ref))


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


CELL_TYPE_GROUPS = {
    "Photoreceptors":   ["R1-6"],
    "Lamina (L1-L3)":   ["L1", "L2", "L3"],
    "Lamina (L4-L5)":   ["L4", "L5"],
    "C2 / C3":          ["C2", "C3"],
    "T4 inputs":        ["Mi1", "Tm3", "Mi4", "Mi9"],
    "T5 inputs":        ["Tm1", "Tm2", "Tm4", "Tm9"],
    "T4":               ["T4a", "T4b", "T4c", "T4d"],
    "T5":               ["T5a", "T5b", "T5c", "T5d"],
}


def main():
    parser = argparse.ArgumentParser(
        description="View natural-scene response for the HandTunedModel."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to saved model .pt file.",
    )
    parser.add_argument(
        "--direction", type=str, default=DEFAULT_DIRECTION,
        choices=["left", "right", "up", "down"],
    )
    parser.add_argument(
        "--velocity", type=float, default=DEFAULT_VELOCITY,
        help="Rotation speed in deg/s.",
    )
    parser.add_argument(
        "--stim-time", type=float, default=DEFAULT_STIM_TIME,
        help="Duration of motion (seconds).",
    )
    parser.add_argument(
        "--pre-stim-time", type=float, default=DEFAULT_PRE_STIM_TIME,
        help="Static panorama before motion (seconds).",
    )
    parser.add_argument(
        "--pano-index", type=int, default=DEFAULT_PANO_INDEX,
        help="Index of the panorama to use from all_panoramic.pt.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--zero-center", action="store_true",
        help="Map stimulus from [0,1] to [-1,1] via 2x-1.",
    )
    parser.add_argument(
        "--log-scale", action="store_true",
        help="Log-compress raw panoramas before filtering.",
    )
    parser.add_argument(
        "--highpass-tau", type=float, default=None,
        help="Time constant (seconds) for causal exponential highpass filter "
             "applied per column before the model. Removes DC offset.",
    )
    readout_group = parser.add_mutually_exclusive_group()
    readout_group.add_argument(
        "--show-voltage", action="store_const", const="voltage", dest="readout",
        help="Plot raw voltage (pre-activation) instead of firing rate.",
    )
    readout_group.add_argument(
        "--show-synaptic-current", action="store_const", const="synaptic_current", dest="readout",
        help="Plot synaptic current (omega * W @ v, before activation). "
             "This is the signal that the activation function clips.",
    )
    parser.set_defaults(readout="firing_rate")
    parser.add_argument(
        "--columns", type=int, nargs="+", default=None,
        help="Column indices to plot as individual traces (e.g. --columns 0 50 100). "
             "Use --columns -1 for the central ommatidium.",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save figure to this path instead of showing interactively.",
    )
    parser.add_argument(
        "--hex", action="store_true",
        help="Render hex mosaic animations of the stimulus and/or a cell "
             "type's response using FlyEye.",
    )
    parser.add_argument(
        "--hex-fps", type=int, default=30,
        help="Playback FPS for hex animations (default 30).",
    )
    parser.add_argument(
        "--hex-cell-type", type=str, nargs="+", default=None,
        help="Cell type(s) whose response to animate on the hex grid "
             "(e.g. --hex-cell-type T4a T5a).  Requires --hex.",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model from {args.model} …")
    model = nm.BaseRnn.load(args.model)
    print(f"  {model.num_neurons} neurons, {model.num_cell_types} cell types, omega={model.get_omega():.4f}")
    
    sample_phi = model.column_phi_theta[:, 0]
    sample_theta = model.column_phi_theta[:, 1]
    dt = model.dt

    output_types = sorted(
        {ct for cts in CELL_TYPE_GROUPS.values() for ct in cts}
    )
    output_types = [ct for ct in output_types if ct in model.unique_cell_types]

    # ── Generate stimulus ─────────────────────────────────────────────
    print(
        f"Generating natural panorama: direction={args.direction}, "
        f"velocity={args.velocity} deg/s, pano_index={args.pano_index}"
    )
    block = get_constant_velocity_pano(
        sample_phi,
        sample_theta,
        args.direction,
        velocity_deg_per_s=args.velocity,
        dt=dt,
        stim_time=args.stim_time,
        pre_stim_time=args.pre_stim_time,
        seed=args.seed,
        pano_indices=np.array([args.pano_index]),
        log_scale=args.log_scale,
    )  # (1, T, N)

    if args.zero_center:
        block = 2.0 * block - 1.0

    if args.highpass_tau is not None:
        print(f"  Applying exponential highpass filter (τ={args.highpass_tau} s)")
        block = _exponential_highpass(block, dt, args.highpass_tau)

    stimulus = torch.tensor(block, device=model.device, dtype=model.dtype)
    time_axis = np.arange(stimulus.shape[1]) * dt - args.pre_stim_time
    print(f"  Stimulus shape: {stimulus.shape}  (batch, time, columns)")

    # ── Run model ─────────────────────────────────────────────────────
    print("Running model …")
    with torch.no_grad():
        result = model.predict(
            stimulus,
            input_neurons=INPUT_NEURONS,
            output_neurons=output_types,
        )
    outputs = result["outputs"]

    # ── Resolve column indices ────────────────────────────────────────
    show_columns = args.columns
    if show_columns is not None:
        center_idx = _central_column_index(
            np.asarray(sample_phi), np.asarray(sample_theta)
        )
        show_columns = [center_idx if c == -1 else c for c in show_columns]
        print(f"  Showing individual columns: {show_columns}")

    # ── Plot ──────────────────────────────────────────────────────────
    readout_key = args.readout
    readout_label = readout_key.replace("_", " ")

    n_groups = len(CELL_TYPE_GROUPS)
    fig, axes = plt.subplots(
        n_groups, 1, figsize=(10, 2.5 * n_groups), sharex=True, squeeze=False
    )

    for row, (group_label, cts) in enumerate(CELL_TYPE_GROUPS.items()):
        ax = axes[row, 0]
        for ct in cts:
            if ct not in outputs[readout_key]:
                continue
            trace = outputs[readout_key][ct][0]  # (time, columns)
            mean_trace = torch.nanmean(trace, dim=1).cpu().numpy()
            line, = ax.plot(time_axis, mean_trace, label=f"{ct} (mean)")
            color = line.get_color()

            if show_columns is not None:
                for ci in show_columns:
                    col_trace = trace[:, ci].cpu().numpy()
                    ax.plot(
                        time_axis, col_trace,
                        color=color, alpha=0.4, linewidth=0.8, linestyle="--",
                        label=f"{ct} col {ci}",
                    )

        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel(group_label, fontsize=9)

    axes[-1, 0].set_xlabel("Time (s)")
    col_tag = ""
    if show_columns is not None:
        col_tag = f" | cols {show_columns}"
    hp_tag = ""
    if args.highpass_tau is not None:
        hp_tag = f" | HP τ={args.highpass_tau}s"
    fig.suptitle(
        f"Natural panorama ({readout_label}) — {args.direction} @ {args.velocity} deg/s "
        f"(pano {args.pano_index}){col_tag}{hp_tag}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    # ── RMS plot ──────────────────────────────────────────────────────
    fig_rms, axes_rms = plt.subplots(
        n_groups, 1, figsize=(10, 2.5 * n_groups), sharex=True, squeeze=False
    )

    for row, (group_label, cts) in enumerate(CELL_TYPE_GROUPS.items()):
        ax = axes_rms[row, 0]
        for ct in cts:
            if ct not in outputs[readout_key]:
                continue
            trace = outputs[readout_key][ct][0]  # (time, columns)
            rms_trace = torch.sqrt(torch.nanmean(trace ** 2, dim=1)).cpu().numpy()
            ax.plot(time_axis, rms_trace, label=f"{ct}")

        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel(group_label, fontsize=9)

    axes_rms[-1, 0].set_xlabel("Time (s)")
    fig_rms.suptitle(
        f"Natural panorama (RMS {readout_label}) — {args.direction} @ {args.velocity} deg/s "
        f"(pano {args.pano_index}){hp_tag}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        rms_path = save_path.parent / (save_path.stem + "_rms" + save_path.suffix)
        fig_rms.savefig(rms_path, dpi=150)
        plt.close("all")
        print(f"Figure saved to {save_path}")
        print(f"Figure saved to {rms_path}")
    else:
        plt.show()

    # ── Hex mosaic animations ─────────────────────────────────────────
    if args.hex:
        from FlyEye.visualize_stimulus import visualize_column_responses
        import matplotlib.cm as mpl_cm

        column_p_q = model.column_p_q
        n_total = stimulus.shape[1]

        target_fps = args.hex_fps
        total_time = args.pre_stim_time + args.stim_time
        target_frames = int(total_time * target_fps)
        step = max(1, n_total // target_frames)
        frame_idx = np.arange(0, n_total, step)
        frame_times = frame_idx * dt - args.pre_stim_time

        save_dir = None
        if args.save:
            save_dir = Path(args.save).parent

        # --- Stimulus animation ---
        stim_frames = block[0, frame_idx, :]  # (F, N_columns)
        stim_titles = [f"pano {args.pano_index} {args.direction} "
                       f"{args.velocity}°/s  t={t:.3f}s" for t in frame_times]

        stim_filename = None
        if save_dir:
            stim_filename = str(
                save_dir / f"hex_stim_pano{args.pano_index}_{args.direction}_{args.velocity:.0f}dps.mp4"
            )

        cmap = mpl_cm.gray(np.linspace(0, 1, 256))
        if args.zero_center or args.highpass_tau is not None:
            cmap = mpl_cm.RdBu_r(np.linspace(0, 1, 256))

        print(f"  Hex stimulus: {len(frame_idx)} frames @ {target_fps} fps")
        visualize_column_responses(
            column_p_q, stim_frames,
            colormap=cmap, fps=target_fps,
            filename=stim_filename,
            plot=(save_dir is None),
            titles=stim_titles,
        )

        # --- Cell-type response animations ---
        if args.hex_cell_type:
            for ct in args.hex_cell_type:
                if ct not in outputs[readout_key]:
                    print(f"  Warning: cell type '{ct}' not in outputs, skipping.")
                    continue
                resp_full = outputs[readout_key][ct][0].cpu().numpy()  # (T, N_columns)
                resp_frames = resp_full[frame_idx]
                resp_titles = [f"{ct} {readout_label}  t={t:.3f}s" for t in frame_times]

                resp_filename = None
                if save_dir:
                    resp_filename = str(
                        save_dir / f"hex_{ct}_pano{args.pano_index}_{args.direction}_{args.velocity:.0f}dps.mp4"
                    )

                print(f"  Hex response: {ct} ({len(frame_idx)} frames)")
                visualize_column_responses(
                    column_p_q, resp_frames,
                    colormap=mpl_cm.coolwarm(np.linspace(0, 1, 256)),
                    fps=target_fps,
                    filename=resp_filename,
                    plot=(save_dir is None),
                    titles=resp_titles,
                )


if __name__ == "__main__":
    main()
