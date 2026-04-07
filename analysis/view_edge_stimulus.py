"""
Visualize moving-edge stimuli and the HandTunedModel responses.

Runs all four polarity × direction conditions (ON-right, ON-left, OFF-right,
OFF-left) through the model and produces:
  1. A stimulus heatmap (time × column) for each condition.
  2. Cell-type response traces (column-averaged and optionally per-column).
  3. (optional) Hex mosaic animations via FlyEye for the raw stimulus
     and/or model responses.

Usage:
    python view_edge_stimulus.py
    python view_edge_stimulus.py --model models/hand_tuned/model.pt --velocity 60
    python view_edge_stimulus.py --columns -1 --show-voltage

    # Hex mosaic animation for ON-right stimulus, saved as mp4:
    python view_edge_stimulus.py --hex --condition "ON right" --save-dir figures/
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt

import neural_models as nm
from stimulus_utilities import make_lr_moving_edge, make_ud_moving_edge

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "models/hand_tuned/model.pt"
DEFAULT_VELOCITY = 30.0      # deg/s
DEFAULT_STIM_TIME = 3.0
DEFAULT_PRE_STIM_TIME = 0.5
DEFAULT_EDGE_WIDTH = 2.5     # degrees

INPUT_NEURONS = ["R1-6"]

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


def _exponential_highpass(block, dt, tau):
    """Subtract a causal exponential lowpass from each column to remove DC.

    First-order IIR lowpass:  y[t] = α·y[t-1] + (1-α)·x[t],  α = exp(-dt/τ)
    Returns:  x[t] - y[t]
    """
    alpha = np.exp(-dt / tau)
    out = np.empty_like(block)
    lp = block[:, 0, :].copy()
    out[:, 0, :] = 0.0
    for t in range(1, block.shape[1]):
        lp = alpha * lp + (1.0 - alpha) * block[:, t, :]
        out[:, t, :] = block[:, t, :] - lp
    return out


def _central_column_index(sample_phi, sample_theta):
    ref = np.array([0.0, 1.0, 0.0])
    xyz = np.stack([
        np.sin(sample_theta) * np.cos(sample_phi),
        np.sin(sample_theta) * np.sin(sample_phi),
        np.cos(sample_theta),
    ], axis=1)
    return int(np.argmax(xyz @ ref))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize moving-edge stimuli and model responses."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to saved model .pt file.",
    )
    parser.add_argument(
        "--velocity", type=float, default=DEFAULT_VELOCITY,
        help="Edge speed in deg/s.",
    )
    parser.add_argument(
        "--stim-time", type=float, default=DEFAULT_STIM_TIME,
        help="Duration of edge motion (seconds).",
    )
    parser.add_argument(
        "--pre-stim-time", type=float, default=DEFAULT_PRE_STIM_TIME,
        help="Static period before edge enters the field (seconds).",
    )
    parser.add_argument(
        "--edge-width", type=float, default=DEFAULT_EDGE_WIDTH,
        help="Gaussian edge smoothing sigma in degrees.",
    )
    parser.add_argument(
        "--zero-center", action="store_true",
        help="Map stimulus from [0,1] to [-1,1] via 2x-1.",
    )
    parser.add_argument(
        "--highpass-tau", type=float, default=None,
        help="Time constant (seconds) for causal exponential highpass filter "
             "applied per column before the model.  Removes DC offset.",
    )
    readout_group = parser.add_mutually_exclusive_group()
    readout_group.add_argument(
        "--show-voltage", action="store_const", const="voltage", dest="readout",
    )
    readout_group.add_argument(
        "--show-synaptic-current", action="store_const", const="synaptic_current",
        dest="readout",
    )
    parser.set_defaults(readout="firing_rate")
    parser.add_argument(
        "--columns", type=int, nargs="+", default=None,
        help="Column indices for individual traces (use -1 for central column).",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Save figures to this directory instead of showing interactively.",
    )
    parser.add_argument(
        "--hex", action="store_true",
        help="Render hex mosaic animations of the stimulus and model responses "
             "using FlyEye.visualize_stimulus.",
    )
    parser.add_argument(
        "--condition", type=str, nargs="+", default=None,
        help="Which edge condition(s) to animate with --hex.  "
             "Choices: 'ON right', 'ON left', 'OFF right', 'OFF left'.  "
             "Defaults to all four.  Quote names with spaces.",
    )
    parser.add_argument(
        "--hex-fps", type=int, default=30,
        help="Playback FPS for hex animations (default 30).",
    )
    parser.add_argument(
        "--hex-cell-type", type=str, default=None,
        help="Also animate this cell type's firing rate on the hex grid "
             "(e.g. 'T4a').  Requires --hex.",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model from {args.model} …")
    model = nm.NeuralModel.load(args.model)
    print(f"  {model.num_neurons} neurons, {model.num_cell_types} cell types, "
          f"omega={model.get_omega():.4f}")

    sample_phi = model.column_phi_theta[:, 0]
    sample_theta = model.column_phi_theta[:, 1]
    dt = model.dt

    output_types = sorted(
        {ct for cts in CELL_TYPE_GROUPS.values() for ct in cts}
    )
    output_types = [ct for ct in output_types if ct in model.unique_cell_types]

    center_col = _central_column_index(
        np.asarray(sample_phi), np.asarray(sample_theta)
    )

    # ── Generate edge stimuli ─────────────────────────────────────────
    conditions = {
        "ON right":  (make_lr_moving_edge,  args.velocity, "on"),
        "ON left":   (make_lr_moving_edge, -args.velocity, "on"),
        "OFF right": (make_lr_moving_edge,  args.velocity, "off"),
        "OFF left":  (make_lr_moving_edge, -args.velocity, "off"),
    }

    stim_dict = {}
    for cond_name, (make_fn, vel, etype) in conditions.items():
        arr = make_fn(
            sample_phi, sample_theta,
            velocity=vel, edge_type=etype,
            dt=dt, stim_time=args.stim_time,
            pre_stim_time=args.pre_stim_time,
            edge_width=args.edge_width,
        )
        if args.zero_center:
            arr = 2.0 * arr - 1.0
        stim_dict[cond_name] = arr  # (time, columns)

    cond_names = list(stim_dict.keys())
    stim_batch = np.stack([stim_dict[c] for c in cond_names], axis=0)  # (4, T, N)

    if args.highpass_tau is not None:
        print(f"  Applying exponential highpass filter (τ={args.highpass_tau} s)")
        stim_batch = _exponential_highpass(stim_batch, dt, args.highpass_tau)
        # Update per-condition dict so heatmaps and hex anims show the filtered stimulus
        for i, cond_name in enumerate(cond_names):
            stim_dict[cond_name] = stim_batch[i]

    stim_tensor = torch.tensor(stim_batch, device=model.device, dtype=model.dtype)
    time_axis = np.arange(stim_batch.shape[1]) * dt - args.pre_stim_time

    print(f"  {len(cond_names)} conditions, {stim_batch.shape[1]} timesteps, "
          f"{stim_batch.shape[2]} columns")

    # ── Run model ─────────────────────────────────────────────────────
    print("Running model …")
    with torch.no_grad():
        result = model.predict(
            stim_tensor,
            input_neurons=INPUT_NEURONS,
            output_neurons=output_types,
        )
    outputs = result["outputs"]

    # ── Resolve column indices ────────────────────────────────────────
    show_columns = args.columns
    if show_columns is not None:
        show_columns = [center_col if c == -1 else c for c in show_columns]
        print(f"  Showing individual columns: {show_columns}")

    # ── Sort columns by phi for heatmap display ───────────────────────
    phi_order = np.argsort(sample_phi)

    # ── Figure 1: stimulus heatmaps ───────────────────────────────────
    n_conds = len(cond_names)
    fig_stim, axes_stim = plt.subplots(
        1, n_conds, figsize=(4 * n_conds, 4), squeeze=False
    )
    for ci, cond_name in enumerate(cond_names):
        ax = axes_stim[0, ci]
        img = stim_dict[cond_name][:, phi_order].T  # (columns_sorted, time)
        vmax = max(abs(img.min()), abs(img.max()))
        ax.imshow(
            img, aspect="auto", origin="lower",
            extent=[time_axis[0], time_axis[-1], 0, img.shape[0]],
            cmap="RdBu_r" if args.zero_center else "gray",
            vmin=-vmax if args.zero_center else 0,
            vmax=vmax if args.zero_center else 1,
        )
        ax.set_title(cond_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ci == 0:
            ax.set_ylabel("Column (sorted by phi)")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

    fig_stim.suptitle(
        f"Edge stimuli — {args.velocity:.0f} deg/s, width={args.edge_width}°",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    # ── Figure 2: model responses ─────────────────────────────────────
    readout_key = args.readout
    readout_label = readout_key.replace("_", " ")
    n_groups = len(CELL_TYPE_GROUPS)

    fig_resp, axes_resp = plt.subplots(
        n_groups, n_conds,
        figsize=(4 * n_conds, 2.5 * n_groups),
        sharex=True, squeeze=False,
    )

    for ci, cond_name in enumerate(cond_names):
        for row, (group_label, cts) in enumerate(CELL_TYPE_GROUPS.items()):
            ax = axes_resp[row, ci]
            for ct in cts:
                if ct not in outputs[readout_key]:
                    continue
                trace = outputs[readout_key][ct][ci]  # (time, columns)
                mean_trace = torch.nanmean(trace, dim=1).cpu().numpy()
                line, = ax.plot(time_axis, mean_trace, label=f"{ct}")
                color = line.get_color()

                if show_columns is not None:
                    for col_i in show_columns:
                        col_trace = trace[:, col_i].cpu().numpy()
                        ax.plot(
                            time_axis, col_trace,
                            color=color, alpha=0.4, linewidth=0.8,
                            linestyle="--",
                        )

            ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.legend(fontsize=6, loc="upper right")
            if row == 0:
                ax.set_title(cond_name, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(group_label, fontsize=9)
            if row == n_groups - 1:
                ax.set_xlabel("Time (s)")

    col_tag = ""
    if show_columns is not None:
        col_tag = f" | cols {show_columns}"
    fig_resp.suptitle(
        f"Edge responses ({readout_label}) — {args.velocity:.0f} deg/s{col_tag}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    # ── Figure 3: RMS responses ──────────────────────────────────────
    fig_rms, axes_rms = plt.subplots(
        n_groups, n_conds,
        figsize=(4 * n_conds, 2.5 * n_groups),
        sharex=True, squeeze=False,
    )

    for ci, cond_name in enumerate(cond_names):
        for row, (group_label, cts) in enumerate(CELL_TYPE_GROUPS.items()):
            ax = axes_rms[row, ci]
            for ct in cts:
                if ct not in outputs[readout_key]:
                    continue
                trace = outputs[readout_key][ct][ci]  # (time, columns)
                rms_trace = torch.sqrt(torch.nanmean(trace ** 2, dim=1)).cpu().numpy()
                ax.plot(time_axis, rms_trace, label=f"{ct}")

            ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.legend(fontsize=6, loc="upper right")
            if row == 0:
                ax.set_title(cond_name, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(group_label, fontsize=9)
            if row == n_groups - 1:
                ax.set_xlabel("Time (s)")

    col_tag = ""
    if show_columns is not None:
        col_tag = f" | cols {show_columns}"
    fig_rms.suptitle(
        f"Edge RMS responses ({readout_label}) — {args.velocity:.0f} deg/s{col_tag}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    # ── Save or show static figures ──────────────────────────────────
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stim_path = save_dir / f"edge_stimulus_{args.velocity:.0f}dps.png"
        resp_path = save_dir / f"edge_responses_{args.velocity:.0f}dps.png"
        rms_path = save_dir / f"edge_responses_rms_{args.velocity:.0f}dps.png"
        fig_stim.savefig(stim_path, dpi=150)
        fig_resp.savefig(resp_path, dpi=150)
        fig_rms.savefig(rms_path, dpi=150)
        plt.close("all")
        print(f"Saved: {stim_path}")
        print(f"Saved: {resp_path}")
        print(f"Saved: {rms_path}")
    else:
        plt.show()

    # ── Hex mosaic animations ─────────────────────────────────────────
    if args.hex:
        from FlyEye.visualize_stimulus import visualize_column_responses
        import matplotlib.cm as mpl_cm

        column_p_q = model.column_p_q
        hex_conds = args.condition if args.condition else cond_names

        # Subsample to keep animations at a reasonable frame count
        target_fps = args.hex_fps
        total_time = args.pre_stim_time + args.stim_time
        target_frames = int(total_time * target_fps)
        n_total = stim_batch.shape[1]
        step = max(1, n_total // target_frames)
        frame_idx = np.arange(0, n_total, step)
        frame_times = frame_idx * dt - args.pre_stim_time

        for cond_name in hex_conds:
            if cond_name not in stim_dict:
                print(f"  Warning: unknown condition '{cond_name}', skipping.")
                continue

            ci = cond_names.index(cond_name)
            safe_name = cond_name.lower().replace(" ", "_")

            # --- Stimulus animation ---
            stim_frames = stim_dict[cond_name][frame_idx]  # (F, N)
            titles = [f"{cond_name}  t={t:.3f}s" for t in frame_times]

            stim_filename = None
            if args.save_dir:
                stim_filename = str(save_dir / f"edge_hex_stim_{safe_name}_{args.velocity:.0f}dps.mp4")
            
            print(f"  Hex stimulus: {stim_filename}")
            cmap = mpl_cm.gray(np.linspace(0, 1, 256))
            if args.zero_center:
                cmap = mpl_cm.RdBu_r(np.linspace(0, 1, 256))

            print(f"  Hex stimulus: {cond_name} ({len(frame_idx)} frames @ {target_fps} fps)")
            visualize_column_responses(
                column_p_q, stim_frames,
                colormap=cmap, fps=target_fps,
                filename=stim_filename,
                plot=(args.save_dir is None),
                titles=titles,
            )

            # --- Cell-type response animation ---
            if args.hex_cell_type:
                ct = args.hex_cell_type
                if ct not in outputs[readout_key]:
                    print(f"  Warning: cell type '{ct}' not in outputs, skipping hex response.")
                    continue
                resp_full = outputs[readout_key][ct][ci].cpu().numpy()  # (T, N_columns)
                resp_frames = resp_full[frame_idx]
                resp_titles = [f"{cond_name} — {ct} {readout_label}  t={t:.3f}s"
                               for t in frame_times]

                resp_filename = None
                if args.save_dir:
                    resp_filename = str(
                        save_dir / f"edge_hex_{ct}_{safe_name}_{args.velocity:.0f}dps.mp4"
                    )
                print(f"  Hex response: {resp_filename}")

                print(f"  Hex response: {ct} / {cond_name} ({len(frame_idx)} frames)")
                visualize_column_responses(
                    column_p_q, resp_frames,
                    colormap=mpl_cm.coolwarm(np.linspace(0, 1, 256)),
                    fps=target_fps,
                    filename=resp_filename,
                    plot=(args.save_dir is None),
                    titles=resp_titles,
                )


if __name__ == "__main__":
    main()
