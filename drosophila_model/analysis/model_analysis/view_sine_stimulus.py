"""
Visualize moving sine-wave stimuli and HandTunedModel responses.

Runs all four direction conditions (right, left, up, down) through the model
and produces:
  1. Stimulus heatmaps (time x column) for each condition.
  2. Cell-type response traces (column-averaged and optionally per-column).
  3. (optional) Hex mosaic animations via FlyEye for the raw stimulus
     and/or model responses.

Usage:
    python view_sine_stimulus.py
    python view_sine_stimulus.py --model models/hand_tuned/model.pt --wavelength 60
    python view_sine_stimulus.py --columns -1 --show-voltage
    python view_sine_stimulus.py --temporal-frequency 8 --wavelength 30
    python view_sine_stimulus.py --hex --hex-cell-type T4a T5a --save-dir figures/
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.utilities.stimulus_utilities import make_lr_sine_wave, make_ud_sine_wave

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "models/hand_tuned/model.pt"
DEFAULT_WAVELENGTH = 30.0        # degrees
DEFAULT_TEMPORAL_FREQ = 4.0      # Hz
DEFAULT_STIM_TIME = 2.0          # seconds
DEFAULT_PRE_STIM_TIME = 0.5      # seconds

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
        description="Visualize moving sine-wave stimuli and model responses."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to saved model .pt file.",
    )
    parser.add_argument(
        "--wavelength", type=float, default=DEFAULT_WAVELENGTH,
        help="Spatial wavelength in degrees.",
    )
    parser.add_argument(
        "--temporal-frequency", type=float, default=DEFAULT_TEMPORAL_FREQ,
        help="Temporal frequency in Hz (positive value; sign set by direction).",
    )
    parser.add_argument(
        "--stim-time", type=float, default=DEFAULT_STIM_TIME,
        help="Duration of sine-wave motion (seconds).",
    )
    parser.add_argument(
        "--pre-stim-time", type=float, default=DEFAULT_PRE_STIM_TIME,
        help="Static period before stimulus (seconds).",
    )
    parser.add_argument(
        "--contrast", type=float, default=1.0,
        help="Sine-wave contrast (0-1).",
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
        "--direction", type=str, nargs="+", default=None,
        choices=["right", "left", "up", "down"],
        help="Which direction(s) to run.  Defaults to all four.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Save figures to this directory instead of showing interactively.",
    )
    parser.add_argument(
        "--hex", action="store_true",
        help="Render hex mosaic animations of the stimulus and model responses.",
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
    print(f"  {model.num_neurons} neurons, {model.num_cell_types} cell types, "
          f"omega={model.get_omega():.4f}")

    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]
    dt = model.dt

    output_types = sorted(
        {ct for cts in CELL_TYPE_GROUPS.values() for ct in cts}
    )
    output_types = [ct for ct in output_types if ct in model.unique_cell_types]

    center_col = _central_column_index(
        np.asarray(sample_phi), np.asarray(sample_theta)
    )

    # ── Determine directions to run ───────────────────────────────────
    all_directions = ["right", "left", "up", "down"]
    directions = args.direction if args.direction else all_directions

    # ── Generate sine-wave stimuli ────────────────────────────────────
    tf = abs(args.temporal_frequency)
    wl = args.wavelength

    stim_dict = {}
    for d in directions:
        if d in ("right", "left"):
            signed_tf = tf if d == "right" else -tf
            sw, _ = make_lr_sine_wave(
                sample_phi, sample_theta,
                contrast=args.contrast, wavelength=wl,
                temporal_frequency=signed_tf,
                dt=dt, stim_time=args.stim_time,
                pre_stim_time=args.pre_stim_time,
            )
        else:
            signed_tf = tf if d == "up" else -tf
            sw, _ = make_ud_sine_wave(
                sample_phi, sample_theta,
                contrast=args.contrast, wavelength=wl,
                temporal_frequency=signed_tf,
                dt=dt, stim_time=args.stim_time,
                pre_stim_time=args.pre_stim_time,
            )
        if args.zero_center:
            sw = 2.0 * sw - 1.0
        stim_dict[d] = sw  # (time, columns)

    cond_names = list(stim_dict.keys())
    stim_batch = np.stack([stim_dict[c] for c in cond_names], axis=0)  # (D, T, N)

    if args.highpass_tau is not None:
        print(f"  Applying exponential highpass filter (τ={args.highpass_tau} s)")
        stim_batch = _exponential_highpass(stim_batch, dt, args.highpass_tau)
        for i, cond_name in enumerate(cond_names):
            stim_dict[cond_name] = stim_batch[i]

    stim_tensor = torch.tensor(stim_batch, device=model.device, dtype=model.dtype)
    time_axis = np.arange(stim_batch.shape[1]) * dt - args.pre_stim_time

    velocity = tf * wl  # deg/s
    print(f"  {len(cond_names)} directions, λ={wl}°, tf={tf} Hz "
          f"(velocity={velocity:.1f} °/s)")
    print(f"  {stim_batch.shape[1]} timesteps, {stim_batch.shape[2]} columns")

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
        vmax_val = max(abs(img.min()), abs(img.max()))
        use_diverging = args.zero_center or args.highpass_tau is not None
        ax.imshow(
            img, aspect="auto", origin="lower",
            extent=[time_axis[0], time_axis[-1], 0, img.shape[0]],
            cmap="RdBu_r" if use_diverging else "gray",
            vmin=-vmax_val if use_diverging else 0,
            vmax=vmax_val if use_diverging else 1,
        )
        ax.set_title(cond_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ci == 0:
            ax.set_ylabel("Column (sorted by phi)")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

    fig_stim.suptitle(
        f"Sine-wave stimuli — λ={wl}°, tf={tf} Hz ({velocity:.0f} °/s)",
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
    hp_tag = ""
    if args.highpass_tau is not None:
        hp_tag = f" | HP τ={args.highpass_tau}s"
    fig_resp.suptitle(
        f"Sine-wave responses ({readout_label}) — λ={wl}°, tf={tf} Hz "
        f"({velocity:.0f} °/s){col_tag}{hp_tag}",
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

    fig_rms.suptitle(
        f"Sine-wave RMS responses ({readout_label}) — λ={wl}°, tf={tf} Hz "
        f"({velocity:.0f} °/s){col_tag}{hp_tag}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    # ── Save or show static figures ──────────────────────────────────
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stim_path = save_dir / f"sine_stimulus_lambda{wl:.0f}_tf{tf:.1f}.png"
        resp_path = save_dir / f"sine_responses_lambda{wl:.0f}_tf{tf:.1f}.png"
        rms_path = save_dir / f"sine_responses_rms_lambda{wl:.0f}_tf{tf:.1f}.png"
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
        from drosophila_model.fly_eye.visualize_stimulus import visualize_column_responses
        import matplotlib.cm as mpl_cm

        column_p_q = model.regions['optic_lobe'].column_p_q

        target_fps = args.hex_fps
        total_time = args.pre_stim_time + args.stim_time
        n_total = stim_batch.shape[1]
        target_frames = int(total_time * target_fps)
        step = max(1, n_total // target_frames)
        frame_idx = np.arange(0, n_total, step)
        frame_times = frame_idx * dt - args.pre_stim_time

        save_dir = Path(args.save_dir) if args.save_dir else None

        for cond_name in cond_names:
            ci = cond_names.index(cond_name)
            safe_name = cond_name.lower().replace(" ", "_")

            # --- Stimulus animation ---
            stim_frames = stim_dict[cond_name][frame_idx]  # (F, N)
            titles = [f"sine {cond_name} λ={wl}° tf={tf}Hz  t={t:.3f}s"
                      for t in frame_times]

            stim_filename = None
            if save_dir:
                stim_filename = str(
                    save_dir / f"sine_hex_stim_{safe_name}_lambda{wl:.0f}_tf{tf:.1f}.mp4"
                )

            use_diverging = args.zero_center or args.highpass_tau is not None
            cmap = mpl_cm.gray(np.linspace(0, 1, 256))
            if use_diverging:
                cmap = mpl_cm.RdBu_r(np.linspace(0, 1, 256))

            print(f"  Hex stimulus: {cond_name} ({len(frame_idx)} frames @ {target_fps} fps)")
            visualize_column_responses(
                column_p_q, stim_frames,
                colormap=cmap, fps=target_fps,
                filename=stim_filename,
                plot=(save_dir is None),
                titles=titles,
            )

            # --- Cell-type response animations ---
            if args.hex_cell_type:
                for ct in args.hex_cell_type:
                    if ct not in outputs[readout_key]:
                        print(f"  Warning: cell type '{ct}' not in outputs, skipping.")
                        continue
                    resp_full = outputs[readout_key][ct][ci].cpu().numpy()  # (T, N)
                    resp_frames = resp_full[frame_idx]
                    resp_titles = [f"{cond_name} — {ct} {readout_label}  t={t:.3f}s"
                                   for t in frame_times]

                    resp_filename = None
                    if save_dir:
                        resp_filename = str(
                            save_dir / f"sine_hex_{ct}_{safe_name}_lambda{wl:.0f}_tf{tf:.1f}.mp4"
                        )

                    print(f"  Hex response: {ct} / {cond_name} ({len(frame_idx)} frames)")
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
