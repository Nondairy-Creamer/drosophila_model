"""
Visualize moving sine-wave stimuli and a model's responses.

Runs all four direction conditions (right, left, up, down) through the model,
driving its default input cells (L1-L3), and produces:
  1. Stimulus heatmaps (time x column) for each condition.
  2. Cell-type response traces (column-averaged and optionally per-column).
  3. (optional) Hex mosaic animations via fly_eye.visualize_stimulus for the
     raw stimulus and/or model responses.

``--model`` is a trained run folder or its model.pt. Figures go to
<model_folder>/figures/view_sine_stimulus unless ``--save-dir`` is given.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.view_sine_stimulus --model models/<run>
    python -m drosophila_model.analysis.model_analysis.view_sine_stimulus --model models/<run> --wavelength 60
    python -m drosophila_model.analysis.model_analysis.view_sine_stimulus --model models/<run> --columns -1 --show-voltage
    python -m drosophila_model.analysis.model_analysis.view_sine_stimulus --model models/<run> --temporal-frequency 8 --wavelength 30
    python -m drosophila_model.analysis.model_analysis.view_sine_stimulus --model models/<run> --hex --hex-cell-type T4a T5a
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling.stimulus_generators import make_lr_sine_wave, make_ud_sine_wave
from drosophila_model.analysis.model_analysis.analysis_utilities import (
    CELL_TYPE_GROUPS, central_column_index, exponential_highpass,
)

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_WAVELENGTH = 30.0        # degrees
DEFAULT_TEMPORAL_FREQ = 4.0      # Hz
DEFAULT_STIM_TIME = 2.0          # seconds
DEFAULT_PRE_STIM_TIME = 0.5      # seconds


def main():
    parser = argparse.ArgumentParser(
        description="Visualize moving sine-wave stimuli and model responses."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Trained run folder or its model.pt.",
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
    parser.add_argument(
        "--show-voltage", action="store_const", const="voltage", dest="readout",
        help="Plot raw voltage (pre-activation) instead of firing rate.",
    )
    parser.set_defaults(readout="firing_rate")
    parser.add_argument(
        "--columns", type=int, nargs="+", default=None,
        help="Column indices for individual traces (use -1 for central column).",
    )
    parser.add_argument(
        "--direction", type=str, nargs="+", default=None,
        choices=["right", "left", "up", "down"],
        help="Which direction(s) to run.  Defaults to all four; when given, "
             "the directions are added to the static figure names.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save figures and animations "
             "(default: <model_folder>/figures/view_sine_stimulus).",
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
    if args.hex_cell_type and not args.hex:
        parser.error("--hex-cell-type requires --hex.")

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model from {args.model} ...")
    model = nm.BaseRnn.load(args.model)
    print(f"  {model.num_neurons} neurons, {model.num_cell_types} cell types")

    model_path = Path(args.model)
    model_folder = model_path.parent if model_path.is_file() else model_path
    save_dir = (Path(args.save_dir) if args.save_dir
                else model_folder / "figures" / "view_sine_stimulus")

    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]
    dt = model.dt

    output_types = sorted(
        {ct for cts in CELL_TYPE_GROUPS.values() for ct in cts}
    )
    output_types = [ct for ct in output_types if ct in model.unique_cell_types]

    center_col = central_column_index(
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
        print(f"  Applying exponential highpass filter (tau={args.highpass_tau} s)")
        stim_batch = exponential_highpass(stim_batch, dt, args.highpass_tau)
        for i, cond_name in enumerate(cond_names):
            stim_dict[cond_name] = stim_batch[i]
    use_diverging = args.zero_center or args.highpass_tau is not None

    stim_tensor = torch.tensor(stim_batch, device=model.device, dtype=model.dtype)
    time_axis = np.arange(stim_batch.shape[1]) * dt - args.pre_stim_time

    velocity = tf * wl  # deg/s
    print(f"  {len(cond_names)} directions, lambda={wl} deg, tf={tf} Hz "
          f"(velocity={velocity:.1f} deg/s)")
    print(f"  {stim_batch.shape[1]} timesteps, {stim_batch.shape[2]} columns")

    # ── Run model ─────────────────────────────────────────────────────
    print("Running model ...")
    with torch.no_grad():
        result = model.predict(stim_tensor, output_neurons=output_types)
    outputs = result["outputs"]["optic_lobe"]

    # ── Resolve column indices ────────────────────────────────────────
    show_columns = args.columns
    if show_columns is not None:
        show_columns = [center_col if c == -1 else c for c in show_columns]
        print(f"  Showing individual columns: {show_columns}")

    # ── Sort columns along the motion axis for heatmap display ────────
    # (phi for left/right gratings, theta for up/down gratings)
    sort_orders = {"phi": np.argsort(sample_phi), "theta": np.argsort(sample_theta)}

    # ── Figure 1: stimulus heatmaps ───────────────────────────────────
    n_conds = len(cond_names)
    fig_stim, axes_stim = plt.subplots(
        1, n_conds, figsize=(4 * n_conds, 4), squeeze=False
    )
    prev_sort_key = None
    for ci, cond_name in enumerate(cond_names):
        ax = axes_stim[0, ci]
        sort_key = "phi" if cond_name in ("right", "left") else "theta"
        img = stim_dict[cond_name][:, sort_orders[sort_key]].T  # (columns_sorted, time)
        vmax_val = max(abs(img.min()), abs(img.max()))
        ax.imshow(
            img, aspect="auto", origin="lower",
            extent=[time_axis[0], time_axis[-1], 0, img.shape[0]],
            cmap="RdBu_r" if use_diverging else "gray",
            vmin=-vmax_val if use_diverging else 0,
            vmax=vmax_val if use_diverging else 1,
        )
        ax.set_title(cond_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if sort_key != prev_sort_key:
            ax.set_ylabel(f"Column (sorted by {sort_key})")
            prev_sort_key = sort_key
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

    # ── Save static figures ───────────────────────────────────────────
    # Runs restricted with --direction get their own names so they do not
    # overwrite each other.
    dir_tag = "_" + "_".join(cond_names) if args.direction else ""
    tag = f"{dir_tag}_lambda{wl:.0f}_tf{tf:.1f}"
    save_dir.mkdir(parents=True, exist_ok=True)
    stim_path = save_dir / f"sine_stimulus{tag}.png"
    resp_path = save_dir / f"sine_responses{tag}.png"
    rms_path = save_dir / f"sine_responses_rms{tag}.png"
    fig_stim.savefig(stim_path, dpi=150)
    fig_resp.savefig(resp_path, dpi=150)
    fig_rms.savefig(rms_path, dpi=150)
    plt.close("all")
    print(f"Saved: {stim_path}")
    print(f"Saved: {resp_path}")
    print(f"Saved: {rms_path}")

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

        stim_cmap = (mpl_cm.RdBu_r if use_diverging else mpl_cm.gray)(
            np.linspace(0, 1, 256))

        for ci, cond_name in enumerate(cond_names):
            safe_name = cond_name.lower().replace(" ", "_")

            # --- Stimulus animation ---
            stim_frames = stim_dict[cond_name][frame_idx]  # (F, N)
            titles = [f"sine {cond_name} λ={wl}° tf={tf}Hz  t={t:.3f}s"
                      for t in frame_times]
            stim_filename = str(
                save_dir / f"sine_hex_stim_{safe_name}_lambda{wl:.0f}_tf{tf:.1f}.mp4"
            )

            print(f"  Hex stimulus: {cond_name} ({len(frame_idx)} frames @ {target_fps} fps)")
            visualize_column_responses(
                column_p_q, stim_frames,
                colormap=stim_cmap, fps=target_fps,
                filename=stim_filename,
                plot=False,
                titles=titles,
            )

            # --- Cell-type response animations ---
            for ct in args.hex_cell_type or []:
                if ct not in outputs[readout_key]:
                    print(f"  Warning: cell type '{ct}' not in outputs, skipping.")
                    continue
                resp_full = outputs[readout_key][ct][ci].cpu().numpy()  # (T, N)
                resp_frames = resp_full[frame_idx]
                resp_titles = [f"{cond_name} — {ct} {readout_label}  t={t:.3f}s"
                               for t in frame_times]
                resp_filename = str(
                    save_dir / f"sine_hex_{ct}_{safe_name}_lambda{wl:.0f}_tf{tf:.1f}.mp4"
                )

                print(f"  Hex response: {ct} / {cond_name} ({len(frame_idx)} frames)")
                visualize_column_responses(
                    column_p_q, resp_frames,
                    colormap=mpl_cm.coolwarm(np.linspace(0, 1, 256)),
                    fps=target_fps,
                    filename=resp_filename,
                    plot=False,
                    titles=resp_titles,
                )


if __name__ == "__main__":
    main()
