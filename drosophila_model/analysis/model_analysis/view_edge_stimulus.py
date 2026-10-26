"""
Visualize moving-edge stimuli and a model's responses.

Runs all four polarity × direction conditions (ON-right, ON-left, OFF-right,
OFF-left) through the model, driving its default input cells (L1-L3), and
produces:
  1. A stimulus heatmap (time × column) for each condition.
  2. Cell-type response traces (column-averaged and optionally per-column).
  3. (optional) Hex mosaic animations via fly_eye.visualize_stimulus for the
     raw stimulus and/or model responses.

``--model`` is a trained run folder or its model.pt. Figures go to
<model_folder>/figures/view_edge_stimulus unless ``--save-dir`` is given.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.view_edge_stimulus --model models/<run>
    python -m drosophila_model.analysis.model_analysis.view_edge_stimulus --model models/<run> --velocity 60
    python -m drosophila_model.analysis.model_analysis.view_edge_stimulus --model models/<run> --columns -1 --show-voltage

    # Hex mosaic animation for the ON-right stimulus, saved as mp4:
    python -m drosophila_model.analysis.model_analysis.view_edge_stimulus --model models/<run> --hex --condition "ON right"
"""
import argparse
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling.stimulus_generators import make_lr_moving_edge
from drosophila_model.analysis.model_analysis.analysis_utilities import (
    CELL_TYPE_GROUPS, central_column_index, exponential_highpass,
)


# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_VELOCITY = 30.0      # deg/s
DEFAULT_STIM_TIME = 3.0
DEFAULT_PRE_STIM_TIME = 0.5
DEFAULT_EDGE_WIDTH = 2.5     # degrees

# Condition name -> (sign of the velocity, edge type)
CONDITIONS = {
    "ON right":  (1, "on"),
    "ON left":   (-1, "on"),
    "OFF right": (1, "off"),
    "OFF left":  (-1, "off"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize moving-edge stimuli and model responses."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Trained run folder or its model.pt.",
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
        "--save-dir", type=str, default=None,
        help="Directory to save figures and animations "
             "(default: <model_folder>/figures/view_edge_stimulus).",
    )
    parser.add_argument(
        "--hex", action="store_true",
        help="Render hex mosaic animations of the stimulus and model responses "
             "using fly_eye.visualize_stimulus.",
    )
    parser.add_argument(
        "--condition", type=str, nargs="+", default=None,
        choices=list(CONDITIONS),
        help="Which edge condition(s) to animate with --hex.  "
             "Defaults to all four.  Quote names with spaces.",
    )
    parser.add_argument(
        "--hex-fps", type=int, default=30,
        help="Playback FPS for hex animations (default 30).",
    )
    parser.add_argument(
        "--hex-cell-type", type=str, nargs="+", default=None,
        help="Cell type(s) whose response to animate on the hex grid "
             "(e.g. --hex-cell-type T4a).  Requires --hex.",
    )
    args = parser.parse_args()
    if not args.hex and (args.hex_cell_type or args.condition):
        parser.error("--hex-cell-type and --condition require --hex.")

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model from {args.model} ...")
    model = nm.BaseRnn.load(args.model)
    print(f"  {model.num_neurons} neurons, {model.num_cell_types} cell types")

    model_path = Path(args.model)
    model_folder = model_path.parent if model_path.is_file() else model_path
    save_dir = (Path(args.save_dir) if args.save_dir
                else model_folder / "figures" / "view_edge_stimulus")

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

    # ── Generate edge stimuli ─────────────────────────────────────────
    stim_dict = {}
    for cond_name, (sign, etype) in CONDITIONS.items():
        arr = make_lr_moving_edge(
            sample_phi, sample_theta,
            velocity=sign * args.velocity, edge_type=etype,
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
        print(f"  Applying exponential highpass filter (tau={args.highpass_tau} s)")
        stim_batch = exponential_highpass(stim_batch, dt, args.highpass_tau)
        # Update per-condition dict so heatmaps and hex anims show the filtered stimulus
        for i, cond_name in enumerate(cond_names):
            stim_dict[cond_name] = stim_batch[i]
    use_diverging = args.zero_center or args.highpass_tau is not None

    stim_tensor = torch.tensor(stim_batch, device=model.device, dtype=model.dtype)
    time_axis = np.arange(stim_batch.shape[1]) * dt - args.pre_stim_time

    print(f"  {len(cond_names)} conditions, {stim_batch.shape[1]} timesteps, "
          f"{stim_batch.shape[2]} columns")

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
            cmap="RdBu_r" if use_diverging else "gray",
            vmin=-vmax if use_diverging else 0,
            vmax=vmax if use_diverging else 1,
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

    fig_rms.suptitle(
        f"Edge RMS responses ({readout_label}) — {args.velocity:.0f} deg/s{col_tag}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    # ── Save static figures ───────────────────────────────────────────
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

    # ── Hex mosaic animations ─────────────────────────────────────────
    if args.hex:
        from drosophila_model.fly_eye.visualize_stimulus import visualize_column_responses
        import matplotlib.cm as mpl_cm

        column_p_q = model.regions['optic_lobe'].column_p_q
        hex_conds = args.condition if args.condition else cond_names

        # Subsample to keep animations at a reasonable frame count
        target_fps = args.hex_fps
        total_time = args.pre_stim_time + args.stim_time
        target_frames = int(total_time * target_fps)
        n_total = stim_batch.shape[1]
        step = max(1, n_total // target_frames)
        frame_idx = np.arange(0, n_total, step)
        frame_times = frame_idx * dt - args.pre_stim_time

        stim_cmap = (mpl_cm.RdBu_r if use_diverging else mpl_cm.gray)(
            np.linspace(0, 1, 256))

        for cond_name in hex_conds:
            ci = cond_names.index(cond_name)
            safe_name = cond_name.lower().replace(" ", "_")

            # --- Stimulus animation ---
            stim_frames = stim_dict[cond_name][frame_idx]  # (F, N)
            titles = [f"{cond_name}  t={t:.3f}s" for t in frame_times]
            stim_filename = str(
                save_dir / f"edge_hex_stim_{safe_name}_{args.velocity:.0f}dps.mp4"
            )

            print(f"  Hex stimulus: {cond_name} ({len(frame_idx)} frames "
                  f"@ {target_fps} fps) -> {stim_filename}")
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
                    print(f"  Warning: cell type '{ct}' not in outputs, skipping hex response.")
                    continue
                resp_full = outputs[readout_key][ct][ci].cpu().numpy()  # (T, N_columns)
                resp_frames = resp_full[frame_idx]
                resp_titles = [f"{cond_name} — {ct} {readout_label}  t={t:.3f}s"
                               for t in frame_times]
                resp_filename = str(
                    save_dir / f"edge_hex_{ct}_{safe_name}_{args.velocity:.0f}dps.mp4"
                )

                print(f"  Hex response: {ct} / {cond_name} ({len(frame_idx)} frames) "
                      f"-> {resp_filename}")
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
