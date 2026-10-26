"""
Visualize full-field impulse (flash) stimuli and a model's responses.

Runs two conditions through the model, driving its default input cells
(L1-L3):
  - Increment: all columns step from 0 to +amplitude for a brief pulse, then back to 0.
  - Decrement: all columns step from 0 to -amplitude for a brief pulse, then back to 0.

Produces:
  1. Stimulus trace (time × value) for each condition.
  2. Cell-type response traces (column-averaged and optionally per-column).
  3. (optional) Hex mosaic animations via fly_eye.visualize_stimulus for
     model responses.

``--model`` is a trained run folder or its model.pt. Figures go to
<model_folder>/figures/view_impulse_stimulus unless ``--save-dir`` is given.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.view_impulse_stimulus --model models/<run>
    python -m drosophila_model.analysis.model_analysis.view_impulse_stimulus --model models/<run> --pulse-duration 0.05 --columns -1
    python -m drosophila_model.analysis.model_analysis.view_impulse_stimulus --model models/<run> --hex --hex-cell-type T4a L1
"""
import argparse
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch


from drosophila_model.model_definition import neural_models as nm
from drosophila_model.analysis.model_analysis.analysis_utilities import central_column_index

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_PRE_STIM_TIME = 0.5     # seconds before pulse
DEFAULT_POST_STIM_TIME = 2.0    # seconds after pulse
DEFAULT_PULSE_DURATION = None    # None = single timestep
DEFAULT_AMPLITUDE = 50.0

CELL_TYPE_GROUPS = {
    "Lamina (L1-L5)":   ["L1", "L2", "L3", "L4", "L5"],
    "Lamina (Other)":   ["Lai", "T1"],
    "C2 / C3":          ["C2", "C3"],
    "T4 inputs":        ["Mi1", "Tm3", "Mi4", "Mi9"],
    "T5 inputs":        ["Tm1", "Tm2", "Tm4", "Tm9", "CT1_L"],
    "T4":               ["T4a", "T4b", "T4c", "T4d"],
    "T5":               ["T5a", "T5b", "T5c", "T5d"],
}


def _make_impulse(n_columns, dt, pre_stim_time, post_stim_time,
                  pulse_duration, amplitude):
    """Build a full-field impulse stimulus array.

    Returns the (time, columns) array and the pulse length in time steps
    (one step when ``pulse_duration`` is None).
    """
    pre_steps = int(round(pre_stim_time / dt))
    if pulse_duration is None:
        pulse_steps = 1
    else:
        pulse_steps = max(1, int(round(pulse_duration / dt)))
    post_steps = int(round(post_stim_time / dt))
    total_steps = pre_steps + pulse_steps + post_steps

    stim = np.zeros((total_steps, n_columns), dtype=np.float64)
    stim[pre_steps:pre_steps + pulse_steps, :] = amplitude
    return stim, pulse_steps


def main():
    parser = argparse.ArgumentParser(
        description="Visualize full-field impulse stimuli and model responses."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Trained run folder or its model.pt.",
    )
    parser.add_argument(
        "--pre-stim-time", type=float, default=DEFAULT_PRE_STIM_TIME,
        help="Baseline period before pulse (seconds).",
    )
    parser.add_argument(
        "--post-stim-time", type=float, default=DEFAULT_POST_STIM_TIME,
        help="Recovery period after pulse (seconds).",
    )
    parser.add_argument(
        "--pulse-duration", type=float, default=DEFAULT_PULSE_DURATION,
        help="Duration of the pulse in seconds (default: single timestep).",
    )
    parser.add_argument(
        "--amplitude", type=float, default=DEFAULT_AMPLITUDE,
        help=f"Pulse amplitude (default {DEFAULT_AMPLITUDE}).",
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
             "(default: <model_folder>/figures/view_impulse_stimulus).",
    )
    parser.add_argument(
        "--hex", action="store_true",
        help="Render hex mosaic animations of model responses "
             "(requires --hex-cell-type).",
    )
    parser.add_argument(
        "--hex-fps", type=int, default=30,
        help="Playback FPS for hex animations (default 30).",
    )
    parser.add_argument(
        "--hex-cell-type", type=str, nargs="+", default=None,
        help="Cell type(s) whose response to animate on the hex grid "
             "(e.g. --hex-cell-type T4a L1).  Requires --hex.",
    )
    args = parser.parse_args()
    if bool(args.hex) != bool(args.hex_cell_type):
        parser.error("--hex and --hex-cell-type must be given together.")

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model from {args.model} ...")
    model = nm.BaseRnn.load(args.model)
    print(f"  {model.num_neurons} neurons, {model.num_cell_types} cell types")

    model_path = Path(args.model)
    model_folder = model_path.parent if model_path.is_file() else model_path
    save_dir = (Path(args.save_dir) if args.save_dir
                else model_folder / "figures" / "view_impulse_stimulus")

    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]
    dt = model.dt
    n_columns = len(sample_phi)

    output_types = sorted(
        {ct for cts in CELL_TYPE_GROUPS.values() for ct in cts}
    )
    output_types = [ct for ct in output_types if ct in model.unique_cell_types]

    center_col = central_column_index(
        np.asarray(sample_phi), np.asarray(sample_theta)
    )

    # ── Build impulse stimuli ─────────────────────────────────────────
    amp = abs(args.amplitude)
    inc_stim, pulse_steps = _make_impulse(
        n_columns, dt, args.pre_stim_time, args.post_stim_time,
        args.pulse_duration, amplitude=amp,
    )
    dec_stim, _ = _make_impulse(
        n_columns, dt, args.pre_stim_time, args.post_stim_time,
        args.pulse_duration, amplitude=-amp,
    )

    cond_names = ["Increment", "Decrement"]
    stim_dict = {"Increment": inc_stim, "Decrement": dec_stim}

    stim_batch = np.stack([inc_stim, dec_stim], axis=0)  # (2, T, N)
    stim_tensor = torch.tensor(stim_batch, device=model.device, dtype=model.dtype)
    time_axis = np.arange(stim_batch.shape[1]) * dt - args.pre_stim_time

    pulse_dur_actual = pulse_steps * dt
    print(f"  Impulse: amplitude=+/-{amp}, pulse={pulse_dur_actual*1000:.1f} ms "
          f"({pulse_steps} steps)")
    print(f"  {stim_batch.shape[1]} timesteps, {n_columns} columns")

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

    # ── Figure 1: stimulus waveform ───────────────────────────────────
    n_conds = len(cond_names)
    fig_stim, axes_stim = plt.subplots(
        1, n_conds, figsize=(4 * n_conds, 2.5), squeeze=False
    )
    for ci, cond_name in enumerate(cond_names):
        ax = axes_stim[0, ci]
        ax.plot(time_axis, stim_dict[cond_name][:, 0], color="k", linewidth=1.5)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(cond_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ci == 0:
            ax.set_ylabel("Stimulus")
        ax.set_ylim(-amp * 1.3, amp * 1.3)

    fig_stim.suptitle(
        f"Full-field impulse — ±{amp}, {pulse_dur_actual*1000:.1f} ms pulse",
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
            # Groups whose cell types are all absent from the model stay empty.
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6, loc="upper right")
            if row == 0:
                ax.set_title(cond_name, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(group_label, fontsize=9)
            if row == n_groups - 1:
                ax.set_xlabel("Time (s)")

    # Unify y-axis limits across conditions for each cell-type group
    for row in range(n_groups):
        ylims = [axes_resp[row, ci].get_ylim() for ci in range(n_conds)]
        ymin = min(lo for lo, _ in ylims)
        ymax = max(hi for _, hi in ylims)
        for ci in range(n_conds):
            axes_resp[row, ci].set_ylim(ymin, ymax)

    col_tag = ""
    if show_columns is not None:
        col_tag = f" | cols {show_columns}"
    fig_resp.suptitle(
        f"Impulse responses ({readout_label}) — ±{amp}, "
        f"{pulse_dur_actual*1000:.1f} ms pulse{col_tag}",
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
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6, loc="upper right")
            if row == 0:
                ax.set_title(cond_name, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(group_label, fontsize=9)
            if row == n_groups - 1:
                ax.set_xlabel("Time (s)")

    for row in range(n_groups):
        ylims = [axes_rms[row, ci].get_ylim() for ci in range(n_conds)]
        ymin = min(lo for lo, _ in ylims)
        ymax = max(hi for _, hi in ylims)
        for ci in range(n_conds):
            axes_rms[row, ci].set_ylim(ymin, ymax)

    fig_rms.suptitle(
        f"Impulse RMS responses ({readout_label}) — ±{amp}, "
        f"{pulse_dur_actual*1000:.1f} ms pulse{col_tag}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    # ── Save static figures ───────────────────────────────────────────
    pulse_tag = f"{pulse_dur_actual*1000:.0f}ms"
    save_dir.mkdir(parents=True, exist_ok=True)
    stim_path = save_dir / f"impulse_stimulus_{pulse_tag}.png"
    resp_path = save_dir / f"impulse_responses_{pulse_tag}.png"
    rms_path = save_dir / f"impulse_responses_rms_{pulse_tag}.png"
    fig_stim.savefig(stim_path, dpi=150)
    fig_resp.savefig(resp_path, dpi=150)
    fig_rms.savefig(rms_path, dpi=150)
    print(f"Saved: {stim_path}")
    print(f"Saved: {resp_path}")
    print(f"Saved: {rms_path}")
    plt.close("all")

    # ── Hex mosaic animations ─────────────────────────────────────────
    if args.hex:
        from drosophila_model.fly_eye.visualize_stimulus import visualize_column_responses
        import matplotlib.cm as mpl_cm

        column_p_q = model.regions['optic_lobe'].column_p_q

        target_fps = args.hex_fps
        total_time = args.pre_stim_time + args.post_stim_time + pulse_dur_actual
        n_total = stim_batch.shape[1]
        target_frames = int(total_time * target_fps)
        step = max(1, n_total // target_frames)
        frame_idx = np.arange(0, n_total, step)
        frame_times = frame_idx * dt - args.pre_stim_time

        for ci, cond_name in enumerate(cond_names):
            safe_name = cond_name.lower()

            for ct in args.hex_cell_type:
                if ct not in outputs[readout_key]:
                    print(f"  Warning: cell type '{ct}' not in outputs, skipping.")
                    continue
                resp_full = outputs[readout_key][ct][ci].cpu().numpy()
                resp_frames = resp_full[frame_idx]
                resp_titles = [
                    f"{cond_name} — {ct} {readout_label}  t={t:.3f}s"
                    for t in frame_times
                ]
                resp_filename = str(
                    save_dir / f"impulse_hex_{ct}_{safe_name}_{pulse_tag}.mp4"
                )

                print(f"  Hex response: {ct} / {cond_name} "
                      f"({len(frame_idx)} frames @ {target_fps} fps)")
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
