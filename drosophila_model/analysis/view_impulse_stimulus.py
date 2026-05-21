"""
Visualize full-field impulse (flash) stimuli and HandTunedModel responses.

Runs two conditions through the model:
  - Increment: all columns step from 0 → +1 for a brief pulse, then back to 0.
  - Decrement: all columns step from 0 → -1 for a brief pulse, then back to 0.

Produces:
  1. Stimulus trace (time × value) for each condition.
  2. Cell-type response traces (column-averaged and optionally per-column).
  3. (optional) Hex mosaic animations via FlyEye for model responses.

Usage:
    python view_impulse_stimulus.py
    python view_impulse_stimulus.py --model models/hand_tuned/model.pt
    python view_impulse_stimulus.py --pulse-duration 0.05 --columns -1
    python view_impulse_stimulus.py --hex --hex-cell-type T4a L1 --save-dir figures/
"""
import argparse
from pathlib import Path
import sys

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch


from drosophila_model.model_definition.model_types import neural_models as nm

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "models/hand_tuned/model.pt"
DEFAULT_PRE_STIM_TIME = 0.5     # seconds before pulse
DEFAULT_POST_STIM_TIME = 2.0    # seconds after pulse
DEFAULT_PULSE_DURATION = None    # None = single timestep
DEFAULT_AMPLITUDE = 50.0

INPUT_NEURONS = ["R1-6"]

CELL_TYPE_GROUPS = {
    "Photoreceptors":   ["R1-6"],
    "Lamina (L1-L6)":   ["L1", "L2", "L3", "L4","L5"],
    "Lamina (Other)":   ["Lai", "T1"],
    "C2 / C3":          ["C2", "C3"],
    "T4 inputs":        ["Mi1", "Tm3", "Mi4", "Mi9"],
    "T5 inputs":        ["Tm1", "Tm2", "Tm4", "Tm9","CT1_L"],
    "T4":               ["T4a", "T4b", "T4c", "T4d"],
    "T5":               ["T5a", "T5b", "T5c", "T5d"],
}


def _run_input_decomposition(model, stim_tensor, input_neurons, target_cell_type):
    """Re-run model step-by-step, decomposing synaptic inputs to *target_cell_type*.

    Returns dict mapping presynaptic type name to (batch, time, num_columns) tensor.
    """
    num_batches, num_time, _ = stim_tensor.shape

    filtered_input = [n for n in input_neurons if n in model.unique_cell_types]
    input_neuron_indices, input_column_indices = model.regions['optic_lobe'].get_shared_columns(filtered_input)
    stim_to_cell_mapping = {
        'neuron_index': torch.tensor(input_neuron_indices, device=model.device, dtype=torch.int64),
        'stimulus_index': torch.tensor(input_column_indices, device=model.device, dtype=torch.int64),
    }

    v = model.steady_state.unsqueeze(0).expand(num_batches, -1)
    filter_state = None
    contributions = {}

    for t in range(num_time):
        u_t = stim_tensor[:, t, :]
        if model.regions['optic_lobe'].use_input_conv:
            u_t, filter_state = model.regions['optic_lobe'].apply_input_conv(u_t, filter_state)
        v = model(v, u_t, stim_to_cell_mapping)
        if model.use_layer_norm:
            v = model.layer_norm(v)

        decomp = model.decompose_synaptic_input(v, target_cell_type)
        for pt, vals in decomp.items():
            if pt not in contributions:
                contributions[pt] = torch.full(
                    (num_batches, num_time, model.regions['optic_lobe'].num_columns), float('nan'),
                    device=model.device, dtype=model.dtype,
                )
            contributions[pt][:, t, :] = vals

    return contributions


def _central_column_index(sample_phi, sample_theta):
    ref = np.array([0.0, 1.0, 0.0])
    xyz = np.stack([
        np.sin(sample_theta) * np.cos(sample_phi),
        np.sin(sample_theta) * np.sin(sample_phi),
        np.cos(sample_theta),
    ], axis=1)
    return int(np.argmax(xyz @ ref))


def _make_impulse(n_columns, dt, pre_stim_time, post_stim_time,
                  pulse_duration, amplitude):
    """Build a full-field impulse stimulus array.

    Returns (time, columns) array and the index of pulse onset.
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
    return stim, pre_steps


def main():
    parser = argparse.ArgumentParser(
        description="Visualize full-field impulse stimuli and model responses."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to saved model .pt file.",
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
        help="Pulse amplitude (default 1.0).",
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
        "--show-inputs-to", type=str, default=None, metavar="CELL_TYPE",
        help="Decompose and plot per-presynaptic-type synaptic inputs "
             "to this cell type (e.g. --show-inputs-to T4a).",
    )
    parser.add_argument(
        "--hex", action="store_true",
        help="Render hex mosaic animations of model responses.",
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

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model from {args.model} …")
    model = nm.BaseRnn.load(args.model)
    print(f"  {model.num_neurons} neurons, {model.num_cell_types} cell types, "
          f"omega={model.get_omega():.4f}")

    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]
    dt = model.dt
    n_columns = len(sample_phi)

    output_types = sorted(
        {ct for cts in CELL_TYPE_GROUPS.values() for ct in cts}
    )
    output_types = [ct for ct in output_types if ct in model.unique_cell_types]

    center_col = _central_column_index(
        np.asarray(sample_phi), np.asarray(sample_theta)
    )

    # ── Build impulse stimuli ─────────────────────────────────────────
    amp = abs(args.amplitude)
    inc_stim, onset_idx = _make_impulse(
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

    pulse_steps = 1 if args.pulse_duration is None else max(1, int(round(args.pulse_duration / dt)))
    pulse_dur_actual = pulse_steps * dt
    print(f"  Impulse: amplitude=±{amp}, pulse={pulse_dur_actual*1000:.1f} ms "
          f"({pulse_steps} steps)")
    print(f"  {stim_batch.shape[1]} timesteps, {n_columns} columns")

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

    # ── Figure 4 (optional): decomposed synaptic inputs ─────────────
    fig_decomp = None
    if args.show_inputs_to:
        target_ct = args.show_inputs_to
        if not hasattr(model, 'decompose_synaptic_input'):
            print(f"  Input decomposition not supported for {type(model).__name__}")
        elif target_ct not in model.unique_cell_types:
            print(f"  Cell type '{target_ct}' not found in model")
        else:
            print(f"Decomposing synaptic inputs to {target_ct} …")
            with torch.no_grad():
                contributions = _run_input_decomposition(
                    model, stim_tensor, INPUT_NEURONS, target_ct,
                )
            if not contributions:
                print(f"  No synaptic inputs found for {target_ct}")
            else:
                # Sort presynaptic types by peak column-mean magnitude
                peak_mag = {}
                for pt, vals in contributions.items():
                    mean_trace = torch.nanmean(vals, dim=2).cpu().numpy()
                    peak_mag[pt] = np.max(np.abs(mean_trace))
                sorted_types = sorted(peak_mag, key=lambda k: peak_mag[k],
                                      reverse=True)

                fig_decomp, axes_decomp = plt.subplots(
                    1, n_conds, figsize=(5 * n_conds, 4), squeeze=False,
                )
                for ci, cond_name in enumerate(cond_names):
                    ax = axes_decomp[0, ci]
                    for pt in sorted_types:
                        trace = torch.nanmean(
                            contributions[pt][ci], dim=1,
                        ).cpu().numpy()
                        ax.plot(time_axis, trace, label=pt, linewidth=1.2)
                    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
                    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
                    ax.legend(fontsize=7, loc="best")
                    ax.set_title(cond_name, fontsize=11, fontweight="bold")
                    ax.set_xlabel("Time (s)")
                    if ci == 0:
                        ax.set_ylabel("Synaptic input")

                # Unify y-axis across conditions
                ylims = [axes_decomp[0, ci].get_ylim() for ci in range(n_conds)]
                ymin = min(lo for lo, _ in ylims)
                ymax = max(hi for _, hi in ylims)
                for ci in range(n_conds):
                    axes_decomp[0, ci].set_ylim(ymin, ymax)

                fig_decomp.suptitle(
                    f"Synaptic inputs to {target_ct} — ±{amp}, "
                    f"{pulse_dur_actual*1000:.1f} ms pulse",
                    fontsize=13, fontweight="bold",
                )
                plt.tight_layout()

    # ── Save or show static figures ──────────────────────────────────
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stim_path = save_dir / f"impulse_stimulus_{pulse_dur_actual*1000:.0f}ms.png"
        resp_path = save_dir / f"impulse_responses_{pulse_dur_actual*1000:.0f}ms.png"
        rms_path = save_dir / f"impulse_responses_rms_{pulse_dur_actual*1000:.0f}ms.png"
        fig_stim.savefig(stim_path, dpi=150)
        fig_resp.savefig(resp_path, dpi=150)
        fig_rms.savefig(rms_path, dpi=150)
        print(f"Saved: {stim_path}")
        print(f"Saved: {resp_path}")
        print(f"Saved: {rms_path}")
        if fig_decomp is not None:
            decomp_path = save_dir / f"impulse_inputs_to_{args.show_inputs_to}_{pulse_dur_actual*1000:.0f}ms.png"
            fig_decomp.savefig(decomp_path, dpi=150)
            print(f"Saved: {decomp_path}")
        plt.close("all")
    else:
        plt.show()

    # ── Hex mosaic animations ─────────────────────────────────────────
    if args.hex:
        from drosophila_model.FlyEye.visualize_stimulus import visualize_column_responses
        import matplotlib.cm as mpl_cm

        column_p_q = model.regions['optic_lobe'].column_p_q

        target_fps = args.hex_fps
        total_time = args.pre_stim_time + args.post_stim_time + pulse_dur_actual
        n_total = stim_batch.shape[1]
        target_frames = int(total_time * target_fps)
        step = max(1, n_total // target_frames)
        frame_idx = np.arange(0, n_total, step)
        frame_times = frame_idx * dt - args.pre_stim_time

        save_dir = Path(args.save_dir) if args.save_dir else None

        if args.hex_cell_type:
            for cond_name in cond_names:
                ci = cond_names.index(cond_name)
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

                    resp_filename = None
                    if save_dir:
                        resp_filename = str(
                            save_dir / f"impulse_hex_{ct}_{safe_name}_{pulse_dur_actual*1000:.0f}ms.mp4"
                        )

                    print(f"  Hex response: {ct} / {cond_name} "
                          f"({len(frame_idx)} frames @ {target_fps} fps)")
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
