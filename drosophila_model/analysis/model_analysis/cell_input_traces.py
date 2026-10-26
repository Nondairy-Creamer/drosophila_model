"""
Time traces of **T4a** and **T5a** on the **same** retinotopic column, each with **all**
presynaptic partners (every unique pre in ``synapse_counts_coo`` onto that neuron).
The column is the **middle** of the L1/L2/L3 shared columns sorted by azimuth φ,
plus ``--column-offset``.  Stimulus is chosen with ``--stim-type``:

* **edge** — same as ``edge_sweep``: light **and** dark edges at 240°/s, right & up,
  FOV 180°, 0.5 s pre-stim.
* **sine** — same as ``sine_wave_sweep``: drifting sine (default 4 Hz, 30° wavelength,
  1 s stim, 0.5 s pre-stim).

Figures go to ``<model_folder>/figures/cell_input_traces/<stim_type>/``; the
partner summaries (``t4a_partners.txt`` / ``t5a_partners.txt``) go to
``<model_folder>/analysis_data/cell_input_traces/<stim_type>/``.

Each presynaptic trace is ``W·g_out·g_in·FR``; each motion-detector FR panel includes
``f(Σ+b)`` on a twin axis, where ``f`` is the model's nonlinearity: the steady-state
firing rate the summed plotted inputs would produce. This input decomposition is
exact only for ``CurrentDynamics`` models (a warning is printed otherwise).  Partners
outside the L1/L2/L3 grid yield NaN FR and thus NaN scaled paths.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.cell_input_traces models/<model>/ --stim-type edge
    python -m drosophila_model.analysis.model_analysis.cell_input_traces models/<model>/ --stim-type sine
    python -m drosophila_model.analysis.model_analysis.cell_input_traces models/<model>/ --stim-type sine --wavelength 30 --temporal-frequency 4
    python -m drosophila_model.analysis.model_analysis.cell_input_traces models/<model>/ --column-offset -1
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List, Optional, Tuple


import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from drosophila_model.model_definition import dynamics as dm
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling.stimulus_generators import make_lr_sine_wave, make_ud_sine_wave
from drosophila_model.analysis.model_analysis.edge_sweep import (
    generate_stimulus as generate_stimulus_edge,
)

# --- Edge defaults (match edge_sweep) ---
VELOCITY_DEG_S = 240.0
FOV_DEG = 180.0
PRE_STIM_TIME_S = 0.5

# --- Sine defaults (match sine_wave_sweep) ---
STIM_TIME_S_SINE_DEFAULT = 1.0
DEFAULT_TEMPORAL_FREQ_HZ = 4.0
DEFAULT_WAVELENGTH_DEG = 30.0


def generate_stimulus_sine(
    direction: str,
    sample_phi,
    sample_theta,
    wavelength_deg: float,
    temporal_frequency_hz: float,
    dt: float,
    stim_time: float,
    pre_stim_time: float,
):
    """Drifting sine of shape (1, time, columns).

    Same direction convention as :func:`sine_wave_sweep.generate_stimulus`,
    parameterised by temporal frequency instead of velocity
    (velocity = temporal_frequency * wavelength).
    """
    direction = direction.lower()
    if direction == "right":
        tf = temporal_frequency_hz
        sine_wave, _ = make_lr_sine_wave(
            sample_phi,
            sample_theta,
            wavelength=wavelength_deg,
            temporal_frequency=tf,
            dt=dt,
            stim_time=stim_time,
            pre_stim_time=pre_stim_time,
        )
    elif direction == "left":
        tf = -temporal_frequency_hz
        sine_wave, _ = make_lr_sine_wave(
            sample_phi,
            sample_theta,
            wavelength=wavelength_deg,
            temporal_frequency=tf,
            dt=dt,
            stim_time=stim_time,
            pre_stim_time=pre_stim_time,
        )
    elif direction == "up":
        tf = temporal_frequency_hz
        sine_wave, _ = make_ud_sine_wave(
            sample_phi,
            sample_theta,
            wavelength=wavelength_deg,
            temporal_frequency=tf,
            dt=dt,
            stim_time=stim_time,
            pre_stim_time=pre_stim_time,
        )
    elif direction == "down":
        tf = -temporal_frequency_hz
        sine_wave, _ = make_ud_sine_wave(
            sample_phi,
            sample_theta,
            wavelength=wavelength_deg,
            temporal_frequency=tf,
            dt=dt,
            stim_time=stim_time,
            pre_stim_time=pre_stim_time,
        )
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return sine_wave[None, :, :]


def shared_columns_sorted_by_phi(model: nm.BaseRnn, input_cols: np.ndarray) -> List[int]:
    """L1/L2/L3 shared column ids sorted by azimuth φ then column index."""
    uc = np.unique(input_cols).astype(int)
    phi_all = model.regions['optic_lobe'].column_phi_theta[:, 0]
    keyed = [(int(c), float(phi_all[c])) for c in uc]
    keyed.sort(key=lambda t: (t[1], t[0]))
    return [t[0] for t in keyed]


def neuron_index_at_column(model: nm.BaseRnn, column_id: int, cell_type: str) -> int:
    """Neuron index of *cell_type* at *column_id* (deterministic if multiple: smallest index)."""
    col_per_neuron = model.regions['optic_lobe'].column_indices_per_neuron
    mask = (model.cell_types == cell_type) & (col_per_neuron == column_id)
    hits = np.where(mask)[0]
    if hits.size == 0:
        raise RuntimeError(
            f"No {cell_type} at column {column_id} (need T4a and T5a on the same shared column).",
        )
    return int(np.min(hits))


def resolve_same_column_t4_t5(
    model: nm.BaseRnn,
    input_cols: np.ndarray,
    column_offset: int,
) -> Tuple[int, int, int, int, int]:
    """Middle column (by φ) + *column_offset* → ``(column_id, t4_idx, t5_idx, middle_k, n_cols)``."""
    cols = shared_columns_sorted_by_phi(model, input_cols)
    n = len(cols)
    if n == 0:
        raise RuntimeError("No shared L1/L2/L3 columns.")
    middle_k = (n - 1) // 2
    k = middle_k + column_offset
    if k < 0 or k >= n:
        raise ValueError(
            f"column_offset={column_offset} → sorted-column index {k} out of range [0, {n}) "
            f"(middle index {middle_k}, {n} shared columns by φ).",
        )
    col = cols[k]
    t4_idx = neuron_index_at_column(model, col, "T4a")
    t5_idx = neuron_index_at_column(model, col, "T5a")
    return col, t4_idx, t5_idx, middle_k, n


def presynaptic_indices_all(model: nm.BaseRnn, post_idx: int) -> List[Tuple[str, int]]:
    """Return (cell_type, pre_neuron_index) for each unique presynaptic neuron onto *post_idx* (full COO)."""
    coo = model.synapse_counts_coo
    on_post = coo[coo[:, 1] == post_idx]
    seen_pre = set()
    out: List[Tuple[str, int]] = []
    for row in on_post:
        pre_i = int(row[0])
        if pre_i in seen_pre:
            continue
        seen_pre.add(pre_i)
        ct = str(model.cell_types[pre_i])
        out.append((ct, pre_i))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def trace_for_neuron(pred: dict, model: nm.BaseRnn, cell_type: str, neuron_idx: int) -> np.ndarray:
    """Firing-rate time series (time,) for one neuron, from predict outputs."""
    ci = model.regions['optic_lobe'].column_indices_per_neuron[neuron_idx]
    if np.isnan(ci):
        return np.full(pred["firing_rate"][cell_type].shape[1], np.nan)
    col = int(ci)
    return pred["firing_rate"][cell_type][0, :, col].detach().cpu().numpy()


def _synapse_weight_post_pre(model: nm.BaseRnn, post_idx: int, pre_idx: int) -> float:
    """Sparse weight W[post, pre] as used by ``forward`` (rows = post, cols = pre).

    Signed for current-based models; for ``ConductanceDynamics`` it is the
    unsigned count in the presynaptic neuron's transmitter matrix.
    """
    if isinstance(model, dm.ConductanceDynamics):
        nt = model.transmitters[pre_idx]
        ri = int(np.where(model.unique_transmitters == nt)[0][0])
        w = model.synapse_counts[ri]
    else:
        w = model.synapse_counts
    # Read row ``post_idx`` straight from the CSR matrix.
    crow = w.crow_indices()
    start, end = int(crow[post_idx]), int(crow[post_idx + 1])
    cols = w.col_indices()[start:end]
    vals = w.values()[start:end]
    return float(vals[cols == pre_idx].sum().cpu().item())


def postsynaptic_input_gain(model: nm.BaseRnn, post_idx: int) -> float:
    """Single scalar g_in for the postsynaptic neuron, applied to every presynaptic path."""
    with torch.no_grad():
        nti = model.neuron_to_type_index
        exp_in = model.input_strength_log.exp()
        if isinstance(model, dm.ConductanceDynamics):
            return float(exp_in[:, nti[post_idx]].mean().cpu().item())
        return float(exp_in[nti[post_idx]].cpu().item())


def postsynaptic_offset(model: nm.BaseRnn, post_idx: int) -> float:
    """Per-cell-type bias *b* in ``z + b``; 0 if the model has no ``offset``."""
    if not hasattr(model, "offset"):
        return 0.0
    with torch.no_grad():
        cti = int(model.neuron_to_type_index[post_idx].item())
        return float(model.offset[cti].cpu().item())


def presynaptic_output_gain(model: nm.BaseRnn, pre_idx: int) -> float:
    """g_out for the presynaptic neuron’s cell type."""
    with torch.no_grad():
        return float(model.output_strength_log.exp()[model.neuron_to_type_index[pre_idx]].cpu().item())


def presynaptic_path_scale(model: nm.BaseRnn, pre_idx: int, post_idx: int) -> float:
    """``W(post,pre) * g_out(pre) * g_in(post)`` — no per-presynapse input row."""
    w = _synapse_weight_post_pre(model, post_idx, pre_idx)
    g_out = presynaptic_output_gain(model, pre_idx)
    g_in_post = postsynaptic_input_gain(model, post_idx)
    return w * g_out * g_in_post


def short_presynaptic_legend_name(cell_type: str, index_among_type: int, n_of_type: int) -> str:
    """Compact legend: ``Mi1`` if only one of that type, else ``Mi1 (1)``, ``Mi1 (2)``, …"""
    if n_of_type <= 1:
        return cell_type
    return f"{cell_type} ({index_among_type})"


def _color_cycle() -> Tuple[List[str], int]:
    cyc = plt.rcParams["axes.prop_cycle"].by_key()
    colors = cyc.get("color")
    if not colors:
        colors = [f"C{i}" for i in range(10)]
    return list(colors), len(colors)


def _color_and_linestyle(series_index: int, n_colors: int) -> Tuple[str, str]:
    colors, _ = _color_cycle()
    n_colors = min(n_colors, len(colors))
    color = colors[series_index % n_colors]
    cycle = series_index // n_colors
    linestyles = ["-", ":", "--", "-."]
    ls = linestyles[cycle % len(linestyles)]
    return color, ls


def run_t4_input_traces(
    *,
    stim_type: str,
    model_folder: Path,
    model: nm.BaseRnn,
    column_offset: int,
    min_trace_peak: float,
    # sine-only (ignored for edge)
    temporal_frequency_hz: float,
    wavelength_deg: float,
    stim_time_sine: float,
) -> None:
    """Core pipeline for one stimulus family (edge or sine)."""
    stim_type = stim_type.lower()
    if stim_type not in ("edge", "sine"):
        raise ValueError(f"stim_type must be 'edge' or 'sine', got {stim_type!r}")
    if not isinstance(model, dm.CurrentDynamics):
        print(
            f"WARNING: {type(model).__name__} is not CurrentDynamics; the "
            "W*g_out*g_in path scales and the f(sum+b) proxy are not its true "
            "input decomposition."
        )

    dt = model.dt
    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]

    if stim_type == "edge":
        stim_time = FOV_DEG / VELOCITY_DEG_S
        pre_stim = PRE_STIM_TIME_S

        stim_summary_line = (
            f"Stimulus (edge): {VELOCITY_DEG_S:g} deg/s, FOV={FOV_DEG:g} deg, "
            f"stim_time={stim_time:g} s, pre_stim={pre_stim:g} s; "
            f"polarities light + dark; directions right + up"
        )
    else:
        stim_time = stim_time_sine
        pre_stim = PRE_STIM_TIME_S
        tf_hz = temporal_frequency_hz
        wl_deg = wavelength_deg

        def gen_stim(direction: str) -> np.ndarray:
            return generate_stimulus_sine(
                direction,
                sample_phi,
                sample_theta,
                wl_deg,
                tf_hz,
                dt,
                stim_time,
                pre_stim,
            )

        stim_labels = {
            "right": f"Sine {tf_hz:g} Hz, λ={wl_deg:g}° → (right)",
            "up": f"Sine {tf_hz:g} Hz, λ={wl_deg:g}° ↑ (up)",
        }

        tf_tag = f"{tf_hz:g}".replace(".", "p")
        wl_tag = f"{wl_deg:g}".replace(".", "p")
        file_prefix = f"sine_{tf_tag}hz_{wl_tag}deg"

        def output_basename(direction: str) -> str:
            return f"{file_prefix}_{direction}"

        stim_summary_line = (
            f"Stimulus (sine): temporal_frequency={tf_hz:g} Hz, wavelength={wl_deg:g} deg, "
            f"stim_time={stim_time:g} s, pre_stim={pre_stim:g} s"
        )

    _, input_cols_arr = model.regions['optic_lobe'].get_shared_columns(["L1", "L2", "L3"])
    input_cols = np.unique(input_cols_arr)

    col, t4_idx, t5_idx, middle_k, n_cols = resolve_same_column_t4_t5(model, input_cols, column_offset)
    t4_col = int(col)
    t5_col = int(col)
    list_index = middle_k + column_offset
    column_ctx = (
        f"column {col} — {n_cols} shared columns sorted by φ, middle index {middle_k}, "
        f"column_offset={column_offset} → list index {list_index}"
    )
    partners_t4 = presynaptic_indices_all(model, t4_idx)
    partners_t5 = presynaptic_indices_all(model, t5_idx)
    g_in_t4 = postsynaptic_input_gain(model, t4_idx)
    g_in_t5 = postsynaptic_input_gain(model, t5_idx)
    t4_offset = postsynaptic_offset(model, t4_idx)
    t5_offset = postsynaptic_offset(model, t5_idx)
    partner_path_scales_t4: dict[int, float] = {
        ni: presynaptic_path_scale(model, ni, t4_idx) for _, ni in partners_t4
    }
    partner_path_scales_t5: dict[int, float] = {
        ni: presynaptic_path_scale(model, ni, t5_idx) for _, ni in partners_t5
    }

    n_of_type_t4 = Counter(ct for ct, _ in partners_t4)
    idx_among_t4 = defaultdict(int)
    plot_specs_t4: List[Tuple[str, str, str, Optional[int]]] = [("T4a", "T4a", "T4a", None)]
    legend_for_trace_key: dict[str, str] = {"T4a": "T4a", "T5a": "T5a"}
    for ct, ni in partners_t4:
        idx_among_t4[ct] += 1
        trace_key = f"t4:{ni}"
        leg = short_presynaptic_legend_name(ct, idx_among_t4[ct], n_of_type_t4[ct])
        plot_specs_t4.append((trace_key, leg, ct, ni))
        legend_for_trace_key[trace_key] = leg

    n_of_type_t5 = Counter(ct for ct, _ in partners_t5)
    idx_among_t5 = defaultdict(int)
    plot_specs_t5: List[Tuple[str, str, str, Optional[int]]] = [("T5a", "T5a", "T5a", None)]
    for ct, ni in partners_t5:
        idx_among_t5[ct] += 1
        trace_key = f"t5:{ni}"
        leg = short_presynaptic_legend_name(ct, idx_among_t5[ct], n_of_type_t5[ct])
        plot_specs_t5.append((trace_key, leg, ct, ni))
        legend_for_trace_key[trace_key] = leg

    plot_specs = plot_specs_t4 + plot_specs_t5
    output_types = sorted({spec[2] for spec in plot_specs})
    output_types = [t for t in output_types if t in model.unique_cell_types]
    if not output_types:
        raise RuntimeError("No output cell types available to plot.")

    directions = ["right", "up"]
    edge_polarities = ("light", "dark")

    num_time = None  # set from the first generated stimulus
    if stim_type == "edge":
        traces_edge: dict[str, dict[str, dict[str, np.ndarray]]] = {
            pol: {d: {} for d in directions} for pol in edge_polarities
        }
        for pol in edge_polarities:
            for direction in directions:
                stim_np = generate_stimulus_edge(
                    direction,
                    sample_phi,
                    sample_theta,
                    VELOCITY_DEG_S,
                    pol,
                    dt,
                    stim_time,
                    pre_stim,
                )
                if num_time is None:
                    num_time = stim_np.shape[1]
                assert stim_np.shape[1] == num_time
                batch = torch.tensor(stim_np, device=model.device, dtype=model.dtype)
                with torch.no_grad():
                    pred = model.predict(
                        batch,
                        input_neurons=["L1", "L2", "L3"],
                        output_neurons=output_types,
                    )["outputs"]["optic_lobe"]

                for trace_key, _leg, ct, ni in plot_specs:
                    if ct not in output_types:
                        continue
                    if trace_key == "T4a":
                        tr = pred["firing_rate"][ct][0, :, t4_col].detach().cpu().numpy()
                    elif trace_key == "T5a":
                        tr = pred["firing_rate"][ct][0, :, t5_col].detach().cpu().numpy()
                    else:
                        assert ni is not None
                        if trace_key.startswith("t4:"):
                            tr = trace_for_neuron(pred, model, ct, ni) * partner_path_scales_t4[ni]
                        else:
                            tr = trace_for_neuron(pred, model, ct, ni) * partner_path_scales_t5[ni]
                    traces_edge[pol][direction][trace_key] = tr
        traces = traces_edge  # type: ignore[assignment]
    else:
        traces_sin: dict[str, dict[str, np.ndarray]] = {d: {} for d in directions}
        for direction in directions:
            stim_np = gen_stim(direction)
            if num_time is None:
                num_time = stim_np.shape[1]
            assert stim_np.shape[1] == num_time
            batch = torch.tensor(stim_np, device=model.device, dtype=model.dtype)
            with torch.no_grad():
                pred = model.predict(
                    batch,
                    input_neurons=["L1", "L2", "L3"],
                    output_neurons=output_types,
                )["outputs"]["optic_lobe"]

            for trace_key, _leg, ct, ni in plot_specs:
                if ct not in output_types:
                    continue
                if trace_key == "T4a":
                    tr = pred["firing_rate"][ct][0, :, t4_col].detach().cpu().numpy()
                elif trace_key == "T5a":
                    tr = pred["firing_rate"][ct][0, :, t5_col].detach().cpu().numpy()
                else:
                    assert ni is not None
                    if trace_key.startswith("t4:"):
                        tr = trace_for_neuron(pred, model, ct, ni) * partner_path_scales_t4[ni]
                    else:
                        tr = trace_for_neuron(pred, model, ct, ni) * partner_path_scales_t5[ni]
                traces_sin[direction][trace_key] = tr
        traces = traces_sin  # type: ignore[assignment]

    time_s = np.arange(num_time, dtype=np.float64) * dt

    partner_keys_all_t4 = [tk for tk, _l, ct, ni in plot_specs_t4 if ni is not None]
    partner_keys_all_t5 = [tk for tk, _l, ct, ni in plot_specs_t5 if ni is not None]

    def _abs_peak(arrays: List[np.ndarray]) -> float:
        """Max |value| ignoring NaN; NaN when every value is NaN (partner off the L1/L2/L3 grid)."""
        stacked = np.abs(np.concatenate(arrays))
        return float(np.nanmax(stacked)) if np.isfinite(stacked).any() else float("nan")

    peak_by_key: dict[str, float] = {}
    for lbl in partner_keys_all_t4 + partner_keys_all_t5:
        if stim_type == "edge":
            peak_by_key[lbl] = _abs_peak(
                [traces[pol][d][lbl] for pol in edge_polarities for d in directions]  # type: ignore[index]
            )
        else:
            peak_by_key[lbl] = _abs_peak([traces[d][lbl] for d in directions])  # type: ignore[index]

    def _filter_plot_keys(keys: List[str]) -> List[str]:
        if min_trace_peak <= 0.0:
            return keys
        return [k for k in keys if peak_by_key[k] >= min_trace_peak]

    partner_keys_plot_t4 = _filter_plot_keys(partner_keys_all_t4)
    partner_keys_plot_t5 = _filter_plot_keys(partner_keys_all_t5)

    print(
        f"Presynaptic partners in COO: T4a<-{len(partner_keys_all_t4)}, T5a<-{len(partner_keys_all_t5)}; "
        f"plotting {len(partner_keys_plot_t4)}/{len(partner_keys_all_t4)} (T4) and "
        f"{len(partner_keys_plot_t5)}/{len(partner_keys_all_t5)} (T5) "
        f"(min-|trace-peak|={min_trace_peak}).",
    )

    fig_folder = model_folder / "figures" / "cell_input_traces" / stim_type
    fig_folder.mkdir(parents=True, exist_ok=True)

    stim_start_s = pre_stim
    _, n_in_cycle = _color_cycle()
    # Twin-axis proxy: steady-state rate f(Σ+b) of the summed plotted inputs,
    # with f the model's own nonlinearity.
    nl_name = type(model.nonlinearity).__name__
    proxy_label = f"{nl_name}(Σ+b)"
    proxy_ylabel = f"{nl_name}(Σ W·g_out·g_in·FR + b)"

    def _edge_stim_title(pol: str, direction: str) -> str:
        lab = "Light" if pol == "light" else "Dark"
        if direction == "right":
            return f"{lab} edge {VELOCITY_DEG_S:g}°/s → (right)"
        return f"{lab} edge {VELOCITY_DEG_S:g}°/s ↑ (up)"

    plot_jobs: List[Tuple[Any, str, str]] = []
    if stim_type == "edge":
        for pol in edge_polarities:
            for direction in directions:
                plot_jobs.append(
                    (
                        traces[pol][direction],  # type: ignore[index]
                        _edge_stim_title(pol, direction),
                        f"{pol}_edge_240deg_s_{direction}",
                    )
                )
    else:
        for direction in directions:
            plot_jobs.append((traces[direction], stim_labels[direction], output_basename(direction)))  # type: ignore[index]

    for tr_dict, stim_line, base in plot_jobs:
        fig_fr, (ax_t4, ax_t5) = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            sharex=True,
        )
        t4_y = tr_dict["T4a"]
        ax_t4.plot(time_s, t4_y, color="k", linewidth=1.5, label="T4a FR")
        if partner_keys_plot_t4:
            sum_linear_t4 = (
                np.nansum(
                    np.stack([tr_dict[k] for k in partner_keys_plot_t4], axis=0),
                    axis=0,
                )
                + t4_offset
            )
            t_lin = torch.tensor(sum_linear_t4, device=model.device, dtype=model.dtype)
            with torch.no_grad():
                sum_fr_proxy = model.nonlinearity(t_lin).cpu().numpy()
            ax_t4r = ax_t4.twinx()
            ax_t4r.plot(
                time_s,
                sum_fr_proxy,
                color="tab:gray",
                linestyle=":",
                linewidth=1.3,
                label=proxy_label,
            )
            ax_t4r.set_ylabel(proxy_ylabel)
            lines_l, lab_l = ax_t4.get_legend_handles_labels()
            lines_r, lab_r = ax_t4r.get_legend_handles_labels()
            ax_t4.legend(lines_l + lines_r, lab_l + lab_r, loc="best")
        else:
            ax_t4.legend(loc="best")
        if not np.any(np.isfinite(t4_y)):
            ax_t4.text(
                0.5,
                0.5,
                "T4a firing rate is all NaN — column not in predict readout set",
                transform=ax_t4.transAxes,
                ha="center",
                va="center",
            )
        ax_t4.axvline(stim_start_s, color="gray", linestyle="--", linewidth=0.8)
        ax_t4.set_ylabel("Firing rate")
        ax_t4.set_title(f"{stim_line}\nT4a neuron_idx={t4_idx}\n{column_ctx}")
        ax_t4.set_xlim(time_s[0], time_s[-1])

        t5_y = tr_dict["T5a"]
        ax_t5.plot(time_s, t5_y, color="tab:blue", linewidth=1.5, label="T5a FR")
        if partner_keys_plot_t5:
            sum_linear_t5 = (
                np.nansum(
                    np.stack([tr_dict[k] for k in partner_keys_plot_t5], axis=0),
                    axis=0,
                )
                + t5_offset
            )
            t_lin5 = torch.tensor(sum_linear_t5, device=model.device, dtype=model.dtype)
            with torch.no_grad():
                sum_fr_proxy5 = model.nonlinearity(t_lin5).cpu().numpy()
            ax_t5r = ax_t5.twinx()
            ax_t5r.plot(
                time_s,
                sum_fr_proxy5,
                color="tab:olive",
                linestyle=":",
                linewidth=1.3,
                label=proxy_label,
            )
            ax_t5r.set_ylabel(proxy_ylabel)
            lines_l5, lab_l5 = ax_t5.get_legend_handles_labels()
            lines_r5, lab_r5 = ax_t5r.get_legend_handles_labels()
            ax_t5.legend(lines_l5 + lines_r5, lab_l5 + lab_r5, loc="best")
        else:
            ax_t5.legend(loc="best")
        if not np.any(np.isfinite(t5_y)):
            ax_t5.text(
                0.5,
                0.5,
                "T5a firing rate is all NaN — column not in predict readout set",
                transform=ax_t5.transAxes,
                ha="center",
                va="center",
            )
        ax_t5.axvline(stim_start_s, color="gray", linestyle="--", linewidth=0.8)
        ax_t5.set_xlabel("Time (s)")
        ax_t5.set_ylabel("Firing rate")
        ax_t5.set_title(f"T5a neuron_idx={t5_idx}; same column as T4a (column {col})")
        fig_fr.tight_layout()
        path_fr = fig_folder / f"{base}_t4_t5_firing_rate.png"
        fig_fr.savefig(path_fr, dpi=150)
        plt.close(fig_fr)
        print(f"Saved {path_fr}")

        for post_slug, label_post, post_idx, pk_plot, pk_all, ylabel_paths in (
            (
                "t4a",
                "T4a",
                t4_idx,
                partner_keys_plot_t4,
                partner_keys_all_t4,
                "W·g_out(pre)·g_in(T4a)·FR",
            ),
            (
                "t5a",
                "T5a",
                t5_idx,
                partner_keys_plot_t5,
                partner_keys_all_t5,
                "W·g_out(pre)·g_in(T5a)·FR",
            ),
        ):
            fig_in, ax_in = plt.subplots(figsize=(10, 6))
            for j, lbl in enumerate(pk_plot):
                if lbl not in tr_dict:
                    continue
                y = tr_dict[lbl]
                color, ls = _color_and_linestyle(j, n_in_cycle)
                ax_in.plot(
                    time_s,
                    y,
                    label=legend_for_trace_key[lbl],
                    linewidth=1.2,
                    color=color,
                    linestyle=ls,
                )
            ax_in.axvline(stim_start_s, color="gray", linestyle=":", linewidth=0.8, label="Stim onset")
            ax_in.set_xlabel("Time (s)")
            ax_in.set_ylabel(ylabel_paths)
            ax_in.set_title(
                f"{stim_line}\n"
                f"Presynaptic paths onto {label_post} neuron_idx={post_idx} "
                f"(showing {len(pk_plot)}/{len(pk_all)} partners, min-|peak|={min_trace_peak})",
            )
            ncol = 2 if len(pk_plot) > 6 else 1
            ax_in.legend(fontsize=7, loc="best", ncol=ncol)
            ax_in.set_xlim(time_s[0], time_s[-1])
            fig_in.tight_layout()
            path_in = fig_folder / f"{base}_{post_slug}_input_paths.png"
            fig_in.savefig(path_in, dpi=150)
            plt.close(fig_in)
            print(f"Saved {path_in}")

    def _partner_summary_lines(
        label: str,
        post_idx: int,
        g_in: float,
        offset: float,
        partners: List[Tuple[str, int]],
        path_scales: dict[int, float],
        pk_plot: List[str],
        pk_all: List[str],
    ) -> List[str]:
        lines_loc = [
            stim_summary_line,
            f"{label}: neuron_index={post_idx}, {column_ctx}",
            f"g_in({label}) single factor applied to all listed inputs: {g_in:.6g}",
            f"b({label}) offset (inside {proxy_label} on the FR plot): {offset:.6g}",
            f"Connectome: {len(partners)} distinct presynaptic neurons (all types) synapse onto this {label}; "
            f"each is a separate column/neighbor. Plotted {len(pk_plot)}/{len(pk_all)} "
            f"with max|trace|>={min_trace_peak} (use --min-trace-peak 0 for all).",
            f"Presynaptic partners (every unique pre in synapse_counts_coo onto this {label}):",
        ]
        if not partners:
            lines_loc.append("  (none found in synapse_counts_coo)")
        for ct, ni in partners:
            ci = model.regions['optic_lobe'].column_indices_per_neuron[ni]
            w = _synapse_weight_post_pre(model, post_idx, ni)
            g_out = presynaptic_output_gain(model, ni)
            pscale = path_scales[ni]
            lines_loc.append(
                f"  {ct} neuron_idx={ni} column={int(ci) if not np.isnan(ci) else 'nan'} "
                f"W={w:.6g} g_out={g_out:.6g} W*g_out*g_in({label})={pscale:.6g}"
            )
        return lines_loc

    # Text summaries are not figures, so they go under analysis_data/.
    summary_folder = model_folder / "analysis_data" / "cell_input_traces" / stim_type
    summary_folder.mkdir(parents=True, exist_ok=True)
    for fname, label, post_idx, g_in, off, partners, scales, pk_plot, pk_all in (
        (
            "t4a_partners.txt",
            "T4a",
            t4_idx,
            g_in_t4,
            t4_offset,
            partners_t4,
            partner_path_scales_t4,
            partner_keys_plot_t4,
            partner_keys_all_t4,
        ),
        (
            "t5a_partners.txt",
            "T5a",
            t5_idx,
            g_in_t5,
            t5_offset,
            partners_t5,
            partner_path_scales_t5,
            partner_keys_plot_t5,
            partner_keys_all_t5,
        ),
    ):
        summary_path = summary_folder / fname
        summary_path.write_text(
            "\n".join(
                _partner_summary_lines(
                    label,
                    post_idx,
                    g_in,
                    off,
                    partners,
                    scales,
                    pk_plot,
                    pk_all,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T4a and T5a + presynaptic time traces (edge or sine stimulus).",
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt",
    )
    parser.add_argument(
        "--stim-type",
        type=str,
        choices=("edge", "sine"),
        default="edge",
        help="Stimulus family: moving light+dark edges (240 deg/s, right & up) or drifting sine grating. Default: %(default)s.",
    )
    parser.add_argument(
        "--temporal-frequency",
        type=float,
        default=DEFAULT_TEMPORAL_FREQ_HZ,
        metavar="HZ",
        help="(sine only) Temporal frequency in Hz. Default: %(default)s.",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=DEFAULT_WAVELENGTH_DEG,
        metavar="DEG",
        help="(sine only) Spatial wavelength in degrees. Default: %(default)s.",
    )
    parser.add_argument(
        "--stim-time",
        type=float,
        default=STIM_TIME_S_SINE_DEFAULT,
        metavar="S",
        help="(sine only) Stimulus duration in seconds (after pre-stim). Default: %(default)s.",
    )
    parser.add_argument(
        "--column-offset",
        type=int,
        default=0,
        metavar="N",
        help="After sorting shared L1/L2/L3 columns by azimuth phi, use the middle column (index "
        "(n-1)//2) plus this integer offset. T4a and T5a are taken from the same column. Default: %(default)s.",
    )
    parser.add_argument(
        "--min-trace-peak",
        type=float,
        default=1e-9,
        metavar="X",
        help="Only plot presynaptic traces whose max |value| (over time and both directions) is >= X. "
        "Use 0 for every partner. Default: %(default)s.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    model = nm.BaseRnn.load(model_path)
    if not hasattr(model, 'input_strength_log'):
        sys.exit(f"cell_input_traces decomposes inputs through per-cell-type gains, "
                 f"which {type(model).__name__} does not have.")

    run_t4_input_traces(
        stim_type=args.stim_type,
        model_folder=model_folder,
        model=model,
        column_offset=args.column_offset,
        min_trace_peak=args.min_trace_peak,
        temporal_frequency_hz=args.temporal_frequency,
        wavelength_deg=args.wavelength,
        stim_time_sine=args.stim_time,
    )


if __name__ == "__main__":
    main()
