"""
Time traces of a central T4a and **all** presynaptic partners (every unique pre in
``synapse_counts_coo`` onto that T4a).  Stimulus is chosen with ``--stim-type``:

* **edge** — same as ``edge_sweep``: light **and** dark edges at 240°/s, right & up,
  FOV 180°, 0.5 s pre-stim (figures under ``figures/t4_input_traces/edge/``).
* **sine** — same as ``sine_wave_sweep``: drifting sine (default 4 Hz, 30° wavelength,
  1 s stim, 0.5 s pre-stim; figures under ``figures/t4_input_traces/sine/``).

Each presynaptic trace is ``W·g_out·g_in·FR``; the T4a FR panel includes
``(1-ρ)·softplus(Σ+b)`` on a twin axis.  Partners outside the L1/L2/L3 grid yield
NaN FR and thus NaN scaled paths.

Usage:
    python analysis/t4_input_traces.py models/100hz_best_drone/ --stim-type edge
    python analysis/t4_input_traces.py models/100hz_best_drone/ --stim-type sine
    python analysis/t4_input_traces.py models/100hz_best_drone/ --stim-type sine --wavelength 30 --temporal-frequency 4
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import data_utilities as du
import neural_models as nm
from stimulus_utilities import make_lr_edge, make_ud_edge, make_lr_sine_wave, make_ud_sine_wave

# --- Edge defaults (match edge_sweep / former t4_input_edge_traces) ---
VELOCITY_DEG_S = 240.0
FOV_DEG = 180.0
PRE_STIM_TIME_S = 0.5

# --- Sine defaults (match sine_wave_sweep / former t4_input_sine_traces) ---
STIM_TIME_S_SINE_DEFAULT = 1.0
DEFAULT_TEMPORAL_FREQ_HZ = 4.0
DEFAULT_WAVELENGTH_DEG = 30.0


def generate_stimulus_edge(
    direction: str,
    sample_phi,
    sample_theta,
    velocity: float,
    edge_polarity: str,
    dt: float,
    stim_time: float,
    pre_stim_time: float,
):
    """Same contract as :func:`analysis.edge_sweep.generate_stimulus`."""
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
    return edge[None, :, :]


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
    """Same convention as :func:`analysis.sine_wave_sweep.generate_stimulus`."""
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


def pick_central_t4a(model: nm.NeuralModel, input_cols: np.ndarray) -> int:
    """Return neuron index of a T4a whose column is in *input_cols* and nearest median (phi, theta)."""
    input_cols = np.unique(input_cols)
    phi_all = model.column_phi_theta[:, 0]
    theta_all = model.column_phi_theta[:, 1]
    phi_med = np.nanmedian(phi_all[input_cols.astype(int)])
    theta_med = np.nanmedian(theta_all[input_cols.astype(int)])

    mask = (
        (model.cell_types == "T4a")
        & ~np.isnan(model.column_indices)
        & np.isin(model.column_indices, input_cols)
    )
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        raise RuntimeError(
            "No T4a with column in L1/L2/L3 shared columns — cannot get non-NaN readouts from predict()."
        )

    best_ni = None
    best_d = np.inf
    for ni in candidates:
        ci = int(model.column_indices[ni])
        d = (phi_all[ci] - phi_med) ** 2 + (theta_all[ci] - theta_med) ** 2
        if d < best_d:
            best_d = d
            best_ni = ni
    return int(best_ni)


def list_t4a_neuron_indices_by_azimuth(model: nm.NeuralModel, input_cols: np.ndarray) -> List[int]:
    """All T4a in *input_cols*, sorted by column azimuth φ then column index."""
    input_cols = np.unique(input_cols)
    phi_all = model.column_phi_theta[:, 0]
    mask = (
        (model.cell_types == "T4a")
        & ~np.isnan(model.column_indices)
        & np.isin(model.column_indices, input_cols)
    )
    indices = np.where(mask)[0]
    keyed = [(ni, float(phi_all[int(model.column_indices[ni])]), int(model.column_indices[ni])) for ni in indices]
    keyed.sort(key=lambda t: (t[1], t[2], t[0]))
    return [t[0] for t in keyed]


def resolve_t4a_neuron(
    model: nm.NeuralModel,
    input_cols: np.ndarray,
    *,
    neuron_index: Optional[int] = None,
    azimuth_steps: int = 0,
) -> int:
    """Pick T4a: explicit index or central ± azimuth steps along φ order."""
    if neuron_index is not None:
        if neuron_index < 0 or neuron_index >= model.num_neurons:
            raise ValueError(f"t4 neuron_index {neuron_index} out of range [0, {model.num_neurons}).")
        if model.cell_types[neuron_index] != "T4a":
            raise ValueError(f"Neuron {neuron_index} is {model.cell_types[neuron_index]!r}, not T4a.")
        ci = model.column_indices[neuron_index]
        uc = np.unique(input_cols)
        if np.isnan(ci) or not np.isin(int(ci), uc):
            raise ValueError(
                f"T4a neuron {neuron_index} column {ci} is not in L1/L2/L3 shared columns — predict() readouts would be NaN.",
            )
        return int(neuron_index)

    ordered = list_t4a_neuron_indices_by_azimuth(model, input_cols)
    if not ordered:
        raise RuntimeError("No T4a in shared input columns.")

    ref = pick_central_t4a(model, input_cols)
    k = ordered.index(ref)
    k_new = k + azimuth_steps
    if k_new < 0 or k_new >= len(ordered):
        raise ValueError(
            f"t4 azimuth_steps={azimuth_steps} → index {k_new} out of range [0, {len(ordered)}) "
            f"(ref T4a at sorted index {k}/{len(ordered)-1}).",
        )
    return int(ordered[k_new])


def presynaptic_indices_all(model: nm.NeuralModel, post_idx: int) -> List[Tuple[str, int]]:
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


def trace_for_neuron(pred: dict, model: nm.NeuralModel, cell_type: str, neuron_idx: int) -> np.ndarray:
    """Firing-rate time series (time,) for one neuron, from predict outputs."""
    ci = model.column_indices[neuron_idx]
    if np.isnan(ci):
        return np.full(pred["firing_rate"][cell_type].shape[1], np.nan)
    col = int(ci)
    return pred["firing_rate"][cell_type][0, :, col].detach().cpu().numpy()


def _synapse_weight_post_pre(model: nm.NeuralModel, post_idx: int, pre_idx: int) -> float:
    """Signed sparse weight W[post, pre] (same layout as ``torch.sparse.mm`` in ``forward``)."""
    if isinstance(model, nm.ConductanceModel):
        nt = model.transmitters[pre_idx]
        ri = int(np.where(model.unique_transmitters == nt)[0][0])
        w = model.synapse_counts[ri]
    else:
        w = model.synapse_counts
    co = w.coalesce()
    idx = co.indices()
    val = co.values()
    m = (idx[0] == post_idx) & (idx[1] == pre_idx)
    if not m.any():
        return 0.0
    return float(val[m].sum().cpu().item())


def postsynaptic_t4a_input_gain(model: nm.NeuralModel, post_idx: int) -> float:
    """Single scalar g_in for T4a (postsynaptic), applied to every presynaptic path."""
    with torch.no_grad():
        nti = model.neuron_to_type_index
        exp_in = model.input_strength_log.exp()
        if isinstance(model, nm.ConductanceModel):
            return float(exp_in[:, nti[post_idx]].mean().cpu().item())
        return float(exp_in[nti[post_idx]].cpu().item())


def postsynaptic_t4a_offset(model: nm.NeuralModel, post_idx: int) -> float:
    """Per-cell-type bias *b* in ``z + b``; 0 if the model has no ``offset``."""
    if not hasattr(model, "offset"):
        return 0.0
    with torch.no_grad():
        cti = int(model.neuron_to_type_index[post_idx].item())
        return float(model.offset[cti].cpu().item())


def postsynaptic_t4a_persistence(model: nm.NeuralModel, post_idx: int) -> float:
    """ρ = sigmoid(persistence_logit) for this cell type. If absent, return 0 so ``(1-ρ)=1``."""
    if not hasattr(model, "persistence_logit"):
        return 0.0
    with torch.no_grad():
        cti = int(model.neuron_to_type_index[post_idx].item())
        return float(model.persistence_logit.sigmoid()[cti].cpu().item())


def presynaptic_output_gain(model: nm.NeuralModel, pre_idx: int) -> float:
    """g_out for the presynaptic neuron’s cell type."""
    with torch.no_grad():
        return float(model.output_strength_log.exp()[model.neuron_to_type_index[pre_idx]].cpu().item())


def presynaptic_path_scale(model: nm.NeuralModel, pre_idx: int, post_idx: int) -> float:
    """``W(post,pre) * g_out(pre) * g_in(T4a)`` — no per-presynapse input row."""
    w = _synapse_weight_post_pre(model, post_idx, pre_idx)
    g_out = presynaptic_output_gain(model, pre_idx)
    g_in_t4a = postsynaptic_t4a_input_gain(model, post_idx)
    return w * g_out * g_in_t4a


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
    model: nm.NeuralModel,
    model_config: dict,
    t4_neuron_index: Optional[int],
    t4_azimuth_steps: int,
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

    dt = model_config["model"]["dt"]
    sample_phi = model.column_phi_theta[:, 0]
    sample_theta = model.column_phi_theta[:, 1]

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

    _, input_cols_arr = model.get_shared_columns(["L1", "L2", "L3"])
    input_cols = np.unique(input_cols_arr)

    t4_idx = resolve_t4a_neuron(
        model,
        input_cols,
        neuron_index=t4_neuron_index,
        azimuth_steps=t4_azimuth_steps,
    )
    t4_col = int(model.column_indices[t4_idx])
    partners = presynaptic_indices_all(model, t4_idx)
    g_in_t4a = postsynaptic_t4a_input_gain(model, t4_idx)
    t4_offset = postsynaptic_t4a_offset(model, t4_idx)
    t4_rho = postsynaptic_t4a_persistence(model, t4_idx)
    partner_path_scales: dict[int, float] = {
        ni: presynaptic_path_scale(model, ni, t4_idx) for _, ni in partners
    }

    n_of_type = Counter(ct for ct, _ in partners)
    idx_among_type = defaultdict(int)
    plot_specs: List[Tuple[str, str, str, Optional[int]]] = [("T4a", "T4a", "T4a", None)]
    legend_for_trace_key: dict[str, str] = {"T4a": "T4a"}
    for ct, ni in partners:
        idx_among_type[ct] += 1
        trace_key = str(ni)
        leg = short_presynaptic_legend_name(ct, idx_among_type[ct], n_of_type[ct])
        plot_specs.append((trace_key, leg, ct, ni))
        legend_for_trace_key[trace_key] = leg

    output_types = sorted({spec[2] for spec in plot_specs})
    output_types = [t for t in output_types if t in model.unique_cell_types]
    if not output_types:
        raise RuntimeError("No output cell types available to plot.")

    directions = ["right", "up"]
    edge_polarities = ("light", "dark")

    if stim_type == "edge":
        _probe = generate_stimulus_edge(
            "right",
            sample_phi,
            sample_theta,
            VELOCITY_DEG_S,
            "light",
            dt,
            stim_time,
            pre_stim,
        )
        num_time = _probe.shape[1]
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
                assert stim_np.shape[1] == num_time
                batch = torch.tensor(stim_np, device=model.device, dtype=model.dtype)
                with torch.no_grad():
                    pred = model.predict(
                        batch,
                        input_neurons=["L1", "L2", "L3"],
                        output_neurons=output_types,
                    )["outputs"]

                for trace_key, _leg, ct, ni in plot_specs:
                    if ct not in output_types:
                        continue
                    if ni is None:
                        tr = pred["firing_rate"][ct][0, :, t4_col].detach().cpu().numpy()
                    else:
                        tr = trace_for_neuron(pred, model, ct, ni) * partner_path_scales[ni]
                    traces_edge[pol][direction][trace_key] = tr
        traces = traces_edge  # type: ignore[assignment]
    else:
        _probe = gen_stim("right")
        num_time = _probe.shape[1]
        traces_sin: dict[str, dict[str, np.ndarray]] = {d: {} for d in directions}
        for direction in directions:
            stim_np = gen_stim(direction)
            assert stim_np.shape[1] == num_time
            batch = torch.tensor(stim_np, device=model.device, dtype=model.dtype)
            with torch.no_grad():
                pred = model.predict(
                    batch,
                    input_neurons=["L1", "L2", "L3"],
                    output_neurons=output_types,
                )["outputs"]

            for trace_key, _leg, ct, ni in plot_specs:
                if ct not in output_types:
                    continue
                if ni is None:
                    tr = pred["firing_rate"][ct][0, :, t4_col].detach().cpu().numpy()
                else:
                    tr = trace_for_neuron(pred, model, ct, ni) * partner_path_scales[ni]
                traces_sin[direction][trace_key] = tr
        traces = traces_sin  # type: ignore[assignment]

    time_s = np.arange(num_time, dtype=np.float64) * dt

    partner_keys_all = [tk for tk, _l, ct, ni in plot_specs if ni is not None]
    peak_by_key: dict[str, float] = {}
    for lbl in partner_keys_all:
        if stim_type == "edge":
            peak_by_key[lbl] = max(
                float(np.nanmax(np.abs(traces[pol][d][lbl])))  # type: ignore[index]
                for pol in edge_polarities
                for d in directions
            )
        else:
            peak_by_key[lbl] = max(
                float(np.nanmax(np.abs(traces[d][lbl]))) for d in directions  # type: ignore[index]
            )
    if min_trace_peak <= 0.0:
        partner_keys_plot = partner_keys_all
    else:
        partner_keys_plot = [k for k in partner_keys_all if peak_by_key[k] >= min_trace_peak]

    print(
        f"Presynaptic partners in COO: {len(partner_keys_all)}; "
        f"plotting {len(partner_keys_plot)} (min-|trace-peak|={min_trace_peak}).",
    )

    fig_folder = model_folder / "figures" / "t4_input_traces" / stim_type
    fig_folder.mkdir(parents=True, exist_ok=True)

    stim_start_s = pre_stim
    _, n_in_cycle = _color_cycle()

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
        fig_t4, ax_t4 = plt.subplots(figsize=(10, 4))
        t4_y = tr_dict["T4a"]
        ax_t4.plot(time_s, t4_y, color="k", linewidth=1.5, label="T4a FR")
        if partner_keys_plot:
            sum_linear = (
                np.nansum(
                    np.stack([tr_dict[k] for k in partner_keys_plot], axis=0),
                    axis=0,
                )
                + t4_offset
            )
            t_lin = torch.tensor(sum_linear, device=model.device, dtype=model.dtype)
            with torch.no_grad():
                sum_fr_proxy = ((1.0 - t4_rho) * torch.nn.functional.softplus(t_lin)).cpu().numpy()
            ax_t4r = ax_t4.twinx()
            ax_t4r.plot(
                time_s,
                sum_fr_proxy,
                color="tab:gray",
                linestyle=":",
                linewidth=1.3,
                label=f"(1-ρ)·softplus(Σ+b), ρ={t4_rho:.3f}",
            )
            ax_t4r.set_ylabel("(1-ρ)·softplus(Σ W·g_out·g_in·FR + b)")
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
        ax_t4.set_xlabel("Time (s)")
        ax_t4.set_ylabel("Firing rate")
        ax_t4.set_title(
            f"{stim_line}\n"
            f"T4a neuron_idx={t4_idx}, column={t4_col} (L1/L2/L3 overlap column)",
        )
        ax_t4.set_xlim(time_s[0], time_s[-1])
        fig_t4.tight_layout()
        path_t4 = fig_folder / f"{base}_t4a_firing_rate.png"
        fig_t4.savefig(path_t4, dpi=150)
        plt.close(fig_t4)
        print(f"Saved {path_t4}")

        fig_in, ax_in = plt.subplots(figsize=(10, 6))
        for j, lbl in enumerate(partner_keys_plot):
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
        ax_in.set_ylabel("W·g_out(pre)·g_in(T4a)·FR")
        ax_in.set_title(
            f"{stim_line}\n"
            f"Presynaptic paths onto T4a neuron_idx={t4_idx} "
            f"(showing {len(partner_keys_plot)}/{len(partner_keys_all)} partners, min-|peak|={min_trace_peak})",
        )
        ncol = 2 if len(partner_keys_plot) > 6 else 1
        ax_in.legend(fontsize=7, loc="best", ncol=ncol)
        ax_in.set_xlim(time_s[0], time_s[-1])
        fig_in.tight_layout()
        path_in = fig_folder / f"{base}_t4a_input_paths.png"
        fig_in.savefig(path_in, dpi=150)
        plt.close(fig_in)
        print(f"Saved {path_in}")

    sel_note = (
        f"explicit --t4-neuron-index={t4_neuron_index}"
        if t4_neuron_index is not None
        else f"central T4a + azimuth_steps={t4_azimuth_steps}"
    )
    lines = [
        stim_summary_line,
        f"T4a ({sel_note}): neuron_index={t4_idx}, column={t4_col}",
        f"g_in(T4a) single factor applied to all listed inputs: {g_in_t4a:.6g}",
        f"b(T4a) offset (inside Σ+b before softplus on FR plot): {t4_offset:.6g}",
        f"ρ(T4a) persistence (sigmoid of persistence_logit): {t4_rho:.6g}",
        f"Connectome: {len(partners)} distinct presynaptic neurons (all types) synapse onto this T4a; "
        f"each is a separate column/neighbor. Plotted {len(partner_keys_plot)}/{len(partner_keys_all)} "
        f"with max|trace|>={min_trace_peak} (use --min-trace-peak 0 for all).",
        "Presynaptic partners (every unique pre in synapse_counts_coo onto this T4a):",
    ]
    if not partners:
        lines.append("  (none found in synapse_counts_coo)")
    for ct, ni in partners:
        ci = model.column_indices[ni]
        w = _synapse_weight_post_pre(model, t4_idx, ni)
        g_out = presynaptic_output_gain(model, ni)
        pscale = partner_path_scales[ni]
        lines.append(
            f"  {ct} neuron_idx={ni} column={int(ci) if not np.isnan(ci) else 'nan'} "
            f"W={w:.6g} g_out={g_out:.6g} W*g_out*g_in(T4a)={pscale:.6g}"
        )
    summary_path = fig_folder / "t4a_partners.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T4a + presynaptic time traces (edge or sine stimulus).",
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder (containing model.pt and config.toml)",
    )
    parser.add_argument(
        "--stim-type",
        type=str,
        choices=("edge", "sine"),
        default="edge",
        help="Stimulus family: moving light+dark edges (240°/s, right & up) or drifting sine grating. Default: %(default)s.",
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
        "--t4-neuron-index",
        type=int,
        default=None,
        metavar="N",
        help="Plot this T4a neuron index (must be in L1/L2/L3 shared columns). Overrides --t4-azimuth-steps.",
    )
    parser.add_argument(
        "--t4-azimuth-steps",
        type=int,
        default=0,
        help="Start from the usual central T4a, then move N steps along columns sorted by azimuth φ.",
    )
    parser.add_argument(
        "--min-trace-peak",
        type=float,
        default=1e-9,
        metavar="X",
        help="Only plot presynaptic traces whose max |value| (over time and both directions) is ≥ X. "
        "Use 0 for every partner. Default: %(default)s.",
    )
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    model_config = du.load_config(model_folder / "config.toml")
    model = nm.NeuralModel.load(model_folder / "model.pt")

    run_t4_input_traces(
        stim_type=args.stim_type,
        model_folder=model_folder,
        model=model,
        model_config=model_config,
        t4_neuron_index=args.t4_neuron_index,
        t4_azimuth_steps=args.t4_azimuth_steps,
        min_trace_peak=args.min_trace_peak,
        temporal_frequency_hz=args.temporal_frequency,
        wavelength_deg=args.wavelength,
        stim_time_sine=args.stim_time,
    )


if __name__ == "__main__":
    main()
