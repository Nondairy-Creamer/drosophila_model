"""White noise analysis: recover a model's linear spatiotemporal filter via
spike-triggered average (STA).

Presents i.i.d. Gaussian white noise independently at each column (ommatidium),
runs the model, and computes the STA pooled across convolutional units.

Because the column-level stimulus is exactly white, the STA is the optimal
linear filter estimate (Wiener filter) without matrix inversion or
regularization. Pooling across convolutional units exploits the shared kernel
to massively overdetermine the estimate.

Usage:
    python analysis/white_noise.py models/best_cells_30hz/
    python analysis/white_noise.py models/best_cells_30hz/ --cell-types L1 L2 L3
    python analysis/white_noise.py models/best_cells_30hz/ --duration 300 --num-lags 20

Outputs in ``output_dir`` include ``temporal_tau_mean_f2.png``,
``temporal_tau_rms_f2.png``, ``temporal_tau_peak_abs.png`` (number-line
summaries), plus ``tau_rms_ms``, ``tau_mean_ms``, ``tau_peak_abs_ms`` in
``sta_data.npz`` (center-column STA; peak lag is argmax of ``|f(τ)|``).
``tau_rms_ms`` is the **standard deviation** of lag (ms), :math:`\\sigma_\\tau
=\\sqrt{\\mathrm{Var}[\\tau]}`, not the variance :math:`\\sigma_\\tau^2`;
:math:`\\tau` is distributed as :math:`p(\\tau)\\propto f(\\tau)^2` over lags
(normalized), not over stimulus intensity.
"""
import argparse
import sys
from pathlib import Path


import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import model_layers as ml

# ── Stimulus parameters ─────────────────────────────────────────────
STIM_DURATION_S = 20.0     # stimulus duration (seconds)
CONTRAST_STD = 0.15        # RMS contrast of column-level stimulus
SEED = 42

# ── Analysis parameters ─────────────────────────────────────────────
# Estimation budget: the STA has K_spatial × num_lags parameters.
# Each of P convolutional units contributes T_eff = T - num_lags
# independent measurements, so the effective sample count is P × T_eff.
# Overdetermination ratio = P × T_eff / (K_spatial × num_lags).
#
# Example at 30 Hz with defaults:
#   K_spatial = 37 (FILTER_SIZE=7), num_lags = 9 (300 ms), params = 333
#   P ≈ 800 columns, T = 600 frames (20 s) → P×T_eff ≈ 473,000
#   Ratio ≈ 1400×  overdetermined — 20 s is more than sufficient.
STA_DURATION_MS = 200.0    # STA window duration in milliseconds
FILTER_SIZE = 9            # hex grid diameter for spatial filter extraction
CHUNK_SIZE = 3000          # frames per model.predict() call


def compute_pooled_sta(stimulus, firing_rates, nearest_columns, num_lags):
    """Compute STA pooled across convolutional units.

    For white noise stimulus the STA directly gives the optimal linear
    filter::

        STA(k, tau) = (1 / sigma^2) * mean_{t,c} r(t,c) * s(t-tau, n_k(c))

    where s is mean-centered, sigma^2 is the stimulus variance, and n_k(c)
    is the k-th hex neighbor of column c.

    Args:
        stimulus: (T, num_columns_all) full stimulus (will be mean-centered).
        firing_rates: (T, num_valid_cols) firing rates for valid columns.
        nearest_columns: (num_valid_cols, K) neighbor indices into the full
            column set.  -1 indicates a missing neighbor (boundary).
        num_lags: number of temporal lags.

    Returns:
        sta: (K, num_lags) pooled spike-triggered average.
    """
    T = firing_rates.shape[0]
    K = nearest_columns.shape[1]

    stim_centered = stimulus - stimulus.mean()
    stim_var = (stim_centered ** 2).mean()

    valid_masks = nearest_columns >= 0
    safe_nn = nearest_columns.copy()
    safe_nn[~valid_masks] = 0

    sta = np.zeros((K, num_lags), dtype=np.float64)

    for tau in range(num_lags):
        T_eff = T - tau
        if T_eff <= 0:
            continue
        r = firing_rates[tau:tau + T_eff]
        s = stim_centered[:T_eff]

        for k in range(K):
            valid = valid_masks[:, k]
            nn_k = safe_nn[valid, k]

            sta[k, tau] = (r[:, valid] * s[:, nn_k]).mean()

    if stim_var > 0:
        sta /= stim_var

    return sta


def squared_kernel_temporal_moments(time_lags_ms, temporal_profile):
    """Temporal moments of lag :math:`\\tau` under :math:`p(\\tau)\\propto f(\\tau)^2`.

    Here :math:`f(\\tau)` is the STA along time at fixed (center) space;
    :math:`p` is normalized over **lags** :math:`\\tau`. We compute
    :math:`\\mathrm{Var}[\\tau]=\\sum_\\tau p(\\tau)(\\tau-\\mu)^2` (variance of
    lag in ms²) and return its **square root** :math:`\\sigma_\\tau` so the
    second return value is a **standard deviation in ms**, not ms². It is **not**
    the variance of :math:`f^2` across intensity/contrast, nor spatial variance.

    Squaring :math:`f` removes sign cancellation for biphasic filters.
    ``mu_ms`` is :math:`\\mathbb{E}[\\tau]` under the same :math:`p`.

    Returns:
        (mu_ms, sigma_ms): ``sigma_ms`` is :math:`\\sigma_\\tau=\\sqrt{\\mathrm{Var}[\\tau]}`
        in **milliseconds** (standard deviation of lag), or ``(nan, nan)`` if no energy.
    """
    w = np.asarray(temporal_profile, dtype=np.float64) ** 2
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.nan, np.nan
    t = np.asarray(time_lags_ms, dtype=np.float64)
    p = w / s
    mu = (p * t).sum()
    var = (p * (t - mu) ** 2).sum()
    var = max(var, 0.0)
    sigma = np.sqrt(var)
    return float(mu), float(sigma)


def peak_abs_lag_ms(time_lags_ms, temporal_profile):
    """Time lag (ms) where ``|f|`` is largest on the center-column STA."""
    prof = np.asarray(temporal_profile, dtype=np.float64)
    t = np.asarray(time_lags_ms, dtype=np.float64)
    if prof.size == 0 or not np.any(np.isfinite(prof)):
        return np.nan
    i = int(np.nanargmax(np.abs(prof)))
    return float(t[i])


def plot_temporal_moment_number_line(
    cell_to_ms,
    output_path,
    *,
    xlabel,
    title,
    color='C0',
    clip_xmin_zero=False,
):
    """One point per cell type on a horizontal axis (ms)."""
    items = [(ct, cell_to_ms[ct])
             for ct in sorted(cell_to_ms)
             if np.isfinite(cell_to_ms[ct])]
    if not items:
        return

    names = [x[0] for x in items]
    xs = np.array([x[1] for x in items], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(10.0, len(items) * 0.35), 3.2))
    span = xs.max() - xs.min()
    pad = max(2.0, 0.05 * (span + 1e-6))
    lo = xs.min() - pad
    hi = xs.max() + pad
    if clip_xmin_zero:
        lo = max(0.0, lo)
    ax.hlines(0, lo, hi, colors='k', linewidth=0.8, zorder=0)
    ax.scatter(xs, np.zeros(len(xs)), s=36, zorder=2, color=color)

    order = np.argsort(xs)
    for rank, j in enumerate(order):
        ct = names[j]
        x = xs[j]
        voff = 14 + (rank % 4) * 11
        ax.annotate(
            ct,
            (x, 0),
            xytext=(0, voff),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=7,
            rotation=0,
            zorder=3,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.12, 0.12)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _build_xt_profile(sta, pq_offsets):
    """Collapse 2D hex STA to a 1D horizontal profile for an x-t plot.

    Selects five bands perpendicular to the hex ``p+q`` direction
    (``|p+q| <= 2``). Positions sharing the same horizontal coordinate
    (``p - q``) are averaged; bins are sorted left-to-right.

    Returns (xt, x_positions):
        xt : (n_x, n_lags) array
        x_positions : (n_x,) sorted unique x coordinates
    """
    pq = pq_offsets.astype(np.float64)
    row_sum = (pq[:, 0] + pq[:, 1]).astype(int)
    mask = np.abs(row_sum) <= 2
    x_coords = np.sqrt(3) / 2 * (pq[mask, 0] - pq[mask, 1])
    sta_sel = sta[mask]

    unique_x = np.unique(np.round(x_coords, 6))
    xt = np.zeros((len(unique_x), sta.shape[1]))
    for i, ux in enumerate(unique_x):
        hits = np.abs(x_coords - ux) < 1e-5
        xt[i] = sta_sel[hits].mean(axis=0)
    return xt, unique_x


def plot_sta(sta, pq_offsets, dt, cell_type, output_path):
    """Plot the recovered spatiotemporal filter as an x-t image.

    Top panel: horizontal position vs time-lag heatmap (averaged over
    the middle five rows of the hex kernel).
    Bottom panel: temporal profile at the center position.
    """
    num_lags = sta.shape[1]
    time_lags_ms = np.arange(num_lags) * dt * 1000
    center_idx = np.argmin(np.abs(pq_offsets).sum(axis=1))

    xt, x_pos = _build_xt_profile(sta, pq_offsets)
    vmax = np.abs(xt).max() or 1

    xy = ml.HexGridConv.pq_to_xy(pq_offsets)
    peak_lag = np.unravel_index(np.argmax(np.abs(sta)), sta.shape)[1]
    spatial_peak = sta[:, peak_lag]
    vmax_spatial = np.abs(spatial_peak).max() or 1

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.4, wspace=0.35)

    # Top-left: x-t heatmap (space on x-axis, time on y-axis)
    ax_xt = fig.add_subplot(gs[0, 0])
    im = ax_xt.imshow(
        xt.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        origin='upper',
        extent=[-0.5, len(x_pos) - 0.5, time_lags_ms[-1], time_lags_ms[0]],
    )
    ax_xt.set_xticks(range(len(x_pos)))
    ax_xt.set_xticklabels([f'{x:.1f}' for x in x_pos])
    ax_xt.set_xlabel('Horizontal position')
    ax_xt.set_ylabel('Time lag (ms)')
    fig.colorbar(im, ax=ax_xt, fraction=0.046, pad=0.04)

    # Top-right: 2D hex spatial filter at lag 0
    ax_hex = fig.add_subplot(gs[0, 1])
    sc = ax_hex.scatter(
        xy[:, 0], xy[:, 1], c=spatial_peak, cmap='RdBu_r',
        s=200, vmin=-vmax_spatial, vmax=vmax_spatial,
        edgecolors='gray', linewidths=0.3,
    )
    ax_hex.set_aspect('equal')
    ax_hex.set_title(f'Spatial filter ({time_lags_ms[peak_lag]:.0f} ms)',
                     fontsize=10)
    ax_hex.set_xticks([])
    ax_hex.set_yticks([])
    fig.colorbar(sc, ax=ax_hex, fraction=0.046, pad=0.04)

    # Bottom: temporal profile at center
    ax_t = fig.add_subplot(gs[1, :])
    ax_t.plot(time_lags_ms, sta[center_idx], 'k-o', markersize=4,
              label='STA (center)')
    ax_t.axhline(0, color='gray', linewidth=0.5)
    ax_t.set_xlabel('Time lag (ms)')
    ax_t.set_ylabel('Filter weight')

    fig.suptitle(f'White noise STA: {cell_type}', fontsize=14, y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


DEFAULT_CELL_TYPES = [
    'L1', 'L2', 'L3', 'L4', 'L5',
    'C2', 'C3', 'CT1_L', 'CT1_M',
    'Mi1', 'Tm3', 'Mi9', 'Mi4',
    'T4a', 'T4b', 'T4c', 'T4d',
    'T5a', 'T5b', 'T5c', 'T5d',
    'Tm1', 'Tm2', 'Tm4', 'Tm9',
    'T1', 'T2', 'T3',
]


def run_white_noise(model, cell_types=None, silenced_neurons=None,
                    output_dir=None, duration=STIM_DURATION_S,
                    sta_duration_ms=STA_DURATION_MS, filter_size=FILTER_SIZE):
    """Run white noise STA analysis on *model* and return extracted filters.

    Args:
        model: A loaded ``BaseRnn`` instance (already on the desired
            device and in eval mode).
        cell_types: List of cell type names to analyse. Defaults to
            ``DEFAULT_CELL_TYPES``.
        silenced_neurons: Optional list of neuron names to silence before
            running the model.  ``output_strength_log`` is saved and
            restored automatically.
        output_dir: Path to write per-cell STA figures, ``sta_data.npz``
            (includes ``tau_mean_ms``, ``tau_rms_ms`` — standard deviation of
            lag in ms, :math:`\\sqrt{\\mathrm{Var}[\\tau]}`, under
            :math:`p(\\tau)\\propto f^2` over **lags**), and
            ``temporal_tau_mean_f2.png``,
            ``temporal_tau_rms_f2.png``, ``temporal_tau_peak_abs.png``.
            If *None*, figure saving is skipped.
        duration: Stimulus duration in seconds.
        sta_duration_ms: STA window duration in milliseconds.
        filter_size: Hex grid diameter for spatial filter extraction.

    Returns:
        dict mapping cell-type name to its STA array (K, num_lags).
    """
    if cell_types is None:
        cell_types = list(DEFAULT_CELL_TYPES)

    dt = model.dt
    num_lags = int(round(sta_duration_ms / (dt * 1000)))
    num_columns = model.regions['optic_lobe'].num_columns

    if model.regions['optic_lobe'].column_p_q is None:
        raise RuntimeError('Model has no column_p_q; '
                           'cannot compute hex neighbor structure.')

    # ── Apply silencing ───────────────────────────────────────────────
    original_output_strength = None
    original_input_strength = None
    if silenced_neurons:
        original_output_strength = model.output_strength_log.data.clone()
        original_input_strength = model.input_strength_log.data.clone()
        model.silence_neurons(silenced_neurons)

    try:
        return _run_white_noise_inner(
            model, cell_types, output_dir, duration,
            sta_duration_ms, filter_size,
            dt, num_lags, num_columns,
        )
    finally:
        if original_output_strength is not None:
            model.output_strength_log.data.copy_(original_output_strength)
            model.input_strength_log.data.copy_(original_input_strength)


def _run_white_noise_inner(model, cell_types, output_dir, duration,
                           sta_duration_ms, filter_size,
                           dt, num_lags, num_columns):
    """Core analysis loop (called by :func:`run_white_noise`)."""

    # ── Generate stimulus ────────────────────────────────────────────
    num_frames = int(round(duration / dt))
    print(f'Generating {num_frames} frames of white noise '
          f'({duration:.0f}s at {1/dt:.0f} Hz, {num_columns} columns)...')
    rng = np.random.default_rng(SEED)
    stimulus = (rng.standard_normal((num_frames, num_columns)).astype(np.float32)
                * CONTRAST_STD + 0.5)

    # ── Hex neighbor structure ───────────────────────────────────────
    nearest = ml.HexGridConv.compute_nearest_columns(
        filter_size, model.regions['optic_lobe'].column_p_q,
    )
    pq_offsets = ml.HexGridConv.hex_grid_pq_offsets(filter_size)
    print(f'Filter size {filter_size}: {pq_offsets.shape[0]} spatial offsets, '
          f'{num_lags} temporal lags')

    # ── Identify valid cell types ────────────────────────────────────
    valid_cell_types = [ct for ct in cell_types
                        if ct in model.unique_cell_types]
    if not valid_cell_types:
        available = sorted(model.unique_cell_types)
        print(f'None of {cell_types} found in model. Available: {available}')
        return {}
    print(f'Cell types: {valid_cell_types}')

    # ── Run model in chunks ──────────────────────────────────────────
    stimulus_tensor = torch.tensor(
        stimulus, device=model.device, dtype=model.dtype,
    )
    all_firing_rates = {ct: [] for ct in valid_cell_types}
    init_state = None
    init_filter_state = None
    cached_mappings = None

    for chunk_start in range(0, num_frames, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, num_frames)
        chunk = stimulus_tensor[None, chunk_start:chunk_end]

        predict_kwargs = {}
        if cached_mappings is not None:
            predict_kwargs['input_neurons'] = cached_mappings['input_mapping']
            predict_kwargs['output_neurons'] = cached_mappings['output_mapping']
        else:
            predict_kwargs['output_neurons'] = valid_cell_types

        print(f'  Running model: frames {chunk_start}–{chunk_end}...')
        with torch.no_grad():
            result = model.predict(
                chunk,
                init_state=init_state,
                init_filter_state=init_filter_state,
                **predict_kwargs,
            )

        init_state = result['model_state']['v']
        init_filter_state = result['model_state']['filter_state']
        if cached_mappings is None:
            cached_mappings = result['mappings']

        for ct in valid_cell_types:
            fr = result['outputs']['firing_rate'][ct][0].cpu().numpy()
            all_firing_rates[ct].append(fr)

    # ── Compute STAs ─────────────────────────────────────────────────
    sta_dict = {}

    for ct in valid_cell_types:
        fr_full = np.concatenate(all_firing_rates[ct], axis=0)
        valid_cols = ~np.isnan(fr_full[0])
        n_valid = valid_cols.sum()
        if n_valid == 0:
            print(f'  {ct}: no valid columns, skipping.')
            continue

        fr_valid = fr_full[:, valid_cols]
        nearest_valid = nearest[valid_cols]

        print(f'  Computing STA for {ct} ({n_valid} columns)...')
        sta = compute_pooled_sta(stimulus, fr_valid, nearest_valid, num_lags)
        sta_dict[ct] = sta

    # ── Plot and save ────────────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        for ct, sta in sta_dict.items():
            out_path = output_dir / f'sta_{ct}.png'
            plot_sta(sta, pq_offsets, dt, ct, out_path)
            print(f'    Saved {out_path}')

        if sta_dict:
            center_idx = int(np.argmin(np.abs(pq_offsets).sum(axis=1)))
            time_lags_ms = np.arange(num_lags) * dt * 1000
            cell_to_mu_ms = {}
            cell_to_sigma_ms = {}
            cell_to_peak_abs_ms = {}
            for ct, sta in sta_dict.items():
                prof = sta[center_idx]
                mu_ms, sigma_ms = squared_kernel_temporal_moments(
                    time_lags_ms, prof,
                )
                cell_to_mu_ms[ct] = mu_ms
                cell_to_sigma_ms[ct] = sigma_ms
                cell_to_peak_abs_ms[ct] = peak_abs_lag_ms(time_lags_ms, prof)

            path_mean = output_dir / 'temporal_tau_mean_f2.png'
            path_rms = output_dir / 'temporal_tau_rms_f2.png'
            path_peak = output_dir / 'temporal_tau_peak_abs.png'
            plot_temporal_moment_number_line(
                cell_to_mu_ms,
                path_mean,
                xlabel=r'Mean lag $\mu_\tau$ (ms), center of mass under $f^2$',
                title='STA temporal moments (center column): mean lag',
                color='C0',
                clip_xmin_zero=False,
            )
            plot_temporal_moment_number_line(
                cell_to_sigma_ms,
                path_rms,
                xlabel=(
                    r'$\sigma_\tau$ (ms) = std.\ dev.\ of lag, '
                    r'$\sqrt{\mathrm{Var}[\tau]}$ with $p(\tau)\propto f(\tau)^2$ '
                    r'over lags (not intensity)'
                ),
                title=(
                    'STA (center column): $\sigma_\tau$ of lag under $f^2$ (ms)'
                ),
                color='C1',
                clip_xmin_zero=True,
            )
            plot_temporal_moment_number_line(
                cell_to_peak_abs_ms,
                path_peak,
                xlabel=r'Lag (ms) at $\mathrm{argmax}\,|f(\tau)|$',
                title='STA (center column): peak $|f|$',
                color='C2',
                clip_xmin_zero=True,
            )
            print(f'    Saved {path_mean}')
            print(f'    Saved {path_rms}')
            print(f'    Saved {path_peak}')
            mu_line = ', '.join(
                f'{ct}={cell_to_mu_ms[ct]:.1f}'
                for ct in sorted(
                    sta_dict,
                    key=lambda c: cell_to_mu_ms.get(c, float('nan')),
                )
                if np.isfinite(cell_to_mu_ms[ct])
            )
            sig_line = ', '.join(
                f'{ct}={cell_to_sigma_ms[ct]:.1f}'
                for ct in sorted(
                    sta_dict,
                    key=lambda c: cell_to_sigma_ms.get(c, float('nan')),
                )
                if np.isfinite(cell_to_sigma_ms[ct])
            )
            peak_line = ', '.join(
                f'{ct}={cell_to_peak_abs_ms[ct]:.1f}'
                for ct in sorted(
                    sta_dict,
                    key=lambda c: cell_to_peak_abs_ms.get(c, float('nan')),
                )
                if np.isfinite(cell_to_peak_abs_ms[ct])
            )
            print(f'  Temporal μ_τ (ms, center STA, f² kernel): {mu_line}')
            print(f'  Temporal σ_τ (ms, center STA, f² kernel): {sig_line}')
            print(f'  Peak |f| lag (ms, center STA): {peak_line}')

            cts_ordered = list(sta_dict.keys())
            data_path = output_dir / 'sta_data.npz'
            np.savez(
                data_path,
                pq_offsets=pq_offsets,
                time_lags_ms=time_lags_ms,
                cell_types=np.array(cts_ordered),
                tau_mean_ms=np.array([cell_to_mu_ms[c] for c in cts_ordered]),
                tau_rms_ms=np.array([cell_to_sigma_ms[c] for c in cts_ordered]),
                tau_peak_abs_ms=np.array(
                    [cell_to_peak_abs_ms[c] for c in cts_ordered]),
                **{f'sta_{ct}': s for ct, s in sta_dict.items()},
            )
            print(f'Saved STA data to {data_path}')

    # ── Report cell polarity ────────────────────────────────────────
    if sta_dict:
        on_cells = []
        off_cells = []
        for ct, sta in sta_dict.items():
            peak_idx = np.unravel_index(np.argmax(np.abs(sta)), sta.shape)
            if sta[peak_idx] >= 0:
                on_cells.append(ct)
            else:
                off_cells.append(ct)
        print(f'  ON  cells (positive polarity): {", ".join(on_cells)}')
        print(f'  OFF cells (negative polarity): {", ".join(off_cells)}')

    return sta_dict


def main():
    parser = argparse.ArgumentParser(
        description='White noise analysis: recover linear filters via STA.',
    )
    parser.add_argument('model_folder', type=str,
                        help='Path to model folder or model.pt')
    parser.add_argument('--cell-types', type=str, nargs='+',
                        default=None,
                        help='Cell types to compute STA for')
    parser.add_argument('--silenced-neurons', type=str, nargs='*',
                        default=None,
                        help='Neuron types to silence before analysis')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory '
                             '(default: <model_folder>/figures/white_noise)')
    parser.add_argument('--duration', type=float, default=None,
                        help=f'Stimulus duration in seconds '
                             f'(default: {STIM_DURATION_S})')
    parser.add_argument('--sta-duration-ms', type=float, default=None,
                        help=f'STA window duration in ms (default: {STA_DURATION_MS})')
    parser.add_argument('--filter-size', type=int, default=None,
                        help=f'Hex grid diameter for spatial filter '
                             f'(default: {FILTER_SIZE})')
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f'Loading model from {model_path}...')
    model = nm.BaseRnn.load(model_path)
    model.eval()

    output_dir = (Path(args.output_dir) if args.output_dir
                  else model_folder / 'figures' / 'white_noise')

    kwargs = {}
    if args.duration is not None:
        kwargs['duration'] = args.duration
    if args.sta_duration_ms is not None:
        kwargs['sta_duration_ms'] = args.sta_duration_ms
    if args.filter_size is not None:
        kwargs['filter_size'] = args.filter_size

    sta_dict = run_white_noise(
        model,
        cell_types=args.cell_types,
        silenced_neurons=args.silenced_neurons,
        output_dir=output_dir,
        **kwargs,
    )

    print(f'\nDone! Recovered filters for {len(sta_dict)} cell types.')


if __name__ == '__main__':
    main()
