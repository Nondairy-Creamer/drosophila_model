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
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import neural_models as nm

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
STA_DURATION_MS = 400.0    # STA window duration in milliseconds
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


def pq_to_xy(pq):
    """Project hex lattice (p, q) coordinates to flat (x, y) for plotting."""
    angle_1 = np.pi / 6
    angle_2 = 2 * np.pi / 3 + angle_1
    basis = np.array([[np.cos(angle_1), np.cos(angle_2)],
                      [np.sin(angle_1), np.sin(angle_2)]])
    return (basis @ pq.T).T


def _build_xt_profile(sta, pq_offsets):
    """Collapse 2D hex STA to a 1D horizontal profile for an x-t plot.

    Selects the middle row (p+q == 0) and one row above/below (p+q == ±1),
    giving 7 hex positions. Positions sharing the same horizontal coordinate
    are averaged, yielding 5 spatial bins sorted left-to-right.

    Returns (xt, x_positions):
        xt : (n_x, n_lags) array
        x_positions : (n_x,) sorted unique x coordinates
    """
    pq = pq_offsets.astype(np.float64)
    row_sum = (pq[:, 0] + pq[:, 1]).astype(int)
    mask = np.abs(row_sum) <= 1
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
    the middle three rows of the hex kernel).
    Bottom panel: temporal profile at the center position.
    """
    num_lags = sta.shape[1]
    time_lags_ms = np.arange(num_lags) * dt * 1000
    center_idx = np.argmin(np.abs(pq_offsets).sum(axis=1))

    xt, x_pos = _build_xt_profile(sta, pq_offsets)
    vmax = np.abs(xt).max() or 1

    xy = pq_to_xy(pq_offsets.astype(np.float64))
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


def main():
    parser = argparse.ArgumentParser(
        description='White noise analysis: recover linear filters via STA.',
    )
    parser.add_argument('model_folder', type=str,
                        help='Path to model folder or model.pt')
    parser.add_argument('--cell-types', type=str, nargs='+',
                        default=[
                            'L1', 'L2', 'L3', 'L4', 'L5',
                            'C2', 'C3', 'CT1_L', 'CT1_M',
                            'Mi1', 'Tm3', 'Mi9', 'Mi4',
                            'T4a', 'T4b', 'T4c', 'T4d',
                            'T5a', 'T5b', 'T5c', 'T5d',
                            'Tm1', 'Tm2', 'Tm4', 'Tm9',
                            'T1', 'T2', 'T3',
                        ],
                        help='Cell types to compute STA for')
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

    duration = args.duration if args.duration is not None else STIM_DURATION_S
    sta_duration_ms = args.sta_duration_ms if args.sta_duration_ms is not None else STA_DURATION_MS
    filter_size = args.filter_size if args.filter_size is not None else FILTER_SIZE

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f'Loading model from {model_path}...')
    model = nm.NeuralModel.load(model_path)
    model.eval()

    dt = model.dt
    num_lags = int(round(sta_duration_ms / (dt * 1000)))
    num_columns = model.num_columns

    if model.column_p_q is None:
        print('Error: model has no column_p_q; '
              'cannot compute hex neighbor structure.')
        sys.exit(1)

    # ── Generate stimulus ────────────────────────────────────────────
    num_frames = int(round(duration / dt))
    print(f'Generating {num_frames} frames of white noise '
          f'({duration:.0f}s at {1/dt:.0f} Hz, {num_columns} columns)...')
    rng = np.random.default_rng(SEED)
    stimulus = (rng.standard_normal((num_frames, num_columns)).astype(np.float32)
                * CONTRAST_STD + 0.5)

    # ── Hex neighbor structure ───────────────────────────────────────
    nearest = nm.NeuralModelConv.compute_nearest_columns(
        filter_size, model.column_p_q,
    )
    pq_offsets = nm.NeuralModelConv.hex_grid_pq_offsets(filter_size)
    print(f'Filter size {filter_size}: {pq_offsets.shape[0]} spatial offsets, '
          f'{num_lags} temporal lags')

    # ── Identify valid cell types ────────────────────────────────────
    cell_types = [ct for ct in args.cell_types
                  if ct in model.unique_cell_types]
    if not cell_types:
        available = sorted(model.unique_cell_types)
        print(f'None of {args.cell_types} found in model.')
        print(f'Available: {available}')
        sys.exit(1)
    print(f'Cell types: {cell_types}')

    # ── Run model in chunks ──────────────────────────────────────────
    stimulus_tensor = torch.tensor(
        stimulus, device=model.device, dtype=model.dtype,
    )
    all_firing_rates = {ct: [] for ct in cell_types}
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
            predict_kwargs['output_neurons'] = cell_types

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

        for ct in cell_types:
            fr = result['outputs']['firing_rate'][ct][0].cpu().numpy()
            all_firing_rates[ct].append(fr)

    # ── Compute and plot STAs ────────────────────────────────────────
    output_dir = (Path(args.output_dir) if args.output_dir
                  else model_folder / 'figures' / 'white_noise')

    sta_dict = {}

    for ct in cell_types:
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

        out_path = output_dir / f'sta_{ct}.png'
        plot_sta(sta, pq_offsets, dt, ct, out_path)
        print(f'    Saved {out_path}')

    # ── Save raw STA data ────────────────────────────────────────────
    if sta_dict:
        data_path = output_dir / 'sta_data.npz'
        np.savez(
            data_path,
            pq_offsets=pq_offsets,
            time_lags_ms=np.arange(num_lags) * dt * 1000,
            cell_types=np.array(list(sta_dict.keys())),
            **{f'sta_{ct}': sta for ct, sta in sta_dict.items()},
        )
        print(f'Saved STA data to {data_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
