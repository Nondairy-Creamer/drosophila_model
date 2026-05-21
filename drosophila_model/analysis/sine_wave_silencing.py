"""Sine wave silencing: compare tuning curves with systematic neuron silencing.

Runs :func:`sine_wave_sweep.run_sine_wave_sweep` once without silencing
(baseline) and then for each silencing configuration.  For every silenced
condition a bar chart shows the relative mean response amplitude of each
cell type compared to the unsilenced baseline.

Usage:
    python analysis/sine_wave_silencing.py models/best_cells_30hz/
"""
import argparse
import sys
from pathlib import Path


import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from drosophila_model.model_definition.model_types import neural_models as nm
from sine_wave_sweep import run_sine_wave_sweep

# ── Silencing configurations ─────────────────────────────────────────
SILENCING_CONFIGS = [
    [],
    ['L1'],
    ['L2'],
    ['L3'],
    ['Y1'],
    ['L1', 'L2'],
    # ['L1', 'L2', 'L3'],
    ['Mi9', 'Mi4'],
    ['Mi9', 'Mi4', 'CT1_M'],
    ['Mi9', 'Mi4', 'CT1_M', 'C3'],
    ['Mi1', 'Tm3'],
    ['Mi1'],
    ['Tm3'],
    ['Tm1'],
    ['Tm2'],
    ['Tm1', 'Tm2'],
    ['Tm9'],
    ['Tm4'],
    ['CT1_L'],
    ['Tm9', 'Tm4', 'CT1_L'],
    ['C2'],
    ['C3'],
    ['C2', 'C3'],
    ['CT1_M'],
    # ['T4a'],
    # ['T4b'],
    # ['T4c'],
    # ['T4d'],
    # ['T5a'],
    # ['T5b'],
    # ['T5c'],
    # ['T5d'],
]


def _config_label(silenced):
    """Human-readable label for a silencing configuration."""
    if not silenced:
        return 'baseline'
    return 'silence_' + '+'.join(silenced)


def _response_amplitude_per_wavelength(cell_results_ct):
    """(max + 0.05) / (min + 0.05) across directions/frequencies, per wavelength."""
    wavelength_vals = {}
    for dir_dict in cell_results_ct.values():
        for wl, arr in dir_dict.items():
            wavelength_vals.setdefault(wl, []).append(arr)
    return {wl: float((np.max(np.concatenate(arrs)) + 0.05)
                       / (np.min(np.concatenate(arrs)) + 0.05))
            for wl, arrs in wavelength_vals.items()}


def relative_response_ratio(baseline_results_ct, silenced_results_ct):
    """Single cell type: mean over wavelengths of (silenced range / baseline range)."""
    base_ranges = _response_amplitude_per_wavelength(baseline_results_ct)
    sil_ranges = _response_amplitude_per_wavelength(silenced_results_ct)
    per_wl_ratios = []
    for wl in base_ranges:
        if base_ranges[wl] > 0:
            per_wl_ratios.append(sil_ranges.get(wl, 0.0) / base_ranges[wl])
        else:
            per_wl_ratios.append(0.0)
    return float(np.mean(per_wl_ratios))


def _mean_response_over_direction(cell_results_ct, direction):
    """Mean firing rate for one direction, over wavelengths and temporal frequencies."""
    arrs = [cell_results_ct[direction][wl]
            for wl in sorted(cell_results_ct[direction].keys())]
    return float(np.mean(np.concatenate(arrs)))


def direction_selectivity_index(cell_results_ct, subtype_letter):
    """T4/T5 DS index from mean responses; subtype_letter is 'a'–'d'.

    A: (R-L)/(R+L); B: (L-R)/(R+L); C: (U-D)/(U+D); D: (D-U)/(U+D) — using
    mean responses for each direction label.
    """
    mr = _mean_response_over_direction(cell_results_ct, 'right')
    ml = _mean_response_over_direction(cell_results_ct, 'left')
    mu = _mean_response_over_direction(cell_results_ct, 'up')
    md = _mean_response_over_direction(cell_results_ct, 'down')
    den_lr = mr + ml
    den_ud = mu + md
    s = subtype_letter.lower()
    if s == 'a':
        return (mr - ml) / den_lr if den_lr > 0 else np.nan
    if s == 'b':
        return (ml - mr) / den_lr if den_lr > 0 else np.nan
    if s == 'c':
        return (mu - md) / den_ud if den_ud > 0 else np.nan
    if s == 'd':
        return (md - mu) / den_ud if den_ud > 0 else np.nan
    raise ValueError(f'Unknown subtype letter: {subtype_letter!r}')


def plot_relative_amplitudes(baseline_results, silenced_results,
                             silenced_neurons, output_path):
    """Bar chart of per-cell-type response range relative to baseline."""
    common_types = sorted(set(baseline_results) & set(silenced_results))
    if not common_types:
        print('  No overlapping cell types to compare — skipping plot.')
        return

    ratios = []
    for ct in common_types:
        ratios.append(relative_response_ratio(baseline_results[ct],
                                              silenced_results[ct]))

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(common_types)), 4))
    x = np.arange(len(common_types))
    ax.bar(x, ratios, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(common_types, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Relative response ratio')
    ax.set_title(f'Silenced: {"+".join(silenced_neurons)}')
    ax.set_ylim(0, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {output_path}')


def plot_t4_t5_ds_vs_ratio(
        baseline_results,
        all_results,
        silencing_configs,
        output_root,
        family,
):
    """Scatter: relative response ratio (x) vs direction selectivity (y) per subtype.

    ``family`` is ``'T4'`` or ``'T5'``. Includes baseline at ratio 1.0 (large markers)
    plus one point per silencing experiment (smaller markers).
    """
    subtypes = [f'{family}{L}' for L in 'abcd']
    xs = {st: [] for st in subtypes}
    ys = {st: [] for st in subtypes}
    exp_tags = {st: [] for st in subtypes}
    x_base = {}
    y_base = {}

    for config in silencing_configs:
        label = _config_label(config)
        silenced_results = all_results.get(label, {})
        is_baseline = not config
        exp_name = '+'.join(config) if config else ''
        for st in subtypes:
            if st not in baseline_results or st not in silenced_results:
                continue
            letter = st[-1]
            if is_baseline:
                ratio = 1.0
            else:
                ratio = relative_response_ratio(baseline_results[st],
                                                silenced_results[st])
            ds = direction_selectivity_index(silenced_results[st], letter)
            if not np.isfinite(ds):
                continue
            if is_baseline:
                x_base[st] = ratio
                y_base[st] = ds
            else:
                xs[st].append(ratio)
                ys[st].append(ds)
                exp_tags[st].append(exp_name)

    if not x_base and not any(xs[st] for st in subtypes):
        print(f'  No T4/T5 data for {family} DS scatter — skipping.')
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.45, 4))
    for i, st in enumerate(subtypes):
        if xs[st]:
            ax.scatter(xs[st], ys[st], color=colors[i], label=st, s=36,
                       edgecolors='white', linewidths=0.4, alpha=0.9, zorder=2)
        if st in x_base:
            bl_label = st if not xs[st] else '_nolegend_'
            ax.scatter([x_base[st]], [y_base[st]], color=colors[i], s=220,
                       edgecolors='black', linewidths=1.6, alpha=0.95,
                       zorder=5, label=bl_label)

    ax.scatter([], [], s=220, facecolors='none', edgecolors='black',
               linewidths=1.6, label='baseline (no silencing)')

    ratio_label_threshold = 0.4
    ds_label_threshold = 0.25
    for i, st in enumerate(subtypes):
        for x, y, tag in zip(xs[st], ys[st], exp_tags[st]):
            low_ratio = x < ratio_label_threshold
            low_ds = y < ds_label_threshold
            if not low_ratio and not low_ds:
                continue
            if low_ds and not low_ratio:
                xytext, va = (5, -11), 'top'
            else:
                xytext, va = (5, 5), 'bottom'
            ax.annotate(
                tag,
                (x, y),
                textcoords='offset points',
                xytext=xytext,
                fontsize=6,
                color=colors[i],
                alpha=0.95,
                ha='left',
                va=va,
                clip_on=True,
            )

    ax.axhline(0.0, color='gray', linewidth=0.6, linestyle=':')
    ax.axvline(1.0, color='k', linewidth=0.7, linestyle='--', alpha=0.7)
    ax.set_xlabel('Relative response ratio (silenced / baseline)')
    ax.set_ylabel('Direction selectivity index')
    ax.set_title(f'{family}: DS vs relative amplitude')
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(-1.05, 1.05)

    output_path = output_root / f'direction_selectivity_vs_relative_ratio_{family}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Sine wave sweep analysis with systematic silencing.',
    )
    parser.add_argument('model_folder', type=str,
                        help='Path to model folder or model.pt')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Root output directory '
                             '(default: <model_folder>/figures/sine_sweep_silencing)')
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f'Loading model from {model_path}...')
    model = nm.BaseRnn.load(model_path)
    model.eval()

    output_root = (Path(args.output_dir) if args.output_dir
                   else model_folder / 'figures' / 'sine_sweep_silencing')

    # ── Run each silencing configuration ─────────────────────────────
    all_results = {}
    for config in SILENCING_CONFIGS:
        label = _config_label(config)
        sub_dir = output_root / label
        print(f'\n{"=" * 60}')
        print(f'Running: {label}')
        print(f'{"=" * 60}')

        cell_results = run_sine_wave_sweep(
            model,
            silenced_neurons=config if config else None,
            output_dir=sub_dir,
            expressive=True,
        )
        all_results[label] = cell_results

    # ── Relative-amplitude comparison plots ──────────────────────────
    baseline_results = all_results.get('baseline', {})
    if not baseline_results:
        print('Baseline run produced no results — cannot compute ratios.')
        return

    for config in SILENCING_CONFIGS:
        if not config:
            continue
        label = _config_label(config)
        silenced_results = all_results.get(label, {})
        out_path = output_root / f'relative_amplitude_{label}.png'
        plot_relative_amplitudes(baseline_results, silenced_results, config,
                                out_path)

    plot_t4_t5_ds_vs_ratio(baseline_results, all_results, SILENCING_CONFIGS,
                           output_root, 'T4')
    plot_t4_t5_ds_vs_ratio(baseline_results, all_results, SILENCING_CONFIGS,
                           output_root, 'T5')

    print('\nAll silencing experiments complete.')


if __name__ == '__main__':
    main()
