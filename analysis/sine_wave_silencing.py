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

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import neural_models as nm
from sine_wave_sweep import run_sine_wave_sweep

# ── Silencing configurations ─────────────────────────────────────────
SILENCING_CONFIGS = [
    [],
    ['L1'],
    ['L2'],
    ['L3'],
    ['L1', 'L2'],
    ['L1', 'L2', 'L3'],
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


def plot_relative_amplitudes(baseline_results, silenced_results,
                             silenced_neurons, output_path):
    """Bar chart of per-cell-type response range relative to baseline."""
    common_types = sorted(set(baseline_results) & set(silenced_results))
    if not common_types:
        print('  No overlapping cell types to compare — skipping plot.')
        return

    ratios = []
    for ct in common_types:
        base_ranges = _response_amplitude_per_wavelength(baseline_results[ct])
        sil_ranges = _response_amplitude_per_wavelength(silenced_results[ct])
        per_wl_ratios = []
        for wl in base_ranges:
            if base_ranges[wl] > 0:
                per_wl_ratios.append(sil_ranges.get(wl, 0.0) / base_ranges[wl])
            else:
                per_wl_ratios.append(0.0)
        ratios.append(float(np.mean(per_wl_ratios)))

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(common_types)), 4))
    x = np.arange(len(common_types))
    ax.bar(x, ratios, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(common_types, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Relative response ratio')
    ax.set_title(f'Silenced: {"+".join(silenced_neurons)}')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 10)

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
    model = nm.NeuralModel.load(model_path)
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

    print('\nAll silencing experiments complete.')


if __name__ == '__main__':
    main()
