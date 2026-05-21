"""White noise silencing: compare STA filters with systematic neuron silencing.

Runs :func:`white_noise.run_white_noise` once without silencing (baseline) and
then for each silencing configuration.  For every silenced condition a bar chart
shows the relative RMS amplitude of each recovered filter compared to the
unsilenced baseline.

Usage:
    python analysis/white_noise_silencing.py models/best_cells_30hz/
    python analysis/white_noise_silencing.py models/best_cells_30hz/ --duration 30
"""
import argparse
import sys
from pathlib import Path


import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from drosophila_model.model_definition.model_types import neural_models as nm
from white_noise import run_white_noise, DEFAULT_CELL_TYPES

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


def _sta_rms(sta):
    """Root-mean-square amplitude of a STA filter."""
    return np.sqrt(np.mean(sta ** 2))


def plot_relative_amplitudes(baseline_stas, silenced_stas, silenced_neurons,
                             output_path):
    """Bar chart of per-cell-type RMS relative to the unsilenced baseline."""
    common_types = sorted(set(baseline_stas) & set(silenced_stas))
    if not common_types:
        print('  No overlapping cell types to compare — skipping plot.')
        return

    ratios = []
    for ct in common_types:
        base_rms = _sta_rms(baseline_stas[ct])
        sil_rms = _sta_rms(silenced_stas[ct])
        ratios.append(sil_rms / base_rms if base_rms > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(common_types)), 4))
    x = np.arange(len(common_types))
    ax.bar(x, ratios, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(common_types, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Relative RMS amplitude')
    ax.set_title(f'Silenced: {"+".join(silenced_neurons)}')
    ax.set_ylim(0, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='White noise STA analysis with systematic silencing.',
    )
    parser.add_argument('model_folder', type=str,
                        help='Path to model folder or model.pt')
    parser.add_argument('--cell-types', type=str, nargs='+', default=None,
                        help='Cell types to analyse')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Root output directory '
                             '(default: <model_folder>/figures/white_noise_silencing)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Stimulus duration in seconds')
    parser.add_argument('--sta-duration-ms', type=float, default=None,
                        help='STA window duration in ms')
    parser.add_argument('--filter-size', type=int, default=None,
                        help='Hex grid diameter for spatial filter')
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f'Loading model from {model_path}...')
    model = nm.BaseRnn.load(model_path)
    model.eval()

    output_root = (Path(args.output_dir) if args.output_dir
                   else model_folder / 'figures' / 'white_noise_silencing')

    wn_kwargs = {}
    if args.duration is not None:
        wn_kwargs['duration'] = args.duration
    if args.sta_duration_ms is not None:
        wn_kwargs['sta_duration_ms'] = args.sta_duration_ms
    if args.filter_size is not None:
        wn_kwargs['filter_size'] = args.filter_size

    cell_types = args.cell_types

    # ── Run each silencing configuration ─────────────────────────────
    all_results = {}
    for config in SILENCING_CONFIGS:
        label = _config_label(config)
        sub_dir = output_root / label
        print(f'\n{"=" * 60}')
        print(f'Running: {label}')
        print(f'{"=" * 60}')

        sta_dict = run_white_noise(
            model,
            cell_types=cell_types,
            silenced_neurons=config if config else None,
            output_dir=sub_dir,
            **wn_kwargs,
        )
        all_results[label] = sta_dict

    # ── Relative-amplitude comparison plots ──────────────────────────
    baseline_stas = all_results.get('baseline', {})
    if not baseline_stas:
        print('Baseline run produced no filters — cannot compute ratios.')
        return

    for config in SILENCING_CONFIGS:
        if not config:
            continue
        label = _config_label(config)
        silenced_stas = all_results.get(label, {})
        out_path = output_root / f'relative_amplitude_{label}.png'
        plot_relative_amplitudes(baseline_stas, silenced_stas, config,
                                 out_path)

    print('\nAll silencing experiments complete.')


if __name__ == '__main__':
    main()
