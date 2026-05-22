"""Bake the canonical connectome bundle, once.

Reads the raw connectome CSVs and writes the standard layout that
:mod:`connectome_loading` and the :class:`regions.RegionView` subclasses
expect. After this runs the bundle is static; per-model slicing happens
at load time via the ``areas`` / ``side`` filters.

Reads raw CSVs from
``config['data']['data_folder'] / 'connectome_data' / 'raw'`` and
writes the baked bundle to
``config['data']['data_folder'] / 'connectome_data' / 'baked'``:

- ``cell_info.parquet`` — one row per neuron in the connectome.
    columns: ``root_id, type, neurotransmitter, area, side, column_index``
    ``column_index`` is populated for optic-lobe neurons that map to a
    hexagonal visual column; NaN otherwise.
- ``synapses.parquet`` — one row per directed synaptic edge.
    columns: ``pre_root_id, post_root_id, synapse_counts, nt_type``

Per-column optic-lobe geometry lives in ``optic_lobe_columns.parquet``
alongside the baked bundle. That file is sourced from a separate
empirical dataset and is NOT written by this script — it must already
exist in the baked directory before models can load.

To add support for another area (e.g. antenna lobe), append a section
that emits more rows to ``cell_info.parquet`` / ``synapses.parquet`` with
``area = '<area_name>'`` and, if the region needs per-element geometry,
drop ``<area_name>_*.parquet`` alongside the core bundle.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from drosophila_model.config import load_config


def _process_optic_lobe(
    *,
    visual_neurons_df: pd.DataFrame,
    synaptic_connections_df: pd.DataFrame,
    columns_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cell_info, synapses) frames for the optic lobe.

    Bakes BOTH hemispheres of intrinsic optic-lobe neurons. ``boundary`` cells
    (those that straddle the optic lobe and a neighbouring area) are dropped:
    they're conceptually shared with another region and weren't used by any
    existing model.

    Per-cell-type modelling exclusions (e.g. R1-6/R7/R8 photoreceptors for the
    visual model) are NOT applied here — they're a region-level decision and
    are applied at load time via ``OpticLobeRegion.default_neurons_to_exclude``
    or ``region_kwargs['optic_lobe']['neurons_to_exclude']``.
    """
    # Boundary cells aren't part of the canonical optic-lobe modelling target.
    visual_neurons_df = visual_neurons_df[
        visual_neurons_df['category'] == 'intrinsic'
    ]

    # Limit to synapses where both endpoints are visual neurons.
    visual_ids = set(visual_neurons_df['root_id'].to_numpy().tolist())
    keep_visual = (
        synaptic_connections_df['pre_root_id'].isin(visual_ids)
        & synaptic_connections_df['post_root_id'].isin(visual_ids)
    )
    synaptic_connections_df = synaptic_connections_df[keep_visual]

    # Build a per-cell-type neurotransmitter table from the (weighted) mode of
    # presynaptic NT calls. Photoreceptors are forced to histamine even after
    # exclusion (they may be added back as an input source later).
    pre_root_to_type = dict(zip(
        visual_neurons_df['root_id'].to_numpy(),
        visual_neurons_df['type'].to_numpy(),
    ))
    pre_types = np.array([
        pre_root_to_type.get(rid, None)
        for rid in synaptic_connections_df['pre_root_id'].to_numpy()
    ])
    nt_per_type = (
        pd.DataFrame({
            'cell_type': pre_types,
            'nt_type': synaptic_connections_df['nt_type'].to_numpy(),
            'syn_count': synaptic_connections_df['syn_count'].to_numpy(),
        })
        .dropna(subset=['cell_type'])
        .groupby(['cell_type', 'nt_type'])['syn_count'].sum()
        .reset_index()
    )
    dominant = nt_per_type.loc[nt_per_type.groupby('cell_type')['syn_count'].idxmax()]
    type_to_nt = dict(zip(dominant['cell_type'], dominant['nt_type']))
    for photoreceptor_type in ('R1-6', 'R7', 'R8'):
        type_to_nt[photoreceptor_type] = 'HIS'

    # Restrict cell info to neurons that actually participate in retained synapses.
    model_root_ids = np.unique(np.concatenate([
        synaptic_connections_df['pre_root_id'].to_numpy(),
        synaptic_connections_df['post_root_id'].to_numpy(),
    ]))
    cell_info = visual_neurons_df[
        visual_neurons_df['root_id'].isin(model_root_ids)
    ].copy()

    # Assign column_index from the column_assignments CSV; NaN elsewhere.
    cols = columns_df[columns_df['root_id'].isin(model_root_ids)]
    col_lookup = dict(zip(cols['root_id'].to_numpy(),
                          (cols['column_id'].to_numpy() - 1).astype(float)))
    cell_info['column_index'] = cell_info['root_id'].map(col_lookup)

    cell_info['neurotransmitter'] = cell_info['type'].map(type_to_nt)
    cell_info['area'] = 'optic_lobe'

    cell_info = cell_info[
        ['root_id', 'type', 'neurotransmitter', 'area', 'side', 'column_index']
    ].reset_index(drop=True)

    synapses = synaptic_connections_df[
        ['pre_root_id', 'post_root_id', 'syn_count', 'nt_type']
    ].rename(columns={'syn_count': 'synapse_counts'}).reset_index(drop=True)

    return cell_info, synapses


_ROOT_ID_DTYPE = np.int64


def main() -> None:
    config = load_config()
    folder = Path(config['data']['data_folder'])
    raw_dir = folder / 'connectome_data' / 'raw'
    baked_dir = folder / 'connectome_data' / 'baked'

    visual_neurons_df = pd.read_csv(
        raw_dir / 'visual_neuron_types_no_threshold.csv',
        dtype={'root_id': _ROOT_ID_DTYPE},
    )
    synaptic_connections_df = pd.read_csv(
        raw_dir / 'connections_no_threshold.csv',
        dtype={
            'pre_root_id': _ROOT_ID_DTYPE,
            'post_root_id': _ROOT_ID_DTYPE,
            'syn_count': _ROOT_ID_DTYPE,
        },
    )
    columns_df = pd.read_csv(
        raw_dir / 'column_assignments_no_threshold.csv',
        dtype={'root_id': _ROOT_ID_DTYPE},
    )

    cell_info_pieces: list[pd.DataFrame] = []
    synapses_pieces: list[pd.DataFrame] = []

    ol_cells, ol_syn = _process_optic_lobe(
        visual_neurons_df=visual_neurons_df,
        synaptic_connections_df=synaptic_connections_df,
        columns_df=columns_df,
    )
    cell_info_pieces.append(ol_cells)
    synapses_pieces.append(ol_syn)

    # TODO: add additional areas (antenna lobe, mushroom body, ...) here
    # by appending more frames to cell_info_pieces / synapses_pieces and
    # writing their extras alongside the core parquets.

    cell_info = pd.concat(cell_info_pieces, ignore_index=True)
    synapses = pd.concat(synapses_pieces, ignore_index=True)

    baked_dir.mkdir(parents=True, exist_ok=True)
    cell_info.to_parquet(baked_dir / 'cell_info.parquet', index=False)
    synapses.to_parquet(baked_dir / 'synapses.parquet', index=False)

    print(f"Wrote {len(cell_info):,} neurons to {baked_dir / 'cell_info.parquet'}")
    print(f"Wrote {len(synapses):,} synapses to {baked_dir / 'synapses.parquet'}")


if __name__ == '__main__':
    main()
