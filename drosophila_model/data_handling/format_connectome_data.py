"""Bake the canonical neuropil-keyed connectome bundle, once.

Reads the raw whole-connectome CSVs and writes the standard layout that
:mod:`connectome_loading` and the :class:`regions.RegionView` subclasses
expect. After this runs the bundle is static; per-model region slicing
happens at load time via :func:`connectome_loading.load_connectome`,
which filters by ``{region: {neuropils, side}}``.

Reads the FlyWire Codex downloads (FAFB v783) in
``<data_folder>/connectome_data/raw`` (``data_folder`` is
``[data].data_folder`` in ``config.toml``), as downloaded:

- ``consolidated_cell_types.csv.gz`` — ``root_id, primary_type,
  additional_type(s)`` for every classified neuron.
- ``connections_princeton_no_threshold.csv.gz`` — ``pre_root_id,
  post_root_id, neuropil, syn_count, nt_type`` for every connected pair and
  neuropil in the whole connectome.
- ``column_assignment.csv.gz`` — optic-lobe per-neuron hex-column
  assignments. Consumed only for the optic-lobe-private bundle files.
- Optional extras merged into ``cell_info`` when present:
  ``cell_stats.csv.gz`` (``length_nm, area_nm, size_nm``) and
  ``classification.csv.gz`` (``flow, super_class, class, sub_class, side``).

CT1 is too large to model as one neuron per optic lobe, so it is replaced by
per-column ``CT1_M`` / ``CT1_L`` pieces from
:mod:`drosophila_model.data_handling.split_ct1` (``ct1_pieces.parquet`` and
``ct1_connections.parquet`` in the baked directory). When those are missing
the split runs first, which needs the raw synapse table.

Writes the baked bundle to ``<data_folder>/connectome_data/baked``:

- ``cell_info.parquet`` — universal per-neuron table.
    columns:
      ``root_id`` (int64),
      ``primary_type`` (str),
      ``neurotransmitter`` (str; per-cell-type dominant pre-synaptic NT,
      with photoreceptors forced to ``'HIS'``; may be NaN for cell types
      that never appear pre-synaptically),
      ``neuropils`` (``list[str]``; union of every neuropil this neuron
      has at least one synapse in. Part of the bundle schema that
      :mod:`connectome_loading` checks for, but never read: the loader
      recomputes region membership from the thresholded synapse table),
      plus the optional extras above (NaN for the CT1 pieces).
- ``synapses.parquet`` — universal directed edges (whole connectome).
    columns:
      ``pre_root_id, post_root_id, synapse_counts, nt_type, neuropil``.
- ``optic_lobe_column_index.parquet`` — per-neuron optic-lobe hex-column
    assignment (loaded by :class:`regions.OpticLobeRegion`).
    columns: ``root_id, column_index`` (0-indexed).

The empirical per-column geometry file ``optic_lobe_columns.parquet`` (the
right-eye viewing direction of each column, from the Reiser lab eye map of
Zhao et al. 2025, keyed by the Codex column ``p, q``) is NOT written here —
it ships with the data bundle and must already exist in the baked directory
before models can load.

To add support for another area, just declare a new ``RegionView``
subclass with ``neuropils = [...]`` in :mod:`regions` and register it in
``regions.REGION_REGISTRY``. No change needed here.

Usage:
    python -m drosophila_model.data_handling.format_connectome_data
"""
import argparse

import numpy as np
import pandas as pd
from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling import split_ct1


_ROOT_ID_DTYPE = np.int64
_PHOTORECEPTOR_TYPES = ('R1-6', 'R7', 'R8')
# Optional per-neuron extras: raw file -> columns merged into cell_info.
_EXTRA_CELL_INFO = {
    'cell_stats.csv.gz': ['length_nm', 'area_nm', 'size_nm'],
    'classification.csv.gz': ['flow', 'super_class', 'class', 'sub_class', 'side'],
}


def _per_neuron_neuropils(connections_df: pd.DataFrame) -> pd.DataFrame:
    """Return ``DataFrame[root_id, neuropils]`` with ``neuropils`` a sorted list.

    Each neuron's neuropil set is the union of neuropils across all its
    synapses (pre or post). Stored as a sorted ``list[str]`` so the parquet
    round-trips as a native ``list<string>`` column.
    """
    long = pd.concat([
        connections_df[['pre_root_id', 'neuropil']].rename(
            columns={'pre_root_id': 'root_id'}),
        connections_df[['post_root_id', 'neuropil']].rename(
            columns={'post_root_id': 'root_id'}),
    ], ignore_index=True)
    long = long.drop_duplicates()
    grouped = (
        long.groupby('root_id')['neuropil']
        .apply(lambda s: sorted(s.unique().tolist()))
        .reset_index()
        .rename(columns={'neuropil': 'neuropils'})
    )
    return grouped


def _per_cell_type_dominant_nt(
    connections_df: pd.DataFrame,
    cell_types_df: pd.DataFrame,
) -> dict[str, str]:
    """Return ``{primary_type: nt_type}`` by weighted majority of pre-synaptic NT.

    Photoreceptors are forced to ``'HIS'`` afterwards — the FlyWire automated
    NT classifier doesn't call histamine, but they're biologically histaminergic.
    """
    type_lookup = cell_types_df.set_index('root_id')['primary_type']
    pre_types = connections_df['pre_root_id'].map(type_lookup)
    weighted = (
        pd.DataFrame({
            'primary_type': pre_types.to_numpy(),
            'nt_type': connections_df['nt_type'].to_numpy(),
            'syn_count': connections_df['syn_count'].to_numpy(),
        })
        .dropna(subset=['primary_type'])
        .groupby(['primary_type', 'nt_type'])['syn_count'].sum()
        .reset_index()
    )
    dominant = weighted.loc[weighted.groupby('primary_type')['syn_count'].idxmax()]
    type_to_nt = dict(zip(dominant['primary_type'], dominant['nt_type']))
    for photoreceptor_type in _PHOTORECEPTOR_TYPES:
        type_to_nt[photoreceptor_type] = 'HIS'
    return type_to_nt


def main() -> None:
    argparse.ArgumentParser(
        description='Bake <data_folder>/connectome_data/baked from the raw CSVs in '
                    '<data_folder>/connectome_data/raw ([data].data_folder of config.toml).'
    ).parse_args()
    folder = get_data_folder(load_config())
    raw_dir = folder / 'connectome_data' / 'raw'
    baked_dir = folder / 'connectome_data' / 'baked'

    if not ((baked_dir / 'ct1_pieces.parquet').exists()
            and (baked_dir / 'ct1_connections.parquet').exists()):
        print("CT1 split not found; running split_ct1 ...")
        split_ct1.split_ct1(folder)
    ct1_pieces = pd.read_parquet(baked_dir / 'ct1_pieces.parquet')
    ct1_connections = pd.read_parquet(baked_dir / 'ct1_connections.parquet')

    print(f"Reading raw CSVs from {raw_dir} ...")
    cell_types_df = pd.read_csv(
        raw_dir / 'consolidated_cell_types.csv.gz',
        dtype={'root_id': _ROOT_ID_DTYPE},
    )
    connections_df = pd.read_csv(
        raw_dir / 'connections_princeton_no_threshold.csv.gz',
        dtype={
            'pre_root_id': _ROOT_ID_DTYPE,
            'post_root_id': _ROOT_ID_DTYPE,
            'syn_count': _ROOT_ID_DTYPE,
        },
    )
    columns_df = pd.read_csv(
        raw_dir / 'column_assignment.csv.gz',
        dtype={'root_id': _ROOT_ID_DTYPE},
    )

    # Swap each CT1 for its per-column CT1_M / CT1_L pieces: their rows
    # replace CT1's connections, and they are typed and given a column here.
    ct1_ids = set(ct1_pieces['parent_root_id'])
    is_ct1 = (connections_df['pre_root_id'].isin(ct1_ids)
              | connections_df['post_root_id'].isin(ct1_ids))
    connections_df = pd.concat(
        [connections_df[~is_ct1], ct1_connections[connections_df.columns]],
        ignore_index=True,
    )
    cell_types_df = pd.concat([
        cell_types_df[~cell_types_df['root_id'].isin(ct1_ids)],
        ct1_pieces[['root_id', 'primary_type']].assign(**{'additional_type(s)': np.nan}),
    ], ignore_index=True)
    columns_df = pd.concat(
        [columns_df, ct1_pieces[['root_id', 'hemisphere', 'primary_type', 'column_id', 'p', 'q']]
         .rename(columns={'primary_type': 'type'})],
        ignore_index=True,
    )
    print(f"Replaced {len(ct1_ids)} CT1 neurons with {len(ct1_pieces):,} per-column pieces.")

    print(f"Loaded {len(cell_types_df):,} typed neurons "
          f"and {len(connections_df):,} synapses.")

    # Per-neuron neuropil union from the (whole-brain) synapse table.
    print("Computing per-neuron neuropil sets ...")
    neuron_neuropils = _per_neuron_neuropils(connections_df)

    # Per-cell-type dominant pre-synaptic NT (photoreceptor override applied).
    print("Computing per-cell-type dominant neurotransmitter ...")
    type_to_nt = _per_cell_type_dominant_nt(connections_df, cell_types_df)

    # Restrict universal cell_info to typed neurons that participate in at
    # least one synapse — anything else can't contribute to the model.
    typed_with_synapses = cell_types_df[
        cell_types_df['root_id'].isin(neuron_neuropils['root_id'])
    ].copy()
    cell_info = typed_with_synapses.merge(neuron_neuropils, on='root_id', how='left')
    cell_info['neurotransmitter'] = cell_info['primary_type'].map(type_to_nt)
    cell_info = cell_info[
        ['root_id', 'primary_type', 'neurotransmitter', 'neuropils']
    ].reset_index(drop=True)
    for file_name, extra_cols in _EXTRA_CELL_INFO.items():
        if (raw_dir / file_name).exists():
            extra = pd.read_csv(raw_dir / file_name, dtype={'root_id': _ROOT_ID_DTYPE})
            cell_info = cell_info.merge(extra[['root_id'] + extra_cols], on='root_id', how='left')

    # Synapses table: keep neuropil column so the loader can filter edges
    # by region neuropil sets without re-deriving from cell_info.
    synapses = connections_df[
        ['pre_root_id', 'post_root_id', 'syn_count', 'nt_type', 'neuropil']
    ].rename(columns={'syn_count': 'synapse_counts'}).reset_index(drop=True)

    # Optic-lobe per-neuron column index (private to OpticLobeRegion).
    # Column IDs in the raw CSV are 1-indexed; we store 0-indexed to line
    # up with the per-column optic_lobe_columns.parquet geometry. Rows
    # without a column assignment are dropped.
    cols_valid = columns_df.dropna(subset=['column_id'])
    column_index_df = pd.DataFrame({
        'root_id': cols_valid['root_id'].astype(_ROOT_ID_DTYPE),
        'column_index': (cols_valid['column_id'].to_numpy() - 1).astype(np.int64),
    })

    baked_dir.mkdir(parents=True, exist_ok=True)
    cell_info.to_parquet(baked_dir / 'cell_info.parquet', index=False)
    synapses.to_parquet(baked_dir / 'synapses.parquet', index=False)
    column_index_df.to_parquet(
        baked_dir / 'optic_lobe_column_index.parquet', index=False,
    )

    print(f"Wrote {len(cell_info):,} neurons to {baked_dir / 'cell_info.parquet'}")
    print(f"Wrote {len(synapses):,} synapses to {baked_dir / 'synapses.parquet'}")
    print(f"Wrote {len(column_index_df):,} optic-lobe column assignments to "
          f"{baked_dir / 'optic_lobe_column_index.parquet'}")


if __name__ == '__main__':
    main()
