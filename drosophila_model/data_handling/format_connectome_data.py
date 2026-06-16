"""Bake the canonical neuropil-keyed connectome bundle, once.

Reads the raw whole-connectome CSVs and writes the standard layout that
:mod:`connectome_loading` and the :class:`regions.RegionView` subclasses
expect. After this runs the bundle is static; per-model region slicing
happens at load time via :func:`connectome_loading.load_connectome`,
which filters by ``{region: {neuropils, side}}``.

Reads raw CSVs from
``config['data']['data_folder'] / 'connectome_data' / 'raw'``:

- ``consolidated_cell_types.csv`` — ``root_id, primary_type,
  additional_type(s)`` for every classified neuron.
- ``connections_no_threshold.csv`` — ``pre_root_id, post_root_id, neuropil,
  syn_count, nt_type`` for every synapse in the whole connectome.
- ``column_assignments_no_threshold.csv`` — optic-lobe-only per-neuron
  hex-column assignments. Consumed only for the optic-lobe-private bundle
  files.
- ``visual_neuron_types_no_threshold.csv`` — read solely to recover the
  ``CT1_L`` / ``CT1_M`` per-column synthetic neuron IDs that aren't in
  ``consolidated_cell_types.csv``. CT1 is too large to treat as two
  whole-neuron entities for optic-lobe modelling, so an upstream split
  assigns one synthetic ``CT1_L`` and one synthetic ``CT1_M`` neuron per
  column. Those synthetic root IDs ARE present in the synapse and column
  CSVs; this is the only place we need them typed.

Writes the baked bundle to
``config['data']['data_folder'] / 'connectome_data' / 'baked'``:

- ``cell_info.parquet`` — universal per-neuron table.
    columns:
      ``root_id`` (int64),
      ``primary_type`` (str),
      ``neurotransmitter`` (str; per-cell-type dominant pre-synaptic NT,
      with photoreceptors forced to ``'HIS'``; may be NaN for cell types
      that never appear pre-synaptically),
      ``neuropils`` (``list[str]``; union of every neuropil this neuron
      has at least one synapse in).
- ``synapses.parquet`` — universal directed edges (whole connectome).
    columns:
      ``pre_root_id, post_root_id, synapse_counts, nt_type, neuropil``.
- ``optic_lobe_column_index.parquet`` — per-neuron optic-lobe hex-column
    assignment (loaded by :class:`regions.OpticLobeRegion`).
    columns: ``root_id, column_index`` (0-indexed).

The empirical per-column geometry file ``optic_lobe_columns.parquet`` is
NOT written here — it ships with the raw data and must already exist in
the baked directory before models can load.

To add support for another area, just declare a new ``RegionView``
subclass with ``neuropils = [...]`` in :mod:`regions`. No change needed
here.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from drosophila_model.config import load_config


_ROOT_ID_DTYPE = np.int64
_PHOTORECEPTOR_TYPES = ('R1-6', 'R7', 'R8')
# CT1 per-column synthetic split: these primary types live only in the
# optic-lobe-specific visual_neuron_types CSV. The synthetic root IDs are
# referenced in connections + column_assignments as if they were normal
# neurons, but consolidated_cell_types.csv only has 2 orphan "real CT1"
# rows (no synapses). We import the split rows here so the universal
# cell_info has them typed.
_CT1_SPLIT_TYPES = ('CT1_L', 'CT1_M')


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
    config = load_config()
    folder = Path(config['data']['data_folder'])
    raw_dir = folder / 'connectome_data' / 'raw'
    baked_dir = folder / 'connectome_data' / 'baked'

    print(f"Reading raw CSVs from {raw_dir} ...")
    cell_types_df = pd.read_csv(
        raw_dir / 'consolidated_cell_types.csv',
        dtype={'root_id': _ROOT_ID_DTYPE},
    )
    connections_df = pd.read_csv(
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
    visual_neurons_df = pd.read_csv(
        raw_dir / 'visual_neuron_types_no_threshold.csv',
        dtype={'root_id': _ROOT_ID_DTYPE},
    )

    # Append CT1_L / CT1_M per-column synthetic neurons from the visual
    # CSV; they're real first-class IDs in connections + column_assignments
    # but only typed in this file.
    ct1_split = visual_neurons_df[
        visual_neurons_df['type'].isin(_CT1_SPLIT_TYPES)
    ][['root_id', 'type']].rename(columns={'type': 'primary_type'})
    ct1_split['additional_type(s)'] = np.nan
    n_before = len(cell_types_df)
    cell_types_df = pd.concat(
        [cell_types_df, ct1_split[cell_types_df.columns.tolist()]],
        ignore_index=True,
    )
    print(f"Appended {len(cell_types_df) - n_before:,} CT1_L/CT1_M synthetic "
          f"per-column neurons from visual_neuron_types_no_threshold.csv.")

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
