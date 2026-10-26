"""Split CT1 into per-column pieces using the FlyWire synapse table.

CT1 is one amacrine neuron per optic lobe whose arbors tile every medulla and
lobula column. That is too large to model as a single cell, so each CT1 is
replaced by one synthetic neuron per column and compartment: ``CT1_M``
(medulla) and ``CT1_L`` (lobula).

Each CT1 synapse in ``ME_*`` / ``LO_*`` goes to the column holding the
majority of its ``K_NEAREST`` nearest synapses made by unicolumnar neurons
(the ``column_assignment`` types other than ``MULTICOLUMNAR_TYPES``, whose
arbors span several columns), in the same neuropil and hemisphere. CT1
synapses in other neuropils (about 0.3%, mostly the lobula plate) and CT1
autapses are dropped; every other CT1 synapse is kept, so the pieces carry
CT1's medulla and lobula connectivity exactly.

Reads from ``<data_folder>/connectome_data/raw`` (FlyWire Codex downloads):

- ``fafb_v783_princeton_synapse_table.csv.gz`` — per-synapse positions.
- ``consolidated_cell_types.csv.gz`` — to find the CT1 root IDs.
- ``column_assignment.csv.gz`` — the reference neurons' columns.
- ``connections_princeton_no_threshold.csv.gz`` — ``nt_type`` per pair and
  neuropil (the synapse table has none).

Writes to ``<data_folder>/connectome_data/baked``:

- ``ct1_pieces.parquet`` — ``root_id`` (synthetic, see ``PIECE_ID_BASE``),
  ``primary_type``, ``parent_root_id``, ``hemisphere``, ``column_id``,
  ``p``, ``q``.
- ``ct1_connections.parquet`` — the connections table's CT1 rows with CT1
  replaced by its pieces: ``pre_root_id, post_root_id, neuropil, syn_count,
  nt_type``.

``format_connectome_data`` swaps these into the baked bundle (and runs this
split itself when the two files are missing). The synapse table pass takes
about a minute.

Usage:
    python -m drosophila_model.data_handling.split_ct1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
from scipy import stats
from scipy.spatial import cKDTree

from drosophila_model.config import get_data_folder, load_config

SYNAPSE_TABLE = 'fafb_v783_princeton_synapse_table.csv.gz'
# The synapse table stores root IDs without this common prefix.
_ROOT_ID_PREFIX = 720575940 * 10 ** 9
_SYNAPSE_COLUMNS = ('ctr_x', 'ctr_y', 'ctr_z', 'pre_root_id_720575940',
                    'post_root_id_720575940', 'neuropil')
# Compartment, hemisphere and piece-ID code of each neuropil CT1 is split in.
COMPARTMENTS = {
    'ME_R': ('CT1_M', 'right', 1), 'LO_R': ('CT1_L', 'right', 2),
    'ME_L': ('CT1_M', 'left', 3), 'LO_L': ('CT1_L', 'left', 4),
}
# Columnar types whose arbors span several columns; their column labels
# would blur the vote, so they are not used as references.
MULTICOLUMNAR_TYPES = ('T2', 'T2a', 'T3', 'Tm3',
                       'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d')
K_NEAREST = 50
# Synthetic piece IDs (stable across re-bakes, far above any FlyWire root ID):
# PIECE_ID_BASE + 100_000 * parent index (CT1 root IDs in sorted order)
# + 10_000 * compartment code (COMPARTMENTS) + column_id.
PIECE_ID_BASE = 900_000_000_000_000_000


def read_synapses(path: Path, ct1_ids: np.ndarray, reference_ids: np.ndarray):
    """Stream the synapse table once.

    Returns ``(ct1_synapses, reference_synapses)``: every synapse touching
    ``ct1_ids``, and every ``ME``/``LO`` synapse touching ``reference_ids``.
    Root IDs come back with the prefix restored; the empty neuropil is
    ``'UNASGD'``, as in the connections table.
    """
    types = {name: pa.int32() for name in _SYNAPSE_COLUMNS[:3]}
    types.update({_SYNAPSE_COLUMNS[3]: pa.int64(), _SYNAPSE_COLUMNS[4]: pa.int64(),
                  'neuropil': pa.string()})
    reader = pv.open_csv(
        path, read_options=pv.ReadOptions(block_size=64 << 20),
        convert_options=pv.ConvertOptions(include_columns=list(_SYNAPSE_COLUMNS),
                                          column_types=types),
    )
    ct1 = pa.array(ct1_ids - _ROOT_ID_PREFIX)
    refs = pa.array(reference_ids - _ROOT_ID_PREFIX)
    split_neuropils = pa.array(list(COMPARTMENTS))
    ct1_parts, ref_parts = [], []
    for batch in reader:
        pre, post = batch.column(_SYNAPSE_COLUMNS[3]), batch.column(_SYNAPSE_COLUMNS[4])
        ct1_parts.append(batch.filter(pc.or_(pc.is_in(pre, ct1), pc.is_in(post, ct1))))
        is_ref = pc.and_(pc.is_in(batch.column('neuropil'), split_neuropils),
                         pc.or_(pc.is_in(pre, refs), pc.is_in(post, refs)))
        ref_parts.append(batch.filter(pc.fill_null(is_ref, False)))

    def to_frame(parts):
        df = pa.Table.from_batches(parts).to_pandas()
        df = df.rename(columns={_SYNAPSE_COLUMNS[3]: 'pre_root_id',
                                _SYNAPSE_COLUMNS[4]: 'post_root_id'})
        df['pre_root_id'] += _ROOT_ID_PREFIX
        df['post_root_id'] += _ROOT_ID_PREFIX
        df['neuropil'] = df['neuropil'].replace('', 'UNASGD').fillna('UNASGD')
        return df

    return to_frame(ct1_parts), to_frame(ref_parts)


def _assign_columns(ct1_synapses: pd.DataFrame, reference_synapses: pd.DataFrame,
                    reference_columns: pd.DataFrame) -> np.ndarray:
    """Column of each CT1 synapse (NaN outside the split neuropils)."""
    info = reference_columns.set_index('root_id')[['column_id', 'hemisphere']]
    xyz = ['ctr_x', 'ctr_y', 'ctr_z']
    # One reference point per (synapse, reference neuron) end, labelled with
    # that neuron's column; only ends in the neuron's own hemisphere count.
    ends = pd.concat([
        reference_synapses[xyz + ['neuropil']].assign(root_id=reference_synapses[end])
        for end in ('pre_root_id', 'post_root_id')
    ]).join(info, on='root_id', how='inner')
    columns = np.full(len(ct1_synapses), np.nan)
    for neuropil, (_, hemisphere, _) in COMPARTMENTS.items():
        ref = ends[(ends['neuropil'] == neuropil) & (ends['hemisphere'] == hemisphere)]
        sel = (ct1_synapses['neuropil'] == neuropil).to_numpy()
        if not sel.any():
            continue
        _, idx = cKDTree(ref[xyz].to_numpy(np.float64)).query(
            ct1_synapses.loc[sel, xyz].to_numpy(np.float64), k=K_NEAREST, workers=-1)
        votes = ref['column_id'].to_numpy()[idx.reshape(int(sel.sum()), K_NEAREST)]
        columns[sel] = stats.mode(votes, axis=1, keepdims=False).mode
    return columns


def split_ct1(data_folder) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split CT1 into per-column pieces; write and return the two tables."""
    raw_dir = Path(data_folder) / 'connectome_data' / 'raw'
    baked_dir = Path(data_folder) / 'connectome_data' / 'baked'
    t0 = time.time()

    cell_types = pd.read_csv(raw_dir / 'consolidated_cell_types.csv.gz')
    ct1_ids = np.sort(cell_types.loc[cell_types['primary_type'] == 'CT1', 'root_id'].to_numpy())
    columns = pd.read_csv(raw_dir / 'column_assignment.csv.gz').dropna(subset=['column_id'])
    columns['column_id'] = columns['column_id'].astype(np.int64)
    reference = columns[~columns['type'].isin(MULTICOLUMNAR_TYPES)]

    print(f"Reading {SYNAPSE_TABLE} ...")
    ct1_syn, ref_syn = read_synapses(raw_dir / SYNAPSE_TABLE, ct1_ids,
                                      reference['root_id'].to_numpy())
    ct1_syn = ct1_syn[ct1_syn['pre_root_id'] != ct1_syn['post_root_id']].reset_index(drop=True)
    print(f"  {len(ct1_syn):,} CT1 synapses, {len(ref_syn):,} reference synapses "
          f"({time.time() - t0:.0f}s)")

    ct1_syn['column_id'] = _assign_columns(ct1_syn, ref_syn, reference)
    kept = ct1_syn.dropna(subset=['column_id']).copy()
    kept['column_id'] = kept['column_id'].astype(np.int64)
    print(f"  assigned {len(kept):,} synapses to columns; dropped {len(ct1_syn) - len(kept):,} "
          f"outside {sorted(COMPARTMENTS)}")

    # One piece per (parent, compartment, column) that received synapses. The
    # parent_* columns keep the original IDs (the partner itself for non-CT1
    # ends), which the nt_type lookup below needs.
    parent_index = pd.Series(np.arange(len(ct1_ids)), index=ct1_ids)
    kept['primary_type'] = kept['neuropil'].map({n: c[0] for n, c in COMPARTMENTS.items()})
    kept['hemisphere'] = kept['neuropil'].map({n: c[1] for n, c in COMPARTMENTS.items()})
    code = kept['neuropil'].map({n: c[2] for n, c in COMPARTMENTS.items()}).to_numpy(np.int64)
    for end in ('pre_root_id', 'post_root_id'):
        kept['parent_' + end] = kept[end]
        is_ct1 = kept[end].isin(ct1_ids).to_numpy()
        piece_ids = (PIECE_ID_BASE
                     + 100_000 * kept[end].map(parent_index).fillna(0).to_numpy(np.int64)
                     + 10_000 * code + kept['column_id'].to_numpy(np.int64))
        kept[end] = np.where(is_ct1, piece_ids, kept[end].to_numpy())

    pieces = pd.concat([
        kept.loc[kept['parent_' + end].isin(ct1_ids),
                 [end, 'primary_type', 'parent_' + end, 'hemisphere', 'column_id']]
        .set_axis(['root_id', 'primary_type', 'parent_root_id', 'hemisphere', 'column_id'], axis=1)
        for end in ('pre_root_id', 'post_root_id')
    ]).drop_duplicates('root_id')
    pq = columns.drop_duplicates(['hemisphere', 'column_id'])[['hemisphere', 'column_id', 'p', 'q']]
    pieces = (pieces.merge(pq, on=['hemisphere', 'column_id'], how='left')
              .sort_values('root_id').reset_index(drop=True))

    # Re-aggregate to the connections layout; nt_type comes from the parent
    # CT1's row of the connections table.
    conn = pd.read_csv(raw_dir / 'connections_princeton_no_threshold.csv.gz')
    conn = conn[conn['pre_root_id'].isin(ct1_ids) | conn['post_root_id'].isin(ct1_ids)]
    keys = ['pre_root_id', 'post_root_id', 'neuropil']
    ct1_conn = (kept.groupby(keys + ['parent_pre_root_id', 'parent_post_root_id'])
                .size().rename('syn_count').reset_index())
    nt = conn.set_index(keys)['nt_type']
    ct1_conn['nt_type'] = nt.reindex(pd.MultiIndex.from_frame(
        ct1_conn[['parent_pre_root_id', 'parent_post_root_id', 'neuropil']])).to_numpy()
    ct1_conn = ct1_conn[keys + ['syn_count', 'nt_type']]

    baked_dir.mkdir(parents=True, exist_ok=True)
    pieces.to_parquet(baked_dir / 'ct1_pieces.parquet', index=False)
    ct1_conn.to_parquet(baked_dir / 'ct1_connections.parquet', index=False)
    counts = pieces.groupby(['hemisphere', 'primary_type']).size().to_dict()
    print(f"Wrote {len(pieces):,} CT1 pieces {counts} and {len(ct1_conn):,} connection rows "
          f"to {baked_dir} ({time.time() - t0:.0f}s)")
    return pieces, ct1_conn


def main() -> None:
    argparse.ArgumentParser(
        description='Split CT1 into per-column CT1_M / CT1_L pieces from the raw '
                    'FlyWire synapse table ([data].data_folder of config.toml).'
    ).parse_args()
    split_ct1(get_data_folder(load_config()))


if __name__ == '__main__':
    main()
