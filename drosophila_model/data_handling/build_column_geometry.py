"""Build the optic-lobe column viewing directions from the Reiser lab eye map.

Zhao et al. (2025, Nature 646:135) measured ommatidium viewing directions by
µCT and matched them to medulla columns traced (as Mi1 neurons) in FAFB. Their
processed map ships with https://github.com/reiserlab/eyemap_T4
(``data/eyemap.RData``): ``med_xyz`` (column positions, FAFB nm),
``ucl_rot_sm`` (unit viewing directions; +x forward, +y left, +z up, as here)
and ``eyemap`` (medulla column -> ommatidium).

1. Each eye-map column is matched to a FlyWire column by a one-to-one
   assignment on its distance to each FlyWire Mi1's medulla synapses. The eye
   map's medulla is the one FlyWire calls left (older CATMAID-era FAFB papers
   name the sides the other way round), so the match uses left-hemisphere Mi1s.
2. Each direction goes to the right-optic-lobe column (the modelled one) with
   the same ``(p, q)``. ``column_id`` is not used here: for some columns it
   names different hex positions in the two hemispheres.
3. Right-lobe columns with no measured direction (mostly at the eye's rim) are
   filled by a local affine fit, over the hex plane, to the ``FILL_NEIGHBOURS``
   nearest measured columns.

Reads from ``<data_folder>/connectome_data/raw``:

- ``eyemap_T4/data/eyemap.RData`` — from eyemap_T4 at ``EYEMAP_T4_COMMIT``.
- ``column_assignment.csv.gz`` and ``fafb_v783_princeton_synapse_table.csv.gz``
  (FlyWire Codex).

Writes ``<data_folder>/connectome_data/baked/optic_lobe_columns.parquet``:
``column_index`` (Codex ``column_id - 1``), ``phi``, ``theta`` (radians; see
``coordinate_conventions.md``), ``p``, ``q``, ``x``, ``y``, ``z`` (unit
vector) and ``source`` (``'eye_map'`` or ``'interpolated'``).

Usage:
    python -m drosophila_model.data_handling.build_column_geometry
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling.split_ct1 import SYNAPSE_TABLE, read_synapses

EYEMAP_T4_COMMIT = '99d2a43123db636cedb55af9ff31a59657e7d17e'
EYEMAP_FILE = Path('eyemap_T4') / 'data' / 'eyemap.RData'
FILL_NEIGHBOURS = 12
# Eye-map columns whose match lies farther than this from the FlyWire Mi1
# (in nm) make the build fail rather than silently mismatch.
MAX_MATCH_DISTANCE_NM = 5_000
# (p, q) -> hex-plane coordinates: unit steps along p, q and p + q are the six
# neighbours of a column, 60 degrees apart.
_HEX_BASIS = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2]])


def _read_eye_map(path: Path) -> pd.DataFrame:
    """``(med_index, x, y, z, dx, dy, dz)``: column position (nm) and viewing direction."""
    import rdata

    objs = rdata.read_rda(path)

    def frame(name, cols):
        da = objs[name]
        return pd.DataFrame(np.asarray(da.values), columns=cols,
                            index=np.asarray(da.coords['dim_0']).astype(np.int64))

    med = frame('med_xyz', ['x', 'y', 'z'])
    directions = frame('ucl_rot_sm', ['dx', 'dy', 'dz'])
    pairs = pd.DataFrame(objs['eyemap'].astype(np.int64), columns=['med', 'lens'])
    eye_map = pairs.join(med, on='med').join(directions, on='lens')
    return eye_map.drop(columns='lens').rename(columns={'med': 'med_index'})


def _match_to_flywire(eye_map: pd.DataFrame, mi1_synapses: pd.DataFrame) -> pd.Series:
    """Codex ``column_id`` of each eye-map column (one-to-one, by Mi1 proximity)."""
    xyz = ['ctr_x', 'ctr_y', 'ctr_z']
    column_ids = np.sort(mi1_synapses['column_id'].unique())
    points = eye_map[['x', 'y', 'z']].to_numpy()
    cost = np.empty((len(points), len(column_ids)))
    for k, (_, syn) in enumerate(mi1_synapses.groupby('column_id', sort=True)):
        cost[:, k], _ = cKDTree(syn[xyz].to_numpy(np.float64)).query(points)
    rows, cols = linear_sum_assignment(cost)
    distance = cost[rows, cols]
    print(f"  matched {len(rows)} eye-map columns to FlyWire Mi1s: median "
          f"{np.median(distance) / 1e3:.2f} um, max {distance.max() / 1e3:.2f} um")
    if distance.max() > MAX_MATCH_DISTANCE_NM:
        raise ValueError(
            f"Eye-map column {eye_map.index[rows[distance.argmax()]]} is "
            f"{distance.max() / 1e3:.1f} um from its matched Mi1; the eye map and "
            "the FlyWire medulla no longer line up.")
    return pd.Series(column_ids[cols], index=eye_map.index[rows])


def _fill_missing(directions: np.ndarray, measured: np.ndarray, pq: np.ndarray) -> np.ndarray:
    """Fill rows of ``directions`` not ``measured`` by local affine fits over the hex plane."""
    hex_xy = pq @ _HEX_BASIS
    known_xy, known_dir = hex_xy[measured], directions[measured]
    tree = cKDTree(known_xy)
    filled = directions.copy()
    for i in np.flatnonzero(~measured):
        _, nb = tree.query(hex_xy[i], k=FILL_NEIGHBOURS)
        design = np.column_stack([known_xy[nb], np.ones(len(nb))])
        coef, *_ = np.linalg.lstsq(design, known_dir[nb], rcond=None)
        d = np.append(hex_xy[i], 1.0) @ coef
        filled[i] = d / np.linalg.norm(d)
    return filled


def build_column_geometry(data_folder) -> pd.DataFrame:
    """Build, write and return ``optic_lobe_columns.parquet``."""
    raw_dir = Path(data_folder) / 'connectome_data' / 'raw'
    baked_dir = Path(data_folder) / 'connectome_data' / 'baked'
    t0 = time.time()

    eye_map = _read_eye_map(raw_dir / EYEMAP_FILE)
    columns = pd.read_csv(raw_dir / 'column_assignment.csv.gz').dropna(subset=['column_id'])
    columns[['column_id', 'p', 'q']] = columns[['column_id', 'p', 'q']].astype(np.int64)
    mi1 = columns[(columns['type'] == 'Mi1') & (columns['hemisphere'] == 'left')]

    print(f"Reading Mi1 synapses from {SYNAPSE_TABLE} ...")
    _, syn = read_synapses(raw_dir / SYNAPSE_TABLE, np.array([], dtype=np.int64),
                           mi1['root_id'].to_numpy())
    syn = syn[syn['neuropil'] == 'ME_L']
    mi1_column = mi1.set_index('root_id')['column_id']
    mi1_synapses = pd.concat([
        syn[['ctr_x', 'ctr_y', 'ctr_z']].assign(column_id=syn[end].map(mi1_column))
        for end in ('pre_root_id', 'post_root_id')
    ]).dropna(subset=['column_id']).astype({'column_id': np.int64})
    print(f"  {len(mi1_synapses):,} left-medulla Mi1 synapse ends ({time.time() - t0:.0f}s)")

    eye_map['column_id'] = _match_to_flywire(eye_map, mi1_synapses)
    left_pq = columns[columns['hemisphere'] == 'left'].drop_duplicates('column_id') \
        .set_index('column_id')[['p', 'q']]
    measured = eye_map.dropna(subset=['column_id']).join(left_pq, on='column_id') \
        .set_index(['p', 'q'])[['dx', 'dy', 'dz']]

    right = (columns[columns['hemisphere'] == 'right'].drop_duplicates('column_id')
             .sort_values('column_id')[['column_id', 'p', 'q']].reset_index(drop=True))
    if not np.array_equal(right['column_id'].to_numpy(), np.arange(1, len(right) + 1)):
        raise ValueError("Right-hemisphere column_ids must run 1..N without gaps.")
    pq = right[['p', 'q']].to_numpy()
    lookup = measured.reindex(pd.MultiIndex.from_arrays([right['p'], right['q']]))
    is_measured = lookup.notna().all(axis=1).to_numpy()
    directions = _fill_missing(lookup.to_numpy(), is_measured, pq)
    print(f"  {int(is_measured.sum())} right-lobe columns measured, "
          f"{int((~is_measured).sum())} interpolated")

    x, y, z = directions.T
    geometry = pd.DataFrame({
        'column_index': right['column_id'].to_numpy() - 1,
        'phi': np.arctan2(y, x),
        'theta': np.arccos(np.clip(z, -1.0, 1.0)),
        'p': right['p'].to_numpy(),
        'q': right['q'].to_numpy(),
        'x': x, 'y': y, 'z': z,
        'source': np.where(is_measured, 'eye_map', 'interpolated'),
    })
    baked_dir.mkdir(parents=True, exist_ok=True)
    out_path = baked_dir / 'optic_lobe_columns.parquet'
    geometry.to_parquet(out_path, index=False)
    print(f"Wrote {len(geometry)} column directions to {out_path} ({time.time() - t0:.0f}s)")
    return geometry


def main() -> None:
    argparse.ArgumentParser(
        description='Build connectome_data/baked/optic_lobe_columns.parquet from the '
                    'Reiser lab eye map ([data].data_folder of config.toml).'
    ).parse_args()
    build_column_geometry(get_data_folder(load_config()))


if __name__ == '__main__':
    main()
