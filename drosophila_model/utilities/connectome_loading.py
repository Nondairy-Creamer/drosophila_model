"""Load slices of the canonical connectome bundle.

The bundle is produced once by :mod:`format_connectome_data` and stored under
``config['data']['data_folder'] / 'connectome_data' / 'baked'``:

- ``cell_info.parquet`` — universal per-neuron table (``root_id, type,
  neurotransmitter, area, side`` plus optional area-specific per-neuron
  columns such as ``column_index``).
- ``synapses.parquet`` — universal directed edges (``pre_root_id,
  post_root_id, synapse_counts, nt_type``).
- ``<area>_*.parquet`` — per-area geometry / lookups, owned and loaded by
  the corresponding :class:`regions.RegionView` subclass (e.g.
  ``optic_lobe_columns.parquet``).

:func:`load_connectome` reads ``cell_info.parquet`` + ``synapses.parquet``,
filters per area according to ``area_sides`` (a ``{area: side}`` dict), and
returns a :class:`ConnectomeData` payload. Each region owns its own side
filter, so e.g. ``{'optic_lobe': 'right', 'central_brain': 'both'}`` is fine.
The model and regions stay anatomy-agnostic about the bundle layout; any
per-neuron column on ``cell_info.parquet`` outside the core set is forwarded
verbatim via :attr:`ConnectomeData.per_neuron_extras` so that regions can
claim what they need (e.g. ``OpticLobeRegion`` reads ``column_index``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from drosophila_model.utilities import data_utilities as du


_CORE_CELL_INFO_COLS = ('root_id', 'type', 'neurotransmitter', 'area', 'side')


@dataclass
class ConnectomeData:
    """Universal connectome payload consumed by :class:`ConnectomeModel`.

    Anatomy-agnostic: holds only fields that make sense for every brain area.
    Per-area geometry (e.g. optic-lobe ``column_phi_theta``) is loaded by the
    corresponding :class:`regions.RegionView` subclass straight from the
    on-disk bundle at ``data_folder / 'connectome_data' / 'baked'``.

    Fields:
        synapse_counts_coo: ``(E, 3)`` int64 — ``[pre_root_id, post_root_id, count]``.
        root_ids: ``(N,)`` int64 — per-neuron root IDs.
        cell_types: ``(N,)`` str — per-neuron cell type label.
        neurotransmitters: ``(N,)`` str.
        area_per_neuron: ``(N,)`` str — area name each neuron belongs to.
        side_per_neuron: ``(N,)`` str — ``'right'`` / ``'left'`` / etc.
        per_neuron_extras: dict of non-core per-neuron columns from
            ``cell_info.parquet`` (each value an ``(N,)`` array aligned with
            ``root_ids``). Today the only entry is ``'column_index'`` for
            optic-lobe neurons.
        area_sides: ``{area: side}`` filter applied at load. Sides are
            ``'right'`` / ``'left'`` / ``'both'``. Each region owns its own
            side, so different areas can pull from different hemispheres.
        data_folder: bundle path; regions use it to find their per-area parquets.
    """

    synapse_counts_coo: np.ndarray
    root_ids: np.ndarray
    cell_types: np.ndarray
    neurotransmitters: np.ndarray
    area_per_neuron: np.ndarray
    side_per_neuron: np.ndarray
    per_neuron_extras: dict[str, np.ndarray] = field(default_factory=dict)
    area_sides: dict[str, str] = field(default_factory=dict)
    data_folder: str = ''

    @property
    def areas(self) -> list[str]:
        """Areas loaded into this payload, in insertion order of ``area_sides``."""
        return list(self.area_sides.keys())


def load_connectome(
    area_sides: dict[str, str],
    *,
    data_folder: Optional[str] = None,
) -> ConnectomeData:
    """Load a multi-area slice of the canonical connectome bundle.

    Args:
        area_sides: Mapping from area name to side filter. Side is one of
            ``'right'`` / ``'left'`` / ``'both'``. ``'both'`` keeps every
            side for that area (including non-bilateral / central neurons).
            Example: ``{'optic_lobe': 'right', 'antenna_lobe': 'both'}``.
        data_folder: Override the data folder. Defaults to
            ``config['data']['data_folder']``.
    """
    if data_folder is None:
        data_folder = du.load_config()['data']['data_folder']
    folder = Path(data_folder)

    baked = folder / 'connectome_data' / 'baked'
    cell_info_df = pd.read_parquet(baked / 'cell_info.parquet')
    synapses_df = pd.read_parquet(baked / 'synapses.parquet')

    return load_connectome_from_dataframes(
        cell_info_df, synapses_df,
        area_sides=area_sides, data_folder=str(folder),
    )


def load_connectome_from_dataframes(
    cell_info_df: pd.DataFrame,
    synapses_df: pd.DataFrame,
    *,
    area_sides: dict[str, str],
    data_folder: str = '',
) -> ConnectomeData:
    """Build a :class:`ConnectomeData` from already-loaded DataFrames.

    Useful for analysis scripts that filtered cell types via
    :func:`data_utilities.get_connectome_subset` before calling this.
    ``data_folder`` should still point at the on-disk bundle so that
    region subclasses can find their per-area parquet files.
    """
    if not area_sides:
        raise ValueError("`area_sides` must be a non-empty {area: side} dict.")
    for area, side in area_sides.items():
        if side not in ('right', 'left', 'both'):
            raise ValueError(
                f"side for area {area!r} must be 'right'|'left'|'both', got {side!r}"
            )
    for col in _CORE_CELL_INFO_COLS:
        if col not in cell_info_df.columns:
            raise ValueError(
                f"cell_info is missing required column {col!r}. "
                f"Re-run format_connectome_data.py to regenerate the bundle."
            )

    # OR per-area keep-masks, each respecting its own side filter.
    keep = np.zeros(len(cell_info_df), dtype=bool)
    for area, side in area_sides.items():
        area_mask = (cell_info_df['area'] == area)
        if side == 'both':
            side_mask = np.ones(len(cell_info_df), dtype=bool)
        else:
            # Keep matched bilateral side plus anything non-lateralised.
            side_mask = cell_info_df['side'].isin([side, 'central', 'left,right', None])
        keep |= area_mask & side_mask
    if not keep.any():
        raise ValueError(
            f"No neurons match area_sides={area_sides}. "
            f"Available areas: {sorted(cell_info_df['area'].unique())}; "
            f"sides: {sorted(cell_info_df['side'].dropna().unique())}."
        )
    cell_info_df = cell_info_df[keep].reset_index(drop=True)

    kept_ids = set(cell_info_df['root_id'].to_numpy().tolist())
    syn = synapses_df[
        synapses_df['pre_root_id'].isin(kept_ids)
        & synapses_df['post_root_id'].isin(kept_ids)
    ]

    extras = {
        col: cell_info_df[col].to_numpy()
        for col in cell_info_df.columns
        if col not in _CORE_CELL_INFO_COLS
    }

    return ConnectomeData(
        synapse_counts_coo=syn[
            ['pre_root_id', 'post_root_id', 'synapse_counts']
        ].to_numpy(),
        root_ids=cell_info_df['root_id'].to_numpy(),
        cell_types=cell_info_df['type'].to_numpy(),
        neurotransmitters=cell_info_df['neurotransmitter'].to_numpy(),
        area_per_neuron=cell_info_df['area'].to_numpy(),
        side_per_neuron=cell_info_df['side'].to_numpy(),
        per_neuron_extras=extras,
        area_sides=dict(area_sides),
        data_folder=data_folder,
    )
