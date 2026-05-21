"""Load slices of the canonical connectome bundle.

The bundle is produced once by :mod:`format_connectome_data` and stored under
``config['data']['connectome_folder']``:

- ``cell_info.parquet`` — universal per-neuron table (``root_id, type,
  neurotransmitter, area, side`` plus optional area-specific per-neuron
  columns such as ``column_index``).
- ``synapses.parquet`` — universal directed edges (``pre_root_id,
  post_root_id, synapse_counts, nt_type``).
- ``area_extras/<area>_*.parquet`` — per-area geometry / lookups, owned and
  loaded by the corresponding :class:`regions.RegionView` subclass.

:func:`load_connectome` reads ``cell_info.parquet`` + ``synapses.parquet``,
filters by ``areas`` and ``side``, and returns a :class:`ConnectomeData`
payload. The model and regions stay anatomy-agnostic about the bundle layout;
any per-neuron column on ``cell_info.parquet`` outside the core set is
forwarded verbatim via :attr:`ConnectomeData.per_neuron_extras` so that
regions can claim what they need (e.g. ``OpticLobeRegion`` reads
``column_index``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from drosophila_model import data_utilities as du


_CORE_CELL_INFO_COLS = ('root_id', 'type', 'neurotransmitter', 'area', 'side')


@dataclass
class ConnectomeData:
    """Universal connectome payload consumed by :class:`ConnectomeModel`.

    Anatomy-agnostic: holds only fields that make sense for every brain area.
    Per-area geometry (e.g. optic-lobe ``column_phi_theta``) is loaded by the
    corresponding :class:`regions.RegionView` subclass straight from the
    on-disk bundle at ``connectome_folder / 'area_extras'``.

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
        areas: list of area names actually present (input filter).
        side: side filter applied at load (``'right'`` / ``'left'`` / ``'both'``).
        connectome_folder: bundle path; regions use it to find ``area_extras``.
    """

    synapse_counts_coo: np.ndarray
    root_ids: np.ndarray
    cell_types: np.ndarray
    neurotransmitters: np.ndarray
    area_per_neuron: np.ndarray
    side_per_neuron: np.ndarray
    per_neuron_extras: dict[str, np.ndarray] = field(default_factory=dict)
    areas: list[str] = field(default_factory=list)
    side: str = 'both'
    connectome_folder: str = ''


def load_connectome(
    areas: list[str],
    *,
    side: str = 'both',
    connectome_folder: Optional[str] = None,
) -> ConnectomeData:
    """Load a multi-area slice of the canonical connectome bundle.

    Args:
        areas: Area names to include, e.g. ``['optic_lobe']`` or
            ``['optic_lobe', 'antenna_lobe']``. Matched against the ``area``
            column of ``cell_info.parquet``.
        side: ``'right'`` / ``'left'`` / ``'both'``. Filters bilateral
            regions by the ``side`` column. ``'both'`` keeps every side,
            including non-bilateral (e.g. central-brain) neurons.
        connectome_folder: Override the data folder. Defaults to
            ``config['data']['connectome_folder']``.
    """
    if connectome_folder is None:
        connectome_folder = du.load_config()['data']['connectome_folder']
    folder = Path(connectome_folder)

    cell_info_df = pd.read_parquet(folder / 'cell_info.parquet')
    synapses_df = pd.read_parquet(folder / 'synapses.parquet')

    return load_connectome_from_dataframes(
        cell_info_df, synapses_df,
        areas=areas, side=side, connectome_folder=str(folder),
    )


def load_connectome_from_dataframes(
    cell_info_df: pd.DataFrame,
    synapses_df: pd.DataFrame,
    *,
    areas: list[str],
    side: str = 'both',
    connectome_folder: str = '',
) -> ConnectomeData:
    """Build a :class:`ConnectomeData` from already-loaded DataFrames.

    Useful for analysis scripts that filtered cell types via
    :func:`data_utilities.get_connectome_subset` before calling this.
    ``connectome_folder`` should still point at the on-disk bundle so that
    region subclasses can find their ``area_extras`` files.
    """
    if not areas:
        raise ValueError("`areas` must be a non-empty list of area names.")
    if side not in ('right', 'left', 'both'):
        raise ValueError(f"side must be 'right'|'left'|'both', got {side!r}")
    for col in _CORE_CELL_INFO_COLS:
        if col not in cell_info_df.columns:
            raise ValueError(
                f"cell_info is missing required column {col!r}. "
                f"Re-run format_connectome_data.py to regenerate the bundle."
            )

    area_mask = cell_info_df['area'].isin(areas)
    if side == 'both':
        side_mask = np.ones(len(cell_info_df), dtype=bool)
    else:
        # Keep matched bilateral side plus anything non-lateralised.
        side_mask = cell_info_df['side'].isin([side, 'central', 'left,right', None])
    keep = area_mask & side_mask
    if not keep.any():
        raise ValueError(
            f"No neurons match areas={areas} side={side!r}. "
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
        areas=list(areas),
        side=side,
        connectome_folder=connectome_folder,
    )
