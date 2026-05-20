"""Standardized connectome loaders.

Single entry point :func:`load_connectome` reads the connectome parquet bundle,
filters by ``areas`` (e.g. ``['optic_lobe']``) and ``side`` (``'right'`` /
``'left'`` / ``'both'``), and returns a :class:`ConnectomeData` dataclass
holding everything :class:`neural_models.ConnectomeModel` needs to set itself
up — including ``area_per_neuron`` and ``side_per_neuron`` arrays so the model
can register one :class:`regions.RegionView` per requested area.

The brain is bilaterally symmetric, so ``side`` is a model-wide filter
(applies to every bilateral region, not just the optic lobe).

Callers that already have the parquet ``DataFrame`` objects in memory (e.g.
analysis scripts that filter to a subset of cell types first) can skip the
parquet round-trip and use :func:`load_connectome_from_dataframes` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import data_utilities as du


# Default mapping from cell-info row to area, used when the parquet doesn't yet
# carry an explicit 'area' column. The current bundle is entirely optic-lobe
# right hemisphere, so every neuron gets ``'optic_lobe'`` by default. When the
# whole-brain bundle lands and adds an 'area' column, this fallback becomes a
# no-op.
_DEFAULT_AREA = 'optic_lobe'


@dataclass
class ConnectomeData:
    """Standard connectome payload consumed by :class:`ConnectomeModel`.

    Fields:
        synapse_counts_coo: ``(E, 3)`` int64 — ``[pre_root_id, post_root_id, count]``.
        root_ids: ``(N,)`` int64 — per-neuron root IDs.
        cell_types: ``(N,)`` object/str — per-neuron cell type label.
        neurotransmitters: ``(N,)`` object/str.
        column_indices: ``(N,)`` float — column index per neuron (NaN allowed).
        column_phi_theta: ``(num_columns, 2)`` float — ``(phi, theta)`` per column.
        column_p_q: optional ``(num_columns, 2)`` int64 — hex p,q coords.
        area_per_neuron: ``(N,)`` object/str — area name each neuron belongs to.
        side_per_neuron: ``(N,)`` object/str — ``'right'`` / ``'left'`` / ``'central'``.
        areas: list of area names actually present (input filter).
        side: side filter applied at load (``'right'`` / ``'left'`` / ``'both'``).
    """

    synapse_counts_coo: np.ndarray
    root_ids: np.ndarray
    cell_types: np.ndarray
    neurotransmitters: np.ndarray
    column_indices: np.ndarray
    column_phi_theta: np.ndarray
    column_p_q: Optional[np.ndarray]
    area_per_neuron: np.ndarray
    side_per_neuron: np.ndarray
    areas: list[str]
    side: str


def load_connectome(
    areas: list[str],
    *,
    side: str = 'both',
    connectome_folder: Optional[str] = None,
) -> ConnectomeData:
    """Load a multi-area slice of the connectome.

    Args:
        areas: Area names to include, e.g. ``['optic_lobe']`` or
            ``['optic_lobe', 'antenna_lobe']``. Matched against the ``area``
            column of ``*_cell_info.parquet`` (defaulted to ``'optic_lobe'``
            when the column is absent).
        side: ``'right'`` / ``'left'`` / ``'both'`` — filters bilateral regions
            by the ``side`` column. ``'both'`` keeps every side, including
            non-bilateral (e.g. central-brain) neurons.
        connectome_folder: Override the data folder. Defaults to
            ``config['data']['connectome_folder']``.
    """
    if not areas:
        raise ValueError("`areas` must be a non-empty list of area names.")
    if side not in ('right', 'left', 'both'):
        raise ValueError(f"side must be 'right'|'left'|'both', got {side!r}")

    if connectome_folder is None:
        connectome_folder = du.load_config()['data']['connectome_folder']
    folder = Path(connectome_folder)

    cell_info_df, column_info_df, synapse_counts_df = _read_parquets(folder)

    return load_connectome_from_dataframes(
        cell_info_df, column_info_df, synapse_counts_df,
        areas=areas, side=side,
    )


def load_connectome_from_dataframes(
    cell_info_df: pd.DataFrame,
    column_info_df: pd.DataFrame,
    synapse_counts_df: pd.DataFrame,
    *,
    areas: list[str],
    side: str = 'both',
) -> ConnectomeData:
    """Filter pre-loaded parquet DataFrames by ``areas`` + ``side`` and wrap them
    into a :class:`ConnectomeData`.

    Useful for analysis scripts that already filtered cell types via
    ``data_utilities.get_connectome_subset`` before calling this.
    """
    if not areas:
        raise ValueError("`areas` must be a non-empty list of area names.")
    if side not in ('right', 'left', 'both'):
        raise ValueError(f"side must be 'right'|'left'|'both', got {side!r}")

    cell_info_df = cell_info_df.copy()

    if 'area' not in cell_info_df.columns:
        cell_info_df['area'] = _DEFAULT_AREA
    if 'side' not in cell_info_df.columns:
        cell_info_df['side'] = 'right'

    area_mask = cell_info_df['area'].isin(areas)
    if side == 'both':
        side_mask = np.ones(len(cell_info_df), dtype=bool)
    else:
        # Keep matched bilateral side + anything labelled as non-lateralised.
        side_mask = cell_info_df['side'].isin([side, 'central', None])
    keep = area_mask & side_mask
    if not keep.any():
        raise ValueError(
            f"No neurons match areas={areas} side={side!r}. "
            f"Available areas: {sorted(cell_info_df['area'].unique())}; "
            f"sides: {sorted(cell_info_df['side'].dropna().unique())}."
        )
    cell_info_df = cell_info_df[keep].reset_index(drop=True)

    kept_ids = set(cell_info_df['root_id'].to_numpy().tolist())
    syn = synapse_counts_df
    syn = syn[syn['pre_root_id'].isin(kept_ids) & syn['post_root_id'].isin(kept_ids)]

    column_p_q = None
    if 'p' in column_info_df.columns and 'q' in column_info_df.columns:
        column_p_q = column_info_df[['p', 'q']].to_numpy()

    return ConnectomeData(
        synapse_counts_coo=syn[
            ['pre_root_id', 'post_root_id', 'synapse_counts']
        ].to_numpy(),
        root_ids=cell_info_df['root_id'].to_numpy(),
        cell_types=cell_info_df['type'].to_numpy(),
        neurotransmitters=cell_info_df['neurotransmitter'].to_numpy(),
        column_indices=cell_info_df['column_index'].to_numpy(),
        column_phi_theta=column_info_df[['phi', 'theta']].to_numpy(),
        column_p_q=column_p_q,
        area_per_neuron=cell_info_df['area'].to_numpy(),
        side_per_neuron=cell_info_df['side'].to_numpy(),
        areas=list(areas),
        side=side,
    )


def _read_parquets(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three connectome parquet files from ``folder``.

    The current bundle uses the ``optic_lobes_*`` prefix. When the whole-brain
    bundle lands with a different naming, add the dispatch here.
    """
    cell_info_df = pd.read_parquet(folder / 'optic_lobes_cell_info.parquet')
    column_info_df = pd.read_parquet(folder / 'optic_lobes_column_info.parquet')
    synapse_counts_df = pd.read_parquet(folder / 'optic_lobes_synapse_counts.parquet')
    return cell_info_df, column_info_df, synapse_counts_df
