"""Standardized connectome loaders.

Single entry point ``load_connectome(name)`` returns a :class:`ConnectomeData`
dataclass holding everything :class:`neural_models.ConnectomeRnn` needs to set
itself up. The intent is that every "anatomy" (optic-lobe right hemisphere,
whole brain, antenna lobe, …) is loaded via the same call, just with a
different tag, so model code stays unaware of where the bytes come from.

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


@dataclass
class ConnectomeData:
    """Standard connectome payload consumed by :class:`ConnectomeRnn`."""

    synapse_counts_coo: np.ndarray  # (E, 3) int64 — [pre_root_id, post_root_id, count]
    root_ids: np.ndarray            # (N,) int64
    cell_types: np.ndarray          # (N,) object/str
    neurotransmitters: np.ndarray   # (N,) object/str
    column_indices: np.ndarray      # (N,) float (NaN allowed)
    column_phi_theta: np.ndarray    # (num_columns, 2) float — (phi, theta)
    column_p_q: Optional[np.ndarray]  # (num_columns, 2) int64 — hex p,q coords
    name: str                       # tag, e.g. 'optic_lobe_right'
    is_single_eye: bool             # True if eye-flip mirroring makes sense


def load_connectome(name: str, *, connectome_folder: Optional[str] = None) -> ConnectomeData:
    """Load a named connectome and return it in the standard format.

    Args:
        name: Connectome identifier. Currently supported:
            - ``'optic_lobe_right'`` — single right-eye optic lobe parquet bundle.
            - ``'whole_brain'`` — not implemented yet.
            - ``'antenna_lobe'`` — not implemented yet.
        connectome_folder: Override the data folder. Defaults to
            ``config['data']['connectome_folder']``.
    """
    if connectome_folder is None:
        connectome_folder = du.load_config()['data']['connectome_folder']
    folder = Path(connectome_folder)

    if name == 'optic_lobe_right':
        return _load_optic_lobe_right(folder)
    if name == 'whole_brain':
        raise NotImplementedError(
            "load_connectome('whole_brain') not implemented yet. "
            "Add the parquet bundle and wire it up here."
        )
    if name == 'antenna_lobe':
        raise NotImplementedError(
            "load_connectome('antenna_lobe') not implemented yet."
        )
    raise ValueError(f"Unknown connectome name: {name!r}")


def load_connectome_from_dataframes(
    cell_info_df: pd.DataFrame,
    column_info_df: pd.DataFrame,
    synapse_counts_df: pd.DataFrame,
    *,
    name: str = 'optic_lobe_right',
    is_single_eye: bool = True,
) -> ConnectomeData:
    """Wrap pre-loaded parquet DataFrames into a :class:`ConnectomeData`.

    Useful for analysis scripts that filter the DataFrames before model
    construction (e.g. ``data_utilities.get_connectome_subset``).
    """
    column_p_q = None
    if 'p' in column_info_df.columns and 'q' in column_info_df.columns:
        column_p_q = column_info_df[['p', 'q']].to_numpy()

    return ConnectomeData(
        synapse_counts_coo=synapse_counts_df[
            ['pre_root_id', 'post_root_id', 'synapse_counts']
        ].to_numpy(),
        root_ids=cell_info_df['root_id'].to_numpy(),
        cell_types=cell_info_df['type'].to_numpy(),
        neurotransmitters=cell_info_df['neurotransmitter'].to_numpy(),
        column_indices=cell_info_df['column_index'].to_numpy(),
        column_phi_theta=column_info_df[['phi', 'theta']].to_numpy(),
        column_p_q=column_p_q,
        name=name,
        is_single_eye=is_single_eye,
    )


def _load_optic_lobe_right(folder: Path) -> ConnectomeData:
    cell_info_df = pd.read_parquet(folder / 'optic_lobes_cell_info.parquet')
    column_info_df = pd.read_parquet(folder / 'optic_lobes_column_info.parquet')
    synapse_counts_df = pd.read_parquet(folder / 'optic_lobes_synapse_counts.parquet')
    return load_connectome_from_dataframes(
        cell_info_df, column_info_df, synapse_counts_df,
        name='optic_lobe_right', is_single_eye=True,
    )
