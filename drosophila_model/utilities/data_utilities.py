'''
Connectome data helpers.

Configuration loading lives in :mod:`drosophila_model.config`.
'''

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def get_connectome_subset(
    cell_info_df: "pd.DataFrame",
    synapse_counts_df: "pd.DataFrame",
    cell_types: list[str],
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Filter connectome data to include only specified cell types.

    Args:
        cell_info_df: DataFrame with cell information (must have 'primary_type'
            and 'root_id' columns, plus the 'neuropils' column the loader
            consumes).
        synapse_counts_df: DataFrame with synapse counts (must have
            'pre_root_id' and 'post_root_id' columns).
        cell_types: List of primary cell-type names to keep.

    Returns:
        Tuple of (filtered_cell_info_df, filtered_synapse_counts_df)
    """
    filtered_cell_info_df = cell_info_df[
        cell_info_df['primary_type'].isin(cell_types)
    ].copy().reset_index(drop=True)

    valid_root_ids = set(filtered_cell_info_df['root_id'])

    filtered_synapse_counts_df = synapse_counts_df[
        synapse_counts_df['pre_root_id'].isin(valid_root_ids) &
        synapse_counts_df['post_root_id'].isin(valid_root_ids)
    ].copy().reset_index(drop=True)

    return filtered_cell_info_df, filtered_synapse_counts_df
