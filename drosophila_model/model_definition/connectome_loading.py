"""Load region slices of the canonical neuropil-keyed connectome bundle.

The bundle is produced once by :mod:`format_connectome_data` and stored under
``<[data].data_folder>/connectome_data/baked``:

- ``cell_info.parquet`` — universal per-neuron table:
  ``root_id, primary_type, neurotransmitter, neuropils`` (``neuropils`` is
  a ``list[str]`` column holding the union of every neuropil this neuron
  has at least one synapse in; the loader requires the column but does not
  use it — membership comes from the thresholded synapse table).
- ``synapses.parquet`` — universal directed edges, one row per
  ``(pre, post, neuropil)``:
  ``pre_root_id, post_root_id, synapse_counts, nt_type, neuropil``.
- ``<region>_*.parquet`` — per-region geometry / lookups, owned and loaded
  by the corresponding :class:`regions.RegionView` subclass (e.g.
  ``optic_lobe_columns.parquet``, written by ``build_column_geometry``, and
  ``optic_lobe_column_index.parquet``).

:func:`load_connectome` takes a ``{region_name: {'neuropils': [...],
'side': 'right'|'left'|'both', 'intrinsic': bool}}`` mapping. Each region
declares its bilateral neuropil base names (e.g. ``['ME', 'LO', 'LOP',
'LA']`` for the optic lobe); the loader resolves them against ``side`` by
appending ``_R`` / ``_L`` suffixes as needed. ``count_threshold`` applies
to the total synapse count of a neuron pair, summed over neuropils. Membership
uses the above-threshold connections: with ``intrinsic=False`` (default) a
neuron is claimed if it has at least one above-threshold connection with
synapses in any of a region's resolved neuropils; with ``intrinsic=True`` only
if *every* one of its above-threshold connections lies within them. Membership is **union** across regions — the same
neuron can belong to multiple regions, but only contributes one row to
the model's neuron axis.

The returned :class:`ConnectomeData` carries the per-region boolean
membership masks (aligned with ``root_ids``); the model builds each
region's ``cell_indices`` from those.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from drosophila_model.config import get_data_folder, load_config


# ``neuropils`` is part of the bundle schema but not read here (membership is
# computed from the synapse table).
_CORE_CELL_INFO_COLS = ('root_id', 'primary_type', 'neurotransmitter', 'neuropils')
_VALID_SIDES = ('right', 'left', 'both')


@dataclass
class ConnectomeData:
    """Universal connectome payload consumed by :class:`ConnectomeModel`.

    Anatomy-agnostic: holds only fields that make sense for every brain
    area, plus per-region boolean membership masks. Per-area geometry
    (e.g. optic-lobe ``column_phi_theta``) is loaded by the corresponding
    :class:`regions.RegionView` subclass straight from the on-disk bundle
    at ``data_folder / 'connectome_data' / 'baked'``.

    Fields:
        synapse_counts_coo: ``(E, 3)`` int64 —
            ``[pre_root_id, post_root_id, count]``, one row per neuron pair
            (the pair's ``synapses.parquet`` rows summed over neuropils), so
            the model's ``synapse_count_threshold`` applies per pair.
        root_ids: ``(N,)`` int64 — per-neuron root IDs.
        cell_types: ``(N,)`` str — per-neuron primary cell-type label.
        neurotransmitters: ``(N,)`` str.
        region_membership: ``{region_name: (N,) bool}`` aligned with
            ``root_ids``. True where the neuron meets that region's
            membership rule against the thresholded synapse table (see
            :func:`load_connectome_from_dataframes`): for
            ``intrinsic=False`` regions, at least one above-threshold
            connection in the region's neuropils; for ``intrinsic=True``,
            all of the neuron's above-threshold connections within them. A
            neuron may be in multiple regions but only contributes one row
            to ``root_ids``.
        region_specs: input filter applied at load — ``{region: {neuropils,
            side, intrinsic}}``. Each region carries its own side and
            membership stringency.
        data_folder: bundle path; regions use it to find their per-region parquets.
    """

    synapse_counts_coo: np.ndarray
    root_ids: np.ndarray
    cell_types: np.ndarray
    neurotransmitters: np.ndarray
    region_membership: dict[str, np.ndarray] = field(default_factory=dict)
    region_specs: dict[str, dict] = field(default_factory=dict)
    data_folder: str = ''

    def to_serializable_dict(self) -> dict:
        """Return this connectome's fields as a plain dict for serialization.

        Every field is already plain data (numpy arrays, dicts, str), so the
        result carries no reference to this class. Saved models store *this*
        rather than the object itself, so a model file never couples to where
        :class:`ConnectomeData` lives. Rebuild with ``ConnectomeData(**d)`` (or
        :meth:`from_serializable_dict`).
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_serializable_dict(cls, d: dict) -> "ConnectomeData":
        """Rebuild a :class:`ConnectomeData` from :meth:`to_serializable_dict`."""
        return cls(**d)

    def drop_cell_types(self, cell_types_to_remove) -> "ConnectomeData":
        """Return a copy with every neuron of ``cell_types_to_remove`` removed.

        Drops the matching per-neuron rows (and the edges touching them) so a
        smaller model can be rebuilt from the result. ``synapse_counts_coo`` is
        in root-ID space, so edges are filtered against the retained root IDs.
        """
        remove = np.asarray(list(cell_types_to_remove))
        keep = ~np.isin(self.cell_types, remove)
        kept_root_ids = self.root_ids[keep]
        coo = self.synapse_counts_coo
        edge_keep = (
            np.isin(coo[:, 0], kept_root_ids) & np.isin(coo[:, 1], kept_root_ids)
        )
        return ConnectomeData(
            synapse_counts_coo=coo[edge_keep],
            root_ids=self.root_ids[keep],
            cell_types=self.cell_types[keep],
            neurotransmitters=self.neurotransmitters[keep],
            region_membership={
                name: mask[keep] for name, mask in self.region_membership.items()
            },
            region_specs={name: dict(spec) for name, spec in self.region_specs.items()},
            data_folder=self.data_folder,
        )


def resolve_sided_neuropils(
    neuropils: list[str],
    side: str,
    *,
    available: Optional[set[str]] = None,
) -> set[str]:
    """Expand a region's bilateral neuropil base names to fully-sided names.

    - Names that already end in ``_L`` / ``_R`` pass through unchanged.
    - Bilateral base names (e.g. ``'ME'``) are expanded to ``'ME_R'`` /
      ``'ME_L'`` per ``side``.
    - When ``available`` is supplied, a bare base name (e.g. ``'EB'``,
      ``'FB'``, ``'GNG'``) is also kept as a candidate if the dataset
      contains it unsided — that's how central / non-lateralised neuropils
      get matched. Without ``available`` we can't distinguish central from
      bilateral, so only sided variants are emitted.

    If ``available`` is supplied, candidates not present in that set are
    dropped so we don't claim neuropils the dataset doesn't actually have.
    """
    if side not in _VALID_SIDES:
        raise ValueError(f"side must be one of {_VALID_SIDES!r}, got {side!r}")
    out: set[str] = set()
    for n in neuropils:
        if n.endswith('_L') or n.endswith('_R'):
            candidates = {n}
        else:
            candidates: set[str] = set()
            if side in ('right', 'both'):
                candidates.add(f"{n}_R")
            if side in ('left', 'both'):
                candidates.add(f"{n}_L")
            if available is not None and n in available:
                # Central / non-lateralised: dataset has the bare name.
                candidates.add(n)
        if available is not None:
            candidates &= available
        out |= candidates
    return out


def load_connectome(
    region_specs: dict[str, dict],
    *,
    count_threshold: int,
    data_folder: Optional[str] = None,
) -> ConnectomeData:
    """Load a multi-region slice of the canonical connectome bundle.

    Args:
        region_specs: ``{region_name: {'neuropils': [...], 'side': str,
            'intrinsic': bool}}``. ``neuropils`` is a list of bilateral base
            names (e.g. ``['ME', 'LO', 'LOP', 'LA']`` for the optic lobe)
            and/or already-sided names (e.g. ``'EB'``, ``'ME_R'``). ``side``
            is ``'right'`` / ``'left'`` / ``'both'`` and controls the
            ``_L``/``_R`` expansion of bilateral names. ``intrinsic``
            (default ``False``) controls membership stringency — see
            :func:`load_connectome_from_dataframes`.
        count_threshold: Minimum total synapse count of a neuron pair (summed
            over neuropils) for the connection to count toward region
            membership. Pass the model's
            ``synapse_count_threshold`` so membership is computed against the
            same thresholded matrix that drives the dynamics.
        data_folder: Override the data folder. Defaults to
            ``[data].data_folder`` of ``config.toml``.
    """
    if data_folder is None:
        data_folder = get_data_folder(load_config())
    folder = Path(data_folder)

    baked = folder / 'connectome_data' / 'baked'
    cell_info_df = pd.read_parquet(baked / 'cell_info.parquet')
    synapses_df = pd.read_parquet(baked / 'synapses.parquet')

    return load_connectome_from_dataframes(
        cell_info_df, synapses_df,
        region_specs=region_specs, data_folder=str(folder),
        count_threshold=count_threshold,
    )


def load_connectome_from_dataframes(
    cell_info_df: pd.DataFrame,
    synapses_df: pd.DataFrame,
    *,
    region_specs: dict[str, dict],
    count_threshold: int,
    data_folder: str = '',
) -> ConnectomeData:
    """Build a :class:`ConnectomeData` from already-loaded DataFrames.

    Used by analysis scripts that pre-filter cell types via
    :func:`get_connectome_subset` before calling this.
    ``data_folder`` should still point at the on-disk bundle so region
    subclasses can find their per-area parquet files.

    Region membership is computed against the **thresholded** connectome:
    only neuron pairs whose total synapse count (summed over neuropils)
    reaches ``count_threshold`` count, so weak connections never opt a neuron
    into a region. A connection is "in" every neuropil where it has synapses.
    Each region's ``intrinsic`` flag controls the rule:

    - ``intrinsic=False`` (default): a neuron is in region R if it has at
      least one above-threshold connection (as pre OR post) in any of R's
      resolved neuropils.
    - ``intrinsic=True``: a neuron is in region R only if it has an
      above-threshold connection in R AND *every* one of its above-threshold
      connections lies within R's resolved neuropils (drops neurons that also
      project/receive strongly outside the area).

    ``count_threshold`` should match the model's ``synapse_count_threshold``
    so membership and dynamics see the same matrix. The returned
    ``synapse_counts_coo`` has one row per neuron pair.
    """
    if not region_specs:
        raise ValueError("`region_specs` must be a non-empty {region: {...}} dict.")
    for col in _CORE_CELL_INFO_COLS:
        if col not in cell_info_df.columns:
            raise ValueError(
                f"cell_info is missing required column {col!r}. "
                f"Re-run format_connectome_data.py to regenerate the bundle."
            )
    for col in ('pre_root_id', 'post_root_id', 'synapse_counts', 'neuropil'):
        if col not in synapses_df.columns:
            raise ValueError(
                f"synapses is missing required column {col!r}. "
                f"Re-run format_connectome_data.py to regenerate the bundle."
            )

    # Universe of neuropils that actually appear in the synapse table; used to
    # drop bogus or out-of-dataset entries from a region's resolved set.
    available_neuropils: set[str] = set(
        synapses_df['neuropil'].dropna().unique().tolist()
    )

    # Validate region_specs shape, then resolve each region's neuropil set.
    resolved: dict[str, set[str]] = {}
    intrinsic_flags: dict[str, bool] = {}
    for name, spec in region_specs.items():
        if not isinstance(spec, dict):
            raise TypeError(
                f"region_specs[{name!r}] must be a dict with keys "
                f"'neuropils', 'side' and optional 'intrinsic', got "
                f"{type(spec).__name__}."
            )
        neuropils = spec.get('neuropils')
        side = spec.get('side', 'both')
        if neuropils is None:
            raise ValueError(
                f"region_specs[{name!r}] is missing required key 'neuropils'."
            )
        intrinsic_flags[name] = bool(spec.get('intrinsic', False))
        resolved[name] = resolve_sided_neuropils(
            list(neuropils), side, available=available_neuropils,
        )
        if not resolved[name]:
            raise ValueError(
                f"Region {name!r} (neuropils={list(neuropils)!r}, side={side!r}) "
                f"resolved to an empty neuropil set against the available bundle "
                f"({sorted(available_neuropils)[:8]}... +{len(available_neuropils)-8}). "
                f"Check the neuropil names or re-bake the bundle."
            )

    # Per-neuron set of neuropils carrying an above-threshold connection (as
    # pre OR post). The threshold applies to each pair's total over all
    # neuropils; a strong pair counts in every neuropil it has synapses in.
    pair_total = synapses_df.groupby(
        ['pre_root_id', 'post_root_id'])['synapse_counts'].transform('sum')
    strong = synapses_df.loc[
        pair_total.abs() >= count_threshold,
        ['pre_root_id', 'post_root_id', 'neuropil'],
    ].dropna(subset=['neuropil'])
    endpoints = pd.concat([
        strong[['pre_root_id', 'neuropil']].rename(columns={'pre_root_id': 'root_id'}),
        strong[['post_root_id', 'neuropil']].rename(columns={'post_root_id': 'root_id'}),
    ], ignore_index=True).drop_duplicates()
    root_ids = cell_info_df['root_id'].to_numpy()

    # Per-region membership mask. ``intrinsic=False``: any above-threshold
    # connection in R. ``intrinsic=True``: at least one in R and all of the
    # neuron's above-threshold connections within R.
    region_membership_indf: dict[str, np.ndarray] = {}
    keep = np.zeros(len(cell_info_df), dtype=bool)
    for name, region_set in resolved.items():
        in_region = endpoints['neuropil'].isin(region_set)
        any_in = np.isin(root_ids, endpoints.loc[in_region, 'root_id'].to_numpy())
        if intrinsic_flags[name]:
            any_out = np.isin(root_ids, endpoints.loc[~in_region, 'root_id'].to_numpy())
            mask = any_in & ~any_out
        else:
            mask = any_in
        region_membership_indf[name] = mask
        keep |= mask

    if not keep.any():
        raise ValueError(
            f"No neurons match region_specs={region_specs} at "
            f"count_threshold={count_threshold}. "
            f"Sample available neuropils: {sorted(available_neuropils)[:12]}."
        )

    cell_info_df = cell_info_df[keep].reset_index(drop=True)
    region_membership = {
        name: mask[keep] for name, mask in region_membership_indf.items()
    }

    kept_ids = set(cell_info_df['root_id'].to_numpy().tolist())
    # Region membership defines the *neuron* set (neurons with at least one
    # synapse in a claimed neuropil). Connectivity is then a property of those
    # neurons: keep every synapse between two kept neurons regardless of which
    # neuropil it sits in. A synapse between two AL neurons that happens to be
    # made outside the AL is still part of this circuit and must be included.
    # One row per pair: sum the pair's per-neuropil rows so the model's
    # synapse_count_threshold applies to whole connections.
    syn = synapses_df[
        synapses_df['pre_root_id'].isin(kept_ids)
        & synapses_df['post_root_id'].isin(kept_ids)
    ].groupby(['pre_root_id', 'post_root_id'], as_index=False)['synapse_counts'].sum()

    return ConnectomeData(
        synapse_counts_coo=syn[
            ['pre_root_id', 'post_root_id', 'synapse_counts']
        ].to_numpy(dtype=np.int64),
        root_ids=cell_info_df['root_id'].to_numpy(),
        cell_types=cell_info_df['primary_type'].to_numpy(),
        neurotransmitters=cell_info_df['neurotransmitter'].to_numpy(),
        region_membership=region_membership,
        region_specs={name: dict(spec) for name, spec in region_specs.items()},
        data_folder=data_folder,
    )


def get_connectome_subset(
    cell_info_df: pd.DataFrame,
    synapse_counts_df: pd.DataFrame,
    cell_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter connectome data to include only specified cell types.

    Used by analysis/training scripts to pre-filter cell types before calling
    :func:`load_connectome_from_dataframes`.

    Args:
        cell_info_df: DataFrame with cell information (must have 'primary_type'
            and 'root_id' columns, plus the other columns
            :func:`load_connectome_from_dataframes` requires).
        synapse_counts_df: DataFrame with synapse counts (must have
            'pre_root_id' and 'post_root_id' columns).
        cell_types: List of primary cell-type names to keep.

    Returns:
        Tuple of (filtered_cell_info_df, filtered_synapse_counts_df).
    """
    missing = sorted(set(cell_types) - set(cell_info_df['primary_type']))
    if missing:
        warnings.warn(
            f"use_connectome_subset: cell types not in the connectome are ignored: {missing}",
            stacklevel=2,
        )
    filtered_cell_info_df = cell_info_df[
        cell_info_df['primary_type'].isin(cell_types)
    ].copy().reset_index(drop=True)

    valid_root_ids = set(filtered_cell_info_df['root_id'])

    filtered_synapse_counts_df = synapse_counts_df[
        synapse_counts_df['pre_root_id'].isin(valid_root_ids) &
        synapse_counts_df['post_root_id'].isin(valid_root_ids)
    ].copy().reset_index(drop=True)

    return filtered_cell_info_df, filtered_synapse_counts_df
