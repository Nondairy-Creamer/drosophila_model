"""Load region slices of the canonical neuropil-keyed connectome bundle.

The bundle is produced once by :mod:`format_connectome_data` and stored under
``config['data']['data_folder'] / 'connectome_data' / 'baked'``:

- ``cell_info.parquet`` — universal per-neuron table:
  ``root_id, primary_type, neurotransmitter, neuropils`` (``neuropils`` is
  a ``list[str]`` column holding the union of every neuropil this neuron
  has at least one synapse in).
- ``synapses.parquet`` — universal directed edges:
  ``pre_root_id, post_root_id, synapse_counts, nt_type, neuropil``.
- ``<region>_*.parquet`` — per-region geometry / lookups, owned and loaded
  by the corresponding :class:`regions.RegionView` subclass (e.g.
  ``optic_lobe_columns.parquet`` and ``optic_lobe_column_index.parquet``).

:func:`load_connectome` takes a ``{region_name: {'neuropils': [...],
'side': 'right'|'left'|'both'}}`` mapping. Each region declares its
bilateral neuropil base names (e.g. ``['ME', 'LO', 'LOP', 'LA']`` for the
optic lobe); the loader resolves them against ``side`` by appending
``_R`` / ``_L`` suffixes as needed. Neurons that have at least one
synapse in any of a region's resolved neuropils are claimed (opted into
that region's membership). Membership is **union** across regions — the
same neuron can belong to multiple regions, but only contributes one
row to the model's neuron axis.

The returned :class:`ConnectomeData` carries the per-region boolean
membership masks (aligned with ``root_ids``); the model builds each
region's ``cell_indices`` from those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from drosophila_model.config import load_config


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
        synapse_counts_coo: ``(E, 3)`` int64 — ``[pre_root_id, post_root_id, count]``.
        root_ids: ``(N,)`` int64 — per-neuron root IDs.
        cell_types: ``(N,)`` str — per-neuron primary cell-type label.
        neurotransmitters: ``(N,)`` str.
        region_membership: ``{region_name: (N,) bool}`` aligned with
            ``root_ids``. True where the neuron has at least one synapse
            in any of that region's resolved neuropils. A neuron may be
            in multiple regions (union membership) but only contributes
            one row to ``root_ids``.
        region_specs: input filter applied at load — ``{region: {neuropils,
            side}}``. Each region carries its own side.
        data_folder: bundle path; regions use it to find their per-region parquets.
    """

    synapse_counts_coo: np.ndarray
    root_ids: np.ndarray
    cell_types: np.ndarray
    neurotransmitters: np.ndarray
    region_membership: dict[str, np.ndarray] = field(default_factory=dict)
    region_specs: dict[str, dict] = field(default_factory=dict)
    data_folder: str = ''

    @property
    def regions(self) -> list[str]:
        """Regions loaded into this payload, in insertion order of ``region_specs``."""
        return list(self.region_specs.keys())


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
    data_folder: Optional[str] = None,
) -> ConnectomeData:
    """Load a multi-region slice of the canonical connectome bundle.

    Args:
        region_specs: ``{region_name: {'neuropils': [...], 'side': str}}``.
            ``neuropils`` is a list of bilateral base names (e.g.
            ``['ME', 'LO', 'LOP', 'LA']`` for the optic lobe) and/or
            already-sided names (e.g. ``'EB'``, ``'ME_R'``). ``side`` is
            ``'right'`` / ``'left'`` / ``'both'`` and controls the
            ``_L``/``_R`` expansion of bilateral names.
        data_folder: Override the data folder. Defaults to
            ``config['data']['data_folder']``.
    """
    if data_folder is None:
        data_folder = load_config()['data']['data_folder']
    folder = Path(data_folder)

    baked = folder / 'connectome_data' / 'baked'
    cell_info_df = pd.read_parquet(baked / 'cell_info.parquet')
    synapses_df = pd.read_parquet(baked / 'synapses.parquet')

    return load_connectome_from_dataframes(
        cell_info_df, synapses_df,
        region_specs=region_specs, data_folder=str(folder),
    )


def load_connectome_from_dataframes(
    cell_info_df: pd.DataFrame,
    synapses_df: pd.DataFrame,
    *,
    region_specs: dict[str, dict],
    data_folder: str = '',
) -> ConnectomeData:
    """Build a :class:`ConnectomeData` from already-loaded DataFrames.

    Used by analysis scripts that pre-filter cell types via
    :func:`data_utilities.get_connectome_subset` before calling this.
    ``data_folder`` should still point at the on-disk bundle so region
    subclasses can find their per-area parquet files.
    """
    if not region_specs:
        raise ValueError("`region_specs` must be a non-empty {region: {...}} dict.")
    for col in _CORE_CELL_INFO_COLS:
        if col not in cell_info_df.columns:
            raise ValueError(
                f"cell_info is missing required column {col!r}. "
                f"Re-run format_connectome_data.py to regenerate the bundle."
            )

    # Normalise neuropils column to per-row sets once for cheap intersection.
    neuropil_sets = cell_info_df['neuropils'].apply(
        lambda v: set(v) if v is not None else set()
    )

    # Universe of neuropils that actually appear in this cell_info; used to
    # drop bogus or out-of-dataset entries from a region's resolved set.
    available_neuropils: set[str] = set()
    for s in neuropil_sets:
        available_neuropils |= s

    # Validate region_specs shape, then resolve each region's neuropil set.
    resolved: dict[str, set[str]] = {}
    for name, spec in region_specs.items():
        if not isinstance(spec, dict):
            raise TypeError(
                f"region_specs[{name!r}] must be a dict with keys "
                f"'neuropils' and 'side', got {type(spec).__name__}."
            )
        neuropils = spec.get('neuropils')
        side = spec.get('side', 'both')
        if neuropils is None:
            raise ValueError(
                f"region_specs[{name!r}] is missing required key 'neuropils'."
            )
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

    # Per-region union-membership mask: a neuron is in region R if its
    # neuropil set intersects R's resolved neuropil set.
    region_membership_indf: dict[str, np.ndarray] = {}
    keep = np.zeros(len(cell_info_df), dtype=bool)
    for name, region_set in resolved.items():
        mask = neuropil_sets.apply(
            lambda s, r=region_set: not s.isdisjoint(r)
        ).to_numpy(dtype=bool)
        region_membership_indf[name] = mask
        keep |= mask

    if not keep.any():
        raise ValueError(
            f"No neurons match region_specs={region_specs}. "
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
    syn = synapses_df[
        synapses_df['pre_root_id'].isin(kept_ids)
        & synapses_df['post_root_id'].isin(kept_ids)
    ]

    return ConnectomeData(
        synapse_counts_coo=syn[
            ['pre_root_id', 'post_root_id', 'synapse_counts']
        ].to_numpy(),
        root_ids=cell_info_df['root_id'].to_numpy(),
        cell_types=cell_info_df['primary_type'].to_numpy(),
        neurotransmitters=cell_info_df['neurotransmitter'].to_numpy(),
        region_membership=region_membership,
        region_specs={name: dict(spec) for name, spec in region_specs.items()},
        data_folder=data_folder,
    )
