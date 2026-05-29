"""Region views: per-anatomical-area handles into a :class:`ConnectomeModel`.

A region is a named subset of the model's neurons that:

- owns the area's on-disk geometry (loaded from the baked bundle in
  :meth:`RegionView._load_extras`),
- owns any area-specific submodules (e.g. the optic-lobe input-conv stack),
- exposes the area-specific operations (e.g. ``predict_two_eyes``).

Regions are :class:`torch.nn.Module` subclasses so their submodules /
parameters land in the parent model's ``state_dict()`` under
``regions.<area>.*`` automatically.

The model itself stays anatomy-agnostic: it asks ``self.regions['optic_lobe']``
for visual things, ``self.regions['antenna_lobe']`` for olfactory things, etc.
Adding a new area means adding a new ``RegionView`` subclass and registering
it under its area name in :data:`REGION_REGISTRY`.

``OpticLobeRegion`` covers both the single-eye case (mirror trick) and the
two-eye case (column slicing) with the same API, chosen by the ``side``
parameter the loader applied (``'right'`` / ``'left'`` / ``'both'``).
"""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import nn

from drosophila_model.utilities import ego_motion
from drosophila_model.model_definition.model_layers import HexGridConv, TemporalLRU

if TYPE_CHECKING:
    from drosophila_model.model_definition.neural_models import ConnectomeModel


class RegionView(nn.Module):
    """A named, area-specific sub-network inside a :class:`ConnectomeModel`.

    Attributes:
        name: Area name (e.g. ``'optic_lobe'``).
        model: Back-reference to the owning ``ConnectomeModel`` (weakref'd to
            avoid the model<->region cycle that would otherwise confuse
            ``nn.Module``'s submodule registration).
        cell_indices: ``(N_r,)`` int64 — neuron indices into the model's flat
            ``num_neurons`` axis.
        side: ``'right'`` / ``'left'`` / ``'both'`` — sides of this bilateral
            region represented in the loaded slice.
        num_sides: 1 if a single hemisphere was loaded, 2 if both.

    Predict-time hooks (override in subclasses that drive the network):
        default_input_neurons / default_output_neurons / default_readouts /
        default_readout_neurons:
            Class-level cell-type / readout-name lists this region exposes
            as defaults. ``ConnectomeModel.predict`` uses
            ``default_input_neurons`` / ``default_output_neurons`` when its
            ``input_neurons`` / ``output_neurons`` args are ``None``.
            ``default_readouts`` lists readout class names this region
            instantiates via :meth:`instantiate_readouts`; per-region
            override via ``[model.regions.<area>].readouts`` in config.
        build_input_mapping:
            Map a cell-type name list to the ``stim_to_cell_mapping`` dict
            consumed by ``ConnectomeModel.forward``. Default ``None`` =
            abstain.
        initialize_recording:
            Allocate this region's per-step output buffers (the recording
            structure is the region's choice — optic-lobe uses column-
            aligned ``(B, T, num_columns)`` tensors with NaN padding; an
            antenna-lobe region would use per-ORN-type ``(B, T, N_ct)``
            tensors with no NaN padding). Returns
            ``(region_outputs, region_mapping)`` or ``None``.
        record_step:
            Per-timestep writer for this region's recording buffers.
        instantiate_readouts:
            Build this region's readout submodules using its own geometry.
            Stores them on ``self._readout_modules``.
        apply_input_pre_filter:
            Apply any per-timestep input pre-filter (e.g. the optic-lobe
            HexGridConv + contrast norm + temporal LRU). Default no-op.
    """

    default_input_neurons: list[str] = []
    default_output_neurons: list[str] = []
    default_readouts: list[str] = []
    default_readout_neurons: list[str] | None = None
    # Cell types to remove from this region's slice of the connectome before
    # the model preprocesses synapses. Consumed by
    # :func:`resolve_neurons_to_exclude` and applied in
    # ``ConnectomeModel._preprocess_connectome``. Override per-instance via
    # ``region_kwargs[area]['neurons_to_exclude']``.
    default_neurons_to_exclude: list[str] = []
    # Bilateral base neuropil names this region claims (no ``_L``/``_R``
    # suffix; the loader expands by ``side``). Names that already carry a
    # suffix pass through unchanged. A neuron with at least one synapse in
    # any of these neuropils is opted into this region (union membership;
    # see :func:`connectome_loading.resolve_sided_neuropils`).
    neuropils: list[str] = []
    # Anatomy-prior sign overrides applied after NT-based signing. Each
    # entry is ``(presynaptic_cell_type, postsynaptic_cell_type, factor)``;
    # ``factor=-1`` flips the sign (e.g. for connections whose bulk-NT
    # classification disagrees with their functional sign), ``factor=0``
    # ablates. Consumed by
    # :meth:`ConnectomeModel._apply_region_sign_overrides`, overridable
    # per-instance via ``region_kwargs[<name>]['sign_overrides']``.
    sign_overrides: list[tuple[str, str, float]] = []

    def __init__(self, *, name: str, model: 'ConnectomeModel',
                 cell_indices: np.ndarray, side: str,
                 region_opts: dict | None = None):
        super().__init__()
        if side not in ('right', 'left', 'both'):
            raise ValueError(f"side must be 'right'|'left'|'both', got {side!r}")
        self.name = name
        # weakref: model owns region (via nn.ModuleDict), region looks back at
        # model. Plain attribute assignment would be fine (weakref isn't an
        # nn.Module) but using object.__setattr__ documents intent.
        object.__setattr__(self, '_model_ref', weakref.ref(model))
        # Buffers (rather than plain attrs) so .to(device) follows the model.
        self.register_buffer(
            'cell_indices',
            torch.as_tensor(cell_indices, dtype=torch.int64),
            persistent=False,
        )
        self.side = side
        self.num_sides = 2 if side == 'both' else 1

        # Empty ModuleDict so torch's submodule registration always sees
        # ``regions.<area>._readout_modules`` even before
        # :meth:`instantiate_readouts` runs.
        self._readout_modules = nn.ModuleDict()

        # Shared (region-agnostic) opts. Pop here so subclass _load_extras
        # validators don't see them as unknown keys. Falling through to the
        # class-level defaults when not set in config.
        opts = dict(region_opts or {})
        if 'input_neurons' in opts:
            self.default_input_neurons = list(opts.pop('input_neurons'))
        if 'output_neurons' in opts:
            self.default_output_neurons = list(opts.pop('output_neurons'))
        if 'sign_overrides' in opts:
            self.sign_overrides = [tuple(t) for t in opts.pop('sign_overrides')]
        readouts_cfg = opts.pop('readouts', None)
        readout_neurons_cfg = opts.pop('readout_neurons', None)
        regularizer_parameters_cfg = opts.pop('regularizer_parameters', None)

        self._load_extras(
            Path(model.data_folder) / 'connectome_data' / 'baked',
            opts,
        )

        # Per-region readout instantiation. Subclasses may override
        # :meth:`instantiate_readouts` to use their own geometry / kwargs.
        # Readouts list falls back to ``default_readouts`` when not in
        # config.
        readout_names = (
            list(readouts_cfg) if readouts_cfg is not None
            else list(self.default_readouts)
        )
        if readout_neurons_cfg is not None:
            readout_neurons = list(readout_neurons_cfg)
        elif self.default_readout_neurons is not None:
            readout_neurons = list(self.default_readout_neurons)
        else:
            readout_neurons = None
        self.instantiate_readouts(
            readout_names=readout_names,
            readout_neurons=readout_neurons,
            regularizer_parameters=regularizer_parameters_cfg,
        )

    @property
    def model(self) -> 'ConnectomeModel':
        m = self._model_ref()
        if m is None:
            raise RuntimeError(
                f"RegionView {self.name!r} has lost its model reference "
                "(model was garbage-collected)."
            )
        return m

    def _load_extras(self, extras_dir: Path, opts: dict) -> None:
        """Override to read this region's per-area parquet files and build
        any area-specific submodules.

        ``opts`` is the per-region config sub-table (from
        ``config['model']['regions'][<area>]``), forwarded through
        :func:`build_regions`. The base no-op ignores it; subclasses should
        extract the keys they accept and may warn / raise on unknown keys.
        """
        pass

    # ------------------------------------------------------------------
    # Predict-time hooks (default no-ops; override per-region)
    # ------------------------------------------------------------------

    @property
    def input_stim_size(self) -> int:
        """Size of this region's slice of the model's input axis.

        Regions that drive input return the number of stimulus channels
        they consume (e.g. hex columns for the optic lobe, glomeruli for
        the antenna lobe). Default ``0`` = this region doesn't drive
        input. Summed across regions by
        :meth:`ConnectomeModel.find_steady_state` to size the zero
        stimulus tensor.
        """
        return 0

    def build_input_mapping(self, input_neurons: list[str]):
        """Return ``(stim_to_cell_mapping, input_meta)`` or ``None``.

        ``stim_to_cell_mapping`` is the dict consumed by
        ``ConnectomeModel.forward``; ``input_meta`` is an opaque dict
        forwarded to :meth:`initialize_recording`.
        """
        return None

    def initialize_recording(self, num_batches: int, num_time: int,
                             output_neurons: list[str], input_meta: dict,
                             record_synaptic: bool):
        """Allocate this region's per-step output buffers + readout buffers.

        Returns ``(region_outputs, region_mapping)`` or ``None`` (this
        region records nothing this predict). ``region_outputs`` is the
        per-region dict that ends up under ``result['outputs'][name]``;
        ``region_mapping`` is an opaque per-region context passed to
        :meth:`record_step` plus surfaced under
        ``result['mappings'][name]``.
        """
        return None

    def record_step(self, t: int, v, fr, syn_current,
                    region_outputs: dict, region_mapping: dict) -> None:
        """Write this region's per-step values + readout outputs.

        Called once per timestep with the full-network voltage and firing
        rate. Default no-op.
        """
        return

    def apply_input_pre_filter(self, u_t, filter_state):
        """Per-timestep input pre-filter. Default no-op."""
        return u_t, filter_state

    def instantiate_readouts(self, readout_names: list[str],
                             readout_neurons: list[str] | None,
                             regularizer_parameters: dict | None) -> None:
        """Build this region's readout submodules from class names.

        Default raises if any readouts are requested, since the base
        ``RegionView`` cannot supply geometry kwargs. Subclasses override
        to wire their own geometry (e.g. ``OpticLobeRegion`` passes
        ``column_p_q`` / ``num_columns``).
        """
        if readout_names:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement instantiate_readouts "
                f"but was asked to build readouts {readout_names!r}."
            )

    def __repr__(self):
        return (f"{type(self).__name__}(name={self.name!r}, side={self.side!r}, "
                f"num_cells={len(self.cell_indices)})")


# ---------------------------------------------------------------------------
# Optic lobe
# ---------------------------------------------------------------------------


class OpticLobeRegion(RegionView):
    """Optic-lobe region. Owns the hex-grid column geometry AND the visual
    input pipeline (HexGridConv pre-filter + contrast normalisation +
    TemporalLRU).

    Configuration keys consumed from ``region_opts`` (a.k.a.
    ``config['model']['regions']['optic_lobe']``):

    - ``use_input_conv`` (default True): build and apply the input-conv stack.
    - ``conv_size_x`` (default 7): hex diameter of the contrast-normalisation
      spatial conv.
    - ``input_pre_conv_size`` (default 5): hex diameter of the pre-filter conv.
    - ``num_temporal_modes`` (default 4): complex modes in the temporal LRU.

    Geometry loaded from ``optic_lobe_columns.parquet`` in the baked bundle:

    - ``column_indices``: ``(C_r,)`` int64 — unique column indices present in
      this slice of the connectome (de-duped from the per-neuron column index).
    - ``column_phi_theta``: ``(C_r, 2)`` float — ``(phi, theta)`` per column.
    - ``column_p_q``: ``(C_r, 2)`` int — hex lattice coords per column.
    - ``column_phi_theta_two_eyes``: torch tensors used by
      :meth:`predict_two_eyes` and :meth:`estimate_ego_motion_from_image`.

    If ``use_input_conv`` is true, also builds the input-conv submodules
    (``input_pre_conv``, ``input_spatial_conv``, ``temporal_lru``) which the
    model invokes via :meth:`apply_input_conv` from inside ``predict``.

    ``num_eyes`` is an alias for ``num_sides`` since the optic lobe maps 1:1
    onto eyes.
    """

    _DEFAULT_OPTS = {
        'use_input_conv':      True,
        'conv_size_x':         7,
        'input_pre_conv_size': 5,
        'num_temporal_modes':  4,
    }

    # Visual input/output cell-type defaults for ConnectomeModel.predict.
    default_input_neurons = ['L1', 'L2', 'L3']
    default_output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
    # Photoreceptor terminals are the model's stimulus input, not connectome
    # nodes. Stripped from the optic-lobe slice before synapse preprocessing.
    default_neurons_to_exclude = ['R1-6', 'R7', 'R8']
    # Optic-lobe neuropils (intrinsic): medulla, lobula, lobula plate, lamina.
    # AME (accessory medulla) is excluded to match the previous
    # ``category == 'intrinsic'`` filter, which dropped boundary cells.
    neuropils = ['ME', 'LO', 'LOP', 'LA']
    # Anatomy-prior sign corrections. C2 and C3 are GABAergic (so the
    # bulk-NT classifier signs them negative) but their contacts onto L1
    # and L2 in the lamina are functionally excitatory — flip them back
    # to positive. Applied automatically after NT signing for every
    # dynamics model; override per-instance via
    # ``region_kwargs['optic_lobe']['sign_overrides']``.
    sign_overrides = [
        ('C2', 'L1', -1.0),
        ('C2', 'L2', -1.0),
        ('C3', 'L1', -1.0),
        ('C3', 'L2', -1.0),
    ]
    # Readouts that produce column-aligned visual outputs.
    default_readouts = ['FlowConv2Readout', 'L1Reg']
    default_readout_neurons = [
        'T1', 'T2', 'T2a', 'T3',
        'T4a', 'T4b', 'T4c', 'T4d',
        'T5a', 'T5b', 'T5c', 'T5d',
        'Tm1', 'Tm2', 'Tm3', 'Tm4',
        'Tm9', 'Tm16', 'Tm20', 'Tm28', 'Tm30',
    ]

    def _load_extras(self, extras_dir: Path, opts: dict) -> None:
        model = self.model

        # Pull recognised opts; reject unknown keys with a clear error so
        # typos in [model.regions.optic_lobe] don't silently no-op.
        unknown = set(opts) - set(self._DEFAULT_OPTS)
        if unknown:
            raise ValueError(
                f"OpticLobeRegion got unknown region_opts keys: {sorted(unknown)}. "
                f"Accepted keys: {sorted(self._DEFAULT_OPTS)}."
            )
        cfg = {**self._DEFAULT_OPTS, **opts}
        self.use_input_conv      = bool(cfg['use_input_conv'])
        self.conv_size_x         = int(cfg['conv_size_x'])
        self.input_pre_conv_size = int(cfg['input_pre_conv_size'])
        self.num_temporal_modes  = int(cfg['num_temporal_modes'])

        # Per-neuron hex-column index: NaN for any model neuron that the
        # optic-lobe column assignments don't cover (non-OL neurons, or OL
        # neurons without a known column). Aligned with model.root_ids.
        col_idx_df = pd.read_parquet(extras_dir / 'optic_lobe_column_index.parquet')
        col_map = dict(zip(
            col_idx_df['root_id'].to_numpy(),
            col_idx_df['column_index'].to_numpy().astype(np.int64),
        ))
        self.column_indices_per_neuron = np.array(
            [col_map.get(int(rid), np.nan) for rid in model.root_ids],
            dtype=np.float64,
        )

        # Region-local set of unique column indices (drop NaNs from neurons
        # without a column assignment).
        col_per_neuron = self.column_indices_per_neuron[
            self.cell_indices.cpu().numpy()
        ]
        col_vals = col_per_neuron[~np.isnan(col_per_neuron)]
        self.column_indices = np.unique(col_vals).astype(np.int64)

        columns_df = pd.read_parquet(extras_dir / 'optic_lobe_columns.parquet')
        columns_df = columns_df.set_index('column_index').loc[self.column_indices]

        self.column_phi_theta = columns_df[['phi', 'theta']].to_numpy()
        self.column_p_q = columns_df[['p', 'q']].to_numpy().astype(np.int64)
        self.num_columns = len(self.column_indices)

        if self.num_sides == 1:
            mirror = self.column_phi_theta.copy()
            mirror[:, 0] = -mirror[:, 0]
            two_eye = np.concatenate((self.column_phi_theta, mirror), axis=0)
        else:
            two_eye = self.column_phi_theta
        self.column_phi_theta_two_eyes = (
            torch.tensor(two_eye[:, 0], device=model.device),
            torch.tensor(two_eye[:, 1], device=model.device),
        )

        if self.use_input_conv:
            self._initialize_input_conv()

    @property
    def num_eyes(self) -> int:
        """Optic-lobe-terminology alias for :attr:`num_sides`."""
        return self.num_sides

    # ------------------------------------------------------------------
    # Visual input pipeline
    # ------------------------------------------------------------------

    def _initialize_input_conv(self) -> None:
        """HexGridConv pre-filter -> local contrast normalisation -> TemporalLRU.

        Initialised with the legacy ``-10.0`` -saturating mask so the very
        first forward pass behaves like a centred identity filter.
        """
        model = self.model
        if self.conv_size_x % 2 == 0:
            raise ValueError("conv_size_x must be odd for centered spatial convolution.")
        if self.input_pre_conv_size % 2 == 0:
            raise ValueError("input_pre_conv_size must be odd for centered spatial convolution.")
        if self.num_temporal_modes < 1:
            raise ValueError("num_temporal_modes must be >= 1.")

        self.input_pre_conv = HexGridConv(
            in_features=1, out_features=1, conv_size=self.input_pre_conv_size,
            column_p_q=self.column_p_q, device=model.device, dtype=model.dtype,
            positive_weights=True, use_bias=False,
        )
        pq_offsets_pre = HexGridConv.hex_grid_pq_offsets(self.input_pre_conv_size)
        with torch.no_grad():
            self.input_pre_conv.weight.fill_(-10.0)
            for i, (dp, dq) in enumerate(pq_offsets_pre):
                if dp == 0 and dq == 0:
                    self.input_pre_conv.weight[0, i, 0] = 0.0

        self.input_spatial_conv = HexGridConv(
            in_features=1, out_features=1, conv_size=self.conv_size_x,
            column_p_q=self.column_p_q, device=model.device, dtype=model.dtype,
            positive_weights=True, use_bias=False,
        )
        pq_offsets = HexGridConv.hex_grid_pq_offsets(self.conv_size_x)
        with torch.no_grad():
            self.input_spatial_conv.weight.fill_(-10.0)
            for i, (dp, dq) in enumerate(pq_offsets):
                if max(abs(dp), abs(dq), abs(dp - dq)) <= 1:
                    self.input_spatial_conv.weight[0, i, 0] = 0.0

        self.temporal_lru = TemporalLRU(
            num_modes=self.num_temporal_modes, dt=model.dt,
            device=model.device, dtype=model.dtype,
        )

    def apply_input_conv(self, u_t, filter_state=None):
        """Apply spatial pre-filter, contrast normalisation, then temporal LRU."""
        filtered = self.input_pre_conv(u_t)

        local_mean = self.input_spatial_conv(filtered)
        centered = filtered - local_mean
        local_var = self.input_spatial_conv(centered ** 2)
        normed = -centered / torch.sqrt(local_var + 1e-12)

        y_t, new_filter_state = self.temporal_lru.step(normed, state=filter_state)
        return y_t, new_filter_state

    # ------------------------------------------------------------------
    # Column-space helpers
    # ------------------------------------------------------------------

    def get_column_xyz(self, radius: float = 1.0) -> np.ndarray:
        """Column locations projected to 3D Cartesian coordinates."""
        phi = self.column_phi_theta[:, 0]
        theta = self.column_phi_theta[:, 1]
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.stack((x, y, z), axis=1)

    def get_shared_columns(self, neuron_list):
        """Columns where all cell types in ``neuron_list`` are present.

        Returns ``(neuron_indices, column_indices)`` over the parent model's
        flat neuron axis. Used by ``ConnectomeModel.predict`` to build the
        stimulus-to-cell mapping for visual readouts.
        """
        from functools import reduce
        model = self.model
        col_per_neuron = self.column_indices_per_neuron

        neuron_indices = np.arange(model.num_neurons)
        per_type_cols = [col_per_neuron[model.cell_types == nt] for nt in neuron_list]
        per_type_cols = [c[~np.isnan(c)] for c in per_type_cols]
        shared_cols = reduce(np.intersect1d, per_type_cols)

        neuron_masks = [(model.cell_types == nt) for nt in neuron_list]
        neuron_is_in_list = np.any(np.stack(neuron_masks), axis=0)
        neuron_is_in_list = neuron_is_in_list & np.isin(col_per_neuron, shared_cols)

        return neuron_indices[neuron_is_in_list], col_per_neuron[neuron_is_in_list]

    # ------------------------------------------------------------------
    # Predict-time hooks
    # ------------------------------------------------------------------

    @property
    def input_stim_size(self) -> int:
        """One stimulus channel per hex column of this eye."""
        return self.num_columns

    def build_input_mapping(self, input_neurons):
        """Map visual cell types to hex-column stimulus indices."""
        model = self.model
        neuron_idx, column_idx = self.get_shared_columns(input_neurons)
        stim_to_cell_mapping = {
            'neuron_index': torch.tensor(
                neuron_idx, device=model.device, dtype=torch.int64,
            ),
            'stimulus_index': torch.tensor(
                column_idx, device=model.device, dtype=torch.int64,
            ),
        }
        return stim_to_cell_mapping, {'stimulus_columns_np': column_idx}

    def initialize_recording(self, num_batches, num_time, output_neurons,
                             input_meta, record_synaptic,
                             precomputed_output_mapping=None):
        """Allocate column-aligned per-cell-type and readout buffers.

        ``region_outputs`` mirrors the optic-lobe contract today:
        ``voltage`` / ``firing_rate`` (and optionally ``synaptic_current``)
        are dicts of ``(B, T, num_columns)`` tensors with NaN padding so
        column index 2 corresponds to the same retinotopic column across
        cell types. Readout buffers are ``(B, T, num_columns, *extras)``.

        If ``precomputed_output_mapping`` is provided (round-trip from a
        previous ``predict`` call), it's reused directly and
        ``output_neurons`` is ignored.
        """
        model = self.model
        device, dtype = model.device, model.dtype

        if precomputed_output_mapping is not None:
            readout_dict = precomputed_output_mapping
            output_neurons = list(readout_dict.keys())
        else:
            col_per_neuron = self.column_indices_per_neuron
            input_cols = input_meta['stimulus_columns_np']
            neuron_index = np.arange(model.num_neurons)

            readout_dict = {}
            for nt in output_neurons:
                mask = (
                    (model.cell_types == nt)
                    & ~np.isnan(col_per_neuron)
                    & np.isin(col_per_neuron, input_cols)
                )
                readout_dict[nt] = {
                    'column_indices': torch.tensor(
                        col_per_neuron[mask], device=device, dtype=torch.int64,
                    ),
                    'neuron_indices': torch.tensor(
                        neuron_index[mask], device=device, dtype=torch.int64,
                    ),
                }

        shape = (num_batches, num_time, self.num_columns)
        region_outputs: dict = {'voltage': {}, 'firing_rate': {}}
        if record_synaptic:
            region_outputs['synaptic_current'] = {}
        for nt in output_neurons:
            region_outputs['voltage'][nt] = torch.full(
                shape, float('nan'), device=device, dtype=dtype,
            )
            region_outputs['firing_rate'][nt] = torch.full(
                shape, float('nan'), device=device, dtype=dtype,
            )
            if record_synaptic:
                region_outputs['synaptic_current'][nt] = torch.full(
                    shape, float('nan'), device=device, dtype=dtype,
                )

        for name, mod in self._readout_modules.items():
            if getattr(mod, 'training_only', False):
                continue
            mod_shape = shape + mod.output_extra_dims
            region_outputs[name] = torch.full(
                mod_shape, float('nan'), device=device, dtype=dtype,
            )

        region_mapping = {
            'input_mapping': None,  # filled in by predict()
            'output_mapping': readout_dict,
            'output_neurons': list(output_neurons),
            'record_synaptic': record_synaptic,
        }
        return region_outputs, region_mapping

    def record_step(self, t, v, fr, syn_current, region_outputs, region_mapping):
        readout_dict = region_mapping['output_mapping']
        record_synaptic = region_mapping['record_synaptic']

        for name, mod in self._readout_modules.items():
            if getattr(mod, 'training_only', False):
                continue
            region_outputs[name][:, t, ...] = mod(fr)

        for nt in region_mapping['output_neurons']:
            ci = readout_dict[nt]['column_indices']
            ni = readout_dict[nt]['neuron_indices']
            valid = ci >= 0
            if torch.any(valid):
                region_outputs['voltage'][nt][:, t, ci[valid]] = v[:, ni[valid]]
                region_outputs['firing_rate'][nt][:, t, ci[valid]] = fr[:, ni[valid]]
                if record_synaptic:
                    region_outputs['synaptic_current'][nt][:, t, ci[valid]] = (
                        syn_current[:, ni[valid]]
                    )

    def apply_input_pre_filter(self, u_t, filter_state):
        if self.use_input_conv:
            return self.apply_input_conv(u_t, filter_state)
        return u_t, filter_state

    def instantiate_readouts(self, readout_names, readout_neurons,
                             regularizer_parameters):
        """Build column-aware readout submodules.

        Each readout class in :mod:`drosophila_model.model_definition.readouts`
        is instantiated with the optic-lobe geometry. Any
        ``_regularizer_config`` declared by a readout is auto-instantiated
        alongside it (e.g. ``DecoderWeightL1Reg`` tied to a specific
        decoder).
        """
        from drosophila_model.model_definition import readouts as readouts_mod

        if not readout_names:
            return

        model = self.model
        kwargs = dict(
            cell_types=model.cell_types,
            column_indices=self.column_indices_per_neuron,
            column_p_q=self.column_p_q,
            num_neurons=model.num_neurons,
            num_columns=self.num_columns,
            device=model.device,
            dtype=model.dtype,
            readout_neurons=readout_neurons,
            regularizer_parameters=regularizer_parameters,
        )
        for name in readout_names:
            cls = getattr(readouts_mod, name, None)
            if cls is None or not isinstance(cls, type):
                raise ValueError(f"Unknown readout class: {name!r}")
            mod = cls(**kwargs)
            self._readout_modules[name] = mod

            for reg_cls, weight in getattr(mod, '_regularizer_config', {}).items():
                if not weight:
                    continue
                reg_name = f"{name}_{reg_cls.__name__}"
                self._readout_modules[reg_name] = reg_cls(
                    readout_name=name, weight=weight,
                )

    # ------------------------------------------------------------------
    # Ego-motion
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_ego_motion_from_flow(phi, theta, d_right, d_up,
                                      terms=('rotation', 'trans')):
        """Solve 6-DOF ego motion from optic flow over the eye's sphere.

        Thin wrapper around :func:`ego_motion.estimate_from_flow`; lives on the
        optic-lobe region because flow is an optic-lobe concept (no other
        region produces optic flow).
        """
        return ego_motion.estimate_from_flow(
            phi, theta, d_right, d_up, terms=terms,
        )

    # ------------------------------------------------------------------
    # Two-eye predict + ego-motion-from-image
    # ------------------------------------------------------------------

    def predict_two_eyes(self, inputs_right, inputs_left,
                         init_state_right=None, init_state_left=None,
                         init_filter_state_right=None, init_filter_state_left=None,
                         **kwargs):
        """Run ``predict`` with right and left eye inputs, return per-eye results.

        Single-side host: concatenates inputs along the batch dim, runs predict
        once, splits the batch, and flips left-eye flow into the shared frame.
        Two-side host: TODO — slice both eye column-spaces from one forward pass
        (not yet exercised; will be wired when whole-brain data lands).
        """
        if self.num_sides != 1:
            raise NotImplementedError(
                "Two-side host two-eye predict not implemented yet; "
                "current data is single-side only."
            )
        model = self.model

        S = inputs_right.shape[0]
        both = torch.cat([inputs_right, inputs_left], dim=0)

        init_state = None
        if init_state_right is not None and init_state_left is not None:
            init_state = torch.cat([init_state_right, init_state_left], dim=0)

        init_filter_state = None
        if init_filter_state_right is not None and init_filter_state_left is not None:
            init_filter_state = torch.cat([init_filter_state_right, init_filter_state_left], dim=0)

        result = model.predict(
            both, init_state=init_state,
            init_filter_state=init_filter_state, **kwargs,
        )

        # Per-region outputs: split each region's tensors along the batch
        # dim. Single-host two-eye predict has only an optic_lobe region
        # today, but the loop is region-agnostic so a future host with
        # multiple regions splits all of them consistently.
        outputs = result['outputs']
        right_outputs, left_outputs = {}, {}
        for region_name, region_out in outputs.items():
            region = model.regions[region_name]
            right_region, left_region = {}, {}
            for key, val in region_out.items():
                if key in ('voltage', 'firing_rate', 'synaptic_current'):
                    right_region[key] = {k: v[:S] for k, v in val.items()}
                    left_region[key] = {k: v[S:] for k, v in val.items()}
                    continue
                mod = region._readout_modules[key] if key in region._readout_modules else None
                if mod is None:
                    right_region[key] = val
                    left_region[key] = val
                    continue
                if getattr(mod, 'training_only', False):
                    right_region[key] = val
                    left_region[key] = val
                    continue
                right_region[key] = val[:S]
                left_val = val[S:].clone()
                if mod.eye_asymmetric:
                    left_val = mod.flip_for_left_eye(left_val)
                left_region[key] = left_val
            right_outputs[region_name] = right_region
            left_outputs[region_name] = left_region

        filter_state = result['model_state']['filter_state']
        right_filter = filter_state[:S] if filter_state is not None else None
        left_filter = filter_state[S:] if filter_state is not None else None

        right_result = {
            'outputs': right_outputs,
            'model_state': {'v': result['model_state']['v'][:S], 'filter_state': right_filter},
            'mappings': result['mappings'],
        }
        left_result = {
            'outputs': left_outputs,
            'model_state': {'v': result['model_state']['v'][S:], 'filter_state': left_filter},
            'mappings': result['mappings'],
        }
        return right_result, left_result

    def estimate_ego_motion_from_image(self,
                                       right_eye_inputs=None, left_eye_inputs=None,
                                       initial_state_right=None, initial_state_left=None,
                                       initial_filter_state_right=None,
                                       initial_filter_state_left=None,
                                       oob_columns=None, cached_mappings=None,
                                       eyes=('right', 'left'),
                                       terms=('rotation', 'trans')):
        """Ego-motion from raw per-eye image inputs.

        Supports both eyes (concatenated flow with mirrored phi) or a single
        eye (flow mirrored to the shared frame). When only ``'trans'`` is
        requested, rotation is not subtracted from the flow before solving
        for translation.
        """
        model = self.model
        eyes = tuple(eyes)
        if not eyes or not set(eyes).issubset({'right', 'left'}):
            raise ValueError(
                f"`eyes` must be a non-empty subset of {{'right','left'}}, got {eyes}"
            )
        eye_inputs_map = {'right': right_eye_inputs, 'left': left_eye_inputs}
        for eye in eyes:
            if eye_inputs_map[eye] is None:
                raise ValueError(
                    f"`eyes` includes '{eye}' but {eye}_eye_inputs is None"
                )

        predict_kwargs = {}
        if cached_mappings is not None:
            optic_cached = cached_mappings.get(self.name, {})
            predict_kwargs['input_neurons'] = optic_cached.get('input_mapping')
            predict_kwargs['output_neurons'] = optic_cached.get('output_mapping')

        two_eye = set(eyes) == {'right', 'left'}

        if two_eye:
            S = right_eye_inputs.shape[0]
            right_result, left_result = self.predict_two_eyes(
                right_eye_inputs, left_eye_inputs,
                init_state_right=initial_state_right, init_state_left=initial_state_left,
                init_filter_state_right=initial_filter_state_right,
                init_filter_state_left=initial_filter_state_left,
                **predict_kwargs,
            )

            right_region_out = right_result['outputs'].get(self.name, {})
            left_region_out = left_result['outputs'].get(self.name, {})
            flow_key = next(
                (k for k, m in self._readout_modules.items()
                 if getattr(m, 'predicts_flow', False) and k in right_region_out),
                None,
            )
            if flow_key is None:
                raise ValueError("No flow readout configured; cannot estimate ego motion.")

            optic_flow_right = right_region_out[flow_key]
            optic_flow_left = left_region_out[flow_key]
            optic_flow = torch.cat((optic_flow_right, optic_flow_left), dim=2)

            phi_tensor = self.column_phi_theta_two_eyes[0].to(dtype=optic_flow.dtype)
            theta_tensor = self.column_phi_theta_two_eyes[1].to(dtype=optic_flow.dtype)
        else:
            eye = eyes[0]
            inputs = eye_inputs_map[eye]
            S = inputs.shape[0]
            init_state = initial_state_right if eye == 'right' else initial_state_left
            init_filter_state = (
                initial_filter_state_right if eye == 'right' else initial_filter_state_left
            )
            result = model.predict(
                inputs,
                init_state=init_state, init_filter_state=init_filter_state,
                **predict_kwargs,
            )

            region_out = result['outputs'].get(self.name, {})
            flow_key = next(
                (k for k, m in self._readout_modules.items()
                 if getattr(m, 'predicts_flow', False) and k in region_out),
                None,
            )
            if flow_key is None:
                raise ValueError("No flow readout configured; cannot estimate ego motion.")

            optic_flow = region_out[flow_key]
            if eye == 'left':
                mod = self._readout_modules[flow_key]
                if mod.eye_asymmetric:
                    optic_flow = mod.flip_for_left_eye(optic_flow.clone())

            phi_np = self.column_phi_theta[:, 0].copy()
            theta_np = self.column_phi_theta[:, 1].copy()
            if eye == 'left':
                phi_np = -phi_np
            phi_tensor = torch.tensor(phi_np, device=model.device, dtype=optic_flow.dtype)
            theta_tensor = torch.tensor(theta_np, device=model.device, dtype=optic_flow.dtype)

            right_result = result if eye == 'right' else None
            left_result = result if eye == 'left' else None

        valid_mask = ~torch.isnan(optic_flow[0, 0, :, 0]) & ~torch.isnan(optic_flow[0, 0, :, 1])
        if oob_columns is not None:
            valid_mask = valid_mask & ~oob_columns
        phi_valid = phi_tensor[valid_mask]
        theta_valid = theta_tensor[valid_mask]

        T_len = optic_flow.shape[1]
        d_right = optic_flow[:, :, valid_mask, 0].reshape(S * T_len, -1)
        d_up = optic_flow[:, :, valid_mask, 1].reshape(S * T_len, -1)

        rot_flat, trans_flat, _ = self.estimate_ego_motion_from_flow(
            phi_valid, theta_valid, d_right, d_up, terms=terms,
        )
        rotations = rot_flat.reshape(S, T_len, 3)
        translations = trans_flat.reshape(S, T_len, 3)

        return rotations, translations, right_result, left_result


# ---------------------------------------------------------------------------
# Antenna lobe (stub)
# ---------------------------------------------------------------------------


class AntennaLobeRegion(RegionView):
    """Antenna-lobe region. Claims every neuron with at least one synapse
    in ``AL_L`` / ``AL_R`` (the connectome's only AL neuropil pair) — so
    ORNs, LNs, and PNs all land in the same region via union membership.

    Defaults for ``predict``:

    - ``default_input_neurons``: every ORN glomerulus type in this slice
      (cell types starting with ``'ORN_'``).
    - ``default_output_neurons``: every projection-neuron type in this
      slice (cell types whose name contains ``'PN'`` — captures
      ``DA1_lPN``, ``VA1d_adPN``, ``VC5_lvPN``, ``M_vPNml53``,
      ``VP3+_l2PN``, etc.).

    Both are computed in :meth:`_load_extras` against the actual cell
    types loaded into this region instead of being hard-coded class
    constants — the connectome ships ~53 ORN types and ~43 PN types and
    their exact set drifts with each upstream re-typing. User overrides
    via ``[model.regions.antenna_lobe].input_neurons`` /
    ``.output_neurons`` still win (see :meth:`RegionView.__init__`).

    Predict-time hooks: one stimulus channel per input cell type
    (i.e. one channel per glomerular ORN type) and per-cell-type recording
    buffers shaped ``(B, T, N_ct)`` — no NaN padding because, unlike the
    optic lobe, there is no shared geometry to align across cell types.
    """

    neuropils = ['AL']

    _ORN_PREFIX = 'ORN_'
    _PN_SUBSTRING = 'PN'
    # Minimum cell count per type for inclusion in dynamic defaults.
    # The substring rule matches a long tail of singleton/doubleton
    # types from ad-hoc upstream typing (a primary type with 1 cell
    # named ``something_PN`` is rarely a real projection-neuron
    # population). Five matches the canonical >=5-cells-per-type bar
    # used elsewhere in the codebase to define ``common'' AL types.
    _MIN_CELLS_PER_TYPE = 5

    _DEFAULT_OPTS: dict = {}

    def _load_extras(self, extras_dir: Path, opts: dict) -> None:
        unknown = set(opts) - set(self._DEFAULT_OPTS)
        if unknown:
            raise ValueError(
                f"AntennaLobeRegion got unknown region_opts keys: {sorted(unknown)}. "
                f"Accepted keys: {sorted(self._DEFAULT_OPTS)}."
            )

        model = self.model
        slice_types = model.cell_types[self.cell_indices.cpu().numpy()]
        unique_types, type_counts = np.unique(slice_types, return_counts=True)
        common_types = {
            str(t) for t, c in zip(unique_types, type_counts)
            if c >= self._MIN_CELLS_PER_TYPE
        }

        # Only fill in defaults the user didn't already pin via
        # region_opts (RegionView.__init__ assigns those before us).
        if not self.default_input_neurons:
            self.default_input_neurons = sorted(
                t for t in common_types if t.startswith(self._ORN_PREFIX)
            )
        if not self.default_output_neurons:
            self.default_output_neurons = sorted(
                t for t in common_types if self._PN_SUBSTRING in t
            )

    # ------------------------------------------------------------------
    # Predict-time hooks
    # ------------------------------------------------------------------

    @property
    def input_stim_size(self) -> int:
        """One stimulus channel per input cell type (per glomerulus).

        Counts only the cell types from :attr:`default_input_neurons`
        that are actually present in the model's cell-type universe
        after preprocessing (a glomerulus dropped by the synapse-count
        threshold won't be drivable).
        """
        present = set(self.model.unique_cell_types)
        return sum(1 for t in self.default_input_neurons if t in present)

    def build_input_mapping(self, input_neurons):
        """Map each input cell type to one stimulus channel.

        Channel ``j`` drives every neuron of cell type
        ``input_neurons[j]`` in this region. Cell types not present in
        the model after preprocessing are silently skipped — their
        channel index is reused by the next present type so the
        stimulus axis stays contiguous.
        """
        model = self.model
        cell_types_np = model.cell_types
        region_neuron_set = set(self.cell_indices.cpu().numpy().tolist())

        neuron_indices: list[int] = []
        stimulus_indices: list[int] = []
        present_types: list[str] = []
        for ct in input_neurons:
            type_neurons = np.flatnonzero(cell_types_np == ct)
            type_neurons = [int(n) for n in type_neurons if int(n) in region_neuron_set]
            if not type_neurons:
                continue
            channel = len(present_types)
            present_types.append(ct)
            neuron_indices.extend(type_neurons)
            stimulus_indices.extend([channel] * len(type_neurons))

        if not present_types:
            return None

        stim_to_cell_mapping = {
            'neuron_index': torch.tensor(
                neuron_indices, device=model.device, dtype=torch.int64,
            ),
            'stimulus_index': torch.tensor(
                stimulus_indices, device=model.device, dtype=torch.int64,
            ),
        }
        return stim_to_cell_mapping, {'input_neurons': list(present_types)}

    def initialize_recording(self, num_batches, num_time, output_neurons,
                             input_meta, record_synaptic,
                             precomputed_output_mapping=None):
        """Allocate per-cell-type ``(B, T, N_ct)`` voltage / firing-rate buffers.

        No NaN padding (unlike :class:`OpticLobeRegion`) since AL has no
        shared geometric axis to align across cell types — each output
        type carries its own trailing axis sized to its own neuron count.
        """
        model = self.model
        device, dtype = model.device, model.dtype
        cell_types_np = model.cell_types
        region_neuron_set = set(self.cell_indices.cpu().numpy().tolist())

        if precomputed_output_mapping is not None:
            readout_dict = precomputed_output_mapping
            output_neurons = list(readout_dict.keys())
        else:
            readout_dict = {}
            for ct in output_neurons:
                type_neurons = np.flatnonzero(cell_types_np == ct)
                type_neurons = np.array(
                    [int(n) for n in type_neurons if int(n) in region_neuron_set],
                    dtype=np.int64,
                )
                if type_neurons.size == 0:
                    continue
                readout_dict[ct] = {
                    'neuron_indices': torch.tensor(
                        type_neurons, device=device, dtype=torch.int64,
                    ),
                }
            output_neurons = list(readout_dict.keys())

        region_outputs: dict = {'voltage': {}, 'firing_rate': {}}
        if record_synaptic:
            region_outputs['synaptic_current'] = {}
        for ct in output_neurons:
            n_ct = readout_dict[ct]['neuron_indices'].numel()
            shape = (num_batches, num_time, n_ct)
            region_outputs['voltage'][ct] = torch.zeros(
                shape, device=device, dtype=dtype,
            )
            region_outputs['firing_rate'][ct] = torch.zeros(
                shape, device=device, dtype=dtype,
            )
            if record_synaptic:
                region_outputs['synaptic_current'][ct] = torch.zeros(
                    shape, device=device, dtype=dtype,
                )

        region_mapping = {
            'input_mapping': None,  # filled in by predict()
            'output_mapping': readout_dict,
            'output_neurons': list(output_neurons),
            'record_synaptic': record_synaptic,
        }
        return region_outputs, region_mapping

    def record_step(self, t, v, fr, syn_current, region_outputs, region_mapping):
        """Per-step writer: copy each output type's neurons into its buffer."""
        readout_dict = region_mapping['output_mapping']
        record_synaptic = region_mapping['record_synaptic']

        for ct in region_mapping['output_neurons']:
            ni = readout_dict[ct]['neuron_indices']
            region_outputs['voltage'][ct][:, t, :] = v[:, ni]
            region_outputs['firing_rate'][ct][:, t, :] = fr[:, ni]
            if record_synaptic:
                region_outputs['synaptic_current'][ct][:, t, :] = syn_current[:, ni]


# ---------------------------------------------------------------------------
# Registry: area name -> RegionView subclass
# ---------------------------------------------------------------------------


REGION_REGISTRY: dict[str, type[RegionView]] = {
    'optic_lobe': OpticLobeRegion,
    'antenna_lobe': AntennaLobeRegion,
}


def build_region_specs(
    region_kwargs: dict[str, dict] | None,
) -> dict[str, dict]:
    """Translate config-style ``region_kwargs`` into loader ``region_specs``.

    For each region name present in ``region_kwargs`` and registered in
    :data:`REGION_REGISTRY`, returns ``{'neuropils': <list>, 'side': <str>}``
    where ``neuropils`` defaults to the class-level :attr:`RegionView.neuropils`
    list (overridable per-instance via ``region_kwargs[<name>]['neuropils']``)
    and ``side`` defaults to ``'both'``. Used by callers to drive
    :func:`connectome_loading.load_connectome` straight from config.
    """
    region_kwargs = region_kwargs or {}
    specs: dict[str, dict] = {}
    for name, opts in region_kwargs.items():
        cls = REGION_REGISTRY.get(name)
        if cls is None:
            continue
        opts = opts or {}
        specs[name] = {
            'neuropils': list(opts.get('neuropils', cls.neuropils)),
            'side': opts.get('side', 'both'),
        }
    return specs


def resolve_neurons_to_exclude(
    regions, region_kwargs: dict[str, dict] | None = None,
) -> dict[str, list[str]]:
    """Return ``{region: [cell_type, ...]}`` of neurons to exclude per region.

    For each region in ``regions`` registered in :data:`REGION_REGISTRY`, uses
    the per-instance override from ``region_kwargs[region]['neurons_to_exclude']``
    if set, else the region class's ``default_neurons_to_exclude``. Used by
    :meth:`ConnectomeModel._preprocess_connectome` to strip cell types that
    a region treats as non-connectome inputs (e.g. photoreceptor terminals
    for the optic lobe).
    """
    region_kwargs = region_kwargs or {}
    out: dict[str, list[str]] = {}
    for name in regions:
        cls = REGION_REGISTRY.get(name)
        if cls is None:
            continue
        override = region_kwargs.get(name, {}).get('neurons_to_exclude')
        if override is not None:
            out[name] = list(override)
        else:
            out[name] = list(cls.default_neurons_to_exclude)
    return out


def build_regions(
    model: 'ConnectomeModel',
    region_membership: dict[str, np.ndarray],
    region_specs: dict[str, dict],
    region_kwargs: dict[str, dict] | None = None,
) -> nn.ModuleDict:
    """Instantiate one ``RegionView`` per region in ``region_specs``.

    Returns an ``nn.ModuleDict`` so the regions and their submodules land in
    ``model.state_dict()`` automatically under ``regions.<name>.*``. Each
    region's ``__init__`` calls ``_load_extras`` to ingest its own on-disk
    data from ``model.data_folder/connectome_data/baked/`` and to build its
    area-specific submodules.

    ``region_membership`` is ``{name: (num_neurons,) bool}`` aligned with the
    model's flat neuron axis (post-preprocessing) — true where the neuron
    belongs to that region. Membership is union, so the same neuron can land
    in multiple regions' ``cell_indices``.

    ``region_kwargs`` maps region name to its config option sub-dict (as it
    appears under ``[model.regions.<name>]`` in the config). ``side`` and
    ``neurons_to_exclude`` are consumed upstream; everything else is
    forwarded verbatim to the region's ``_load_extras`` as ``region_opts``.
    """
    region_kwargs = dict(region_kwargs or {})
    regions = nn.ModuleDict()
    for name, spec in region_specs.items():
        cls = REGION_REGISTRY.get(name)
        if cls is None:
            continue

        mask = region_membership.get(name)
        if mask is None:
            raise ValueError(
                f"region_membership is missing entry for region {name!r}. "
                f"Loaded regions: {sorted(region_membership)!r}."
            )
        cell_idx = np.flatnonzero(mask).astype(np.int64)

        opts = dict(region_kwargs.get(name, {}))
        # Consumed upstream / by RegionView.__init__; drop so subclass
        # _load_extras unknown-key checks don't see them.
        opts.pop('neurons_to_exclude', None)
        opts.pop('neuropils', None)
        opts.pop('side', None)
        side = spec.get('side', 'both')

        regions[name] = cls(
            name=name, model=model,
            cell_indices=cell_idx, side=side,
            region_opts=opts,
        )
    return regions
