"""Region views: per-anatomical-area handles into a :class:`ConnectomeModel`.

A region is a named subset of the model's neurons that:

- owns the area's on-disk geometry (loaded from ``area_extras/`` in
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

from drosophila_model import ego_motion
from drosophila_model.model_definition.model_layers import HexGridConv, TemporalLRU

if TYPE_CHECKING:
    from drosophila_model.model_definition.model_types.neural_models import ConnectomeModel


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
    """

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
        self._load_extras(
            Path(model.connectome_folder) / 'area_extras',
            dict(region_opts or {}),
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
        """Override to read this region's ``area_extras/`` files and build
        any area-specific submodules.

        ``opts`` is the per-region config sub-table (from
        ``config['model']['regions'][<area>]``), forwarded through
        :func:`build_regions`. The base no-op ignores it; subclasses should
        extract the keys they accept and may warn / raise on unknown keys.
        """
        pass

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

    Geometry loaded from ``area_extras/optic_lobe_columns.parquet``:

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

        # Region-local set of unique column indices (drop NaNs from neurons
        # without a column assignment, like the boundary cell types).
        col_per_neuron = model.column_indices_per_neuron[
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
        col_per_neuron = model.column_indices_per_neuron

        neuron_indices = np.arange(model.num_neurons)
        per_type_cols = [col_per_neuron[model.cell_types == nt] for nt in neuron_list]
        per_type_cols = [c[~np.isnan(c)] for c in per_type_cols]
        shared_cols = reduce(np.intersect1d, per_type_cols)

        neuron_masks = [(model.cell_types == nt) for nt in neuron_list]
        neuron_is_in_list = np.any(np.stack(neuron_masks), axis=0)
        neuron_is_in_list = neuron_is_in_list & np.isin(col_per_neuron, shared_cols)

        return neuron_indices[neuron_is_in_list], col_per_neuron[neuron_is_in_list]

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

        outputs = result['outputs']
        right_outputs, left_outputs = {}, {}

        for key in ('voltage', 'firing_rate', 'synaptic_current'):
            if key in outputs:
                right_outputs[key] = {k: v[:S] for k, v in outputs[key].items()}
                left_outputs[key] = {k: v[S:] for k, v in outputs[key].items()}

        for rname, mod in model._readout_modules.items():
            if rname not in outputs:
                continue
            if getattr(mod, 'training_only', False):
                right_outputs[rname] = outputs[rname]
                left_outputs[rname] = outputs[rname]
                continue
            right_outputs[rname] = outputs[rname][:S]
            left_val = outputs[rname][S:].clone()
            if mod.eye_asymmetric:
                left_val = mod.flip_for_left_eye(left_val)
            left_outputs[rname] = left_val

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
            predict_kwargs['input_neurons'] = cached_mappings['input_mapping']
            predict_kwargs['output_neurons'] = cached_mappings['output_mapping']

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

            flow_key = next(
                (k for k, m in model._readout_modules.items()
                 if getattr(m, 'predicts_flow', False) and k in right_result['outputs']),
                None,
            )
            if flow_key is None:
                raise ValueError("No flow readout configured; cannot estimate ego motion.")

            optic_flow_right = right_result['outputs'][flow_key]
            optic_flow_left = left_result['outputs'][flow_key]
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

            flow_key = next(
                (k for k, m in model._readout_modules.items()
                 if getattr(m, 'predicts_flow', False) and k in result['outputs']),
                None,
            )
            if flow_key is None:
                raise ValueError("No flow readout configured; cannot estimate ego motion.")

            optic_flow = result['outputs'][flow_key]
            if eye == 'left':
                mod = model._readout_modules[flow_key]
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
    """Placeholder antenna-lobe region. Add olfactory-specific methods and
    ``_load_extras`` to read e.g. ``area_extras/antenna_lobe_glomeruli.parquet``
    when antenna-lobe data lands and modeling work begins.
    """


# ---------------------------------------------------------------------------
# Registry: area name -> RegionView subclass
# ---------------------------------------------------------------------------


REGION_REGISTRY: dict[str, type[RegionView]] = {
    'optic_lobe': OpticLobeRegion,
    'antenna_lobe': AntennaLobeRegion,
}


def build_regions(model: 'ConnectomeModel',
                  region_kwargs: dict[str, dict] | None = None) -> nn.ModuleDict:
    """Instantiate one ``RegionView`` per area present in ``model``'s connectome.

    Returns an ``nn.ModuleDict`` so the regions and their submodules land in
    ``model.state_dict()`` automatically under ``regions.<area>.*``. Each
    region's ``__init__`` calls ``_load_extras`` to ingest its own on-disk
    data from ``model.connectome_folder/area_extras/`` and to build its
    area-specific submodules.

    ``region_kwargs`` maps area name to that region's option sub-dict (as it
    appears under ``[model.regions.<area>]`` in the config). Sub-dicts are
    forwarded verbatim to the region's ``_load_extras``.
    """
    region_kwargs = dict(region_kwargs or {})
    regions = nn.ModuleDict()
    for area in sorted(set(model.area_per_neuron)):
        cls = REGION_REGISTRY.get(area)
        if cls is None:
            continue

        cell_mask = (model.area_per_neuron == area)
        cell_idx = np.flatnonzero(cell_mask).astype(np.int64)

        regions[area] = cls(
            name=area, model=model,
            cell_indices=cell_idx, side=model.side,
            region_opts=region_kwargs.get(area, {}),
        )
    return regions
