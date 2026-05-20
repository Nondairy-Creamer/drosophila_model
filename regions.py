"""Region views: per-anatomical-area handles into a :class:`ConnectomeModel`.

A region is a named subset of the model's neurons that exposes area-specific
operations *and* owns any area-specific on-disk data. The model itself stays
anatomy-agnostic; adding a new brain area means adding a new
``RegionView`` subclass, registering it under its area name, and (if the area
has geometry) shipping an ``area_extras/<name>_*.parquet`` file that the
region's ``_load_extras`` reads at construction time.

A single ``ConnectomeModel`` may host any combination of regions, picked at
construction by the config ``areas`` list.

``OpticLobeRegion`` covers both the single-eye case (mirror trick) and the
two-eye case (column slicing) with the same API, chosen by the ``side``
parameter the loader applied (``'right'`` / ``'left'`` / ``'both'``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from neural_models import ConnectomeModel


class RegionView:
    """A named subset of neurons inside a :class:`ConnectomeModel`.

    Attributes:
        name: Area name (e.g. ``'optic_lobe'``).
        model: Back-reference to the owning ``ConnectomeModel``.
        cell_indices: ``(N_r,)`` int64 — neuron indices into the model's flat
            ``num_neurons`` axis.
        side: ``'right'`` / ``'left'`` / ``'both'`` — sides of this bilateral
            region represented in the loaded slice.
        num_sides: 1 if a single hemisphere was loaded, 2 if both.
    """

    def __init__(self, *, name: str, model: 'ConnectomeModel',
                 cell_indices: np.ndarray, side: str):
        if side not in ('right', 'left', 'both'):
            raise ValueError(f"side must be 'right'|'left'|'both', got {side!r}")
        self.name = name
        self.model = model
        self.cell_indices = cell_indices
        self.side = side
        self.num_sides = 2 if side == 'both' else 1
        self._load_extras(Path(model.connectome_folder) / 'area_extras')

    def _load_extras(self, extras_dir: Path) -> None:
        """Override to read this region's ``area_extras/<name>_*.parquet`` file.

        Default is a no-op for areas that don't need per-element geometry.
        """
        pass

    def __repr__(self):
        return (f"{type(self).__name__}(name={self.name!r}, side={self.side!r}, "
                f"num_cells={len(self.cell_indices)})")


# ---------------------------------------------------------------------------
# Optic lobe
# ---------------------------------------------------------------------------


class OpticLobeRegion(RegionView):
    """Optic-lobe region. Owns the hex-grid column geometry.

    On construction reads ``area_extras/optic_lobe_columns.parquet`` and
    populates:

    - ``column_indices``: ``(C_r,)`` int64 — unique column indices present in
      this slice of the connectome (de-duped from the per-neuron column index).
    - ``column_phi_theta``: ``(C_r, 2)`` float — ``(phi, theta)`` per column,
      ordered to match ``column_indices``.
    - ``column_p_q``: ``(C_r, 2)`` int — hex lattice coords per column.
    - ``column_phi_theta_two_eyes``: torch tensors used by
      :meth:`predict_two_eyes` and :meth:`estimate_ego_motion_from_image`.
      Single-side host mirrors phi for a virtual second eye; two-side host
      uses native columns directly.

    ``num_eyes`` is an alias for ``num_sides`` since the optic lobe maps 1:1
    onto eyes.
    """

    def _load_extras(self, extras_dir: Path) -> None:
        model = self.model

        # Region-local set of unique column indices (drop NaNs from neurons
        # without a column assignment, like the boundary cell types).
        col_per_neuron = model.column_indices_per_neuron[self.cell_indices]
        col_vals = col_per_neuron[~np.isnan(col_per_neuron)]
        self.column_indices = np.unique(col_vals).astype(np.int64)

        columns_df = pd.read_parquet(extras_dir / 'optic_lobe_columns.parquet')
        columns_df = columns_df.set_index('column_index').loc[self.column_indices]

        self.column_phi_theta = columns_df[['phi', 'theta']].to_numpy()
        self.column_p_q = columns_df[['p', 'q']].to_numpy().astype(np.int64)

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

    @property
    def num_eyes(self) -> int:
        """Optic-lobe-terminology alias for :attr:`num_sides`."""
        return self.num_sides

    # ------------------------------------------------------------------
    # Two-eye predict + ego-motion
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

        rot_flat, trans_flat, _ = model.estimate_ego_motion_from_flow(
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


def build_regions(model: 'ConnectomeModel') -> dict[str, RegionView]:
    """Instantiate one ``RegionView`` per area present in ``model``'s connectome.

    Uses ``model.area_per_neuron`` to assign neuron indices to each registered
    area. Each region's ``__init__`` calls ``_load_extras`` to ingest its own
    on-disk data from ``model.connectome_folder/area_extras/``.
    """
    regions: dict[str, RegionView] = {}
    for area in sorted(set(model.area_per_neuron)):
        cls = REGION_REGISTRY.get(area)
        if cls is None:
            continue

        cell_mask = (model.area_per_neuron == area)
        cell_idx = np.flatnonzero(cell_mask).astype(np.int64)

        regions[area] = cls(
            name=area, model=model,
            cell_indices=cell_idx, side=model.side,
        )
    return regions
