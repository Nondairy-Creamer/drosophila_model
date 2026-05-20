"""Connectome-based recurrent neural models.

The class hierarchy is::

    SerializableModel               (mixin: save/load)
    BaseRnn(SerializableModel, ...) (RNN training/eval scaffolding, no connectome)
    ConnectomeModel(BaseRnn)        (loads a multi-area connectome; owns regions)

Dynamics live in :mod:`dynamics_models` as mixins. Anatomy-region-specific
operations (e.g. optic-lobe two-eye prediction) live on
:class:`regions.RegionView` subclasses, registered onto ``self.regions`` at
construction based on the requested ``areas``. The model itself stays
anatomy-agnostic — adding a new brain area means adding a new region class,
not a new model subclass.

Concrete models are composed at runtime via :func:`compose_model_class`, which
takes the dynamics-mixin name and returns a class that mixes ``ConnectomeModel``
with that dynamics::

    ModelClass = compose_model_class('CurrentDynamics')
    cdata = connectome_loading.load_connectome(['optic_lobe'], side='right')
    model = ModelClass(connectome=cdata, ...)

Config files name the dynamics + areas + side::

    [model]
    model_type    = 'ConnectomeModel'   # only valid value today
    dynamics_type = 'CurrentDynamics'
    areas         = ['optic_lobe']
    side          = 'right'             # 'right' | 'left' | 'both'

Saved checkpoints carry the dynamics string so ``BaseRnn.load`` can re-compose
the class on load.
"""

from __future__ import annotations

from logging import warning
import inspect
import os
import subprocess
import tempfile
import warnings
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path

import numpy as np
import torch

import dynamics_models
import data_utilities as du
from connectome_loading import (
    ConnectomeData,
    load_connectome,
    load_connectome_from_dataframes,
)
from model_layers import CellTypeLayerNorm, HexGridConv, TemporalLRU
import regions as regions_mod


# ---------------------------------------------------------------------------
# SerializableModel: save/load mixin (no class registry)
# ---------------------------------------------------------------------------


class SerializableModel:
    """Save / load / init-input tracking for any ``torch.nn.Module`` subclass.

    Used by both :class:`BaseRnn` and :class:`engineering_models.ConvModel` so
    they share a single checkpoint format. Class identity on load is resolved
    via the ``_resolve_class_from_payload`` hook: the default returns the
    calling ``cls`` (e.g. ``ConvModel.load(path)`` instantiates ``ConvModel``);
    :class:`BaseRnn` overrides it to compose dynamics + anatomy from the
    ``dynamics_type`` and ``model_type`` strings saved in the payload.
    """

    # Subclasses opt into recording extra (non-parameter, non-buffer) state by
    # overriding ``_collect_extra_state`` / ``_restore_extra_state``.

    def _save_init_inputs(self):
        """Snapshot constructor arguments from the calling frame.

        Call at the start (or end) of each ``__init__``. When a child class
        calls this after ``super().__init__()``, parent-resolved defaults are
        preserved and the child's inputs are merged on top. ``**kwargs`` are
        unpacked so the saved dict stays flat.
        """
        frame = inspect.currentframe().f_back
        params = frame.f_locals.copy()
        params.pop('self', None)
        params.pop('__class__', None)
        if 'kwargs' in params:
            kwargs = params.pop('kwargs')
            params.update(kwargs)
        if hasattr(self, 'init_inputs'):
            merged = dict(self.init_inputs)
            merged.update(params)
            self.init_inputs = merged
        else:
            self.init_inputs = params

    def _collect_extra_state(self):
        """Return a dict of non-parameter, non-buffer state to serialise."""
        return {}

    def _restore_extra_state(self, state, device, dtype):
        """Restore state saved by :meth:`_collect_extra_state` (no-op default)."""
        return

    def _save_class_identity(self):
        """Class-identity fields to embed in the saved payload.

        Default just stores the class's own ``__name__`` so ``cls.load(path)``
        knows which class it's instantiating. :class:`BaseRnn` overrides this
        to additionally store ``model_type`` and ``dynamics_type`` strings
        used by :func:`compose_model_class` on load.
        """
        return {'class_name': type(self).__name__}

    @classmethod
    def _resolve_class_from_payload(cls, payload):
        """Return the concrete class to instantiate when loading.

        Default: the calling class (e.g. ``ConvModel.load(...)`` instantiates
        ``ConvModel``). :class:`BaseRnn` overrides this to compose dynamics
        + anatomy from the saved ``dynamics_type`` and ``model_type`` strings.
        """
        return cls

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'inputs': self.init_inputs,
            'state_dict': {k: v.detach().cpu() for k, v in self.state_dict().items()},
            'extra_state': self._collect_extra_state(),
            **self._save_class_identity(),
        }

        # Atomic write to prevent corruption on network filesystems.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                'wb', delete=False, dir=str(p.parent), suffix='.tmp'
            ) as tmp:
                tmp_path = tmp.name
                torch.save(payload, tmp_path)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, str(p))
        except Exception:
            if tmp_path is not None:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass
            raise

    @classmethod
    def load(cls, path, device=None, dtype=None, freeze_params: bool = True):
        path = Path(path)
        if path.is_dir():
            path = path / "model.pt"
        data = torch.load(path, map_location='cpu', weights_only=False)
        inputs = dict(data['inputs'])

        if device is not None:
            inputs['device'] = device
        else:
            try:
                inputs['device'] = du.load_config()['model']['device']
            except Exception:
                inputs.setdefault('device', 'cpu')

        if dtype is not None:
            inputs['dtype'] = dtype

        model_cls = cls._resolve_class_from_payload(data)
        inputs['init_new_model'] = False

        # Strip saved keys that no constructor in the MRO accepts (guards
        # against stray locals that older _save_init_inputs may have captured).
        accepted = set()
        for klass in model_cls.__mro__:
            if klass is object:
                continue
            init = getattr(klass, '__init__', None)
            if init is None:
                continue
            for name, p in inspect.signature(init).parameters.items():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    continue
                if name != 'self':
                    accepted.add(name)
        inputs = {k: v for k, v in inputs.items() if k in accepted}

        model = model_cls(**inputs)
        model.load_state_dict(data['state_dict'])

        if 'extra_state' in data:
            model_device = torch.device(inputs.get('device', 'cpu'))
            model_dtype = getattr(torch, inputs.get('dtype', 'float32'))
            model._restore_extra_state(data['extra_state'], model_device, model_dtype)

        if freeze_params:
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()

        # Provenance check: warn if the local code is on a different commit
        # than the model was trained on.
        model_commit = getattr(model, 'commit_hash', None)
        try:
            current_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True,
                cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            current_commit = None

        if model_commit is not None:
            print(f"Model git commit: {model_commit}")
        if current_commit is not None:
            print(f"Local git commit: {current_commit}")
        else:
            print("local commit hash not detected")

        return model


# ---------------------------------------------------------------------------
# BaseRnn: connectome-agnostic RNN training / eval scaffolding
# ---------------------------------------------------------------------------


class BaseRnn(SerializableModel, torch.nn.Module, ABC):
    """Connectome-agnostic RNN base.

    Owns ``predict``, ``set_sigma_max``, ``find_steady_state``,
    ``add_noise_to_parameters``, save/load, and the static ego-motion math.
    All connectome-specific state (synapse matrix, cell types, columns) lives
    on :class:`ConnectomeModel` so a hypothetical non-connectome RNN can sit
    alongside it.

    Concrete subclasses (via the dynamics mixins) supply:
        - ``forward(v_prev, u_t, stim_to_cell_mapping=None) -> v_next``
        - ``firing_rate(v) -> fr``
        - ``calc_sigma_max() -> sigma`` (only required if ``set_sigma_max`` is called)
        - ``_apply_sigma_scale(scale_factor)``  (mutates whichever parameter
          controls overall recurrent gain)
    """

    # Default spectral-norm target for the linearised dynamics matrix.
    max_allowed_sigma = 0.95
    # HandTunedDynamics flips this to True so predict() records synaptic current.
    records_synaptic_current = False

    def __init__(self, *,
                 nonlinearity='Softplus', device='cpu', dtype='float32',
                 dt=0.01, seed=None, init_std=0.1, commit_hash=None,
                 readouts=None, readout_neurons=None, regularizer_parameters=None,
                 use_input_conv=True, conv_size_x=7, input_pre_conv_size=5,
                 num_temporal_modes=4, use_layer_norm=True, **kwargs):
        super().__init__(**kwargs)

        self.commit_hash = commit_hash
        self.silence_val = 1e-8
        self.init_std = init_std
        self.readouts = list(readouts) if readouts else []
        self.readout_neurons = list(readout_neurons) if readout_neurons else None
        self.regularizer_parameters = dict(regularizer_parameters) if regularizer_parameters else {}
        self.use_input_conv = use_input_conv
        self.conv_size_x = conv_size_x
        self.input_pre_conv_size = input_pre_conv_size
        self.num_temporal_modes = num_temporal_modes
        self.use_layer_norm = use_layer_norm

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        if nonlinearity in ("identity", "linear"):
            self.nonlinearity = torch.nn.Identity()
        else:
            self.nonlinearity = getattr(torch.nn, nonlinearity)()

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.dt = dt

    @abstractmethod
    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        """One timestep of recurrent dynamics."""

    @abstractmethod
    def firing_rate(self, v):
        """Convert voltage to firing rate."""

    # ------------------------------------------------------------------
    # Input pipeline
    # ------------------------------------------------------------------

    def _initialize_input_conv(self, column_p_q):
        """Set up input convolution using HexGridConv + TemporalLRU.

        Pipeline: pre-filter -> local contrast normalisation -> temporal LRU.
        The pre-filter is a learned spatial conv applied before contrast
        normalisation. Called from :class:`ConnectomeModel` once column_p_q is
        known.
        """
        if self.conv_size_x % 2 == 0:
            raise ValueError("conv_size_x must be odd for centered spatial convolution.")
        if self.input_pre_conv_size % 2 == 0:
            raise ValueError("input_pre_conv_size must be odd for centered spatial convolution.")
        if self.num_temporal_modes < 1:
            raise ValueError("num_temporal_modes must be >= 1.")

        self.input_conv_size_x = self.conv_size_x

        self.input_pre_conv = HexGridConv(
            in_features=1, out_features=1, conv_size=self.input_pre_conv_size,
            column_p_q=column_p_q, device=self.device, dtype=self.dtype,
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
            column_p_q=column_p_q, device=self.device, dtype=self.dtype,
            positive_weights=True, use_bias=False,
        )
        pq_offsets = HexGridConv.hex_grid_pq_offsets(self.conv_size_x)
        with torch.no_grad():
            self.input_spatial_conv.weight.fill_(-10.0)
            for i, (dp, dq) in enumerate(pq_offsets):
                if max(abs(dp), abs(dq), abs(dp - dq)) <= 1:
                    self.input_spatial_conv.weight[0, i, 0] = 0.0

        self.temporal_lru = TemporalLRU(
            num_modes=self.num_temporal_modes, dt=self.dt,
            device=self.device, dtype=self.dtype,
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
    # Training noise / spectral-norm rescaling
    # ------------------------------------------------------------------

    def add_noise_to_parameters(self, noise_std):
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.tensor(
                    self.rng.normal(0, noise_std, param.shape),
                    dtype=param.dtype, device=param.device,
                )
                param.add_(noise)

    def set_sigma_max(self):
        """Iteratively rescale the recurrent operator to hit ``max_allowed_sigma``.

        Subclasses (dynamics mixins) must implement ``calc_sigma_max`` and
        ``_apply_sigma_scale``.
        """
        with torch.no_grad():
            max_scaling_iters = 100
            rho_tolerance = 0.01
            damping_factor = 0.3

            for iter_idx in range(max_scaling_iters):
                sigma_max = self.calc_sigma_max()
                print(iter_idx, ':', sigma_max)
                if abs(sigma_max - self.max_allowed_sigma) < rho_tolerance:
                    break
                scale_factor = (self.max_allowed_sigma / sigma_max) ** damping_factor
                self._apply_sigma_scale(scale_factor)

    # ------------------------------------------------------------------
    # Static ego-motion math (no connectome refs)
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_matrix(a):
        ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
        O = torch.zeros_like(ax)
        return torch.stack([
            torch.stack([   O, -az,  ay], dim=-1),
            torch.stack([  az,   O, -ax], dim=-1),
            torch.stack([ -ay,  ax,   O], dim=-1),
        ], dim=-2)

    @staticmethod
    def _spherical_basis(phi, theta):
        x = torch.stack((
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ), dim=1)
        e_theta = torch.stack((
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            -torch.sin(theta),
        ), dim=1)
        e_phi = torch.stack((
            -torch.sin(phi),
            torch.cos(phi),
            torch.zeros_like(phi),
        ), dim=1)
        return x, e_phi, e_theta

    @classmethod
    def _estimate_rotation_from_flow(cls, phi, theta, d_right, d_up):
        x, e_phi, e_theta = cls._spherical_basis(phi, theta)
        u = -d_right[..., None] * e_phi - d_up[..., None] * e_theta
        T_len, N = d_right.shape
        A = cls._cross_matrix(x)
        A_flat = A.reshape(3 * N, 3)
        u_flat = u.reshape(T_len, 3 * N).T
        AtA = A_flat.T @ A_flat
        Atb = A_flat.T @ u_flat
        omega_star = torch.linalg.solve(AtA, Atb).T
        return omega_star

    @classmethod
    def _estimate_translation_from_flow(cls, phi, theta, d_right, d_up):
        x, e_phi, e_theta = cls._spherical_basis(phi, theta)
        v = -d_right[..., None] * e_phi - d_up[..., None] * e_theta
        T_len, N = d_right.shape

        outer = x.unsqueeze(2) @ x.unsqueeze(1)
        I = torch.eye(3, device=x.device, dtype=x.dtype).expand(N, 3, 3)
        P = I - outer

        vx = cls._cross_matrix(v)
        C_blocks = vx @ P
        C = C_blocks.reshape(T_len, 3 * N, 3)
        _, _, Vh = torch.linalg.svd(C, full_matrices=False)
        T_dir = Vh[:, -1]

        w_i = torch.einsum('nij,tj->tni', P, T_dir)
        v_dot_w = (v * w_i).sum(dim=-1)
        w_norm_sq = (w_i * w_i).sum(dim=-1)
        weighted_sign = (v_dot_w * w_norm_sq).sum(dim=-1)
        sign_flip = torch.where(
            weighted_sign > 0,
            torch.tensor(-1.0, device=x.device, dtype=d_right.dtype),
            torch.tensor(1.0, device=x.device, dtype=d_right.dtype),
        )
        T_dir = T_dir * sign_flip[:, None]
        v_dot_w = v_dot_w * sign_flip[:, None]

        rho_star = -v_dot_w / (w_norm_sq + 1e-12)

        focus_threshold = 0.1
        valid = w_norm_sq > focus_threshold
        rho_masked = torch.where(valid, rho_star, torch.full_like(rho_star, float('nan')))
        rho_agg = torch.nanmedian(rho_masked, dim=-1).values
        rho_agg = torch.where(
            torch.isnan(rho_agg),
            torch.nanmedian(rho_star, dim=-1).values,
            rho_agg,
        )
        T_star = rho_agg[:, None] * T_dir
        return T_star, rho_star

    def estimate_ego_motion_from_flow(self, phi, theta, d_right, d_up,
                                      terms=('rotation', 'trans')):
        """Estimate ego-motion (rotation + translation) from optic flow.

        Coordinate system:
            +X: Forward (phi=0, theta=pi/2)
            +Y: Left    (phi=pi/2, theta=pi/2)
            +Z: Up      (theta=0)

        Returns ``(omega_star, T_star, rho_star)``. See the inline doc on the
        old BaseRnn for the full sign/convention/de-rotation explanation.
        """
        terms = tuple(terms)
        valid_terms = {'rotation', 'trans'}
        if not terms or not set(terms).issubset(valid_terms):
            raise ValueError(
                f"`terms` must be a non-empty subset of {valid_terms}, got {terms}"
            )

        unbatched = d_right.dim() == 1
        if unbatched:
            d_right = d_right.unsqueeze(0)
            d_up = d_up.unsqueeze(0)

        T_len, N = d_right.shape
        device, dtype = d_right.device, d_right.dtype

        if 'rotation' in terms:
            omega_star = self._estimate_rotation_from_flow(phi, theta, d_right, d_up)
        else:
            omega_star = torch.zeros(T_len, 3, device=device, dtype=dtype)

        if 'trans' in terms:
            if 'rotation' in terms:
                x, e_phi, e_theta = self._spherical_basis(phi, theta)
                A = self._cross_matrix(x)
                u_rot = torch.einsum('nij,tj->tni', A, omega_star)
                d_right_rot = -(u_rot * e_phi).sum(dim=-1)
                d_up_rot = -(u_rot * e_theta).sum(dim=-1)
                d_right_t = d_right - d_right_rot
                d_up_t = d_up - d_up_rot
            else:
                d_right_t = d_right
                d_up_t = d_up
            T_star, rho_star = self._estimate_translation_from_flow(
                phi, theta, d_right_t, d_up_t,
            )
        else:
            T_star = torch.zeros(T_len, 3, device=device, dtype=dtype)
            rho_star = torch.zeros(T_len, N, device=device, dtype=dtype)

        if unbatched:
            omega_star = omega_star.squeeze(0)
            T_star = T_star.squeeze(0)
            rho_star = rho_star.squeeze(0)

        return omega_star, T_star, rho_star

    # ------------------------------------------------------------------
    # to(): move plain (non-Parameter, non-Buffer) tensor attributes too
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if hasattr(result, 'synapse_counts') and isinstance(result.synapse_counts, torch.Tensor):
            result.synapse_counts = result.synapse_counts.to(*args, **kwargs)
        if hasattr(result, 'neuron_to_type_index') and isinstance(result.neuron_to_type_index, torch.Tensor):
            result.neuron_to_type_index = result.neuron_to_type_index.to(*args, **kwargs)
        device = None
        if args and isinstance(args[0], (torch.device, str)):
            device = torch.device(args[0])
        elif 'device' in kwargs:
            device = torch.device(kwargs['device'])
        if device is not None:
            result.device = device
        return result


# ---------------------------------------------------------------------------
# ConnectomeModel: loads a multi-area connectome and runs the predict() loop
# ---------------------------------------------------------------------------


class ConnectomeModel(BaseRnn):
    """``BaseRnn`` augmented with a multi-area connectome.

    Takes a pre-loaded :class:`ConnectomeData` (built via
    :func:`connectome_loading.load_connectome` with the desired ``areas`` /
    ``side`` filters), preprocesses it (synapse-count threshold +
    neurotransmitter filter), then registers one
    :class:`regions.RegionView` per area in ``self.regions``. Area-specific
    operations (e.g. ``predict_two_eyes`` for the optic lobe) live on those
    region objects — the model itself stays anatomy-agnostic.
    """

    def __init__(self, *,
                 connectome,
                 excitatory_nt=['ACH'], inhibitory_nt=['GLUT', 'GABA'],
                 count_threshold=2,
                 **kwargs):
        super().__init__(**kwargs)

        self.excitatory_nt = excitatory_nt
        self.inhibitory_nt = inhibitory_nt
        self.count_threshold = count_threshold

        if not isinstance(connectome, ConnectomeData):
            raise TypeError(
                f"`connectome` must be a ConnectomeData (build it via "
                f"connectome_loading.load_connectome(areas=..., side=...)). "
                f"Got {type(connectome).__name__}."
            )
        cdata = connectome
        self.areas = list(cdata.areas)
        self.side = cdata.side

        # Store the resolved ConnectomeData in init_inputs so saved
        # checkpoints are self-contained (no parquet path required to reload).
        self._save_init_inputs()
        self.init_inputs['connectome'] = cdata

        valid_neurotransmitters = self.excitatory_nt + self.inhibitory_nt
        (
            root_ids, cell_types, neurotransmitters, column_indices,
            area_per_neuron, side_per_neuron, synapse_counts_coo,
        ) = self._preprocess_connectome(
            cdata.synapse_counts_coo, cdata.root_ids, cdata.cell_types,
            cdata.neurotransmitters, cdata.column_indices,
            cdata.area_per_neuron, cdata.side_per_neuron,
            valid_neurotransmitters, count_threshold,
        )

        self.root_ids = root_ids
        self.cell_types = cell_types
        self.num_neurons = cell_types.shape[0]
        self.unique_cell_types = np.unique(cell_types)
        self.num_cell_types = self.unique_cell_types.shape[0]

        self.transmitters = neurotransmitters
        self.cell_type_transmitters = np.array([
            neurotransmitters[cell_types == ct][0] for ct in self.unique_cell_types
        ])
        self.unique_transmitters = np.unique(self.transmitters)
        self.num_transmitters = len(self.unique_transmitters)

        self.column_indices = column_indices
        self.column_phi_theta = cdata.column_phi_theta
        self.column_p_q = cdata.column_p_q
        self.num_columns = np.unique(column_indices[~np.isnan(column_indices)]).size

        self.area_per_neuron = area_per_neuron
        self.side_per_neuron = side_per_neuron

        self.register_buffer(
            'steady_state',
            torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype),
        )

        # Filtered COO (matrix-index space) and CSR synapse_counts.
        self.synapse_counts_coo = synapse_counts_coo

        rows = torch.tensor(synapse_counts_coo[:, 1], device=self.device, dtype=torch.int64)
        cols = torch.tensor(synapse_counts_coo[:, 0], device=self.device, dtype=torch.int64)
        values = torch.tensor(synapse_counts_coo[:, 2], device=self.device, dtype=self.dtype)
        indices = torch.stack([rows, cols], dim=0)
        self.synapse_counts = torch.sparse_coo_tensor(
            indices, values, size=(self.num_neurons, self.num_neurons),
            dtype=self.dtype, device=self.device,
        ).coalesce().to_sparse_csr()

        # Per-neuron -> per-cell-type index.
        self.neuron_to_type_index = torch.zeros(
            self.num_neurons, device=self.device, dtype=torch.int64,
        ) - 1
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask_np = (self.cell_types == ct)
            ct_mask = torch.from_numpy(ct_mask_np).to(device=self.device)
            self.neuron_to_type_index[ct_mask] = cti
        assert torch.all(self.neuron_to_type_index != -1), \
            'some cell type to neuron mappings not assigned'

        if self.use_layer_norm:
            self.layer_norm = CellTypeLayerNorm(
                self.num_cell_types, self.neuron_to_type_index,
                device=self.device, dtype=self.dtype,
            )

        self._initialize_readouts()
        if self.use_input_conv:
            self._initialize_input_conv(self.column_p_q)

        self.regions = regions_mod.build_regions(self)

    # ------------------------------------------------------------------
    # Connectome preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_connectome(synapse_counts_coo, root_ids, cell_types,
                               neurotransmitters, column_indices,
                               area_per_neuron, side_per_neuron,
                               valid_neurotransmitters, count_threshold):
        """Filter the raw connectome and convert root IDs to matrix indices.

        Filtering (in order):
          1. Remove connections below the synapse count threshold.
          2. Remove neurons with invalid neurotransmitters.
          3. Remove connections involving removed neurons.
          4. Remove neurons with no remaining connections.

        All per-neuron arrays (including ``area_per_neuron`` and
        ``side_per_neuron``) are reduced in lockstep. ``synapse_counts_coo``
        is rewritten from root-ID space into matrix-index space.
        """
        import pandas as pd

        neurotransmitters = np.array(neurotransmitters)
        area_per_neuron = np.array(area_per_neuron)
        side_per_neuron = np.array(side_per_neuron)
        assert root_ids.dtype == np.int64, 'root_ids must have dtype np.int64'
        assert synapse_counts_coo.dtype == np.int64, \
            'synapse_counts_coo must have dtype np.int64'

        above_threshold = np.abs(synapse_counts_coo[:, 2]) >= count_threshold
        synapse_counts_coo = synapse_counts_coo[above_threshold]

        valid_nt_mask = np.isin(neurotransmitters, valid_neurotransmitters)
        valid_ids = root_ids[valid_nt_mask]
        valid_connections = (
            np.isin(synapse_counts_coo[:, 0], valid_ids)
            & np.isin(synapse_counts_coo[:, 1], valid_ids)
        )
        synapse_counts_coo = synapse_counts_coo[valid_connections]

        connected_ids = np.unique(synapse_counts_coo[:, 0:2])
        valid_neurons = valid_nt_mask & np.isin(root_ids, connected_ids)

        root_ids = root_ids[valid_neurons]
        cell_types = cell_types[valid_neurons]
        neurotransmitters = neurotransmitters[valid_neurons]
        column_indices = column_indices[valid_neurons]
        area_per_neuron = area_per_neuron[valid_neurons]
        side_per_neuron = side_per_neuron[valid_neurons]

        root_id_index = pd.Index(root_ids)
        pre_idx = root_id_index.get_indexer(synapse_counts_coo[:, 0])
        post_idx = root_id_index.get_indexer(synapse_counts_coo[:, 1])
        synapse_counts_coo = np.column_stack([pre_idx, post_idx, synapse_counts_coo[:, 2]])

        return (root_ids, cell_types, neurotransmitters, column_indices,
                area_per_neuron, side_per_neuron, synapse_counts_coo)

    # ------------------------------------------------------------------
    # Readouts
    # ------------------------------------------------------------------

    @property
    def has_decoder(self):
        return any(getattr(mod, 'predicts_flow', False)
                   for mod in self._readout_modules.values())

    def _initialize_readouts(self):
        """Instantiate readout modules by class name from ``readout_utilities``."""
        import readout_utilities
        self._readout_modules = torch.nn.ModuleDict()
        kwargs = dict(
            cell_types=self.cell_types,
            column_indices=self.column_indices,
            column_p_q=self.column_p_q,
            num_neurons=self.num_neurons,
            num_columns=self.num_columns,
            device=self.device,
            dtype=self.dtype,
            readout_neurons=self.readout_neurons,
            regularizer_parameters=self.regularizer_parameters,
        )
        for name in self.readouts:
            cls = getattr(readout_utilities, name, None)
            if cls is None or not isinstance(cls, type):
                raise ValueError(f"Unknown readout class: {name!r}")
            mod = cls(**kwargs)
            self._readout_modules[name] = mod

            for reg_cls, weight in getattr(mod, '_regularizer_config', {}).items():
                if not weight:
                    continue
                reg_name = f"{name}_{reg_cls.__name__}"
                self._readout_modules[reg_name] = reg_cls(readout_name=name, weight=weight)

    # ------------------------------------------------------------------
    # Connectome helpers used by dynamics mixins
    # ------------------------------------------------------------------

    @staticmethod
    def _csr_row_indices(csr):
        """Expand crow_indices into a per-nnz row index tensor."""
        crow = csr.crow_indices()
        n_rows = csr.size(0)
        return torch.repeat_interleave(
            torch.arange(n_rows, device=crow.device), crow.diff()
        )

    def _csr_abs_row_col_sums(self, csr):
        """Return ``(|row|.sum, |col|.sum)`` for a CSR tensor as dense vectors."""
        col = csr.col_indices()
        row = self._csr_row_indices(csr)
        abs_vals = csr.values().abs()
        row_sum = torch.zeros(csr.size(0), device=self.device, dtype=self.dtype)
        col_sum = torch.zeros(csr.size(1), device=self.device, dtype=self.dtype)
        row_sum.index_add_(0, row, abs_vals)
        col_sum.index_add_(0, col, abs_vals)
        return row_sum, col_sum

    def _sign_synapses(self):
        """Sign ``synapse_counts`` in place by presynaptic neurotransmitter.

        Excitatory NTs -> positive, inhibitory NTs -> negative. Modifies both
        ``synapse_counts`` (CSR) and ``synapse_counts_coo`` (numpy).
        """
        sc = self.synapse_counts
        col = sc.col_indices()

        pre_neuron_indices = col.cpu().numpy()
        pre_transmitters = self.transmitters[pre_neuron_indices]

        synapse_sign = np.zeros(len(pre_transmitters), dtype=np.float32)
        for nt in self.excitatory_nt:
            synapse_sign[pre_transmitters == nt] = 1
        for nt in self.inhibitory_nt:
            synapse_sign[pre_transmitters == nt] = -1

        unknown_mask = synapse_sign == 0
        if np.any(unknown_mask):
            unknown_nts = np.unique(pre_transmitters[unknown_mask])
            warning(f"Unknown neurotransmitters {unknown_nts} - these synapses will be set to 0")

        synapse_sign_tensor = torch.tensor(synapse_sign, device=self.device, dtype=self.dtype)
        signed_values = sc.values() * synapse_sign_tensor
        self.synapse_counts = torch.sparse_csr_tensor(
            sc.crow_indices(), col, signed_values, size=sc.size(),
        )

        row = self._csr_row_indices(sc)
        self.synapse_counts_coo = np.column_stack([
            col.cpu().numpy(),
            row.cpu().numpy(),
            signed_values.detach().cpu().numpy(),
        ])

    def get_shared_columns(self, neuron_list):
        """Columns where all ``neuron_list`` cell types are present."""
        neuron_indices = np.arange(self.num_neurons)

        column_indices = [self.column_indices[self.cell_types == i] for i in neuron_list]
        column_indices = [i[~np.isnan(i)] for i in column_indices]
        column_indices = reduce(np.intersect1d, column_indices)

        neuron_masks = [(self.cell_types == i) for i in neuron_list]
        neuron_is_in_list = np.any(np.stack(neuron_masks), axis=0)
        neuron_is_in_list = neuron_is_in_list & np.isin(self.column_indices, column_indices)

        selected_neuron_indices = neuron_indices[neuron_is_in_list]
        selected_column_indices = self.column_indices[neuron_is_in_list]
        return selected_neuron_indices, selected_column_indices

    def _silencing_mask(self, neurons_to_silence, invert_silencing=False):
        """Return a boolean mask over cell types for neurons to silence."""
        if not neurons_to_silence:
            print('No neurons selected for silencing')
            return None

        if not invert_silencing:
            print("\n--- WARNING silencing neurons", neurons_to_silence, '---\n')
            chosen = np.zeros(self.num_cell_types, dtype=bool)
            for ns in neurons_to_silence:
                chosen[self.unique_cell_types == ns] = True
        else:
            print("\n--- WARNING silencing everything but", neurons_to_silence, '---\n')
            chosen = np.ones(self.num_cell_types, dtype=bool)
            for ns in neurons_to_silence:
                chosen[self.unique_cell_types == ns] = False

        return chosen

    def _cell_type_name_to_index(self, name):
        """Return index into ``unique_cell_types`` for *name*, or ``None``."""
        matches = np.where(self.unique_cell_types == name)[0]
        if len(matches) == 0:
            warnings.warn(f"cell type '{name}' not found")
            return None
        return int(matches[0])

    def get_column_xyz(self, radius: float = 1.0) -> np.ndarray:
        """Column locations projected to 3D Cartesian coordinates."""
        phi = self.column_phi_theta[:, 0]
        theta = self.column_phi_theta[:, 1]
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.stack((x, y, z), axis=1)

    # ------------------------------------------------------------------
    # Extra-state serialisation for the CSR synapse matrix
    # ------------------------------------------------------------------

    def _collect_extra_state(self):
        state = super()._collect_extra_state()
        state['synapse_counts'] = self.synapse_counts
        return state

    def _restore_extra_state(self, state, device, dtype):
        super()._restore_extra_state(state, device, dtype)
        if 'synapse_counts' not in state:
            return

        def _to_csr(t):
            t = t.to(device=device, dtype=dtype)
            if t.layout == torch.sparse_coo:
                t = t.coalesce().to_sparse_csr()
            return t

        sc = state['synapse_counts']
        if isinstance(sc, torch.Tensor):
            self.synapse_counts = _to_csr(sc)
        elif isinstance(sc, list):
            self.synapse_counts = [_to_csr(t) for t in sc]

        if isinstance(self.synapse_counts, list) and hasattr(self, 'synapse_counts_T'):
            self.synapse_counts_T = [A.t().to_sparse_csr() for A in self.synapse_counts]

    # ------------------------------------------------------------------
    # predict() and steady-state finder
    # ------------------------------------------------------------------

    def predict(self, inputs,
                input_neurons=['L1', 'L2', 'L3'],
                output_neurons=['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'],
                init_state=None, init_filter_state=None):
        num_batches, num_time, num_neurons = inputs.shape

        filter_state = init_filter_state

        if isinstance(input_neurons, list):
            input_neurons = [i for i in input_neurons if i in self.unique_cell_types]
            output_neurons = [i for i in output_neurons if i in self.unique_cell_types]

            input_neuron_indices, input_column_indices = self.get_shared_columns(input_neurons)
            input_column_indices_np = input_column_indices
            stim_to_cell_mapping = {
                'neuron_index': torch.tensor(input_neuron_indices, device=self.device, dtype=torch.int64),
                'stimulus_index': torch.tensor(input_column_indices, device=self.device, dtype=torch.int64),
            }
        else:
            stim_to_cell_mapping = input_neurons
            input_column_indices_np = stim_to_cell_mapping['stimulus_index'].detach().cpu().numpy()

        if isinstance(output_neurons, list):
            readout_dict = {}
            for i in output_neurons:
                is_correct_type = (
                    (self.cell_types == i)
                    & ~np.isnan(self.column_indices)
                    & np.isin(self.column_indices, input_column_indices_np)
                )
                neuron_index = np.arange(self.num_neurons)
                readout_dict[i] = {
                    'column_indices': torch.tensor(self.column_indices[is_correct_type], device=self.device, dtype=torch.int64),
                    'neuron_indices': torch.tensor(neuron_index[is_correct_type], device=self.device, dtype=torch.int64),
                }
            readout_keys = output_neurons
        else:
            readout_dict = output_neurons
            readout_keys = list(readout_dict.keys())

        if init_state is None:
            v = self.steady_state.unsqueeze(0).expand(num_batches, -1)
        else:
            v = init_state

        outputs = {'voltage': {}, 'firing_rate': {}}
        record_synaptic = getattr(self, 'records_synaptic_current', False)
        if record_synaptic:
            outputs['synaptic_current'] = {}
        for i in readout_keys:
            outputs['voltage'][i] = torch.full((num_batches, num_time, self.num_columns), float('nan'), device=self.device, dtype=self.dtype)
            outputs['firing_rate'][i] = torch.full((num_batches, num_time, self.num_columns), float('nan'), device=self.device, dtype=self.dtype)
            if record_synaptic:
                outputs['synaptic_current'][i] = torch.full((num_batches, num_time, self.num_columns), float('nan'), device=self.device, dtype=self.dtype)

        for name, mod in self._readout_modules.items():
            if getattr(mod, 'training_only', False):
                continue
            shape = (num_batches, num_time, self.num_columns) + mod.output_extra_dims
            outputs[name] = torch.full(shape, float('nan'),
                                       device=self.device, dtype=self.dtype)

        for t in range(num_time):
            u_t = inputs[:, t, :]

            if self.use_input_conv:
                u_t, filter_state = self.apply_input_conv(u_t, filter_state)

            v = self(v, u_t, stim_to_cell_mapping)

            if self.use_layer_norm:
                v = self.layer_norm(v)

            fr = self.firing_rate(v)

            for name, mod in self._readout_modules.items():
                if getattr(mod, 'training_only', False):
                    continue
                outputs[name][:, t, ...] = mod(fr)

            if record_synaptic:
                syn_current = self.compute_synaptic_current(v)

            for i in readout_keys:
                ci = readout_dict[i]['column_indices']
                ni = readout_dict[i]['neuron_indices']
                valid = ci >= 0
                if torch.any(valid):
                    outputs['voltage'][i][:, t, ci[valid]] = v[:, ni[valid]]
                    outputs['firing_rate'][i][:, t, ci[valid]] = fr[:, ni[valid]]
                    if record_synaptic:
                        outputs['synaptic_current'][i][:, t, ci[valid]] = syn_current[:, ni[valid]]

        if self.training:
            for name, mod in self._readout_modules.items():
                if getattr(mod, 'training_only', False):
                    outputs[name] = mod(self)

        return {
            'outputs': outputs,
            'model_state': {'v': v.detach(), 'filter_state': filter_state},
            'mappings': {'input_mapping': stim_to_cell_mapping, 'output_mapping': readout_dict},
        }

    def find_steady_state(self, run_time=100, max_iter=25):
        print("Finding steady state ...")
        inputs = torch.zeros(
            (1, run_time, self.num_columns), device=self.device, dtype=self.dtype,
        )
        initial_state = self.steady_state.unsqueeze(0)
        initial_ave = initial_state.abs().mean()

        for i in range(max_iter):
            final_state = self.predict(inputs, init_state=initial_state)['model_state']['v']

            if torch.allclose(final_state, initial_state, atol=1e-3):
                print('steady state found after', i + 1, 'iterations')
                self.steady_state.copy_(final_state.detach().squeeze(0))
                return

            initial_state = final_state.detach().clone()

        final_ave = final_state.abs().mean()
        if final_ave / (initial_ave + 1) > 1e6:
            print('Warning: cannot find steady state, dynamics may be unstable')
            return

        print('Average abs diff:', (final_state - initial_state).abs().mean())
        print('steady state not found after', max_iter, 'iterations')
        self.steady_state.copy_(final_state.detach().squeeze(0))

# ---------------------------------------------------------------------------
# Dynamic class composition: dynamics x ConnectomeModel
# ---------------------------------------------------------------------------


def compose_model_class(dynamics_type: str, model_type: str = 'ConnectomeModel'):
    """Return a class that mixes ``dynamics_type`` onto :class:`ConnectomeModel`.

    ``dynamics_type`` is a class name in :mod:`dynamics_models` (e.g.
    ``'CurrentDynamics'``, ``'ConductanceDynamics'``).

    ``model_type`` is kept as an optional string label so the save format can
    grow more anatomy classes later, but today the only valid value is
    ``'ConnectomeModel'``. Anatomical scope is set by the ``areas``/``side``
    filters at connectome-load time; brain-region-specific operations live on
    :class:`regions.RegionView` objects in ``model.regions``.

    Result is cached so repeated calls with the same pair return the *same*
    class object — important for ``isinstance`` and for save/load identity.
    """
    if model_type != 'ConnectomeModel':
        raise ValueError(
            f"Unknown model_type {model_type!r}. Only 'ConnectomeModel' is "
            f"supported; pick areas via the connectome loader instead."
        )

    key = (dynamics_type, model_type)
    cached = _COMPOSED_CLASS_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        dynamics_cls = getattr(dynamics_models, dynamics_type)
    except AttributeError as e:
        raise ValueError(
            f"Unknown dynamics_type {dynamics_type!r}. Add it to dynamics_models.py "
            f"or fix the config."
        ) from e

    anatomy_cls = ConnectomeModel
    name = f'{model_type}_{dynamics_type}'
    cls = type(name, (dynamics_cls, anatomy_cls), {})
    cls._dynamics_type_name = dynamics_type
    cls._model_type_name = model_type

    _COMPOSED_CLASS_CACHE[key] = cls
    return cls


_COMPOSED_CLASS_CACHE: dict[tuple[str, str], type] = {}


def _baseRnn_save_class_identity(self):
    return {
        'model_type': type(self)._model_type_name,
        'dynamics_type': type(self)._dynamics_type_name,
    }


@classmethod
def _baseRnn_resolve_class_from_payload(cls, payload):
    return compose_model_class(
        payload['dynamics_type'],
        payload.get('model_type', 'ConnectomeModel'),
    )


# Attach to BaseRnn here (defined out-of-class so the factory above can
# reference BaseRnn and the cache without forward-reference gymnastics).
BaseRnn._save_class_identity = _baseRnn_save_class_identity
BaseRnn._resolve_class_from_payload = _baseRnn_resolve_class_from_payload
