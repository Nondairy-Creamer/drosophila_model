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
from model_layers import CellTypeLayerNorm
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
                 use_layer_norm=True, **kwargs):
        super().__init__(**kwargs)

        self.commit_hash = commit_hash
        self.silence_val = 1e-8
        self.init_std = init_std
        self.readouts = list(readouts) if readouts else []
        self.readout_neurons = list(readout_neurons) if readout_neurons else None
        self.regularizer_parameters = dict(regularizer_parameters) if regularizer_parameters else {}
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
        return self.nonlinearity(v)
        """Convert voltage to firing rate."""

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
                 region_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.excitatory_nt = excitatory_nt
        self.inhibitory_nt = inhibitory_nt
        self.count_threshold = count_threshold
        # Per-region option sub-dicts from config['model']['regions']; passed
        # through to each RegionView's _load_extras. Saved into init_inputs by
        # the _save_init_inputs() snapshot below so checkpoints round-trip.
        self.region_kwargs = dict(region_kwargs or {})

        if not isinstance(connectome, ConnectomeData):
            raise TypeError(
                f"`connectome` must be a ConnectomeData (build it via "
                f"connectome_loading.load_connectome(areas=..., side=...)). "
                f"Got {type(connectome).__name__}."
            )
        cdata = connectome
        self.areas = list(cdata.areas)
        self.side = cdata.side
        self.connectome_folder = cdata.connectome_folder

        # Store the resolved ConnectomeData in init_inputs so saved
        # checkpoints are self-contained (no parquet path required to reload).
        self._save_init_inputs()
        self.init_inputs['connectome'] = cdata

        valid_neurotransmitters = self.excitatory_nt + self.inhibitory_nt
        (
            root_ids, cell_types, neurotransmitters,
            area_per_neuron, side_per_neuron, per_neuron_extras,
            synapse_counts_coo,
        ) = self._preprocess_connectome(
            cdata.synapse_counts_coo, cdata.root_ids, cdata.cell_types,
            cdata.neurotransmitters,
            cdata.area_per_neuron, cdata.side_per_neuron,
            cdata.per_neuron_extras,
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

        self.area_per_neuron = area_per_neuron
        self.side_per_neuron = side_per_neuron
        # Universal per-neuron extras forwarded from the bundle. The
        # optic-lobe column index lives here; future areas may add their own.
        self.per_neuron_extras = per_neuron_extras
        self.column_indices_per_neuron = per_neuron_extras.get(
            'column_index',
            np.full(self.num_neurons, np.nan),
        )

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

        # Build per-area RegionView submodules. Each region reads its own
        # area_extras parquet, consumes its own option sub-dict from
        # region_kwargs, and owns its area-specific submodules (e.g.
        # OpticLobeRegion builds the input_conv stack when
        # region_kwargs['optic_lobe']['use_input_conv'] is True).
        self.regions = regions_mod.build_regions(self, self.region_kwargs)

        self._initialize_readouts()

    # ------------------------------------------------------------------
    # Connectome preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_connectome(synapse_counts_coo, root_ids, cell_types,
                               neurotransmitters,
                               area_per_neuron, side_per_neuron,
                               per_neuron_extras,
                               valid_neurotransmitters, count_threshold):
        """Filter the raw connectome and convert root IDs to matrix indices.

        Filtering (in order):
          1. Remove connections below the synapse count threshold.
          2. Remove neurons with invalid neurotransmitters.
          3. Remove connections involving removed neurons.
          4. Remove neurons with no remaining connections.

        All per-neuron arrays (including everything in ``per_neuron_extras``)
        are reduced in lockstep. ``synapse_counts_coo`` is rewritten from
        root-ID space into matrix-index space.
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
        area_per_neuron = area_per_neuron[valid_neurons]
        side_per_neuron = side_per_neuron[valid_neurons]
        per_neuron_extras = {
            k: np.asarray(v)[valid_neurons] for k, v in per_neuron_extras.items()
        }

        root_id_index = pd.Index(root_ids)
        pre_idx = root_id_index.get_indexer(synapse_counts_coo[:, 0])
        post_idx = root_id_index.get_indexer(synapse_counts_coo[:, 1])
        synapse_counts_coo = np.column_stack([pre_idx, post_idx, synapse_counts_coo[:, 2]])

        return (root_ids, cell_types, neurotransmitters,
                area_per_neuron, side_per_neuron, per_neuron_extras,
                synapse_counts_coo)

    # ------------------------------------------------------------------
    # Readouts
    # ------------------------------------------------------------------

    @property
    def has_decoder(self):
        return any(getattr(mod, 'predicts_flow', False)
                   for mod in self._readout_modules.values())

    def _initialize_readouts(self):
        """Instantiate readout modules by class name from ``readout_utilities``.

        Visual readouts (anything that wants ``column_p_q`` / ``num_columns``)
        require the ``optic_lobe`` region. The user is responsible for only
        listing visual readouts when their config includes ``'optic_lobe'`` in
        ``areas``.
        """
        import readout_utilities
        self._readout_modules = torch.nn.ModuleDict()
        if not self.readouts:
            return

        if 'optic_lobe' not in self.regions:
            raise ValueError(
                f"Readouts {self.readouts} were configured but no 'optic_lobe' "
                "region is present. Visual readouts require the optic lobe; "
                "either add 'optic_lobe' to areas or drop the readouts."
            )
        optic = self.regions['optic_lobe']

        kwargs = dict(
            cell_types=self.cell_types,
            column_indices=self.column_indices_per_neuron,
            column_p_q=optic.column_p_q,
            num_neurons=self.num_neurons,
            num_columns=optic.num_columns,
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

        # predict() currently assumes a visual input stream over hex columns;
        # the optic-lobe region owns the column geometry, the input-conv
        # pipeline, and the cell-type->column shared-membership lookup.
        optic = self.regions['optic_lobe']
        num_columns = optic.num_columns
        col_per_neuron = self.column_indices_per_neuron

        if isinstance(input_neurons, list):
            input_neurons = [i for i in input_neurons if i in self.unique_cell_types]
            output_neurons = [i for i in output_neurons if i in self.unique_cell_types]

            input_neuron_indices, input_column_indices = optic.get_shared_columns(input_neurons)
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
                    & ~np.isnan(col_per_neuron)
                    & np.isin(col_per_neuron, input_column_indices_np)
                )
                neuron_index = np.arange(self.num_neurons)
                readout_dict[i] = {
                    'column_indices': torch.tensor(col_per_neuron[is_correct_type], device=self.device, dtype=torch.int64),
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
            outputs['voltage'][i] = torch.full((num_batches, num_time, num_columns), float('nan'), device=self.device, dtype=self.dtype)
            outputs['firing_rate'][i] = torch.full((num_batches, num_time, num_columns), float('nan'), device=self.device, dtype=self.dtype)
            if record_synaptic:
                outputs['synaptic_current'][i] = torch.full((num_batches, num_time, num_columns), float('nan'), device=self.device, dtype=self.dtype)

        for name, mod in self._readout_modules.items():
            if getattr(mod, 'training_only', False):
                continue
            shape = (num_batches, num_time, num_columns) + mod.output_extra_dims
            outputs[name] = torch.full(shape, float('nan'),
                                       device=self.device, dtype=self.dtype)

        for t in range(num_time):
            u_t = inputs[:, t, :]

            if optic.use_input_conv:
                u_t, filter_state = optic.apply_input_conv(u_t, filter_state)

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
            (1, run_time, self.regions['optic_lobe'].num_columns), device=self.device, dtype=self.dtype,
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
