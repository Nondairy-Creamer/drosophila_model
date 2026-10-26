"""Connectome-based recurrent neural models.

The class hierarchy is::

    SerializableModel               (mixin: save/load)
    BaseRnn(SerializableModel, ...) (RNN training/eval scaffolding, no connectome)
    ConnectomeModel(BaseRnn)        (loads a multi-area connectome; owns regions)

Dynamics live in :mod:`dynamics` as mixins. Anatomy-region-specific
operations (e.g. optic-lobe two-eye prediction) live on
:class:`regions.RegionView` subclasses, registered onto ``self.regions`` at
construction based on the requested ``areas``. The model itself stays
anatomy-agnostic — adding a new brain area means adding a new region class,
not a new model subclass.

Concrete models are composed at runtime via :func:`compose_model_class`, which
takes the dynamics-mixin name and returns a class that mixes ``ConnectomeModel``
with that dynamics::

    ModelClass = compose_model_class('CurrentDynamics')
    cdata = connectome_loading.load_connectome(
        {'optic_lobe': {'neuropils': ['ME', 'LO', 'LOP', 'LA'], 'side': 'right'}},
        count_threshold=2,
    )
    model = ModelClass(connectome=cdata, ...)

Config files name the dynamics + the per-region sides::

    [model]
    model_type    = 'ConnectomeModel'   # only valid value today
    dynamics_type = 'CurrentDynamics'

    [model.regions.optic_lobe]
    side = 'right'                       # the optic lobe supports only 'right'

Saved checkpoints carry the dynamics string so ``BaseRnn.load`` can re-compose
the class on load.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import subprocess
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from drosophila_model.model_definition import dynamics
from drosophila_model.config import get_data_folder, load_config, resolve_device
from drosophila_model.model_definition.connectome_loading import ConnectomeData
from drosophila_model.model_definition.model_layers import CellTypeLayerNorm
from drosophila_model.model_definition import regions as regions_mod


# torch warns once per process, on the first sparse COO / CSR tensor it
# builds, that CSR support is beta and that COO invariant checks are off.
# Neither is actionable here, so both are silenced where the model's sparse
# matrices are built or loaded.
_SPARSE_WARNING_MESSAGES = (
    'Sparse CSR tensor support is in beta state',
    'Sparse invariant checks are implicitly disabled',
)


@contextlib.contextmanager
def _quiet_sparse_warnings():
    with warnings.catch_warnings():
        for message in _SPARSE_WARNING_MESSAGES:
            warnings.filterwarnings('ignore', message=message, category=UserWarning)
        yield


# ---------------------------------------------------------------------------
# SerializableModel: save/load mixin (no class registry)
# ---------------------------------------------------------------------------


class SerializableModel:
    """Save / load / init-input tracking for any ``torch.nn.Module`` subclass.

    Defines the checkpoint format used by :class:`BaseRnn`. Class identity on
    load is resolved via the ``_resolve_class_from_payload`` hook: the default
    returns the calling ``cls`` (``cls.load(path)`` instantiates ``cls``);
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

        Default: the calling class (``cls.load(...)`` instantiates ``cls``).
        :class:`BaseRnn` overrides this to compose dynamics + anatomy from the
        saved ``dynamics_type`` and ``model_type`` strings.
        """
        return cls

    @staticmethod
    def _accepted_init_input_names(model_cls):
        """Constructor argument names accepted across ``model_cls``'s MRO.

        Used to strip saved/forwarded ``init_inputs`` down to keys some
        ``__init__`` actually accepts (``_save_init_inputs`` snapshots every
        local of the ``__init__`` frame, not only its arguments).
        """
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
        return accepted

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Serialize the connectome as plain data (dict of arrays) rather than
        # the ConnectomeData object, so a saved model never couples to that
        # class's import path. Rebuilt in load() via from_serializable_dict.
        inputs = dict(self.init_inputs)
        connectome = inputs.get('connectome', None)
        if isinstance(connectome, ConnectomeData):
            inputs['connectome'] = connectome.to_serializable_dict()

        payload = {
            'inputs': inputs,
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
        """Load a model saved by :meth:`save` (``path`` may be its folder).

        ``device`` defaults to ``[model].device`` of the ``config.toml`` in the
        working directory (CPU when CUDA is unavailable). The connectome's
        parquets are read from that config's ``[data].data_folder`` (or the
        saved folder when config.toml doesn't set one). ``freeze_params``
        disables gradients and puts the model in eval mode.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "model.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"No saved model at {path.resolve()}. "
                "Pass a trained model's folder or its model.pt."
            )
        with _quiet_sparse_warnings():
            data = torch.load(path, map_location='cpu', weights_only=False)
        inputs = dict(data['inputs'])

        # The device comes from the argument, else this machine's config.toml
        # (falling back to CPU when CUDA is unavailable); config.toml also
        # says where the data bundle lives. An explicit device makes the
        # config's device irrelevant, so its CUDA fallback warning is dropped.
        with warnings.catch_warnings():
            if device is not None:
                warnings.filterwarnings(
                    'ignore', message=r"Config device '.*' requested but CUDA is not available",
                    category=UserWarning)
            local_config = load_config()
        inputs['device'] = resolve_device(device if device is not None
                                          else local_config['model']['device'])

        if dtype is not None:
            inputs['dtype'] = dtype

        # Rebuild the ConnectomeData from its plain-data dict (see save()).
        connectome = inputs.get('connectome', None)
        if isinstance(connectome, dict):
            connectome = ConnectomeData.from_serializable_dict(connectome)
            inputs['connectome'] = connectome

        # The saved connectome carries the data_folder of the training
        # machine. Point it at this machine's data bundle when config.toml
        # sets one (so region parquets resolve); keep the saved folder only
        # when config.toml leaves it unset and the saved folder still exists
        # (an empty saved folder, e.g. from load_connectome_from_dataframes'
        # default, counts as unset). get_data_folder raises a clear error
        # otherwise.
        if connectome is not None and (
            local_config['data'].get('data_folder')
            or not connectome.data_folder
            or not Path(connectome.data_folder).is_dir()
        ):
            connectome.data_folder = str(get_data_folder(local_config))

        model_cls = cls._resolve_class_from_payload(data)
        inputs['init_new_model'] = False

        # Strip saved keys that no constructor in the MRO accepts.
        accepted = cls._accepted_init_input_names(model_cls)
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

        if model_commit and current_commit and model_commit != current_commit:
            warnings.warn(
                f"{path} was trained on git commit {model_commit[:8]}; the "
                f"local code is at {current_commit[:8]}.",
                stacklevel=2,
            )

        return model


# ---------------------------------------------------------------------------
# BaseRnn: connectome-agnostic RNN training / eval scaffolding
# ---------------------------------------------------------------------------


class BaseRnn(SerializableModel, torch.nn.Module, ABC):
    """Connectome-agnostic RNN base.

    Owns the nonlinearity, the seeded ``rng``, ``device`` / ``dtype`` / ``dt``,
    ``add_noise_to_parameters``, ``set_sigma_max`` and save/load (with the
    class-identity hooks that re-compose the dynamics class on load).
    ``predict``, ``find_steady_state`` and all connectome-specific state
    (synapse matrix, cell types, regions) live on :class:`ConnectomeModel`.

    The ``nonlinearity`` / ``use_layer_norm`` defaults here apply only to
    direct construction; config-driven builds (``run_model``) pass
    ``[model].nonlinearity`` and ``[model].use_layer_norm`` explicitly.

    Concrete subclasses (via the dynamics mixins) supply:
        - ``forward(v_prev, u_t, stim_to_cell_mapping=None) -> v_next``
        - ``firing_rate(v) -> fr``
        - ``calc_sigma_max() -> sigma``
        - ``_apply_sigma_scale(scale_factor)``  (mutates whichever parameter
          controls overall recurrent gain)
    """

    # Default spectral-norm target for the linearised dynamics matrix.
    max_allowed_sigma = 0.95

    def __init__(self, *,
                 nonlinearity='Softplus', device='cpu', dtype='float32',
                 dt=0.01, seed=None, init_std=0.1, commit_hash=None,
                 use_layer_norm=True, **kwargs):
        super().__init__(**kwargs)

        self.commit_hash = commit_hash
        self.silence_val = 1e-8
        self.init_std = init_std
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

            for _ in range(max_scaling_iters):
                sigma_max = self.calc_sigma_max()
                if abs(sigma_max - self.max_allowed_sigma) < rho_tolerance:
                    break
                scale_factor = (self.max_allowed_sigma / sigma_max) ** damping_factor
                self._apply_sigma_scale(scale_factor)

    # ------------------------------------------------------------------
    # to(): move plain (non-Parameter, non-Buffer) tensor attributes too
    # ------------------------------------------------------------------

    # Plain attributes (a tensor or a list of tensors) that nn.Module.to()
    # doesn't see: the sparse synapse matrices kept by ConnectomeModel and
    # the dynamics mixins, and the integer neuron -> cell-type index.
    _plain_tensor_attrs = (
        'synapse_counts', 'synapse_counts_T',
        'synapse_counts_pos', 'synapse_counts_neg',
        'neuron_to_type_index',
    )

    def to(self, *args, **kwargs):
        """``nn.Module.to`` that also moves :attr:`_plain_tensor_attrs`.

        As in ``nn.Module.to``, a dtype only applies to floating-point
        tensors (integer indices just change device). ``self.device`` and
        ``self.dtype`` are kept in sync.
        """
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        result = super().to(*args, **kwargs)

        def _move(t):
            if dtype is not None and t.is_floating_point():
                return t.to(device=device, dtype=dtype, non_blocking=non_blocking)
            return t.to(device=device, non_blocking=non_blocking)

        for name in self._plain_tensor_attrs:
            value = getattr(result, name, None)
            if isinstance(value, torch.Tensor):
                setattr(result, name, _move(value))
            elif isinstance(value, list) and all(isinstance(t, torch.Tensor) for t in value):
                setattr(result, name, [_move(t) for t in value])

        if device is not None:
            result.device = device
        if dtype is not None:
            result.dtype = dtype
        return result

    # ------------------------------------------------------------------
    # Save/load class identity: re-compose dynamics x anatomy on load
    # ------------------------------------------------------------------

    def _save_class_identity(self):
        return {
            'model_type': type(self)._model_type_name,
            'dynamics_type': type(self)._dynamics_type_name,
        }

    @classmethod
    def _resolve_class_from_payload(cls, payload):
        return compose_model_class(
            payload['dynamics_type'],
            payload.get('model_type', 'ConnectomeModel'),
        )


# ---------------------------------------------------------------------------
# ConnectomeModel: loads a multi-area connectome and runs the predict() loop
# ---------------------------------------------------------------------------


class ConnectomeModel(BaseRnn):
    """``BaseRnn`` augmented with a multi-area connectome.

    Takes a pre-loaded :class:`ConnectomeData` (built via
    :func:`connectome_loading.load_connectome` with the desired
    ``{area: side}`` filter), preprocesses it (synapse-count threshold +
    neurotransmitter filter), then registers one
    :class:`regions.RegionView` per area in ``self.regions``. Each region
    owns its own side (``'right'`` / ``'left'`` / ``'both'``; the optic lobe
    supports only ``'right'``). Area-specific
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
                f"connectome_loading.load_connectome({{region: {{...}}}})). "
                f"Got {type(connectome).__name__}."
            )
        self.region_specs = {
            name: dict(spec) for name, spec in connectome.region_specs.items()
        }
        self.regions_loaded = list(connectome.region_specs.keys())
        self.data_folder = connectome.data_folder

        # Store the resolved ConnectomeData in init_inputs so saved
        # checkpoints are self-contained (no parquet path required to reload).
        self._save_init_inputs()
        self.init_inputs['connectome'] = connectome

        valid_neurotransmitters = self.excitatory_nt + self.inhibitory_nt
        # Per-region cell-type exclusions are owned by the region classes;
        # the model just collects them and applies them in preprocessing.
        neurons_to_exclude_per_region = regions_mod.resolve_neurons_to_exclude(
            self.regions_loaded, self.region_kwargs,
        )
        (
            root_ids, cell_types, neurotransmitters,
            region_membership, synapse_counts_coo,
        ) = self._preprocess_connectome(
            connectome.synapse_counts_coo, connectome.root_ids, connectome.cell_types,
            connectome.neurotransmitters,
            connectome.region_membership,
            valid_neurotransmitters, count_threshold,
            neurons_to_exclude_per_region,
        )

        self.root_ids = root_ids
        self.cell_types = cell_types
        self.num_neurons = cell_types.shape[0]
        self.unique_cell_types = np.unique(cell_types)
        self.num_cell_types = self.unique_cell_types.shape[0]

        self.transmitters = neurotransmitters
        self.unique_transmitters = np.unique(self.transmitters)
        self.num_transmitters = len(self.unique_transmitters)

        # Per-region union-membership masks aligned with the model's flat
        # neuron axis. Same neuron can land in multiple regions; each
        # RegionView builds its own ``cell_indices`` from its mask.
        self.region_membership = region_membership

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
        with _quiet_sparse_warnings():
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

        # Build per-region RegionView submodules. Each region reads its own
        # per-region parquet, consumes its own option sub-dict from
        # region_kwargs, and owns its area-specific submodules (e.g.
        # OpticLobeRegion builds the input_conv stack when
        # region_kwargs['optic_lobe']['use_input_conv'] is True). Each
        # region also instantiates its own readouts from
        # region_kwargs[name]['readouts'].
        self.regions = regions_mod.build_regions(
            self, self.region_membership, self.region_specs, self.region_kwargs,
        )

    # ------------------------------------------------------------------
    # Connectome preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_connectome(synapse_counts_coo, root_ids, cell_types,
                               neurotransmitters,
                               region_membership,
                               valid_neurotransmitters, count_threshold,
                               neurons_to_exclude_per_region):
        """Filter the raw connectome and convert root IDs to matrix indices.

        Filtering (in order):
          1. Remove ``synapse_counts_coo`` rows below the synapse count
             threshold. :class:`ConnectomeData` has one row per neuron pair
             (summed over neuropils), so the threshold applies per pair.
          2. Remove neurons with invalid neurotransmitters.
          3. Remove per-region cell-type exclusions (e.g. photoreceptor
             terminals from the optic lobe; see
             :func:`regions.resolve_neurons_to_exclude`). A cell type
             excluded by region R is dropped wherever it appears IN R's
             union membership (so a photoreceptor that only lives in the
             optic lobe is dropped globally; a type that also exists in
             another region keeps its non-R copies).
          4. Remove connections involving removed neurons.
          5. Remove neurons with no remaining connections.

        All per-neuron arrays (and the per-region membership masks) are
        reduced in lockstep. ``synapse_counts_coo`` is rewritten from
        root-ID space into matrix-index space.
        """
        import pandas as pd

        neurotransmitters = np.array(neurotransmitters)
        cell_types = np.array(cell_types)
        region_membership = {
            name: np.asarray(mask, dtype=bool)
            for name, mask in region_membership.items()
        }
        assert root_ids.dtype == np.int64, 'root_ids must have dtype np.int64'
        assert synapse_counts_coo.dtype == np.int64, \
            'synapse_counts_coo must have dtype np.int64'

        above_threshold = np.abs(synapse_counts_coo[:, 2]) >= count_threshold
        synapse_counts_coo = synapse_counts_coo[above_threshold]

        valid_nt_mask = np.isin(neurotransmitters, valid_neurotransmitters)
        for name, excluded_types in (neurons_to_exclude_per_region or {}).items():
            if not excluded_types:
                continue
            in_region = region_membership.get(name)
            if in_region is None:
                continue
            valid_nt_mask &= ~(
                in_region & np.isin(cell_types, list(excluded_types))
            )
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
        region_membership = {
            name: mask[valid_neurons] for name, mask in region_membership.items()
        }

        root_id_index = pd.Index(root_ids)
        pre_idx = root_id_index.get_indexer(synapse_counts_coo[:, 0])
        post_idx = root_id_index.get_indexer(synapse_counts_coo[:, 1])
        synapse_counts_coo = np.column_stack([pre_idx, post_idx, synapse_counts_coo[:, 2]])

        return (root_ids, cell_types, neurotransmitters,
                region_membership, synapse_counts_coo)

    # ------------------------------------------------------------------
    # Readouts (region-owned; aggregated here for convenient lookup)
    # ------------------------------------------------------------------

    @property
    def _readout_modules(self):
        """Flat view of region-owned readout modules.

        Each region owns its own ``_readout_modules`` ModuleDict; this
        property aggregates them for callers that just want to find or
        iterate readouts by name. Names are expected to be unique across
        regions; later regions win on conflict (today only optic_lobe
        publishes readouts so there's no conflict).
        """
        merged = {}
        for region in self.regions.values():
            for name, mod in region._readout_modules.items():
                merged[name] = mod
        return merged

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

        Excitatory NTs -> positive, inhibitory NTs -> negative. Every
        presynaptic NT is one of the two (``_preprocess_connectome`` drops
        neurons with any other NT). Modifies both ``synapse_counts`` (CSR)
        and ``synapse_counts_coo`` (numpy).
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

    def _sign_override_factors(self, pre_neurons, post_neurons):
        """Per-edge factor from every region's ``sign_overrides``.

        ``pre_neurons`` / ``post_neurons`` are matrix indices of the edges.
        Each region declares anatomy-prior corrections as
        ``(pre_cell_type, post_cell_type, factor)``: connection types whose
        functional sign disagrees with the bulk-NT classification (e.g. the
        optic lobe's GABAergic C2 / C3 contacts onto L1 / L2 are excitatory).
        ``factor=-1`` inverts the connection's sign, ``factor=0`` ablates it,
        other magnitudes scale it. Returns a float array (1 where no override
        applies). Every dynamics applies these factors in its own terms: the
        signed-matrix dynamics multiply the edge weight
        (:meth:`_apply_region_sign_overrides`), ``ConductanceDynamics`` moves a
        sign-inverted edge to an opposite-sign transmitter's conductance.
        Cell types missing from the loaded connectome are skipped with a
        warning.
        """
        pre_neurons = np.asarray(pre_neurons, dtype=np.int64)
        post_neurons = np.asarray(post_neurons, dtype=np.int64)
        type_index = self.neuron_to_type_index.cpu().numpy()
        pre_types = type_index[pre_neurons]
        post_types = type_index[post_neurons]
        factors = np.ones(len(pre_neurons), dtype=np.float64)
        for region in self.regions.values():
            for pre, post, factor in region.sign_overrides:
                pre_ti = self._cell_type_name_to_index(pre)
                post_ti = self._cell_type_name_to_index(post)
                if pre_ti is None or post_ti is None:
                    continue
                factors[(pre_types == pre_ti) & (post_types == post_ti)] *= factor
        return factors

    def _apply_region_sign_overrides(self):
        """Multiply the signed ``synapse_counts`` by the region sign overrides.

        Used by the signed-matrix dynamics (Current, CurrentDivisive) after
        :meth:`_sign_synapses`; see :meth:`_sign_override_factors`. Mutates
        both ``synapse_counts`` (CSR) and ``synapse_counts_coo`` (numpy).
        """
        sc = self.synapse_counts
        pre_n = sc.col_indices()
        post_n = self._csr_row_indices(sc)
        factors = self._sign_override_factors(pre_n.cpu().numpy(), post_n.cpu().numpy())
        values = sc.values() * torch.as_tensor(
            factors, device=sc.values().device, dtype=sc.values().dtype)

        self.synapse_counts = torch.sparse_csr_tensor(
            sc.crow_indices(), pre_n, values, size=sc.size(),
        )
        self.synapse_counts_coo = np.column_stack([
            pre_n.cpu().numpy(),
            post_n.cpu().numpy(),
            values.detach().cpu().numpy(),
        ])

    def _initialize_synapse_signs(self):
        """Sign synapses by NT, then apply region-level anatomy overrides.

        Single entry point shared by the signed-matrix dynamics ``__init__``
        methods (``ConductanceDynamics`` applies the overrides through its
        per-transmitter matrices instead).
        """
        self._sign_synapses()
        self._apply_region_sign_overrides()

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
    # Structural trimming: fully remove neurons (vs. silence_neurons)
    # ------------------------------------------------------------------

    def remove_cell_types(self, cell_types_to_remove):
        """Return a new, smaller model with those cell types' neurons deleted.

        Unlike :meth:`silence_neurons` (which only zeroes a cell type's output
        strength but leaves it in the network), this rebuilds the model from a
        connectome filtered to drop those neurons and their edges — so the
        connectome-derived structure (``synapse_counts``,
        ``neuron_to_type_index``, region ``cell_indices``) is regenerated by the
        normal construction path — then transfers this model's trained
        parameters onto it via :meth:`_transfer_parameters_to`. ``steady_state``
        (the initial state predictions and training start from) is copied over
        for the retained neurons, matched by root id, so the trimmed model
        starts from the same state as this one.

        Relies on every dynamics (Current, CurrentDivisive, Conductance)
        keeping its state in parameters indexed by cell type, by
        neurotransmitter, or by fixed readout geometry.
        """
        connectome = self.init_inputs.get('connectome')
        if connectome is None:
            raise RuntimeError(
                "model has no stored connectome to rebuild from; cannot remove "
                "cell types structurally."
            )

        remove = set(cell_types_to_remove) & set(self.unique_cell_types.tolist())
        if not remove:
            warnings.warn("remove_cell_types: no matching cell types to remove")

        filtered = connectome.drop_cell_types(remove)

        model_cls = type(self)
        inputs = dict(self.init_inputs)
        inputs['connectome'] = filtered
        inputs['init_new_model'] = False
        inputs['device'] = self.device
        accepted = self._accepted_init_input_names(model_cls)
        inputs = {k: v for k, v in inputs.items() if k in accepted}

        new_model = model_cls(**inputs)
        self._transfer_parameters_to(new_model)

        # Keep this model's initial state for the retained neurons.
        old_pos = {int(rid): i for i, rid in enumerate(self.root_ids)}
        pos = torch.tensor(
            [old_pos[int(rid)] for rid in new_model.root_ids],
            dtype=torch.int64, device=self.steady_state.device,
        )
        with torch.no_grad():
            new_model.steady_state.copy_(self.steady_state[pos])

        new_model.eval()
        for param in new_model.parameters():
            param.requires_grad_(False)
        return new_model

    def _transfer_dynamics_parameters(self, new_model):
        """Hook for dynamics whose parameters are not indexed by cell type.

        Copy such parameters into the trimmed ``new_model`` and return their
        names (e.g. ``CellAnnDynamics``'s per-type-pair weights). Default:
        nothing to do.
        """
        return set()

    @contextlib.contextmanager
    def silenced(self, neurons_to_silence, invert_silencing=False, silence_inputs=False):
        """Context manager: :meth:`silence_neurons`, restoring the model on exit.

        Works for every dynamics: the whole ``state_dict`` is snapshotted and
        reloaded, whatever parameters the dynamics' silencing touches.
        """
        saved = {k: v.detach().clone() for k, v in self.state_dict().items()}
        self.silence_neurons(neurons_to_silence, invert_silencing=invert_silencing,
                             silence_inputs=silence_inputs)
        try:
            yield self
        finally:
            self.load_state_dict(saved)

    def _transfer_parameters_to(self, new_model):
        """Copy this model's trained parameters into a trimmed ``new_model``.

        Parameters whose shape is unchanged (readout / geometry weights) are
        copied directly. Parameters with a shrunk cell-type or neurotransmitter
        dimension are sub-indexed by matching ``new_model``'s retained cell-type
        names / transmitters back to this model's ordering.
        """
        old_ct = {name: i for i, name in enumerate(self.unique_cell_types)}
        ct_index = torch.tensor(
            [old_ct[name] for name in new_model.unique_cell_types],
            dtype=torch.int64, device=self.device,
        )
        old_tx = {name: i for i, name in enumerate(self.unique_transmitters)}
        tx_index = torch.tensor(
            [old_tx[name] for name in new_model.unique_transmitters],
            dtype=torch.int64, device=self.device,
        )
        # Map a (shrunk) new dimension size -> (expected old size, gather index).
        dim_maps = {
            new_model.num_cell_types: (self.num_cell_types, ct_index),
            new_model.num_transmitters: (self.num_transmitters, tx_index),
        }

        handled = self._transfer_dynamics_parameters(new_model)
        old_params = dict(self.named_parameters())
        with torch.no_grad():
            for name, new_p in new_model.named_parameters():
                if name in handled:
                    continue
                if name not in old_params:
                    raise KeyError(
                        f"parameter {name!r} present in trimmed model but not "
                        f"in source model"
                    )
                old_p = old_params[name].detach().to(new_p.device)
                if old_p.shape == new_p.shape:
                    new_p.copy_(old_p)
                    continue

                result = old_p
                for d in range(old_p.dim()):
                    if old_p.shape[d] == new_p.shape[d]:
                        continue
                    target = new_p.shape[d]
                    mapping = dim_maps.get(target)
                    if mapping is None or mapping[0] != old_p.shape[d]:
                        raise ValueError(
                            f"cannot remap dim {d} of parameter {name!r} "
                            f"({tuple(old_p.shape)} -> {tuple(new_p.shape)}); "
                            f"only cell-type / transmitter dimensions are "
                            f"supported."
                        )
                    result = result.index_select(d, mapping[1].to(result.device))

                if result.shape != new_p.shape:
                    raise ValueError(
                        f"shape mismatch after remapping {name!r}: "
                        f"{tuple(result.shape)} vs {tuple(new_p.shape)}"
                    )
                new_p.copy_(result)

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
                input_neurons=None, output_neurons=None,
                init_state=None, init_filter_state=None):
        """Step the network forward over ``inputs``.

        Anatomy-agnostic: I/O mapping, per-timestep input pre-filtering,
        recording structure, and readouts are all delegated to
        :class:`RegionView` subclasses. Outputs and mappings are nested
        per-region (e.g. ``result['outputs']['optic_lobe']['firing_rate']``).

        ``input_neurons`` / ``output_neurons`` are lists of cell-type
        names; passing ``None`` collects each region's
        ``default_input_neurons`` / ``default_output_neurons``. Either may
        also be a pre-built mapping dict (round-trip from a previous
        ``predict`` call via the per-region mappings).
        """
        num_batches, num_time, _ = inputs.shape
        filter_state = init_filter_state

        stim_to_cell_mapping, input_meta_per_region = self._resolve_input_mapping(
            input_neurons,
        )
        output_per_region = self._resolve_output_neurons(output_neurons)

        region_outputs: dict = {}
        region_mappings: dict = {}
        for name, region in self.regions.items():
            spec = output_per_region.get(name)
            if spec is None:
                continue
            meta = input_meta_per_region.get(name, {})
            if isinstance(spec, dict):
                # Round-trip: spec is a precomputed readout_dict.
                result = region.initialize_recording(
                    num_batches, num_time, [], meta,
                    precomputed_output_mapping=spec,
                )
            else:
                # spec is a cell-type-name list.
                if not spec:
                    continue
                result = region.initialize_recording(
                    num_batches, num_time, spec, meta,
                )
            if result is None:
                continue
            outs, mapping = result
            mapping['input_mapping'] = (
                stim_to_cell_mapping if name in input_meta_per_region else None
            )
            region_outputs[name] = outs
            region_mappings[name] = mapping

        if init_state is None:
            v = self.steady_state.unsqueeze(0).expand(num_batches, -1)
        else:
            v = init_state

        for t in range(num_time):
            u_t = inputs[:, t, :]
            u_t, filter_state = self._apply_input_pre_filters(u_t, filter_state)

            v = self(v, u_t, stim_to_cell_mapping)
            if self.use_layer_norm:
                v = self.layer_norm(v)
            fr = self.firing_rate(v)

            for name in region_outputs:
                self.regions[name].record_step(
                    t, v, fr,
                    region_outputs[name], region_mappings[name],
                )

        if self.training:
            for name, region in self.regions.items():
                if name not in region_outputs:
                    continue
                for rname, mod in region._readout_modules.items():
                    if getattr(mod, 'training_only', False):
                        region_outputs[name][rname] = mod(self)

        return {
            'outputs': region_outputs,
            'model_state': {'v': v.detach(), 'filter_state': filter_state},
            'mappings': region_mappings,
        }

    # ------------------------------------------------------------------
    # predict() helpers: region dispatch
    # ------------------------------------------------------------------

    def _resolve_input_mapping(self, input_neurons):
        """Build the stim->cell mapping plus per-region ``input_meta``.

        ``input_neurons`` may be:
          - ``None``: each region uses its own ``default_input_neurons``.
          - list of cell-type names: filtered to types present in the model
            (a ``ValueError`` if none are), then handed to the first region
            that contains any of them and whose ``build_input_mapping``
            claims them. Types outside that region are dropped with a
            warning.
          - pre-built ``stim_to_cell_mapping`` dict: passed through
            unchanged; ``input_meta`` is reconstructed enough to drive
            ``initialize_recording``.

        Returns ``(stim_to_cell_mapping, input_meta_per_region)`` where
        ``input_meta_per_region`` is keyed by region name. Today only one
        region drives input (optic_lobe) so the mapping is single-valued,
        but multiple-input-region models would aggregate here.
        """
        if isinstance(input_neurons, dict):
            stimulus_columns_np = (
                input_neurons['stimulus_index'].detach().cpu().numpy()
            )
            # Round-trip path: today only the optic-lobe builds an input
            # mapping, so attribute the meta to it.
            owner = next(iter(self.regions))
            return input_neurons, {owner: {'stimulus_columns_np': stimulus_columns_np}}

        if input_neurons is None:
            # Each region contributes its own defaults; if any region
            # provides defaults we build a single combined mapping.
            for name, region in self.regions.items():
                cell_types = [
                    n for n in region.default_input_neurons
                    if n in self.unique_cell_types
                ]
                if not cell_types:
                    continue
                result = region.build_input_mapping(cell_types)
                if result is not None:
                    mapping, meta = result
                    return mapping, {name: meta}
            raise ValueError(
                "No region provided default input_neurons. Pass "
                "input_neurons explicitly or add defaults to a region."
            )

        filtered = [n for n in input_neurons if n in self.unique_cell_types]
        if not filtered:
            raise ValueError(
                f"None of input_neurons={list(input_neurons)!r} are in the "
                f"model (cell types can be dropped by a region's "
                f"neurons_to_exclude, an unsupported neurotransmitter, the "
                f"synapse-count threshold or use_connectome_subset)."
            )
        for name, region in self.regions.items():
            region_types = set(self.cell_types[region.cell_indices.cpu().numpy()])
            claimed = [n for n in filtered if n in region_types]
            if not claimed:
                continue
            result = region.build_input_mapping(claimed)
            if result is not None:
                dropped = [n for n in input_neurons if n not in claimed]
                if dropped:
                    warnings.warn(
                        f"input_neurons {dropped!r} are not in region "
                        f"{name!r} (or not in the model) and get no input."
                    )
                mapping, meta = result
                return mapping, {name: meta}

        raise ValueError(
            f"No region could build an input mapping for input_neurons="
            f"{filtered!r} (e.g. optic-lobe input types that share no column)."
        )

    def _resolve_output_neurons(self, output_neurons):
        """Resolve ``output_neurons`` into a per-region spec.

        Returns ``{region_name: spec}`` where each ``spec`` is either:
          - a list of cell-type names (``initialize_recording`` will build
            the readout dict from scratch), or
          - a pre-built ``readout_dict`` (round-trip from a prior predict;
            ``initialize_recording`` reuses it via
            ``precomputed_output_mapping``).

        Input forms:
          - ``None``: each region uses its own ``default_output_neurons``.
          - list of cell-type names: each entry routed to the first region
            whose cell-type set includes it (today: all to optic_lobe).
          - ``{region_name: readout_dict}``: per-region passthrough.
        """
        if isinstance(output_neurons, dict):
            unknown = [k for k in output_neurons if k not in self.regions]
            if unknown:
                raise ValueError(
                    f"output_neurons mapping keys must be region names "
                    f"{list(self.regions)}; got {unknown}."
                )
            return dict(output_neurons)

        if output_neurons is None:
            return {
                name: [
                    n for n in region.default_output_neurons
                    if n in self.unique_cell_types
                ]
                for name, region in self.regions.items()
            }

        filtered = [n for n in output_neurons if n in self.unique_cell_types]
        per_region: dict[str, list[str]] = {name: [] for name in self.regions}
        for ct in filtered:
            for name, region in self.regions.items():
                cell_idx_np = region.cell_indices.cpu().numpy()
                if ct in self.cell_types[cell_idx_np]:
                    per_region[name].append(ct)
                    break
        return per_region

    def _apply_input_pre_filters(self, u_t, filter_state):
        """Compose per-region input pre-filters in region-iteration order."""
        for region in self.regions.values():
            u_t, filter_state = region.apply_input_pre_filter(u_t, filter_state)
        return u_t, filter_state

    def find_steady_state(self, run_time=100, max_iter=25):
        """Run the model on zero input until its state stops changing.

        Stores the final state in ``steady_state`` and returns ``True``,
        unless the dynamics blow up (non-finite or exploding state): then
        ``steady_state`` is left unchanged and ``False`` is returned.
        """
        print("Finding steady state ...")
        # Region-agnostic: each region reports the size of its slice of
        # the input axis. Summed gives the total stimulus width.
        stim_size = sum(region.input_stim_size for region in self.regions.values())
        inputs = torch.zeros(
            (1, run_time, stim_size), device=self.device, dtype=self.dtype,
        )
        initial_state = self.steady_state.unsqueeze(0)
        initial_ave = initial_state.abs().mean()

        for i in range(max_iter):
            final_state = self.predict(inputs, init_state=initial_state)['model_state']['v']

            if torch.allclose(final_state, initial_state, atol=1e-3):
                print('steady state found after', i + 1, 'iterations')
                self.steady_state.copy_(final_state.detach().squeeze(0))
                return True

            last_diff = (final_state - initial_state).abs().mean()
            initial_state = final_state.detach().clone()

        # NaN fails the ratio test below, so check finiteness explicitly
        # rather than storing a NaN steady state.
        final_ave = final_state.abs().mean()
        if not torch.isfinite(final_state).all() or final_ave / (initial_ave + 1) > 1e6:
            print('Warning: cannot find steady state, dynamics may be unstable')
            return False

        print('Average abs diff:', last_diff)
        print('steady state not found after', max_iter, 'iterations')
        self.steady_state.copy_(final_state.detach().squeeze(0))
        return True

# ---------------------------------------------------------------------------
# Dynamic class composition: dynamics x ConnectomeModel
# ---------------------------------------------------------------------------


def dynamics_kwargs(model_config: dict) -> dict:
    """Constructor kwargs specific to ``model_config['dynamics_type']``.

    Only ``CellAnnDynamics`` takes one (``[model].cell_ann_depth``).
    """
    if model_config['dynamics_type'] == 'CellAnnDynamics':
        return {'cell_ann_depth': model_config['cell_ann_depth']}
    return {}


def compose_model_class(dynamics_type: str, model_type: str = 'ConnectomeModel'):
    """Return a class that mixes ``dynamics_type`` onto :class:`ConnectomeModel`.

    ``dynamics_type`` is a class name in :mod:`dynamics` (e.g.
    ``'CurrentDynamics'``, ``'ConductanceDynamics'``).

    ``model_type`` is kept as an optional string label so the save format can
    grow more anatomy classes later, but today the only valid value is
    ``'ConnectomeModel'``. Anatomical scope is set by the ``{area: side}``
    filter at connectome-load time; brain-region-specific operations live on
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

    # The valid names are the *Dynamics classes defined in dynamics.py.
    available = sorted(
        name for name, obj in vars(dynamics).items()
        if isinstance(obj, type) and obj.__module__ == dynamics.__name__
        and name.endswith('Dynamics')
    )
    if dynamics_type not in available:
        raise ValueError(
            f"Unknown dynamics_type {dynamics_type!r}; available: "
            f"{', '.join(available)}. Add it to dynamics.py or fix the config."
        )
    dynamics_cls = getattr(dynamics, dynamics_type)

    anatomy_cls = ConnectomeModel
    name = f'{model_type}_{dynamics_type}'
    cls = type(name, (dynamics_cls, anatomy_cls), {})
    cls._dynamics_type_name = dynamics_type
    cls._model_type_name = model_type

    _COMPOSED_CLASS_CACHE[key] = cls
    return cls


_COMPOSED_CLASS_CACHE: dict[tuple[str, str], type] = {}
