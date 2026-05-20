"""Connectome-based recurrent neural models.

The class hierarchy splits orthogonally along *anatomy* (which connectome) and
*dynamics* (Current / Conductance / CellAnn / HandTuned / Arbitrary):

    SerializableModel               (mixin: save/load)
    BaseRnn(SerializableModel, ...) (RNN training/eval scaffolding, no connectome)
    ConnectomeRnn(BaseRnn)          (loads a connectome; whole-brain ego-motion)
    OpticLobeRightRnn(ConnectomeRnn)(single-eye extras: two-eye predict + flip)

Dynamics live in :mod:`dynamics_models` as mixins. Concrete models are composed
at runtime via :func:`compose_model_class`, which takes the dynamics-mixin name
and the anatomy-class name and returns a new class that mixes them together::

    ModelClass = compose_model_class('CurrentDynamics', 'OpticLobeRightRnn')
    model = ModelClass(connectome='optic_lobe_right', ...)

Config files name the two axes separately::

    [model]
    model_type    = 'OpticLobeRightRnn'   # anatomy
    dynamics_type = 'CurrentDynamics'     # dynamics

Saved checkpoints carry both strings so ``BaseRnn.load`` can re-compose the
class on load.
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
# Shared submodules: CellTypeLayerNorm, NeuralModelConv, TemporalLRU
# ---------------------------------------------------------------------------


class CellTypeLayerNorm(torch.nn.Module):
    """LayerNorm with weight and bias shared per cell type instead of per neuron.

    Stores one weight (in log space, so always positive) and one bias per cell
    type, and expands them to per-neuron via a neuron-to-type index mapping
    before applying.
    """

    def __init__(self, num_cell_types, neuron_to_type_index, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.num_cell_types = num_cell_types
        self.eps = eps
        self.register_buffer('neuron_to_type_index', neuron_to_type_index)
        init_weight = 1.0
        init_weights = np.full(num_cell_types, np.log(init_weight))
        self.weight_log = torch.nn.Parameter(
            torch.tensor(init_weights, device=device, dtype=dtype)
        )
        self.bias = torch.nn.Parameter(torch.zeros(num_cell_types, device=device, dtype=dtype))

    def forward(self, x):
        idx = self.neuron_to_type_index
        batch_shape = x.shape[:-1]
        num_neurons = x.shape[-1]
        x_flat = x.reshape(-1, num_neurons)
        B = x_flat.shape[0]
        idx_expanded = idx.unsqueeze(0).expand(B, -1)
        type_sum = torch.zeros(B, self.num_cell_types, device=x.device, dtype=x.dtype)
        type_sum.scatter_add_(1, idx_expanded, x_flat)
        type_count = torch.zeros(B, self.num_cell_types, device=x.device, dtype=x.dtype)
        type_count.scatter_add_(1, idx_expanded, torch.ones_like(x_flat))
        type_count = type_count.clamp(min=1)
        type_mean = type_sum / type_count
        neuron_mean = type_mean.gather(1, idx_expanded)
        type_var_sum = torch.zeros(B, self.num_cell_types, device=x.device, dtype=x.dtype)
        type_var_sum.scatter_add_(1, idx_expanded, (x_flat - neuron_mean) ** 2)
        type_var = type_var_sum / type_count
        neuron_var = type_var.gather(1, idx_expanded)
        x_norm_flat = (x_flat - neuron_mean) / torch.sqrt(neuron_var + self.eps)
        x_norm = x_norm_flat.reshape(x.shape)
        w = self.weight_log.exp()[idx]
        b = self.bias[idx]
        return x_norm * w + b

    def extra_repr(self):
        return f'num_cell_types={self.num_cell_types}, eps={self.eps}'


class NeuralModelConv(torch.nn.Module):
    """Hex-grid spatial convolution layer.

    Gathers per-column features from hex-grid neighbours, applies a learned
    weight tensor via ``einsum``, and adds a bias. Missing neighbours (boundary
    columns marked ``-1``) and ``NaN`` input features (e.g. absent neuron
    types) are zeroed out before the contraction.
    """

    def __init__(self, in_features, out_features, conv_size, column_p_q,
                 device=None, dtype=None, positive_weights=False, use_bias=True):
        super().__init__()
        self.positive_weights = positive_weights
        nearest_column_indices = self.compute_nearest_columns(conv_size, column_p_q)
        self.register_buffer(
            'nearest_column_indices',
            torch.tensor(nearest_column_indices, device=device, dtype=torch.int64),
        )
        num_nearest = nearest_column_indices.shape[1]
        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, num_nearest, in_features,
                        device=device, dtype=dtype)
        )
        if use_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.bias = None

    def forward(self, x):
        squeeze_features = x.dim() >= 1 and x.shape[-1] != self.weight.shape[-1]
        if squeeze_features:
            x = x.unsqueeze(-1)

        leading = x.shape[:-2]
        x_flat = x.reshape(-1, x.shape[-2], x.shape[-1])

        safe_idx = self.nearest_column_indices.clamp(min=0)
        valid_mask = self.nearest_column_indices >= 0

        gathered = x_flat[:, safe_idx]
        gathered = gathered.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(-1), 0.0)
        gathered = torch.where(torch.isnan(gathered),
                               torch.zeros_like(gathered), gathered)

        if self.positive_weights:
            w = self.weight.exp()
            w_sum = torch.einsum('cn,knf->ck', valid_mask.float(), w).clamp(min=1e-8)
            out = torch.einsum('scnf,knf->sck', gathered, w) / w_sum.unsqueeze(0)
        else:
            out = torch.einsum('scnf,knf->sck', gathered, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        out = out.reshape(*leading, out.shape[-2], out.shape[-1])
        if squeeze_features:
            out = out.squeeze(-1)
        return out

    @staticmethod
    def hex_grid_axial_coords(conv_size: int) -> np.ndarray:
        """Return axial (q, r) coordinates for a flat-top hex grid."""
        if conv_size % 2 == 0:
            raise ValueError("conv_size must be odd for centered convolution.")
        radius = conv_size // 2
        coords = []
        for r in range(-radius, radius + 1):
            q_min = max(-radius, -r - radius)
            q_max = min(radius, -r + radius)
            for q in range(q_min, q_max + 1):
                coords.append((q, r))
        return np.array(coords, dtype=np.int64)

    @staticmethod
    def hex_grid_points(conv_size: int) -> np.ndarray:
        """Return flat-top hex grid points centered at (0,0)."""
        axial = NeuralModelConv.hex_grid_axial_coords(conv_size).astype(np.float64)
        q, r = axial[:, 0], axial[:, 1]
        return np.stack([1.5 * q, np.sqrt(3.0) * (r + 0.5 * q)], axis=1)

    @staticmethod
    def hex_grid_pq_offsets(conv_size: int) -> np.ndarray:
        """Return ``(dp, dq)`` offsets in the connectome's p,q lattice."""
        if conv_size % 2 == 0:
            raise ValueError("conv_size must be odd for centered convolution.")
        radius = conv_size // 2
        coords = []
        for dp in range(-radius, radius + 1):
            for dq in range(-radius, radius + 1):
                if max(abs(dp), abs(dq), abs(dp - dq)) <= radius:
                    coords.append((dp, dq))
        return np.array(coords, dtype=np.int64)

    @staticmethod
    def pq_to_xy(pq: np.ndarray) -> np.ndarray:
        """Project hex-lattice (p, q) coordinates to flat (x, y) for plotting."""
        angle_1 = np.pi / 6
        angle_2 = 2 * np.pi / 3 + angle_1
        basis = np.array([[np.cos(angle_1), np.cos(angle_2)],
                          [np.sin(angle_1), np.sin(angle_2)]])
        return (basis @ np.asarray(pq, dtype=np.float64).T).T

    @staticmethod
    def compute_nearest_columns(conv_size, column_p_q):
        """Compute neighbour column indices for every column on a p,q hex lattice."""
        if conv_size % 2 == 0:
            raise ValueError("conv_size must be odd for centered convolution.")
        if column_p_q is None:
            raise ValueError("column_p_q is required for convolution.")

        pq_offsets = NeuralModelConv.hex_grid_pq_offsets(conv_size)
        num_grid_points = pq_offsets.shape[0]
        p = column_p_q[:, 0].astype(np.int64)
        q = column_p_q[:, 1].astype(np.int64)
        num_columns = len(p)
        pq_to_col = {(p[i], q[i]): i for i in range(num_columns)}

        result = np.full((num_columns, num_grid_points), -1, dtype=np.int64)
        for col_idx in range(num_columns):
            pc, qc = p[col_idx], q[col_idx]
            for gi in range(num_grid_points):
                key = (pc + pq_offsets[gi, 0], qc + pq_offsets[gi, 1])
                if key in pq_to_col:
                    result[col_idx, gi] = pq_to_col[key]
        return result


class TemporalLRU(torch.nn.Module):
    """Temporal filter via complex diagonal linear recurrence.

    Each of the *N* modes is a damped oscillator whose frequency and damping
    are independently learnable. The per-timestep update is::

        z_k(t) = a_k * z_k(t-1) + b_k * u(t)      (complex state)
        y(t)   = sum_k Re(c_k * z_k(t))  + d * u(t) (real output)

    where ``a_k = exp(-gamma_k + i*omega_k)`` with ``gamma_k > 0`` for stability.
    """

    _COMPLEX_DTYPE = {torch.float32: torch.complex64, torch.float64: torch.complex128}

    def __init__(self, num_modes=4, dt=0.01, device='cpu', dtype='float32'):
        super().__init__()
        self.num_modes = num_modes
        self.dt = dt
        pt_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.log_gamma = torch.nn.Parameter(
            torch.zeros(num_modes, device=device, dtype=pt_dtype)
        )
        self.logit_omega = torch.nn.Parameter(
            torch.zeros(num_modes, device=device, dtype=pt_dtype)
        )
        self.b_re = torch.nn.Parameter(
            torch.randn(num_modes, device=device, dtype=pt_dtype) * 0.1
        )
        self.b_im = torch.nn.Parameter(
            torch.zeros(num_modes, device=device, dtype=pt_dtype)
        )
        self.c_re = torch.nn.Parameter(
            torch.randn(num_modes, device=device, dtype=pt_dtype) * 0.1
        )
        self.c_im = torch.nn.Parameter(
            torch.zeros(num_modes, device=device, dtype=pt_dtype)
        )
        self.log_d = torch.nn.Parameter(
            torch.zeros(1, device=device, dtype=pt_dtype)
        )

    @property
    def gamma(self):
        return self.log_gamma.exp()

    @property
    def omega(self):
        return torch.sigmoid(self.logit_omega) * np.pi

    def poles(self):
        return torch.exp(torch.complex(-self.gamma * self.dt, self.omega))

    def step(self, u_t, state=None):
        cdtype = self._COMPLEX_DTYPE[u_t.dtype]
        a = self.poles()
        b = torch.complex(self.b_re, self.b_im)
        c = torch.complex(self.c_re, self.c_im)

        if state is None:
            state = torch.zeros(
                u_t.shape[0], self.num_modes, u_t.shape[1],
                dtype=cdtype, device=u_t.device,
            )

        u_cx = u_t.to(cdtype)
        new_state = a[None, :, None] * state + b[None, :, None] * u_cx[:, None, :]
        y_t = (c[None, :, None] * new_state).real.sum(dim=1) + self.log_d.exp() * u_t
        return y_t, new_state

    def impulse_response(self, num_t=20):
        a = self.poles()
        cb = torch.complex(self.c_re, self.c_im) * torch.complex(self.b_re, self.b_im)
        h = torch.zeros(num_t, device=a.device, dtype=a.real.dtype)
        n = torch.arange(num_t, device=a.device, dtype=a.real.dtype)
        a_n = a[:, None] ** n[None, :]
        h[:] = (cb[:, None] * a_n).real.sum(dim=0)
        h[0] = h[0] + self.log_d.exp()
        return h

    def impulse_response_components(self, num_t=20):
        a = self.poles()
        cb = torch.complex(self.c_re, self.c_im) * torch.complex(self.b_re, self.b_im)
        n = torch.arange(num_t, device=a.device, dtype=a.real.dtype)
        a_n = a[:, None] ** n[None, :]
        modes = (cb[:, None] * a_n).real
        feedthrough = self.log_d.exp()
        return modes, feedthrough


# ---------------------------------------------------------------------------
# Ego-motion math helpers (rotation integration; standalone)
# ---------------------------------------------------------------------------


def _rodrigues(rotvec):
    """Rotation vectors (axis * angle, radians) to rotation matrices.

    Uses Taylor expansions for ``sin(theta)/theta`` and ``(1-cos(theta))/theta^2``
    at small angles to avoid catastrophic cancellation in the backward pass.
    """
    theta_sq = (rotvec * rotvec).sum(dim=-1, keepdim=True)

    x, y, z = rotvec[..., 0], rotvec[..., 1], rotvec[..., 2]
    O = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([O, -z, y], dim=-1),
        torch.stack([z, O, -x], dim=-1),
        torch.stack([-y, x, O], dim=-1),
    ], dim=-2)

    small = (theta_sq < 1e-6).squeeze(-1)
    safe_sq = theta_sq.clamp(min=1e-24)
    theta = safe_sq.sqrt()

    sinc = torch.where(small.unsqueeze(-1), 1 - theta_sq / 6, torch.sin(theta) / theta)
    cosc = torch.where(small.unsqueeze(-1), 0.5 - theta_sq / 24, (1 - torch.cos(theta)) / safe_sq)

    sinc = sinc.unsqueeze(-1)
    cosc = cosc.unsqueeze(-1)
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    return I + sinc * K + cosc * (K @ K)


def _rotmat_to_rotvec(R):
    """Rotation matrices to rotation vectors (axis * angle, radians)."""
    v = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)

    sin_angle = (v * v).sum(dim=-1).clamp(min=1e-24).sqrt() / 2
    cos_angle = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2

    angle = torch.atan2(sin_angle, cos_angle)

    angle_sq = angle * angle
    scale_small = 0.5 + angle_sq / 12
    scale_large = angle / (2 * sin_angle)
    scale = torch.where(sin_angle.abs() < 1e-4, scale_small, scale_large)

    return v * scale.unsqueeze(-1)


def integrate_angular_velocity(omega_deg, dt):
    """Integrate angular velocity to cumulative orientation via SO(3) composition."""
    squeeze = omega_deg.dim() == 2
    if squeeze:
        omega_deg = omega_deg.unsqueeze(0)

    rotvec_rad = omega_deg * (np.pi / 180) * dt
    R_inc = _rodrigues(rotvec_rad)

    B, T = omega_deg.shape[:2]
    I = torch.eye(3, device=omega_deg.device, dtype=omega_deg.dtype)
    R = I.unsqueeze(0).expand(B, -1, -1).clone()
    rotvecs = []
    for t in range(T):
        R = R_inc[:, t] @ R
        rotvecs.append(_rotmat_to_rotvec(R))
    result = torch.stack(rotvecs, dim=1) * (180 / np.pi)

    if squeeze:
        result = result.squeeze(0)
    return result


def integrate_translational_velocity(v, omega_deg, dt):
    """Integrate body-frame translational velocity to world-frame position."""
    squeeze = v.dim() == 2
    if squeeze:
        v = v.unsqueeze(0)
        omega_deg = omega_deg.unsqueeze(0)

    rotvec_rad = omega_deg * (np.pi / 180) * dt
    R_inc = _rodrigues(rotvec_rad)

    B, T = v.shape[:2]
    I = torch.eye(3, device=v.device, dtype=v.dtype)
    R = I.unsqueeze(0).expand(B, -1, -1).clone()
    pos = torch.zeros(B, 3, device=v.device, dtype=v.dtype)
    positions = []
    for t in range(T):
        R = R_inc[:, t] @ R
        v_world = (R @ v[:, t, :, None]).squeeze(-1)
        pos = pos + v_world * dt
        positions.append(pos.clone())
    result = torch.stack(positions, dim=1)

    if squeeze:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# BaseRnn: connectome-agnostic RNN training / eval scaffolding
# ---------------------------------------------------------------------------


class BaseRnn(SerializableModel, torch.nn.Module, ABC):
    """Connectome-agnostic RNN base.

    Owns ``predict``, ``set_sigma_max``, ``find_steady_state``,
    ``add_noise_to_parameters``, save/load, and the static ego-motion math.
    All connectome-specific state (synapse matrix, cell types, columns) lives
    on :class:`ConnectomeRnn` so a hypothetical non-connectome RNN can sit
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
        """Set up input convolution using NeuralModelConv + TemporalLRU.

        Pipeline: pre-filter -> local contrast normalisation -> temporal LRU.
        The pre-filter is a learned spatial conv applied before contrast
        normalisation. Called from :class:`ConnectomeRnn` once column_p_q is
        known.
        """
        if self.conv_size_x % 2 == 0:
            raise ValueError("conv_size_x must be odd for centered spatial convolution.")
        if self.input_pre_conv_size % 2 == 0:
            raise ValueError("input_pre_conv_size must be odd for centered spatial convolution.")
        if self.num_temporal_modes < 1:
            raise ValueError("num_temporal_modes must be >= 1.")

        self.input_conv_size_x = self.conv_size_x

        self.input_pre_conv = NeuralModelConv(
            in_features=1, out_features=1, conv_size=self.input_pre_conv_size,
            column_p_q=column_p_q, device=self.device, dtype=self.dtype,
            positive_weights=True, use_bias=False,
        )
        pq_offsets_pre = NeuralModelConv.hex_grid_pq_offsets(self.input_pre_conv_size)
        with torch.no_grad():
            self.input_pre_conv.weight.fill_(-10.0)
            for i, (dp, dq) in enumerate(pq_offsets_pre):
                if dp == 0 and dq == 0:
                    self.input_pre_conv.weight[0, i, 0] = 0.0

        self.input_spatial_conv = NeuralModelConv(
            in_features=1, out_features=1, conv_size=self.conv_size_x,
            column_p_q=column_p_q, device=self.device, dtype=self.dtype,
            positive_weights=True, use_bias=False,
        )
        pq_offsets = NeuralModelConv.hex_grid_pq_offsets(self.conv_size_x)
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

    def integrate_angular_velocity(self, omega_deg, dt=None):
        if dt is None:
            dt = self.dt
        return integrate_angular_velocity(omega_deg, dt)

    def integrate_translational_velocity(self, v, omega_deg, dt=None):
        if dt is None:
            dt = self.dt
        return integrate_translational_velocity(v, omega_deg, dt)

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
# ConnectomeRnn: loads a connectome and runs the predict() loop
# ---------------------------------------------------------------------------


class ConnectomeRnn(BaseRnn):
    """``BaseRnn`` augmented with a connectome.

    Loads connectome data via :func:`connectome_loading.load_connectome` (or
    accepts a pre-loaded :class:`ConnectomeData`, or raw kwargs for backward
    compatibility), filters / signs / indexes the synapse matrix, and
    initialises the input conv pipeline + readouts.

    The default behaviour assumes a whole-brain connectome — no left-eye
    mirroring; ``estimate_ego_motion_from_image`` runs a single pass over the
    full ``(B, T, num_columns)`` input. :class:`OpticLobeRightRnn` overrides
    the eye-flip behaviour for single-eye connectomes.
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

        cdata = self._resolve_connectome(connectome)
        self.connectome_name = cdata.name
        self.is_single_eye = cdata.is_single_eye

        # Store the resolved ConnectomeData in init_inputs so saved
        # checkpoints are self-contained (no parquet path required to reload).
        self._save_init_inputs()
        self.init_inputs['connectome'] = cdata

        valid_neurotransmitters = self.excitatory_nt + self.inhibitory_nt
        (
            root_ids, cell_types, neurotransmitters, column_indices,
            synapse_counts_coo,
        ) = self._preprocess_connectome(
            cdata.synapse_counts_coo, cdata.root_ids, cdata.cell_types,
            cdata.neurotransmitters, cdata.column_indices,
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
        self.column_phi_theta_two_eyes = self._build_column_phi_theta_two_eyes(
            cdata.column_phi_theta
        )
        self.column_p_q = cdata.column_p_q
        self.num_columns = np.unique(column_indices[~np.isnan(column_indices)]).size

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

    # ------------------------------------------------------------------
    # Connectome resolution & preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_connectome(connectome):
        """Coerce ``connectome`` (str tag or :class:`ConnectomeData`) into ConnectomeData."""
        if isinstance(connectome, ConnectomeData):
            return connectome
        if isinstance(connectome, str):
            return load_connectome(connectome)
        raise TypeError(
            f"`connectome` must be a str tag or ConnectomeData, "
            f"got {type(connectome).__name__}"
        )

    @staticmethod
    def _preprocess_connectome(synapse_counts_coo, root_ids, cell_types,
                               neurotransmitters, column_indices,
                               valid_neurotransmitters, count_threshold):
        """Filter the raw connectome and convert root IDs to matrix indices.

        Filtering (in order):
          1. Remove connections below the synapse count threshold.
          2. Remove neurons with invalid neurotransmitters.
          3. Remove connections involving removed neurons.
          4. Remove neurons with no remaining connections.

        Returns the filtered ``(root_ids, cell_types, neurotransmitters,
        column_indices, synapse_counts_coo)`` where the COO now stores matrix
        indices instead of root IDs.
        """
        import pandas as pd

        neurotransmitters = np.array(neurotransmitters)
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

        root_id_index = pd.Index(root_ids)
        pre_idx = root_id_index.get_indexer(synapse_counts_coo[:, 0])
        post_idx = root_id_index.get_indexer(synapse_counts_coo[:, 1])
        synapse_counts_coo = np.column_stack([pre_idx, post_idx, synapse_counts_coo[:, 2]])

        return root_ids, cell_types, neurotransmitters, column_indices, synapse_counts_coo

    # ------------------------------------------------------------------
    # Two-eye phi/theta: whole-brain default (no mirroring)
    # ------------------------------------------------------------------

    def _build_column_phi_theta_two_eyes(self, column_phi_theta):
        """Default: connectome already contains both eyes; pass through."""
        return (
            torch.tensor(column_phi_theta[:, 0], device=self.device),
            torch.tensor(column_phi_theta[:, 1], device=self.device),
        )

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

    # ------------------------------------------------------------------
    # Whole-brain ego-motion from image (single forward pass)
    # ------------------------------------------------------------------

    def estimate_ego_motion_from_image(self, inputs, *,
                                       initial_state=None,
                                       initial_filter_state=None,
                                       oob_columns=None,
                                       cached_mappings=None,
                                       terms=('rotation', 'trans')):
        """Single-pass ego-motion from a both-eyes ``(B, T, num_columns)`` stimulus.

        Returns ``(rotations, translations, result)`` where ``result`` is the
        dict produced by ``predict()``. For single-eye connectomes, use
        :class:`OpticLobeRightRnn`'s override which handles two-eye batching
        and the left-eye flip.
        """
        predict_kwargs = {}
        if cached_mappings is not None:
            predict_kwargs['input_neurons'] = cached_mappings['input_mapping']
            predict_kwargs['output_neurons'] = cached_mappings['output_mapping']

        S = inputs.shape[0]
        result = self.predict(
            inputs,
            init_state=initial_state, init_filter_state=initial_filter_state,
            **predict_kwargs,
        )

        flow_key = next(
            (k for k, m in self._readout_modules.items()
             if getattr(m, 'predicts_flow', False) and k in result['outputs']),
            None,
        )
        if flow_key is None:
            raise ValueError("No flow readout configured; cannot estimate ego motion.")

        optic_flow = result['outputs'][flow_key]
        phi_tensor = torch.tensor(
            self.column_phi_theta[:, 0], device=self.device, dtype=optic_flow.dtype,
        )
        theta_tensor = torch.tensor(
            self.column_phi_theta[:, 1], device=self.device, dtype=optic_flow.dtype,
        )

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

        return rotations, translations, result


# ---------------------------------------------------------------------------
# OpticLobeRightRnn: single-eye extras (mirror + two-eye predict + flip)
# ---------------------------------------------------------------------------


class OpticLobeRightRnn(ConnectomeRnn):
    """``ConnectomeRnn`` specialised for a single right-eye optic lobe connectome.

    Adds:
      - left-eye phi mirroring when building ``column_phi_theta_two_eyes``
      - ``predict_two_eyes`` that batches both eyes into one forward pass
      - a two-eye-aware ``estimate_ego_motion_from_image``
    """

    def _build_column_phi_theta_two_eyes(self, column_phi_theta):
        """Mirror the right eye's phi to synthesise the left eye."""
        left_phi_theta = column_phi_theta.copy()
        left_phi_theta[:, 0] = -left_phi_theta[:, 0]
        two_eye = np.concatenate((column_phi_theta, left_phi_theta), axis=0)
        return (
            torch.tensor(two_eye[:, 0], device=self.device),
            torch.tensor(two_eye[:, 1], device=self.device),
        )

    def predict_two_eyes(self, inputs_right, inputs_left,
                         init_state_right=None, init_state_left=None,
                         init_filter_state_right=None, init_filter_state_left=None,
                         **kwargs):
        """Run ``predict`` with both eyes batched into a single forward pass.

        Concatenates right and left inputs along the batch dimension, runs
        ``predict`` once, then splits and returns separate result dicts. The
        left-eye flow readout has its horizontal component flipped via
        ``flip_for_left_eye`` so both eyes share a consistent frame.
        """
        S = inputs_right.shape[0]
        both = torch.cat([inputs_right, inputs_left], dim=0)

        init_state = None
        if init_state_right is not None and init_state_left is not None:
            init_state = torch.cat([init_state_right, init_state_left], dim=0)

        init_filter_state = None
        if init_filter_state_right is not None and init_filter_state_left is not None:
            init_filter_state = torch.cat([init_filter_state_right, init_filter_state_left], dim=0)

        result = self.predict(
            both, init_state=init_state,
            init_filter_state=init_filter_state, **kwargs,
        )

        outputs = result['outputs']
        right_outputs = {}
        left_outputs = {}

        for key in ('voltage', 'firing_rate', 'synaptic_current'):
            if key in outputs:
                right_outputs[key] = {k: v[:S] for k, v in outputs[key].items()}
                left_outputs[key] = {k: v[S:] for k, v in outputs[key].items()}

        for rname, mod in self._readout_modules.items():
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
                (k for k, m in self._readout_modules.items()
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
            result = self.predict(
                inputs,
                init_state=init_state, init_filter_state=init_filter_state,
                **predict_kwargs,
            )

            flow_key = next(
                (k for k, m in self._readout_modules.items()
                 if getattr(m, 'predicts_flow', False) and k in result['outputs']),
                None,
            )
            if flow_key is None:
                raise ValueError("No flow readout configured; cannot estimate ego motion.")

            optic_flow = result['outputs'][flow_key]
            if eye == 'left':
                mod = self._readout_modules[flow_key]
                if mod.eye_asymmetric:
                    optic_flow = mod.flip_for_left_eye(optic_flow.clone())

            phi_np = self.column_phi_theta[:, 0].copy()
            theta_np = self.column_phi_theta[:, 1].copy()
            if eye == 'left':
                phi_np = -phi_np
            phi_tensor = torch.tensor(phi_np, device=self.device, dtype=optic_flow.dtype)
            theta_tensor = torch.tensor(theta_np, device=self.device, dtype=optic_flow.dtype)

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
# Dynamic class composition: dynamics x anatomy
# ---------------------------------------------------------------------------


def compose_model_class(dynamics_type: str, model_type: str):
    """Return a class that mixes ``dynamics_type`` onto ``model_type``.

    ``dynamics_type`` is a class name in :mod:`dynamics_models` (e.g.
    ``'CurrentDynamics'``). ``model_type`` is an anatomy class name in this
    module (e.g. ``'OpticLobeRightRnn'``, ``'ConnectomeRnn'``).

    Result is cached so repeated calls with the same pair return the *same*
    class object — important for ``isinstance`` and for save/load identity.
    """
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

    anatomy_cls = globals().get(model_type)
    if anatomy_cls is None or not (isinstance(anatomy_cls, type)
                                   and issubclass(anatomy_cls, BaseRnn)):
        raise ValueError(
            f"Unknown model_type {model_type!r}. Expected a BaseRnn subclass "
            f"defined in neural_models.py (e.g. 'OpticLobeRightRnn', "
            f"'ConnectomeRnn')."
        )

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
    return compose_model_class(payload['dynamics_type'], payload['model_type'])


# Attach to BaseRnn here (defined out-of-class so the factory above can
# reference BaseRnn and the cache without forward-reference gymnastics).
BaseRnn._save_class_identity = _baseRnn_save_class_identity
BaseRnn._resolve_class_from_payload = _baseRnn_resolve_class_from_payload
