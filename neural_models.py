from logging import warning
import platform
import subprocess
import warnings
import torch
import numpy as np
from pathlib import Path
from functools import reduce
import tempfile
import os
import inspect
from abc import ABC, abstractmethod

import data_utilities as du

MODEL_CLASS_REGISTRY = {}


def _register_model_class(model_cls):
    """Register model class for checkpoint dispatch."""
    MODEL_CLASS_REGISTRY[model_cls.__name__] = model_cls
    return model_cls


class CellTypeLayerNorm(torch.nn.Module):
    """Layer normalization with weight and bias shared per cell type instead of per neuron.

    Works like torch.nn.LayerNorm but stores only one weight and one bias per cell type,
    expanding them to per-neuron via a neuron-to-type index mapping before applying.
    """

    def __init__(self, num_cell_types, neuron_to_type_index, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.num_cell_types = num_cell_types
        self.eps = eps
        # neuron_to_type_index is a (num_neurons,) int64 tensor mapping each neuron to its cell type
        self.register_buffer('neuron_to_type_index', neuron_to_type_index)
        # Store weight in log space so exp() is always positive
        init_weight = 1.0
        init_weights = np.full(num_cell_types, np.log(init_weight))
        self.weight_log = torch.nn.Parameter(
            torch.tensor(init_weights, device=device, dtype=dtype)
        )
        self.bias = torch.nn.Parameter(torch.zeros(num_cell_types, device=device, dtype=dtype))

    def forward(self, x):
        # x shape: (..., num_neurons)
        # Compute mean and var per cell type, then scatter back to per-neuron
        idx = self.neuron_to_type_index  # (num_neurons,)
        batch_shape = x.shape[:-1]
        num_neurons = x.shape[-1]
        # Flatten batch dims for scatter_add
        x_flat = x.reshape(-1, num_neurons)  # (B, N)
        B = x_flat.shape[0]
        idx_expanded = idx.unsqueeze(0).expand(B, -1)  # (B, N)
        # Sum and count per cell type
        type_sum = torch.zeros(B, self.num_cell_types, device=x.device, dtype=x.dtype)
        type_sum.scatter_add_(1, idx_expanded, x_flat)
        type_count = torch.zeros(B, self.num_cell_types, device=x.device, dtype=x.dtype)
        type_count.scatter_add_(1, idx_expanded, torch.ones_like(x_flat))
        type_count = type_count.clamp(min=1)
        type_mean = type_sum / type_count  # (B, C)
        # Gather per-neuron mean
        neuron_mean = type_mean.gather(1, idx_expanded)  # (B, N)
        # Compute per cell type variance
        type_var_sum = torch.zeros(B, self.num_cell_types, device=x.device, dtype=x.dtype)
        type_var_sum.scatter_add_(1, idx_expanded, (x_flat - neuron_mean) ** 2)
        type_var = type_var_sum / type_count  # (B, C)
        # Gather per-neuron var
        neuron_var = type_var.gather(1, idx_expanded)  # (B, N)
        # Normalize
        x_norm_flat = (x_flat - neuron_mean) / torch.sqrt(neuron_var + self.eps)
        x_norm = x_norm_flat.reshape(x.shape)
        # Expand per-cell-type params to per-neuron
        w = self.weight_log.exp()[idx]
        b = self.bias[idx]
        return x_norm * w + b

    def extra_repr(self):
        return f'num_cell_types={self.num_cell_types}, eps={self.eps}'


class NeuralModelConv(torch.nn.Module):
    """Hex-grid spatial convolution layer.

    Gathers per-column features from hex-grid neighbours, applies a learned
    weight tensor via einsum, and adds a bias.  Missing neighbours (boundary
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
        """Apply hex-grid convolution.

        Args:
            x: ``(..., num_columns, in_features)`` or ``(..., num_columns)``
               when ``in_features == 1``.  Arbitrary leading dimensions are
               preserved.  May contain ``NaN`` for missing features (they
               are replaced with 0).

        Returns:
            ``(..., num_columns, out_features)`` or ``(..., num_columns)``
            when ``out_features == 1`` and the input had no feature dim.
        """
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
            # Per-column normalization: divide by sum of valid weights so they sum to 1
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

    # ------------------------------------------------------------------
    # Hex-grid geometry helpers (static, shared by all conv users)
    # ------------------------------------------------------------------

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
        """Return (dp, dq) offsets for a hex grid in the connectome's p,q
        lattice coordinates.

        The p,q basis vectors are at 120 degrees apart.  A point (dp, dq) is
        within radius *r* of the origin when ``max(|dp|, |dq|, |dp-dq|) <= r``.
        """
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
        """Project hex-lattice (p, q) coordinates to flat (x, y) for plotting.

        Companion to ``hex_grid_pq_offsets``: applies the canonical projection
        with basis vectors at 120 degrees apart, used everywhere we visualise
        per-column data or convolution kernels indexed in p,q order.
        """
        angle_1 = np.pi / 6
        angle_2 = 2 * np.pi / 3 + angle_1
        basis = np.array([[np.cos(angle_1), np.cos(angle_2)],
                          [np.sin(angle_1), np.sin(angle_2)]])
        return (basis @ np.asarray(pq, dtype=np.float64).T).T

    @staticmethod
    def compute_nearest_columns(conv_size, column_p_q):
        """Compute neighbour column indices for every column on a p,q hex lattice.

        Args:
            conv_size: Odd integer giving the hex-grid diameter.
            column_p_q: ``(num_columns, 2)`` array of integer lattice coords.

        Returns:
            ``(num_columns, num_grid_points)`` int64 array (``-1`` = missing).
        """
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

    Each of the *N* modes is a damped oscillator whose frequency and
    damping are independently learnable, providing natural bandpass
    behaviour.  The per-timestep update is::

        z_k(t) = a_k * z_k(t-1) + b_k * u(t)      (complex state)
        y(t)   = sum_k Re(c_k * z_k(t))  + d * u(t) (real output)

    where ``a_k = exp(-gamma_k + i*omega_k)`` with ``gamma_k > 0`` for
    stability.

    Parameters are learned in constrained spaces:
      - ``gamma = exp(log_gamma)``  -- positive damping rate (1/s)
      - ``omega = sigmoid(logit_omega) * pi``  -- frequency in ``[0, pi]`` rad/step
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
        """Damping coefficients (positive)."""
        return self.log_gamma.exp()

    @property
    def omega(self):
        """Angular frequencies in [0, pi]."""
        return torch.sigmoid(self.logit_omega) * np.pi

    def poles(self):
        """Complex poles ``a_k = exp(-gamma_k * dt + i*omega_k)``."""
        return torch.exp(torch.complex(-self.gamma * self.dt, self.omega))

    def step(self, u_t, state=None):
        """Single-timestep update.

        Args:
            u_t: ``(batch, columns)`` real input.
            state: ``(batch, num_modes, columns)`` complex, or *None*.

        Returns:
            y_t: ``(batch, columns)`` real output.
            new_state: ``(batch, num_modes, columns)`` complex.
        """
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
        """Analytically compute the impulse response h[0..num_t-1].

        Modes contribute from t=0 (using updated state), feedthrough at t=0.

        Returns:
            ``(num_t,)`` real tensor.
        """
        a = self.poles()
        cb = torch.complex(self.c_re, self.c_im) * torch.complex(self.b_re, self.b_im)
        h = torch.zeros(num_t, device=a.device, dtype=a.real.dtype)
        n = torch.arange(num_t, device=a.device, dtype=a.real.dtype)
        a_n = a[:, None] ** n[None, :]
        h[:] = (cb[:, None] * a_n).real.sum(dim=0)
        h[0] = h[0] + self.log_d.exp()
        return h

    def impulse_response_components(self, num_t=20):
        """Per-mode impulse response components and feedthrough.

        Returns:
            modes: ``(num_modes, num_t)`` real tensor — each mode's
                contribution ``Re(c_k * b_k * a_k^n)``.
            feedthrough: scalar ``d = exp(log_d)``.
        """
        a = self.poles()
        cb = torch.complex(self.c_re, self.c_im) * torch.complex(self.b_re, self.b_im)
        n = torch.arange(num_t, device=a.device, dtype=a.real.dtype)
        a_n = a[:, None] ** n[None, :]
        modes = (cb[:, None] * a_n).real
        feedthrough = self.log_d.exp()
        return modes, feedthrough


def _rodrigues(rotvec):
    """Rotation vectors (axis * angle, radians) to rotation matrices.

    Uses Taylor expansions for the ``sin(θ)/θ`` and ``(1-cos(θ))/θ²``
    coefficients at small angles to avoid catastrophic cancellation in
    the backward pass (autograd sees two large nearly-equal terms that
    subtract to zero in float32).

    Args:
        rotvec: (..., 3) rotation vectors.
    Returns:
        (..., 3, 3) rotation matrices.
    """
    theta_sq = (rotvec * rotvec).sum(dim=-1, keepdim=True)

    x, y, z = rotvec[..., 0], rotvec[..., 1], rotvec[..., 2]
    O = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([O, -z, y], dim=-1),
        torch.stack([z, O, -x], dim=-1),
        torch.stack([-y, x, O], dim=-1),
    ], dim=-2)

    # sin(θ)/θ  → 1 - θ²/6 + θ⁴/120   for small θ
    # (1-cos(θ))/θ² → 1/2 - θ²/24 + θ⁴/720  for small θ
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
    """Rotation matrices to rotation vectors (axis * angle, radians).

    Uses ``atan2`` instead of ``acos`` for the angle so that gradients stay
    bounded near the identity rotation (small angles).  The scale factor
    ``angle / (2 sin angle)`` is replaced by its Taylor expansion when
    ``sin angle`` is tiny, avoiding the 0/0 indeterminate form.

    Args:
        R: (..., 3, 3) rotation matrices.
    Returns:
        (..., 3) rotation vectors.
    """
    v = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)

    sin_angle = (v * v).sum(dim=-1).clamp(min=1e-24).sqrt() / 2
    cos_angle = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2

    angle = torch.atan2(sin_angle, cos_angle)

    # scale = angle / (2 sin_angle).  Taylor: 1/2 + angle²/12 + 7·angle⁴/720 + …
    angle_sq = angle * angle
    scale_small = 0.5 + angle_sq / 12
    scale_large = angle / (2 * sin_angle)
    scale = torch.where(sin_angle.abs() < 1e-4, scale_small, scale_large)

    return v * scale.unsqueeze(-1)


def integrate_angular_velocity(omega_deg, dt):
    """Integrate angular velocity to cumulative orientation via SO(3) composition.

    Composes incremental rotation matrices (Rodrigues) at each timestep and
    extracts the cumulative rotation vector (axis * angle).

    Args:
        omega_deg: (batch, T, 3) or (T, 3) angular velocity in deg/s.
            A 2-D input is treated as batch size 1 and the output is squeezed.
        dt: timestep in seconds.
    Returns:
        Cumulative rotation vectors in degrees, same shape as input.
    """
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
    """Integrate body-frame translational velocity to world-frame position.

    At each timestep the cumulative rotation (from *omega_deg*) is used to
    rotate the body-frame velocity into the world frame before accumulating.

    Args:
        v: (batch, T, 3) or (T, 3) body-frame translational velocity.
        omega_deg: (batch, T, 3) or (T, 3) angular velocity in deg/s
            (same shape as *v*).
        dt: timestep in seconds.
    Returns:
        World-frame cumulative displacement, same shape as input.
    """
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


class NeuralModel(torch.nn.Module, ABC):
    """Base class for neural models. Contains shared functionality for both current and conductance models."""
    
    def __init__(self, *, synapse_counts_coo, root_ids, cell_types, neurotransmitters, 
                 column_indices, column_phi_theta, column_p_q=None,
                 excitatory_nt=['ACH'], inhibitory_nt=['GLUT', 'GABA'], nonlinearity='Softplus', device='cpu', dtype='float32',
                 init_new_model=True, dt=0.01, seed=None, init_std=0.1, count_threshold=2, commit_hash=None,
                 readouts=None, readout_neurons=None, regularizer_parameters=None,
                 use_input_conv=True, conv_size_x=7, input_pre_conv_size=5, num_temporal_modes=4, use_layer_norm=True):
        super().__init__()
        # Save constructor inputs for serialization (child classes will merge their own on top)
        self._save_init_inputs()
        self.commit_hash = commit_hash
        self.silence_val = 1e-8  # when silencing neurons this is the value we set their connections to
        self.init_std = init_std
        self.count_threshold = count_threshold
        self.excitatory_nt = excitatory_nt
        self.inhibitory_nt = inhibitory_nt
        self.readouts = list(readouts) if readouts else []
        self.readout_neurons = list(readout_neurons) if readout_neurons else None
        self.regularizer_parameters = dict(regularizer_parameters) if regularizer_parameters else {}
        self.use_input_conv = use_input_conv
        valid_neurotransmitters = self.excitatory_nt + self.inhibitory_nt

        # Create RNG from seed for random initialization
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # --- model properties ---
        if nonlinearity in ("identity", "linear"):
            self.nonlinearity = torch.nn.Identity()
        else:
            self.nonlinearity = getattr(torch.nn, nonlinearity)()

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.dt = dt
        
        root_ids, cell_types, neurotransmitters, column_indices, synapse_counts_coo = (
            self._preprocess_connectome(
                synapse_counts_coo, root_ids, cell_types, neurotransmitters,
                column_indices, valid_neurotransmitters, count_threshold,
            )
        )

        # --- Now set up the model with filtered data ---
        self.root_ids = root_ids
        self.cell_types = cell_types
        self.num_neurons = cell_types.shape[0]
        self.unique_cell_types = np.unique(cell_types)
        self.num_cell_types = self.unique_cell_types.shape[0]
        
        # Store per-neuron transmitters
        self.transmitters = neurotransmitters
        
        # Build cell_type_transmitters (one per cell type, corresponding 1:1 to unique_cell_types)
        # For each unique cell type, get the neurotransmitter of the first neuron of that type
        self.cell_type_transmitters = np.array([
            neurotransmitters[cell_types == ct][0] for ct in self.unique_cell_types
        ])
        
        self.unique_transmitters = np.unique(self.transmitters)
        self.num_transmitters = len(self.unique_transmitters)
        self.column_indices = column_indices
        self.column_phi_theta = column_phi_theta
        left_phi_theta = column_phi_theta.copy()
        left_phi_theta[:, 0] = -left_phi_theta[:, 0]
        two_eye = np.concatenate((column_phi_theta, left_phi_theta), axis=0)
        self.column_phi_theta_two_eyes = (
            torch.tensor(two_eye[:, 0], device=self.device),
            torch.tensor(two_eye[:, 1], device=self.device),
        )
        self.column_p_q = column_p_q
        self.num_columns = np.unique(column_indices[~np.isnan(column_indices)]).size
        self.register_buffer('steady_state', torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype))

        # Layer norm flag (actual module created after neuron_to_type_index is built)
        self.use_layer_norm = use_layer_norm

        # --- create the matrix of synapse counts ---
        # Store filtered COO for subclasses that may need it (e.g., ConductanceModel)
        self.synapse_counts_coo = synapse_counts_coo
        
        rows = torch.tensor(synapse_counts_coo[:, 1], device=device, dtype=torch.int64)
        cols = torch.tensor(synapse_counts_coo[:, 0], device=device, dtype=torch.int64)
        values = torch.tensor(synapse_counts_coo[:, 2], device=device, dtype=self.dtype)

        indices = torch.stack([rows, cols], dim=0)
        self.synapse_counts = torch.sparse_coo_tensor(indices, values, size=(self.num_neurons, self.num_neurons), dtype=self.dtype, device=self.device).coalesce().to_sparse_csr()

        # Pre-compute the mapping from a cell type to the indicies of its individual neurons
        self.neuron_to_type_index = torch.zeros(self.num_neurons, device=self.device, dtype=torch.int64) - 1
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask_np = (self.cell_types == ct)
            ct_mask = torch.from_numpy(ct_mask_np).to(device=self.device)
            self.neuron_to_type_index[ct_mask] = cti
        assert torch.all(self.neuron_to_type_index != -1), 'some cell type to neuron mappings not assigned'

        # Optional layer normalization with shared weight/bias per cell type
        if self.use_layer_norm:
            self.layer_norm = CellTypeLayerNorm(
                self.num_cell_types, self.neuron_to_type_index,
                device=self.device, dtype=self.dtype,
            )

        self._initialize_readouts()
        if self.use_input_conv:
            self._initialize_input_conv(conv_size_x=conv_size_x, input_pre_conv_size=input_pre_conv_size, num_temporal_modes=num_temporal_modes)


    @property
    def has_decoder(self):
        """Backward-compat: True when any flow readout is configured."""
        return any(getattr(mod, 'predicts_flow', False) for mod in self._readout_modules.values())

    def get_column_xyz(self, radius: float = 1.0) -> np.ndarray:
        """Return column locations projected to 3D Cartesian coordinates."""
        phi = self.column_phi_theta[:, 0]
        theta = self.column_phi_theta[:, 1]
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.stack((x, y, z), axis=1)

    @staticmethod
    def _preprocess_connectome(synapse_counts_coo, root_ids, cell_types,
                               neurotransmitters, column_indices,
                               valid_neurotransmitters, count_threshold):
        """Filter the raw connectome and convert root IDs to matrix indices.

        Filtering steps (in order):
        1. Remove connections below the synapse count threshold.
        2. Remove neurons with invalid neurotransmitters.
        3. Remove connections involving removed neurons.
        4. Remove neurons with no remaining connections.

        Returns the filtered ``(root_ids, cell_types, neurotransmitters,
        column_indices, synapse_counts_coo)`` where the COO now stores
        matrix indices instead of root IDs.
        """
        import pandas as pd

        neurotransmitters = np.array(neurotransmitters)
        assert root_ids.dtype == np.int64, 'root_ids must have dtype np.int64'
        assert synapse_counts_coo.dtype == np.int64, 'synapse_counts_coo must have dtype np.int64'

        # Remove connections below threshold
        above_threshold = np.abs(synapse_counts_coo[:, 2]) >= count_threshold
        synapse_counts_coo = synapse_counts_coo[above_threshold]

        # Keep only neurons with valid neurotransmitters
        valid_nt_mask = np.isin(neurotransmitters, valid_neurotransmitters)

        # Remove connections involving invalid neurons
        valid_ids = root_ids[valid_nt_mask]
        valid_connections = (
            np.isin(synapse_counts_coo[:, 0], valid_ids)
            & np.isin(synapse_counts_coo[:, 1], valid_ids)
        )
        synapse_counts_coo = synapse_counts_coo[valid_connections]

        # Remove neurons with no remaining connections
        connected_ids = np.unique(synapse_counts_coo[:, 0:2])
        valid_neurons = valid_nt_mask & np.isin(root_ids, connected_ids)

        root_ids = root_ids[valid_neurons]
        cell_types = cell_types[valid_neurons]
        neurotransmitters = neurotransmitters[valid_neurons]
        column_indices = column_indices[valid_neurons]

        # Convert root IDs to matrix indices
        root_id_index = pd.Index(root_ids)
        pre_idx = root_id_index.get_indexer(synapse_counts_coo[:, 0])
        post_idx = root_id_index.get_indexer(synapse_counts_coo[:, 1])
        synapse_counts_coo = np.column_stack([pre_idx, post_idx, synapse_counts_coo[:, 2]])

        return root_ids, cell_types, neurotransmitters, column_indices, synapse_counts_coo

    def _initialize_input_conv(self, conv_size_x=7, input_pre_conv_size=5, num_temporal_modes=4):
        """Set up input convolution using a ``NeuralModelConv`` for the spatial
        part and a ``TemporalLRU`` for the temporal part.

        The pipeline is: pre-filter -> local contrast normalization -> temporal LRU.
        The pre-filter is a learned spatial convolution applied before contrast
        normalization.
        """
        if conv_size_x % 2 == 0:
            raise ValueError("conv_size_x must be odd for centered spatial convolution.")
        if input_pre_conv_size % 2 == 0:
            raise ValueError("input_pre_conv_size must be odd for centered spatial convolution.")
        if num_temporal_modes < 1:
            raise ValueError("num_temporal_modes must be >= 1.")

        self.input_conv_size_x = conv_size_x
        self.input_pre_conv_size = input_pre_conv_size

        self.input_pre_conv = NeuralModelConv(
            in_features=1, out_features=1, conv_size=input_pre_conv_size,
            column_p_q=self.column_p_q, device=self.device, dtype=self.dtype,
            positive_weights=True, use_bias=False,
        )

        pq_offsets_pre = NeuralModelConv.hex_grid_pq_offsets(input_pre_conv_size)
        with torch.no_grad():
            self.input_pre_conv.weight.fill_(-10.0)
            for i, (dp, dq) in enumerate(pq_offsets_pre):
                if dp == 0 and dq == 0:
                    self.input_pre_conv.weight[0, i, 0] = 0.0


        self.input_spatial_conv = NeuralModelConv(
            in_features=1, out_features=1, conv_size=conv_size_x,
            column_p_q=self.column_p_q, device=self.device, dtype=self.dtype,
            positive_weights=True, use_bias=False,
        )

        pq_offsets = NeuralModelConv.hex_grid_pq_offsets(conv_size_x)
        with torch.no_grad():
            self.input_spatial_conv.weight.fill_(-10.0)
            for i, (dp, dq) in enumerate(pq_offsets):
                if max(abs(dp), abs(dq), abs(dp - dq)) <= 1:
                    self.input_spatial_conv.weight[0, i, 0] = 0.0

        self.temporal_lru = TemporalLRU(
            num_modes=num_temporal_modes, dt=self.dt, device=self.device, dtype=self.dtype,
        )

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

    def apply_input_conv(self, u_t, filter_state=None):
        """Apply spatial pre-filter, contrast normalization, then temporal LRU.

        Args:
            u_t: ``(batch, columns)`` raw input for a single timestep.
            filter_state: complex tensor from a previous call, or *None*.

        Returns:
            y_t: ``(batch, columns)`` filtered output.
            new_filter_state: updated complex state for the next timestep.
        """
        filtered = self.input_pre_conv(u_t)

        local_mean = self.input_spatial_conv(filtered)
        centered = filtered - local_mean
        local_var = self.input_spatial_conv(centered ** 2)
        normed = -centered / torch.sqrt(local_var + 1e-12)

        y_t, new_filter_state = self.temporal_lru.step(normed, state=filter_state)

        return y_t, new_filter_state

    @abstractmethod
    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        """Forward pass for the neural model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def firing_rate(self, v):
        """Convert voltage to firing rate. Must be implemented by subclasses."""
        pass

    def _save_init_inputs(self):
        """Save the __init__ inputs from the calling frame to self.init_inputs.
        
        Call this at the start of each __init__ to capture the constructor arguments for
        serialization. Handles **kwargs by unpacking them into the saved dict so
        serialization stays flat.
        
        When a child class calls this after super().__init__(), the parent's resolved
        defaults are preserved and the child's inputs are merged on top. This ensures
        every parameter (including defaults the caller didn't explicitly pass) is saved,
        so old checkpoints remain valid even if defaults later change.
        """
        frame = inspect.currentframe().f_back
        params = frame.f_locals.copy()
        params.pop('self', None)
        params.pop('__class__', None)
        # Unpack **kwargs so serialized inputs stay flat (e.g. {'device': 'cpu'} not {'kwargs': {...}})
        if 'kwargs' in params:
            kwargs = params.pop('kwargs')
            params.update(kwargs)
        # Merge with previously saved inputs (e.g. parent class defaults) so that
        # resolved default values are never lost when a child class overwrites.
        if hasattr(self, 'init_inputs'):
            merged = dict(self.init_inputs)
            merged.update(params)
            self.init_inputs = merged
        else:
            self.init_inputs = params
    
    def add_noise_to_parameters(self, noise_std):
        """Add Gaussian noise to all trainable parameters."""
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.tensor(
                    self.rng.normal(0, noise_std, param.shape),
                    dtype=param.dtype,
                    device=param.device
                )
                param.add_(noise)

    def predict(self, inputs, input_neurons=['L1', 'L2', 'L3'],  output_neurons=['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'], 
                init_state=None, init_filter_state=None):
        num_batches, num_time, num_neurons = inputs.shape

        filter_state = init_filter_state

        # Handle input mapping (either compute from names list, or reuse provided indices)
        if isinstance(input_neurons, list):
            input_neurons = [i for i in input_neurons if i in self.unique_cell_types]
            output_neurons = [i for i in output_neurons if i in self.unique_cell_types]

            # get a mapping from the stimulus (batch, time, columns) columns to neurons
            input_neuron_indices, input_column_indices = self.get_shared_columns(input_neurons)
            input_column_indices_np = input_column_indices
            stim_to_cell_mapping = {
                'neuron_index': torch.tensor(input_neuron_indices, device=self.device, dtype=torch.int64),
                'stimulus_index': torch.tensor(input_column_indices, device=self.device, dtype=torch.int64),
            }
        else:
            # assume provided indices mapping (dict with 'neuron_index' and 'stimulus_index')
            stim_to_cell_mapping = input_neurons
            # derive numpy column indices for readout filtering
            input_column_indices_np = stim_to_cell_mapping['stimulus_index'].detach().cpu().numpy()

        # Handle readouts (either compute from names list, or reuse provided dict of indices)
        if isinstance(output_neurons, list):
            # if the output is columnar get the columns where we have all the readouts
            readout_dict = {}
            for i in output_neurons:
                # only use output neurons that have an assigned column, are of the correct type, and have an input
                # we require an input in that column so we don't waste time trying to fit neurons that are getting limited input
                is_correct_type = (self.cell_types == i) & ~np.isnan(self.column_indices) & np.isin(self.column_indices, input_column_indices_np)
                neuron_index = np.arange(self.num_neurons)
                readout_dict[i] = {
                    'column_indices': torch.tensor(self.column_indices[is_correct_type], device=self.device, dtype=torch.int64),
                    'neuron_indices': torch.tensor(neuron_index[is_correct_type], device=self.device, dtype=torch.int64),
                }
            readout_keys = output_neurons
        else:
            # assume provided mapping dict keyed by readout names
            readout_dict = output_neurons
            readout_keys = list(readout_dict.keys())

        # initial neural activity values
        if init_state is None:
            v = self.steady_state.unsqueeze(0).expand(num_batches, -1)
        else:
            v = init_state

        # outputs for each neuron in output_neurons are (batches, time, number of columns)
        outputs = {'voltage': {}, 'firing_rate': {}}
        record_synaptic = isinstance(self, HandTunedModel)
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

    def predict_two_eyes(self, inputs_right, inputs_left,
                         init_state_right=None, init_state_left=None,
                         init_filter_state_right=None, init_filter_state_left=None,
                         **kwargs):
        """Run predict with both eyes batched into a single forward pass.

        Concatenates right and left inputs along the batch dimension, runs
        ``predict`` once, then splits and returns separate result dicts.  The
        left eye decoder output has its horizontal flow component flipped so
        both eyes share a consistent coordinate frame.

        Returns:
            (right_result, left_result): each a dict with keys ``outputs``,
            ``model_state``, and ``mappings`` (same structure as ``predict``).
        """
        S = inputs_right.shape[0]
        both = torch.cat([inputs_right, inputs_left], dim=0)

        init_state = None
        if init_state_right is not None and init_state_left is not None:
            init_state = torch.cat([init_state_right, init_state_left], dim=0)

        init_filter_state = None
        if init_filter_state_right is not None and init_filter_state_left is not None:
            init_filter_state = torch.cat([init_filter_state_right, init_filter_state_left], dim=0)

        result = self.predict(both, init_state=init_state,
                              init_filter_state=init_filter_state, **kwargs)

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

    # Default spectral-norm target for the linearized dynamics matrix.
    # Subclasses that implement calc_sigma_max() can call set_sigma_max() to
    # iteratively rescale output_strength_log so the linearized recurrent
    # operator stays under this bound at initialization.
    max_allowed_sigma = 0.95

    def set_sigma_max(self):
        """Iteratively rescale the recurrent operator to hit ``max_allowed_sigma``.

        Subclasses must implement ``calc_sigma_max``. The actual rescaling step
        is delegated to ``_apply_sigma_scale`` so subclasses can scale whatever
        parameter controls overall recurrent gain (e.g. ``output_strength_log``
        for the connectome-based models, ``W`` for ``ArbitraryModel``).
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

    def _apply_sigma_scale(self, scale_factor):
        """Multiply the recurrent gain by ``scale_factor`` (in linear units).

        Default: shift ``output_strength_log`` by ``log(scale_factor)``, which
        scales every per-type output gain by ``scale_factor``. Override in
        subclasses that parameterize recurrence differently.
        """
        self.output_strength_log.add_(torch.log(scale_factor))

    def find_steady_state(self, run_time=100, max_iter=25):
        print("Finding steady state ...")
        inputs = torch.zeros((1, run_time, self.num_columns), device=self.device, dtype=self.dtype)
        # Start from current steady state buffer
        initial_state = self.steady_state.unsqueeze(0)
        initial_ave = initial_state.abs().mean()

        for i in range(max_iter):
            final_state = self.predict(inputs, init_state=initial_state)['model_state']['v']

            # Check if state converged (didn't change from input to output)
            if torch.allclose(final_state, initial_state, atol=1e-3):
                print('steady state found after', i + 1, 'iterations')
                self.steady_state.copy_(final_state.detach().squeeze(0))
                return
            
            initial_state = final_state.detach().clone()

        # Check for exploding dynamics: if new state is much larger than previous
        final_ave = final_state.abs().mean()
        if final_ave / (initial_ave + 1) > 1e6:
            print('Warning: cannot find steady state, dynamics may be unstable')
            return
            
        print('Average abs diff:', (final_state - initial_state).abs().mean())

        print('steady state not found after', max_iter, 'iterations')
        self.steady_state.copy_(final_state.detach().squeeze(0))

        return

    def to(self, *args, **kwargs):
        """Override to also move non-parameter/buffer tensors and update self.device."""
        result = super().to(*args, **kwargs)
        # Move plain tensor attributes that nn.Module.to() doesn't handle
        if hasattr(result, 'synapse_counts') and isinstance(result.synapse_counts, torch.Tensor):
            result.synapse_counts = result.synapse_counts.to(*args, **kwargs)
        if hasattr(result, 'neuron_to_type_index') and isinstance(result.neuron_to_type_index, torch.Tensor):
            # Must stay integer: forwarding ``.to(dtype=bfloat16)`` (or any float dtype)
            # from user code would cast this mapping to float and break ``tensor[idx]``
            # on CUDA with IndexError.
            non_blocking = kwargs.get('non_blocking', False)
            result.neuron_to_type_index = result.neuron_to_type_index.to(
                device=result.synapse_counts.device,
                dtype=torch.int64,
                non_blocking=non_blocking,
            )
        # Update self.device if a device was specified
        device = None
        if args and isinstance(args[0], (torch.device, str)):
            device = torch.device(args[0])
        elif 'device' in kwargs:
            device = torch.device(kwargs['device'])
        if device is not None:
            result.device = device
        return result

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

    def _collect_extra_state(self):
        """Collect non-parameter, non-buffer state for serialization.

        Override in subclasses to persist tensor attributes (like
        ``synapse_counts``) that are not registered as buffers or parameters
        and would therefore be lost by ``state_dict()`` alone.
        """
        return {'synapse_counts': self.synapse_counts}

    def _restore_extra_state(self, state, device, dtype):
        """Restore state saved by :meth:`_collect_extra_state`.

        Legacy checkpoints stored ``synapse_counts`` as COO; the canonical
        in-memory layout is now CSR. Convert here so loaded models match
        freshly-constructed ones.
        """
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

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'inputs': self.init_inputs,
            'class_name': self.__class__.__name__,
            # store weights on CPU for portability (includes buffers like optic_flow_*_readout)
            'state_dict': {k: v.detach().cpu() for k, v in self.state_dict().items()},
            'extra_state': self._collect_extra_state(),
        }

        # Use atomic write to prevent corruption on network filesystems
        # Write to temporary file first, then atomically replace target
        # This ensures all-or-nothing writes and prevents partial/corrupted files
        try:
            with tempfile.NamedTemporaryFile('wb', delete=False, dir=str(p.parent), suffix='.tmp') as tmp:
                tmp_path = tmp.name
                # torch.save uses pickle under the hood and handles tensors well
                torch.save(payload, tmp_path)
                tmp.flush()
                os.fsync(tmp.fileno())
            # Atomically replace target file

            os.replace(tmp_path, str(p))
        except Exception:
            # Clean up temp file on error
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            raise

    def get_shared_columns(self, neuron_list):
        # get all the neurons that are of the type in the list and their indices

        # get a mapping from the stimulus (batch, time, columns) columns to neurons
        neuron_indices = np.arange(self.num_neurons)

        # get the list of column indices which have all of the input neurons
        column_indices = [self.column_indices[self.cell_types == i] for i in neuron_list]
        column_indices = [i[~np.isnan(i)] for i in column_indices]
        column_indices = reduce(np.intersect1d, column_indices)

        # get the neuron indices of the neurons in those columns with the right type
        neuron_masks = [(self.cell_types == i) for i in neuron_list]
        neuron_is_in_list = np.any(np.stack(neuron_masks), axis=0)
        neuron_is_in_list = neuron_is_in_list & np.isin(self.column_indices, column_indices)

        selected_neuron_indices = neuron_indices[neuron_is_in_list]
        selected_column_indices = self.column_indices[neuron_is_in_list]

        return selected_neuron_indices, selected_column_indices

    def _silencing_mask(self, neurons_to_silence, invert_silencing=False):
        """Return a boolean mask over cell types for the neurons to silence."""
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

    @classmethod
    def load(cls, path, device=None, dtype=None, freeze_params: bool = True):
        # should not have any backwards compatibility. Old models will be updated with other code if the
        # model structure changes
        path = Path(path)
        if path.is_dir():
            path = path / "model.pt"
        data = torch.load(path, map_location='cpu', weights_only=False)
        inputs = dict(data['inputs'])

        if device is not None:
            inputs['device'] = device
        else:
            inputs['device'] = du.load_config()['model']['device']

        if dtype is not None:
            inputs['dtype'] = dtype

        model_cls = MODEL_CLASS_REGISTRY[data['class_name']]
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
            model_device = torch.device(inputs['device'])
            model_dtype = getattr(torch, inputs.get('dtype', 'float32'))
            model._restore_extra_state(data['extra_state'], model_device, model_dtype)

        if freeze_params:
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()

        # Verify that the current codebase matches the commit the model was trained on.
        # A mismatch may mean the model architecture or weights are incompatible with
        # the current code, which could silently produce incorrect results.
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
            print(f"local commit hash not detected")

        return model

    def estimate_ego_motion_from_image(self,
                                       right_eye_inputs=None, left_eye_inputs=None,
                                       initial_state_right=None, initial_state_left=None,
                                       initial_filter_state_right=None, initial_filter_state_left=None,
                                       oob_columns=None, cached_mappings=None,
                                       eyes=('right', 'left'),
                                       terms=('rotation', 'trans')):
        """Ego-motion pipeline from raw eye image inputs.

        Args:
            right_eye_inputs, left_eye_inputs: per-eye stimulus tensors. Each is
                required only if the corresponding eye is listed in ``eyes``.
            eyes: subset of {'right','left'} indicating which eyes to use.
                Default uses both eyes (concatenated flow with mirrored phi).
            terms: subset of {'rotation','trans'} indicating which ego-motion
                components to estimate. Skipped components are returned as
                zeros (shape preserved). When only ``'trans'`` is requested,
                rotation is not subtracted from the flow before solving for
                translation.
            cached_mappings: optional dict with 'input_mapping' and
                'output_mapping' from a previous call's result['mappings'].
                Skips expensive per-call recomputation of neuron index mappings.

        Returns:
            (rotations, translations, right_result, left_result). The eye
            result dicts have the same structure as ``predict`` /
            ``predict_two_eyes``; the result for an eye not in ``eyes`` is
            ``None``.
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
                # Match the shared coordinate frame used by predict_two_eyes:
                # apply the same flip the two-eye path applies to the left eye.
                mod = self._readout_modules[flow_key]
                if mod.eye_asymmetric:
                    optic_flow = mod.flip_for_left_eye(optic_flow.clone())

            # Single-eye phi/theta in the shared coordinate frame: the left eye
            # has its azimuthal angle mirrored, mirroring the construction at
            # __init__ for column_phi_theta_two_eyes.
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

    def integrate_angular_velocity(self, omega_deg, dt=None):
        """Integrate angular velocity to cumulative orientation via SO(3).

        Convenience wrapper around the module-level ``integrate_angular_velocity``
        that defaults *dt* to ``self.dt``.

        Args:
            omega_deg: (batch, T, 3) or (T, 3) angular velocity in deg/s.
            dt: timestep in seconds (default: ``self.dt``).
        Returns:
            Cumulative rotation vectors in degrees, same shape as input.
        """
        if dt is None:
            dt = self.dt
        return integrate_angular_velocity(omega_deg, dt)

    def integrate_translational_velocity(self, v, omega_deg, dt=None):
        """Integrate body-frame translational velocity to world-frame position.

        Convenience wrapper around the module-level
        ``integrate_translational_velocity`` that defaults *dt* to ``self.dt``.

        Args:
            v: (batch, T, 3) or (T, 3) body-frame translational velocity.
            omega_deg: (batch, T, 3) or (T, 3) angular velocity in deg/s.
            dt: timestep in seconds (default: ``self.dt``).
        Returns:
            World-frame cumulative displacement, same shape as input.
        """
        if dt is None:
            dt = self.dt
        return integrate_translational_velocity(v, omega_deg, dt)

    @staticmethod
    def _cross_matrix(a):
        """Skew-symmetric matrix [a]_x for ``a`` shaped (..., 3) -> (..., 3, 3)."""
        ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
        O = torch.zeros_like(ax)
        return torch.stack([
            torch.stack([   O, -az,  ay], dim=-1),
            torch.stack([  az,   O, -ax], dim=-1),
            torch.stack([ -ay,  ax,   O], dim=-1)
        ], dim=-2)

    @staticmethod
    def _spherical_basis(phi, theta):
        """Unit viewing direction ``x`` and tangent basis ``e_phi``, ``e_theta``.

        Returns three (N, 3) tensors. ``e_phi`` points east (toward +phi),
        ``e_theta`` points south (toward +theta, i.e. down).
        """
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
        """Solve u ≈ [x]_× Ω via normal equations.

        Inputs are (T, N). Returns ``omega_star`` shape (T, 3). The rotation
        convention follows :py:meth:`estimate_ego_motion_from_flow`.
        """
        x, e_phi, e_theta = cls._spherical_basis(phi, theta)
        # u = -d_right * e_phi - d_up * e_theta (rightward = -e_phi, upward = -e_theta)
        u = -d_right[..., None] * e_phi - d_up[..., None] * e_theta  # (T, N, 3)
        T_len, N = d_right.shape
        A = cls._cross_matrix(x)                  # (N, 3, 3)
        A_flat = A.reshape(3 * N, 3)              # (3N, 3)
        u_flat = u.reshape(T_len, 3 * N).T        # (3N, T)
        AtA = A_flat.T @ A_flat                   # (3, 3) — constant across time
        Atb = A_flat.T @ u_flat                   # (3, T)
        omega_star = torch.linalg.solve(AtA, Atb).T  # (T, 3)
        return omega_star

    @classmethod
    def _estimate_translation_from_flow(cls, phi, theta, d_right, d_up):
        """Solve translation direction + per-point ρ from (de-rotated) flow.

        Inputs are (T, N) angular flow components in the same convention as
        :py:meth:`estimate_ego_motion_from_flow`; the caller is responsible
        for de-rotating the flow first when rotation is also being estimated.
        Returns ``(T_star, rho_star)`` shapes ``(T, 3)`` and ``(T, N)``.
        """
        x, e_phi, e_theta = cls._spherical_basis(phi, theta)
        # Use ``v`` for the (de-rotated) translational flow in 3D.
        v = -d_right[..., None] * e_phi - d_up[..., None] * e_theta  # (T, N, 3)
        T_len, N = d_right.shape

        # Projection matrices Π_i = I - x_i x_iᵀ — shared across time.
        outer = x.unsqueeze(2) @ x.unsqueeze(1)                       # (N, 3, 3)
        I = torch.eye(3, device=x.device, dtype=x.dtype).expand(N, 3, 3)
        P = I - outer                                                  # (N, 3, 3)

        # Stacked C matrix: each block = [v_i]_× Π_i; solve C T = 0 via SVD.
        vx = cls._cross_matrix(v)                                     # (T, N, 3, 3)
        C_blocks = vx @ P                                              # (T, N, 3, 3)
        C = C_blocks.reshape(T_len, 3 * N, 3)                          # (T, 3N, 3)
        _, _, Vh = torch.linalg.svd(C, full_matrices=False)            # Vh: (T, 3, 3)
        T_dir = Vh[:, -1]                                              # (T, 3)

        # Disambiguate sign via depth-positivity (see doc stage 3).
        w_i = torch.einsum('nij,tj->tni', P, T_dir)                    # (T, N, 3)
        v_dot_w = (v * w_i).sum(dim=-1)                                # (T, N)
        w_norm_sq = (w_i * w_i).sum(dim=-1)                            # (T, N)
        weighted_sign = (v_dot_w * w_norm_sq).sum(dim=-1)              # (T,)
        sign_flip = torch.where(
            weighted_sign > 0,
            torch.tensor(-1.0, device=x.device, dtype=d_right.dtype),
            torch.tensor(1.0, device=x.device, dtype=d_right.dtype),
        )
        T_dir = T_dir * sign_flip[:, None]
        v_dot_w = v_dot_w * sign_flip[:, None]

        # Stage 4: per-point ρ_i = ‖T‖/Z_i (positive after flip).
        rho_star = -v_dot_w / (w_norm_sq + 1e-12)                      # (T, N)

        # Stage 5: aggregate ρ_i into a scalar via median over points away
        # from the focus of expansion (where ‖Π t̂‖² ≈ 0 is ill-conditioned).
        focus_threshold = 0.1
        valid = w_norm_sq > focus_threshold                            # (T, N)
        rho_masked = torch.where(valid, rho_star, torch.full_like(rho_star, float('nan')))
        rho_agg = torch.nanmedian(rho_masked, dim=-1).values           # (T,)
        rho_agg = torch.where(
            torch.isnan(rho_agg),
            torch.nanmedian(rho_star, dim=-1).values,
            rho_agg,
        )
        T_star = rho_agg[:, None] * T_dir                              # (T, 3)
        return T_star, rho_star

    def estimate_ego_motion_from_flow(self, phi, theta, d_right, d_up,
                                      terms=('rotation', 'trans')):
        """
        Estimate ego-motion (rotation and translation) from optic flow.

        Supports batched computation over time: if d_right and d_up have a leading
        time dimension, all time steps are solved in parallel.

        Coordinate system:
            - +X: Forward (phi=0, theta=π/2)
            - +Y: Left (phi=π/2, theta=π/2)
            - +Z: Up (theta=0)

        Flow sign convention:
            - Positive d_right: rightward motion on retina → leftward observer rotation
            - Positive d_up: upward motion on retina → downward observer motion

        Args:
            phi: (N,) azimuthal angles of viewing directions
            theta: (N,) polar angles of viewing directions
            d_right: (N,) or (T, N) angular flow in phi direction (positive = rightward on retina)
            d_up: (N,) or (T, N) angular flow in theta direction (positive = upward on retina)
            terms: subset of {'rotation','trans'} indicating which components
                to estimate. Skipped components are returned as zeros (shape
                preserved). When only ``'trans'`` is requested, rotation is
                NOT subtracted from the flow before solving for translation.

        Returns:
            omega_star: (3,) or (T, 3) estimated rotation rate [roll, pitch, yaw]
                in physical observer convention (positive omega_z = leftward
                rotation). Zeros if ``'rotation'`` is not in ``terms``.
            T_star: (3,) or (T, 3) estimated observer translational velocity,
                T* = ρ_agg · t̂ (doc stage 5), where t̂ is the sign-disambiguated
                unit direction (stage 3) and ρ_agg is the median of ρ_i over
                points away from the focus of expansion (‖Π t̂‖² > 0.1).
                Zeros if ``'trans'`` is not in ``terms``.
            rho_star: (N,) or (T, N) per-point ρ_i = ‖T‖/Z_i, the recoverable
                translational-flow-over-depth magnitude (doc stage 4). Absolute
                depth Z_i and absolute ‖T‖ cannot be separated from monocular
                flow alone; only their ratio ρ_i is observable. Together with
                t̂ this fully reconstructs the de-rotated flow:
                v_i = -ρ_i · Π_i · t̂. Zeros if ``'trans'`` is not in ``terms``.
        """
        terms = tuple(terms)
        valid_terms = {'rotation', 'trans'}
        if not terms or not set(terms).issubset(valid_terms):
            raise ValueError(
                f"`terms` must be a non-empty subset of {valid_terms}, got {terms}"
            )

        # Handle unbatched input: (N,) → (1, N)
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
                # De-rotate flow before translation: subtract the rotational
                # contribution u_rot = [x]_× Ω in 3D, then project back to
                # angular components via the spherical basis.
                x, e_phi, e_theta = self._spherical_basis(phi, theta)
                A = self._cross_matrix(x)                              # (N, 3, 3)
                u_rot = torch.einsum('nij,tj->tni', A, omega_star)     # (T, N, 3)
                d_right_rot = -(u_rot * e_phi).sum(dim=-1)             # (T, N)
                d_up_rot = -(u_rot * e_theta).sum(dim=-1)              # (T, N)
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

    def _sign_synapses(self):
        """
        Sign synapse counts in-place based on neurotransmitter type.
        
        Converts unsigned synapse counts to signed counts where:
        - Excitatory synapses (from neurons using excitatory_nt) are positive
        - Inhibitory synapses (from neurons using inhibitory_nt) are negative
        
        Modifies self.synapse_counts and self.synapse_counts_coo in-place.
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


@_register_model_class
class CurrentModel(NeuralModel):
    """Current-based neural model where synaptic input is modeled as current injection."""
    
    def __init__(self, *, init_new_model=True, **kwargs):
        # Call parent init (handles NT filtering and orphan removal)
        # All NeuralModel args (data + config) pass through via **kwargs
        super().__init__(**kwargs)
        
        # Sign the synapse counts based on neurotransmitter type
        # This converts unsigned counts to signed counts (excitatory = +, inhibitory = -)
        self._sign_synapses()
        
        # Save init inputs after super().__init__() to overwrite with this class's inputs
        self._save_init_inputs()
        
        # Initialize power vector for spectral norm calculation
        power_vec_np = self.rng.standard_normal(self.num_neurons)
        power_vec = torch.tensor(power_vec_np, device=self.device, dtype=self.dtype)
        power_vec = power_vec / (power_vec.norm() + 1e-12)
        self.register_buffer('_power_vec', power_vec)
        
        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))
        
        # --- CurrentModel-specific parameters ---
        persistence_init = 0.5
        stimulus_strength_init = 10.0
        offset_init = 0.0
        
        persistence_logit_init = logit(persistence_init)
        self.persistence_logit = torch.full((self.num_cell_types,), persistence_logit_init, dtype=self.dtype, device=self.device)
        
        stimulus_strength_log_init = np.log(stimulus_strength_init)
        self.stimulus_strength_log = torch.full((self.num_cell_types,), stimulus_strength_log_init, dtype=self.dtype, device=self.device)
        
        self.offset = torch.full((self.num_cell_types,), offset_init, dtype=self.dtype, device=self.device)
        
        # Initialize input strength per cell type based on the inverse of fan-in.
        # CSR row-sums come for free from the structure: we already have
        # crow_indices (cumulative nnz per row) and values, no need to
        # reduce a sparse tensor.
        synapses_in_per_neuron, synapses_out_per_neuron = self._csr_abs_row_col_sums(
            self.synapse_counts
        )

        synapses_in_per_cell_type = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            synapses_in_per_cell_type[cti] = synapses_in_per_neuron[ct_mask].mean().clamp_min(1.0)

        input_strength_init = synapses_in_per_cell_type.pow(-1 / 2)
        self.input_strength_log = torch.log(input_strength_init)
        synapses_out_per_cell_type = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            synapses_out_per_cell_type[cti] = synapses_out_per_neuron[ct_mask].mean().clamp_min(1.0)
        output_strength_init = synapses_out_per_cell_type.pow(-1 / 2)
        self.output_strength_log = torch.log(output_strength_init)
        
        # Convert to Parameters
        self.persistence_logit = torch.nn.Parameter(self.persistence_logit)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.offset = torch.nn.Parameter(self.offset)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.output_strength_log = torch.nn.Parameter(self.output_strength_log)
        
        # Add noise to all parameters
        self.add_noise_to_parameters(self.init_std)
        
        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()
                # self.find_steady_state()

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        """Calculate the maximum singular value (spectral norm) of the dynamics matrix."""
        A = self.synapse_counts
        A_T = A.t().to_sparse_csr()
        D_in = self.input_strength_log.exp()[self.neuron_to_type_index]
        D_out = self.output_strength_log.exp()[self.neuron_to_type_index]
        d = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        one_minus_d = 1.0 - d


        def apply_M(v):
            v_scaled = v * D_out
            y = torch.sparse.mm(A, v_scaled.unsqueeze(1)).squeeze(1)
            return y * D_in

        def apply_M_T(v):
            v_scaled = v * D_in
            y = torch.sparse.mm(A_T, v_scaled.unsqueeze(1)).squeeze(1)
            return y * D_out

        def apply_S(v):
            return d * v + one_minus_d * apply_M(v)

        def apply_S_T(v):
            return d * v + apply_M_T(one_minus_d * v)

        x = self._power_vec.clone()
        x = x / (x.norm() + eps)

        for _ in range(max_iter):
            y = apply_S(x)
            x = apply_S_T(y)
            x = x / (x.norm() + eps)

        y = apply_S(x)
        sigma = y.norm()

        self._power_vec.copy_(x.detach())

        return sigma

    def silence_neurons(self, neurons_to_silence, invert_silencing=False, silence_inputs=False):
        chosen = self._silencing_mask(neurons_to_silence, invert_silencing)
        if chosen is not None:
            with torch.no_grad():
                self.output_strength_log.data[chosen] = np.log(self.silence_val)
                if silence_inputs:
                    self.input_strength_log.data[chosen] = np.log(self.silence_val)

    def firing_rate(self, v):
        """Convert voltage to firing rate."""
        return self.nonlinearity(v)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        input_strength = self.input_strength_log.exp()[self.neuron_to_type_index]
        output_strength = self.output_strength_log.exp()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        offset_all_neurons = self.offset[self.neuron_to_type_index]

        v_scaled = self.nonlinearity(v_prev) * output_strength[None, :]
        z = torch.sparse.mm(self.synapse_counts, v_scaled.transpose(0, 1)).transpose(0, 1)
        z = z * input_strength[None, :]

        if stim_to_cell_mapping is None:
            z = z + u_t * stimulus_strength[None, :]
        else:
            cell_idx = stim_to_cell_mapping['neuron_index']
            stim_idx = stim_to_cell_mapping['stimulus_index']
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                z.index_add_(1, ci, u_t[:, si] * stimulus_strength[None, ci])

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (z + offset_all_neurons[None, :])

        return v_next


@_register_model_class
class CurrentModelFast(CurrentModel):
    """Inference-optimized variant of :class:`CurrentModel`.

    Produces identical outputs to ``CurrentModel.forward`` (up to quantization
    error when ``quantize_activations=True``) but is considerably faster at
    inference because:

    * Per-cell-type parameters are expanded to per-neuron tensors **once** via
      :meth:`prepare_for_inference`, so ``sigmoid``/``exp`` are not recomputed
      every timestep.
    * The ``(1 - persistence)`` factor is fused into the post-matmul scale,
      the stimulus scale, and the bias, removing several per-step multiplies.
    * The sparse synapse matmul backend is selectable via ``sparse_impl``
      (default ``'auto'``). On x86 / Apple Silicon the CSR path
      (``torch.sparse.mm`` against a CSR view) is fastest; on Linux ARM /
      Raspberry Pi a hand-rolled gather + ``index_add_`` (``'scatter'``) is
      ~2.7x faster than CSR because the MKL/Accelerate sparse kernel isn't
      available there.
    * ``forward`` can optionally be wrapped with :func:`torch.compile`
      (``enable_compile=True``). On CPU this is roughly break-even because
      ``torch.sparse.mm`` forces a graph break so Inductor can only fuse the
      cheap element-wise tail; the dispatch overhead cancels the fusion
      savings. On CUDA it's typically worth enabling.
    * All forward math runs under ``torch.no_grad`` — no autograd bookkeeping.

    Optional activation quantization (``quantize_activations=True``) makes
    the dominant sparse matrix–vector product integer-valued:

    * Synapse counts are stored as an int32 sparse tensor (CSR if the backend
      supports int sparse CSR mm, otherwise COO).
    * Pre-matmul activations are dynamically quantized each step with
      symmetric per-batch-row scaling. Bit depth is configurable via
      ``activation_bits`` (8 or 16; default 16). int8 is usually too lossy on
      trained models because ``fr * output_strength`` has a large dynamic
      range across cell types (a single outlier dictates the per-tensor scale
      for everyone else); int16 gives ~256× the resolution.
    * The matmul is int{8,16} × int32 → int32; the result is rescaled to
      float once for the small per-neuron post-scale / persistence update.
      The best available integer path is detected once at prepare time so the
      hot path is branch-free.
    * On current PyTorch CPU builds the integer sparse mm kernel is *less*
      optimized than MKL's float32 sparse kernel, so quantization currently
      does **not** provide a CPU speedup — it's primarily useful for targets
      with dedicated low-precision sparse support (accelerators, some GPUs).

    This class can be trained like ``CurrentModel``, or constructed from an
    already-trained ``CurrentModel`` via :meth:`from_trained`.
    """

    SPARSE_IMPLS = ('auto', 'csr', 'coo', 'scatter')

    def __init__(self, *, quantize_activations=False, activation_bits=16,
                 enable_compile=False, sparse_impl='auto',
                 init_new_model=True, **kwargs):
        super().__init__(init_new_model=init_new_model, **kwargs)

        if activation_bits not in (8, 16):
            raise ValueError(f"activation_bits must be 8 or 16, got {activation_bits}")
        if sparse_impl not in self.SPARSE_IMPLS:
            raise ValueError(
                f"sparse_impl must be one of {self.SPARSE_IMPLS}, got {sparse_impl!r}"
            )
        self.quantize_activations = bool(quantize_activations)
        self.activation_bits = int(activation_bits)
        self.enable_compile = bool(enable_compile)
        # ``sparse_impl`` selects the synapse matmul backend:
        #   'auto'    - pick at construction time based on architecture: 'csr'
        #               on x86 (MKL-backed sparse mm is well-tuned), 'scatter'
        #               on Linux ARM / Raspberry Pi (no MKL; gather + index_add_
        #               is 2-3x faster there than torch.sparse.mm). This is the
        #               default and the right choice for both deployment shapes.
        #   'csr'     - torch.sparse.mm with CSR layout (best on x86 with MKL).
        #   'coo'     - torch.sparse.mm with coalesced COO (matches CSR on ARM
        #               without the conversion).
        #   'scatter' - hand-rolled gather + index_add_ on the COO triplet.
        #               Bypasses torch.sparse.mm's dispatch entirely; on Linux
        #               ARM this can be ~2.7x faster than CSR.
        if sparse_impl == 'auto':
            sparse_impl = self._auto_select_sparse_impl()
        self.sparse_impl = sparse_impl
        self._inference_prepared = False
        self._compiled_forward = None

        self._save_init_inputs()

        self.prepare_for_inference()

    @staticmethod
    def _auto_select_sparse_impl():
        """Pick the best ``sparse_impl`` for the current OS + CPU.

        The sparse-mm kernel quality depends on which BLAS backend PyTorch
        is built against:

        * Linux/Windows/macOS x86_64 : MKL-backed ``torch.sparse.mm`` (CSR)
          is the fastest path.
        * macOS arm64 (Apple Silicon) : Accelerate-backed sparse mm is
          well-tuned; CSR wins by ~12% over scatter at typical nnz.
        * Linux ARM / aarch64 (Raspberry Pi, Jetson, AWS Graviton) :
          PyTorch's sparse mm fallback has no tuned backend; the hand-rolled
          scatter (``index_select`` + ``index_add_``) path is ~2.7x faster
          than CSR / COO on these targets.

        ``'auto'`` picks ``'scatter'`` only for Linux ARM (the case where it
        wins by a wide margin) and ``'csr'`` everywhere else.
        """
        machine = (platform.machine() or '').lower()
        system = (platform.system() or '').lower()
        is_arm = machine.startswith('arm') or machine.startswith('aarch')
        if system == 'linux' and is_arm:
            return 'scatter'
        return 'csr'

    @classmethod
    def from_trained(cls, trained_model, quantize_activations=False,
                     activation_bits=16, enable_compile=False,
                     sparse_impl='auto'):
        """Build a ``CurrentModelFast`` that mirrors an already-trained ``CurrentModel``.

        The trained model's constructor inputs and parameter tensors are copied
        over; the expensive ``set_sigma_max`` pass is skipped.
        """
        if not isinstance(trained_model, CurrentModel):
            raise TypeError(
                f"from_trained expects a CurrentModel instance, got {type(trained_model).__name__}"
            )

        inputs = dict(trained_model.init_inputs)
        inputs['init_new_model'] = False
        inputs['quantize_activations'] = quantize_activations
        inputs['activation_bits'] = activation_bits
        inputs['enable_compile'] = enable_compile
        inputs['sparse_impl'] = sparse_impl

        fast = cls(**inputs)
        fast.load_state_dict(trained_model.state_dict(), strict=False)
        fast.prepare_for_inference()
        fast.eval()
        return fast

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        # Re-fuse precomputed buffers from the freshly loaded parameters.
        self.prepare_for_inference()
        return result

    def to(self, *args, **kwargs):
        """Move parameters/buffers then rebuild sparse views on the new device.

        ``_A_csr``, ``_A_rows/_cols/_vals``, and ``_int_A`` are plain
        attributes, so ``nn.Module.to()`` would leave them behind on the old
        device and the synapse matmul would mix CUDA dense activations with
        a CPU sparse matrix. ``prepare_for_inference()`` rebuilds them from
        ``self.synapse_counts`` (which ``super().to()`` does move correctly).
        """
        result = super().to(*args, **kwargs)
        result.prepare_for_inference()
        return result

    @torch.no_grad()
    def prepare_for_inference(self):
        """Precompute per-neuron fused buffers from current parameter values.

        Safe to call repeatedly — used after parameter updates or state loads.
        """
        idx = self.neuron_to_type_index

        persistence = self.persistence_logit.sigmoid()[idx]
        input_strength = self.input_strength_log.exp()[idx]
        output_strength = self.output_strength_log.exp()[idx]
        stimulus_strength = self.stimulus_strength_log.exp()[idx]
        offset = self.offset[idx]

        one_minus_p = 1.0 - persistence

        fused = {
            # Used as: v_next += persistence * v_prev
            '_persistence_n': persistence.contiguous(),
            # Applied to pre-matmul activations (column-scale on nonlinearity(v_prev)).
            '_output_strength_n': output_strength.contiguous(),
            # Folds (1 - p) and input_strength into one per-row multiply on z.
            '_post_mm_scale_n': (one_minus_p * input_strength).contiguous(),
            # Folds (1 - p) * stimulus_strength into a single scale on u_t.
            '_stim_scale_n': (one_minus_p * stimulus_strength).contiguous(),
            # Folds (1 - p) * offset into a constant additive bias each step.
            '_bias_n': (one_minus_p * offset).contiguous(),
        }
        for name, tensor in fused.items():
            if name in self._buffers:
                self._buffers[name] = tensor
            else:
                self.register_buffer(name, tensor)

        # Build the synapse matmul state matching ``self.sparse_impl``. Sparse
        # tensors can't always live in ``_buffers`` across PyTorch versions,
        # so we hold them as plain attributes.
        impl = self.sparse_impl
        A_coal = self.synapse_counts.coalesce()
        if impl == 'csr':
            self._A_csr = self._to_csr_or_coo(A_coal)
            self._A_rows = self._A_cols = self._A_vals = None
        elif impl == 'coo':
            self._A_csr = A_coal
            self._A_rows = self._A_cols = self._A_vals = None
        elif impl == 'scatter':
            self._A_csr = None
            indices = A_coal.indices()
            self._A_rows = indices[0].contiguous()
            self._A_cols = indices[1].contiguous()
            self._A_vals = A_coal.values().contiguous()
        else:
            raise ValueError(f"unknown sparse_impl: {impl!r}")

        if self.quantize_activations:
            self._prepare_int_synapse_matrix()
            self._int_A, self._int_mm_ok = self._detect_int_mm_path()
        else:
            self._int_A = None
            self._int_mm_ok = False

        # Any rebuild of buffers / sparse tensors invalidates the compiled fn.
        self._compiled_forward = None
        self._inference_prepared = True

    @staticmethod
    def _to_csr_or_coo(A):
        """Return a CSR view of a sparse tensor if supported, else the coalesced COO."""
        A_coal = A.coalesce() if A.is_sparse else A
        try:
            return A_coal.to_sparse_csr()
        except (RuntimeError, NotImplementedError):
            return A_coal

    @torch.no_grad()
    def _prepare_int_synapse_matrix(self):
        A = self.synapse_counts.coalesce()
        int_values = A.values().round().to(torch.int32)
        self._synapse_counts_int = torch.sparse_coo_tensor(
            A.indices(), int_values, size=A.size(),
            dtype=torch.int32, device=A.device,
        ).coalesce()
        try:
            self._synapse_counts_int_csr = self._synapse_counts_int.to_sparse_csr()
        except (RuntimeError, NotImplementedError):
            self._synapse_counts_int_csr = None

    @torch.no_grad()
    def _detect_int_mm_path(self):
        """Pick the fastest available int sparse mm path (CSR > COO > float fallback).

        Returns (matrix, ok) where ok=True means int sparse mm works with the
        returned matrix, ok=False means the caller should use the float fallback.
        """
        probe = torch.zeros(
            self.num_neurons, 1, dtype=torch.int32, device=self.synapse_counts.device,
        )
        for A in (self._synapse_counts_int_csr, self._synapse_counts_int):
            if A is None:
                continue
            try:
                torch.sparse.mm(A, probe)
                return A, True
            except (RuntimeError, NotImplementedError):
                continue
        return None, False

    def _scatter_sparse_mm(self, v):
        """Compute ``A @ v.T`` (transposed back) via gather + index_add_.

        Mathematically identical to ``torch.sparse.mm(A, v.T).T`` for the
        synapse matrix ``A`` of shape ``(N, N)``: each non-zero
        ``(post=row, pre=col, weight=val)`` contributes
        ``z[:, row] += val * v[:, col]``. The advantage over
        ``torch.sparse.mm`` on Linux ARM is that ``index_select`` and
        ``index_add_`` map to tighter, more vectorized kernels than the
        sparse mm fallback path that lacks an MKL backend.

        Args:
            v: ``(B, N)`` activations to be multiplied by the synapse matrix.

        Returns:
            ``(B, N)`` post-synaptic contributions.
        """
        contrib = v.index_select(1, self._A_cols) * self._A_vals.unsqueeze(0)
        z = torch.zeros_like(v)
        z.index_add_(1, self._A_rows, contrib)
        return z

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        if not self._inference_prepared:
            self.prepare_for_inference()

        if self.enable_compile:
            if self._compiled_forward is None:
                try:
                    self._compiled_forward = torch.compile(
                        self._forward_impl, dynamic=False, fullgraph=False,
                    )
                except Exception:
                    self.enable_compile = False
                    self._compiled_forward = None
            if self._compiled_forward is not None:
                try:
                    return self._compiled_forward(v_prev, u_t, stim_to_cell_mapping)
                except Exception:
                    # Compiler tripped over something at runtime — drop back
                    # to eager and disable for future calls.
                    self._compiled_forward = None
                    self.enable_compile = False

        return self._forward_impl(v_prev, u_t, stim_to_cell_mapping)

    @torch.no_grad()
    def _forward_impl(self, v_prev, u_t, stim_to_cell_mapping=None):
        fr = self.nonlinearity(v_prev)
        pre = fr * self._output_strength_n  # (B, N)

        if self.quantize_activations:
            # Dynamic symmetric per-batch-row quantization of activations.
            # Per-row (not per-tensor) so different batch samples don't share a
            # single outlier-driven scale; still cheap (one amax per row).
            max_val = float((1 << (self.activation_bits - 1)) - 1)  # 127 or 32767
            row_scale = (pre.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                         / max_val)
            q_int32 = (pre / row_scale).round().clamp(-max_val, max_val).to(torch.int32)

            if self._int_mm_ok:
                z_int = torch.sparse.mm(self._int_A,
                                        q_int32.transpose(0, 1)).transpose(0, 1)
                z = z_int.to(pre.dtype) * row_scale
            elif self.sparse_impl == 'scatter':
                # Hand-rolled int-quant scatter path: still float storage
                # for the gather product, but exact integer multiplies.
                q_f = q_int32.to(pre.dtype)
                z = self._scatter_sparse_mm(q_f) * row_scale
            else:
                # Backend fallback: float sparse mm on integer-valued operands —
                # multiplies are still exact integer ops, just widened.
                q_f = q_int32.to(pre.dtype)
                z = torch.sparse.mm(self._A_csr,
                                    q_f.transpose(0, 1)).transpose(0, 1) * row_scale
        else:
            if self.sparse_impl == 'scatter':
                z = self._scatter_sparse_mm(pre)
            else:
                z = torch.sparse.mm(self._A_csr,
                                    pre.transpose(0, 1)).transpose(0, 1)

        z = z * self._post_mm_scale_n

        if stim_to_cell_mapping is None:
            z = z + u_t * self._stim_scale_n
        else:
            cell_idx = stim_to_cell_mapping['neuron_index']
            stim_idx = stim_to_cell_mapping['stimulus_index']
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                z.index_add_(1, ci, u_t[:, si] * self._stim_scale_n[ci])

        return self._persistence_n * v_prev + z + self._bias_n


@_register_model_class
class ConductanceModel(NeuralModel):
    """Conductance-based neural model where synaptic input is modeled as conductance changes."""
    
    def __init__(self, *, init_new_model=True, **kwargs):
        """Initialize a conductance-based neural model.
        
        Args:
            init_new_model: Whether to run initialization steps like finding steady state (default: True)
            **kwargs: All arguments passed to NeuralModel (synapse_counts_coo, root_ids,
                cell_types, neurotransmitters, column_indices, column_phi_theta,
                column_p_q, device, dtype, dt, seed, etc.)
        """
        # Call parent init (handles NT filtering, count_threshold filtering, and orphan removal)
        # All NeuralModel args (data + config) pass through via **kwargs
        super().__init__(**kwargs)
        
        # Save init inputs after super().__init__() to overwrite with this class's inputs
        self._save_init_inputs()
        
        # --- Split synapse_counts_coo into per-neurotransmitter sparse matrices ---
        # Use the filtered COO from parent class
        synapse_counts_coo_filtered = self.synapse_counts_coo
        dtype_torch = self.dtype
        
        # Get the neurotransmitter for each presynaptic neuron in the COO matrix (directly from per-neuron array)
        pre_neuron_indices = synapse_counts_coo_filtered[:, 0]
        pre_transmitters = self.transmitters[pre_neuron_indices]
        
        # Map from neuron index to cell type index (needed for tracking which cell types receive each transmitter)
        neuron_to_type_idx = self.neuron_to_type_index.cpu().numpy()
        
        # Build a sparse matrix for each neurotransmitter
        self.synapse_counts = []
        
        for ri, r in enumerate(self.unique_transmitters):
            mask = (pre_transmitters == r)
            coo_r = synapse_counts_coo_filtered[mask]
            
            rows = torch.tensor(coo_r[:, 1], device=self.device, dtype=torch.int64)
            cols = torch.tensor(coo_r[:, 0], device=self.device, dtype=torch.int64)
            values = torch.tensor(coo_r[:, 2], device=self.device, dtype=dtype_torch)
            
            indices = torch.stack([rows, cols], dim=0)
            sparse_mat = torch.sparse_coo_tensor(indices, values,
                                                  size=(self.num_neurons, self.num_neurons),
                                                  dtype=dtype_torch, device=self.device).coalesce().to_sparse_csr()
            self.synapse_counts.append(sparse_mat)

        self.synapse_counts_T = [A.t().to_sparse_csr() for A in self.synapse_counts]
        
        print(f"ConductanceModel: {self.num_transmitters} transmitters, {self.num_cell_types} cell types, {self.num_neurons} neurons")
        print(f"  Output params: {self.num_cell_types} (one per cell type)")
        print(f"  Input/RP params: {self.num_transmitters} x {self.num_cell_types} (per transmitter per cell type)")
        
        # --- Conductance-specific parameters ---
        # Reversal potentials in mV: acetylcholine is excitatory, glutamate/GABA are inhibitory
        rp_default = {'ACH': 10.0, 'GLUT': -70.0, 'GABA': -70.0}
        rp_leak_default = -50.0
        g_leak_log_default = np.log(1.0)
        # Membrane time constant tau = C / g_leak. With g_leak=1 and dt in
        # seconds, picking C = 0.04 s gives tau = 40 ms at rest, so the
        # low-pass cutoff (~8 Hz) is high enough to pass typical motion stimuli
        # (e.g. 4 Hz). The previous value (C=40) gave tau=40 s and wiped out
        # all temporal structure before it could propagate past L1.
        capacitance_inv_log_default = np.log(1.0 / 0.04)
        threshold_default = -50
        
        # Check that all neurotransmitters have defined reversal potentials
        unknown_transmitters = set(self.unique_transmitters) - set(rp_default.keys())
        if unknown_transmitters:
            raise ValueError(f"Unknown neurotransmitters: {unknown_transmitters}. "
                           f"Please add reversal potentials to default_rp for these transmitters.")
        
        # --- Reversal potential: per cell type per transmitter ---
        # Shape: (num_transmitters, num_cell_types)
        # Initialize based on transmitter type defaults
        rp_transmitter_init = torch.zeros(self.num_transmitters, self.num_cell_types, dtype=self.dtype, device=self.device)
        for ri, r in enumerate(self.unique_transmitters):
            rp_transmitter_init[ri, :] = rp_default[r]

        # HARDCODED OVERRIDE: make GABA look excitatory (like ACH) onto L1 and L2.
        # In the lamina, L1/L2 actually receive histaminergic (sign-inverted)
        # input from photoreceptors, so flipping GABA's reversal potential at
        # these targets is a quick way to test what happens when an otherwise
        # inhibitory transmitter is treated as depolarizing on these cells.
        if 'GABA' in self.unique_transmitters:
            gaba_idx = int(np.where(self.unique_transmitters == 'GABA')[0][0])
            for ct_name in ('L1', 'L2'):
                ct_match = np.where(self.unique_cell_types == ct_name)[0]
                if ct_match.size > 0:
                    ct_idx = int(ct_match[0])
                    rp_transmitter_init[gaba_idx, ct_idx] = rp_default['ACH']
                    print(f"  [override] GABA reversal potential on {ct_name} "
                          f"set to {rp_default['ACH']} mV (excitatory like ACH)")

        self.rp_transmitter = rp_transmitter_init
        
        # Leak reversal potential per cell type: (num_cell_types,)
        self.rp_leak = torch.full((self.num_cell_types,), rp_leak_default, dtype=self.dtype, device=self.device)
        
        # Leak conductance (log scale) per cell type: (num_cell_types,)
        self.g_leak_log = torch.full((self.num_cell_types,), g_leak_log_default, dtype=self.dtype, device=self.device)
        
        # Inverse capacitance (log scale) per cell type: (num_cell_types,)
        self.capacitance_inv_log = torch.full((self.num_cell_types,), capacitance_inv_log_default, dtype=self.dtype, device=self.device)
        
        # Threshold (firing threshold) - initialized to match leak reversal potential
        self.threshold = torch.full((self.num_cell_types,), threshold_default, dtype=self.dtype, device=self.device)
        
        # Stimulus strength (log scale) per cell type. With u_t ~ O(1) and
        # g_tot ~ g_leak at rest, this is the mV swing of v_inf per unit
        # stimulus, so ~30 puts L1 across most of its dynamic range.
        stimulus_strength_init = 30.0
        stimulus_strength_log_init = np.log(stimulus_strength_init)
        self.stimulus_strength_log = torch.full((self.num_cell_types,), stimulus_strength_log_init, dtype=self.dtype, device=self.device)
        
        # --- Input/output strength: same fan-in / fan-out scheme as CurrentModel ---
        # Use the *total* synapse count (summed across transmitters) per cell
        # type. input_strength is broadcast across the transmitter dim so its
        # shape stays (num_transmitters, num_cell_types) for the per-transmitter
        # forward pass; values are identical across transmitters at init.
        synapses_in_per_neuron = torch.zeros(self.num_neurons, dtype=self.dtype, device=self.device)
        synapses_out_per_neuron = torch.zeros(self.num_neurons, dtype=self.dtype, device=self.device)
        for s in self.synapse_counts:
            r, c = self._csr_abs_row_col_sums(s)
            synapses_in_per_neuron = synapses_in_per_neuron + r
            synapses_out_per_neuron = synapses_out_per_neuron + c
        
        in_log_per_ct = torch.zeros(self.num_cell_types, dtype=self.dtype, device=self.device)
        out_log_per_ct = torch.zeros(self.num_cell_types, dtype=self.dtype, device=self.device)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            fan_in = synapses_in_per_neuron[ct_mask].mean().clamp_min(1.0)
            fan_out = synapses_out_per_neuron[ct_mask].mean().clamp_min(1.0)
            in_log_per_ct[cti] = -0.5 * torch.log(fan_in)
            out_log_per_ct[cti] = -0.5 * torch.log(fan_out)
        
        self.input_strength_log = in_log_per_ct.unsqueeze(0).expand(self.num_transmitters, -1).clone()
        self.output_strength_log = out_log_per_ct
        
        # --- Convert to Parameters ---
        self.rp_transmitter = torch.nn.Parameter(self.rp_transmitter)
        self.rp_leak = torch.nn.Parameter(self.rp_leak)
        self.g_leak_log = torch.nn.Parameter(self.g_leak_log)
        self.capacitance_inv_log = torch.nn.Parameter(self.capacitance_inv_log)
        self.threshold = torch.nn.Parameter(self.threshold)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.output_strength_log = torch.nn.Parameter(self.output_strength_log)
        
        # Power vector for spectral-norm calculation (used by set_sigma_max).
        power_vec_np = self.rng.standard_normal(self.num_neurons)
        power_vec = torch.tensor(power_vec_np, device=self.device, dtype=self.dtype)
        power_vec = power_vec / (power_vec.norm() + 1e-12)
        self.register_buffer('_power_vec', power_vec)
        
        # Add noise to all parameters
        self.add_noise_to_parameters(self.init_std)
        
        # Initialize steady_state to rp_leak (better starting point than zeros)
        with torch.no_grad():
            rp_leak_all = self.rp_leak[self.neuron_to_type_index]
            self.steady_state.copy_(rp_leak_all)
        
        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def silence_neurons(self, neurons_to_silence, invert_silencing=False, silence_inputs=False):
        chosen = self._silencing_mask(neurons_to_silence, invert_silencing)
        if chosen is not None:
            with torch.no_grad():
                self.output_strength_log.data[chosen] = np.log(self.silence_val)
                if silence_inputs:
                    self.input_strength_log.data[:, chosen] = np.log(self.silence_val)

    def firing_rate(self, v):
        """Convert voltage to firing rate with threshold subtraction.
        
        Args:
            v: Voltage tensor with shape (..., num_neurons) where neuron dimension is last.
        """
        threshold_all = self.threshold[self.neuron_to_type_index]
        # threshold_all has shape (num_neurons,), which broadcasts against the last dim of v
        return self.nonlinearity(v - threshold_all)

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        """Spectral norm of the linearized one-step Jacobian at rest.

        Linearizes around v_0 = E_leak with the baseline g_tot = g_leak
        (no synaptic activity yet) and a HARDCODED nonlinearity slope of 1.

        The slope-1 assumption is deliberate: the actual derivative of ReLU
        at v_0 = threshold is 0 (we sit on the kink), which would zero out
        M_eff and make set_sigma_max do nothing. Treating f'(v_0) = 1
        instead bounds the worst-case (active) gain, which is what we want
        the spectral target to constrain.

        The recurrent operator that gets bounded is:

            M_eff = sum_r [(E_r - E_leak) / g_leak] * D_in_r * A_r * D_out

        and the full Jacobian is

            S = D_persist + (1 - D_persist) * M_eff,

        with D_persist = exp(-dt * g_leak * C_inv). The (E_r - E_leak)/g_leak
        weighting carries the sign of each transmitter (positive for ACH,
        negative for GABA/GLUT) so excitatory and inhibitory paths combine
        correctly through the power iteration.
        """
        D_out = self.output_strength_log.exp()[self.neuron_to_type_index]            # (N,)
        D_in = self.input_strength_log.exp()[:, self.neuron_to_type_index]           # (R, N)
        E_r = self.rp_transmitter[:, self.neuron_to_type_index]                      # (R, N)
        E_leak = self.rp_leak[self.neuron_to_type_index]                             # (N,)
        g_leak = self.g_leak_log.exp()[self.neuron_to_type_index]                    # (N,)
        C_inv = self.capacitance_inv_log.exp()[self.neuron_to_type_index]            # (N,)

        d = torch.exp(-self.dt * g_leak * C_inv)                                     # (N,)
        one_minus_d = 1.0 - d
        W_r = (E_r - E_leak[None, :]) / g_leak[None, :]                              # (R, N)

        def apply_M(v):
            v_scaled = v * D_out
            out = torch.zeros_like(v)
            for ri, A_r in enumerate(self.synapse_counts):
                y = torch.sparse.mm(A_r, v_scaled.unsqueeze(1)).squeeze(1)
                out = out + W_r[ri] * D_in[ri] * y
            return out

        def apply_M_T(v):
            out = torch.zeros_like(v)
            for ri in range(len(self.synapse_counts)):
                v_weighted = v * W_r[ri] * D_in[ri]
                y = torch.sparse.mm(self.synapse_counts_T[ri], v_weighted.unsqueeze(1)).squeeze(1)
                out = out + y
            return out * D_out

        def apply_S(v):
            return d * v + one_minus_d * apply_M(v)

        def apply_S_T(v):
            return d * v + apply_M_T(one_minus_d * v)

        x = self._power_vec.clone()
        x = x / (x.norm() + eps)

        for _ in range(max_iter):
            y = apply_S(x)
            x = apply_S_T(y)
            x = x / (x.norm() + eps)

        y = apply_S(x)
        sigma = y.norm()

        self._power_vec.copy_(x.detach())

        return sigma

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None, eps=1e-6):
        """
        Semi-implicit / exponential (frozen-conductance) update per step:

        dv/dt = -(g_tot/C) * v + (1/C) * (sum_r g_r E_r + g_leak E_leak + b u)

        => v_next = persistence * v_prev + (1-persistence) * v_inf
        persistence = exp(-dt * g_tot / C)
        v_inf = (sum_r g_r E_r + g_leak E_leak + b u) / g_tot
        """
        # --- Expand parameters to per-neuron tensors ---
        C_inv = self.capacitance_inv_log.exp()[self.neuron_to_type_index]          # 1/C

        E_leak = self.rp_leak[self.neuron_to_type_index]
        g_leak = self.g_leak_log.exp()[self.neuron_to_type_index]

        b = self.stimulus_strength_log.exp()[self.neuron_to_type_index]

        out_strength = self.output_strength_log.exp()[self.neuron_to_type_index]  # (N,)
        in_strength = self.input_strength_log.exp()[:, self.neuron_to_type_index] # (R,N)
        E_r_all = self.rp_transmitter[:, self.neuron_to_type_index]               # (R,N)

        # --- Compute synaptic conductances g_r(v_prev) ---
        firing_rate_scaled = out_strength[None, :] * self.firing_rate(v_prev)     # (B,N)

        B, N = v_prev.shape
        g_syn_sum = torch.zeros_like(v_prev)                                      # (B,N)
        num = (g_leak[None, :] * E_leak[None, :])                                 # (B,N)  numerator piece

        # sum_r g_r and sum_r g_r * E_r
        for ri, _ in enumerate(self.unique_transmitters):
            # sparse mm expects (N,N) @ (N,B) -> (N,B)
            g_r = torch.sparse.mm(
                self.synapse_counts[ri], firing_rate_scaled.transpose(0, 1)
            ).transpose(0, 1)                                                     # (B,N)

            g_r = in_strength[ri][None, :] * g_r                                  # (B,N)

            g_syn_sum = g_syn_sum + g_r
            num = num + g_r * E_r_all[ri][None, :]                                # (B,N)

        # total conductance
        g_tot = g_leak[None, :] + g_syn_sum                                       # (B,N)

        # --- Add stimulus term into numerator: + b*u (same sign as your Euler form) ---
        if stim_to_cell_mapping is None:
            num = num + b[None, :] * u_t                                          # (B,N)
        else:
            # u_t is (B, num_stimuli), map into neuron space sparsely
            cell_idx = stim_to_cell_mapping["neuron_index"]
            stim_idx = stim_to_cell_mapping["stimulus_index"]
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                # num[:, ci] += b[ci] * u_t[:, si]
                num.index_add_(1, ci, b[ci][None, :] * u_t[:, si])

        # --- Exponential step ---
        # a = exp(-dt * g_tot / C) = exp(-dt * (g_tot * C_inv))
        persistence = torch.exp(-self.dt * g_tot * C_inv)                                                # (B,N)

        v_inf = num / (g_tot + eps)                                                # (B,N)
        v_next = persistence * v_prev + (1.0 - persistence) * v_inf

        return v_next


class CellAnn(torch.nn.Module):
    """Per-neuron multi-layer network using sparse connectivity from the connectome.

    Models each neuron as a small feed-forward network with layer sizes
    N, N-1, ..., 2, 1.  The first layer uses sparse connectivity derived from
    the connectome; subsequent layers are dense per-neuron matmuls that don't
    mix across neurons.

    All weights and biases are shared per cell type.  The sparse first layer
    stores one weight (in log space) per *connected* (post_type, pre_type)
    pair per hidden unit.  At forward time these are exponentiated (ensuring
    positivity) and multiplied by the biological sign derived from the
    presynaptic neurotransmitter.  The dense layers store one weight matrix
    per cell type.

    Index buffers (``_sparse_idx``, ``_neuron_to_type``, ``_edge_to_pair``,
    ``_edge_sign``) are computed by :class:`CellAnnModel` and passed in.

    Args:
        num_neurons: Total number of neurons (X).
        num_cell_types: Number of unique cell types.
        num_type_pairs: Number of connected (post_type, pre_type) pairs.
        cell_ann_depth: N, the width of the first hidden layer.
        nonlinearity: Activation module applied between layers.
        sparse_idx: ``(2, num_edges)`` int64 index tensor for the sparse layer.
        neuron_to_type: ``(num_neurons,)`` int64 neuron → cell-type mapping.
        edge_to_pair: ``(num_edges,)`` int64 edge → type-pair mapping.
        edge_sign: ``(num_edges,)`` +1/−1 biological sign per edge.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(self, num_neurons, num_cell_types, num_type_pairs,
                 cell_ann_depth, nonlinearity,
                 sparse_idx, neuron_to_type, edge_to_pair, edge_sign,
                 device=None, dtype=None):
        super().__init__()
        self.num_neurons = num_neurons
        self.cell_ann_depth = cell_ann_depth
        self.nonlinearity = nonlinearity

        # Pre-computed index buffers (owned by CellAnnModel)
        self.register_buffer('_sparse_idx', sparse_idx)
        self.register_buffer('_neuron_to_type', neuron_to_type)
        self.register_buffer('_edge_to_pair', edge_to_pair)
        self.register_buffer('_edge_sign', edge_sign)

        # Sparse layer: learned in log space, one per connected type pair per hidden unit
        self.sparse_weights_log = torch.nn.Parameter(
            torch.zeros(num_type_pairs, cell_ann_depth, device=device, dtype=dtype)
        )
        self.sparse_bias = torch.nn.Parameter(
            torch.zeros(num_cell_types, cell_ann_depth, device=device, dtype=dtype)
        )

        # Dense layers: per cell type, sizes N→N-1→…→2→1
        self.dense_weights = torch.nn.ParameterList()
        self.dense_biases = torch.nn.ParameterList()
        for k in range(cell_ann_depth - 1):
            in_size = cell_ann_depth - k
            out_size = in_size - 1
            self.dense_weights.append(torch.nn.Parameter(
                torch.zeros(num_cell_types, in_size, out_size,
                            device=device, dtype=dtype)
            ))
            self.dense_biases.append(torch.nn.Parameter(
                torch.zeros(num_cell_types, out_size, device=device, dtype=dtype)
            ))

    def forward(self, v):
        """
        Args:
            v: ``(batch, num_neurons)`` voltage vector.
        Returns:
            ``(batch, num_neurons)`` output after the per-neuron ANN.
        """
        batch_size = v.shape[0]

        # Convert voltage to firing rate before synaptic transmission
        fr = self.nonlinearity(v)

        # Positive learned magnitude × biological sign → signed per-edge weights
        edge_weights = (self.sparse_weights_log[self._edge_to_pair].exp()
                        * self._edge_sign.unsqueeze(-1))

        # Sparse first layer: N separate sparse matmuls → (batch, X, N)
        h = torch.zeros(batch_size, self.num_neurons, self.cell_ann_depth,
                        device=v.device, dtype=v.dtype)
        for n in range(self.cell_ann_depth):
            sparse_mat = torch.sparse_coo_tensor(
                self._sparse_idx, edge_weights[:, n],
                size=(self.num_neurons, self.num_neurons),
                dtype=v.dtype, device=v.device,
            )
            h[:, :, n] = torch.sparse.mm(sparse_mat, fr.transpose(0, 1)).transpose(0, 1)

        h = h + self.sparse_bias[self._neuron_to_type]
        if self.cell_ann_depth > 1:
            h = self.nonlinearity(h)

        # Dense per-neuron layers (weights shared by cell type)
        for k in range(len(self.dense_weights)):
            w = self.dense_weights[k][self._neuron_to_type]
            b = self.dense_biases[k][self._neuron_to_type]
            h = torch.einsum('bxi,xio->bxo', h, w) + b
            if k < len(self.dense_weights) - 1:
                h = self.nonlinearity(h)

        return h.squeeze(-1)


@_register_model_class
class CellAnnModel(NeuralModel):
    """Neural model where each neuron is modeled as a small feed-forward ANN.

    Processes the connectome the same way as :class:`CurrentModel`: synapses
    are signed by neurotransmitter type (excitatory / inhibitory).  The
    connectome provides the sparsity pattern and biological sign for the first
    layer of the per-neuron network; all weight magnitudes are learned.

    Forward dynamics::

        v_{t+1} = p * v_t + (1 - p) * (cell_ann(v_t) + b + c * u_t)

    Args:
        cell_ann_depth: N, the width of the first hidden layer in the
            per-neuron ANN (layer sizes: N, N-1, ..., 2, 1).
        init_new_model: Whether to run post-init setup (default True).
        **kwargs: Forwarded to :class:`NeuralModel`.
    """

    def __init__(self, *, init_new_model=True, cell_ann_depth=1, **kwargs):
        super().__init__(**kwargs)

        # Sign synapse counts by neurotransmitter (same as CurrentModel)
        self._sign_synapses()

        self._save_init_inputs()

        # --- Extract sparsity pattern and biological sign ---
        sc_coo = self.synapse_counts.to_sparse_coo()
        sparse_indices = sc_coo.indices()
        edge_sign = torch.sign(sc_coo.values())

        # --- Compute type-pair mapping (only connected pairs get parameters) ---
        edge_post_type = self.neuron_to_type_index[sparse_indices[0]]
        edge_pre_type = self.neuron_to_type_index[sparse_indices[1]]
        pair_keys = edge_post_type * self.num_cell_types + edge_pre_type
        unique_keys, edge_to_pair = torch.unique(pair_keys, return_inverse=True)
        num_type_pairs = unique_keys.shape[0]

        # --- Create CellAnn (initializes its own parameters) ---
        self.cell_ann = CellAnn(
            num_neurons=self.num_neurons,
            num_cell_types=self.num_cell_types,
            num_type_pairs=num_type_pairs,
            cell_ann_depth=cell_ann_depth,
            nonlinearity=self.nonlinearity,
            sparse_idx=sparse_indices,
            neuron_to_type=self.neuron_to_type_index,
            edge_to_pair=edge_to_pair,
            edge_sign=edge_sign,
            device=self.device,
            dtype=self.dtype,
        )

        # --- CellAnnModel-specific parameters (per cell type) ---
        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        persistence_init = 0.5
        stimulus_strength_init = 10.0
        offset_init = 0.0

        self.persistence_logit = torch.nn.Parameter(
            torch.full((self.num_cell_types,), logit(persistence_init),
                       dtype=self.dtype, device=self.device)
        )
        self.stimulus_strength_log = torch.nn.Parameter(
            torch.full((self.num_cell_types,), np.log(stimulus_strength_init),
                       dtype=self.dtype, device=self.device)
        )
        self.offset = torch.nn.Parameter(
            torch.full((self.num_cell_types,), offset_init,
                       dtype=self.dtype, device=self.device)
        )

        # --- Init sparse first-layer weights to match CurrentModel's effective
        #     per-edge magnitude. ---
        # CurrentModel's per-edge weight at v=0 is
        #     count[i,j] * sign * fan_in_post^(-1/2) * fan_out_pre^(-1/2).
        # CellAnn's per-edge weight is  exp(W_pair[p,c]) * sign  with no count
        # term, so we fold the type-pair mean count into the init magnitude:
        #     W_pair[p,c] = log(mean_count[p,c])
        #                   - 0.5*log(fan_in_post[p]) - 0.5*log(fan_out_pre[c])
        # This keeps biologically strong pathways (e.g. L2->Tm1) hot at init so
        # signal propagates through the feed-forward path. set_sigma_max() then
        # rescales uniformly across pairs to hit the spectral target.
        synapses_in_per_neuron = sc_coo.abs().sum(dim=1).to_dense()
        synapses_out_per_neuron = sc_coo.abs().sum(dim=0).to_dense()
        synapses_in_per_cell_type = torch.zeros(
            self.num_cell_types, device=self.device, dtype=self.dtype)
        synapses_out_per_cell_type = torch.zeros(
            self.num_cell_types, device=self.device, dtype=self.dtype)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            synapses_in_per_cell_type[cti] = synapses_in_per_neuron[ct_mask].mean().clamp_min(1.0)
            synapses_out_per_cell_type[cti] = synapses_out_per_neuron[ct_mask].mean().clamp_min(1.0)

        # Mean |synapse count| per connected (post_type, pre_type) pair.
        edge_abs_counts = sc_coo.values().abs().to(self.dtype)
        pair_count_sum = torch.zeros(
            num_type_pairs, dtype=self.dtype, device=self.device)
        pair_edge_count = torch.zeros(
            num_type_pairs, dtype=self.dtype, device=self.device)
        pair_count_sum.index_add_(0, edge_to_pair, edge_abs_counts)
        pair_edge_count.index_add_(
            0, edge_to_pair, torch.ones_like(edge_abs_counts))
        pair_mean_count = (pair_count_sum / pair_edge_count.clamp_min(1.0)).clamp_min(1.0)

        pair_post_type = unique_keys // self.num_cell_types  # (num_type_pairs,)
        pair_pre_type = unique_keys % self.num_cell_types
        fan_in = synapses_in_per_cell_type[pair_post_type]
        fan_out = synapses_out_per_cell_type[pair_pre_type]
        sparse_weights_log_init = (
            torch.log(pair_mean_count)
            - 0.5 * torch.log(fan_in)
            - 0.5 * torch.log(fan_out)
        ).unsqueeze(-1)
        self.cell_ann.sparse_weights_log.data.copy_(
            sparse_weights_log_init.expand(-1, cell_ann_depth)
        )

        # --- Glorot init for the dense per-neuron layers (only used when depth>1) ---
        # std = sqrt(2 / (in + out)). Uses self.rng so init is reproducible
        # under the model seed, like the rest of the noise injection.
        for k, w in enumerate(self.cell_ann.dense_weights):
            in_size = cell_ann_depth - k
            out_size = in_size - 1
            std = (2.0 / (in_size + out_size)) ** 0.5
            w_np = self.rng.normal(0.0, std, w.shape)
            w.data.copy_(torch.tensor(w_np, dtype=w.dtype, device=w.device))

        # --- Power vector for spectral-norm calc (used by set_sigma_max) ---
        power_vec_np = self.rng.standard_normal(self.num_neurons)
        power_vec = torch.tensor(power_vec_np, device=self.device, dtype=self.dtype)
        power_vec = power_vec / (power_vec.norm() + 1e-12)
        self.register_buffer('_power_vec', power_vec)

        self.add_noise_to_parameters(self.init_std)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def _apply_sigma_scale(self, scale_factor):
        """Scale recurrent gain by shifting all sparse_weights_log uniformly.

        Shifting by log(scale_factor) multiplies every per-pair magnitude by
        scale_factor identically across hidden units, so relative structure is
        preserved while overall recurrent gain rescales.
        """
        self.cell_ann.sparse_weights_log.add_(torch.log(scale_factor))

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        """Spectral norm of the linearized one-step Jacobian.

        Linearizes assuming ``cell_ann_depth == 1`` and ``f'(v) = 1`` (same
        slope-1 convention as CurrentModel/ConductanceModel). When the actual
        depth is >1 the dense layers and inter-layer Softplus distort this,
        but we use it as the init-time spectral target anyway and trust that
        training picks up the slack -- only the n=0 sparse hidden unit's
        magnitudes feed the calculation, but ``_apply_sigma_scale`` shifts
        every column equally so the rescaling stays well-defined.
        """
        sparse_idx = self.cell_ann._sparse_idx
        edge_sign = self.cell_ann._edge_sign
        edge_to_pair = self.cell_ann._edge_to_pair

        # Effective signed weight per edge from the n=0 hidden unit.
        edge_w = self.cell_ann.sparse_weights_log[edge_to_pair, 0].exp() * edge_sign
        A = torch.sparse_coo_tensor(
            sparse_idx, edge_w,
            size=(self.num_neurons, self.num_neurons),
            dtype=self.dtype, device=self.device,
        ).coalesce().to_sparse_csr()
        A_T = A.t().to_sparse_csr()

        d = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        one_minus_d = 1.0 - d

        def apply_M(v):
            return torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)

        def apply_M_T(v):
            return torch.sparse.mm(A_T, v.unsqueeze(1)).squeeze(1)

        def apply_S(v):
            return d * v + one_minus_d * apply_M(v)

        def apply_S_T(v):
            return d * v + apply_M_T(one_minus_d * v)

        x = self._power_vec.clone()
        x = x / (x.norm() + eps)

        for _ in range(max_iter):
            y = apply_S(x)
            x = apply_S_T(y)
            x = x / (x.norm() + eps)

        y = apply_S(x)
        sigma = y.norm()

        self._power_vec.copy_(x.detach())

        return sigma

    def firing_rate(self, v):
        return self.nonlinearity(v)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        offset = self.offset[self.neuron_to_type_index]

        z = self.cell_ann(v_prev)

        stim = torch.zeros_like(v_prev)
        if stim_to_cell_mapping is None:
            stim = u_t * stimulus_strength[None, :]
        else:
            cell_idx = stim_to_cell_mapping['neuron_index']
            stim_idx = stim_to_cell_mapping['stimulus_index']
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                stim.index_add_(1, ci, u_t[:, si] * stimulus_strength[None, ci])

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (z + offset[None, :] + stim)

        return v_next


@_register_model_class
class HandTunedModel(NeuralModel):
    """Hand-tuned model with per-cell-type dynamics parameters.

    Implements Julia optic-lobe-dynamics-style Euler integration:

        dx/dt = -x/tau + f_type(omega * W_norm @ x) + stimulus + offset

    Discretized (forward Euler):

        x_new = x + dt * (-x/tau + f_type(h) + stimulus + offset)

    where h = omega * W_norm @ x, and W_norm is the signed synapse matrix
    with each neuron's incoming weights normalized to sum to 1 (in absolute
    value).  Entry W_norm[i, j] is the fraction of neuron i's total input
    contributed by neuron j (with sign from the presynaptic neurotransmitter).

    Per-cell-type parameters:
        tau               -> time constants (positive)
        offset            -> bias
        stimulus_strength -> stimulus gain (positive)

    Per-cell-type configuration (buffer, not learned):
        activation_type -> nonlinearity index (0=tanh, 1=relu, 2=softplus, 3=identity)

    Global parameter:
        omega            -> gain factor (positive)

    Constructor accepts override dicts keyed by cell-type name for convenient
    hand-tuning of tau, offset, and activation function per cell type.
    """

    ACTIVATION_NAMES = ['tanh', 'relu', 'softplus', 'identity','sigmoid']

    def __init__(self, *, init_new_model=True,
                 tau_overrides=None, offset_overrides=None,
                 activation_overrides=None,
                 synapse_amplification=None,
                 omega_init=1.0, tau_default=0.05,
                 stimulus_strength_init=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._save_init_inputs()

        # Snapshot the raw (unsigned, unnormalized) connectome so we can
        # re-derive the weight matrix when amplification factors change.
        self._raw_synapse_counts = self.synapse_counts.clone()
        self._sign_flips = []

        self._sign_synapses()
        self.activation_overrides = activation_overrides
        # Per-cell-type amplification applied before row-wise normalization.
        # A factor > 1 for a presynaptic cell type makes its synapses occupy
        # a larger share of each postsynaptic neuron's normalized input.
        self.register_buffer('_synapse_amplification',
                             torch.ones(self.num_cell_types, dtype=self.dtype, device=self.device))
        if synapse_amplification:
            for ct_name, factor in synapse_amplification.items():
                idx = self._cell_type_name_to_index(ct_name)
                if idx is not None:
                    self._synapse_amplification[idx] = factor

        # Sign-based amplification: scale negative (inhibitory) or positive
        # (excitatory) synapses based on their actual sign in the matrix,
        # which accounts for manual flip_sign_between_cell_types calls.
        self.register_buffer('_negative_amplification',
                             torch.tensor(1.0, dtype=self.dtype, device=self.device))
        self.register_buffer('_positive_amplification',
                             torch.tensor(1.0, dtype=self.dtype, device=self.device))

        self._normalize_synapse_counts()

        # --- Global gain (omega) ---
        self.omega = torch.nn.Parameter(
            torch.tensor(omega_init, dtype=self.dtype, device=self.device))

        # --- Per-cell-type tau (time constant) ---
        tau_values = torch.full((self.num_cell_types,), tau_default,
                                dtype=self.dtype, device=self.device)
        if tau_overrides:
            for ct_name, tau_val in tau_overrides.items():
                idx = self._cell_type_name_to_index(ct_name)
                if idx is not None:
                    tau_values[idx] = tau_val
        self.tau = torch.nn.Parameter(tau_values)

        # --- Per-cell-type offset (bias) ---
        offset_values = torch.zeros(self.num_cell_types,
                                    dtype=self.dtype, device=self.device)
        if offset_overrides:
            for ct_name, off_val in offset_overrides.items():
                idx = self._cell_type_name_to_index(ct_name)
                if idx is not None:
                    offset_values[idx] = off_val
        self.offset = torch.nn.Parameter(offset_values)

        # --- Per-cell-type activation function (buffer, not learned) ---
        act_indices = torch.zeros(self.num_cell_types,
                                  dtype=torch.int64, device=self.device)
        if activation_overrides:
            for ct_name, act_name in activation_overrides.items():
                idx = self._cell_type_name_to_index(ct_name)
                if idx is not None:
                    act_indices[idx] = self.ACTIVATION_NAMES.index(act_name)
        self.register_buffer('activation_type', act_indices)

        # --- Per-cell-type stimulus strength ---
        self.stimulus_strength = torch.nn.Parameter(
            torch.full((self.num_cell_types,), stimulus_strength_init,
                        dtype=self.dtype, device=self.device))

        #self.add_noise_to_parameters(self.init_std)

        with torch.no_grad():
            self.omega.clamp_(min=1e-6)
            self.tau.clamp_(min=1e-6)
            self.stimulus_strength.clamp_(min=1e-6)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _collect_extra_state(self):
        state = super()._collect_extra_state()
        state['_raw_synapse_counts'] = self._raw_synapse_counts
        state['_sign_flips'] = self._sign_flips
        return state

    def _restore_extra_state(self, state, device, dtype):
        super()._restore_extra_state(state, device, dtype)
        if '_raw_synapse_counts' in state:
            self._raw_synapse_counts = state['_raw_synapse_counts'].to(
                device=device, dtype=dtype)
        if '_sign_flips' in state:
            self._sign_flips = state['_sign_flips']

    # ------------------------------------------------------------------
    # Weight-matrix normalization
    # ------------------------------------------------------------------

    def _normalize_synapse_counts(self):
        """Normalize each neuron's incoming weights to sum to 1 (absolute value).

        After this, synapse_counts[i, j] is the signed fraction of neuron i's
        total input that comes from neuron j.  Matches the Julia code's
        ``WW = WW * spdiagm(1 ./ indegree)`` normalization.

        Amplification is applied in two stages before normalizing:

        1. **Per-cell-type** (``_synapse_amplification``): scales each synapse
           by its presynaptic cell type's factor.
        2. **Sign-based** (``_negative_amplification`` /
           ``_positive_amplification``): scales synapses by their actual sign
           in the matrix (after NT signing and manual flips).
        """
        sc = self.synapse_counts
        col_indices = sc.col_indices()
        values = sc.values()
        row_indices = self._csr_row_indices(sc)

        # Per-presynaptic-cell-type amplification
        if hasattr(self, '_synapse_amplification'):
            pre_type = self.neuron_to_type_index[col_indices]
            values = values * self._synapse_amplification[pre_type]

        # Sign-based amplification (uses actual sign after NT signing + flips)
        if hasattr(self, '_negative_amplification') and self._negative_amplification != 1.0:
            values = torch.where(values < 0,
                                 values * self._negative_amplification,
                                 values)
        if hasattr(self, '_positive_amplification') and self._positive_amplification != 1.0:
            values = torch.where(values > 0,
                                 values * self._positive_amplification,
                                 values)

        row_abs_sums = torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype)
        row_abs_sums.index_add_(0, row_indices, values.abs())
        row_abs_sums.clamp_(min=1.0)

        normalized_values = values / row_abs_sums[row_indices]
        self.synapse_counts = torch.sparse_csr_tensor(
            sc.crow_indices(), col_indices, normalized_values, size=sc.size(),
        )

    # ------------------------------------------------------------------
    # Cell-type lookup helpers
    # ------------------------------------------------------------------

    def _cell_type_name_to_index(self, name):
        """Return index into ``unique_cell_types`` for *name*, or ``None``."""
        matches = np.where(self.unique_cell_types == name)[0]
        if len(matches) == 0:
            warnings.warn(f"HandTunedModel: cell type '{name}' not found")
            return None
        return int(matches[0])

    # ------------------------------------------------------------------
    # Convenience setters for hand-tuning after construction
    # ------------------------------------------------------------------

    def set_tau(self, cell_type_name, value):
        assert value > 0, f"tau must be positive, got {value}"
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            with torch.no_grad():
                self.tau[idx] = value

    def set_offset(self, cell_type_name, value):
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            with torch.no_grad():
                self.offset[idx] = value

    def set_activation(self, cell_type_name, activation_name):
        """*activation_name* is one of
        ``ACTIVATION_NAMES`` ('tanh', 'relu', 'softplus', 'identity')."""
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            self.activation_type[idx] = self.ACTIVATION_NAMES.index(activation_name)
    
    def set_default_activation(self, activation_name):
        with torch.no_grad():
            # apply new default to all cells
            self.activation_type[:] = self.ACTIVATION_NAMES.index(activation_name)
            # reapply
            if self.activation_overrides:
                for ct_name, act_name in self.activation_overrides.items():
                    idx = self._cell_type_name_to_index(ct_name)
                    if idx is not None:
                        self.activation_type[idx] = self.ACTIVATION_NAMES.index(act_name)

    def set_omega(self, value):
        assert value > 0, f"omega must be positive, got {value}"
        with torch.no_grad():
            self.omega.fill_(value)

    def set_stimulus_strength(self, cell_type_name, value):
        assert value > 0, f"stimulus_strength must be positive, got {value}"
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            with torch.no_grad():
                self.stimulus_strength[idx] = value

    def set_synapse_amplification(self, cell_type_name, value, renormalize=True):
        """Set the synapse amplification factor for a presynaptic cell type.

        After updating, re-signs and re-normalizes the weight matrix from
        the raw connectome so the new factor takes effect, unless
        *renormalize* is ``False``.
        """
        assert value > 0, f"synapse_amplification must be positive, got {value}"
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            self._synapse_amplification[idx] = value
            if renormalize:
                self._renormalize_from_raw()

    def amplify_inhibitory(self, factor):
        """Amplify all functionally inhibitory (negative) synapses.

        Uses the actual sign of each synapse in the matrix (after NT-based
        signing *and* any manual ``flip_sign_between_cell_types`` calls), so
        connections that were flipped to act as inhibitory are included.
        """
        assert factor > 0, f"factor must be positive, got {factor}"
        self._negative_amplification.fill_(factor)
        self._renormalize_from_raw()

    def amplify_excitatory(self, factor):
        """Amplify all functionally excitatory (positive) synapses."""
        assert factor > 0, f"factor must be positive, got {factor}"
        self._positive_amplification.fill_(factor)
        self._renormalize_from_raw()

    def _renormalize_from_raw(self):
        """Re-apply sign, stored sign flips, and normalization from the
        original connectome."""
        self.synapse_counts = self._raw_synapse_counts.clone()
        self._sign_synapses()
        for pre, post, fac in self._sign_flips:
            self._apply_sign_flip(pre, post, fac)
        self._normalize_synapse_counts()

    def flip_sign_between_cell_types(
        self, presynaptic_cell_type, postsynaptic_cell_type, factor=-1.0
    ):
        """Scale synaptic weights for edges from *presynaptic_cell_type* → *postsynaptic_cell_type*.

        ``synapse_counts`` is stored with row = post, column = pre (see ``forward``).
        Default *factor* is ``-1`` (flip inhibitory/excitatory effect for that pair).
        Row-wise absolute normalization is unchanged, so you do not need to call
        ``_normalize_synapse_counts`` again after this.

        Also updates ``synapse_counts_coo`` to match the sparse tensor.
        The flip is recorded so that ``_renormalize_from_raw`` can replay it.
        """
        self._sign_flips.append((presynaptic_cell_type, postsynaptic_cell_type, factor))
        self._apply_sign_flip(presynaptic_cell_type, postsynaptic_cell_type, factor)

    def _apply_sign_flip(self, presynaptic_cell_type, postsynaptic_cell_type, factor):
        """Apply a single sign flip to the current synapse_counts matrix."""
        pre_ti = self._cell_type_name_to_index(presynaptic_cell_type)
        post_ti = self._cell_type_name_to_index(postsynaptic_cell_type)
        if pre_ti is None or post_ti is None:
            return

        sc = self.synapse_counts
        pre_n = sc.col_indices()
        post_n = self._csr_row_indices(sc)
        values = sc.values().clone()

        pre_types = self.neuron_to_type_index[pre_n]
        post_types = self.neuron_to_type_index[post_n]
        mask = (pre_types == pre_ti) & (post_types == post_ti)

        fac = torch.as_tensor(factor, device=values.device, dtype=values.dtype)
        values = torch.where(mask, values * fac, values)

        self.synapse_counts = torch.sparse_csr_tensor(
            sc.crow_indices(), pre_n, values, size=sc.size(),
        )

        self.synapse_counts_coo = np.column_stack([
            pre_n.cpu().numpy(),
            post_n.cpu().numpy(),
            values.detach().cpu().numpy(),
        ])

    # ------------------------------------------------------------------
    # Ablation
    # ------------------------------------------------------------------

    def ablate_cell_type(self, cell_type_name):
        """Remove a cell type from the network by zeroing all its connections.

        Zeroes every incoming and outgoing synapse for neurons of
        *cell_type_name* in both the current weight matrix and the raw
        connectome snapshot, so the ablation persists across
        ``_renormalize_from_raw`` calls.
        """
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is None:
            return

        neuron_mask = self.neuron_to_type_index == idx  # (num_neurons,)

        for sc_attr in ('synapse_counts', '_raw_synapse_counts'):
            sc = getattr(self, sc_attr)
            pre_n = sc.col_indices()
            post_n = self._csr_row_indices(sc)
            values = sc.values().clone()

            hit = neuron_mask[post_n] | neuron_mask[pre_n]
            values[hit] = 0.0

            setattr(self, sc_attr, torch.sparse_csr_tensor(
                sc.crow_indices(), pre_n, values, size=sc.size(),
            ))

        # Re-normalize so remaining neurons' weights still sum to 1
        self._renormalize_from_raw()
        print(f"Ablated cell type '{cell_type_name}' "
              f"({neuron_mask.sum().item()} neurons disconnected)")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_omega(self):
        return self.omega.item()

    def get_cell_type_params(self, cell_type_name):
        """Return a dict of current parameter values for *cell_type_name*."""
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is None:
            return None
        return {
            'tau': self.tau[idx].item(),
            'offset': self.offset[idx].item(),
            'activation': self.ACTIVATION_NAMES[self.activation_type[idx].item()],
            'stimulus_strength': self.stimulus_strength[idx].item(),
            'synapse_amplification': self._synapse_amplification[idx].item(),
        }

    # ------------------------------------------------------------------
    # Cell-type readout helpers
    # ------------------------------------------------------------------

    def _cell_type_column_mapping(self, cell_type_name):
        """Return (neuron_indices, column_indices) for *cell_type_name*."""
        mask = self.cell_types == cell_type_name
        neuron_idx = np.where(mask)[0]
        col_idx = self.column_indices[mask]
        valid = ~np.isnan(col_idx)
        return neuron_idx[valid].astype(int), col_idx[valid].astype(int)

    def get_cell_voltage(self, v, cell_type_name):
        """Extract raw (pre-activation) voltage for a cell type, by column.

        Args:
            v: ``(batch, num_neurons)`` or ``(num_neurons,)`` state vector.
            cell_type_name: e.g. ``'T4a'``.

        Returns:
            ``(batch, num_columns)`` or ``(num_columns,)`` tensor with NaN for
            columns that lack a neuron of this type.
        """
        squeeze = v.dim() == 1
        if squeeze:
            v = v.unsqueeze(0)
        neuron_idx, col_idx = self._cell_type_column_mapping(cell_type_name)
        out = torch.full((v.shape[0], self.num_columns), float('nan'),
                         device=v.device, dtype=v.dtype)
        if len(neuron_idx) > 0:
            ni = torch.tensor(neuron_idx, device=v.device, dtype=torch.int64)
            ci = torch.tensor(col_idx, device=v.device, dtype=torch.int64)
            out[:, ci] = v[:, ni]
        return out.squeeze(0) if squeeze else out

    # ------------------------------------------------------------------
    # Per-cell-type nonlinearity
    # ------------------------------------------------------------------

    def _apply_activation(self, h):
        """Apply per-cell-type activation to post-synaptic input *h*.

        Activation indices (must match ACTIVATION_NAMES):
            0=tanh  1=relu  2=softplus  3=identity  4=sigmoid
        """
        unique_acts = self.activation_type.unique()

        if len(unique_acts) == 1:
            act = unique_acts.item()
            if act == 0:
                return torch.tanh(h)
            if act == 1:
                return torch.relu(h)
            if act == 2:
                return torch.nn.functional.softplus(h)
            if act == 3:
                return h
            if act == 4:
                return torch.sigmoid(h)
            return h

        act_per_neuron = self.activation_type[self.neuron_to_type_index]
        result = torch.tanh(h)
        for act_idx in unique_acts.tolist():
            if act_idx == 0:
                continue
            mask = (act_per_neuron == act_idx).unsqueeze(0)
            if act_idx == 1:
                result = torch.where(mask, torch.relu(h), result)
            elif act_idx == 2:
                result = torch.where(mask, torch.nn.functional.softplus(h), result)
            elif act_idx == 3:
                result = torch.where(mask, h, result)
            elif act_idx == 4:
                result = torch.where(mask, torch.sigmoid(h), result)
        return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def firing_rate(self, v):
        """Apply per-cell-type activations (``activation_type`` / ``_apply_activation``).

        Same family as in ``forward`` (tanh / relu / softplus / identity), used by
        ``predict`` for readout.  *v* has shape ``(batch, num_neurons)``.
        """
        return self._apply_activation(v)

    def compute_synaptic_current(self, v):
        """Total synaptic current into each neuron: omega * W_norm @ f(v).

        Args:
            v: ``(batch, num_neurons)`` state vector.

        Returns:
            ``(batch, num_neurons)`` synaptic current tensor.
        """
        nonlin = self._apply_activation(v)
        h = torch.sparse.mm(self.synapse_counts, nonlin.transpose(0, 1)).transpose(0, 1)
        return self.omega * h

    def decompose_synaptic_input(self, v, target_cell_type):
        """Decompose synaptic input to *target_cell_type* by presynaptic type.

        For each presynaptic type P structurally connected to the target::

            contribution_P = omega * sum_{j in P} W_norm[i,j] * f(v_j)

        mapped to the target neuron's column.

        Args:
            v: ``(batch, num_neurons)`` state vector.
            target_cell_type: name of the postsynaptic cell type.

        Returns:
            dict mapping presynaptic type name to ``(batch, num_columns)``
            tensor.  Target columns have real values; others are NaN.
        """
        neuron_idx, col_idx = self._cell_type_column_mapping(target_cell_type)
        if len(neuron_idx) == 0:
            return {}

        ti = torch.tensor(neuron_idx, device=self.device, dtype=torch.int64)
        ci = torch.tensor(col_idx, device=self.device, dtype=torch.int64)
        B = v.shape[0]

        nonlin = self._apply_activation(v)

        sc = self.synapse_counts.to_sparse_coo()
        sc_idx = sc.indices()
        sc_val = sc.values()

        row_mask = torch.isin(sc_idx[0], ti)
        sub_rows = sc_idx[0, row_mask]
        sub_cols = sc_idx[1, row_mask]
        sub_vals = sc_val[row_mask]

        if sub_vals.numel() == 0:
            return {}

        neuron_to_col = torch.full((self.num_neurons,), -1,
                                   device=self.device, dtype=torch.int64)
        neuron_to_col[ti] = ci
        target_cols = neuron_to_col[sub_rows]

        pre_type_idx = self.neuron_to_type_index[sub_cols]

        result = {}
        for tidx in pre_type_idx.unique():
            mask = pre_type_idx == tidx
            w = sub_vals[mask]
            pre_n = sub_cols[mask]
            tgt_c = target_cols[mask]

            weighted = self.omega * w.unsqueeze(0) * nonlin[:, pre_n]

            out = torch.full((B, self.num_columns), float('nan'),
                             device=self.device, dtype=self.dtype)
            out[:, ci] = 0.0
            out.scatter_add_(1, tgt_c.unsqueeze(0).expand(B, -1), weighted)

            type_name = self.unique_cell_types[tidx.item()]
            result[type_name] = out

        return result

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        """One Euler step of the dynamics.

        x_new = x + dt * (-x/tau + f(omega * W_norm @ x) + stim + offset)
        """
        tau = self.tau[self.neuron_to_type_index]
        omega = self.omega
        stimulus_strength = self.stimulus_strength[self.neuron_to_type_index]
        offset_all = self.offset[self.neuron_to_type_index]

        # Stimulus drive
class ArbitraryModel(NeuralModel):
    """All-to-all model with learned cell-type-to-cell-type weight matrix.

    Like CurrentModel but without connectome sparsity or sign constraints.
    The effective connectivity A is a dense (num_neurons x num_neurons) matrix
    parameterized by a (num_cell_types x num_cell_types) weight matrix W,
    where A[i,j] = W[type(i), type(j)].

    Dynamics: v_{t+1} = p * v_t + (1-p) * (A @ g(v_t) + b + c*u_t)
    """

    def __init__(self, *, init_new_model=True, **kwargs):
        super().__init__(**kwargs)
        self._save_init_inputs()

        power_vec_np = self.rng.standard_normal(self.num_neurons)
        power_vec = torch.tensor(power_vec_np, device=self.device, dtype=self.dtype)
        power_vec = power_vec / (power_vec.norm() + 1e-12)
        self.register_buffer('_power_vec', power_vec)

        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        persistence_init = 0.5
        stimulus_strength_init = 1.0
        offset_init = 0.0

        self.persistence_logit = torch.nn.Parameter(
            torch.full((self.num_cell_types,), logit(persistence_init),
                       dtype=self.dtype, device=self.device)
        )
        self.stimulus_strength_log = torch.nn.Parameter(
            torch.full((self.num_cell_types,), np.log(stimulus_strength_init),
                       dtype=self.dtype, device=self.device)
        )
        self.offset = torch.nn.Parameter(
            torch.full((self.num_cell_types,), offset_init,
                       dtype=self.dtype, device=self.device)
        )

        # Cell-type-to-cell-type weight matrix (no sign or sparsity constraints)
        self.W = torch.nn.Parameter(
            torch.zeros(self.num_cell_types, self.num_cell_types,
                        dtype=self.dtype, device=self.device)
        )

        self.add_noise_to_parameters(self.init_std)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def _apply_A(self, v, transpose=False):
        """Efficiently compute A @ v where A[i,j] = W[type(i), type(j)].

        Aggregates v by pre-synaptic cell type, multiplies by W in type-space,
        then expands back to per-neuron. Avoids materializing the full
        (num_neurons x num_neurons) matrix.
        """
        if v.dim() == 1:
            idx = self.neuron_to_type_index
        else:
            idx = self.neuron_to_type_index.unsqueeze(0).expand_as(v)

        v_by_type = torch.zeros(*v.shape[:-1], self.num_cell_types,
                                device=v.device, dtype=v.dtype)
        v_by_type.scatter_add_(-1, idx, v)

        if transpose:
            z_by_type = v_by_type @ self.W
        else:
            z_by_type = v_by_type @ self.W.T

        return z_by_type[..., self.neuron_to_type_index]

    def _apply_sigma_scale(self, scale_factor):
        """Scale the recurrent gain by multiplying the cell-type weight matrix."""
        self.W.mul_(scale_factor)

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        """Calculate the maximum singular value (spectral norm) of the dynamics matrix."""
        d = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        one_minus_d = 1.0 - d

        def apply_M(v):
            return self._apply_A(v)

        def apply_M_T(v):
            return self._apply_A(v, transpose=True)

        def apply_S(v):
            return d * v + one_minus_d * apply_M(v)

        def apply_S_T(v):
            return d * v + apply_M_T(one_minus_d * v)

        x = self._power_vec.clone()
        x = x / (x.norm() + eps)

        for _ in range(max_iter):
            y = apply_S(x)
            x = apply_S_T(y)
            x = x / (x.norm() + eps)

        y = apply_S(x)
        sigma = y.norm()

        self._power_vec.copy_(x.detach())

        return sigma

    def firing_rate(self, v):
        return self.nonlinearity(v)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        offset = self.offset[self.neuron_to_type_index]

        fr = self.nonlinearity(v_prev)
        z = self._apply_A(fr)

        stim = torch.zeros_like(v_prev)
        if stim_to_cell_mapping is None:
            stim = u_t * stimulus_strength[None, :]
        else:
            cell_idx = stim_to_cell_mapping['neuron_index']
            stim_idx = stim_to_cell_mapping['stimulus_index']
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                stim.index_add_(1, ci, u_t[:, si] * stimulus_strength[None, ci])

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (z + offset[None, :] + stim)

        return v_next

