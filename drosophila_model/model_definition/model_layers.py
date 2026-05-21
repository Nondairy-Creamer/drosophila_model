"""Reusable ``torch.nn.Module`` building blocks used by the connectome RNNs.

Each layer here is self-contained — no dependencies on
:mod:`neural_models` — so they can be combined freely (e.g.
:class:`HexGridConv` is also used by :class:`engineering_models.ConvModel`).

Contents:
    - :class:`CellTypeLayerNorm` — layer norm with weight/bias shared per
      cell type instead of per neuron.
    - :class:`HexGridConv` — spatial conv over a hex-grid column lattice.
    - :class:`TemporalLRU` — complex diagonal linear recurrence (temporal filter).
"""

from __future__ import annotations

import numpy as np
import torch


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


class HexGridConv(torch.nn.Module):
    """Spatial convolution over a hex-grid column lattice.

    Designed for the Drosophila optic-lobe connectome where every column has
    six nearest neighbours arranged on a hexagonal lattice (indexed by
    ``column_p_q`` axial coordinates), but works for any hex lattice. Gathers
    per-column features from hex-grid neighbours, applies a learned weight
    tensor via ``einsum``, and adds a bias. Missing neighbours (boundary
    columns marked ``-1``) and ``NaN`` input features (e.g. absent neuron
    types at a column) are zeroed out before the contraction.
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
        axial = HexGridConv.hex_grid_axial_coords(conv_size).astype(np.float64)
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

        pq_offsets = HexGridConv.hex_grid_pq_offsets(conv_size)
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
