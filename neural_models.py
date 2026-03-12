from logging import warning
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


class MotionDecoder(torch.nn.Module):
    """N-layer hex-grid convolutional decoder: firing rates -> optic flow.

    Each entry in *num_channels* adds a ``NeuralModelConv`` layer.  All layers
    except the last are followed by BatchNorm + softplus.  The final layer's
    first two channels are divided by softplus of the third to produce 2-D
    optic flow per column.

    Args:
        num_channels: List of output channel counts, one per conv layer.
        conv_sizes: List of hex-grid diameters (odd ints), same length as
            *num_channels*.
        neurons_to_decode: Ordered list of cell-type names used as decoder
            input features.
        column_p_q: ``(num_columns, 2)`` p,q lattice coordinates.
        cell_types: Per-neuron cell type array (length *num_neurons*).
        column_indices: Per-neuron column index array (length *num_neurons*).
        num_neurons: Total neuron count in the model.
        num_columns: Total column count in the model.
        denom_eps: Clamping floor for the normalising denominator.
    """

    def __init__(self, num_channels, conv_sizes, neurons_to_decode,
                 column_p_q, cell_types, column_indices, num_neurons,
                 num_columns, denom_eps=1e-2, device=None, dtype=None):
        super().__init__()
        if len(num_channels) != len(conv_sizes):
            raise ValueError("num_channels and conv_sizes must have the same length.")
        self.denom_eps = denom_eps
        self.neurons_to_decode = list(neurons_to_decode)
        self.conv_sizes = list(conv_sizes)

        self._build_column_to_neuron_mapping(
            cell_types, column_indices, num_neurons, num_columns, device,
        )

        num_decoder_neuron_types = len(self.neurons_to_decode)
        in_features = num_decoder_neuron_types
        self.conv_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i, (channels, csz) in enumerate(zip(num_channels, conv_sizes)):
            self.conv_layers.append(NeuralModelConv(
                in_features=in_features,
                out_features=channels,
                conv_size=csz,
                column_p_q=column_p_q,
                device=device,
                dtype=dtype,
            ))
            if i < len(num_channels) - 1:
                self.batch_norms.append(
                    torch.nn.BatchNorm1d(channels).to(device=device, dtype=dtype)
                )
            in_features = channels

    # ------------------------------------------------------------------
    # Neuron-to-column mapping (built once at init)
    # ------------------------------------------------------------------

    def _build_column_to_neuron_mapping(self, cell_types, column_indices,
                                        num_neurons, num_columns, device):
        """Create ``column_to_decoder_neurons`` buffer.

        Shape ``(num_columns, num_decoder_neuron_types)`` with neuron indices;
        ``-1`` marks absent neuron types in a given column.
        """
        cell_types = np.asarray(cell_types)
        column_indices = np.asarray(column_indices, dtype=np.float64)
        cell_type_to_idx = {ct: i for i, ct in enumerate(self.neurons_to_decode)}
        num_types = len(self.neurons_to_decode)

        decoder_mask = np.zeros(num_neurons, dtype=bool)
        for ct in self.neurons_to_decode:
            decoder_mask |= (cell_types == ct)

        mapping = np.full((num_columns, num_types), -1, dtype=np.int64)
        for neuron_idx in np.where(decoder_mask)[0]:
            col = column_indices[neuron_idx]
            if np.isnan(col):
                continue
            col = int(col)
            type_idx = cell_type_to_idx.get(cell_types[neuron_idx])
            if type_idx is not None and mapping[col, type_idx] == -1:
                mapping[col, type_idx] = neuron_idx

        self.register_buffer(
            'column_to_decoder_neurons',
            torch.tensor(mapping, device=device, dtype=torch.int64),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _extract_column_features(self, v):
        """Map raw neuron firing rates to per-column features.

        Returns ``(batch, num_columns, num_decoder_neuron_types)`` with
        ``NaN`` for neuron types absent from a column.
        """
        valid = self.column_to_decoder_neurons >= 0
        safe = self.column_to_decoder_neurons.clamp(min=0)
        features = v[:, safe]
        return features.masked_fill(~valid.unsqueeze(0), float('nan'))

    def forward(self, v):
        """Decode optic flow from firing rates.

        Args:
            v: ``(stimuli, num_neurons)`` or ``(num_neurons,)``.

        Returns:
            Optic flow ``(stimuli, num_columns, 2)`` or ``(num_columns, 2)``.
        """
        if v.dim() == 1:
            v = v.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        x = self._extract_column_features(v)

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x.permute(0, 2, 1)).permute(0, 2, 1)
                x = torch.nn.functional.softplus(x)

        denom = torch.nn.functional.softplus(x[:, :, -1:].clone())
        denom = torch.clamp(denom, min=self.denom_eps)
        optic_flow = x[:, :, :-1] / denom

        if squeeze:
            optic_flow = optic_flow.squeeze(0)
        return optic_flow


class NeuralModel(torch.nn.Module, ABC):
    """Base class for neural models. Contains shared functionality for both current and conductance models."""
    
    def __init__(self, *, synapse_counts_coo, root_ids, cell_types, neurotransmitters, 
                 column_indices, column_phi_theta, column_p_q=None,
                 excitatory_nt=['ACH'], inhibitory_nt=['GLUT', 'GABA'], nonlinearity='Softplus', device='cpu', dtype='float32',
                 init_new_model=True, dt=0.01, seed=None, init_std=0.1, count_threshold=2, commit_hash=None,
                 decoder_type='none', use_input_conv=True, conv_size_x=7, input_pre_conv_size=5, num_temporal_modes=4, use_layer_norm=True):
        super().__init__()
        # Save constructor inputs for serialization (child classes will merge their own on top)
        self._save_init_inputs()
        self.commit_hash = commit_hash
        self.silence_val = 1e-8  # when silencing neurons this is the value we set their connections to
        self.init_std = init_std
        self.count_threshold = count_threshold
        self.excitatory_nt = excitatory_nt
        self.inhibitory_nt = inhibitory_nt
        self.decoder_type = decoder_type
        self.use_input_conv = use_input_conv
        self.decoder_denom_eps = 1e-2
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
        self.synapse_counts = torch.sparse_coo_tensor(indices, values, size=(self.num_neurons, self.num_neurons), dtype=self.dtype, device=self.device).coalesce()

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

        if self.decoder_type == 'conv_1':
            self._initialize_decoder(num_channels=[3], conv_sizes=[5])
        if self.decoder_type == 'conv_2':
            self._initialize_decoder()
        if self.use_input_conv:
            self._initialize_input_conv(conv_size_x=conv_size_x, input_pre_conv_size=input_pre_conv_size, num_temporal_modes=num_temporal_modes)


    @property
    def has_decoder(self):
        """Whether the model has a trained decoder (any type other than 'none')."""
        return self.decoder_type != 'none'

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

    def _initialize_decoder(self, num_channels=None, conv_sizes=None):
        """Create the ``MotionDecoder`` sub-module.

        All decoder-specific computation (neuron mapping, hex-grid neighbour
        lookup, conv layers) lives inside ``MotionDecoder`` and
        ``NeuralModelConv``.  This method simply passes the model's structural
        data through.
        """
        if num_channels is None:
            num_channels = [8, 3]
        if conv_sizes is None:
            conv_sizes = [5, 5]

        neurons_to_decode = [
            'T1', 'T2', 'T2a', 'T3',
            'T4a', 'T4b', 'T4c', 'T4d',
            'T5a', 'T5b', 'T5c', 'T5d',
            'Tm1', 'Tm2', 'Tm3', 'Tm4',
            'Tm5Y', 'Tm5a', 'Tm5b', 'Tm5c',
            'Tm9', 'Tm16', 'Tm20', 'Tm28', 'Tm30',
            'TmY3', 'TmY4', 'TmY9', 'TmY10', 'TmY15', 'TmY18', 'TmY5a'
        ]

        self.decoder = MotionDecoder(
            num_channels=num_channels,
            conv_sizes=conv_sizes,
            neurons_to_decode=neurons_to_decode,
            column_p_q=self.column_p_q,
            cell_types=self.cell_types,
            column_indices=self.column_indices,
            num_neurons=self.num_neurons,
            num_columns=self.num_columns,
            denom_eps=self.decoder_denom_eps,
            device=self.device,
            dtype=self.dtype,
        )

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
        # given some input, generate the outputs of the rnn
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
        for i in readout_keys:
            outputs['voltage'][i] = torch.full((num_batches, num_time, self.num_columns), float('nan'), device=self.device, dtype=self.dtype)
            outputs['firing_rate'][i] = torch.full((num_batches, num_time, self.num_columns), float('nan'), device=self.device, dtype=self.dtype)

        if self.has_decoder:
            outputs['decoder'] = torch.full((num_batches, num_time, self.num_columns, 2), float('nan'), device=self.device, dtype=self.dtype)

        for t in range(num_time):
            u_t = inputs[:, t, :]
            if self.use_input_conv:
                u_t, filter_state = self.apply_input_conv(u_t, filter_state)
            v = self(v, u_t, stim_to_cell_mapping)
            if self.use_layer_norm:
                v = self.layer_norm(v)
            fr = self.firing_rate(v)

            if self.has_decoder:
                outputs['decoder'][:, t, :, :] = self.decoder(fr)

            for i in readout_keys:
                ci = readout_dict[i]['column_indices']
                ni = readout_dict[i]['neuron_indices']
                # Ensure we only write to valid column positions
                valid = ci >= 0
                if torch.any(valid):
                    outputs['voltage'][i][:, t, ci[valid]] = v[:, ni[valid]]
                    outputs['firing_rate'][i][:, t, ci[valid]] = fr[:, ni[valid]]

        # Check for NaN predictions indicating unstable dynamics
        total_elements = 0
        nan_elements = 0
        for key in readout_keys:
            fr_tensor = outputs['firing_rate'][key]
            total_elements += fr_tensor.numel()
            nan_elements += torch.isnan(fr_tensor).sum().item()
        if total_elements > 0 and nan_elements / total_elements > 0.5:
            print(f'Warning: >50% of predictions are NaN ({nan_elements}/{total_elements}), potentially unstable dynamics')

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

        for key in ('voltage', 'firing_rate'):
            if key in outputs:
                right_outputs[key] = {k: v[:S] for k, v in outputs[key].items()}
                left_outputs[key] = {k: v[S:] for k, v in outputs[key].items()}

        if 'decoder' in outputs:
            right_outputs['decoder'] = outputs['decoder'][:S]
            left_decoder = outputs['decoder'][S:].clone()
            left_decoder[..., 0] *= -1
            left_outputs['decoder'] = left_decoder

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
            result.neuron_to_type_index = result.neuron_to_type_index.to(*args, **kwargs)
        # Update self.device if a device was specified
        device = None
        if args and isinstance(args[0], (torch.device, str)):
            device = torch.device(args[0])
        elif 'device' in kwargs:
            device = torch.device(kwargs['device'])
        if device is not None:
            result.device = device
        return result

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'inputs': self.init_inputs,
            'class_name': self.__class__.__name__,
            # store weights on CPU for portability (includes buffers like optic_flow_*_readout)
            'state_dict': {k: v.detach().cpu() for k, v in self.state_dict().items()},
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

    def silence_neurons(self, neurons_to_silence, invert_silencing=False):
        if len(neurons_to_silence) > 0:

            if not invert_silencing:
                print("\n--- WARNING silencing neurons", neurons_to_silence, '---\n')
                # set the input and output strength of these neurons to a small value
                chosen_neurons = np.zeros(self.num_cell_types) == 1
                for ns in neurons_to_silence:
                    chosen_neurons[self.unique_cell_types == ns] = True
            else:
                print("\n--- WARNING silencing everything but", neurons_to_silence, '---\n')
                # set the input and output strength of these neurons to a small value
                chosen_neurons = np.zeros(self.num_cell_types) == 0
                for ns in neurons_to_silence:
                    chosen_neurons[self.unique_cell_types == ns] = False

            with torch.no_grad():
                self.output_strength_log.data[chosen_neurons] = np.log(self.silence_val)
        else:
            print('No neurons selected for silencing')

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
        model = model_cls(**inputs)
        model.load_state_dict(data['state_dict'])

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

    def estimate_ego_motion_from_image(self, right_eye_inputs, left_eye_inputs,
                                       initial_state_right=None, initial_state_left=None,
                                       initial_filter_state_right=None, initial_filter_state_left=None,
                                       oob_columns=None):
        """Two-eye ego-motion pipeline.

        Returns:
            (rotations, translations, right_result, left_result) where the
            result dicts have the same structure as ``predict`` / ``predict_two_eyes``.
        """
        S = right_eye_inputs.shape[0]

        right_result, left_result = self.predict_two_eyes(
            right_eye_inputs, left_eye_inputs,
            init_state_right=initial_state_right, init_state_left=initial_state_left,
            init_filter_state_right=initial_filter_state_right,
            init_filter_state_left=initial_filter_state_left,
        )

        optic_flow_right = right_result['outputs']['decoder']
        optic_flow_left = left_result['outputs']['decoder']
        optic_flow = torch.cat((optic_flow_right, optic_flow_left), dim=2)

        # Generate column locations for both eyes
        left_column_locations = self.column_phi_theta.copy()
        left_column_locations[:, 0] = -left_column_locations[:, 0]
        column_locations = np.concatenate((self.column_phi_theta, left_column_locations), axis=0)

        phi_tensor = torch.tensor(column_locations[:, 0], device=optic_flow.device, dtype=optic_flow.dtype)
        theta_tensor = torch.tensor(column_locations[:, 1], device=optic_flow.device, dtype=optic_flow.dtype)

        valid_mask = ~torch.isnan(optic_flow[0, 0, :, 0]) & ~torch.isnan(optic_flow[0, 0, :, 1])
        if oob_columns is not None:
            valid_mask = valid_mask & ~oob_columns
        phi_valid = phi_tensor[valid_mask]
        theta_valid = theta_tensor[valid_mask]

        T_len = optic_flow.shape[1]
        d_right = optic_flow[:, :, valid_mask, 0].reshape(S * T_len, -1)
        d_up = optic_flow[:, :, valid_mask, 1].reshape(S * T_len, -1)

        rot_flat, trans_flat, _ = self.estimate_ego_motion_from_flow(
            phi_valid, theta_valid, d_right, d_up
        )
        rotations = rot_flat.reshape(S, T_len, 3)
        translations = trans_flat.reshape(S, T_len, 3)

        return rotations, translations, right_result, left_result

    def estimate_ego_motion_from_flow(self, phi, theta, d_right, d_up):
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
        Returns:
            omega_star: (3,) or (T, 3) estimated rotation rate [roll, pitch, yaw]
                in physical observer convention (positive omega_z = leftward rotation)
            T_star: (3,) or (T, 3) estimated observer velocity direction, scaled by
                RMS of translational flow
            Z_star: (N,) or (T, N) estimated depths (positive = in front of observer)
        """

        def cross_matrix(a):
            """
            a: (..., 3) tensor
            returns (..., 3, 3) skew-symmetric matrix [a]_x
            """
            ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
            O = torch.zeros_like(ax)
            return torch.stack([
                torch.stack([   O, -az,  ay], dim=-1),
                torch.stack([  az,   O, -ax], dim=-1),
                torch.stack([ -ay,  ax,   O], dim=-1)
            ], dim=-2)

        # Handle unbatched input: (N,) → (1, N)
        unbatched = d_right.dim() == 1
        if unbatched:
            d_right = d_right.unsqueeze(0)
            d_up = d_up.unsqueeze(0)

        T, N = d_right.shape

        # === Convert to 3D viewing directions (N, 3) — shared across time ===
        x = torch.stack((
            torch.sin(theta)*torch.cos(phi),
            torch.sin(theta)*torch.sin(phi),
            torch.cos(theta)
        ), dim=1)                              # (N, 3)

        # === Basis vectors of sphere — shared across time ===
        e_theta = torch.stack((
            torch.cos(theta)*torch.cos(phi),
            torch.cos(theta)*torch.sin(phi),
            -torch.sin(theta)
        ), dim=1)                              # (N, 3)

        e_phi = torch.stack((
            -torch.sin(phi),
            torch.cos(phi),
            torch.zeros_like(phi)
        ), dim=1)                              # (N, 3)

        # Reconstruct 3D flow from angular components
        # d_right is rightward on retina; rightward = -e_phi (since e_phi points left)
        # d_up is upward on retina; upward = -e_theta (since e_theta points down)
        u = -d_right[..., None] * e_phi - d_up[..., None] * e_theta  # (T, N, 3)

        # === Rotation estimation: solve u ≈ [x]_× Ω for each time step ===
        # Physical relationship: image motion u = -Ω × x = x × Ω = [x]_× Ω
        # Normal equations: (A^T A) Ω = A^T u  (differentiable via torch.linalg.solve)
        A = cross_matrix(x)                    # (N, 3, 3)
        A_flat = A.reshape(3*N, 3)             # (3N, 3)
        u_flat = u.reshape(T, 3*N).T           # (3N, T)

        AtA = A_flat.T @ A_flat                # (3, 3) — constant across time
        Atb = A_flat.T @ u_flat                # (3, T)
        omega_star = torch.linalg.solve(AtA, Atb).T  # (T, 3)

        # === De-rotated flow: v = u - u_rot where u_rot = [x]_× Ω ===
        v = u - torch.einsum('nij,tj->tni', A, omega_star)  # (T, N, 3)

        # === Measure translation signal strength (per time step) ===
        translation_magnitude = torch.sqrt((v**2).mean(dim=(-2, -1)))

        # === Projection matrices Π_i = I - x_i x_iᵀ — shared across time ===
        outer = x.unsqueeze(2) @ x.unsqueeze(1)   # (N, 3, 3)
        I = torch.eye(3, device=x.device).expand(N, 3, 3)
        P = I - outer                              # (N, 3, 3)

        # === Construct stacked C matrix: each block = [v_i]_× Π_i ===
        vx = cross_matrix(v)                   # (T, N, 3, 3)
        C_blocks = vx @ P                      # (T, N, 3, 3)
        C = C_blocks.reshape(T, 3*N, 3)        # (T, 3N, 3)

        # === Solve C T = 0 via smallest singular vector (batched SVD) ===
        _, _, Vh = torch.linalg.svd(C, full_matrices=False)  # Vh: (T, 3, 3)
        T_dir = Vh[:, -1]                                     # (T, 3)

        # === Disambiguate sign using depth positivity constraint ===
        # Physical flow: v = -(1/Z) P T_scene where T_scene = -V_observer
        # w = P @ T_dir. For positive depth: v·w > 0 ↔ T_dir ∝ T_scene
        w_i = torch.einsum('nij,tj->tni', P, T_dir)          # (T, N, 3)
        v_dot_w = (v * w_i).sum(dim=-1)                      # (T, N)
        w_norm_sq = (w_i * w_i).sum(dim=-1)                   # (T, N)
        weighted_sign = (v_dot_w * w_norm_sq).sum(dim=-1)     # (T,)

        sign_flip = torch.where(weighted_sign < 0,
                                torch.tensor(-1.0, device=x.device, dtype=d_right.dtype),
                                torch.tensor(1.0, device=x.device, dtype=d_right.dtype))
        T_dir = T_dir * sign_flip[:, None]
        v_dot_w = v_dot_w * sign_flip[:, None]

        # Scale T by translation magnitude and negate to convert from the SVD's
        # -V_observer convention to physical observer velocity convention.
        T_star = -T_dir * translation_magnitude[:, None]  # (T, 3)

        # Depth: Z = ||w||² / (v · w) with v · w > 0 after disambiguation
        Z_star = w_norm_sq / (v_dot_w + 1e-12)  # (T, N)

        # Squeeze back if input was unbatched
        if unbatched:
            omega_star = omega_star.squeeze(0)
            T_star = T_star.squeeze(0)
            Z_star = Z_star.squeeze(0)

        return omega_star, T_star, Z_star


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
        
        self.max_allowed_sigma = 0.90
        
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
        
        # Initialize strength per cell type based on the inverse of fan in / out
        synapses_in_per_neuron = self.synapse_counts.abs().sum(dim=1).to_dense()
        synapses_out_per_neuron = self.synapse_counts.abs().sum(dim=0).to_dense()
        
        synapses_in_per_cell_type = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype)
        synapses_out_per_cell_type = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            synapses_in_per_cell_type[cti] = synapses_in_per_neuron[ct_mask].mean().clamp_min(1.0)
            synapses_out_per_cell_type[cti] = synapses_out_per_neuron[ct_mask].mean().clamp_min(1.0)
        
        input_strength_init = synapses_in_per_cell_type.pow(-1 / 2)
        output_strength_init = synapses_out_per_cell_type.pow(-1 / 2)
        self.input_strength_log = torch.log(input_strength_init)
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

    def _sign_synapses(self):
        """
        Sign synapse counts in-place based on neurotransmitter type.
        
        Converts unsigned synapse counts to signed counts where:
        - Excitatory synapses (from neurons using excitatory_nt) are positive
        - Inhibitory synapses (from neurons using inhibitory_nt) are negative
        
        Modifies self.synapse_counts and self.synapse_counts_coo in-place.
        """
        # Get presynaptic neuron indices from the sparse tensor (column indices = presynaptic)
        indices = self.synapse_counts.coalesce().indices()
        
        # Get transmitter for each presynaptic neuron (directly from per-neuron array)
        pre_neuron_indices = indices[1].cpu().numpy()
        pre_transmitters = self.transmitters[pre_neuron_indices]
        
        # Compute sign based on transmitter type
        synapse_sign = np.zeros(len(pre_transmitters), dtype=np.float32)
        for nt in self.excitatory_nt:
            synapse_sign[pre_transmitters == nt] = 1
        for nt in self.inhibitory_nt:
            synapse_sign[pre_transmitters == nt] = -1
        
        # Warn about any synapses with unknown neurotransmitter (sign = 0)
        unknown_mask = synapse_sign == 0
        if np.any(unknown_mask):
            unknown_nts = np.unique(pre_transmitters[unknown_mask])
            warning(f"Unknown neurotransmitters {unknown_nts} - these synapses will be set to 0")
        
        # Apply sign to the sparse tensor values
        synapse_sign_tensor = torch.tensor(synapse_sign, device=self.device, dtype=self.dtype)
        signed_values = self.synapse_counts.values() * synapse_sign_tensor
        self.synapse_counts = torch.sparse_coo_tensor(
            indices, signed_values, size=self.synapse_counts.size(), 
            dtype=self.dtype, device=self.device
        ).coalesce()
        
        # Update the COO array from the sparse tensor for consistency
        # (coalesce may have combined duplicate entries, so sizes may differ)
        coalesced = self.synapse_counts.coalesce()
        self.synapse_counts_coo = np.column_stack([
            coalesced.indices()[1].cpu().numpy(),  # pre (columns)
            coalesced.indices()[0].cpu().numpy(),  # post (rows)
            coalesced.values().cpu().numpy()       # signed counts
        ])

    def set_sigma_max(self):
        """Iteratively scale output_strength to achieve target max sigma (spectral norm)."""
        with torch.no_grad():
            max_scaling_iters = 100
            rho_tolerance = 0.01
            damping_factor = 0.3

            for iter_idx in range(max_scaling_iters):
                sigma_max = self.calc_sigma_max()

                print(iter_idx, ':', sigma_max)
                if abs(sigma_max - self.max_allowed_sigma) < rho_tolerance:
                    break

                scale_factor = (self.max_allowed_sigma / sigma_max)**damping_factor
                self.output_strength_log.add_(torch.log(scale_factor))

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        """Calculate the maximum singular value (spectral norm) of the dynamics matrix."""
        A = self.synapse_counts.coalesce()
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
            y = torch.sparse.mm(A.transpose(0, 1), v_scaled.unsqueeze(1)).squeeze(1)
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

    def firing_rate(self, v):
        """Convert voltage to firing rate."""
        return self.nonlinearity(v)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        input_strength = self.input_strength_log.exp()[self.neuron_to_type_index]
        output_strength = self.output_strength_log.exp()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        offset_all_neurons = self.offset[self.neuron_to_type_index]

        # model is
        # z = D_in @ synapse_counts @ D_out @ nonlinearity(v_prev) + stimulus * stimulus_strength
        # where D_in scales post-synaptic (rows) and D_out scales pre-synaptic (columns)
        # v_next = persistence * v_prev + (1 - persistence) * (z + offset)
        # implementation is slightly more complicated to allow synapse_counts to be sparse and not run into
        # issues with passing gradients due to pytorch's limited sparse support.
        # we can't precompute the dynamics matrix because the broadcast operation of the input/output vectors doesn't
        # allow for passing the gradient when synapse_counts is sparse
        # we also have to separately apply the diagonal to x_prev to avoid converting it to a sparse diagonal which takes time

        v_scaled = self.nonlinearity(v_prev) * output_strength[None, :]
        z = torch.sparse.mm(self.synapse_counts, v_scaled.transpose(0, 1)).transpose(0, 1)
        z = z * input_strength[None, :]

        if stim_to_cell_mapping is None:
            z = z + u_t * stimulus_strength[None, :]
        else:
            # because stimulus can be very sparse (only stimulate 3 cell types)
            # I added this option to include a mapping so that we dont generate an enormous stimulus matrix
            cell_idx = stim_to_cell_mapping['neuron_index']  # neuron indices (int64)
            stim_idx = stim_to_cell_mapping['stimulus_index']  # stimulus indices (int64)
            # Guard against any negative indices
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                z.index_add_(1, ci, u_t[:, si] * stimulus_strength[None, ci])

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (z + offset_all_neurons[None, :])

        return v_next


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
                                                  dtype=dtype_torch, device=self.device).coalesce()
            self.synapse_counts.append(sparse_mat)
        
        print(f"ConductanceModel: {self.num_transmitters} transmitters, {self.num_cell_types} cell types, {self.num_neurons} neurons")
        print(f"  Output params: {self.num_cell_types} (one per cell type)")
        print(f"  Input/RP params: {self.num_transmitters} x {self.num_cell_types} (per transmitter per cell type)")
        
        # --- Conductance-specific parameters ---
        # Reversal potentials in mV: acetylcholine is excitatory, glutamate/GABA are inhibitory
        rp_default = {'ACH': 10.0, 'GLUT': -70.0, 'GABA': -70.0}
        rp_leak_default = -50.0
        g_leak_log_default = np.log(1.0)
        capacitance_inv_log_default = np.log(1.0 / 40.0)
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
        self.rp_transmitter = rp_transmitter_init
        
        # Leak reversal potential per cell type: (num_cell_types,)
        self.rp_leak = torch.full((self.num_cell_types,), rp_leak_default, dtype=self.dtype, device=self.device)
        
        # Leak conductance (log scale) per cell type: (num_cell_types,)
        self.g_leak_log = torch.full((self.num_cell_types,), g_leak_log_default, dtype=self.dtype, device=self.device)
        
        # Inverse capacitance (log scale) per cell type: (num_cell_types,)
        self.capacitance_inv_log = torch.full((self.num_cell_types,), capacitance_inv_log_default, dtype=self.dtype, device=self.device)
        
        # Threshold (firing threshold) - initialized to match leak reversal potential
        self.threshold = torch.full((self.num_cell_types,), threshold_default, dtype=self.dtype, device=self.device)
        
        # Stimulus strength (log scale) per cell type
        stimulus_strength_init = 10.0
        stimulus_strength_log_init = np.log(stimulus_strength_init)
        self.stimulus_strength_log = torch.full((self.num_cell_types,), stimulus_strength_log_init, dtype=self.dtype, device=self.device)
        
        # --- Input strength: per cell type per transmitter ---
        # Shape: (num_transmitters, num_cell_types)
        # Initialize based on fan-in: how many synapses each cell type receives from each transmitter
        self.input_strength_log = torch.zeros(self.num_transmitters, self.num_cell_types, dtype=self.dtype, device=self.device)
        
        for ri in range(self.num_transmitters):
            # Fan-in per neuron for this transmitter (sum over presynaptic, i.e., columns)
            # synapse_counts[ri] is (post, pre), so sum(dim=1) gives fan-in per post-synaptic neuron
            synapses_in_per_neuron = self.synapse_counts[ri].abs().sum(dim=1).to_dense()
            
            for cti, ct in enumerate(self.unique_cell_types):
                ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
                fan_in = synapses_in_per_neuron[ct_mask].mean().clamp_min(1.0)
                self.input_strength_log[ri, cti] = -0.5 * torch.log(fan_in)
        
        # --- Output strength: per cell type ---
        # Shape: (num_cell_types,)
        # Initialize based on fan-out for the transmitter that each cell type expresses
        self.output_strength_log = torch.zeros(self.num_cell_types, dtype=self.dtype, device=self.device)
        
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            # Find which transmitter this cell type expresses
            ct_transmitter = self.cell_type_transmitters[cti]
            ri = np.where(self.unique_transmitters == ct_transmitter)[0][0]
            
            # Fan-out per neuron for this transmitter (sum over postsynaptic, i.e., rows)
            # synapse_counts[ri] is (post, pre), so sum(dim=0) gives fan-out per pre-synaptic neuron
            synapses_out_per_neuron = self.synapse_counts[ri].abs().sum(dim=0).to_dense()
            fan_out = synapses_out_per_neuron[ct_mask].mean().clamp_min(1.0)
            self.output_strength_log[cti] = -0.5 * torch.log(fan_out)
        
        # --- Convert to Parameters ---
        self.rp_transmitter = torch.nn.Parameter(self.rp_transmitter)
        self.rp_leak = torch.nn.Parameter(self.rp_leak)
        self.g_leak_log = torch.nn.Parameter(self.g_leak_log)
        self.capacitance_inv_log = torch.nn.Parameter(self.capacitance_inv_log)
        self.threshold = torch.nn.Parameter(self.threshold)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.output_strength_log = torch.nn.Parameter(self.output_strength_log)
        
        # Add noise to all parameters
        self.add_noise_to_parameters(self.init_std)
        
        # Initialize steady_state to rp_leak (better starting point than zeros)
        with torch.no_grad():
            rp_leak_all = self.rp_leak[self.neuron_to_type_index]
            self.steady_state.copy_(rp_leak_all)
        
        # Find steady state
        # if init_new_model:
        #     with torch.no_grad():
        #         self.find_steady_state()

    def firing_rate(self, v):
        """Convert voltage to firing rate with threshold subtraction.
        
        Args:
            v: Voltage tensor with shape (..., num_neurons) where neuron dimension is last.
        """
        threshold_all = self.threshold[self.neuron_to_type_index]
        # threshold_all has shape (num_neurons,), which broadcasts against the last dim of v
        return self.nonlinearity(v - threshold_all)

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

