from logging import warning
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


class NeuralModel(torch.nn.Module, ABC):
    """Base class for neural models. Contains shared functionality for both current and conductance models."""
    
    def __init__(self, *, synapse_counts_coo, root_ids, cell_types, neurotransmitters, column_indices, column_phi_theta, 
                 excitatory_nt=['ACH'], inhibitory_nt=['GLUT', 'GABA'], nonlinearity='Softplus', device='cpu', dtype='float32',
                 init_new_model=True, dt=0.01, seed=None, init_std=0.1, count_threshold=2, commit_hash=None,
                 use_decoder=True, use_input_conv=True, conv_size_x=5, conv_size_t=20, use_layer_norm=True):
        super().__init__()
        # Save constructor inputs for serialization (child classes will merge their own on top)
        self._save_init_inputs()
        self.commit_hash = commit_hash
        self.silence_val = 1e-8  # when silencing neurons this is the value we set their connections to
        self.init_std = init_std
        self.count_threshold = count_threshold
        self.excitatory_nt = excitatory_nt
        self.inhibitory_nt = inhibitory_nt
        self.use_decoder = use_decoder
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
        
        # Ensure arrays are numpy with correct dtypes
        neurotransmitters = np.array(neurotransmitters)
        assert root_ids.dtype == np.int64, 'root_ids must have dtype np.int64'
        assert synapse_counts_coo.dtype == np.int64, 'root_ids must have dtype np.int64'
        
        # --- Filter out neurons in order ---
        # remove connections that have too few synapses
        # remove neurons with invalid neurotransmitters
        # remove connections to invalid neurons
        # remove neurons with no connections

        # --- Remove Small Connections ---
        # Apply synapse count threshold and make sure connections are to valid neurons
        above_threshold_connections = np.abs(synapse_counts_coo[:, 2]) >= count_threshold
        synapse_counts_coo = synapse_counts_coo[above_threshold_connections, :]

        # --- Filter out neurons ---
        # Filter neurons by valid neurotransmitters
        valid_nt_mask = np.isin(neurotransmitters, valid_neurotransmitters)

        # remove connections to invalid neurons
        valid_connections = np.isin(synapse_counts_coo[:, 0], root_ids[valid_nt_mask]) & np.isin(synapse_counts_coo[:, 1], root_ids[valid_nt_mask])
        synapse_counts_coo = synapse_counts_coo[valid_connections, :]
        
        # Filter if the neuron has no connections
        connected_root_ids = np.unique(synapse_counts_coo[:, 0:2])
        connected_neurons = np.isin(root_ids, connected_root_ids)
        valid_neurons = valid_nt_mask & connected_neurons
        
        # remove the invalid neurons based on neurotransmitter and if it has no connections
        root_ids = root_ids[valid_neurons]
        cell_types = cell_types[valid_neurons]
        neurotransmitters = neurotransmitters[valid_neurons]
        column_indices = column_indices[valid_neurons]

        # --- Convert root IDs to matrix indices (done once after all filtering) ---
        self.root_ids = root_ids
        import pandas as pd
        root_id_index = pd.Index(root_ids)
        pre_matrix_indices = root_id_index.get_indexer(synapse_counts_coo[:, 0])
        post_matrix_indices = root_id_index.get_indexer(synapse_counts_coo[:, 1])
        synapse_counts_coo = np.column_stack([pre_matrix_indices, post_matrix_indices, synapse_counts_coo[:, 2]])
        
        # --- Now set up the model with filtered data ---
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
        self.num_columns = np.unique(column_indices[~np.isnan(column_indices)]).size
        self.register_buffer('steady_state', torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype))

        # Layer norm flag (actual module created after neuron_to_type_index is built)
        self.use_layer_norm = use_layer_norm

        # readout from the neurons to optic flow in units of degrees per second
        # Cell lists stored as attributes, readout tensors as buffers (auto-saved in state_dict)
        self.optic_flow_horizontal_cells = ['T4a', 'T4b', 'T5a', 'T5b']
        self.optic_flow_vertical_cells = ['T4c', 'T4d', 'T5c', 'T5d']
        self.register_buffer('optic_flow_horizontal_readout', 
                             torch.tensor([1., -1., 1., -1., 0.], device=self.device, dtype=torch.double))
        self.register_buffer('optic_flow_vertical_readout',
                             torch.tensor([1., -1., 1., -1., 0.], device=self.device, dtype=torch.double))

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

        if self.use_decoder:
            # Initialize decoder
            self._initialize_decoder()
        if self.use_input_conv:
            # Initialize input convolution layer
            self._initialize_input_conv(conv_size_x=conv_size_x, conv_size_t=conv_size_t)

    def get_column_xyz(self, radius: float = 1.0) -> np.ndarray:
        """Return column locations projected to 3D Cartesian coordinates."""
        phi = self.column_phi_theta[:, 0]
        theta = self.column_phi_theta[:, 1]
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.stack((x, y, z), axis=1)

    def _initialize_input_conv(self, conv_size_x=5, conv_size_t=20):
        """Initialize input convolution parameters and precompute indices."""
        if conv_size_x % 2 == 0:
            raise ValueError("conv_size_x must be odd for centered spatial convolution.")
        if conv_size_t < 1:
            raise ValueError("conv_size_t must be >= 1 for temporal convolution.")

        self.input_conv_size_x = conv_size_x
        self.input_conv_size_t = conv_size_t
        axial_coords = self._hex_grid_axial_coords(conv_size_x)
        self.input_num_nearest_columns = axial_coords.shape[0]

        # Initialize spatial filter: 1 for inner 7 hexes (center + 6 neighbors), small value elsewhere
        # Store as log so exp() is always non-negative
        spatial_floor = 1e-4
        spatial_init = np.full((conv_size_x, conv_size_x), spatial_floor)
        radius = conv_size_x // 2
        for q, r in axial_coords:
            if max(abs(q), abs(r), abs(q + r)) <= 1:
                spatial_init[r + radius, q + radius] = 1.0
        # Initialize temporal filter as delta at current time (index 0=1, rest=0)
        temporal_init = np.zeros(conv_size_t)
        temporal_init[0] = 1.0

        self.input_conv_xy_log = torch.nn.Parameter(
            torch.tensor(np.log(spatial_init), device=self.device, dtype=self.dtype)
        )
        self.input_conv_t = torch.nn.Parameter(
            torch.tensor(temporal_init, device=self.device, dtype=self.dtype)
        )
        self.input_conv_bias = torch.nn.Parameter(
            torch.zeros((), device=self.device, dtype=self.dtype)
        )

        self.register_buffer(
            'input_conv_axial_coords',
            torch.tensor(axial_coords, device=self.device, dtype=torch.int64)
        )

        ordered_nearest_indices = self._compute_nearest_columns(conv_size_x)
        self.register_buffer(
            'input_nearest_column_indices',
            torch.tensor(ordered_nearest_indices, device=self.device, dtype=torch.int64)
        )

    def _initialize_decoder(self, conv_size=5):
        """Initialize decoder parameters and precompute neuron indices."""
        # conv_1: nearest columns on hex grid, 8 channels, weights per neuron
        # conv_2: nearest columns on hex grid, 3 channels
        self.conv_size = conv_size
        self.num_nearest_columns = self._hex_grid_points(conv_size).shape[0]
        self.channels_1 = 8
        self.channels_2 = 3
        decoder_init = 0.0
        
        # Set list of neurons to use for decoding (from black box in figure panel g)
        self.neurons_to_decode = [
            # T-shaped cells
            'T1', 'T2', 'T2a', 'T3',
            'T4a', 'T4b', 'T4c', 'T4d',
            'T5a', 'T5b', 'T5c', 'T5d',
            # Transmedullary cells (Tm)
            'Tm1', 'Tm2', 'Tm3', 'Tm4',
            'Tm5Y', 'Tm5a', 'Tm5b', 'Tm5c',
            'Tm9', 'Tm16', 'Tm20', 'Tm28', 'Tm30',
            # Transmedullary Y cells (TmY)
            'TmY3', 'TmY4', 'TmY9', 'TmY10', 'TmY15', 'TmY18', 'TmY5a'
        ]
        
        # Pre-compute neuron indices for decoder neurons
        decoder_neuron_mask_np = np.zeros(self.num_neurons, dtype=bool)
        for ct_name in self.neurons_to_decode:
            decoder_neuron_mask_np |= (self.cell_types == ct_name)
        decoder_neuron_indices_np = np.where(decoder_neuron_mask_np)[0]
        # Store as buffer so it's saved with the model
        self.register_buffer('decoder_neuron_indices', 
                           torch.tensor(decoder_neuron_indices_np, device=self.device, dtype=torch.int64))
        # Also store mask for efficient column lookups
        self.register_buffer('decoder_neuron_mask',
                           torch.tensor(decoder_neuron_mask_np, device=self.device, dtype=torch.bool))
        
        # Pre-compute column_indices as tensor (used in decode_motion)
        self.register_buffer('column_indices_tensor',
                           torch.from_numpy(self.column_indices).to(device=self.device))
        
        # Number of decoder neuron types (fixed size for the tensor)
        num_decoder_neuron_types = len(self.neurons_to_decode)
        
        # Initialize conv_1 with weights per neuron: (channels_1, num_nearest_columns, num_decoder_neuron_types)
        self.decoder_conv_1 = torch.nn.Parameter(
            torch.full(
                (self.channels_1, self.num_nearest_columns, num_decoder_neuron_types),
                decoder_init,
                device=self.device,
                dtype=self.dtype,
            )
        )
        self.decoder_conv_1_bias = torch.nn.Parameter(
            torch.zeros(self.channels_1, device=self.device, dtype=self.dtype)
        )
        self.decoder_conv_1_bn = torch.nn.BatchNorm1d(self.channels_1).to(device=self.device, dtype=self.dtype)
        
        # Pre-compute mapping from column to decoder neuron indices
        self._precompute_column_to_decoder_neurons(num_decoder_neuron_types)
        
        # Initialize conv_2 (operates on conv_1_output channels from nearest columns)
        # Shape: (channels_2, num_nearest_columns, channels_1)
        self.decoder_conv_2 = torch.nn.Parameter(
            torch.full(
                (self.channels_2, self.num_nearest_columns, self.channels_1),
                decoder_init,
                device=self.device,
                dtype=self.dtype,
            )
        )
        self.decoder_conv_2_bias = torch.nn.Parameter(
            torch.zeros(self.channels_2, device=self.device, dtype=self.dtype)
        )
        
        # Pre-compute nearest column indices for each column
        self._precompute_nearest_columns(self.conv_size)
        # Pre-compute decoder neuron indices for nearest columns (used in decode_motion)
        self._precompute_nearest_decoder_neurons()

    def _precompute_column_to_decoder_neurons(self, num_decoder_neuron_types):
        """Pre-compute mapping from column index to decoder neuron indices in that column.
        
        Creates a tensor (num_columns, num_decoder_neuron_types) with neuron indices.
        Each position corresponds to a specific neuron type (by index in self.neurons_to_decode),
        ensuring the same neuron type always maps to the same filter weight index.
        Missing neurons are padded with -1 (will be converted to NaN when indexing firing rates).
        
        Args:
            num_decoder_neuron_types: Number of decoder neuron types (from len(self.neurons_to_decode))
        """
        # Create mapping from cell type name to index in self.neurons_to_decode
        # This ensures each neuron type always maps to the same filter weight index
        cell_type_to_index = {ct_name: idx for idx, ct_name in enumerate(self.neurons_to_decode)}
        
        # Create tensor: (num_columns, num_decoder_neuron_types)
        # Use -1 as sentinel value (will be converted to NaN when indexing)
        column_to_decoder_neurons_tensor = torch.full(
            (self.num_columns, num_decoder_neuron_types),
            -1,
            device=self.device,
            dtype=torch.int64
        )
        
        for col_idx in range(self.num_columns):
            # Find neurons in this column
            in_column_mask = (self.column_indices_tensor == col_idx) & ~torch.isnan(self.column_indices_tensor)
            # Intersect with decoder neurons
            decoder_in_column_mask = in_column_mask & self.decoder_neuron_mask
            decoder_neuron_indices = torch.where(decoder_in_column_mask)[0]
            
            # Map each neuron to its type index
            for neuron_idx in decoder_neuron_indices:
                neuron_idx_np = neuron_idx.item()
                cell_type = self.cell_types[neuron_idx_np]
                # Find which type index this neuron corresponds to
                if cell_type in cell_type_to_index:
                    type_idx = cell_type_to_index[cell_type]
                    # Store neuron index at the correct type-index position
                    # If multiple neurons of same type exist in column, keep the first one
                    if column_to_decoder_neurons_tensor[col_idx, type_idx] == -1:
                        column_to_decoder_neurons_tensor[col_idx, type_idx] = neuron_idx
        
        # Store as buffer so it's saved with the model
        self.register_buffer('column_to_decoder_neurons', column_to_decoder_neurons_tensor)

    def _precompute_nearest_columns(self, conv_size):
        """Pre-compute nearest columns using a shift + stretch aligned hex grid.
        
        For each central column, take the nearest columns matching the hex grid count,
        fit a best rigid transform from a fixed flat-top hex grid to those points
        (including the central column), then order the nearest columns by the grid's
        row-major ordering to keep convolution weights consistently aligned across columns.
        """
        ordered_nearest_indices = self._compute_nearest_columns(conv_size)
        self.register_buffer(
            'nearest_column_indices',
            torch.tensor(ordered_nearest_indices, device=self.device, dtype=torch.int64)
        )

    @staticmethod
    def _hex_grid_axial_coords(conv_size: int) -> np.ndarray:
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
    def _hex_grid_points(conv_size: int) -> np.ndarray:
        """Return flat-top hex grid points centered at (0,0)."""
        axial = NeuralModel._hex_grid_axial_coords(conv_size).astype(np.float64)
        q = axial[:, 0]
        r = axial[:, 1]
        grid_x = 1.5 * q
        grid_y = np.sqrt(3.0) * (r + 0.5 * q)
        return np.stack([grid_x, grid_y], axis=1)

    def _compute_nearest_columns(self, conv_size):
        if conv_size % 2 == 0:
            raise ValueError("conv_size must be odd for centered convolution.")

        # (num_columns, 3) with [x, y, z]
        column_xyz = self.get_column_xyz()
        num_columns = column_xyz.shape[0]

        base_grid = self._hex_grid_points(conv_size)
        num_nearest_columns = base_grid.shape[0]
        
        # Compute pairwise Euclidean distances between xyz vectors
        xyz_i = column_xyz[:, None, :]  # (num_columns, 1, 3)
        xyz_j = column_xyz[None, :, :]  # (1, num_columns, 3)
        diff = xyz_i - xyz_j  # (num_columns, num_columns, 3)
        distances = np.sqrt(np.sum(diff**2, axis=2))  # (num_columns, num_columns)
        
        # For each column, find nearest columns (including itself)
        nearest_indices = np.argsort(distances, axis=1)[:, :num_nearest_columns]
        
        # Use median nearest-neighbor distance as global spacing
        nearest_neighbor_dist = np.partition(distances, 1, axis=1)[:, 1]
        grid_spacing = np.nanmedian(nearest_neighbor_dist)
        if not np.isfinite(grid_spacing) or grid_spacing <= 0:
            grid_spacing = 1.0
        grid_points = base_grid * grid_spacing
        
        center_grid_idx = np.where(
            (grid_points[:, 0] == 0) & (grid_points[:, 1] == 0)
        )[0][0]
        
        def _no_rotation_affine(src_points, dst_points):
            """Return per-axis scale and translation (no rotation) best fit."""
            # Solve independently per axis: dst = src * s + t
            A = np.column_stack([src_points[:, 0], np.ones(src_points.shape[0])])
            sx, tx = np.linalg.lstsq(A, dst_points[:, 0], rcond=None)[0]
            A = np.column_stack([src_points[:, 1], np.ones(src_points.shape[0])])
            sy, ty = np.linalg.lstsq(A, dst_points[:, 1], rcond=None)[0]
            scale = np.array([sx, sy])
            translation = np.array([tx, ty])
            return scale, translation
        
        def _hungarian_assignment(distance_matrix, fixed_pairs):
            """Optimal one-to-one assignment with optional fixed grid->actual pairs."""
            n = distance_matrix.shape[0]
            assignment = np.full(n, -1, dtype=np.int64)
            grid_used = np.zeros(n, dtype=bool)
            actual_used = np.zeros(n, dtype=bool)
            for grid_idx, actual_idx in fixed_pairs:
                assignment[grid_idx] = actual_idx
                grid_used[grid_idx] = True
                actual_used[actual_idx] = True
            
            remaining_grid = np.where(~grid_used)[0]
            remaining_actual = np.where(~actual_used)[0]
            if remaining_grid.size > 0:
                reduced_cost = distance_matrix[np.ix_(remaining_grid, remaining_actual)]
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(reduced_cost)
                assignment[remaining_grid[row_ind]] = remaining_actual[col_ind]
            return assignment
        
        ordered_nearest_indices = np.zeros_like(nearest_indices)
        max_icp_iters = 6
        
        for col_idx in range(num_columns):
            actual_points = self.column_phi_theta[nearest_indices[col_idx]].astype(np.float64)
            center_actual_pos = np.where(nearest_indices[col_idx] == col_idx)[0]
            if center_actual_pos.size == 0:
                center_actual_pos = 0
            else:
                center_actual_pos = int(center_actual_pos[0])
            
            scale = np.ones(2, dtype=np.float64)
            translation = actual_points.mean(axis=0) - grid_points.mean(axis=0)
            assignment = None
            
            for _ in range(max_icp_iters):
                transformed_grid = (grid_points * scale[None, :]) + translation
                dists = np.linalg.norm(
                    transformed_grid[:, None, :] - actual_points[None, :, :],
                    axis=2
                )
                assignment_new = _hungarian_assignment(
                    dists,
                    fixed_pairs=[(center_grid_idx, center_actual_pos)]
                )
                matched_actual = actual_points[assignment_new]
                scale_new, translation_new = _no_rotation_affine(grid_points, matched_actual)
                
                if assignment is not None and np.array_equal(assignment_new, assignment):
                    scale, translation = scale_new, translation_new
                    break
                assignment = assignment_new
                scale, translation = scale_new, translation_new
            
            ordered_nearest_indices[col_idx] = nearest_indices[col_idx][assignment]

        return ordered_nearest_indices

    def _apply_spatial_conv(self, inputs):
        """Apply spatial convolution only (no temporal filtering).

        Args:
            inputs: Tensor with shape (stimuli, time, columns).

        Returns:
            Convolved tensor with same shape.
        """
        # Gather spatial neighbors: (stimuli, time, columns, neighbors)
        spatial_windows = inputs[:, :, self.input_nearest_column_indices]

        radius = self.input_conv_size_x // 2
        q_idx = self.input_conv_axial_coords[:, 0] + radius
        r_idx = self.input_conv_axial_coords[:, 1] + radius
        spatial_kernel = self.input_conv_xy_log.exp()[r_idx, q_idx]
        spatial_kernel = spatial_kernel / spatial_kernel.sum()
        return torch.einsum('stcn,n->stc', spatial_windows, spatial_kernel)

    def _apply_temporal_conv(self, inputs):
        """Apply temporal convolution only (no spatial filtering).

        Args:
            inputs: Tensor with shape (stimuli, time, columns).

        Returns:
            Convolved tensor with same shape.
        """
        pad = self.input_conv_size_t - 1
        inputs_padded = torch.nn.functional.pad(inputs, (0, 0, pad, 0))
        time_windows = inputs_padded.unfold(1, self.input_conv_size_t, 1)
        # time_windows: (stimuli, time, columns, temporal_window)
        temporal_kernel = torch.flip(self.input_conv_t, dims=(0,))
        return torch.einsum('stcq,q->stc', time_windows, temporal_kernel)

    def apply_input_conv(self, inputs):
        """Apply local contrast normalization (spatial) then temporal filtering.

        Computes:
            local_mean = spatial_conv(input)
            input_centered = input - local_mean
            local_var = spatial_conv(input_centered ** 2)
            input_norm = input_centered / sqrt(local_var + eps)
            output = temporal_conv(-input_norm)
        """
        if not self.use_input_conv:
            return inputs

        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        elif inputs.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError("inputs must have shape (time, columns) or (stimuli, time, columns).")

        num_columns = self.input_nearest_column_indices.shape[0]
        if inputs.shape[2] != num_columns:
            raise ValueError(
                f"inputs has {inputs.shape[2]} columns, expected {num_columns}."
            )

        # Spatial contrast normalization
        local_mean = self._apply_spatial_conv(inputs)
        input_centered = inputs - local_mean
        local_var = self._apply_spatial_conv(input_centered ** 2)
        eps = 1e-3
        input_norm = input_centered / torch.sqrt(local_var + eps)

        # Negate and apply temporal kernel
        result = self._apply_temporal_conv(-input_norm)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _precompute_nearest_decoder_neurons(self):
        """Pre-compute decoder neuron indices for each column's nearest columns.
        
        Creates tensor (num_columns, num_nearest, num_decoder_neuron_types) with neuron indices.
        Missing neurons are marked as -1.
        """
        # nearest_column_indices: (num_columns, num_nearest)
        # column_to_decoder_neurons: (num_columns, num_decoder_neuron_types)
        nearest_decoder_neuron_indices = self.column_to_decoder_neurons[self.nearest_column_indices]
        self.register_buffer('nearest_decoder_neuron_indices', nearest_decoder_neuron_indices)

    def decode_motion(self, v):
        """
        Decode motion from firing rates using hand-implemented convolutions.
        
        Args:
            v: Firing rate tensor with shape (stimuli, num_neurons) or (num_neurons,)
                All operations broadcast over the first dimension (stimuli)
        
        Returns:
            motion: Decoded motion with shape (stimuli, num_columns, 2) or (num_columns, 2)
                First 2 channels divided by 3rd channel
        """
        # Handle both batched and unbatched inputs
        if v.dim() == 1:
            v = v.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # v has shape (stimuli, num_neurons)
        # All operations will broadcast over the first dimension (stimuli)
        stimuli_size = v.shape[0]
        num_columns = self.num_columns
        num_nearest = self.nearest_column_indices.shape[1]

        # Apply conv_1: For each column, operate on individual decoder neurons in N nearest columns
        # Collect firing rates from decoder neurons for each nearest column (precomputed mapping)
        # Shape: (stimuli, num_columns, num_nearest, num_decoder_neuron_types)
        decoder_indices = self.nearest_decoder_neuron_indices  # (num_columns, num_nearest, num_decoder_neuron_types)
        valid_mask = decoder_indices >= 0
        safe_indices = decoder_indices.clamp(min=0)
        nearest_decoder_voltages = v[:, safe_indices]
        nearest_decoder_voltages = nearest_decoder_voltages.masked_fill(~valid_mask.unsqueeze(0), float('nan'))
        
        # Apply conv_1 filter: (stimuli, num_columns, num_nearest, num_decoder_neuron_types) @ (channels_1, num_nearest, num_decoder_neuron_types)
        # decoder_conv_1: (channels_1, num_nearest, num_decoder_neuron_types)
        # nearest_decoder_voltages: (stimuli, num_columns, num_nearest, num_decoder_neuron_types)
        # Broadcasts over stimuli dimension
        # Handle NaNs by setting them to 0 before multiplication, then mask out contributions
        # Replace NaN with 0 for computation, but track which values were NaN
        valid_mask = ~torch.isnan(nearest_decoder_voltages)  # (stimuli, num_columns, num_nearest, num_decoder_neuron_types)
        nearest_decoder_voltages_clean = torch.where(
            valid_mask,
            nearest_decoder_voltages,
            torch.zeros_like(nearest_decoder_voltages)
        )
        
        # Single fused contraction: (stimuli, columns, nearest, types) x (channels_1, nearest, types)
        # -> (stimuli, columns, channels_1)
        conv_1_output = torch.einsum(
            'scnt,knt->sck',
            nearest_decoder_voltages_clean,
            self.decoder_conv_1
        )
        conv_1_output = conv_1_output + self.decoder_conv_1_bias.view(1, 1, -1)
        conv_1_output = self.decoder_conv_1_bn(conv_1_output.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Apply softplus activation after conv_1
        conv_1_output = torch.nn.functional.softplus(conv_1_output)
        
        # Apply conv_2: (stimuli, num_columns, channels_1) -> (stimuli, num_columns, channels_2)
        # conv_2 operates on conv_1_output from nearest columns, weighting the channels_1 channels
        # Get conv_1_output from nearest columns: (stimuli, num_columns, num_nearest, channels_1)
        # Advanced indexing: conv_1_output[:, nearest_indices] broadcasts over stimuli dimension
        nearest_conv_1_output = conv_1_output[:, self.nearest_column_indices]  # (stimuli, num_columns, num_nearest, channels_1)
        
        # Apply conv_2 filter: (stimuli, num_columns, num_nearest, channels_1) @ (channels_2, num_nearest, channels_1)
        # decoder_conv_2: (channels_2, num_nearest, channels_1)
        # nearest_conv_1_output: (stimuli, num_columns, num_nearest, channels_1)
        # Broadcasts over stimuli dimension
        # Single fused contraction: (stimuli, columns, nearest, channels_1) x (channels_2, nearest, channels_1)
        # -> (stimuli, columns, channels_2)
        conv_2_output = torch.einsum(
            'scnf,knf->sck',
            nearest_conv_1_output,
            self.decoder_conv_2
        )
        conv_2_output = conv_2_output + self.decoder_conv_2_bias.view(1, 1, -1)
        
        # Output: first 2 channels divided by 3rd channel
        # Avoid near-zero division to stabilize early training
        denominator = conv_2_output[:, :, 2:3].clone()  # (stimuli, num_columns, 1)
        denominator = torch.nn.functional.softplus(denominator)
        denominator = torch.clamp(denominator, min=self.decoder_denom_eps)
        optic_flow = conv_2_output[:, :, :2] / denominator  # (stimuli, num_columns, 2)
        
        if squeeze_output:
            optic_flow = optic_flow.squeeze(0)
        
        return optic_flow

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
                init_state=None):
        # given some input, generate the outputs of the rnn
        num_batches, num_time, num_neurons = inputs.shape

        if self.use_input_conv:
            inputs = self.apply_input_conv(inputs)

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

        # Initialize decoder output if decoder is enabled (only once, outside the loop)
        if self.use_decoder:
            outputs['decoder'] = torch.full((num_batches, num_time, self.num_columns, 2), float('nan'), device=self.device, dtype=self.dtype)

        for t in range(num_time):
            v = self(v, inputs[:, t, :], stim_to_cell_mapping)
            if self.use_layer_norm:
                v = self.layer_norm(v)
            # Compute firing_rate on full v (has neuron dimension for proper threshold broadcasting)
            fr = self.firing_rate(v)

            if self.use_decoder:
                outputs['decoder'][:, t, :, :] = self.decode_motion(fr)

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

        return outputs, v.detach(), {'input_mapping': stim_to_cell_mapping, 'output_mapping': readout_dict}

    def find_steady_state(self, run_time=100, max_iter=25):
        print("Finding steady state ...")
        inputs = torch.zeros((1, run_time, self.num_columns), device=self.device, dtype=self.dtype)
        # Start from current steady state buffer
        initial_state = self.steady_state.unsqueeze(0)
        initial_ave = initial_state.abs().mean()

        for i in range(max_iter):
            final_state = self.predict(inputs, init_state=initial_state)[1]

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

    def calculate_optic_flow(self, inputs, input_neurons=['L1', 'L2', 'L3'], eye='right', initial_state=None):
        # The eye is assumed to be bilaterally symmetric and is trained on the right eye
        # To estimate the response of the left eye, flip the inputs in space, calculate optic flow
        # then flip the optic flow estimate to get the flow estimate in a consistant basis

        # inputs should be given with the form (stimuli, time, columns) where columns correspond to
        # ommatidia at the locations in self.column_phi_theta
        # if eye='left' then the stimulus should be generated such that it corresponds to locations with
        # self.column_phi_theta but with phi = -phi (flipped over the mid point of the fly)

        horizontal_cells = self.optic_flow_horizontal_cells
        vertical_cells = self.optic_flow_vertical_cells

        outputs, final_state = self.predict(inputs, input_neurons=input_neurons, 
                               output_neurons=horizontal_cells+vertical_cells, init_state=initial_state)[0:2]

        horizontal = torch.zeros_like(outputs['firing_rate'][horizontal_cells[0]]) + self.optic_flow_horizontal_readout[-1]
        vertical = torch.zeros_like(outputs['firing_rate'][vertical_cells[0]]) + self.optic_flow_vertical_readout[-1]

        for hi, h in enumerate(horizontal_cells):
            horizontal += self.optic_flow_horizontal_readout[hi] * outputs['firing_rate'][h]

        for vi, v in enumerate(vertical_cells):
            vertical += self.optic_flow_vertical_readout[vi] * outputs['firing_rate'][v]

        if eye == 'left':
            horizontal = -horizontal

        optic_flow = torch.stack((horizontal, vertical), axis=-1)
        
        return optic_flow, final_state

    def fit_optic_flow_readout(self, inputs, target):
        outputs = self.predict(inputs, input_neurons=['L1', 'L2', 'L3'], 
                               output_neurons=self.optic_flow_horizontal_cells+self.optic_flow_vertical_cells, 
                               init_state=None)[0]

        horizontal_target = target[:, :, :, 0].reshape(-1)
        vertical_target = target[:, :, :, 1].reshape(-1)
        horizontal_list = []
        vertical_list = []

        for h in self.optic_flow_horizontal_cells:
            horizontal_list.append(outputs['firing_rate'][h].reshape(-1))
        for v in self.optic_flow_vertical_cells:
            vertical_list.append(outputs['firing_rate'][v].reshape(-1))

        horizontal_list.append(torch.ones_like(horizontal_list[0]))
        vertical_list.append(torch.ones_like(vertical_list[0]))

        horizontal_rates = torch.stack(horizontal_list, axis=-1)
        vertical_rates = torch.stack(vertical_list, axis=-1)

        # get the indices where there are no nans and keep the non nan values
        horizontal_valid_inds = ~torch.any(torch.isnan(horizontal_rates), axis=1) & ~torch.isnan(horizontal_target)
        horizontal_rates = horizontal_rates[horizontal_valid_inds, :]
        horizontal_target = horizontal_target[horizontal_valid_inds]

        vertical_valid_inds = ~torch.any(torch.isnan(vertical_rates), axis=1) & ~torch.isnan(vertical_target)
        vertical_rates = vertical_rates[vertical_valid_inds, :]
        vertical_target = vertical_target[vertical_valid_inds]

        # find the best fit between the rates and the target
        self.optic_flow_horizontal_readout = torch.linalg.lstsq(horizontal_rates.double(), horizontal_target.double())[0]
        self.optic_flow_vertical_readout = torch.linalg.lstsq(vertical_rates.double(), vertical_target.double())[0]

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
        path = Path(path)
        if path.is_dir():
            path = path / "model.pt"
        data = torch.load(path, map_location='cpu', weights_only=False)
        inputs = dict(data['inputs'])

        # Determine target device up front so we construct directly on it
        # (avoids expensive CPU construction + .to(device) round-trip)
        if device is not None:
            target_device = device
        else:
            local_config = du.load_config()
            target_device = local_config['model']['device']
        inputs['device'] = target_device

        if dtype is not None:
            inputs['dtype'] = dtype

        # Choose the correct class based on the saved class name.
        # Fall back to the class this was called on for older checkpoints.
        class_name = data.get('class_name')
        model_cls = cls
        if class_name is not None:
            model_cls = MODEL_CLASS_REGISTRY.get(class_name, cls)
            if model_cls is cls and cls is NeuralModel:
                raise ValueError(
                    f"Unknown model class '{class_name}' in checkpoint. "
                    "Register it in MODEL_CLASS_REGISTRY."
                )

        # construct directly from saved (or overridden) inputs
        inputs['init_new_model'] = False
        model = model_cls(**inputs)

        state_dict = data['state_dict']
        # Migrate old parameter names to _log variants
        state_dict = dict(state_dict)
        if 'input_conv_x' in state_dict and 'input_conv_y' in state_dict:
            input_conv_xy = torch.outer(state_dict.pop('input_conv_y'), state_dict.pop('input_conv_x'))
            state_dict['input_conv_xy_log'] = torch.log(input_conv_xy.clamp(min=1e-8))
        if 'input_conv_xy' in state_dict and 'input_conv_xy_log' not in state_dict:
            state_dict['input_conv_xy_log'] = torch.log(state_dict.pop('input_conv_xy').clamp(min=1e-8))
        if 'input_conv_t_log' in state_dict and 'input_conv_t' not in state_dict:
            state_dict['input_conv_t'] = state_dict.pop('input_conv_t_log').exp()
        if 'layer_norm.weight' in state_dict and 'layer_norm.weight_log' not in state_dict:
            state_dict['layer_norm.weight_log'] = torch.log(state_dict.pop('layer_norm.weight').clamp(min=1e-8))

        model.load_state_dict(state_dict, strict=False)

        # Optionally freeze all parameters to disable gradients
        if freeze_params:
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()

        return model

    def estimate_ego_motion_from_image(self, right_eye_inputs, left_eye_inputs, initial_state_right=None, initial_state_left=None):
        # right_eye_inputs: (stimuli, time, columns) where columns correspond to
        # ommatidia at the locations in self.column_phi_theta
        # left_eye_inputs: (stimuli, time, columns) where columns correspond to
        # ommatidia at the locations in self.column_phi_theta but with phi = -phi (flipped over the mid point of the fly)
        # initial_state: None or tuple (right_state, left_state) from a previous call

        # Run predictions for each eye to get decoder optic flow and final states
        outputs_right, final_state_right = self.predict(right_eye_inputs, init_state=initial_state_right)[0:2]
        outputs_left, final_state_left = self.predict(left_eye_inputs, init_state=initial_state_left)[0:2]

        # Get decoder outputs: (stimuli, time, columns, 2) where last dim is (d_right, d_up)
        # Flip horizontal (d_right) for left eye to put it in a consistent coordinate frame
        optic_flow_right = outputs_right['decoder']
        optic_flow_left = outputs_left['decoder'].clone()
        optic_flow_left[:, :, :, 0] = -optic_flow_left[:, :, :, 0]

        # Concatenate along column dimension
        optic_flow = torch.cat((optic_flow_right, optic_flow_left), dim=2)
        # optic_flow shape: (stimuli, time, columns*2, 2)

        # Generate column locations for both eyes
        left_column_locations = self.column_phi_theta.copy()
        left_column_locations[:, 0] = -left_column_locations[:, 0]
        column_locations = np.concatenate((self.column_phi_theta, left_column_locations), axis=0)

        phi_tensor = torch.tensor(column_locations[:, 0], device=optic_flow.device, dtype=optic_flow.dtype)
        theta_tensor = torch.tensor(column_locations[:, 1], device=optic_flow.device, dtype=optic_flow.dtype)

        # Decoder NaN columns are constant across time, so one mask suffices.
        valid_mask = ~torch.isnan(optic_flow[0, 0, :, 0]) & ~torch.isnan(optic_flow[0, 0, :, 1])
        phi_valid = phi_tensor[valid_mask]
        theta_valid = theta_tensor[valid_mask]

        num_stimuli = optic_flow.shape[0]
        rotations = []
        translations = []

        for s in range(num_stimuli):
            d_right = optic_flow[s, :, valid_mask, 0]  # (time, valid_columns)
            d_up = optic_flow[s, :, valid_mask, 1]

            rotation, translation, _ = self.estimate_ego_motion_from_flow(
                phi_valid, theta_valid, d_right, d_up
            )
            rotations.append(rotation)       # (time, 3)
            translations.append(translation) # (time, 3)

        rotations = torch.stack(rotations)       # (stimuli, time, 3)
        translations = torch.stack(translations)  # (stimuli, time, 3)

        return rotations, translations, final_state_right, final_state_left

    @staticmethod
    def estimate_ego_motion_from_flow(phi, theta, d_right, d_up):
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
            omega_star: (3,) or (T, 3) estimated rotation vector [roll, pitch, yaw]
                Positive omega_z = leftward rotation (world moves right on retina)
            T_star: (3,) or (T, 3) estimated translation direction (up to sign ambiguity)
            Z_star: (N,) or (T, N) estimated depths (positive when T_star sign is correct)
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
        # So Ω = lstsq([x]_×, u)
        A = cross_matrix(x)                    # (N, 3, 3)
        A_flat = A.reshape(3*N, 3)             # (3N, 3)
        u_flat = u.reshape(T, 3*N).T           # (3N, T) — multiple RHS for lstsq

        omega_star = torch.linalg.lstsq(A_flat, u_flat).solution.T  # (T, 3)

        # === De-rotated flow: v = u - u_rot where u_rot = [x]_× Ω ===
        v = u - torch.einsum('nij,tj->tni', A, omega_star)  # (T, N, 3)

        # === Measure translation signal strength (per time step) ===
        # RMS of de-rotated flow - if near zero, there's no translation
        translation_magnitude = torch.sqrt((v**2).mean(dim=(-2, -1)))  # (T,)

        # === Projection matrices Π_i = I - x_i x_iᵀ — shared across time ===
        outer = x.unsqueeze(2) @ x.unsqueeze(1)   # (N, 3, 3)
        I = torch.eye(3, device=x.device).expand(N, 3, 3)
        P = I - outer                              # (N, 3, 3)

        # === Construct stacked C matrix: each block = [v_i]_× Π_i ===
        vx = cross_matrix(v)                   # (T, N, 3, 3)
        C_blocks = vx @ P                      # (T, N, 3, 3)
        C = C_blocks.reshape(T, 3*N, 3)        # (T, 3N, 3)

        # === Solve C T = 0 via smallest singular vector (batched SVD) ===
        # For pure rotation the de-rotated flow is ~0, making C ill-conditioned.
        # Only run SVD on time steps with meaningful translational signal.
        trans_eps = 1e-6
        has_translation = translation_magnitude > trans_eps  # (T,)

        T_dir = torch.zeros(T, 3, device=x.device, dtype=d_right.dtype)
        v_dot_w = torch.zeros(T, N, device=x.device, dtype=d_right.dtype)
        w_norm_sq = torch.zeros(T, N, device=x.device, dtype=d_right.dtype)

        if has_translation.any():
            C_valid = C[has_translation]                         # (T_valid, 3N, 3)
            _, _, Vh = torch.linalg.svd(C_valid, full_matrices=False)  # Vh: (T_valid, 3, 3)
            T_dir_valid = Vh[:, -1]                              # (T_valid, 3)

            # === Disambiguate sign using physical constraint ===
            # Convention: progressive flow (expansion) = positive T_x (forward motion)
            # For expansion at point x: flow v points away from focus of expansion
            # w = Π @ T points toward focus, so v · w < 0 for expansion with correct T
            # We flip T if v · w > 0 (wrong sign), keeping v · w < 0
            w_i_valid = torch.einsum('nij,tj->tni', P, T_dir_valid)  # (T_valid, N, 3)
            v_dot_w_valid = (v[has_translation] * w_i_valid).sum(dim=-1)  # (T_valid, N)

            # Use weighted sum (weight by ||w||² to emphasize reliable points)
            w_norm_sq_valid = (w_i_valid * w_i_valid).sum(dim=-1)  # (T_valid, N)
            weighted_sign = (v_dot_w_valid * w_norm_sq_valid).sum(dim=-1)  # (T_valid,)

            # Flip T where weighted_sign < 0 so that expansion gives positive T
            flip_mask = weighted_sign < 0
            T_dir_valid[flip_mask] = -T_dir_valid[flip_mask]
            v_dot_w_valid[flip_mask] = -v_dot_w_valid[flip_mask]

            T_dir[has_translation] = T_dir_valid
            v_dot_w[has_translation] = v_dot_w_valid
            w_norm_sq[has_translation] = w_norm_sq_valid

        # Scale T by translation magnitude
        T_star = T_dir * translation_magnitude[:, None]  # (T, 3)

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
                device, dtype, dt, seed, etc.)
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
        
        # Build a sparse matrix for each neurotransmitter and track which cell types receive each
        self.synapse_counts = []
        # Track which cell types receive input from each transmitter: (num_transmitters, num_cell_types)
        receives_transmitter = torch.zeros((self.num_transmitters, self.num_cell_types), dtype=torch.bool, device=self.device)
        
        for ri, r in enumerate(self.unique_transmitters):
            # Filter to synapses where presynaptic neuron expresses this transmitter
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
            
            # Determine which cell types receive this transmitter (have non-zero post-synaptic input)
            post_neuron_indices = coo_r[:, 1].astype(np.int64)
            post_cell_type_indices = neuron_to_type_idx[post_neuron_indices]
            unique_receiving_types = np.unique(post_cell_type_indices)
            receives_transmitter[ri, unique_receiving_types] = True
        
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
            synapses_in_per_neuron = self.synapse_counts[ri].pow(2).sum(dim=1).to_dense()
            
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
            synapses_out_per_neuron = self.synapse_counts[ri].pow(2).sum(dim=0).to_dense()
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

