import torch
import numpy as np
import time

class SimpleRnn(torch.nn.Module):
    def __init__(self, synapse_counts, cell_types, nonlinearity=torch.relu, device=torch.device('cpu'), dtype=torch.float32, use_sparse=False):
        diagonal_init = 0.9
        synapse_strength_init = 0.001
        input_strength_init = 0.01

        super().__init__()
        self.nonlinearity = nonlinearity
        self.cell_types = cell_types
        self.device = device
        self.dtype = dtype
        self.use_sparse = use_sparse
        self.num_neurons = synapse_counts.shape[0]
        self.synapse_counts = synapse_counts
        self.unique_cell_types = np.unique(cell_types)
        self.diagonal_log = torch.full((self.unique_cell_types.shape[0],), np.log(diagonal_init), dtype=self.dtype, device=self.device)
        self.synapse_strength_log = torch.tensor(np.log(synapse_strength_init), dtype=self.dtype, device=self.device)
        self.input_strength_log = torch.full((self.num_neurons,), np.log(input_strength_init), device=self.device, dtype=self.dtype)

        self.diagonal_log = torch.nn.Parameter(self.diagonal_log)
        self.synapse_strength_log = torch.nn.Parameter(self.synapse_strength_log)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)

        self.update_dynamics_matrix()
        self.b = torch.nn.Parameter(torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype))

    def update_dynamics_matrix(self):
        # define the diagonal or self terms. we want to learn a self term for each cell type, but shared accross individual cells
        diagonal = torch.zeros(self.num_neurons, dtype=self.dtype, device=self.device)
        for cell_type_index, cell_type in enumerate(self.unique_cell_types):
            mask = self.cell_types == cell_type
            diagonal[mask] = torch.exp(self.diagonal_log[cell_type_index])

        if self.use_sparse:
            # Create diagonal matrix as sparse tensor
            indices = torch.arange(self.num_neurons, device=self.device, dtype=torch.int64)
            diag_indices = torch.stack([indices, indices], dim=0)

            diag_sparse = torch.sparse_coo_tensor(
                indices=diag_indices,
                values=diagonal,
                size=(self.num_neurons, self.num_neurons),
                device=self.device,
                dtype=self.dtype,
            )

            coo = torch.exp(self.synapse_strength_log) * self.synapse_counts + diag_sparse
            self.dynamics_matrix = coo.to_sparse_csr()
        else:
            self.dynamics_matrix = torch.exp(self.synapse_strength_log) * self.synapse_counts + torch.diag(diagonal)

    def forward(self, x_prev, u_t):
        return self.nonlinearity(x_prev) @ self.dynamics_matrix.transpose(0, 1) + u_t * torch.exp(self.input_strength_log) + self.b
