import torch
import numpy as np
import time

class SimpleRnn(torch.nn.Module):
    def __init__(self, synapse_counts, nonlinearity=torch.relu, device=torch.device('cpu'), dtype=torch.float32, use_sparse=False):
        """
        n: hidden size
        m: input size
        nonlinearity: callable that takes a tensor (default torch.tanh).
                      Can also be a functools.partial with extra args.
        """
        super().__init__()
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype
        self.use_sparse = use_sparse
        self.num_neurons = synapse_counts.shape[0]
        self.synapse_counts = synapse_counts
        self.diagonal_log = torch.full((self.num_neurons,), np.log(0.9), dtype=self.dtype, device=self.device)
        self.synapse_strength_log = torch.tensor(np.log(0.001), dtype=self.dtype, device=self.device)
        self.input_strength_log = torch.full((self.num_neurons,), np.log(0.01), device=self.device, dtype=self.dtype)

        self.diagonal_log = torch.nn.Parameter(self.diagonal_log)
        self.synapse_strength_log = torch.nn.Parameter(self.synapse_strength_log)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)

        self.update_dynamics_matrix()
        self.b = torch.nn.Parameter(torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype))

    def update_dynamics_matrix(self):
        if self.use_sparse:
            indices = torch.arange(self.num_neurons, device=self.device, dtype=self.dtype)
            diag_weights = torch.sparse_coo_tensor(
                indices=torch.stack([indices, indices]),
                values=torch.exp(self.diagonal_log),
                size=(self.num_neurons, self.num_neurons),
            )

            coo = torch.exp(self.synapse_strength_log) * self.synapse_counts + diag_weights
            self.dynamics_matrix = coo.to_sparse_csr()
        else:
            self.dynamics_matrix = torch.exp(self.synapse_strength_log) * self.synapse_counts + torch.diag(torch.exp(self.diagonal_log))

    def forward(self, x_prev, u_t):
        return self.nonlinearity(x_prev) @ self.dynamics_matrix.transpose(0, 1) + u_t * torch.exp(self.input_strength_log) + self.b
