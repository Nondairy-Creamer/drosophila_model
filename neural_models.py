import torch
import numpy as np
import time

class SimpleRnn(torch.nn.Module):
    def __init__(self, synapse_counts, cell_types, nonlinearity=torch.relu, device=torch.device('cpu'), dtype=torch.float32, use_sparse=False):
        diagonal_init = 0.9
        synapse_strength_init = 0.001
        input_strength_init = 0.01
        readout_weight_init = 1
        readout_bias_init = 0

        super().__init__()
        self.nonlinearity = nonlinearity
        self.cell_types = cell_types
        self.device = device
        self.dtype = dtype
        self.use_sparse = use_sparse
        self.num_neurons = synapse_counts.shape[0]
        self.unique_cell_types = np.unique(cell_types)
        self.num_cell_types = self.unique_cell_types.shape[0]
        self.synapse_counts = synapse_counts
        self.diagonal_log = torch.full((self.num_cell_types,), np.log(diagonal_init), dtype=self.dtype, device=self.device)
        self.synapse_strength_log = torch.tensor(np.log(synapse_strength_init), dtype=self.dtype, device=self.device)
        self.input_strength_log = torch.full((self.num_cell_types,), np.log(input_strength_init), device=self.device, dtype=self.dtype)
        self.offset_cell_type = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype)
        self.readout_weight = torch.tensor(1, device=self.device, dtype=self.dtype)
        self.readout_bias = torch.tensor(0, device=self.device, dtype=self.dtype)

        self.diagonal_log = torch.nn.Parameter(self.diagonal_log)
        self.synapse_strength_log = torch.nn.Parameter(self.synapse_strength_log)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.offset_cell_type = torch.nn.Parameter(self.offset_cell_type)
        self.readout_weight = torch.nn.Parameter(self.readout_weight)
        self.readout_bias = torch.nn.Parameter(self.readout_bias)

        # set readout neurons and masks
        self.neuron_pro_name = 'T4a'
        self.neuron_reg_name = 'T4b'
        self.mask_pro = self.cell_types == self.neuron_pro_name
        self.mask_reg = self.cell_types == self.neuron_reg_name

        # Pre-compute boolean masks to map from each cell type to each cell
        self.cell_type_masks = {}
        for cell_type_index, cell_type in enumerate(self.unique_cell_types):
            self.cell_type_masks[cell_type_index] = torch.tensor(
                self.cell_types == cell_type, 
                device=self.device, 
                dtype=torch.bool
            )
        
        # Diag indices only used if using sparse matricies
        self.diag_indices = torch.stack([
            torch.arange(self.num_neurons, device=self.device, dtype=torch.int64),
            torch.arange(self.num_neurons, device=self.device, dtype=torch.int64)
        ], dim=0)
        
        # Pre-allocate tensors to avoid repeated allocation
        self.diagonal = torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype)
        self.input_strength = torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype)
        self.offset = torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype)

        self.construct_model_parameters()


    def forward(self, x_prev, u_t):
        return self.nonlinearity(x_prev) @ self.dynamics_matrix.transpose(0, 1) + u_t * self.input_strength + self.offset

    def construct_model_parameters(self):
        self.diagonal.detach_()
        self.input_strength.detach_()
        self.offset.detach_()

        # Vectorized assignment using pre-computed masks
        for cell_type_index in range(self.num_cell_types):
            mask = self.cell_type_masks[cell_type_index]
            self.diagonal[mask] = torch.exp(self.diagonal_log[cell_type_index])
            self.input_strength[mask] = torch.exp(self.input_strength_log[cell_type_index])
            self.offset[mask] = torch.exp(self.offset_cell_type[cell_type_index])

        if self.use_sparse:
            # Use pre-computed diagonal indices for faster sparse tensor creation
            diag_sparse = torch.sparse_coo_tensor(
                indices=self.diag_indices,
                values=self.diagonal,
                size=(self.num_neurons, self.num_neurons),
                device=self.device,
                dtype=self.dtype,
            )

            coo = torch.exp(self.synapse_strength_log) * self.synapse_counts + diag_sparse
            self.dynamics_matrix = coo.to_sparse_csr()
        else:
            self.dynamics_matrix = torch.exp(self.synapse_strength_log) * self.synapse_counts + torch.diag(self.diagonal)

    def estimate_readout_weights(self, inputs, target):
        with torch.no_grad():
            # find the optimal starting readout weight and bias using least squares linear regression
            prediction = self.readout(inputs)

            # x, y are 1D float tensors of shape (N,)
            X = torch.stack([prediction, torch.ones_like(prediction)], dim=1)  # [N, 2]
            beta = torch.linalg.lstsq(X, target).solution         # [2]
            slope, intercept = beta[0], beta[1]

            self.readout_weight.copy_(slope)
            self.readout_bias.copy_(intercept)

            # from matplotlib import pyplot as plt
            # plt.figure()
            # plt.plot(prediction.detach().cpu().numpy())
            # plt.plot(target.detach().cpu().numpy())
            # plt.savefig('figures/before_readout_fit.png')
            # plt.close()

            # prediction = self.readout(inputs)

            # plt.figure()
            # plt.plot(prediction.detach().cpu().numpy())
            # plt.plot(target.detach().cpu().numpy())
            # plt.savefig('figures/after_readout_fit.png')
            # plt.close()

    def predict(self, inputs):
        """
        cell: SimpleRnn instance
        U: (batch, T, m)  # input sequences
        h0: (batch, n) initial state, defaults to zeros
        """
        num_batches, num_time, num_neurons = inputs.shape
        outputs = torch.zeros((num_batches, num_time, self.num_neurons), device=inputs.device, dtype=inputs.dtype)

        for t in range(1, num_time):
            outputs[:, t, :] = self(outputs[:, t-1, :], inputs[:, t-1, :])   # (B, n)

        return outputs
    
    def readout(self, inputs):
        self.construct_model_parameters()
        outputs = self.predict(inputs)            # (batch, T, n)

        neurons_pro = outputs[:, -1, self.mask_pro].mean(dim=1)  # (batch,)
        neurons_reg = outputs[:, -1, self.mask_reg].mean(dim=1)  # (batch,)

        return self.readout_weight * (neurons_pro - neurons_reg) + self.readout_bias


