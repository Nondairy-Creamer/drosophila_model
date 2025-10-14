import torch
import numpy as np
from scipy.optimize import lsq_linear
import pickle
from pathlib import Path
import time


class NeuralRnn(torch.nn.Module):
    def __init__(self, synapse_counts_df, cell_types, nonlinearity=torch.relu, device=torch.device('cpu'), dtype=torch.float32,
                 use_sparse=False):
        super().__init__()
        diagonal_init = 0.9
        input_strength_init = 0.2
        output_strength_init = input_strength_init
        stimulus_strength_init = 0.1
        readout_weight_init = 1
        readout_bias_init = 0

        # --- model properties ---
        self.nonlinearity = nonlinearity
        self.cell_types = cell_types
        self.device = device
        self.dtype = dtype
        self.use_sparse = use_sparse
        self.num_neurons = cell_types.shape[0]
        self.unique_cell_types = np.unique(cell_types)
        self.num_cell_types = self.unique_cell_types.shape[0]

        # set readout neurons and masks
        self.neuron_pro_name = 'T4a'
        self.neuron_reg_name = 'T4b'
        self.mask_pro = self.cell_types == self.neuron_pro_name
        self.mask_reg = self.cell_types == self.neuron_reg_name

        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        # --- model parameters
        self.diagonal_logit = torch.full((self.num_cell_types,), logit(diagonal_init), dtype=self.dtype, device=self.device)
        self.synapse_strength_log = torch.tensor(np.log(1.1), device=self.device, dtype=self.dtype)
        # self.synapse_strength_log = torch.tensor(np.log(1.1*0.04), device=self.device, dtype=self.dtype)
        self.input_strength_log = torch.full((self.num_cell_types,), np.log(input_strength_init), device=self.device, dtype=self.dtype)
        self.output_strength_log = torch.full((self.num_cell_types,), np.log(output_strength_init), device=self.device, dtype=self.dtype)
        self.stimulus_strength_log = torch.full((self.num_cell_types,), np.log(stimulus_strength_init), device=self.device, dtype=self.dtype)
        self.offset = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype)
        self.readout_weight_log = torch.tensor(np.log(readout_weight_init), device=self.device, dtype=self.dtype)
        self.readout_bias = torch.tensor(readout_bias_init, device=self.device, dtype=self.dtype)

        self.diagonal_logit = torch.nn.Parameter(self.diagonal_logit)
        # self.synapse_strength_log = torch.nn.Parameter(self.synapse_strength_log)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.output_strength_log = torch.nn.Parameter(self.output_strength_log)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.offset = torch.nn.Parameter(self.offset)
        self.readout_weight_log = torch.nn.Parameter(self.readout_weight_log)
        self.readout_bias = torch.nn.Parameter(self.readout_bias)

        # --- pre-compute some variables for efficiency ---

        # create synapse counts matrix with either sparse or dense format
        rows = torch.tensor(synapse_counts_df.post_matrix_ind.values, device=device, dtype=torch.int64)
        cols = torch.tensor(synapse_counts_df.pre_matrix_ind.values, device=device, dtype=torch.int64)
        values = torch.tensor(synapse_counts_df.synapse_counts.values, device=device, dtype=self.dtype)

        if use_sparse:
            # Diag indices only used if using sparse matricies when adding the diagonal to the dynamics matrix
            self.diag_indices = torch.stack([
                torch.arange(self.num_neurons, device=self.device, dtype=torch.int64),
                torch.arange(self.num_neurons, device=self.device, dtype=torch.int64)
            ], dim=0)

            # indices must be int64
            indices = torch.stack([rows, cols], dim=0)
            self.synapse_counts = torch.sparse_coo_tensor(indices, values, size=(self.num_neurons, self.num_neurons), dtype=self.dtype, device=self.device).coalesce()
            self.synapse_counts = self.synapse_counts.coalesce()
        else:
            # create a dense matrix
            self.synapse_counts = torch.zeros((self.num_neurons, self.num_neurons), device=self.device, dtype=self.dtype)
            self.synapse_counts.index_put_((rows, cols), values, accumulate=True).to(device=self.device, dtype=self.dtype)

        # Pre-compute the mapping from a cell type to the indicies of its individual neurons
        self.neuron_to_type_index = torch.zeros(self.num_neurons, device=self.device, dtype=torch.int64) - 1
        for cti, ct in enumerate(self.unique_cell_types):
            self.neuron_to_type_index[self.cell_types == ct] = cti
        assert torch.all(self.neuron_to_type_index != -1), 'some cell type to neuron mappings not assigned'
        
    def forward(self, x_prev, u_t):
        diagonal = torch.sigmoid(self.diagonal_logit)[self.neuron_to_type_index]
        synapse_strength = torch.exp(self.synapse_strength_log)
        input_strength = torch.exp(self.input_strength_log)[self.neuron_to_type_index]
        output_strength = torch.exp(self.output_strength_log)[self.neuron_to_type_index]
        stimulus_strength = torch.exp(self.stimulus_strength_log)[self.neuron_to_type_index]
        offset_all_neurons = self.offset[self.neuron_to_type_index]

        # model is
        # dynamics_matrix = out_scaling_column_vector * synapse_counts * in_scaling_row_vector + diagonal
        # x_next = nonlin( dynamics_matrix @ x_prev + stimulus * stimulus_strength + offset )
        # implementation is slightly more complicated to allow synapse_counts to be sparse and not run into
        # issues with passing gradients due to pytorch's limited sparse support.
        # we can't precompute the dynamics matrix because the broacast opperation of hte input/output vectors doesn't
        # allow for passing the gradient when synapse_counts is sparse
        # we also have to separately apply the diagonal to x_prev to avoid converting it to a sparse diagonal which takes time

        x_scaled = x_prev * input_strength[None, :]
        z = x_scaled @ self.synapse_counts.T
        z = synapse_strength * z * output_strength[None, :]
        z = z + x_prev * diagonal
        x_next = self.nonlinearity(z + u_t * stimulus_strength + offset_all_neurons)

        return x_next

    def estimate_readout_weights(self, inputs, target):
        with torch.no_grad():
            epsilon = 1e-6
            # find the optimal starting readout weight and bias using least squares linear regression
            readout = self.predict_and_readout(inputs)[1]
            # get rid of the current readout / bias
            readout = readout - self.readout_bias
            readout = readout / torch.exp(self.readout_weight_log)

            # readout, target are 1D torch tensors of shape (N,)
            readout_np = readout.detach().cpu().numpy().astype(np.float64)
            target_np  = target.detach().cpu().numpy().astype(np.float64)

            # fit with constraint that the slope is positive. this is to keep it from learningn to swap left and right
            # so when give it an initial condition we need to make sure its positive
            A = np.c_[readout_np, np.ones_like(readout_np, dtype=np.float64)]
            res = lsq_linear(A, target_np, bounds=([epsilon, -np.inf], [np.inf, np.inf]))
            slope, intercept = res.x  # slope >= 0, intercept free

            # set the initial values
            self.readout_weight_log.copy_(np.log(slope))
            self.readout_bias.copy_(intercept)

    def predict(self, inputs):
        # given some input, generate the outputs of the rnn
        num_batches, num_time, num_neurons = inputs.shape

        x = inputs.new_zeros(num_batches, self.num_neurons)
        outputs = inputs.new_zeros(num_batches, num_time, self.num_neurons)
        for t in range(1, num_time):
            x = self(x, inputs[:, t, :])
            outputs[:, t, :] = x

        return outputs

    def readout(self, outputs):
        # given some outputs of the rnn calcualte the linear decoding readout
        neurons_pro = outputs[:, :, self.mask_pro].mean(dim=(1, 2))  # (batch,)
        neurons_reg = outputs[:, :, self.mask_reg].mean(dim=(1, 2))  # (batch,)

        return torch.exp(self.readout_weight_log) * (neurons_pro - neurons_reg) + self.readout_bias
    
    def predict_and_readout(self, inputs):
        # predict the rnn outputs from an input and then read them out
        outputs = self.predict(inputs)
        return outputs, self.readout(outputs)

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

