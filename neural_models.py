import torch
import numpy as np
from pathlib import Path


class NeuralRnn(torch.nn.Module):
    def __init__(self, synapse_counts_coo, cell_types, readout_indexes, nonlinearity='relu', device='cpu', dtype='float32',
                 scale_in_out=False, use_sparse=False):

        self.readout_indexes = readout_indexes

        # capture constructor inputs early (before we mutate or derive types)
        # we'll save these to so that we can load the class easily
        params = locals().copy()
        params.pop('self', None)
        params.pop('__class__', None)
        self.init_inputs = params
        super().__init__()

        diagonal_init = 0.9
        synapse_strength_init = 1e-3
        input_strength_init = np.sqrt(synapse_strength_init)
        output_strength_init = input_strength_init
        stimulus_strength_init = 1e-1
        readout_weight_init = 1
        readout_bias_init = 0

        # --- model properties ---
        if nonlinearity in ("identity", "linear"):
            self.nonlinearity = torch.nn.Identity()
        else:
            self.nonlinearity = getattr(torch, nonlinearity)
            
        self.cell_types = cell_types
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.use_sparse = use_sparse
        self.num_neurons = cell_types.shape[0]
        self.unique_cell_types = np.unique(cell_types)
        self.num_cell_types = self.unique_cell_types.shape[0]
        self.synapse_counts_coo = synapse_counts_coo
        self.scale_in_out = scale_in_out

        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        # --- model parameters
        self.diagonal_logit = torch.full((self.num_cell_types,), logit(diagonal_init), dtype=self.dtype, device=self.device)
        self.stimulus_strength_log = torch.full((self.num_cell_types,), np.log(stimulus_strength_init), device=self.device, dtype=self.dtype)
        self.offset = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype)
        self.readout_weight_log = torch.tensor(np.log(readout_weight_init), device=self.device, dtype=self.dtype)
        self.readout_bias = torch.tensor(readout_bias_init, device=self.device, dtype=self.dtype)

        self.diagonal_logit = torch.nn.Parameter(self.diagonal_logit)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.offset = torch.nn.Parameter(self.offset)
        # self.readout_weight_log = torch.nn.Parameter(self.readout_weight_log)
        # self.readout_bias = torch.nn.Parameter(self.readout_bias)

        # if we're scaling inputs and outputs then define those vectors as parameters and
        # set synapse strength to a buffer since it is redundant.
        synapse_strength_log = torch.tensor(np.log(synapse_strength_init), device=self.device, dtype=self.dtype)
        input_strength_log = torch.full((self.num_cell_types,), np.log(input_strength_init), device=self.device, dtype=self.dtype)
        output_strength_log = torch.full((self.num_cell_types,), np.log(output_strength_init), device=self.device, dtype=self.dtype)

        if self.scale_in_out:
            self.input_strength_log = torch.nn.Parameter(input_strength_log)
            self.output_strength_log = torch.nn.Parameter(output_strength_log)

            synapse_strength_log[...] = np.log(1)
            self.register_buffer('synapse_strength_log', synapse_strength_log)
        else:
            self.synapse_strength_log = torch.nn.Parameter(synapse_strength_log)

            input_strength_log[:] = np.log(1)
            output_strength_log[:] = np.log(1)
            self.register_buffer('input_strength_log', input_strength_log)
            self.register_buffer('output_strength_log', output_strength_log)
        
        # --- create the matrix of synapse counts ---
        # create synapse counts matrix with either sparse or dense format
        # synapse counts sparse must be in the form N x 3 with [indices, (pre_synaptic, post_synaptic, count)]
        rows = torch.tensor(self.synapse_counts_coo[:, 1], device=device, dtype=torch.int64)
        cols = torch.tensor(self.synapse_counts_coo[:, 0], device=device, dtype=torch.int64)
        values = torch.tensor(self.synapse_counts_coo[:, 2], device=device, dtype=self.dtype)

        if use_sparse:
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
            ct_mask_np = (self.cell_types == ct)
            ct_mask = torch.from_numpy(ct_mask_np).to(device=self.device)
            self.neuron_to_type_index[ct_mask] = cti
        assert torch.all(self.neuron_to_type_index != -1), 'some cell type to neuron mappings not assigned'
        
    def forward(self, x_prev, u_t):
        diagonal = self.diagonal_logit.sigmoid()[self.neuron_to_type_index]
        synapse_strength = self.synapse_strength_log.exp()
        input_strength = self.input_strength_log.exp()[self.neuron_to_type_index]
        output_strength = self.output_strength_log.exp()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
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
        x_next = self.nonlinearity(z + u_t * stimulus_strength[None, :] + offset_all_neurons[None, :])

        return x_next

    def estimate_readout_weights(self, inputs, target):
        with torch.no_grad():
            # just set the wieght so that the readout has the same standard deviation as the data and the bias centers the readout on the data
            # can't do linear regression because slope needs to be positive. least squares with bounds often fits to 0 and kills the gradient
            epsilon = 1e-6

            readout = self.predict_and_readout(inputs)[1]
            # get rid of the current readout / bias
            readout = readout - self.readout_bias
            readout = readout / self.readout_weight_log.exp()

            # find the STD of the prediction and the bias that will center it on the target
            readout_std = torch.std(readout, unbiased=False).clamp_min(epsilon)
            target_std = torch.std(target, unbiased=False).clamp_min(epsilon)
            slope = target_std / readout_std
            intercept = torch.mean(target - slope * readout)

            # set the initial values
            self.readout_weight_log.copy_(torch.log(slope))
            self.readout_bias.copy_(intercept)

    def predict(self, inputs):
        # given some input, generate the outputs of the rnn
        num_batches, num_time, num_neurons = inputs.shape

        # initial neural activity values
        x = inputs.new_zeros(num_batches, self.num_neurons)
        outputs = inputs.new_zeros(num_batches, num_time, self.num_neurons)
        for t in range(num_time):
            x = self(x, inputs[:, t, :])
            outputs[:, t, :] = x

        return outputs

    def readout(self, outputs):
        readout_keys = list(self.readout_indexes.keys())
        readout_size = len(self.readout_indexes[readout_keys[0]])
        num_stimuli = outputs.shape[0]
        num_readouts = len(readout_keys)
        readout = torch.zeros((num_readouts, num_stimuli, readout_size), device=self.device, dtype=self.dtype)
        halftime = outputs.shape[1] // 2

        for ki, key in enumerate(readout_keys):
            # average over the second half of stimulus presentation
            # and select the neurons using the readout_indexes
            readout[ki, :, :] = outputs[:, halftime:, self.readout_indexes[key]].mean(dim=1)  # (batch,)

        return self.readout_weight_log.exp() * readout + self.readout_bias
    
    def predict_and_readout(self, inputs):
        # predict the rnn outputs from an input and then read them out
        outputs = self.predict(inputs)
        return outputs, self.readout(outputs)

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'inputs': self.init_inputs,
            # store weights on CPU for portability
            'state_dict': {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }

        # torch.save uses pickle under the hood and handles tensors well
        torch.save(payload, p)

    @classmethod
    def load(cls, path, device=None, dtype=None):
        data = torch.load(Path(path), map_location='cpu', weights_only=False)
        inputs = dict(data['inputs'])
        if device is not None:
            inputs['device'] = device
        if dtype is not None:
            inputs['dtype'] = dtype

        # construct directly from saved (or overridden) inputs
        model = cls(**inputs)

        model.load_state_dict(data['state_dict'])

        return model

