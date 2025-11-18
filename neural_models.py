import torch
import numpy as np
from pathlib import Path
from functools import reduce


class NeuralRnn(torch.nn.Module):
    def __init__(self, synapse_counts_coo, cell_types, column_indices, nonlinearity='relu', device='cpu', dtype='float32',
                 use_layer_norm=False, set_spectral_radius=True, dt=0.01):
        # capture constructor inputs early (before we mutate or derive types)
        # we'll save these to so that we can load the class easily
        params = locals().copy()
        params.pop('self', None)
        params.pop('__class__', None)
        self.init_inputs = params
        super().__init__()

        diagonal_init = 0.5
        stimulus_strength_init = 10.0
        offset_init = 0.0

        # --- model properties ---
        if nonlinearity in ("identity", "linear"):
            self.nonlinearity = torch.nn.Identity()
        else:
            self.nonlinearity = getattr(torch.nn, nonlinearity)()
            
        self.cell_types = cell_types
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.use_layer_norm = use_layer_norm
        self.num_neurons = cell_types.shape[0]
        self.unique_cell_types = np.unique(cell_types)
        self.num_cell_types = self.unique_cell_types.shape[0]
        self.synapse_counts_coo = synapse_counts_coo
        self.column_indices = column_indices
        self.num_columns = np.unique(column_indices[~np.isnan(column_indices)]).size
        self.dt = dt  # not currently used in the model, but the model implicitly expects a time step

        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        # --- model parameters
        self.diagonal_logit = torch.full((self.num_cell_types,), logit(diagonal_init), dtype=self.dtype, device=self.device)
        self.stimulus_strength_log = torch.full((self.num_cell_types,), np.log(stimulus_strength_init), device=self.device, dtype=self.dtype)
        self.offset = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype) + offset_init

        # --- create the matrix of synapse counts ---
        # create synapse counts matrix with either sparse or dense format
        # synapse counts sparse must be in the form N x 3 with [indices, (pre_synaptic, post_synaptic, count)]
        rows = torch.tensor(self.synapse_counts_coo[:, 1], device=device, dtype=torch.int64)
        cols = torch.tensor(self.synapse_counts_coo[:, 0], device=device, dtype=torch.int64)
        values = torch.tensor(self.synapse_counts_coo[:, 2], device=device, dtype=self.dtype)

        indices = torch.stack([rows, cols], dim=0)
        self.synapse_counts = torch.sparse_coo_tensor(indices, values, size=(self.num_neurons, self.num_neurons), dtype=self.dtype, device=self.device).coalesce()
        self.synapse_counts = self.synapse_counts.coalesce()

        # Pre-compute the mapping from a cell type to the indicies of its individual neurons
        self.neuron_to_type_index = torch.zeros(self.num_neurons, device=self.device, dtype=torch.int64) - 1
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask_np = (self.cell_types == ct)
            ct_mask = torch.from_numpy(ct_mask_np).to(device=self.device)
            self.neuron_to_type_index[ct_mask] = cti
        assert torch.all(self.neuron_to_type_index != -1), 'some cell type to neuron mappings not assigned'
        
        # Optional layer normalization for better signal propagation
        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(self.num_neurons, elementwise_affine=False, device=self.device, dtype=self.dtype)

        # Target: keep variance roughly constant across layers
        # Compute fan-in per neuron
        synapses_in_per_neuron = self.synapse_counts.pow(2).sum(dim=1).to_dense()  # (num_neurons,)
        synapses_out_per_neuron = self.synapse_counts.pow(2).sum(dim=0).to_dense()  # (num_neurons,)

        # Average fan-in for each cell type
        synapses_in_per_cell_type = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype)
        synapses_out_per_cell_type = torch.zeros(self.num_cell_types, device=self.device, dtype=self.dtype)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask_np = (self.cell_types == ct)
            ct_mask = torch.from_numpy(ct_mask_np).to(device=self.device)
            synapses_in_per_cell_type[cti] = synapses_in_per_neuron[ct_mask].mean().clamp_min(1.0)
            synapses_out_per_cell_type[cti] = synapses_out_per_neuron[ct_mask].mean().clamp_min(1.0)

        # Initialize strength per cell type based on the inverse of its fan in / out
        input_strength_init = synapses_in_per_cell_type.pow(-1 / 2)
        output_strength_init = synapses_out_per_cell_type.pow(-1 / 2)
        self.input_strength_log = torch.log(input_strength_init)
        self.output_strength_log = torch.log(output_strength_init)

        if set_spectral_radius:
            with torch.no_grad():
                # Iteratively scale output_strength to achieve target spectral radius
                target_rho = 0.90
                max_scaling_iters = 100
                rho_tolerance = 0.001
                damping_factor = 0.5

                for iter_idx in range(max_scaling_iters):
                    rho, iterations, converged = self.calc_sigma_max()

                    print(iter_idx, ':', rho)
                    assert converged, f'spectral radius calculation did not converge at scaling iteration {iter_idx}'
                    
                    if abs(rho - target_rho) < rho_tolerance:
                        break
                    
                    # Scale output_strength to move toward target
                    scale_factor = (target_rho / rho)**damping_factor
                    self.output_strength_log.add_(torch.log(scale_factor))

        self.diagonal_logit = torch.nn.Parameter(self.diagonal_logit)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.offset = torch.nn.Parameter(self.offset)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.output_strength_log = torch.nn.Parameter(self.output_strength_log)

    def forward(self, x_prev, u_t, stim_to_cell_mapping=None):
        diagonal = self.diagonal_logit.sigmoid()[self.neuron_to_type_index]
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
        z = z * output_strength[None, :]
        z = z + x_prev * diagonal[None, :]
        
        # Apply layer normalization before nonlinearity if enabled
        if self.use_layer_norm:
            z = self.layer_norm(z)
        
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

        x_next = self.nonlinearity(z + offset_all_neurons[None, :])

        return x_next

    def calc_sigma_max(self, max_iter=200, tol=1e-4):
        A = self.synapse_counts.coalesce()
        n = self.num_neurons
        device = self.device
        diagonal = self.diagonal_logit.sigmoid()[self.neuron_to_type_index]
        D_in  = self.input_strength_log.exp()[self.neuron_to_type_index]   # (n,)
        D_out = self.output_strength_log.exp()[self.neuron_to_type_index]  # (n,)

        def mv(v):   # y = M @ v = D_out * (A @ (D_in * v)) + diagonal * v
            v_scaled = v * D_in
            y = v_scaled @ A.T
            y = y * D_out
            return y + diagonal * v

        def mtv(v):  # y = M^T @ v = D_in * (A^T @ (D_out * v)) + diagonal * v
            v_scaled = v * D_out
            y = v_scaled @ A
            y = y * D_in
            return y + diagonal * v

        # Use seeded random initialization for reproducibility
        rng = torch.Generator(device=device).manual_seed(42)
        x = torch.randn(n, device=device, generator=rng)
        x = x / (x.norm() + 1e-12)

        prev = None
        rho = torch.tensor(float("inf"), device=device)
        for k in range(1, max_iter + 1):
            y = mv(x)
            z = mtv(y)                 # z = (A^T A)_scaled x
            # Rayleigh quotient λ = x^T (A^T A) x must be computed BEFORE updating x
            lam = (x * z).sum()
            zn = z.norm()
            if zn == 0:
                return 0.0, k, True
            x = z / zn
            rho = torch.sqrt(torch.clamp(lam, min=0.0))
            if prev is not None and torch.abs(rho - prev) / (prev + 1e-12) < tol:
                return rho, k, True
            prev = rho
        return rho, max_iter, False

    def predict(self, inputs, input_neurons=['L1'], output_neurons=['T4a'], init_state=None, return_mappings=False):
        # given some input, generate the outputs of the rnn
        num_batches, num_time, num_neurons = inputs.shape

        # Handle input mapping (either compute from names list, or reuse provided indices)
        if isinstance(input_neurons, list):
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
            x = inputs.new_zeros(num_batches, self.num_neurons)
        else:
            x = init_state

        # outputs for each neuron in output_neurons are (batches, time, number of columns)
        outputs = {}
        for i in readout_keys:
            outputs[i] = torch.full((num_batches, num_time, self.num_columns), float('nan'),
                                    device=self.device, dtype=self.dtype)
                                    
        for t in range(num_time):
            x = self(x, inputs[:, t, :], stim_to_cell_mapping)

            for i in readout_keys:
                ci = readout_dict[i]['column_indices']
                ni = readout_dict[i]['neuron_indices']
                # Ensure we only write to valid column positions
                valid = ci >= 0
                if torch.any(valid):
                    outputs[i][:, t, ci[valid]] = x[:, ni[valid]]

        if return_mappings:
            return outputs, x.detach(), {'stim_to_cell_mapping': stim_to_cell_mapping, 'readout_dict': readout_dict}
        else:
            return outputs, x.detach()

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
        
    @classmethod
    def load(cls, path, device=None, dtype=None, freeze_params: bool = True):
        data = torch.load(Path(path), map_location='cpu', weights_only=False)
        inputs = dict(data['inputs'])
        if device is not None:
            inputs['device'] = device
        if dtype is not None:
            inputs['dtype'] = dtype

        # construct directly from saved (or overridden) inputs
        inputs['set_spectral_radius'] = False
        model = cls(**inputs)

        model.load_state_dict(data['state_dict'])

        # Optionally freeze all parameters to disable gradients
        if freeze_params:
            for param in model.parameters():
                param.requires_grad_(False)

        return model

