import torch
import numpy as np
from pathlib import Path
from functools import reduce
import tempfile
import os


class NeuralRnn(torch.nn.Module):
    def __init__(self, synapse_counts_coo, cell_types, column_indices, nonlinearity='relu', device='cpu', dtype='float32',
                 use_layer_norm=False, set_max_sigma=True, dt=0.01, seed=None, init_std=0.1):
        # capture constructor inputs early (before we mutate or derive types)
        # we'll save these to so that we can load the class easily
        params = locals().copy()
        params.pop('self', None)
        params.pop('__class__', None)
        self.init_inputs = params
        super().__init__()

        persistence_init = 0.5
        stimulus_strength_init = 10.0
        offset_init = 0.0
        self.max_allowed_sigma = 0.95
        self.silence_val = 1e-8  # when silencing neurons this is the value we set their connections to
        # Prevent input strength from going to exactly zero, which would silence cells
        # This is especially important for cells with sparse targets (like T4a/T5a for rightward motion)
        # where predicting zero can give misleadingly good correlation scores
        self.input_strength_minimum = 0  # minimum input strength (prevents complete silencing)

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

        self.cell_types = cell_types
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.use_layer_norm = use_layer_norm
        self.num_neurons = cell_types.shape[0]
        self.unique_cell_types = np.unique(cell_types)
        self.num_cell_types = self.unique_cell_types.shape[0]
        self.column_indices = column_indices
        self.num_columns = np.unique(column_indices[~np.isnan(column_indices)]).size
        self.dt = dt  # not currently used in the model, but the model implicitly expects a time step
        # Initialize persistent power-iteration vector for warm starts using RNG
        # Register as buffer so it's saved/restored with state_dict
        power_vec_np = self.rng.standard_normal(self.num_neurons)
        power_vec = torch.tensor(power_vec_np, device=self.device, dtype=self.dtype)
        power_vec = power_vec / (power_vec.norm() + 1e-12)
        self.register_buffer('_power_vec', power_vec)

        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        # --- model parameters with random initialization
        # Initialize as numpy arrays with Gaussian noise, then convert to torch tensors
        persistence_logit_init = logit(persistence_init)
        persistence_logit_np = persistence_logit_init + self.rng.normal(0, init_std, self.num_cell_types)
        self.persistence_logit = torch.tensor(persistence_logit_np, dtype=self.dtype, device=self.device)

        stimulus_strength_log_init = np.log(stimulus_strength_init)
        stimulus_strength_log_np = stimulus_strength_log_init + self.rng.normal(0, init_std, self.num_cell_types)
        self.stimulus_strength_log = torch.tensor(stimulus_strength_log_np, dtype=self.dtype, device=self.device)

        offset_np = offset_init + self.rng.normal(0, init_std, self.num_cell_types)
        self.offset = torch.tensor(offset_np, dtype=self.dtype, device=self.device)

        # --- create the matrix of synapse counts ---
        # create synapse counts matrix with either sparse or dense format
        # synapse counts sparse must be in the form N x 3 with [indices, (pre_synaptic, post_synaptic, count)]
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
        # Ensure input_strength_init is at least input_strength_minimum to prevent cells from being silenced
        input_strength_init = input_strength_init.clamp_min(self.input_strength_minimum * 2)  # Start above minimum
        # Initialize log such that exp(log) + minimum = input_strength_init
        input_strength_log_init = torch.log(input_strength_init - self.input_strength_minimum + 1e-10)
        output_strength_log_init = torch.log(output_strength_init)

        # Add Gaussian noise to strength parameters
        input_strength_log_np = input_strength_log_init.detach().cpu().numpy() + self.rng.normal(0, init_std, self.num_cell_types)
        output_strength_log_np = output_strength_log_init.detach().cpu().numpy() + self.rng.normal(0, init_std, self.num_cell_types)
        self.input_strength_log = torch.tensor(input_strength_log_np, dtype=self.dtype, device=self.device)
        self.output_strength_log = torch.tensor(output_strength_log_np, dtype=self.dtype, device=self.device)

        if set_max_sigma:
            self.set_sigma_max()

        self.persistence_logit = torch.nn.Parameter(self.persistence_logit)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.offset = torch.nn.Parameter(self.offset)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.output_strength_log = torch.nn.Parameter(self.output_strength_log)

    def set_sigma_max(self):
        with torch.no_grad():
            # Iteratively scale output_strength to achieve target max sigma (spectral norm, largest singular value)
            max_scaling_iters = 100
            rho_tolerance = 0.01
            damping_factor = 0.3

            for iter_idx in range(max_scaling_iters):
                sigma_max = self.calc_sigma_max()

                print(iter_idx, ':', sigma_max)
                if abs(sigma_max - self.max_allowed_sigma) < rho_tolerance:
                    break

                # Scale output_strength to move toward target
                scale_factor = (self.max_allowed_sigma / sigma_max)**damping_factor
                self.output_strength_log.add_(torch.log(scale_factor))

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        input_strength = self.input_strength_log.exp()[self.neuron_to_type_index] + self.input_strength_minimum
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

        # Optionally normalize recurrent term before adding inputs/offset
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

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (z + offset_all_neurons[None, :])

        return v_next

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        A = self.synapse_counts.coalesce()
        n = self.num_neurons
        device = self.device
        dtype = self.dtype
        # Per-neuron scalings
        D_in  = self.input_strength_log.exp()[self.neuron_to_type_index] + self.input_strength_minimum   # (n,)
        D_out = self.output_strength_log.exp()[self.neuron_to_type_index]  # (n,)
        d = self.persistence_logit.sigmoid()[self.neuron_to_type_index]       # (n,)
        one_minus_d = 1.0 - d

        # M v = D_in * (A @ (D_out * v))
        def apply_M(v):
            v_scaled = v * D_out
            y = torch.sparse.mm(A, v_scaled.unsqueeze(1)).squeeze(1)
            return y * D_in

        # M^T v = D_out * (A^T @ (D_in * v))
        def apply_M_T(v):
            v_scaled = v * D_in
            y = torch.sparse.mm(A.transpose(0, 1), v_scaled.unsqueeze(1)).squeeze(1)
            return y * D_out

        # S v = diag(d) v + diag(1-d) (M v)
        def apply_S(v):
            return d * v + one_minus_d * apply_M(v)

        # S^T v = diag(d) v + M^T (diag(1-d) v)
        def apply_S_T(v):
            return d * v + apply_M_T(one_minus_d * v)

        # Warm start from registered buffer (saved/restored with state_dict)
        x = self._power_vec.clone()

        x = x / (x.norm() + eps)  # normalize, but no gradient needed w.r.t x
        # x is treated as a constant initial direction for autograd

        for _ in range(max_iter):
            y = apply_S(x)     # y ≈ σ u
            x = apply_S_T(y)   # x ≈ σ^2 v
            x = x / (x.norm() + eps)

        # At convergence, ||S v||_2 ≈ σ_max
        y = apply_S(x)
        sigma = y.norm()  # differentiable w.r.t. parameters in S

        # Update warm-start vector in-place (registered buffer), but cut history
        self._power_vec.copy_(x.detach())

        return sigma

    def predict(self, inputs, input_neurons=['L1'], output_neurons=['T4a'], init_state=None):
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
            v = inputs.new_zeros(num_batches, self.num_neurons)
        else:
            v = init_state

        # outputs for each neuron in output_neurons are (batches, time, number of columns)
        outputs = {'voltage': {}, 'firing_rate': {}}
        for i in readout_keys:
            outputs['voltage'][i] = torch.full((num_batches, num_time, self.num_columns), float('nan'), device=self.device, dtype=self.dtype)
            outputs['firing_rate'][i] = torch.full((num_batches, num_time, self.num_columns), float('nan'), device=self.device, dtype=self.dtype)

        for t in range(num_time):
            v = self(v, inputs[:, t, :], stim_to_cell_mapping)

            for i in readout_keys:
                ci = readout_dict[i]['column_indices']
                ni = readout_dict[i]['neuron_indices']
                # Ensure we only write to valid column positions
                valid = ci >= 0
                if torch.any(valid):
                    outputs['voltage'][i][:, t, ci[valid]] = v[:, ni[valid]]

        # Compute firing_rate from the complete voltage tensor (more memory efficient)
        # This creates one computation graph node instead of one per timestep
        for i in readout_keys:
            outputs['firing_rate'][i] = self.nonlinearity(outputs['voltage'][i])

        return outputs, v.detach(), {'input_mapping': stim_to_cell_mapping, 'output_mapping': readout_dict}

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'inputs': self.init_inputs,
            # store weights on CPU for portability
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

    def silence_neurons(self, neurons_to_silence):
        if len(neurons_to_silence) > 0:
            print("\n--- WARNING silencing neurons", neurons_to_silence, '---\n')
            # set the input and output strength of these neurons to a small value
            chosen_neurons = np.zeros(self.num_cell_types) == 1
            for ns in neurons_to_silence:
                chosen_neurons[self.unique_cell_types == ns] = True

            with torch.no_grad():
                self.output_strength_log.data[chosen_neurons] = np.log(self.silence_val)
        else:
            print('No neurons selected for silencing')

    @classmethod
    def load(cls, path, device=None, dtype=None, freeze_params: bool = True):
        data = torch.load(Path(path), map_location='cpu', weights_only=False)
        inputs = dict(data['inputs'])
        if device is not None:
            inputs['device'] = device
        if dtype is not None:
            inputs['dtype'] = dtype

        # construct directly from saved (or overridden) inputs
        inputs['set_max_sigma'] = False
        model = cls(**inputs)
        model.load_state_dict(data['state_dict'])

        # Optionally freeze all parameters to disable gradients
        if freeze_params:
            for param in model.parameters():
                param.requires_grad_(False)

        return model

