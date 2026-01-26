from logging import warning
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
import tempfile
import os
import inspect
from abc import ABC, abstractmethod


class NeuralModel(torch.nn.Module, ABC):
    """Base class for neural models. Contains shared functionality for both current and conductance models."""
    
    def __init__(self, synapse_counts_coo, cell_types, neurotransmitters, column_indices, column_phi_theta,
                 excitatory_nt=['ACH'], inhibitory_nt=['GLUT', 'GABA'], nonlinearity='Softplus', device='cpu', dtype='float32',
                 init_new_model=True, dt=0.01, seed=None, init_std=0.1, count_threshold=2, commit_hash=None):
        super().__init__()
        # Save constructor inputs for serialization (child classes will overwrite with their own)
        self._save_init_inputs()
        self.commit_hash = commit_hash
        self.silence_val = 1e-8  # when silencing neurons this is the value we set their connections to
        self.init_std = init_std
        self.count_threshold = count_threshold
        self.excitatory_nt = excitatory_nt
        self.inhibitory_nt = inhibitory_nt
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
        
        # Ensure neurotransmitters is a numpy array
        neurotransmitters = np.array(neurotransmitters)
        
        # synapse_counts_coo has [pre_root_id, post_root_id, count]
        # Get the root_ids that define neuron ordering (from cell_info which was ordered by unique synapse root IDs)
        root_ids = np.unique(synapse_counts_coo[:, 0:2])
        
        # --- Filter neurons by valid neurotransmitters (while still using root IDs) ---
        valid_nt_mask = np.isin(neurotransmitters, valid_neurotransmitters)
        valid_root_ids = set(root_ids[valid_nt_mask])
        num_filtered_nt = np.sum(~valid_nt_mask)
        if num_filtered_nt > 0:
            # Filter synapses where both pre and post have valid neurotransmitters
            pre_valid = np.isin(synapse_counts_coo[:, 0], list(valid_root_ids))
            post_valid = np.isin(synapse_counts_coo[:, 1], list(valid_root_ids))
            synapse_counts_coo = synapse_counts_coo[pre_valid & post_valid]
            print(f"Filtered {num_filtered_nt} neurons with invalid neurotransmitters")
        
        # --- Apply synapse count threshold (still using root IDs) ---
        if count_threshold > 0:
            mask = np.abs(synapse_counts_coo[:, 2]) >= count_threshold
            synapse_counts_coo = synapse_counts_coo[mask]
        
        # --- Remove orphaned neurons: keep only neurons that appear in remaining synapses ---
        connected_root_ids = np.unique(synapse_counts_coo[:, 0:2])
        num_orphaned = len(root_ids) - len(connected_root_ids)
        if num_orphaned > 0:
            print(f"Removed {num_orphaned} orphaned neurons (no connections after thresholding)")
        
        # Filter cell arrays to only include connected neurons (use root_ids to find which to keep)
        keep_mask = np.isin(root_ids, connected_root_ids)
        root_ids = root_ids[keep_mask]
        cell_types = cell_types[keep_mask]
        neurotransmitters = neurotransmitters[keep_mask]
        column_indices = column_indices[keep_mask]
        
        # --- Convert root IDs to matrix indices (done once after all filtering) ---
        self.root_ids = root_ids
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
        
        Call this at the end of __init__ to capture the constructor arguments for serialization.
        Child classes should call this after super().__init__() to overwrite with their own inputs.
        """
        frame = inspect.currentframe().f_back
        params = frame.f_locals.copy()
        params.pop('self', None)
        params.pop('__class__', None)
        self.init_inputs = params
    
    def add_noise_to_parameters(self):
        """Add Gaussian noise to all trainable parameters."""
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.tensor(
                    self.rng.normal(0, self.init_std, param.shape),
                    dtype=param.dtype,
                    device=param.device
                )
                param.add_(noise)

    def predict(self, inputs, input_neurons=['L1', 'L2', 'L3'],  output_neurons=['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'], 
                init_state=None, invert_input=True):
        # given some input, generate the outputs of the rnn
        num_batches, num_time, num_neurons = inputs.shape
        if invert_input:
            # inputs are usually given to L1 L2 and L3 mimicking input from the photo receptors
            # however in Drosophila photo receptors inhibit these cells, so we invert the input here
            inputs = -inputs

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
            v = self.steady_state.unsqueeze(0).expand(num_batches, -1)
        else:
            v = init_state

        # outputs for each neuron in output_neurons are (batches, time, number of columns)
        outputs = {'voltage': {}, 'firing_rate': {}}
        for i in readout_keys:
            outputs['voltage'][i] = torch.full((num_batches, num_time, self.num_columns), float('nan'), device=self.device, dtype=self.dtype)
            outputs['firing_rate'][i] = torch.full((num_batches, num_time, self.num_columns), float('nan'), device=self.device, dtype=self.dtype)

        for t in range(num_time):
            v = self(v, inputs[:, t, :], stim_to_cell_mapping)
            # Compute firing_rate on full v (has neuron dimension for proper threshold broadcasting)
            fr = self.firing_rate(v)

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
        inputs = torch.zeros((1, run_time, self.num_columns), device=self.device, dtype=self.dtype)
        # Start from current steady state buffer
        original_state = self.steady_state.clone()
        initial_state = self.steady_state.unsqueeze(0)
        initial_ave = initial_state.abs().mean()

        for i in range(max_iter):
            final_state = self.predict(inputs, init_state=initial_state)[1]

            # Check if state converged (didn't change from input to output)
            if torch.allclose(final_state, initial_state, rtol=1e-3, atol=1e-5):
                print('steady state found after', i + 1, 'iterations')
                self.steady_state.copy_(final_state.detach().squeeze(0))
                return
            
            initial_state = final_state.detach().clone()

        # Check for exploding dynamics: if new state is much larger than previous
        final_ave = final_state.abs().mean()
        if final_ave / (initial_ave + 1) > 1e6:
            print('Warning: cannot find steady state, dynamics may be unstable')
            return
            
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
        data = torch.load(Path(path), map_location='cpu', weights_only=False)
        inputs = dict(data['inputs'])
        if device is not None:
            inputs['device'] = device
        if dtype is not None:
            inputs['dtype'] = dtype

        # construct directly from saved (or overridden) inputs
        inputs['init_new_model'] = False
        model = cls(**inputs)
        model.load_state_dict(data['state_dict'], strict=False)

        # Optionally freeze all parameters to disable gradients
        if freeze_params:
            for param in model.parameters():
                param.requires_grad_(False)

        return model

    def estimate_self_motion_from_image(self, right_eye_inputs, left_eye_inputs, equilibration_steps=0):
        # right_eye_inputs: (stimuli, time, columns) where columns correspond to
        # ommatidia at the locations in self.column_phi_theta
        # left_eye_inputs: (stimuli, time, columns) where columns correspond to
        # ommatidia at the locations in self.column_phi_theta but with phi = -phi (flipped over the mid point of the fly)
        # equilibration_steps: number of time indices to skip at the beginning for equilibration (default: 0)

        # calculate optic flow from the left eye and right eye stimulus. Concatenate them together for total optic flow
        optic_flow_right = self.calculate_optic_flow(right_eye_inputs, eye='right')[0]
        optic_flow_left = self.calculate_optic_flow(left_eye_inputs, eye='left')[0]
        optic_flow = torch.cat((optic_flow_right, optic_flow_left), axis=2)
        # optic_flow shape: (stimuli, time, columns*2, 2)

        # generate the column locations for both eyes
        left_column_locations = self.column_phi_theta.copy()
        left_column_locations[:, 0] = -left_column_locations[:, 0]
        column_locations = np.concatenate((self.column_phi_theta, left_column_locations), axis=0)

        # Convert column locations to tensors for estimate_self_motion_from_flow
        phi_tensor = torch.tensor(column_locations[:, 0], device=optic_flow.device, dtype=optic_flow.dtype)
        theta_tensor = torch.tensor(column_locations[:, 1], device=optic_flow.device, dtype=optic_flow.dtype)

        # Estimate self-motion for each stimulus and time point
        num_stimuli, num_time, num_columns_total, _ = optic_flow.shape
        num_columns_per_eye = self.column_phi_theta.shape[0]
        rotations = []
        translations = []
        depths = []
        
        for s in range(num_stimuli):
            rot_t = []
            trans_t = []
            depth_t = []
            
            # Skip equilibration_steps at the beginning
            for t in range(equilibration_steps, num_time):
                d_phi = optic_flow[s, t, :, 0]  # (columns*2,)
                d_theta = optic_flow[s, t, :, 1]  # (columns*2,)
                
                # Filter out NaN values (columns without valid optic flow)
                valid_mask = ~torch.isnan(d_phi) & ~torch.isnan(d_theta)
                phi_valid = phi_tensor[valid_mask]
                theta_valid = theta_tensor[valid_mask]
                d_phi_valid = d_phi[valid_mask]
                d_theta_valid = d_theta[valid_mask]
                
                rotation, translation, depth = self.estimate_self_motion_from_flow(
                    phi_valid, theta_valid, d_phi_valid, d_theta_valid
                )
                rot_t.append(rotation)
                trans_t.append(translation)
                
                # Expand depth back to full size with NaN for invalid columns
                depth_full = torch.full((num_columns_total,), float('nan'), 
                                        device=depth.device, dtype=depth.dtype)
                depth_full[valid_mask] = depth
                depth_t.append(depth_full)
            
            rotations.append(torch.stack(rot_t))      # (time, 3)
            translations.append(torch.stack(trans_t))  # (time, 3)
            depths.append(torch.stack(depth_t))        # (time, columns*2)
        
        # Stack results: (stimuli, time, ...) 
        rotations = torch.stack(rotations)      # (stimuli, time, 3)
        translations = torch.stack(translations) # (stimuli, time, 3)
        depths = torch.stack(depths)             # (stimuli, time, columns*2)
        
        return rotations, translations, depths

    @staticmethod
    def estimate_self_motion_from_flow(phi, theta, d_phi, d_theta):
        """
        Estimate self-motion (rotation and translation) from optic flow.
        
        Coordinate system:
            - +X: Forward (phi=0, theta=π/2)
            - +Y: Left (phi=π/2, theta=π/2)  
            - +Z: Up (theta=0)
        
        Flow sign convention:
            - Positive d_phi: rightward motion on retina → leftward observer rotation
            - Positive d_theta: upward motion on retina → downward observer motion
        
        Args:
            phi: (N,) azimuthal angles of viewing directions  
            theta: (N,) polar angles of viewing directions
            d_phi: (N,) angular flow in phi direction (positive = rightward on retina)
            d_theta: (N,) angular flow in theta direction (positive = upward on retina)
        
        Returns:
            omega_star: (3,) estimated rotation vector [roll, pitch, yaw]
                Positive omega_z = leftward rotation (world moves right on retina)
            T_star: (3,) estimated translation direction (up to sign ambiguity)
            Z_star: (N,) estimated depths (positive when T_star sign is correct)
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

        N = theta.shape[0]

        # === Convert to 3D viewing directions (N,3) ===
        x = torch.stack((
            torch.sin(theta)*torch.cos(phi),
            torch.sin(theta)*torch.sin(phi),
            torch.cos(theta)
        ), dim=1)                              # (N,3)

        # === Convert angular flow to 3D tangent vectors u_i ∈ R^3 ===
        # Basis vectors of sphere
        e_theta = torch.stack((
            torch.cos(theta)*torch.cos(phi),
            torch.cos(theta)*torch.sin(phi),
            -torch.sin(theta)
        ), dim=1)                              # (N,3)

        e_phi = torch.stack((
            -torch.sin(phi),
            torch.cos(phi),
            torch.zeros_like(phi)
        ), dim=1)                              # (N,3)

        # Reconstruct 3D flow from angular components
        # d_phi: positive = rightward motion on retina (leftward observer rotation)
        # d_theta: positive = upward motion on retina (downward observer motion)
        # Negate d_theta because e_theta points downward but we want up=positive
        u = -d_theta[:, None] * e_theta + d_phi[:, None] * e_phi   # (N,3)

        # === Rotation estimation: solve u ≈ [x]_× Ω ===
        # Physical relationship: image motion u = Ω × x = -x × Ω = -[x]_× Ω
        # So [x]_× Ω = -u, meaning Ω = -solve([x]_× Ω = u)
        A = cross_matrix(x)                    # (N,3,3)
        A_flat = A.reshape(3*N, 3)
        u_flat = u.reshape(3*N)

        omega_star = -torch.linalg.lstsq(A_flat, u_flat)[0]  # Negate for correct physical sign

        # === De-rotated flow: v_i = u_i - u_rot where u_rot = -[x]_× Ω ===
        # So v = u - (-[x]_× Ω) = u + [x]_× Ω
        v = u + (A @ omega_star)               # (N,3)
        
        # === Measure translation signal strength ===
        # RMS of de-rotated flow - if near zero, there's no translation
        translation_magnitude = torch.sqrt((v**2).mean())

        # === Projection matrices Π_i = I - x_i x_iᵀ ===
        # Each Pi is (N,3,3)
        outer = x.unsqueeze(2) @ x.unsqueeze(1)   # (N,3,3)
        I = torch.eye(3, device=x.device).expand(N,3,3)
        P = I - outer

        # === Construct stacked C matrix: each block = [v_i]_× Π_i ===
        vx = cross_matrix(v)                   # (N,3,3)
        C_blocks = vx @ P                      # (N,3,3)

        C = C_blocks.reshape(3*N, 3)           # (3N,3)

        # === Solve C T = 0 via smallest singular vector ===
        _, _, Vh = torch.linalg.svd(C, full_matrices=False)
        T_dir = Vh[-1]  # unit direction (sign ambiguous)

        # === Disambiguate sign using physical constraint ===
        # Convention: progressive flow (expansion) = positive T_x (forward motion)
        # For expansion at point x: flow v points away from focus of expansion
        # w = Π @ T points toward focus, so v · w < 0 for expansion with correct T
        # We flip T if v · w > 0 (wrong sign), keeping v · w < 0
        w_i = (P @ T_dir)                      # (N,3)
        v_dot_w = (v * w_i).sum(dim=1)         # (N,)
        
        # Use weighted sum (weight by ||w||² to emphasize reliable points)
        w_norm_sq = (w_i * w_i).sum(dim=1)     # (N,)
        weighted_sign = (v_dot_w * w_norm_sq).sum()
        
        # Flip T if weighted_sign < 0 so that expansion gives positive T
        if weighted_sign < 0:
            T_dir = -T_dir
            w_i = -w_i
            v_dot_w = -v_dot_w

        # Scale T by translation magnitude
        T_star = T_dir * translation_magnitude

        # Depth: Z = ||w||² / (v · w) with v · w > 0 after disambiguation
        Z_star = w_norm_sq / (v_dot_w + 1e-12)

        return omega_star, T_star, Z_star


class CurrentModel(NeuralModel):
    """Current-based neural model where synaptic input is modeled as current injection."""
    
    def __init__(self, synapse_counts_coo, cell_types, neurotransmitters, column_indices, column_phi_theta,
                 excitatory_nt=['ACH'], inhibitory_nt=['GLUT', 'GABA'], nonlinearity='Softplus', device='cpu', dtype='float32',
                 init_new_model=True, dt=0.01, seed=None, init_std=0.1, count_threshold=2, commit_hash=None):
        # Call parent init (handles NT filtering and orphan removal)
        super().__init__(synapse_counts_coo, cell_types, neurotransmitters, column_indices, column_phi_theta,
                         excitatory_nt=excitatory_nt, inhibitory_nt=inhibitory_nt, 
                         nonlinearity=nonlinearity, device=device, dtype=dtype,
                         init_new_model=init_new_model, dt=dt, seed=seed, init_std=init_std, count_threshold=count_threshold, commit_hash=commit_hash)
        
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
        synapses_in_per_neuron = self.synapse_counts.pow(2).sum(dim=1).to_dense()
        synapses_out_per_neuron = self.synapse_counts.pow(2).sum(dim=0).to_dense()
        
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
        self.add_noise_to_parameters()
        
        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()
                self.find_steady_state()

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


class ConductanceModel(NeuralModel):
    """Conductance-based neural model where synaptic input is modeled as conductance changes."""
    
    def __init__(self, synapse_counts_coo, cell_types, neurotransmitters, column_indices, column_phi_theta,
                 excitatory_nt=['ACH'], inhibitory_nt=['GLUT', 'GABA'], nonlinearity='Softplus', device='cpu', dtype='float32', init_new_model=True,
                 dt=0.01, seed=None, init_std=0.1, count_threshold=2, commit_hash=None):
        """
        Initialize a conductance-based neural model.
        
        Args:
            synapse_counts_coo: COO matrix (N x 3 array with [pre_root_id, post_root_id, count])
            cell_types: Array of cell type labels for each neuron
            neurotransmitters: Array of neurotransmitters for each neuron (same size as cell_types)
            column_indices: Array of column indices for each neuron
            column_phi_theta: Array of (phi, theta) coordinates for each column
            excitatory_nt: List of excitatory neurotransmitter names (default: ['ACH'])
            inhibitory_nt: List of inhibitory neurotransmitter names (default: ['GLUT', 'GABA'])
            nonlinearity: Activation function name (default: 'Softplus')
            device: Torch device (default: 'cpu')
            dtype: Torch dtype as string (default: 'float32')
            init_new_model: Do anything a new model needs like finding a baseline (default: True)
            dt: Time step in seconds (default: 0.01)
            seed: Random seed for initialization (default: None)
            init_std: Standard deviation for parameter initialization noise (default: 0.1)
            count_threshold: Minimum synapse count to include a connection (default: 2)
            commit_hash: Git commit hash the model was created with (default: None)
        """
        # Call parent init (handles NT filtering, count_threshold filtering, and orphan removal)
        super().__init__(synapse_counts_coo, cell_types, neurotransmitters, column_indices, column_phi_theta,
                         excitatory_nt=excitatory_nt, inhibitory_nt=inhibitory_nt, 
                         nonlinearity=nonlinearity, device=device, dtype=dtype,
                         init_new_model=init_new_model, dt=dt, seed=seed, init_std=init_std, count_threshold=count_threshold, commit_hash=commit_hash)
        
        # Save init inputs after super().__init__() to overwrite with this class's inputs
        self._save_init_inputs()
        
        # --- Split synapse_counts_coo into per-neurotransmitter sparse matrices ---
        # Use the filtered COO from parent class
        synapse_counts_coo_filtered = self.synapse_counts_coo
        dtype_torch = getattr(torch, dtype)
        
        # Get the neurotransmitter for each presynaptic neuron in the COO matrix (directly from per-neuron array)
        pre_neuron_indices = synapse_counts_coo_filtered[:, 0].astype(np.int64)
        pre_transmitters = self.transmitters[pre_neuron_indices]
        
        # Map from neuron index to cell type index (needed for tracking which cell types receive each transmitter)
        neuron_to_type_idx = self.neuron_to_type_index.cpu().numpy()
        
        # Build a sparse matrix for each neurotransmitter and track which cell types receive each
        self.synapse_counts = []
        # Track which cell types receive input from each transmitter: (num_transmitters, num_cell_types)
        receives_transmitter = torch.zeros((self.num_transmitters, self.num_cell_types), dtype=torch.bool, device=device)
        
        for ri, r in enumerate(self.unique_transmitters):
            # Filter to synapses where presynaptic neuron expresses this transmitter
            mask = (pre_transmitters == r)
            coo_r = synapse_counts_coo_filtered[mask]
            
            rows = torch.tensor(coo_r[:, 1], device=device, dtype=torch.int64)
            cols = torch.tensor(coo_r[:, 0], device=device, dtype=torch.int64)
            values = torch.tensor(coo_r[:, 2], device=device, dtype=dtype_torch)
            
            indices = torch.stack([rows, cols], dim=0)
            sparse_mat = torch.sparse_coo_tensor(indices, values, 
                                                  size=(self.num_neurons, self.num_neurons), 
                                                  dtype=dtype_torch, device=device).coalesce()
            self.synapse_counts.append(sparse_mat)
            
            # Determine which cell types receive this transmitter (have non-zero post-synaptic input)
            post_neuron_indices = coo_r[:, 1].astype(np.int64)
            post_cell_type_indices = neuron_to_type_idx[post_neuron_indices]
            unique_receiving_types = np.unique(post_cell_type_indices)
            receives_transmitter[ri, unique_receiving_types] = True
        
        # --- Precompute mappings for sparse parameter storage ---
        
        # 1. Which transmitter does each cell type EXPRESS (for output_strength)
        #    Shape: (num_cell_types,) with values in [0, num_transmitters)
        cell_type_transmitter_idx = torch.zeros(self.num_cell_types, dtype=torch.int64, device=device)
        for ri, r in enumerate(self.unique_transmitters):
            expressing_mask = (self.cell_type_transmitters == r)
            cell_type_transmitter_idx[expressing_mask] = ri
        self.register_buffer('cell_type_transmitter_idx', cell_type_transmitter_idx)
        
        # 2. Which (transmitter, cell_type) pairs RECEIVE input (for input_strength, rp_transmitter)
        #    Store as boolean mask and as flat indices for scatter
        self.register_buffer('receives_transmitter_mask', receives_transmitter)
        
        # Get indices of active (receiving) pairs for sparse -> dense expansion
        active_indices = receives_transmitter.nonzero(as_tuple=False)  # (num_active, 2)
        self.register_buffer('active_input_transmitter_idx', active_indices[:, 0])  # transmitter indices
        self.register_buffer('active_input_cell_type_idx', active_indices[:, 1])    # cell type indices
        self.num_active_input_pairs = active_indices.shape[0]
        
        print(f"ConductanceModel: {self.num_transmitters} transmitters, {self.num_cell_types} cell types")
        print(f"  Output params: {self.num_cell_types} (one per cell type)")
        print(f"  Input/RP params: {self.num_active_input_pairs} active pairs "
              f"(vs {self.num_transmitters * self.num_cell_types} if dense)")
        
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
        
        # --- Compact parameter storage ---
        # Only store parameters for biologically-relevant (transmitter, cell_type) pairs
        
        # Reversal potential: only for (transmitter, cell_type) pairs that RECEIVE that transmitter
        # Shape: (num_active_input_pairs,)
        rp_transmitter_compact = torch.zeros(self.num_active_input_pairs, dtype=self.dtype, device=self.device)
        for ri, r in enumerate(self.unique_transmitters):
            # Find which entries in the compact array correspond to this transmitter
            mask = (self.active_input_transmitter_idx == ri)
            rp_transmitter_compact[mask] = rp_default[r]
        self.rp_transmitter = rp_transmitter_compact
        
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
        
        # --- Input strength: compact storage for receiving pairs only ---
        # Shape: (num_active_input_pairs,) - only pairs where cell type receives that transmitter
        self.input_strength_log = torch.zeros(self.num_active_input_pairs, dtype=self.dtype, device=self.device)
        
        # --- Output strength: one per cell type (each expresses exactly one transmitter) ---
        # Shape: (num_cell_types,)
        # Initialize based on fan-out from the transmitter that cell type expresses
        output_strength_log_init = torch.zeros(self.num_cell_types, dtype=self.dtype, device=self.device)
        for cti, ct in enumerate(self.unique_cell_types):
            # Get which transmitter this cell type expresses
            ri = self.cell_type_transmitter_idx[cti].item()
            syn_counts = self.synapse_counts[ri]
            synapses_out = syn_counts.pow(2).sum(dim=0).to_dense()
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            fan_out = synapses_out[ct_mask].mean().clamp_min(1.0)
            output_strength_log_init[cti] = -0.5 * torch.log(fan_out)
        self.output_strength_log = output_strength_log_init
        
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
        self.add_noise_to_parameters()
        
        # Initialize steady_state to rp_leak (better starting point than zeros)
        with torch.no_grad():
            rp_leak_all = self.rp_leak[self.neuron_to_type_index]
            self.steady_state.copy_(rp_leak_all)
        
        # Find steady state
        if init_new_model:
            with torch.no_grad():
                self.find_steady_state()

    def firing_rate(self, v):
        """Convert voltage to firing rate with threshold subtraction."""
        threshold_all = self.threshold[self.neuron_to_type_index]
        # Reshape threshold to broadcast with v: find the neuron dimension and expand others to 1
        neuron_dim = v.shape.index(self.num_neurons)
        shape = [1] * v.ndim
        shape[neuron_dim] = self.num_neurons
        threshold_all = threshold_all.view(*shape)
        return self.nonlinearity(v - threshold_all)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        """
        Conductance-based forward pass.
        
        We're implementing this model:
        
        v_{t+1} = v_t - Δt * C_m^{-1} @ [Σ_{r∈{glu,ach,gaba}} (v_t - E_r) ⊙ g_r(v_t) + (v_t - E_leak) ⊙ g_leak - b ⊙ u_t]
        
        g_r(t) = D_{in,r} @ S_r @ D_{out,r} @ f(v(t) - θ)
        
        Where:
            - V_t: membrane voltage at time t
            - C_m: membrane capacitance
            - E_r: reversal potential for neurotransmitter r (glu, ach, gaba)
            - E_leak: leak reversal potential
            - g_r(t): conductance for neurotransmitter r
            - g_leak: leak conductance
            - b: stimulus strength
            - u_t: external stimulus
            - D_{in,r}, D_{out,r}: input/output scaling matrices for neurotransmitter r
            - S_r: synapse count matrix for neurotransmitter r
            - f: nonlinearity (e.g., ReLU/Softplus)
            - θ: firing threshold offset
        
        Args:
            v_prev: tensor with shape (stimuli, neurons)
            u_t: tensor with shape (stimuli, neurons)
            stim_to_cell_mapping: sparse mapping from stimulus to individual neurons
        """
        
        # --- Expand compact parameters to full tensors ---
        
        # Simple per-cell-type params -> per-neuron
        capacitance_inv = self.capacitance_inv_log.exp()[self.neuron_to_type_index]
        rp_leak_all = self.rp_leak[self.neuron_to_type_index]
        g_leak = self.g_leak_log.exp()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        
        # Output strength: (num_cell_types,) -> (num_transmitters, num_neurons)
        # Each cell type only contributes to the transmitter it expresses
        output_strength_full = torch.zeros(self.num_transmitters, self.num_neurons, 
                                           dtype=self.dtype, device=self.device)
        output_strength_exp = self.output_strength_log.exp()  # (num_cell_types,)
        for ri in range(self.num_transmitters):
            # Get neurons whose cell type expresses this transmitter
            expresses_ri = (self.cell_type_transmitter_idx[self.neuron_to_type_index] == ri)
            # Scatter the output strength values for neurons expressing this transmitter
            output_strength_full[ri, expresses_ri] = output_strength_exp[self.neuron_to_type_index[expresses_ri]]
        
        # Input strength: (num_active_input_pairs,) -> (num_transmitters, num_neurons)
        input_strength_full = torch.zeros(self.num_transmitters, self.num_neurons,
                                          dtype=self.dtype, device=self.device)
        input_strength_exp = self.input_strength_log.exp()  # (num_active_input_pairs,)
        # Scatter: active_input_transmitter_idx and active_input_cell_type_idx give (transmitter, cell_type) pairs
        # We need to expand to neurons
        input_strength_by_type = torch.zeros(self.num_transmitters, self.num_cell_types,
                                             dtype=self.dtype, device=self.device)
        input_strength_by_type[self.active_input_transmitter_idx, self.active_input_cell_type_idx] = input_strength_exp
        input_strength_full = input_strength_by_type[:, self.neuron_to_type_index]
        
        # Reversal potential: (num_active_input_pairs,) -> (num_transmitters, num_neurons)
        rp_transmitter_by_type = torch.zeros(self.num_transmitters, self.num_cell_types,
                                             dtype=self.dtype, device=self.device)
        rp_transmitter_by_type[self.active_input_transmitter_idx, self.active_input_cell_type_idx] = self.rp_transmitter
        rp_transmitter_all = rp_transmitter_by_type[:, self.neuron_to_type_index]

        z = torch.zeros_like(v_prev)
        # get the contribution of each neurotransmitter
        # Σ_{r∈{glu,ach,gaba}} (v_t - E_r) ⊙ g_r(v_t)
        firing_rate = self.firing_rate(v_prev)
        for ri, r in enumerate(self.unique_transmitters):
            r_firing_rate = output_strength_full[None, ri, :] * firing_rate
            r_conductance = torch.sparse.mm(self.synapse_counts[ri], r_firing_rate.transpose(0, 1)).transpose(0, 1)
            r_conductance = input_strength_full[None, ri, :] * r_conductance
            z = z + (v_prev - rp_transmitter_all[None, ri, :]) * r_conductance

        # add in the leak current
        # (v_t - E_leak) ⊙ g_leak
        z = z + (v_prev - rp_leak_all[None, :]) * g_leak

        # add in the inputs
        if stim_to_cell_mapping is None:
            z = z - u_t * stimulus_strength[None, :]
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
                z.index_add_(1, ci, -stimulus_strength[None, ci] * u_t[:, si])

        v_next = v_prev - self.dt * capacitance_inv * z

        return v_next
