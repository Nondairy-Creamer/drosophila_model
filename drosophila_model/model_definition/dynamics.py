"""Recurrent dynamics mixins for connectome-based neural models.

Each class here defines a *flavour of dynamics* (Current / Conductance /
CellAnn / HandTuned / Arbitrary) as a mixin that owns its own parameters and
the methods that touch them (``forward``, ``firing_rate``, ``calc_sigma_max``,
``_apply_sigma_scale``, ``silence_neurons``).

The mixins are designed to be combined with :class:`neural_models.ConnectomeModel`
via multiple inheritance — done at runtime by
:func:`neural_models.compose_model_class`::

    ModelCls = compose_model_class('CurrentDynamics')
    # equivalent to: class ModelCls(CurrentDynamics, ConnectomeModel): pass

This module imports nothing from :mod:`neural_models` to keep the dependency
one-way (``neural_models -> dynamics``).

Host interface
==============

The mixins are duck-typed against the following attributes/methods. A
:class:`neural_models.ConnectomeModel` host is what they're written for;
plugging them onto a non-connectome host would require synthesising these.

From ``BaseRnn``:
    ``device``, ``dtype``, ``rng``, ``nonlinearity``, ``dt``, ``init_std``,
    ``add_noise_to_parameters(std)``, ``set_sigma_max()``,
    ``max_allowed_sigma`` (class attr), ``silence_val``,
    ``_save_init_inputs()``.

From ``ConnectomeModel``:
    ``synapse_counts`` (CSR sparse tensor; some mixins replace it with a
    list of CSRs at init), ``synapse_counts_coo`` (numpy COO),
    ``neuron_to_type_index``, ``num_neurons``, ``num_cell_types``,
    ``unique_cell_types``, ``cell_types``, ``transmitters``,
    ``unique_transmitters``, ``num_transmitters``, ``steady_state`` buffer,
    ``_csr_abs_row_col_sums(csr)``, ``_csr_row_indices(csr)``,
    ``_sign_synapses()``, ``_silencing_mask(neurons, invert)``,
    ``_cell_type_name_to_index(name)``.

Connectome-layout contract
==========================

``ConnectomeModel`` sets ``self.synapse_counts`` to a single CSR sparse tensor.
A dynamics mixin may replace this with a Python list of CSR tensors (one per
neurotransmitter, as ``ConductanceDynamics`` does) or mutate it in place (as
``HandTunedDynamics`` does). Code that serialises/restores
``synapse_counts`` already handles both layouts.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch


# ---------------------------------------------------------------------------
# CurrentDynamics
# ---------------------------------------------------------------------------


class CurrentDynamics:
    """Current-based dynamics: synaptic input modelled as current injection."""

    def __init__(self, *, init_new_model=True, **kwargs):
        super().__init__(**kwargs)

        # Convert unsigned connectome counts to signed (excit. +, inhib. -)
        # and apply region-level anatomy-prior sign overrides.
        self._initialize_synapse_signs()

        self._save_init_inputs()

        # Power vector for spectral-norm calculation (used by set_sigma_max).
        power_vec_np = self.rng.standard_normal(self.num_neurons)
        power_vec = torch.tensor(power_vec_np, device=self.device, dtype=self.dtype)
        power_vec = power_vec / (power_vec.norm() + 1e-12)
        self.register_buffer('_power_vec', power_vec)

        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        persistence_init = 0.5
        stimulus_strength_init = 10.0
        offset_init = 0.0

        self.persistence_logit = torch.full(
            (self.num_cell_types,), logit(persistence_init),
            dtype=self.dtype, device=self.device,
        )
        self.stimulus_strength_log = torch.full(
            (self.num_cell_types,), np.log(stimulus_strength_init),
            dtype=self.dtype, device=self.device,
        )
        self.offset = torch.full(
            (self.num_cell_types,), offset_init,
            dtype=self.dtype, device=self.device,
        )

        # Initialise input/output strength per cell type from inverse fan-in /
        # fan-out. CSR row sums come for free from the structure.
        synapses_in_per_neuron, synapses_out_per_neuron = self._csr_abs_row_col_sums(
            self.synapse_counts
        )

        synapses_in_per_cell_type = torch.zeros(
            self.num_cell_types, device=self.device, dtype=self.dtype)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            synapses_in_per_cell_type[cti] = synapses_in_per_neuron[ct_mask].mean().clamp_min(1.0)

        input_strength_init = synapses_in_per_cell_type.pow(-1 / 2)
        self.input_strength_log = torch.log(input_strength_init)

        synapses_out_per_cell_type = torch.zeros(
            self.num_cell_types, device=self.device, dtype=self.dtype)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            synapses_out_per_cell_type[cti] = synapses_out_per_neuron[ct_mask].mean().clamp_min(1.0)
        output_strength_init = synapses_out_per_cell_type.pow(-1 / 2)
        self.output_strength_log = torch.log(output_strength_init)

        self.persistence_logit = torch.nn.Parameter(self.persistence_logit)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.offset = torch.nn.Parameter(self.offset)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.output_strength_log = torch.nn.Parameter(self.output_strength_log)

        self.add_noise_to_parameters(self.init_std)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def _apply_sigma_scale(self, scale_factor):
        """Shift ``output_strength_log`` by ``log(scale_factor)``."""
        self.output_strength_log.add_(torch.log(scale_factor))

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        """Spectral norm of the linearised dynamics matrix."""
        A = self.synapse_counts
        A_T = A.t().to_sparse_csr()
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
            y = torch.sparse.mm(A_T, v_scaled.unsqueeze(1)).squeeze(1)
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

    def silence_neurons(self, neurons_to_silence, invert_silencing=False, silence_inputs=False):
        chosen = self._silencing_mask(neurons_to_silence, invert_silencing)
        if chosen is not None:
            with torch.no_grad():
                self.output_strength_log.data[chosen] = np.log(self.silence_val)
                if silence_inputs:
                    self.input_strength_log.data[chosen] = np.log(self.silence_val)

    def firing_rate(self, v):
        return self.nonlinearity(v)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        input_strength = self.input_strength_log.exp()[self.neuron_to_type_index]
        output_strength = self.output_strength_log.exp()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        offset_all_neurons = self.offset[self.neuron_to_type_index]

        v_scaled = self.nonlinearity(v_prev) * output_strength[None, :]
        z = torch.sparse.mm(self.synapse_counts, v_scaled.transpose(0, 1)).transpose(0, 1)
        z = z * input_strength[None, :]

        if stim_to_cell_mapping is None:
            z = z + u_t * stimulus_strength[None, :]
        else:
            cell_idx = stim_to_cell_mapping['neuron_index']
            stim_idx = stim_to_cell_mapping['stimulus_index']
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                z.index_add_(1, ci, u_t[:, si] * stimulus_strength[None, ci])

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (
            z + offset_all_neurons[None, :]
        )
        return v_next


# ---------------------------------------------------------------------------
# ConductanceDynamics
# ---------------------------------------------------------------------------


class ConductanceDynamics:
    """Conductance-based dynamics: synaptic input modelled as conductance changes."""

    def __init__(self, *, init_new_model=True, **kwargs):
        super().__init__(**kwargs)

        self._save_init_inputs()

        # --- Split synapse_counts_coo into per-neurotransmitter sparse matrices ---
        synapse_counts_coo_filtered = self.synapse_counts_coo
        dtype_torch = self.dtype

        pre_neuron_indices = synapse_counts_coo_filtered[:, 0]
        pre_transmitters = self.transmitters[pre_neuron_indices]

        self.synapse_counts = []
        for ri, r in enumerate(self.unique_transmitters):
            mask = (pre_transmitters == r)
            coo_r = synapse_counts_coo_filtered[mask]

            rows = torch.tensor(coo_r[:, 1], device=self.device, dtype=torch.int64)
            cols = torch.tensor(coo_r[:, 0], device=self.device, dtype=torch.int64)
            values = torch.tensor(coo_r[:, 2], device=self.device, dtype=dtype_torch)

            indices = torch.stack([rows, cols], dim=0)
            sparse_mat = torch.sparse_coo_tensor(
                indices, values,
                size=(self.num_neurons, self.num_neurons),
                dtype=dtype_torch, device=self.device,
            ).coalesce().to_sparse_csr()
            self.synapse_counts.append(sparse_mat)

        self.synapse_counts_T = [A.t().to_sparse_csr() for A in self.synapse_counts]

        print(f"ConductanceDynamics: {self.num_transmitters} transmitters, "
              f"{self.num_cell_types} cell types, {self.num_neurons} neurons")
        print(f"  Output params: {self.num_cell_types} (one per cell type)")
        print(f"  Input/RP params: {self.num_transmitters} x {self.num_cell_types} "
              f"(per transmitter per cell type)")

        # --- Conductance-specific parameter defaults ---
        rp_default = {'ACH': 10.0, 'GLUT': -70.0, 'GABA': -70.0}
        rp_leak_default = -50.0
        g_leak_log_default = np.log(1.0)
        # Membrane time constant tau = C / g_leak. With g_leak=1 and dt in
        # seconds, picking C = 0.04 s gives tau = 40 ms at rest.
        capacitance_inv_log_default = np.log(1.0 / 0.04)
        threshold_default = -50

        unknown_transmitters = set(self.unique_transmitters) - set(rp_default.keys())
        if unknown_transmitters:
            raise ValueError(
                f"Unknown neurotransmitters: {unknown_transmitters}. "
                f"Please add reversal potentials to rp_default for these transmitters."
            )

        # Reversal potential per (transmitter, cell type)
        rp_transmitter_init = torch.zeros(
            self.num_transmitters, self.num_cell_types,
            dtype=self.dtype, device=self.device,
        )
        for ri, r in enumerate(self.unique_transmitters):
            rp_transmitter_init[ri, :] = rp_default[r]

        # HARDCODED OVERRIDE: make GABA look excitatory onto L1 and L2 (cells
        # actually receive histaminergic, sign-inverted input from
        # photoreceptors; flipping GABA's reversal potential is a quick
        # stand-in for that biology).
        if 'GABA' in self.unique_transmitters:
            gaba_idx = int(np.where(self.unique_transmitters == 'GABA')[0][0])
            for ct_name in ('L1', 'L2'):
                ct_match = np.where(self.unique_cell_types == ct_name)[0]
                if ct_match.size > 0:
                    ct_idx = int(ct_match[0])
                    rp_transmitter_init[gaba_idx, ct_idx] = rp_default['ACH']
                    print(f"  [override] GABA reversal potential on {ct_name} "
                          f"set to {rp_default['ACH']} mV (excitatory like ACH)")

        self.rp_transmitter = rp_transmitter_init
        self.rp_leak = torch.full(
            (self.num_cell_types,), rp_leak_default,
            dtype=self.dtype, device=self.device,
        )
        self.g_leak_log = torch.full(
            (self.num_cell_types,), g_leak_log_default,
            dtype=self.dtype, device=self.device,
        )
        self.capacitance_inv_log = torch.full(
            (self.num_cell_types,), capacitance_inv_log_default,
            dtype=self.dtype, device=self.device,
        )
        self.threshold = torch.full(
            (self.num_cell_types,), threshold_default,
            dtype=self.dtype, device=self.device,
        )

        stimulus_strength_init = 30.0
        self.stimulus_strength_log = torch.full(
            (self.num_cell_types,), np.log(stimulus_strength_init),
            dtype=self.dtype, device=self.device,
        )

        # Input/output strength: same fan-in / fan-out scheme as CurrentDynamics,
        # but using totals across all transmitters.
        synapses_in_per_neuron = torch.zeros(
            self.num_neurons, dtype=self.dtype, device=self.device)
        synapses_out_per_neuron = torch.zeros(
            self.num_neurons, dtype=self.dtype, device=self.device)
        for s in self.synapse_counts:
            r, c = self._csr_abs_row_col_sums(s)
            synapses_in_per_neuron = synapses_in_per_neuron + r
            synapses_out_per_neuron = synapses_out_per_neuron + c

        in_log_per_ct = torch.zeros(self.num_cell_types, dtype=self.dtype, device=self.device)
        out_log_per_ct = torch.zeros(self.num_cell_types, dtype=self.dtype, device=self.device)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            fan_in = synapses_in_per_neuron[ct_mask].mean().clamp_min(1.0)
            fan_out = synapses_out_per_neuron[ct_mask].mean().clamp_min(1.0)
            in_log_per_ct[cti] = -0.5 * torch.log(fan_in)
            out_log_per_ct[cti] = -0.5 * torch.log(fan_out)

        self.input_strength_log = (
            in_log_per_ct.unsqueeze(0).expand(self.num_transmitters, -1).clone()
        )
        self.output_strength_log = out_log_per_ct

        # Convert to Parameters
        self.rp_transmitter = torch.nn.Parameter(self.rp_transmitter)
        self.rp_leak = torch.nn.Parameter(self.rp_leak)
        self.g_leak_log = torch.nn.Parameter(self.g_leak_log)
        self.capacitance_inv_log = torch.nn.Parameter(self.capacitance_inv_log)
        self.threshold = torch.nn.Parameter(self.threshold)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.output_strength_log = torch.nn.Parameter(self.output_strength_log)

        # Power vector for spectral-norm calc
        power_vec_np = self.rng.standard_normal(self.num_neurons)
        power_vec = torch.tensor(power_vec_np, device=self.device, dtype=self.dtype)
        power_vec = power_vec / (power_vec.norm() + 1e-12)
        self.register_buffer('_power_vec', power_vec)

        self.add_noise_to_parameters(self.init_std)

        # Initialise steady_state to rp_leak (better starting point than zeros)
        with torch.no_grad():
            rp_leak_all = self.rp_leak[self.neuron_to_type_index]
            self.steady_state.copy_(rp_leak_all)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def silence_neurons(self, neurons_to_silence, invert_silencing=False, silence_inputs=False):
        chosen = self._silencing_mask(neurons_to_silence, invert_silencing)
        if chosen is not None:
            with torch.no_grad():
                self.output_strength_log.data[chosen] = np.log(self.silence_val)
                if silence_inputs:
                    self.input_strength_log.data[:, chosen] = np.log(self.silence_val)

    def firing_rate(self, v):
        """Voltage -> firing rate with per-cell-type threshold subtraction."""
        threshold_all = self.threshold[self.neuron_to_type_index]
        return self.nonlinearity(v - threshold_all)

    def _apply_sigma_scale(self, scale_factor):
        """Shift ``output_strength_log`` by ``log(scale_factor)``."""
        self.output_strength_log.add_(torch.log(scale_factor))

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        """Spectral norm of the linearised one-step Jacobian at rest.

        Linearises around v_0 = E_leak with baseline g_tot = g_leak (no
        synaptic activity yet) and a hardcoded f'(v_0) = 1. ReLU's actual
        derivative at threshold is 0, which would zero out M_eff and make
        set_sigma_max a no-op; using slope 1 bounds the worst-case active
        gain, which is what the spectral target should constrain.
        """
        D_out = self.output_strength_log.exp()[self.neuron_to_type_index]
        D_in = self.input_strength_log.exp()[:, self.neuron_to_type_index]
        E_r = self.rp_transmitter[:, self.neuron_to_type_index]
        E_leak = self.rp_leak[self.neuron_to_type_index]
        g_leak = self.g_leak_log.exp()[self.neuron_to_type_index]
        C_inv = self.capacitance_inv_log.exp()[self.neuron_to_type_index]

        d = torch.exp(-self.dt * g_leak * C_inv)
        one_minus_d = 1.0 - d
        W_r = (E_r - E_leak[None, :]) / g_leak[None, :]

        def apply_M(v):
            v_scaled = v * D_out
            out = torch.zeros_like(v)
            for ri, A_r in enumerate(self.synapse_counts):
                y = torch.sparse.mm(A_r, v_scaled.unsqueeze(1)).squeeze(1)
                out = out + W_r[ri] * D_in[ri] * y
            return out

        def apply_M_T(v):
            out = torch.zeros_like(v)
            for ri in range(len(self.synapse_counts)):
                v_weighted = v * W_r[ri] * D_in[ri]
                y = torch.sparse.mm(
                    self.synapse_counts_T[ri], v_weighted.unsqueeze(1)
                ).squeeze(1)
                out = out + y
            return out * D_out

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

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None, eps=1e-6):
        """Semi-implicit exponential (frozen-conductance) update per step.

        dv/dt = -(g_tot/C) * v + (1/C) * (sum_r g_r E_r + g_leak E_leak + b u)

        => v_next = persistence * v_prev + (1 - persistence) * v_inf
           persistence = exp(-dt * g_tot / C)
           v_inf = (sum_r g_r E_r + g_leak E_leak + b u) / g_tot
        """
        C_inv = self.capacitance_inv_log.exp()[self.neuron_to_type_index]
        E_leak = self.rp_leak[self.neuron_to_type_index]
        g_leak = self.g_leak_log.exp()[self.neuron_to_type_index]
        b = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        out_strength = self.output_strength_log.exp()[self.neuron_to_type_index]
        in_strength = self.input_strength_log.exp()[:, self.neuron_to_type_index]
        E_r_all = self.rp_transmitter[:, self.neuron_to_type_index]

        firing_rate_scaled = out_strength[None, :] * self.firing_rate(v_prev)

        g_syn_sum = torch.zeros_like(v_prev)
        num = (g_leak[None, :] * E_leak[None, :])

        for ri, _ in enumerate(self.unique_transmitters):
            g_r = torch.sparse.mm(
                self.synapse_counts[ri], firing_rate_scaled.transpose(0, 1)
            ).transpose(0, 1)
            g_r = in_strength[ri][None, :] * g_r
            g_syn_sum = g_syn_sum + g_r
            num = num + g_r * E_r_all[ri][None, :]

        g_tot = g_leak[None, :] + g_syn_sum

        if stim_to_cell_mapping is None:
            num = num + b[None, :] * u_t
        else:
            cell_idx = stim_to_cell_mapping['neuron_index']
            stim_idx = stim_to_cell_mapping['stimulus_index']
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                num.index_add_(1, ci, b[ci][None, :] * u_t[:, si])

        persistence = torch.exp(-self.dt * g_tot * C_inv)
        v_inf = num / (g_tot + eps)
        v_next = persistence * v_prev + (1.0 - persistence) * v_inf
        return v_next


# ---------------------------------------------------------------------------
# CellAnnDynamics — per-neuron multi-layer ANN with connectome sparsity
# ---------------------------------------------------------------------------


class CellAnn(torch.nn.Module):
    """Per-neuron multi-layer network using sparse connectome connectivity.

    Each neuron is modelled as a small feed-forward network with layer
    sizes N, N-1, ..., 2, 1. The first layer uses connectome-derived
    sparse connectivity; subsequent layers are dense per-neuron matmuls
    that don't mix across neurons. All weights are shared per cell type.
    Sparse first-layer weights are stored in log space (so always positive)
    and multiplied by the biological sign from the presynaptic
    neurotransmitter.
    """

    def __init__(self, num_neurons, num_cell_types, num_type_pairs,
                 cell_ann_depth, nonlinearity,
                 sparse_idx, neuron_to_type, edge_to_pair, edge_sign,
                 device=None, dtype=None):
        super().__init__()
        self.num_neurons = num_neurons
        self.cell_ann_depth = cell_ann_depth
        self.nonlinearity = nonlinearity

        self.register_buffer('_sparse_idx', sparse_idx)
        self.register_buffer('_neuron_to_type', neuron_to_type)
        self.register_buffer('_edge_to_pair', edge_to_pair)
        self.register_buffer('_edge_sign', edge_sign)

        self.sparse_weights_log = torch.nn.Parameter(
            torch.zeros(num_type_pairs, cell_ann_depth, device=device, dtype=dtype)
        )
        self.sparse_bias = torch.nn.Parameter(
            torch.zeros(num_cell_types, cell_ann_depth, device=device, dtype=dtype)
        )

        self.dense_weights = torch.nn.ParameterList()
        self.dense_biases = torch.nn.ParameterList()
        for k in range(cell_ann_depth - 1):
            in_size = cell_ann_depth - k
            out_size = in_size - 1
            self.dense_weights.append(torch.nn.Parameter(
                torch.zeros(num_cell_types, in_size, out_size,
                            device=device, dtype=dtype)
            ))
            self.dense_biases.append(torch.nn.Parameter(
                torch.zeros(num_cell_types, out_size, device=device, dtype=dtype)
            ))

    def forward(self, v):
        batch_size = v.shape[0]

        fr = self.nonlinearity(v)

        edge_weights = (self.sparse_weights_log[self._edge_to_pair].exp()
                        * self._edge_sign.unsqueeze(-1))

        h = torch.zeros(batch_size, self.num_neurons, self.cell_ann_depth,
                        device=v.device, dtype=v.dtype)
        for n in range(self.cell_ann_depth):
            sparse_mat = torch.sparse_coo_tensor(
                self._sparse_idx, edge_weights[:, n],
                size=(self.num_neurons, self.num_neurons),
                dtype=v.dtype, device=v.device,
            )
            h[:, :, n] = torch.sparse.mm(sparse_mat, fr.transpose(0, 1)).transpose(0, 1)

        h = h + self.sparse_bias[self._neuron_to_type]
        if self.cell_ann_depth > 1:
            h = self.nonlinearity(h)

        for k in range(len(self.dense_weights)):
            w = self.dense_weights[k][self._neuron_to_type]
            b = self.dense_biases[k][self._neuron_to_type]
            h = torch.einsum('bxi,xio->bxo', h, w) + b
            if k < len(self.dense_weights) - 1:
                h = self.nonlinearity(h)

        return h.squeeze(-1)


class CellAnnDynamics:
    """Each neuron is a small feed-forward ANN constrained by the connectome.

    Forward dynamics:

        v_{t+1} = p * v_t + (1 - p) * (cell_ann(v_t) + b + c * u_t)
    """

    def __init__(self, *, init_new_model=True, cell_ann_depth=1, **kwargs):
        super().__init__(**kwargs)

        self._initialize_synapse_signs()

        self._save_init_inputs()

        sc_coo = self.synapse_counts.to_sparse_coo()
        sparse_indices = sc_coo.indices()
        edge_sign = torch.sign(sc_coo.values())

        edge_post_type = self.neuron_to_type_index[sparse_indices[0]]
        edge_pre_type = self.neuron_to_type_index[sparse_indices[1]]
        pair_keys = edge_post_type * self.num_cell_types + edge_pre_type
        unique_keys, edge_to_pair = torch.unique(pair_keys, return_inverse=True)
        num_type_pairs = unique_keys.shape[0]

        self.cell_ann = CellAnn(
            num_neurons=self.num_neurons,
            num_cell_types=self.num_cell_types,
            num_type_pairs=num_type_pairs,
            cell_ann_depth=cell_ann_depth,
            nonlinearity=self.nonlinearity,
            sparse_idx=sparse_indices,
            neuron_to_type=self.neuron_to_type_index,
            edge_to_pair=edge_to_pair,
            edge_sign=edge_sign,
            device=self.device,
            dtype=self.dtype,
        )

        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        persistence_init = 0.5
        stimulus_strength_init = 10.0
        offset_init = 0.0

        self.persistence_logit = torch.nn.Parameter(
            torch.full((self.num_cell_types,), logit(persistence_init),
                       dtype=self.dtype, device=self.device)
        )
        self.stimulus_strength_log = torch.nn.Parameter(
            torch.full((self.num_cell_types,), np.log(stimulus_strength_init),
                       dtype=self.dtype, device=self.device)
        )
        self.offset = torch.nn.Parameter(
            torch.full((self.num_cell_types,), offset_init,
                       dtype=self.dtype, device=self.device)
        )

        # Initialise sparse first-layer weights to match CurrentDynamics's
        # effective per-edge magnitude (keeps biologically strong pathways
        # hot at init so signal propagates through the feed-forward path).
        synapses_in_per_neuron = sc_coo.abs().sum(dim=1).to_dense()
        synapses_out_per_neuron = sc_coo.abs().sum(dim=0).to_dense()
        synapses_in_per_cell_type = torch.zeros(
            self.num_cell_types, device=self.device, dtype=self.dtype)
        synapses_out_per_cell_type = torch.zeros(
            self.num_cell_types, device=self.device, dtype=self.dtype)
        for cti, ct in enumerate(self.unique_cell_types):
            ct_mask = torch.from_numpy(self.cell_types == ct).to(device=self.device)
            synapses_in_per_cell_type[cti] = synapses_in_per_neuron[ct_mask].mean().clamp_min(1.0)
            synapses_out_per_cell_type[cti] = synapses_out_per_neuron[ct_mask].mean().clamp_min(1.0)

        edge_abs_counts = sc_coo.values().abs().to(self.dtype)
        pair_count_sum = torch.zeros(num_type_pairs, dtype=self.dtype, device=self.device)
        pair_edge_count = torch.zeros(num_type_pairs, dtype=self.dtype, device=self.device)
        pair_count_sum.index_add_(0, edge_to_pair, edge_abs_counts)
        pair_edge_count.index_add_(0, edge_to_pair, torch.ones_like(edge_abs_counts))
        pair_mean_count = (pair_count_sum / pair_edge_count.clamp_min(1.0)).clamp_min(1.0)

        pair_post_type = unique_keys // self.num_cell_types
        pair_pre_type = unique_keys % self.num_cell_types
        fan_in = synapses_in_per_cell_type[pair_post_type]
        fan_out = synapses_out_per_cell_type[pair_pre_type]
        sparse_weights_log_init = (
            torch.log(pair_mean_count)
            - 0.5 * torch.log(fan_in)
            - 0.5 * torch.log(fan_out)
        ).unsqueeze(-1)
        self.cell_ann.sparse_weights_log.data.copy_(
            sparse_weights_log_init.expand(-1, cell_ann_depth)
        )

        for k, w in enumerate(self.cell_ann.dense_weights):
            in_size = cell_ann_depth - k
            out_size = in_size - 1
            std = (2.0 / (in_size + out_size)) ** 0.5
            w_np = self.rng.normal(0.0, std, w.shape)
            w.data.copy_(torch.tensor(w_np, dtype=w.dtype, device=w.device))

        power_vec_np = self.rng.standard_normal(self.num_neurons)
        power_vec = torch.tensor(power_vec_np, device=self.device, dtype=self.dtype)
        power_vec = power_vec / (power_vec.norm() + 1e-12)
        self.register_buffer('_power_vec', power_vec)

        self.add_noise_to_parameters(self.init_std)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def _apply_sigma_scale(self, scale_factor):
        """Shift all sparse_weights_log uniformly to scale recurrent gain."""
        self.cell_ann.sparse_weights_log.add_(torch.log(scale_factor))

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        """Spectral norm of the linearised one-step Jacobian.

        Linearises assuming ``cell_ann_depth == 1`` and ``f'(v) = 1``. Only
        the n=0 sparse hidden unit's magnitudes feed the calculation, but
        ``_apply_sigma_scale`` shifts every column equally so the rescaling
        stays well-defined.
        """
        sparse_idx = self.cell_ann._sparse_idx
        edge_sign = self.cell_ann._edge_sign
        edge_to_pair = self.cell_ann._edge_to_pair

        edge_w = self.cell_ann.sparse_weights_log[edge_to_pair, 0].exp() * edge_sign
        A = torch.sparse_coo_tensor(
            sparse_idx, edge_w,
            size=(self.num_neurons, self.num_neurons),
            dtype=self.dtype, device=self.device,
        ).coalesce().to_sparse_csr()
        A_T = A.t().to_sparse_csr()

        d = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        one_minus_d = 1.0 - d

        def apply_M(v):
            return torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)

        def apply_M_T(v):
            return torch.sparse.mm(A_T, v.unsqueeze(1)).squeeze(1)

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
        return self.nonlinearity(v)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        offset = self.offset[self.neuron_to_type_index]

        z = self.cell_ann(v_prev)

        stim = torch.zeros_like(v_prev)
        if stim_to_cell_mapping is None:
            stim = u_t * stimulus_strength[None, :]
        else:
            cell_idx = stim_to_cell_mapping['neuron_index']
            stim_idx = stim_to_cell_mapping['stimulus_index']
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                stim.index_add_(1, ci, u_t[:, si] * stimulus_strength[None, ci])

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (
            z + offset[None, :] + stim
        )
        return v_next


# ---------------------------------------------------------------------------
# HandTunedDynamics
# ---------------------------------------------------------------------------


class HandTunedDynamics:
    """Hand-tuned model with per-cell-type dynamics parameters.

    Implements Julia optic-lobe-dynamics-style Euler integration:

        dx/dt = -x/tau + f_type(omega * W_norm @ x) + stimulus + offset

    Discretised (forward Euler):

        x_new = x + dt * (-x/tau + f_type(h) + stimulus + offset)

    where ``h = omega * W_norm @ x`` and ``W_norm`` is the signed synapse
    matrix with each neuron's incoming weights normalised to absolute sum 1.
    """

    ACTIVATION_NAMES = ['tanh', 'relu', 'softplus', 'identity', 'sigmoid']
    records_synaptic_current = True

    def __init__(self, *, init_new_model=True,
                 tau_overrides=None, offset_overrides=None,
                 activation_overrides=None,
                 synapse_amplification=None,
                 omega_init=1.0, tau_default=0.05,
                 stimulus_strength_init=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._save_init_inputs()

        # Snapshot the raw (unsigned, unnormalised) connectome so we can
        # re-derive the weight matrix when amplification factors change.
        self._raw_synapse_counts = self.synapse_counts.clone()
        self._sign_flips = []

        self._initialize_synapse_signs()
        self.activation_overrides = activation_overrides

        # Per-presynaptic-cell-type amplification (factor > 1 makes a cell
        # type's synapses occupy a larger share of each postsynaptic neuron's
        # normalised input).
        self.register_buffer(
            '_synapse_amplification',
            torch.ones(self.num_cell_types, dtype=self.dtype, device=self.device),
        )
        if synapse_amplification:
            for ct_name, factor in synapse_amplification.items():
                idx = self._cell_type_name_to_index(ct_name)
                if idx is not None:
                    self._synapse_amplification[idx] = factor

        # Sign-based amplification: scales by the actual sign in the matrix
        # (after NT signing + manual flips), so flipped connections count too.
        self.register_buffer(
            '_negative_amplification',
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            '_positive_amplification',
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )

        self._normalize_synapse_counts()

        self.omega = torch.nn.Parameter(
            torch.tensor(omega_init, dtype=self.dtype, device=self.device))

        tau_values = torch.full((self.num_cell_types,), tau_default,
                                dtype=self.dtype, device=self.device)
        if tau_overrides:
            for ct_name, tau_val in tau_overrides.items():
                idx = self._cell_type_name_to_index(ct_name)
                if idx is not None:
                    tau_values[idx] = tau_val
        self.tau = torch.nn.Parameter(tau_values)

        offset_values = torch.zeros(self.num_cell_types,
                                    dtype=self.dtype, device=self.device)
        if offset_overrides:
            for ct_name, off_val in offset_overrides.items():
                idx = self._cell_type_name_to_index(ct_name)
                if idx is not None:
                    offset_values[idx] = off_val
        self.offset = torch.nn.Parameter(offset_values)

        act_indices = torch.zeros(self.num_cell_types,
                                  dtype=torch.int64, device=self.device)
        if activation_overrides:
            for ct_name, act_name in activation_overrides.items():
                idx = self._cell_type_name_to_index(ct_name)
                if idx is not None:
                    act_indices[idx] = self.ACTIVATION_NAMES.index(act_name)
        self.register_buffer('activation_type', act_indices)

        self.stimulus_strength = torch.nn.Parameter(
            torch.full((self.num_cell_types,), stimulus_strength_init,
                       dtype=self.dtype, device=self.device))

        with torch.no_grad():
            self.omega.clamp_(min=1e-6)
            self.tau.clamp_(min=1e-6)
            self.stimulus_strength.clamp_(min=1e-6)

    # ----- serialisation -------------------------------------------------

    def _collect_extra_state(self):
        state = super()._collect_extra_state()
        state['_raw_synapse_counts'] = self._raw_synapse_counts
        state['_sign_flips'] = self._sign_flips
        return state

    def _restore_extra_state(self, state, device, dtype):
        super()._restore_extra_state(state, device, dtype)
        if '_raw_synapse_counts' in state:
            self._raw_synapse_counts = state['_raw_synapse_counts'].to(
                device=device, dtype=dtype)
        if '_sign_flips' in state:
            self._sign_flips = state['_sign_flips']

    # ----- weight-matrix normalisation -----------------------------------

    def _normalize_synapse_counts(self):
        """Normalise each neuron's incoming weights to absolute-sum 1.

        After this, ``synapse_counts[i, j]`` is the signed fraction of
        neuron i's total input that comes from neuron j. Per-cell-type
        amplification and sign-based amplification are applied before
        normalising.
        """
        sc = self.synapse_counts
        col_indices = sc.col_indices()
        values = sc.values()
        row_indices = self._csr_row_indices(sc)

        if hasattr(self, '_synapse_amplification'):
            pre_type = self.neuron_to_type_index[col_indices]
            values = values * self._synapse_amplification[pre_type]

        if hasattr(self, '_negative_amplification') and self._negative_amplification != 1.0:
            values = torch.where(values < 0,
                                 values * self._negative_amplification,
                                 values)
        if hasattr(self, '_positive_amplification') and self._positive_amplification != 1.0:
            values = torch.where(values > 0,
                                 values * self._positive_amplification,
                                 values)

        row_abs_sums = torch.zeros(self.num_neurons, device=self.device, dtype=self.dtype)
        row_abs_sums.index_add_(0, row_indices, values.abs())
        row_abs_sums.clamp_(min=1.0)

        normalized_values = values / row_abs_sums[row_indices]
        self.synapse_counts = torch.sparse_csr_tensor(
            sc.crow_indices(), col_indices, normalized_values, size=sc.size(),
        )

    # ----- convenience setters -------------------------------------------

    def set_tau(self, cell_type_name, value):
        assert value > 0, f"tau must be positive, got {value}"
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            with torch.no_grad():
                self.tau[idx] = value

    def set_offset(self, cell_type_name, value):
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            with torch.no_grad():
                self.offset[idx] = value

    def set_activation(self, cell_type_name, activation_name):
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            self.activation_type[idx] = self.ACTIVATION_NAMES.index(activation_name)

    def set_default_activation(self, activation_name):
        with torch.no_grad():
            self.activation_type[:] = self.ACTIVATION_NAMES.index(activation_name)
            if self.activation_overrides:
                for ct_name, act_name in self.activation_overrides.items():
                    idx = self._cell_type_name_to_index(ct_name)
                    if idx is not None:
                        self.activation_type[idx] = self.ACTIVATION_NAMES.index(act_name)

    def set_omega(self, value):
        assert value > 0, f"omega must be positive, got {value}"
        with torch.no_grad():
            self.omega.fill_(value)

    def set_stimulus_strength(self, cell_type_name, value):
        assert value > 0, f"stimulus_strength must be positive, got {value}"
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            with torch.no_grad():
                self.stimulus_strength[idx] = value

    def set_synapse_amplification(self, cell_type_name, value, renormalize=True):
        """Set the amplification factor for a presynaptic cell type."""
        assert value > 0, f"synapse_amplification must be positive, got {value}"
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is not None:
            self._synapse_amplification[idx] = value
            if renormalize:
                self._renormalize_from_raw()

    def amplify_inhibitory(self, factor):
        """Amplify all functionally inhibitory (negative) synapses."""
        assert factor > 0, f"factor must be positive, got {factor}"
        self._negative_amplification.fill_(factor)
        self._renormalize_from_raw()

    def amplify_excitatory(self, factor):
        """Amplify all functionally excitatory (positive) synapses."""
        assert factor > 0, f"factor must be positive, got {factor}"
        self._positive_amplification.fill_(factor)
        self._renormalize_from_raw()

    def _renormalize_from_raw(self):
        """Re-sign, re-flip, and re-normalise from the raw connectome snapshot.

        Region anatomy overrides are applied inside
        :meth:`ConnectomeModel._initialize_synapse_signs`; user-driven
        ``_sign_flips`` are replayed on top, in the order they were
        registered.
        """
        self.synapse_counts = self._raw_synapse_counts.clone()
        self._initialize_synapse_signs()
        for pre, post, fac in self._sign_flips:
            self._apply_sign_flip(pre, post, fac)
        self._normalize_synapse_counts()

    def flip_sign_between_cell_types(
        self, presynaptic_cell_type, postsynaptic_cell_type, factor=-1.0
    ):
        """Scale weights for edges from ``presynaptic`` -> ``postsynaptic``.

        Default factor is -1 (flip sign). Recorded for replay across
        ``_renormalize_from_raw``. ``_apply_sign_flip`` itself is inherited
        from :class:`ConnectomeModel`.
        """
        self._sign_flips.append((presynaptic_cell_type, postsynaptic_cell_type, factor))
        self._apply_sign_flip(presynaptic_cell_type, postsynaptic_cell_type, factor)

    # ----- ablation ------------------------------------------------------

    def ablate_cell_type(self, cell_type_name):
        """Zero all incoming and outgoing synapses for a cell type."""
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is None:
            return

        neuron_mask = self.neuron_to_type_index == idx

        for sc_attr in ('synapse_counts', '_raw_synapse_counts'):
            sc = getattr(self, sc_attr)
            pre_n = sc.col_indices()
            post_n = self._csr_row_indices(sc)
            values = sc.values().clone()

            hit = neuron_mask[post_n] | neuron_mask[pre_n]
            values[hit] = 0.0

            setattr(self, sc_attr, torch.sparse_csr_tensor(
                sc.crow_indices(), pre_n, values, size=sc.size(),
            ))

        self._renormalize_from_raw()
        print(f"Ablated cell type '{cell_type_name}' "
              f"({neuron_mask.sum().item()} neurons disconnected)")

    # ----- inspection ----------------------------------------------------

    def get_omega(self):
        return self.omega.item()

    def get_cell_type_params(self, cell_type_name):
        idx = self._cell_type_name_to_index(cell_type_name)
        if idx is None:
            return None
        return {
            'tau': self.tau[idx].item(),
            'offset': self.offset[idx].item(),
            'activation': self.ACTIVATION_NAMES[self.activation_type[idx].item()],
            'stimulus_strength': self.stimulus_strength[idx].item(),
            'synapse_amplification': self._synapse_amplification[idx].item(),
        }

    def _cell_type_column_mapping(self, cell_type_name):
        mask = self.cell_types == cell_type_name
        neuron_idx = np.where(mask)[0]
        col_idx = self.column_indices[mask]
        valid = ~np.isnan(col_idx)
        return neuron_idx[valid].astype(int), col_idx[valid].astype(int)

    def get_cell_voltage(self, v, cell_type_name):
        squeeze = v.dim() == 1
        if squeeze:
            v = v.unsqueeze(0)
        neuron_idx, col_idx = self._cell_type_column_mapping(cell_type_name)
        out = torch.full((v.shape[0], self.num_columns), float('nan'),
                         device=v.device, dtype=v.dtype)
        if len(neuron_idx) > 0:
            ni = torch.tensor(neuron_idx, device=v.device, dtype=torch.int64)
            ci = torch.tensor(col_idx, device=v.device, dtype=torch.int64)
            out[:, ci] = v[:, ni]
        return out.squeeze(0) if squeeze else out

    # ----- per-cell-type activation --------------------------------------

    def _apply_activation(self, h):
        """Apply per-cell-type activation to ``h``."""
        unique_acts = self.activation_type.unique()

        if len(unique_acts) == 1:
            act = unique_acts.item()
            if act == 0:
                return torch.tanh(h)
            if act == 1:
                return torch.relu(h)
            if act == 2:
                return torch.nn.functional.softplus(h)
            if act == 3:
                return h
            if act == 4:
                return torch.sigmoid(h)
            return h

        act_per_neuron = self.activation_type[self.neuron_to_type_index]
        result = torch.tanh(h)
        for act_idx in unique_acts.tolist():
            if act_idx == 0:
                continue
            mask = (act_per_neuron == act_idx).unsqueeze(0)
            if act_idx == 1:
                result = torch.where(mask, torch.relu(h), result)
            elif act_idx == 2:
                result = torch.where(mask, torch.nn.functional.softplus(h), result)
            elif act_idx == 3:
                result = torch.where(mask, h, result)
            elif act_idx == 4:
                result = torch.where(mask, torch.sigmoid(h), result)
        return result

    # ----- forward / firing rate -----------------------------------------

    def firing_rate(self, v):
        return self._apply_activation(v)

    def compute_synaptic_current(self, v):
        """Total synaptic current into each neuron: omega * W_norm @ f(v)."""
        nonlin = self._apply_activation(v)
        h = torch.sparse.mm(self.synapse_counts, nonlin.transpose(0, 1)).transpose(0, 1)
        return self.omega * h

    def decompose_synaptic_input(self, v, target_cell_type):
        """Decompose synaptic input to ``target_cell_type`` by presynaptic type."""
        neuron_idx, col_idx = self._cell_type_column_mapping(target_cell_type)
        if len(neuron_idx) == 0:
            return {}

        ti = torch.tensor(neuron_idx, device=self.device, dtype=torch.int64)
        ci = torch.tensor(col_idx, device=self.device, dtype=torch.int64)
        B = v.shape[0]

        nonlin = self._apply_activation(v)

        sc = self.synapse_counts.to_sparse_coo()
        sc_idx = sc.indices()
        sc_val = sc.values()

        row_mask = torch.isin(sc_idx[0], ti)
        sub_rows = sc_idx[0, row_mask]
        sub_cols = sc_idx[1, row_mask]
        sub_vals = sc_val[row_mask]

        if sub_vals.numel() == 0:
            return {}

        neuron_to_col = torch.full((self.num_neurons,), -1,
                                   device=self.device, dtype=torch.int64)
        neuron_to_col[ti] = ci
        target_cols = neuron_to_col[sub_rows]

        pre_type_idx = self.neuron_to_type_index[sub_cols]

        result = {}
        for tidx in pre_type_idx.unique():
            mask = pre_type_idx == tidx
            w = sub_vals[mask]
            pre_n = sub_cols[mask]
            tgt_c = target_cols[mask]

            weighted = self.omega * w.unsqueeze(0) * nonlin[:, pre_n]

            out = torch.full((B, self.num_columns), float('nan'),
                             device=self.device, dtype=self.dtype)
            out[:, ci] = 0.0
            out.scatter_add_(1, tgt_c.unsqueeze(0).expand(B, -1), weighted)

            type_name = self.unique_cell_types[tidx.item()]
            result[type_name] = out

        return result

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        """One Euler step:

        x_new = x + dt * (-x/tau + f(omega * W_norm @ x) + stim + offset)
        """
        tau = self.tau[self.neuron_to_type_index]
        omega = self.omega
        stimulus_strength = self.stimulus_strength[self.neuron_to_type_index]
        offset_all = self.offset[self.neuron_to_type_index]

        stim = torch.zeros_like(v_prev)
        if stim_to_cell_mapping is None:
            stim = u_t * stimulus_strength[None, :]
        else:
            cell_idx = stim_to_cell_mapping['neuron_index']
            stim_idx = stim_to_cell_mapping['stimulus_index']
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                stim.index_add_(1, ci, u_t[:, si] * stimulus_strength[None, ci])

        nonlin = self._apply_activation(v_prev)
        h = torch.sparse.mm(self.synapse_counts, nonlin.transpose(0, 1)).transpose(0, 1)
        h_nonlin = omega * h
        v_next = v_prev + self.dt / tau[None, :] * (
            -v_prev + h_nonlin + stim + offset_all[None, :]
        )
        return v_next


# ---------------------------------------------------------------------------
# ArbitraryDynamics — all-to-all per-cell-type weight matrix
# ---------------------------------------------------------------------------


class ArbitraryDynamics:
    """All-to-all dynamics with learned per-cell-type weight matrix.

    Like CurrentDynamics but without connectome sparsity or sign constraints.
    Effective connectivity ``A[i, j] = W[type(i), type(j)]``.

        v_{t+1} = p * v_t + (1 - p) * (A @ g(v_t) + b + c * u_t)
    """

    def __init__(self, *, init_new_model=True, **kwargs):
        super().__init__(**kwargs)
        self._save_init_inputs()

        power_vec_np = self.rng.standard_normal(self.num_neurons)
        power_vec = torch.tensor(power_vec_np, device=self.device, dtype=self.dtype)
        power_vec = power_vec / (power_vec.norm() + 1e-12)
        self.register_buffer('_power_vec', power_vec)

        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        persistence_init = 0.5
        stimulus_strength_init = 1.0
        offset_init = 0.0

        self.persistence_logit = torch.nn.Parameter(
            torch.full((self.num_cell_types,), logit(persistence_init),
                       dtype=self.dtype, device=self.device)
        )
        self.stimulus_strength_log = torch.nn.Parameter(
            torch.full((self.num_cell_types,), np.log(stimulus_strength_init),
                       dtype=self.dtype, device=self.device)
        )
        self.offset = torch.nn.Parameter(
            torch.full((self.num_cell_types,), offset_init,
                       dtype=self.dtype, device=self.device)
        )

        self.W = torch.nn.Parameter(
            torch.zeros(self.num_cell_types, self.num_cell_types,
                        dtype=self.dtype, device=self.device)
        )

        self.add_noise_to_parameters(self.init_std)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def _apply_A(self, v, transpose=False):
        """Compute ``A @ v`` where ``A[i, j] = W[type(i), type(j)]``."""
        if v.dim() == 1:
            idx = self.neuron_to_type_index
        else:
            idx = self.neuron_to_type_index.unsqueeze(0).expand_as(v)

        v_by_type = torch.zeros(*v.shape[:-1], self.num_cell_types,
                                device=v.device, dtype=v.dtype)
        v_by_type.scatter_add_(-1, idx, v)

        if transpose:
            z_by_type = v_by_type @ self.W
        else:
            z_by_type = v_by_type @ self.W.T

        return z_by_type[..., self.neuron_to_type_index]

    def _apply_sigma_scale(self, scale_factor):
        self.W.mul_(scale_factor)

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        d = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        one_minus_d = 1.0 - d

        def apply_M(v):
            return self._apply_A(v)

        def apply_M_T(v):
            return self._apply_A(v, transpose=True)

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
        return self.nonlinearity(v)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        offset = self.offset[self.neuron_to_type_index]

        fr = self.nonlinearity(v_prev)
        z = self._apply_A(fr)

        stim = torch.zeros_like(v_prev)
        if stim_to_cell_mapping is None:
            stim = u_t * stimulus_strength[None, :]
        else:
            cell_idx = stim_to_cell_mapping['neuron_index']
            stim_idx = stim_to_cell_mapping['stimulus_index']
            valid = (cell_idx >= 0) & (stim_idx >= 0)
            if torch.any(valid):
                ci = cell_idx[valid]
                si = stim_idx[valid]
                stim.index_add_(1, ci, u_t[:, si] * stimulus_strength[None, ci])

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (
            z + offset[None, :] + stim
        )
        return v_next
