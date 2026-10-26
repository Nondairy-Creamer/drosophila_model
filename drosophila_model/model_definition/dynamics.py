"""Recurrent dynamics mixins for connectome-based neural models.

Each class here defines a *flavour of dynamics* (Current / CurrentDivisive /
Conductance / CellAnn / Arbitrary) as a mixin that owns its own parameters and
the methods that touch them (``forward``, ``firing_rate``, ``calc_sigma_max``,
``_apply_sigma_scale`` and ``silence_neurons``). Select one with
``[model].dynamics_type``.

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
    ``_csr_abs_row_col_sums(csr)``, ``_initialize_synapse_signs()``,
    ``_silencing_mask(neurons, invert)``.

Connectome-layout contract
==========================

``ConnectomeModel`` sets ``self.synapse_counts`` to a single CSR sparse tensor.
A dynamics mixin may replace this with a Python list of CSR tensors (one per
neurotransmitter, as ``ConductanceDynamics`` does). Code that serialises/restores
``synapse_counts`` already handles both layouts. Plain (non-parameter,
non-buffer) tensor attributes a mixin adds must be listed in
``BaseRnn._plain_tensor_attrs`` so ``model.to()`` moves them.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers shared by the mixins
# ---------------------------------------------------------------------------


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _init_power_vec(model):
    """Register the random unit-norm ``_power_vec`` buffer for ``calc_sigma_max``.

    Draws from ``model.rng``, so each ``__init__`` calls it at a fixed point
    to keep seeded builds reproducible. The buffer name is part of saved
    checkpoints.
    """
    power_vec_np = model.rng.standard_normal(model.num_neurons)
    power_vec = torch.tensor(power_vec_np, device=model.device, dtype=model.dtype)
    power_vec = power_vec / (power_vec.norm() + 1e-12)
    model.register_buffer('_power_vec', power_vec)


def _spectral_norm(model, apply_M, apply_M_T, d, max_iter, eps, rtol=1e-6):
    """Spectral norm of ``S = diag(d) + diag(1 - d) @ M`` by power iteration.

    ``d`` is the per-neuron persistence and ``apply_M`` / ``apply_M_T`` apply
    the linearised recurrent operator and its transpose. Warm-starts from,
    and updates, ``model._power_vec``; stops early once the estimate changes
    by less than ``rtol`` (relative) between iterations.
    """
    one_minus_d = 1.0 - d

    def apply_S(v):
        return d * v + one_minus_d * apply_M(v)

    def apply_S_T(v):
        return d * v + apply_M_T(one_minus_d * v)

    x = model._power_vec.clone()
    x = x / (x.norm() + eps)

    prev = None
    for _ in range(max_iter):
        y = apply_S(x)
        sigma = y.norm()
        x = apply_S_T(y)
        x = x / (x.norm() + eps)
        if prev is not None and abs(sigma - prev) <= rtol * sigma:
            break
        prev = sigma

    y = apply_S(x)
    sigma = y.norm()

    model._power_vec.copy_(x.detach())
    return sigma


def _stimulus_drive(u_t, gain, stim_to_cell_mapping):
    """Per-neuron stimulus drive ``(B, num_neurons)``.

    Without a mapping ``u_t`` is already per-neuron and the drive is
    ``gain * u_t``. Otherwise stimulus channel ``stimulus_index[k]`` drives
    neuron ``neuron_index[k]`` with that neuron's ``gain`` (entries with a
    negative index are skipped) and every other neuron gets zero.
    """
    if stim_to_cell_mapping is None:
        return u_t * gain[None, :]
    drive = torch.zeros(u_t.shape[0], gain.shape[0], device=u_t.device,
                        dtype=torch.promote_types(u_t.dtype, gain.dtype))
    cell_idx = stim_to_cell_mapping['neuron_index']
    stim_idx = stim_to_cell_mapping['stimulus_index']
    valid = (cell_idx >= 0) & (stim_idx >= 0)
    if torch.any(valid):
        ci = cell_idx[valid]
        si = stim_idx[valid]
        drive.index_add_(1, ci, u_t[:, si] * gain[None, ci])
    return drive


def _mean_per_cell_type(model, per_neuron):
    """Mean of ``per_neuron`` over each cell type's neurons, clamped to >= 1."""
    out = torch.zeros(model.num_cell_types, device=model.device, dtype=model.dtype)
    for cti, ct in enumerate(model.unique_cell_types):
        ct_mask = torch.from_numpy(model.cell_types == ct).to(device=model.device)
        out[cti] = per_neuron[ct_mask].mean().clamp_min(1.0)
    return out


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
        _init_power_vec(self)

        persistence_init = 0.5
        stimulus_strength_init = 10.0
        offset_init = 0.0

        self.persistence_logit = torch.full(
            (self.num_cell_types,), _logit(persistence_init),
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
        synapses_in_per_cell_type = _mean_per_cell_type(self, synapses_in_per_neuron)
        input_strength_init = synapses_in_per_cell_type.pow(-1 / 2)
        self.input_strength_log = torch.log(input_strength_init)

        synapses_out_per_cell_type = _mean_per_cell_type(self, synapses_out_per_neuron)
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

        def apply_M(v):
            v_scaled = v * D_out
            y = torch.sparse.mm(A, v_scaled.unsqueeze(1)).squeeze(1)
            return y * D_in

        def apply_M_T(v):
            v_scaled = v * D_in
            y = torch.sparse.mm(A_T, v_scaled.unsqueeze(1)).squeeze(1)
            return y * D_out

        return _spectral_norm(self, apply_M, apply_M_T, d, max_iter, eps)

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
        z = z + _stimulus_drive(u_t, stimulus_strength, stim_to_cell_mapping)

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (
            z + offset_all_neurons[None, :]
        )
        return v_next


# ---------------------------------------------------------------------------
# CurrentDivisiveDynamics
# ---------------------------------------------------------------------------


class CurrentDivisiveDynamics:
    """Current-based dynamics with divisive normalisation by inhibition.

    Update rule::

        v_{t+1} = p * v_t + (1 - p) * (
            (W+ @ g(v_t) + stim) / (W- @ g(v_t) + sigma)
            + offset
        )

    where ``W+`` / ``W-`` are the positive and negative parts of the
    signed connectome weight matrix (``W-`` stored as a positive-magnitude
    sparse tensor). Synaptic weights and per-cell-type input/output
    strengths are parameterised exactly like :class:`CurrentDynamics`;
    ``sigma`` is a per-cell-type positive semi-saturation constant
    (stored in log space) that sets the gain at low inhibitory drive.

    The numerator and denominator both feed off the same strength-scaled
    presynaptic firing rates, so this is a Carandini-Heeger-style
    canonical normalisation rather than a shunting-conductance model
    (compare :class:`ConductanceDynamics`, where division by ``g_tot``
    falls out of biophysical reversal-potential dynamics). The stimulus
    enters the numerator alongside the excitatory recurrent drive, so the
    feedforward input is divisively normalised by inhibition too — the
    Carandini-Heeger form where the driving input is what gets normalised.
    Offset is added *outside* the ratio.
    """

    def __init__(self, *, init_new_model=True, **kwargs):
        super().__init__(**kwargs)

        # Sign synapses (excit. +, inhib. -) and apply region anatomy
        # overrides before splitting; identical to CurrentDynamics so the
        # signs and overrides stay consistent across dynamics flavours.
        self._initialize_synapse_signs()

        self._save_init_inputs()

        self._build_signed_split()

        # Power vector for spectral-norm calculation (used by set_sigma_max).
        _init_power_vec(self)

        persistence_init = 0.5
        stimulus_strength_init = 10.0
        offset_init = 0.0
        # Sigma sets the divisive gain at low inhibitory drive: at
        # W- @ g(v) ~ 0, the divisive term collapses to (W+ @ g(v)) / sigma.
        # Initialised to 1.0 so init looks like CurrentDynamics minus the
        # negative path, then training pulls it down as inhibition kicks in.
        sigma_init = 1.0

        self.persistence_logit = torch.full(
            (self.num_cell_types,), _logit(persistence_init),
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
        self.sigma_log = torch.full(
            (self.num_cell_types,), np.log(sigma_init),
            dtype=self.dtype, device=self.device,
        )

        # Same fan-in / fan-out init as CurrentDynamics. We deliberately
        # use the full signed-magnitude row/col sums here so a neuron's
        # init scaling reflects its total connectome footprint, not just
        # the excitatory or inhibitory share.
        synapses_in_per_neuron, synapses_out_per_neuron = self._csr_abs_row_col_sums(
            self.synapse_counts
        )
        synapses_in_per_cell_type = _mean_per_cell_type(self, synapses_in_per_neuron)
        self.input_strength_log = torch.log(synapses_in_per_cell_type.pow(-1 / 2))

        synapses_out_per_cell_type = _mean_per_cell_type(self, synapses_out_per_neuron)
        self.output_strength_log = torch.log(synapses_out_per_cell_type.pow(-1 / 2))

        self.persistence_logit = torch.nn.Parameter(self.persistence_logit)
        self.stimulus_strength_log = torch.nn.Parameter(self.stimulus_strength_log)
        self.offset = torch.nn.Parameter(self.offset)
        self.input_strength_log = torch.nn.Parameter(self.input_strength_log)
        self.output_strength_log = torch.nn.Parameter(self.output_strength_log)
        self.sigma_log = torch.nn.Parameter(self.sigma_log)

        self.add_noise_to_parameters(self.init_std)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def _build_signed_split(self):
        """Split signed ``synapse_counts`` into ``W+`` and ``|W-|`` CSRs.

        ``W-`` is stored as positive magnitudes so it can sit inside the
        divisor without sign-flipping at every forward pass. Zero-valued
        entries (e.g. ablations from sign overrides) end up in neither.
        """
        sc = self.synapse_counts.to_sparse_coo().coalesce()
        idx = sc.indices()
        vals = sc.values()

        pos_mask = vals > 0
        neg_mask = vals < 0

        size = sc.size()
        self.synapse_counts_pos = torch.sparse_coo_tensor(
            idx[:, pos_mask], vals[pos_mask],
            size=size, dtype=self.dtype, device=self.device,
        ).coalesce().to_sparse_csr()
        self.synapse_counts_neg = torch.sparse_coo_tensor(
            idx[:, neg_mask], -vals[neg_mask],
            size=size, dtype=self.dtype, device=self.device,
        ).coalesce().to_sparse_csr()

    def _apply_sigma_scale(self, scale_factor):
        """Shift ``output_strength_log`` by ``log(scale_factor)``."""
        self.output_strength_log.add_(torch.log(scale_factor))

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        """Spectral norm of the linearised one-step Jacobian at v=0.

        Linearises around ``v_0 = 0`` with ``f'(v_0) = 1`` (the same
        slope-1 hack :class:`ConductanceDynamics` uses to avoid ReLU's
        zero derivative at threshold). At ``v_0 = 0`` the inhibitory
        denominator collapses to ``sigma`` since ``g(0) = 0``, which gives
        a strict upper bound on the recurrent gain (the denominator only
        grows with activity). The spectral target therefore constrains
        the worst-case excitatory gain — which is what we want for the
        init-time rescaling.
        """
        A_pos = self.synapse_counts_pos
        A_pos_T = A_pos.t().to_sparse_csr()

        D_in = self.input_strength_log.exp()[self.neuron_to_type_index]
        D_out = self.output_strength_log.exp()[self.neuron_to_type_index]
        sigma = self.sigma_log.exp()[self.neuron_to_type_index]
        d = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        gain = D_in / sigma

        def apply_M(v):
            v_scaled = v * D_out
            y = torch.sparse.mm(A_pos, v_scaled.unsqueeze(1)).squeeze(1)
            return y * gain

        def apply_M_T(v):
            v_scaled = v * gain
            y = torch.sparse.mm(A_pos_T, v_scaled.unsqueeze(1)).squeeze(1)
            return y * D_out

        return _spectral_norm(self, apply_M, apply_M_T, d, max_iter, eps)

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
        offset_all = self.offset[self.neuron_to_type_index]
        sigma = self.sigma_log.exp()[self.neuron_to_type_index]

        fr_scaled = self.nonlinearity(v_prev) * output_strength[None, :]

        z_pos = torch.sparse.mm(
            self.synapse_counts_pos, fr_scaled.transpose(0, 1),
        ).transpose(0, 1) * input_strength[None, :]
        z_neg = torch.sparse.mm(
            self.synapse_counts_neg, fr_scaled.transpose(0, 1),
        ).transpose(0, 1) * input_strength[None, :]

        stim = _stimulus_drive(u_t, stimulus_strength, stim_to_cell_mapping)
        divisive = (z_pos + stim) / (z_neg + sigma[None, :])

        v_next = persistence[None, :] * v_prev + (1 - persistence[None, :]) * (
            divisive + offset_all[None, :]
        )
        return v_next


# ---------------------------------------------------------------------------
# ConductanceDynamics
# ---------------------------------------------------------------------------


class ConductanceDynamics:
    """Conductance-based dynamics: synaptic input modelled as conductance changes.

    Synapses stay unsigned, split into one matrix per presynaptic
    neurotransmitter; each transmitter's sign comes from its reversal
    potential. Region ``sign_overrides`` (e.g. the optic lobe's C2/C3 -> L1/L2)
    are applied per edge: a sign-inverted edge is moved into the matrix of an
    opposite-sign transmitter (inhibitory -> the first excitatory one, e.g.
    GABA -> ACH, and vice versa), scaled by ``|factor|``; ``factor=0`` drops it.
    """

    def __init__(self, *, init_new_model=True, **kwargs):
        super().__init__(**kwargs)

        self._save_init_inputs()

        # --- Split synapse_counts_coo into per-neurotransmitter sparse matrices ---
        dtype_torch = self.dtype
        coo, edge_transmitters = self._override_edge_transmitters()

        self.synapse_counts = []
        for ri, r in enumerate(self.unique_transmitters):
            mask = (edge_transmitters == r)
            coo_r = coo[mask]

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

        in_log_per_ct = -0.5 * torch.log(_mean_per_cell_type(self, synapses_in_per_neuron))
        out_log_per_ct = -0.5 * torch.log(_mean_per_cell_type(self, synapses_out_per_neuron))

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
        _init_power_vec(self)

        self.add_noise_to_parameters(self.init_std)

        # Initialise steady_state to rp_leak (better starting point than zeros)
        with torch.no_grad():
            rp_leak_all = self.rp_leak[self.neuron_to_type_index]
            self.steady_state.copy_(rp_leak_all)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def _override_edge_transmitters(self):
        """Edges and their effective transmitter after the region sign overrides.

        Returns ``(coo, transmitters)``: ``synapse_counts_coo`` with ablated
        edges dropped and counts scaled by ``|factor|``, and per edge the
        transmitter whose conductance carries it -- the presynaptic one, or
        for a sign-inverted edge the first transmitter of the opposite class
        (``excitatory_nt`` / ``inhibitory_nt``) present in the model.
        """
        coo = np.array(self.synapse_counts_coo, copy=True)
        pre = coo[:, 0].astype(np.int64)
        edge_transmitters = self.transmitters[pre].copy()
        factors = self._sign_override_factors(pre, coo[:, 1].astype(np.int64))

        flipped = factors < 0
        if flipped.any():
            present = set(self.unique_transmitters)
            opposite = {}
            for same, other in ((self.excitatory_nt, self.inhibitory_nt),
                                (self.inhibitory_nt, self.excitatory_nt)):
                target = next((t for t in other if t in present), None)
                for t in same:
                    opposite[t] = target
            for t in np.unique(edge_transmitters[flipped]):
                if opposite.get(t) is None:
                    raise ValueError(
                        f"A sign override inverts {t} edges, but the model has no "
                        f"transmitter of the opposite sign to carry them."
                    )
                edge_transmitters[flipped & (edge_transmitters == t)] = opposite[t]

        coo[:, 2] = coo[:, 2] * np.abs(factors)
        keep = factors != 0
        return coo[keep], edge_transmitters[keep]

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

        return _spectral_norm(self, apply_M, apply_M_T, d, max_iter, eps)

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
        num = num + _stimulus_drive(u_t, b, stim_to_cell_mapping)

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
    that don't mix across neurons. All weights are shared per cell type
    (first-layer weights per (post, pre) cell-type pair). Sparse first-layer
    weights are stored in log space (so always positive) and multiplied by
    the edge's sign from the signed synapse matrix (presynaptic
    neurotransmitter plus region sign overrides).
    """

    def __init__(self, num_neurons, num_cell_types, pair_post_type, pair_pre_type,
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
        # Cell types of each first-layer weight row (one row per type pair).
        self.register_buffer('_pair_post_type', pair_post_type)
        self.register_buffer('_pair_pre_type', pair_pre_type)

        self.sparse_weights_log = torch.nn.Parameter(
            torch.zeros(len(pair_post_type), cell_ann_depth, device=device, dtype=dtype)
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

    ``cell_ann_depth`` (``[model].cell_ann_depth``) sets the per-neuron
    network depth; depth 1 is a learned per-type-pair weighting of the signed
    connectome. Each step rebuilds the sparse first layer, so it is several
    times slower than :class:`CurrentDynamics`.
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
        pair_post_type = unique_keys // self.num_cell_types
        pair_pre_type = unique_keys % self.num_cell_types
        num_type_pairs = unique_keys.shape[0]

        self.cell_ann = CellAnn(
            num_neurons=self.num_neurons,
            num_cell_types=self.num_cell_types,
            pair_post_type=pair_post_type,
            pair_pre_type=pair_pre_type,
            cell_ann_depth=cell_ann_depth,
            nonlinearity=self.nonlinearity,
            sparse_idx=sparse_indices,
            neuron_to_type=self.neuron_to_type_index,
            edge_to_pair=edge_to_pair,
            edge_sign=edge_sign,
            device=self.device,
            dtype=self.dtype,
        )

        persistence_init = 0.5
        stimulus_strength_init = 10.0
        offset_init = 0.0

        self.persistence_logit = torch.nn.Parameter(torch.full(
            (self.num_cell_types,), _logit(persistence_init),
            dtype=self.dtype, device=self.device))
        self.stimulus_strength_log = torch.nn.Parameter(torch.full(
            (self.num_cell_types,), np.log(stimulus_strength_init),
            dtype=self.dtype, device=self.device))
        self.offset = torch.nn.Parameter(torch.full(
            (self.num_cell_types,), offset_init,
            dtype=self.dtype, device=self.device))

        # Initialise sparse first-layer weights to match CurrentDynamics's
        # effective per-edge magnitude (keeps biologically strong pathways
        # hot at init so signal propagates through the feed-forward path).
        synapses_in_per_cell_type = _mean_per_cell_type(
            self, sc_coo.abs().sum(dim=1).to_dense())
        synapses_out_per_cell_type = _mean_per_cell_type(
            self, sc_coo.abs().sum(dim=0).to_dense())

        edge_abs_counts = sc_coo.values().abs().to(self.dtype)
        pair_count_sum = torch.zeros(num_type_pairs, dtype=self.dtype, device=self.device)
        pair_edge_count = torch.zeros(num_type_pairs, dtype=self.dtype, device=self.device)
        pair_count_sum.index_add_(0, edge_to_pair, edge_abs_counts)
        pair_edge_count.index_add_(0, edge_to_pair, torch.ones_like(edge_abs_counts))
        pair_mean_count = (pair_count_sum / pair_edge_count.clamp_min(1.0)).clamp_min(1.0)

        sparse_weights_log_init = (
            torch.log(pair_mean_count)
            - 0.5 * torch.log(synapses_in_per_cell_type[pair_post_type])
            - 0.5 * torch.log(synapses_out_per_cell_type[pair_pre_type])
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

        _init_power_vec(self)

        self.add_noise_to_parameters(self.init_std)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def silence_neurons(self, neurons_to_silence, invert_silencing=False, silence_inputs=False):
        """Silence cell types' outputs (and optionally inputs).

        Sets the first-layer weights of every type pair whose presynaptic (and,
        with ``silence_inputs``, postsynaptic) type is silenced to
        ``silence_val``; ``silence_inputs`` also silences the stimulus input.
        """
        chosen = self._silencing_mask(neurons_to_silence, invert_silencing)
        if chosen is None:
            return
        chosen_t = torch.as_tensor(chosen, device=self.device)
        rows = chosen_t[self.cell_ann._pair_pre_type]
        if silence_inputs:
            rows = rows | chosen_t[self.cell_ann._pair_post_type]
        with torch.no_grad():
            self.cell_ann.sparse_weights_log.data[rows] = np.log(self.silence_val)
            if silence_inputs:
                self.stimulus_strength_log.data[chosen_t] = np.log(self.silence_val)

    def _transfer_dynamics_parameters(self, new_model):
        """Copy the per-type-pair first-layer weights into a trimmed model.

        Rows are matched by (post, pre) cell-type names; the remaining
        parameters are per cell type and handled by
        :meth:`ConnectomeModel._transfer_parameters_to`.
        """
        def pair_names(model):
            names = model.unique_cell_types
            post = model.cell_ann._pair_post_type.cpu().numpy()
            pre = model.cell_ann._pair_pre_type.cpu().numpy()
            return [(names[a], names[b]) for a, b in zip(post, pre)]

        old_rows = {pair: i for i, pair in enumerate(pair_names(self))}
        index = torch.tensor([old_rows[p] for p in pair_names(new_model)],
                             dtype=torch.int64, device=self.device)
        with torch.no_grad():
            new_model.cell_ann.sparse_weights_log.copy_(
                self.cell_ann.sparse_weights_log.index_select(0, index))
        return {'cell_ann.sparse_weights_log'}

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
        cell_ann = self.cell_ann
        edge_w = (cell_ann.sparse_weights_log[cell_ann._edge_to_pair, 0].exp()
                  * cell_ann._edge_sign)
        A = torch.sparse_coo_tensor(
            cell_ann._sparse_idx, edge_w,
            size=(self.num_neurons, self.num_neurons),
            dtype=self.dtype, device=self.device,
        ).coalesce().to_sparse_csr()
        A_T = A.t().to_sparse_csr()

        def apply_M(v):
            return torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)

        def apply_M_T(v):
            return torch.sparse.mm(A_T, v.unsqueeze(1)).squeeze(1)

        d = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        return _spectral_norm(self, apply_M, apply_M_T, d, max_iter, eps)

    def firing_rate(self, v):
        return self.nonlinearity(v)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        offset = self.offset[self.neuron_to_type_index]

        z = self.cell_ann(v_prev)
        stim = _stimulus_drive(u_t, stimulus_strength, stim_to_cell_mapping)

        return persistence[None, :] * v_prev + (1 - persistence[None, :]) * (
            z + offset[None, :] + stim
        )


# ---------------------------------------------------------------------------
# ArbitraryDynamics — all-to-all per-cell-type weight matrix
# ---------------------------------------------------------------------------


class ArbitraryDynamics:
    """All-to-all dynamics with a learned per-cell-type weight matrix.

    Like CurrentDynamics but without connectome sparsity or sign constraints:
    a connectome-free baseline that keeps only the cell-type identities.
    Effective connectivity ``A[i, j] = W[type(i), type(j)] / n[type(j)]``
    (``n`` = neurons per type), i.e. each neuron sees the *mean* rate of every
    presynaptic type, so ``W`` stays O(1) whatever the type sizes. Region
    ``sign_overrides`` do not apply.

        v_{t+1} = p * v_t + (1 - p) * (A @ g(v_t) + b + c * u_t)
    """

    def __init__(self, *, init_new_model=True, **kwargs):
        super().__init__(**kwargs)
        self._save_init_inputs()

        _init_power_vec(self)
        self.register_buffer('_type_counts', torch.bincount(
            self.neuron_to_type_index, minlength=self.num_cell_types,
        ).to(self.dtype).clamp_min(1.0))

        persistence_init = 0.5
        stimulus_strength_init = 1.0
        offset_init = 0.0

        self.persistence_logit = torch.nn.Parameter(torch.full(
            (self.num_cell_types,), _logit(persistence_init),
            dtype=self.dtype, device=self.device))
        self.stimulus_strength_log = torch.nn.Parameter(torch.full(
            (self.num_cell_types,), np.log(stimulus_strength_init),
            dtype=self.dtype, device=self.device))
        self.offset = torch.nn.Parameter(torch.full(
            (self.num_cell_types,), offset_init,
            dtype=self.dtype, device=self.device))

        self.W = torch.nn.Parameter(torch.zeros(
            self.num_cell_types, self.num_cell_types,
            dtype=self.dtype, device=self.device))

        self.add_noise_to_parameters(self.init_std)

        if init_new_model:
            with torch.no_grad():
                self.set_sigma_max()

    def silence_neurons(self, neurons_to_silence, invert_silencing=False, silence_inputs=False):
        """Zero the chosen cell types' outgoing (and optionally incoming) weights."""
        chosen = self._silencing_mask(neurons_to_silence, invert_silencing)
        if chosen is None:
            return
        chosen_t = torch.as_tensor(chosen, device=self.device)
        with torch.no_grad():
            self.W.data[:, chosen_t] = 0.0
            if silence_inputs:
                self.W.data[chosen_t, :] = 0.0
                self.stimulus_strength_log.data[chosen_t] = np.log(self.silence_val)

    def _apply_A(self, v, transpose=False):
        """Compute ``A @ v`` (or ``A.T @ v``), ``A[i, j] = W[type(i), type(j)] / n[type(j)]``."""
        if v.dim() == 1:
            idx = self.neuron_to_type_index
        else:
            idx = self.neuron_to_type_index.unsqueeze(0).expand_as(v)

        v_by_type = torch.zeros(*v.shape[:-1], self.num_cell_types,
                                device=v.device, dtype=v.dtype)
        v_by_type.scatter_add_(-1, idx, v)

        if transpose:
            z_by_type = (v_by_type @ self.W) / self._type_counts
        else:
            z_by_type = (v_by_type / self._type_counts) @ self.W.T

        return z_by_type[..., self.neuron_to_type_index]

    def _apply_sigma_scale(self, scale_factor):
        self.W.mul_(scale_factor)

    def calc_sigma_max(self, max_iter=100, eps=1e-12):
        d = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        return _spectral_norm(
            self, self._apply_A, lambda v: self._apply_A(v, transpose=True),
            d, max_iter, eps)

    def firing_rate(self, v):
        return self.nonlinearity(v)

    def forward(self, v_prev, u_t, stim_to_cell_mapping=None):
        persistence = self.persistence_logit.sigmoid()[self.neuron_to_type_index]
        stimulus_strength = self.stimulus_strength_log.exp()[self.neuron_to_type_index]
        offset = self.offset[self.neuron_to_type_index]

        z = self._apply_A(self.nonlinearity(v_prev))
        stim = _stimulus_drive(u_t, stimulus_strength, stim_to_cell_mapping)

        return persistence[None, :] * v_prev + (1 - persistence[None, :]) * (
            z + offset[None, :] + stim
        )
