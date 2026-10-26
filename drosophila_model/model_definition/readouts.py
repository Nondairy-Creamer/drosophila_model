"""Readout and regularizer modules for the optic-lobe region.

**Readouts** are ``torch.nn.Module`` subclasses with these class-level
attributes:

* ``output_extra_dims`` – tuple of ints giving the trailing shape beyond
  ``(batch, num_columns)``.  For example ``(2,)`` for 2-D optic flow.
* ``eye_asymmetric`` – ``True`` when the readout produces an output that
  differs between left and right eyes (e.g. horizontal flow is flipped).
* ``predicts_flow`` – ``True`` for 2-D (right, up) optic-flow readouts (used
  by the ``flow_loss`` loss and the ego-motion solver).
* ``predicts_motion_mag`` – ``True`` for scalar motion-magnitude readouts
  (used by the ``motion_mag`` loss).

**Regularizers** are ``torch.nn.Module`` subclasses with ``training_only = True``.
They are called once per ``predict()`` (after the time loop) with the model as
input and return a scalar loss term.  They are skipped when the model is in
eval mode.

Both are listed by class name in ``[model.regions.optic_lobe].readouts`` and
built by ``OpticLobeRegion.instantiate_readouts`` (looked up by name in this
module, so there is no registry to update).
"""

import numpy as np
import torch

from drosophila_model.model_definition.model_layers import HexGridConv

_FLOW_NEURON_TYPES = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

_DEFAULT_INPUT_LAYER_L1 = 0.0


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class HexGridReadout(torch.nn.Module):
    """Hex-grid convolutional readout with N layers.

    Handles:
    * Column-to-neuron mapping (shared across all hex-grid readouts).
    * Stacking ``HexGridConv`` layers with softplus between them.
    * Optional denominator normalisation (last channel passed through softplus
      and used to divide the remaining channels).

    Subclasses only need to set the class-level flags (``output_extra_dims``,
    ``eye_asymmetric``, ``predicts_flow`` / ``predicts_motion_mag``) and call
    ``super().__init__`` with the desired conv configuration.
    """

    output_extra_dims: tuple = ()
    eye_asymmetric: bool = False
    predicts_flow: bool = False
    predicts_motion_mag: bool = False

    def __init__(self, *, neurons_to_decode=None, column_p_q, cell_types,
                 column_indices, num_neurons, num_columns,
                 num_channels, conv_sizes,
                 positive_weights=False, use_bias=True,
                 use_denominator=False, denom_eps=1e-2,
                 regularizer_parameters=None,
                 device=None, dtype=None, **_kwargs):
        super().__init__()
        if len(num_channels) != len(conv_sizes):
            raise ValueError("num_channels and conv_sizes must have the same length.")

        self.use_denominator = use_denominator
        self.denom_eps = denom_eps

        reg_params = regularizer_parameters or {}
        unknown = set(reg_params) - {'input_layer_l1'}
        if unknown:
            raise ValueError(
                f"Unknown regularizer_parameters {sorted(unknown)}; the only "
                f"supported key is 'input_layer_l1'."
            )
        l1_weight = reg_params.get('input_layer_l1', _DEFAULT_INPUT_LAYER_L1)
        self._regularizer_config = {DecoderWeightL1Reg: l1_weight} if l1_weight else {}
        if neurons_to_decode is None:
            neurons_to_decode = sorted(set(np.asarray(cell_types)))
        self.neurons_to_decode = list(neurons_to_decode)

        self._build_column_to_neuron_mapping(
            cell_types, column_indices, num_neurons, num_columns, device,
        )

        in_features = len(self.neurons_to_decode)
        self.conv_sizes = list(conv_sizes)
        self.conv_layers = torch.nn.ModuleList()
        for i, (channels, csz) in enumerate(zip(num_channels, conv_sizes)):
            is_last = (i == len(num_channels) - 1)
            self.conv_layers.append(HexGridConv(
                in_features=in_features,
                out_features=channels,
                conv_size=csz,
                column_p_q=column_p_q,
                device=device,
                dtype=dtype,
                positive_weights=positive_weights if is_last else False,
                use_bias=use_bias if is_last else True,
            ))
            in_features = channels

    def _build_column_to_neuron_mapping(self, cell_types, column_indices,
                                        num_neurons, num_columns, device):
        cell_types = np.asarray(cell_types)
        column_indices = np.asarray(column_indices, dtype=np.float64)
        cell_type_to_idx = {ct: i for i, ct in enumerate(self.neurons_to_decode)}
        num_types = len(self.neurons_to_decode)

        decoder_mask = np.zeros(num_neurons, dtype=bool)
        for ct in self.neurons_to_decode:
            decoder_mask |= (cell_types == ct)

        mapping = np.full((num_columns, num_types), -1, dtype=np.int64)
        for neuron_idx in np.where(decoder_mask)[0]:
            col = column_indices[neuron_idx]
            if np.isnan(col):
                continue
            col = int(col)
            type_idx = cell_type_to_idx.get(cell_types[neuron_idx])
            if type_idx is not None and mapping[col, type_idx] == -1:
                mapping[col, type_idx] = neuron_idx

        self.register_buffer(
            'column_to_decoder_neurons',
            torch.tensor(mapping, device=device, dtype=torch.int64),
        )

    def _extract_column_features(self, v):
        """``(batch, num_neurons)`` -> ``(batch, num_columns, num_types)``."""
        valid = self.column_to_decoder_neurons >= 0
        safe = self.column_to_decoder_neurons.clamp(min=0)
        features = v[:, safe]
        return features.masked_fill(~valid.unsqueeze(0), float('nan'))

    def flip_for_left_eye(self, x):
        """Transform readout output for left-eye convention.

        Default: negate the first channel (horizontal flow sign flip).
        Subclasses with different output layouts should override.
        """
        x[..., 0] *= -1
        return x

    def forward(self, fr):
        if fr.dim() == 1:
            fr = fr.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        x = self._extract_column_features(fr)

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            is_intermediate = (i < len(self.conv_layers) - 1)
            if is_intermediate:
                x = torch.nn.functional.softplus(x)

        if self.use_denominator:
            denom = torch.nn.functional.softplus(x[:, :, -1:].clone())
            denom = torch.clamp(denom, min=self.denom_eps)
            x = x[:, :, :-1] / denom

        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        if squeeze:
            x = x.squeeze(0)
        return x


# ---------------------------------------------------------------------------
# Concrete readout classes
# ---------------------------------------------------------------------------

class FlowNeuronReadout(torch.nn.Module):
    """Optic flow from weighted T4/T5 direction-selective cells.

    Learns 8 positive (log-space) weights to combine T4a-d and T5a-d firing
    rates into a 2-D flow vector per column.
    """

    output_extra_dims = (2,)
    eye_asymmetric = True
    predicts_flow = True

    def __init__(self, *, cell_types, column_indices, num_neurons, num_columns,
                 device=None, dtype=None, **_kwargs):
        super().__init__()
        flow_types = _FLOW_NEURON_TYPES
        mapping = np.full((num_columns, len(flow_types)), -1, dtype=np.int64)
        ct_to_idx = {ct: i for i, ct in enumerate(flow_types)}
        cell_types = np.asarray(cell_types)
        column_indices_np = np.asarray(column_indices, dtype=np.float64)
        for neuron_idx in range(num_neurons):
            ct = cell_types[neuron_idx]
            if ct not in ct_to_idx:
                continue
            col = column_indices_np[neuron_idx]
            if np.isnan(col):
                continue
            col = int(col)
            ti = ct_to_idx[ct]
            if mapping[col, ti] == -1:
                mapping[col, ti] = neuron_idx
        self.register_buffer(
            'column_indices',
            torch.tensor(mapping, device=device, dtype=torch.int64),
        )
        self.weights_log = torch.nn.Parameter(
            torch.zeros(8, device=device, dtype=dtype)
        )

    def flip_for_left_eye(self, x):
        """Negate horizontal flow channel for the left eye."""
        x[..., 0] *= -1
        return x

    def forward(self, fr):
        idx = self.column_indices                       # (cols, 8)
        valid = idx >= 0
        safe = idx.clamp(min=0)
        vals = fr[:, safe]                              # (batch, cols, 8)
        vals = vals * valid.unsqueeze(0).float()

        w = self.weights_log.exp()
        # Index order: T4a(0) T4b(1) T4c(2) T4d(3) T5a(4) T5b(5) T5c(6) T5d(7)
        lr = (w[0] * vals[:, :, 0] + w[4] * vals[:, :, 4]) \
           - (w[1] * vals[:, :, 1] + w[5] * vals[:, :, 5])
        ud = (w[2] * vals[:, :, 2] + w[6] * vals[:, :, 6]) \
           - (w[3] * vals[:, :, 3] + w[7] * vals[:, :, 7])
        return torch.stack([lr, ud], dim=-1)


class FlowLinearReadout(HexGridReadout):
    """Single-layer hex-grid conv decoder producing 2-D optic flow."""

    output_extra_dims = (2,)
    eye_asymmetric = True
    predicts_flow = True

    def __init__(self, *, readout_neurons=None, **kwargs):
        super().__init__(
            neurons_to_decode=readout_neurons,
            num_channels=[2], conv_sizes=[3], use_denominator=False,
            **kwargs,
        )


class FlowConv2Readout(HexGridReadout):
    """Two-layer hex-grid conv decoder producing 2-D optic flow."""

    output_extra_dims = (2,)
    eye_asymmetric = True
    predicts_flow = True

    def __init__(self, *, readout_neurons=None, **kwargs):
        super().__init__(
            neurons_to_decode=readout_neurons,
            num_channels=[8, 3], conv_sizes=[5, 5], use_denominator=True,
            **kwargs,
        )


class FlowConv3Readout(HexGridReadout):
    """Three-layer hex-grid conv decoder producing 2-D optic flow."""

    output_extra_dims = (2,)
    eye_asymmetric = True
    predicts_flow = True

    def __init__(self, *, readout_neurons=None, **kwargs):
        super().__init__(
            neurons_to_decode=readout_neurons,
            num_channels=[8, 6, 3], conv_sizes=[5, 5, 5], use_denominator=True,
            **kwargs,
        )


class MotionMagReadout(HexGridReadout):
    """Positive-weight single-layer conv producing scalar motion magnitude."""

    output_extra_dims = ()
    eye_asymmetric = False
    predicts_motion_mag = True

    def __init__(self, *, readout_neurons=None, **kwargs):
        super().__init__(
            neurons_to_decode=readout_neurons,
            num_channels=[1], conv_sizes=[5],
            positive_weights=True, use_bias=False, use_denominator=False,
            **kwargs,
        )


class MotionMagConv2Readout(HexGridReadout):
    """Two-layer conv producing scalar motion magnitude (positive weights in
    the last layer)."""

    output_extra_dims = ()
    eye_asymmetric = False
    predicts_motion_mag = True

    def __init__(self, *, readout_neurons=None, **kwargs):
        super().__init__(
            neurons_to_decode=readout_neurons,
            num_channels=[8, 1], conv_sizes=[5, 5],
            positive_weights=True, use_bias=False, use_denominator=False,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Regularizer base class and concrete regularizers
# ---------------------------------------------------------------------------

class Regularizer(torch.nn.Module):
    """Base class for regularizer modules.

    Regularizers are called once per ``predict()`` with the model as input and
    return a scalar.  They are skipped when the model is in eval mode.
    """

    training_only: bool = True

    def __init__(self, **_kwargs):
        super().__init__()

    def forward(self, model):
        raise NotImplementedError


class L1Reg(Regularizer):
    """L1 on input/output strength parameters and readout conv weights."""

    def forward(self, model):
        terms = [
            model.input_strength_log.exp().mean(),
            model.output_strength_log.exp().mean(),
        ]
        for mod in model._readout_modules.values():
            if getattr(mod, 'training_only', False):
                continue
            for sub in mod.modules():
                if hasattr(sub, 'conv_layers'):
                    terms.append(
                        torch.stack([p.abs().mean() for p in sub.conv_layers.parameters()]).mean()
                    )
                    break
        return sum(terms) / len(terms)


class L2Reg(Regularizer):
    """L2 on input/output strength parameters."""

    def forward(self, model):
        return (model.input_strength_log.exp().pow(2).mean()
                + model.output_strength_log.exp().pow(2).mean()) / 2


class SigmaReg(Regularizer):
    """Penalty when spectral radius exceeds the allowed maximum."""

    def forward(self, model):
        sigma_max = model.calc_sigma_max(max_iter=10)
        return torch.relu(sigma_max - model.max_allowed_sigma) ** 2


class DecoderWeightL1Reg(Regularizer):
    """L1 on the first conv layer of a specific decoder to encourage cell-type sparsity.

    Each instance is bound to a single readout (by name) and penalizes only
    ``conv_layers[0].weight`` — the layer whose input dimension corresponds
    to the decoder cell types.  The penalty is scaled by ``weight``.
    """

    def __init__(self, readout_name, weight=1.0, **_kwargs):
        super().__init__()
        self.readout_name = readout_name
        self.l1_weight = weight

    def forward(self, model):
        readout = model._readout_modules[self.readout_name]
        return self.l1_weight * readout.conv_layers[0].weight.abs().sum()


def is_configurable(cls) -> bool:
    """Whether ``cls`` may be listed in ``[model.regions.optic_lobe].readouts``.

    True for the concrete readouts (``predicts_flow`` / ``predicts_motion_mag``)
    and the standalone regularizers; False for base classes, imported helpers
    and ``DecoderWeightL1Reg`` (built automatically, tied to one readout).
    """
    if not (isinstance(cls, type) and issubclass(cls, torch.nn.Module)):
        return False
    if issubclass(cls, Regularizer):
        return cls not in (Regularizer, DecoderWeightL1Reg)
    return bool(getattr(cls, 'predicts_flow', False)
                or getattr(cls, 'predicts_motion_mag', False))


def configurable_names() -> list[str]:
    """Sorted names of the classes :func:`is_configurable` accepts."""
    return sorted(name for name, obj in globals().items()
                  if not name.startswith('_') and is_configurable(obj))
