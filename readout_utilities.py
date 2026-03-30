"""Readout and regularizer modules for NeuralModel.

**Readouts** are ``torch.nn.Module`` subclasses with two class-level attributes:

* ``output_extra_dims`` – tuple of ints giving the trailing shape beyond
  ``(batch, num_columns)``.  For example ``(2,)`` for 2-D optic flow.
* ``eye_asymmetric`` – ``True`` when the readout produces an output that
  differs between left and right eyes (e.g. horizontal flow is flipped).

**Regularizers** are ``torch.nn.Module`` subclasses with ``training_only = True``.
They are called once per ``predict()`` (after the time loop) with the model as
input and return a scalar loss term.  They are skipped when the model is in
eval mode.

Both are listed in the config ``readouts`` key and discovered automatically by
``NeuralModel`` — no registry to update.
"""

import numpy as np
import torch

from neural_models import NeuralModelConv

# Cell types used as input features for hex-grid conv readouts.
_DECODER_NEURON_TYPES = [
    'T1', 'T2', 'T2a', 'T3',
    'T4a', 'T4b', 'T4c', 'T4d',
    'T5a', 'T5b', 'T5c', 'T5d',
    'Tm1', 'Tm2', 'Tm3', 'Tm4',
    'Tm9', 'Tm16', 'Tm20', 'Tm28', 'Tm30',
]

_FLOW_NEURON_TYPES = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class HexGridReadout(torch.nn.Module):
    """Hex-grid convolutional readout with N layers.

    Handles:
    * Column-to-neuron mapping (shared across all hex-grid readouts).
    * Stacking ``NeuralModelConv`` layers with optional BatchNorm + softplus
      between them.
    * Optional denominator normalisation (last channel passed through softplus
      and used to divide the remaining channels).

    Subclasses only need to set ``output_extra_dims``, ``eye_asymmetric``, and call
    ``super().__init__`` with the desired conv configuration.
    """

    output_extra_dims: tuple = ()
    eye_asymmetric: bool = False

    def __init__(self, *, neurons_to_decode, column_p_q, cell_types,
                 column_indices, num_neurons, num_columns,
                 num_channels, conv_sizes,
                 positive_weights=False, use_bias=True,
                 use_denominator=False, denom_eps=1e-2,
                 device=None, dtype=None, **_kwargs):
        super().__init__()
        if len(num_channels) != len(conv_sizes):
            raise ValueError("num_channels and conv_sizes must have the same length.")

        self.use_denominator = use_denominator
        self.denom_eps = denom_eps
        self.neurons_to_decode = list(neurons_to_decode)

        self._build_column_to_neuron_mapping(
            cell_types, column_indices, num_neurons, num_columns, device,
        )

        in_features = len(self.neurons_to_decode)
        self.conv_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i, (channels, csz) in enumerate(zip(num_channels, conv_sizes)):
            is_last = (i == len(num_channels) - 1)
            self.conv_layers.append(NeuralModelConv(
                in_features=in_features,
                out_features=channels,
                conv_size=csz,
                column_p_q=column_p_q,
                device=device,
                dtype=dtype,
                positive_weights=positive_weights if is_last else False,
                use_bias=use_bias if is_last else True,
            ))
            if not is_last:
                self.batch_norms.append(
                    torch.nn.BatchNorm1d(channels).to(device=device, dtype=dtype)
                )
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

    def forward(self, fr):
        if fr.dim() == 1:
            fr = fr.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        x = self._extract_column_features(fr)

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x.permute(0, 2, 1)).permute(0, 2, 1)
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

    def __init__(self, **kwargs):
        super().__init__(
            neurons_to_decode=_DECODER_NEURON_TYPES,
            num_channels=[3], conv_sizes=[5], use_denominator=True,
            **kwargs,
        )


class FlowConv2Readout(HexGridReadout):
    """Two-layer hex-grid conv decoder producing 2-D optic flow."""

    output_extra_dims = (2,)
    eye_asymmetric = True

    def __init__(self, **kwargs):
        super().__init__(
            neurons_to_decode=_DECODER_NEURON_TYPES,
            num_channels=[8, 3], conv_sizes=[5, 5], use_denominator=True,
            **kwargs,
        )


class MotionMagReadout(HexGridReadout):
    """Positive-weight single-layer conv producing scalar motion magnitude."""

    output_extra_dims = ()
    eye_asymmetric = False

    def __init__(self, **kwargs):
        super().__init__(
            neurons_to_decode=_DECODER_NEURON_TYPES,
            num_channels=[1], conv_sizes=[5],
            positive_weights=True, use_bias=False,
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
