"""Engineering (non-biological) models that operate on the hex column grid.

These models share the spatial structure of the connectome models -- they
still convolve over the columnar hex lattice using
:class:`neural_models.NeuralModelConv` -- but they have no neurons,
synapses, or recurrent dynamics.  Each forward pass is a pure
feed-forward map over a fixed temporal window of input frames.

Sister to :class:`neural_models.BaseRnn` rather than a subclass: ``ConvModel``
shares only save/load infrastructure via :class:`SerializableModel`, since
its stateless feed-forward semantics don't match the RNN's per-timestep
``predict`` loop.
"""

import torch

from neural_models import NeuralModelConv, SerializableModel


class ConvModel(SerializableModel, torch.nn.Module):
    """Stack of hex-grid convolutions over the last ``num_input_frames`` frames.

    Stateless: the caller supplies the temporal window (newest frame
    last) and the first conv treats those frames as input channels.
    Subsequent convs operate on the learned feature channels produced by
    the previous layer.

    Args:
        column_p_q: ``(num_columns, 2)`` integer p,q lattice coordinates
            for the columns.  Defines the hex-grid neighbourhood.
        num_input_frames: Length of the temporal window fed to the first
            conv layer (channels of the first layer).
        channels: Output channel count for each conv layer.  The number
            of layers equals ``len(channels)``.
        conv_size: Hex-grid diameter (must be odd).
        nonlinearity: Name of an activation in ``torch.nn`` (or
            ``"identity"``) applied between conv layers.
        device, dtype: Standard torch placement / precision.

    Forward:
        ``frames``: ``(batch, num_input_frames, num_columns)`` --
        oldest-to-newest stack of per-column scalars.

        Returns: ``(batch, num_columns, channels[-1])``.
    """

    def __init__(self, *, column_p_q, num_input_frames=5, channels=(5, 8, 8),
                 conv_size=5, nonlinearity='Softplus',
                 device='cpu', dtype='float32'):
        super().__init__()

        if column_p_q is None:
            raise ValueError("ConvModel requires column_p_q for hex convolutions.")

        self._save_init_inputs()

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.num_input_frames = int(num_input_frames)
        self.channels = tuple(channels)
        self.conv_size = int(conv_size)
        self.column_p_q = column_p_q
        self.num_columns = column_p_q.shape[0]

        if nonlinearity in ("identity", "linear"):
            self.nonlinearity = torch.nn.Identity()
        else:
            self.nonlinearity = getattr(torch.nn, nonlinearity)()

        in_channels = (self.num_input_frames,) + self.channels[:-1]
        self.conv_layers = torch.nn.ModuleList([
            NeuralModelConv(
                in_features=cin,
                out_features=cout,
                conv_size=self.conv_size,
                column_p_q=self.column_p_q,
                device=self.device, dtype=self.dtype,
            )
            for cin, cout in zip(in_channels, self.channels)
        ])

    def forward(self, frames):
        """Apply the conv stack to a temporal window of frames.

        Args:
            frames: ``(batch, num_input_frames, num_columns)`` per-column
                scalars over the temporal window (oldest first).

        Returns:
            ``(batch, num_columns, channels[-1])`` per-column features.
        """
        if frames.dim() != 3:
            raise ValueError(
                f"frames must be (batch, num_input_frames, num_columns); "
                f"got shape {tuple(frames.shape)}."
            )
        if frames.shape[1] != self.num_input_frames:
            raise ValueError(
                f"expected {self.num_input_frames} input frames, "
                f"got {frames.shape[1]}."
            )
        if frames.shape[2] != self.num_columns:
            raise ValueError(
                f"expected {self.num_columns} columns, got {frames.shape[2]}."
            )

        # NeuralModelConv expects (..., num_columns, in_features); the
        # 5 frames become the input channels of the first conv.
        x = frames.transpose(-1, -2)
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if i < len(self.conv_layers) - 1:
                x = self.nonlinearity(x)
        return x
