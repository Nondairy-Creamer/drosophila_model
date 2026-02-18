import torch

import neural_models as nm
from test_model_factory import build_model_from_test_config


def _set_identity_filter(model):
    """Set spatial and temporal filters to identity (center=1, rest=0)."""
    with torch.no_grad():
        model.input_spatial_conv.weight.zero_()
        model.input_spatial_conv.bias.zero_()
        pq_offsets = nm.NeuralModelConv.hex_grid_pq_offsets(model.input_conv_size_x)
        for i, (dp, dq) in enumerate(pq_offsets):
            if dp == 0 and dq == 0:
                model.input_spatial_conv.weight[0, i, 0] = 1.0
                break
        model.input_conv_temporal.zero_()
        model.input_conv_temporal[0] = 1.0


def test_uniform_input_gives_zero_output():
    """A spatially uniform input has no local contrast, so output should be ~0."""
    model, _, _, _ = build_model_from_test_config(decoder_type='none', use_input_conv=True)
    model.eval()
    _set_identity_filter(model)

    num_time = 10
    inputs = torch.full(
        (num_time, model.num_columns), 5.0,
        dtype=model.dtype, device=model.device,
    )

    with torch.no_grad():
        outputs = model.apply_input_conv(inputs)

    assert torch.allclose(outputs, torch.zeros_like(outputs), atol=1e-3, rtol=0.0)


def test_spatial_identity_gives_zero_output():
    """With a spatial identity filter, centered = input - input = 0, so output is ~0."""
    model, _, _, _ = build_model_from_test_config(decoder_type='none', use_input_conv=True)
    model.eval()
    _set_identity_filter(model)

    num_time = 10
    inputs = torch.randn(
        (num_time, model.num_columns),
        dtype=model.dtype, device=model.device,
    )

    with torch.no_grad():
        outputs = model.apply_input_conv(inputs)

    assert torch.allclose(outputs, torch.zeros_like(outputs), atol=1e-3, rtol=0.0)


def test_full_pipeline_spatial_norm_negate_temporal():
    """Verify the full pipeline: spatial contrast norm -> negate -> temporal filter."""
    model, _, _, _ = build_model_from_test_config(decoder_type='none', use_input_conv=True)
    model.eval()

    num_time = 10
    torch.manual_seed(42)
    inputs = torch.randn(
        (1, num_time, model.num_columns),
        dtype=model.dtype, device=model.device,
    )

    with torch.no_grad():
        w = model.input_spatial_conv.weight[0, :, 0]
        valid = model.input_spatial_conv.nearest_column_indices >= 0
        eff_sum = (w.unsqueeze(0) * valid.float()).sum(dim=1).clamp(min=1e-8)

        raw_mean = model.input_spatial_conv(inputs.unsqueeze(-1)).squeeze(-1)
        local_mean = raw_mean / eff_sum
        centered = inputs - local_mean
        raw_var = model.input_spatial_conv((centered ** 2).unsqueeze(-1)).squeeze(-1)
        local_var = raw_var / eff_sum
        normed = -centered / torch.sqrt(local_var + 1e-3)

        pad = model.input_conv_size_t - 1
        padded = torch.nn.functional.pad(normed, (0, 0, pad, 0))
        windows = padded.unfold(1, model.input_conv_size_t, 1)
        kernel = torch.flip(model.input_conv_temporal, dims=(0,))
        expected = torch.einsum('stcq,q->stc', windows, kernel)

        outputs = model.apply_input_conv(inputs)

    nan_mask = torch.isnan(outputs) | torch.isnan(expected)
    assert (torch.isnan(outputs) == torch.isnan(expected)).all(), "NaN positions differ"
    assert torch.allclose(outputs[~nan_mask], expected[~nan_mask], atol=1e-6, rtol=0.0)
