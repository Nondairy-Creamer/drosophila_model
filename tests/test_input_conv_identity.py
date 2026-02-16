import torch
import math

from test_model_factory import build_model_from_test_config


def _set_identity_filter(model):
    """Set spatial and temporal filters to identity (center=1, rest≈0)."""
    with torch.no_grad():
        # Spatial: log(~0) everywhere, log(1)=0 at center
        model.input_conv_xy_log.fill_(math.log(1e-8))
        center = model.input_conv_size_x // 2
        model.input_conv_xy_log[center, center] = 0.0
        # Temporal: 0 everywhere, 1 at index 0
        model.input_conv_t.zero_()
        model.input_conv_t[0] = 1.0


def test_uniform_input_gives_zero_output():
    """A spatially uniform input has no local contrast, so output should be ~0."""
    model, _, _, _ = build_model_from_test_config(use_decoder=False, use_input_conv=True)
    model.eval()
    _set_identity_filter(model)

    # Uniform input across all columns (constant value 5.0)
    num_time = 10
    inputs = torch.full(
        (num_time, model.num_columns), 5.0,
        dtype=model.dtype, device=model.device,
    )

    with torch.no_grad():
        outputs = model.apply_input_conv(inputs)

    # spatial_conv(5.0) ≈ 5.0 (identity)
    # input_centered ≈ 0.0
    # temporal_conv(-0.0) ≈ 0.0
    assert torch.allclose(outputs, torch.zeros_like(outputs), atol=1e-3, rtol=0.0)


def test_spatial_identity_gives_zero_output():
    """With a spatial identity filter, centered = input - input = 0, so output is ~0."""
    model, _, _, _ = build_model_from_test_config(use_decoder=False, use_input_conv=True)
    model.eval()
    _set_identity_filter(model)

    num_time = 10
    inputs = torch.randn(
        (num_time, model.num_columns),
        dtype=model.dtype, device=model.device,
    )

    with torch.no_grad():
        outputs = model.apply_input_conv(inputs)

    # spatial identity → local_mean ≈ input → centered ≈ 0 → output ≈ 0
    assert torch.allclose(outputs, torch.zeros_like(outputs), atol=1e-3, rtol=0.0)


def test_full_pipeline_spatial_norm_negate_temporal():
    """Verify the full pipeline: spatial contrast norm → negate → temporal filter."""
    model, _, _, _ = build_model_from_test_config(use_decoder=False, use_input_conv=True)
    model.eval()

    num_time = 10
    torch.manual_seed(42)
    inputs = torch.randn(
        (1, num_time, model.num_columns),
        dtype=model.dtype, device=model.device,
    )

    with torch.no_grad():
        # Manually compute expected output step by step
        local_mean = model._apply_spatial_conv(inputs)
        input_centered = inputs - local_mean
        local_var = model._apply_spatial_conv(input_centered ** 2)
        eps = 1e-6
        input_norm = input_centered / torch.sqrt(local_var + eps)
        expected = model._apply_temporal_conv(-input_norm)

        outputs = model.apply_input_conv(inputs)

    # NaN-safe comparison: check NaN positions match, then compare non-NaN values
    nan_mask = torch.isnan(outputs) | torch.isnan(expected)
    assert (torch.isnan(outputs) == torch.isnan(expected)).all(), "NaN positions differ"
    assert torch.allclose(outputs[~nan_mask], expected[~nan_mask], atol=1e-6, rtol=0.0)
