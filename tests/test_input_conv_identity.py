import torch

import neural_models as nm
import model_layers as ml
from test_model_factory import build_model_from_test_config


def _set_identity_filter(model):
    """Set spatial filter to center-only and temporal LRU to passthrough (d=1, modes off).

    The spatial conv uses ``positive_weights=True`` (log-space), so we need
    large negative values for non-center weights to make ``exp(w) ≈ 0``.
    """
    with torch.no_grad():
        model.input_spatial_conv.weight.fill_(-100.0)
        pq_offsets = ml.HexGridConv.hex_grid_pq_offsets(model.input_conv_size_x)
        for i, (dp, dq) in enumerate(pq_offsets):
            if dp == 0 and dq == 0:
                model.input_spatial_conv.weight[0, i, 0] = 0.0
                break
        model.temporal_lru.log_d.fill_(0.0)
        model.temporal_lru.b_re.zero_()
        model.temporal_lru.b_im.zero_()
        model.temporal_lru.c_re.zero_()
        model.temporal_lru.c_im.zero_()


def test_uniform_input_gives_zero_conv_output():
    """A spatially uniform input has no local contrast, so conv output should be ~0."""
    model, _, _, _ = build_model_from_test_config(readouts=[], use_input_conv=True)
    model.eval()
    _set_identity_filter(model)

    batch = 2
    u_t = torch.full(
        (batch, model.num_columns), 5.0,
        dtype=model.dtype, device=model.device,
    )

    with torch.no_grad():
        y_t, _ = model.apply_input_conv(u_t)

    assert torch.allclose(y_t, torch.zeros_like(y_t), atol=1e-3, rtol=0.0)


def test_spatial_identity_gives_zero_output():
    """With a spatial identity filter, centered = input - input = 0, so output is ~0."""
    model, _, _, _ = build_model_from_test_config(readouts=[], use_input_conv=True)
    model.eval()
    _set_identity_filter(model)

    batch = 2
    inputs = torch.randn(
        (batch, 1, model.num_columns),
        dtype=model.dtype, device=model.device,
    )

    with torch.no_grad():
        u_t = inputs[:, 0, :]
        y_t, _ = model.apply_input_conv(u_t)

    assert torch.allclose(y_t, torch.zeros_like(y_t), atol=1e-3, rtol=0.0)


def test_lru_step_shapes():
    """Verify the LRU step returns correct shapes and state."""
    model, _, _, _ = build_model_from_test_config(readouts=[], use_input_conv=True)
    model.eval()

    batch = 3
    u_t = torch.randn(batch, model.num_columns, dtype=model.dtype, device=model.device)

    with torch.no_grad():
        y_t, state = model.temporal_lru.step(u_t)

    assert y_t.shape == (batch, model.num_columns)
    assert state.shape == (batch, model.temporal_lru.num_modes, model.num_columns)
    assert state.is_complex()

    with torch.no_grad():
        y_t2, state2 = model.temporal_lru.step(u_t, state=state)

    assert y_t2.shape == y_t.shape
    assert state2.shape == state.shape
