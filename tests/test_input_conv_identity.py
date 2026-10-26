import torch

from drosophila_model.model_definition import model_layers as ml
from tests.model_factory import build_model_from_test_config


def _set_identity_filter(model):
    """Set spatial filter to center-only and temporal LRU to passthrough (d=1, modes off).

    The spatial conv uses ``positive_weights=True`` (log-space), so we need
    large negative values for non-center weights to make ``exp(w) ≈ 0``.
    """
    with torch.no_grad():
        model.regions['optic_lobe'].input_spatial_conv.weight.fill_(-100.0)
        pq_offsets = ml.HexGridConv.hex_grid_pq_offsets(model.regions['optic_lobe'].conv_size_x)
        for i, (dp, dq) in enumerate(pq_offsets):
            if dp == 0 and dq == 0:
                model.regions['optic_lobe'].input_spatial_conv.weight[0, i, 0] = 0.0
                break
        model.regions['optic_lobe'].temporal_lru.log_d.fill_(0.0)
        model.regions['optic_lobe'].temporal_lru.b_re.zero_()
        model.regions['optic_lobe'].temporal_lru.b_im.zero_()
        model.regions['optic_lobe'].temporal_lru.c_re.zero_()
        model.regions['optic_lobe'].temporal_lru.c_im.zero_()


def test_spatial_identity_gives_zero_output():
    """With a spatial identity filter, centered = input - input = 0, so output is ~0."""
    model, _ = build_model_from_test_config(readouts=[], use_input_conv=True)
    model.eval()
    _set_identity_filter(model)

    batch = 2
    inputs = torch.randn(
        (batch, 1, model.regions['optic_lobe'].num_columns),
        dtype=model.dtype, device=model.device,
    )

    with torch.no_grad():
        u_t = inputs[:, 0, :]
        y_t, _ = model.regions['optic_lobe'].apply_input_conv(u_t)

    assert torch.allclose(y_t, torch.zeros_like(y_t), atol=1e-3, rtol=0.0)


def test_lru_step_shapes():
    """Verify the LRU step returns correct shapes and state."""
    model, _ = build_model_from_test_config(readouts=[], use_input_conv=True)
    model.eval()

    batch = 3
    u_t = torch.randn(batch, model.regions['optic_lobe'].num_columns, dtype=model.dtype, device=model.device)

    with torch.no_grad():
        y_t, state = model.regions['optic_lobe'].temporal_lru.step(u_t)

    assert y_t.shape == (batch, model.regions['optic_lobe'].num_columns)
    assert state.shape == (batch, model.regions['optic_lobe'].temporal_lru.num_modes, model.regions['optic_lobe'].num_columns)
    assert state.is_complex()

    with torch.no_grad():
        y_t2, state2 = model.regions['optic_lobe'].temporal_lru.step(u_t, state=state)

    assert y_t2.shape == y_t.shape
    assert state2.shape == state.shape
