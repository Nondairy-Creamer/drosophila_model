import pytest
import torch

from drosophila_model.training import loss_functions as lf
from drosophila_model.training import training_utilities as tu
from tests.model_factory import build_model_from_test_config


# Each helper as a function of (prediction, target), in its own argument order:
# mse_loss / cos_loss take (y_true, y_hat); normalized_mse / cosine_loss take
# (pred, target).
_EVAL_LOSS_FNS = {
    'mse_loss': lambda pred, target: lf.mse_loss(target, pred),
    'cos_loss': lambda pred, target: lf.cos_loss(target, pred),
    'normalized_mse': lf.normalized_mse,
    'cosine_loss': lf.cosine_loss,
}


def test_loss_functions_handle_nans():
    y_true = torch.tensor([[1.0, float("nan")], [0.0, 2.0]])
    y_hat = torch.tensor([[1.0, 1.0], [float("nan"), 2.0]])

    for name, loss_fn in _EVAL_LOSS_FNS.items():
        loss_val = loss_fn(y_hat, y_true)
        assert torch.isfinite(loss_val), f"{name} returned non-finite {loss_val}"

        loss_val = loss_fn(y_true, y_true)
        assert loss_val < 1e-6, f"{name} returned {loss_val} when y_hat == y_true"


def _one_eye_batch(model, num_time=4):
    """Random one-eye inputs, flow target and an all-true train mask."""
    gen = torch.Generator().manual_seed(0)
    num_cols = model.regions['optic_lobe'].num_columns
    inputs = torch.rand(1, num_time, num_cols, generator=gen, dtype=model.dtype)
    flow = torch.randn(1, num_time, num_cols, 2, generator=gen, dtype=model.dtype)
    train_mask = torch.ones(inputs.shape, dtype=torch.bool)
    return (inputs.to(model.device), {'optic_flow': flow.to(model.device)},
            train_mask.to(model.device))


def test_motion_mag_trains_conv2_magnitude_readout():
    """motion_mag scores MotionMagConv2Readout (it used to return a silent 0)."""
    model, _ = build_model_from_test_config(
        readouts=['FlowConv2Readout', 'MotionMagConv2Readout'])
    inputs, target, train_mask = _one_eye_batch(model)

    outputs, _ = tu.model_outputs(model, inputs, train_mask, num_eyes=1)
    loss = lf.motion_mag(model, outputs, target, train_mask,
                         tu.loss_start_step(train_mask))
    assert torch.isfinite(loss) and loss > 0

    loss.backward()
    readout = model._readout_modules['MotionMagConv2Readout']
    grads = [p.grad for p in readout.conv_layers.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_motion_mag_requires_a_magnitude_readout():
    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'])
    inputs, target, train_mask = _one_eye_batch(model)
    with torch.no_grad():
        outputs, _ = tu.model_outputs(model, inputs, train_mask, num_eyes=1)
    with pytest.raises(ValueError, match="motion-magnitude readout"):
        lf.motion_mag(model, outputs, target, train_mask, 0)
