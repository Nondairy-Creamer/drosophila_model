import torch

from drosophila_model.training import loss_functions as lf
from tests.model_factory import build_model_from_test_config


def test_model_gradients():
    model, _ = build_model_from_test_config(
        readouts=['FlowConv2Readout'],
        use_input_conv=True,
    )
    model.train()

    torch.manual_seed(42)
    # New models start the flow readout at zero output, which blocks every
    # gradient behind its last layer until the first update; give it weights
    # so the whole gradient path is tested.
    last = model.regions['optic_lobe']._readout_modules['FlowConv2Readout'].conv_layers[-1]
    with torch.no_grad():
        last.weight.normal_(0.0, 0.1)
    inputs = torch.randn((1, 20, model.regions['optic_lobe'].num_columns), dtype=model.dtype, device=model.device)
    outputs = model.predict(inputs, output_neurons=model.unique_cell_types.tolist())['outputs']
    optic_outputs = outputs['optic_lobe']
    target = torch.zeros_like(optic_outputs["FlowConv2Readout"]) + 1

    loss = lf.mse_loss(target, optic_outputs["FlowConv2Readout"])
    loss.backward()

    grad_nonzero = {}
    for name, param in model.named_parameters():
        assert param.requires_grad
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        grad_nonzero[name] = bool(torch.any(param.grad != 0))

    zero_grad_names = [name for name, has_grad in grad_nonzero.items() if not has_grad]
    assert not zero_grad_names, f"Zero gradients for: {zero_grad_names}"


def test_flow_readout_starts_at_zero():
    """A new model predicts zero flow, so its first loss is ~1 (no init spike)."""
    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'])
    model.eval()
    torch.manual_seed(0)
    inputs = torch.rand((2, 10, model.regions['optic_lobe'].num_columns), dtype=model.dtype, device=model.device)
    with torch.no_grad():
        flow = model.predict(inputs)['outputs']['optic_lobe']['FlowConv2Readout']
    assert torch.all(flow[torch.isfinite(flow)] == 0)
