import torch

import loss_functions as lf
from tests.test_model_factory import build_model_from_test_config


def test_model_gradients():
    model, _, input_neurons, _ = build_model_from_test_config(
        readouts=['FlowConv2Readout'],
        use_input_conv=True,
    )
    model.train()

    torch.manual_seed(42)
    inputs = torch.randn((1, 20, model.regions['optic_lobe'].num_columns), dtype=model.dtype, device=model.device)
    outputs = model.predict(inputs, input_neurons=input_neurons, output_neurons=model.unique_cell_types.tolist())['outputs']
    target = torch.zeros_like(outputs["FlowConv2Readout"]) + 1

    loss = lf.mse_loss(target, outputs["FlowConv2Readout"])
    loss.backward()

    grad_nonzero = {}
    for name, param in model.named_parameters():
        print(name)
        assert param.requires_grad
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        grad_nonzero[name] = bool(torch.any(param.grad != 0))

    zero_grad_names = [name for name, has_grad in grad_nonzero.items() if not has_grad]
    assert not zero_grad_names, f"Zero gradients for: {zero_grad_names}"
