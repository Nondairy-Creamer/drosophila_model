import torch

from test_model_factory import build_model_from_test_config


def test_input_conv_identity_no_shift():
    model, _, _, _ = build_model_from_test_config(use_decoder=False, use_input_conv=True)
    model.eval()

    with torch.no_grad():
        model.input_conv_xy.zero_()
        model.input_conv_t.zero_()

        center = model.input_conv_size_x // 2
        model.input_conv_xy[center, center] = 1.0
        # By convention, index 0 corresponds to the current time.
        model.input_conv_t[0] = 1.0
        model.input_conv_bias.zero_()

    inputs = torch.randn((5, model.num_columns), dtype=model.dtype, device=model.device)
    outputs = model.apply_input_conv(inputs)

    assert torch.allclose(outputs, inputs, atol=1e-6, rtol=0.0)

