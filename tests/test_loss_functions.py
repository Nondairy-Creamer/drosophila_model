import torch

import loss_functions as lf


def test_loss_functions_handle_nans():
    y_true = torch.tensor([[1.0, float("nan")], [0.0, 2.0]])
    y_hat = torch.tensor([[1.0, 1.0], [float("nan"), 2.0]])

    for name in dir(lf):
        if not name.endswith("_loss"):
            continue
        loss_fn = getattr(lf, name)
        if not callable(loss_fn):
            continue
        loss_val = loss_fn(y_true, y_hat)
        assert torch.isfinite(loss_val), f"{name} returned non-finite {loss_val}"

        loss_val = loss_fn(y_true, y_true)
        assert loss_val < 1e-6, f"{name} returned {loss_val} when y_hat == y_true"
