import numpy as np
import torch

from drosophila_model.training import training_utilities as tu
from tests.model_factory import build_model_from_test_config


def test_fit_plateau_reload(capsys, tmp_path):
    model, _ = build_model_from_test_config(readouts=['L1Reg'])
    model_save_folder = tmp_path / "model_run"
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    inputs = torch.zeros((1, 1, model.regions['optic_lobe'].num_columns), dtype=model.dtype, device=model.device)
    target = {'optic_flow': torch.zeros((1, 1, model.regions['optic_lobe'].num_columns, 2), dtype=model.dtype, device=model.device)}

    trained_model, loss_history, opt_state, info = tu.fit(
        model,
        inputs,
        target,
        output_neurons=None,
        input_neurons=None,
        loss_components=[],
        rng=np.random.default_rng(0),
        model_save_folder=model_save_folder,
        num_epochs=2,
        save_every=1,
        alpha=0.0,
        patience=0,
        alpha_factor=0.5,
        takeoff_fraction=0.0,  # the loss never moves here, so count patience from the start
        input_noise_std=0.0,
        contrast_std=0.0,
    )

    stdout = capsys.readouterr().out
    assert "Reloaded best checkpoint" in stdout

    reloaded_state = trained_model.state_dict()
    for key in initial_state:
        torch.testing.assert_close(reloaded_state[key], initial_state[key])

    saved_opt = torch.load(model_save_folder / "model_opt.pt", map_location="cpu", weights_only=False)
    assert info["last_reload_state"] is not None
    for key in initial_state:
        torch.testing.assert_close(info["last_reload_state"][key], initial_state[key])
    def _assert_state_dicts_equal(a, b):
        """Recursively compare state dicts that may contain tensors."""
        assert type(a) == type(b), f"Type mismatch: {type(a)} vs {type(b)}"
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                _assert_state_dicts_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                _assert_state_dicts_equal(x, y)
        elif isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b)
        else:
            assert a == b
    _assert_state_dicts_equal(info["last_reload_opt_state"], saved_opt["optimizer_state_dict"])
    assert len(loss_history) > 0
