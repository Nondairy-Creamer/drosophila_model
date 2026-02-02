import numpy as np
import torch

import loss_functions as lf
import training_utilities as tu
from tests.test_model_factory import build_model_from_test_config


def test_train_model_plateau_reload(capsys, tmp_path):
    model, _, input_neurons, output_neurons = build_model_from_test_config(use_decoder=False)
    model_save_folder = tmp_path / "model_run"
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    if len(output_neurons) == 0:
        raise AssertionError("No output neurons available for plateau test")

    inputs = torch.zeros((1, 1, model.num_columns), dtype=model.dtype, device=model.device)
    target = {n: torch.zeros((1, 1, model.num_columns), dtype=model.dtype, device=model.device) for n in output_neurons}

    trained_model, loss_history, opt_state, info = tu.train_model(
        model,
        inputs,
        target,
        output_neurons=output_neurons,
        input_neurons=input_neurons,
        rng=np.random.default_rng(0),
        model_save_folder=model_save_folder,
        num_steps=2,
        save_every=1,
        alpha=0.0,
        loss_fn=lf.mse_loss,
        patience=0,
        alpha_factor=0.5,
        input_noise_std=0.0,
        contrast_std=0.0,
        reset_init_every=1,
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
    assert info["last_reload_opt_state"] == saved_opt["optimizer_state_dict"]
    assert len(loss_history) > 0
