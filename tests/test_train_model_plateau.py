from pathlib import Path

import numpy as np
import torch

import loss_functions as lf
import training_utilities as tu


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.use_decoder = False
        self.call_count = 0

    def predict(self, inputs, init_state=None, input_neurons=None, output_neurons=None):
        output_neurons = output_neurons or []
        batch = inputs.shape[0]
        time = inputs.shape[1]
        cols = inputs.shape[2]
        offset = 0.0 if self.call_count == 0 else 10.0
        self.call_count += 1
        pred = {
            "firing_rate": {
                n: (self.weight + offset)
                * torch.ones((batch, time, cols), device=self.device, dtype=self.dtype)
                for n in output_neurons
            }
        }
        return pred, None, {"input_mapping": input_neurons, "output_mapping": output_neurons}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    def find_steady_state(self):
        return


def test_train_model_plateau_reload(capsys, tmp_path):
    model = DummyModel()
    model_save_folder = tmp_path / "model_run"
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    inputs = torch.zeros((1, 1, 1), dtype=model.dtype, device=model.device)
    target = {"T4a": torch.ones((1, 1, 1), dtype=model.dtype, device=model.device)}

    trained_model, loss_history, opt_state, info = tu.train_model(
        model,
        inputs,
        target,
        output_neurons=["T4a"],
        input_neurons=["L1"],
        rng=np.random.default_rng(0),
        model_save_folder=model_save_folder,
        num_steps=2,
        save_every=1,
        alpha=0.1,
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
