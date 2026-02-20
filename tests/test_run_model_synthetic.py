import time
from pathlib import Path

import torch

import data_utilities as du
import neural_models as nm
import run_model
import training_utilities as tu


def _load_test_config(repo_root: Path) -> tuple[dict, Path]:
    test_config_path = repo_root / "tests" / "config_test.toml"
    return du.load_config(str(test_config_path)), test_config_path


def _get_run_folder(repo_root: Path, timestamp: str, run_id: str) -> Path:
    models_dir = repo_root / "models"
    model_dir = models_dir / f"{timestamp}_{run_id}"
    if not model_dir.exists():
        matches = list(models_dir.glob(f"*_{run_id}"))
        if not matches:
            raise AssertionError(f"No model folder created for run_id={run_id}")
        model_dir = max(matches, key=lambda p: p.stat().st_mtime)
    return model_dir


def _get_io_neurons(config: dict) -> tuple[list[str], list[str]]:
    input_neurons = ["L1", "L2", "L3"]
    output_neurons = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
    subset = config["model"].get("use_connectome_subset", [])
    if len(subset) > 0:
        input_neurons = [n for n in input_neurons if n in subset]
        output_neurons = [n for n in output_neurons if n in subset]
    return input_neurons, output_neurons


def _calc_loss(model, stimulus, target, input_neurons, output_neurons, loss_fn):
    with torch.no_grad():
        pred, _, _ = model.predict(
            stimulus,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            init_state=None,
        )
        if model.has_decoder:
            return loss_fn(target, pred["decoder"]).item()
        return tu.calc_loss_dict(loss_fn, pred["firing_rate"], target).item()


def test_run_model_synthetic_learns(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = f"pytest_{int(time.time())}"
    run_id = "synthetic"
    config, test_config_path = _load_test_config(repo_root)
    run_model.run_single_model(
        seed=0,
        device_override="cpu",
        run_id=run_id,
        timestamp=timestamp,
        config_path=str(test_config_path),
    )

    model_dir = _get_run_folder(repo_root, timestamp, run_id)
    stimulus = torch.load(model_dir / "cell_inputs.pt")
    target = torch.load(model_dir / "target.pt")

    input_neurons, output_neurons = _get_io_neurons(config)
    loss_fn = getattr(tu, config["training"]["loss_fn"])
    init_model = nm.NeuralModel.load(model_dir / "model_init.pt", device="cpu", freeze_params=False)
    trained_model = nm.NeuralModel.load(model_dir / "model.pt", device="cpu", freeze_params=False)

    device = init_model.device
    stimulus = stimulus.to(device=device, dtype=init_model.dtype)
    if isinstance(target, dict):
        target = {k: v.to(device=device, dtype=init_model.dtype) for k, v in target.items()}
    else:
        target = target.to(device=device, dtype=init_model.dtype)

    initial_loss = _calc_loss(init_model, stimulus, target, input_neurons, output_neurons, loss_fn)
    final_loss = _calc_loss(trained_model, stimulus, target, input_neurons, output_neurons, loss_fn)

    assert initial_loss > 0.0
    assert final_loss < 0.1 * initial_loss
