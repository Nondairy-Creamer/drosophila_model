import shutil
import time
from pathlib import Path

import torch

import data_utilities as du
import neural_models as nm
import run_model
import training_utilities as tu


def _write_test_config(repo_root: Path) -> dict:
    config = du.load_config(str(repo_root / "tests" / "config_test.toml"))
    config["training"]["use_synthetic_data"] = True
    config["training"]["num_steps"] = 100
    config["model"]["use_decoder"] = False
    config["model"]["device"] = "cpu"
    config["model"]["dt"] = 0.01
    config["stimulus"]["stim_time"] = 1.0
    config["stimulus"]["pre_stim_time"] = 0.0
    config["stimulus"]["post_stim_time"] = 0.0
    config["stimulus"]["num_repeats"] = 1
    du.save_config(config, repo_root / "config.toml")
    return config


def _restore_config(repo_root: Path, backup_path: Path | None) -> None:
    config_path = repo_root / "config.toml"
    if backup_path is None:
        if config_path.exists():
            config_path.unlink()
        return
    shutil.copy2(backup_path, config_path)


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
        return tu.calc_loss_dict(loss_fn, pred["firing_rate"], target).item()


def test_run_model_synthetic_learns(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "config.toml"
    backup_path = None
    if config_path.exists():
        backup_path = tmp_path / "config_backup.toml"
        shutil.copy2(config_path, backup_path)

    timestamp = f"pytest_{int(time.time())}"
    run_id = "synthetic"
    try:
        config = _write_test_config(repo_root)
        run_model.run_single_model(
            seed=0,
            device_override="cpu",
            run_id=run_id,
            timestamp=timestamp,
        )
    finally:
        _restore_config(repo_root, backup_path)

    model_dir = _get_run_folder(repo_root, timestamp, run_id)
    stimulus = torch.load(model_dir / "cell_inputs.pt")
    target = torch.load(model_dir / "target.pt")

    assert stimulus.shape[1] == 100, f"Expected 100 time points, got {stimulus.shape[1]}"

    input_neurons, output_neurons = _get_io_neurons(config)
    loss_fn = getattr(tu, config["training"]["loss_fn"])
    model_class = getattr(nm, config["model"]["model_type"])

    init_model = model_class.load(model_dir / "model_init.pt", device="cpu", freeze_params=False)
    trained_model = model_class.load(model_dir / "model.pt", device="cpu", freeze_params=False)

    device = init_model.device
    stimulus = stimulus.to(device=device, dtype=init_model.dtype)
    target = {k: v.to(device=device, dtype=init_model.dtype) for k, v in target.items()}

    initial_loss = _calc_loss(init_model, stimulus, target, input_neurons, output_neurons, loss_fn)
    final_loss = _calc_loss(trained_model, stimulus, target, input_neurons, output_neurons, loss_fn)

    assert initial_loss > 0.0
    assert final_loss < 0.001, f"Final loss not close to 0: {final_loss}"
    assert final_loss < initial_loss
