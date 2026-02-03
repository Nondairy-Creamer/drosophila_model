"""
Evaluate model optic flow predictions against movie stimuli.
"""
import argparse
import sys
from copy import deepcopy
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su
import training_utilities as tu


def _resolve_model_paths(model_path: Path) -> tuple[Path, Path]:
    if model_path.is_file():
        return model_path, model_path.parent
    model_pt = model_path / "model.pt"
    if model_pt.exists():
        return model_pt, model_path
    raise FileNotFoundError(f"Could not find model.pt at {model_path}")


def _load_model(model_pt: Path, config: dict, device: str) -> nm.CurrentModel:
    model_type = config["model"].get("model_type")
    ModelClass = getattr(nm, model_type)
    return ModelClass.load(model_pt, device=device)


def _prepare_stimulus_config(config: dict, device: str) -> dict:
    config_copy = deepcopy(config)
    config_copy["model"]["device"] = device
    config_copy["stimulus"]["pre_stim_time"] = 0.0
    return config_copy


def _strip_blank_stimulus(stimulus: torch.Tensor, optic_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if stimulus.shape[0] > 1:
        return stimulus[1:], optic_flow[1:]
    return stimulus, optic_flow


def _predict_optic_flow(model: nm.CurrentModel, stimulus: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if model.use_decoder:
            outputs = model.predict(stimulus, input_neurons=["L1", "L2", "L3"])[0]
            return outputs["decoder"]

        return model.calculate_optic_flow(stimulus)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RNN optic flow predictions.")
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="models/best_decoder/20260202_214948_gpu4_run0",
        help="Path to model folder or model.pt (default: models/best_model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (default: cpu)",
    )
    args = parser.parse_args()

    model_pt, model_folder = _resolve_model_paths(Path(args.model_path))
    config = du.load_config(model_folder / "config.toml")
    model = _load_model(model_pt, config, args.device)

    stim_config = _prepare_stimulus_config(config, str(model.device))
    column_phi_theta = model.column_phi_theta

    stimulus_types = [
        ("sintel", None),
        ("random_sine", column_phi_theta),
        ("flow_360", None),
    ]

    for stim_type, column_info in stimulus_types:
        stimulus, optic_flow = su.get_stimulus(stim_type, stim_config, column_info)
        stimulus, optic_flow = _strip_blank_stimulus(stimulus, optic_flow)

        pred_flow = _predict_optic_flow(model, stimulus)
        cosine_similarity = (1.0 - tu.cos_loss(optic_flow, pred_flow)).item()

        print(f"{stim_type}: cosine similarity = {cosine_similarity:.4f}")


if __name__ == "__main__":
    main()
