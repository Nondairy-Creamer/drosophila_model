"""
Utilities for running models on common stimuli.
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


def _resolve_model_paths(model_path: Path) -> tuple[Path, Path]:
    if model_path.is_file():
        return model_path, model_path.parent
    model_pt = model_path / "model.pt"
    if model_pt.exists():
        return model_pt, model_path
    raise FileNotFoundError(f"Could not find model.pt at {model_path}")


def _prepare_stimulus_config(config: dict, device: str) -> dict:
    config_copy = deepcopy(config)
    config_copy["model"]["device"] = device
    config_copy["stimulus"]["pre_stim_time"] = 0.0
    return config_copy


def _strip_blank_stimulus(stimulus: torch.Tensor, optic_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if stimulus.shape[0] > 1:
        return stimulus[1:], optic_flow[1:]
    return stimulus, optic_flow


def _predict_optic_flow(model: nm.NeuralModel, stimulus: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if model.use_decoder:
            outputs = model.predict(stimulus, input_neurons=["L1", "L2", "L3"])[0]
            return outputs["decoder"]

        return model.calculate_optic_flow(stimulus)[0]


def run_model_on_stimuli(
    model_path: str | Path = "models/best_model",
    device: str = "cpu",
    stimulus_types: list[str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    model_pt, model_folder = _resolve_model_paths(Path(model_path))
    config = du.load_config(model_folder / "config.toml")
    model = nm.NeuralModel.load(model_pt, device=device)

    stim_config = _prepare_stimulus_config(config, str(model.device))
    column_phi_theta = model.column_phi_theta
    column_xyz = model.get_column_xyz()

    if stimulus_types is None:
        stimulus_types = ["sintel", "flow360", "random_sine"]

    stimulus_map = {
        "sintel": ("sintel", None),
        "flow360": ("flow_360", None),
        "random_sine": ("random_sine", column_phi_theta),
    }

    responses: dict[str, dict[str, torch.Tensor]] = {}
    for stim_name in stimulus_types:
        if stim_name not in stimulus_map:
            raise ValueError(f"Unsupported stimulus type: {stim_name}")
        stim_type, column_info = stimulus_map[stim_name]
        stimulus, optic_flow = su.get_stimulus(stim_type, stim_config, column_info)
        stimulus, optic_flow = _strip_blank_stimulus(stimulus, optic_flow)

        pred_flow = _predict_optic_flow(model, stimulus)
        responses[stim_name] = {
            "stimulus": stimulus,
            "optic_flow": optic_flow,
            "pred_flow": pred_flow,
            "column_xyz": torch.tensor(column_xyz, device=model.device, dtype=model.dtype),
            "dt": torch.tensor(model.dt, device=model.device, dtype=model.dtype),
        }

    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RNN optic flow predictions.")
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="models/best_model",
        help="Path to model folder or model.pt (default: models/best_model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (default: cpu)",
    )
    args = parser.parse_args()

    responses = run_model_on_stimuli(args.model_path, device=args.device)
    print(f"Loaded responses for: {', '.join(responses.keys())}")


if __name__ == "__main__":
    main()
