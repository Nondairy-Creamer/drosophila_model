"""
Convert saved model checkpoints to the current NeuralModel save format.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

import data_utilities as du
import neural_models as nm


def _load_raw_checkpoint(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "inputs" not in data or "state_dict" not in data:
        raise ValueError(f"Unexpected checkpoint format at {path}")
    return data


def _restore_model(data: dict, use_input_conv: bool, device: str) -> nm.NeuralModel:
    class_name = data.get("class_name", "CurrentModel")
    ModelClass = getattr(nm, class_name)
    inputs = dict(data["inputs"])
    inputs["init_new_model"] = False
    if use_input_conv:
        inputs["use_input_conv"] = True
    inputs["device"] = device
    init_params = inspect.signature(ModelClass.__init__).parameters
    valid_keys = set(init_params.keys()) - {"self"}
    filtered_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
    model = ModelClass(**filtered_inputs)
    model.load_state_dict(data["state_dict"], strict=False)
    return model


def convert_model_file(path: Path, use_input_conv: bool, device: str) -> None:
    data = _load_raw_checkpoint(path)
    model = _restore_model(data, use_input_conv=use_input_conv, device=device)
    model.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert model checkpoints to current format.")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("models/best_model"),
        help="Model folder containing model_init.pt and model.pt",
    )
    parser.add_argument(
        "--use-input-conv",
        action="store_true",
        help="Enable input convolution (default: False)",
    )
    args = parser.parse_args()

    model_dir = args.model_folder
    # Use local config (repo config.toml) for device placement
    config = du.load_config()
    device = config["model"]["device"]
    for name in ("model_init.pt", "model.pt"):
        convert_model_file(model_dir / name, use_input_conv=args.use_input_conv, device=device)


if __name__ == "__main__":
    main()
