from pathlib import Path

import pytest
import torch

from drosophila_model.model_definition import neural_models as nm
from tests.model_factory import build_model_from_test_config


def test_model_save_load_roundtrip(tmp_path):
    model, _ = build_model_from_test_config(readouts=[])
    model_path = Path(tmp_path) / "model.pt"

    model.save(model_path)
    loaded = nm.BaseRnn.load(model_path, device="cpu", dtype="float32", freeze_params=False)

    original_state = model.state_dict()
    loaded_state = loaded.state_dict()

    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        torch.testing.assert_close(original_state[key], loaded_state[key])


def _save_under_old_name(model, path, new_name, old_name):
    """Save ``model`` as a model saved before the readout rename would be."""
    model.save(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    data["inputs"]["region_kwargs"]["optic_lobe"]["readouts"] = [old_name]
    data["state_dict"] = {k.replace(f"._readout_modules.{new_name}.", f"._readout_modules.{old_name}."): v
                          for k, v in data["state_dict"].items()}
    torch.save(data, path)


@pytest.mark.parametrize("new_name, old_name", [
    ("FlowConv2DenomReadout", "FlowConv2Readout"),
    ("FlowConv2Readout", "FlowConv2NoDenomReadout"),
])
def test_load_renamed_readouts(tmp_path, new_name, old_name):
    model, _ = build_model_from_test_config(readouts=[new_name])
    model_path = Path(tmp_path) / "model.pt"
    _save_under_old_name(model, model_path, new_name, old_name)

    loaded = nm.BaseRnn.load(model_path, device="cpu", dtype="float32", freeze_params=False)

    assert list(loaded.regions["optic_lobe"]._readout_modules) == [new_name]
    original_state = model.state_dict()
    loaded_state = loaded.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        torch.testing.assert_close(original_state[key], loaded_state[key])
