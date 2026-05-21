from pathlib import Path

import torch

from drosophila_model.model_definition.model_types import neural_models as nm
from tests.test_model_factory import build_model_from_test_config


def test_model_save_load_roundtrip(tmp_path):
    model, _, _, _ = build_model_from_test_config(readouts=[])
    model_path = Path(tmp_path) / "model.pt"

    model.save(model_path)
    loaded = nm.BaseRnn.load(model_path, device="cpu", dtype="float32", freeze_params=False)

    original_state = model.state_dict()
    loaded_state = loaded.state_dict()

    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        torch.testing.assert_close(original_state[key], loaded_state[key])
