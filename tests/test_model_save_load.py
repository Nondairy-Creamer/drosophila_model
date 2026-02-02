from pathlib import Path

import numpy as np
import torch

import neural_models as nm


def _build_minimal_model():
    root_ids = np.array([1, 2], dtype=np.int64)
    synapse_counts_coo = np.array(
        [
            [1, 2, 3],
            [2, 1, 3],
        ],
        dtype=np.int64,
    )
    cell_types = np.array(["L1", "T4a"])
    neurotransmitters = np.array(["ACH", "GABA"])
    column_indices = np.array([0, 0], dtype=float)
    column_phi_theta = np.array([[0.1, 0.2], [0.1, 0.2]], dtype=float)

    return nm.CurrentModel(
        synapse_counts_coo=synapse_counts_coo,
        root_ids=root_ids,
        cell_types=cell_types,
        neurotransmitters=neurotransmitters,
        column_indices=column_indices,
        column_phi_theta=column_phi_theta,
        device="cpu",
        dtype="float32",
        use_decoder=False,
        count_threshold=1,
        seed=0,
    )


def test_model_save_load_roundtrip(tmp_path):
    model = _build_minimal_model()
    model_path = Path(tmp_path) / "model.pt"

    model.save(model_path)
    loaded = nm.CurrentModel.load(model_path, device="cpu", dtype="float32", freeze_params=False)

    original_state = model.state_dict()
    loaded_state = loaded.state_dict()

    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        torch.testing.assert_close(original_state[key], loaded_state[key])
