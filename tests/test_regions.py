"""Tests for region-based composition on :class:`ConnectomeModel`."""

import numpy as np
import torch

from drosophila_model.model_definition import regions as regions_mod
from test_model_factory import build_model_from_test_config


def test_optic_lobe_region_registered():
    """A model configured with [model.regions.optic_lobe] exposes an OpticLobeRegion."""
    model, config, _, _ = build_model_from_test_config(readouts=[])

    assert "optic_lobe" in config["model"]["regions"]
    assert "optic_lobe" in model.regions

    region = model.regions["optic_lobe"]
    assert isinstance(region, regions_mod.OpticLobeRegion)
    assert region.side == "right"
    assert region.num_sides == 1
    assert region.num_eyes == 1  # eye-terminology alias


def test_optic_lobe_region_indices_consistent():
    """Region cell/column indices match the loader's region_membership."""
    model, _, _, _ = build_model_from_test_config(readouts=[])
    region = model.regions["optic_lobe"]

    expected_cells = np.flatnonzero(model.region_membership["optic_lobe"])
    np.testing.assert_array_equal(region.cell_indices, expected_cells)

    col_vals = region.column_indices_per_neuron[
        model.region_membership["optic_lobe"]
    ]
    expected_cols = np.unique(col_vals[~np.isnan(col_vals)]).astype(np.int64)
    np.testing.assert_array_equal(region.column_indices, expected_cols)


def test_region_specs_round_trip_to_loader():
    """build_region_specs + load_connectome plumbing yields the right neuropil set."""
    from drosophila_model.model_definition import regions as regions_mod
    from drosophila_model.utilities import connectome_loading as cl

    model, _, _, _ = build_model_from_test_config(readouts=[])
    spec = model.region_specs["optic_lobe"]
    resolved = cl.resolve_sided_neuropils(spec["neuropils"], spec["side"])
    # Right-side optic lobe = right variants of ME/LO/LOP/LA only.
    assert resolved == {"ME_R", "LO_R", "LOP_R", "LA_R"}


def test_union_membership_supports_multiple_regions():
    """A region's cell_indices comes straight from its membership mask."""
    model, _, _, _ = build_model_from_test_config(readouts=[])
    # Every optic-lobe neuron in the membership mask shows up in the region's
    # cell_indices, and every cell_index points back into the mask.
    mask = model.region_membership["optic_lobe"]
    assert mask.sum() == len(model.regions["optic_lobe"].cell_indices)
    assert mask[model.regions["optic_lobe"].cell_indices.cpu().numpy()].all()


def test_optic_lobe_column_phi_theta_two_eyes_mirrored():
    """Single-side host: column_phi_theta_two_eyes mirrors phi for the fake left eye."""
    model, _, _, _ = build_model_from_test_config(readouts=[])
    region = model.regions["optic_lobe"]

    phi_2e, theta_2e = region.column_phi_theta_two_eyes
    n = model.regions['optic_lobe'].num_columns

    assert phi_2e.shape[0] == 2 * n
    assert theta_2e.shape[0] == 2 * n

    phi_np = model.regions['optic_lobe'].column_phi_theta[:, 0]
    np.testing.assert_allclose(phi_2e[:n].cpu().numpy(), phi_np)
    np.testing.assert_allclose(phi_2e[n:].cpu().numpy(), -phi_np)


def test_predict_two_eyes_runs():
    """Single-eye OpticLobeRegion.predict_two_eyes returns per-eye result dicts."""
    model, _, _, _ = build_model_from_test_config(readouts=["FlowConv2Readout"])
    model.eval()
    region = model.regions["optic_lobe"]

    S, T = 1, 3
    inputs_right = torch.zeros(S, T, model.regions['optic_lobe'].num_columns,
                               dtype=model.dtype, device=model.device)
    inputs_left = torch.zeros_like(inputs_right)

    with torch.no_grad():
        right_result, left_result = region.predict_two_eyes(
            inputs_right, inputs_left,
        )

    assert "model_state" in right_result and "model_state" in left_result
    assert right_result["model_state"]["v"].shape[0] == S
    assert left_result["model_state"]["v"].shape[0] == S
