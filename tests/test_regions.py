"""Tests for region-based composition on :class:`ConnectomeModel`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from drosophila_model.model_definition import connectome_loading as cl
from drosophila_model.model_definition import regions as regions_mod
from tests.model_factory import build_model_from_test_config


def test_optic_lobe_region_registered():
    """A model configured with [model.regions.optic_lobe] exposes an OpticLobeRegion."""
    model, config = build_model_from_test_config(readouts=[])

    assert "optic_lobe" in config["model"]["regions"]
    assert "optic_lobe" in model.regions

    region = model.regions["optic_lobe"]
    assert isinstance(region, regions_mod.OpticLobeRegion)
    assert region.side == "right"
    assert region.num_sides == 1
    assert region.num_eyes == 1  # eye-terminology alias


def test_optic_lobe_region_indices_consistent():
    """Region cell indices match the loader's region_membership, and the
    column set is the full baked geometry (column id == position)."""
    model, config = build_model_from_test_config(readouts=[])
    region = model.regions["optic_lobe"]

    expected_cells = np.flatnonzero(model.region_membership["optic_lobe"])
    np.testing.assert_array_equal(region.cell_indices, expected_cells)

    columns_path = (Path(config["data"]["data_folder"]) / "connectome_data"
                    / "baked" / "optic_lobe_columns.parquet")
    expected_cols = np.arange(len(pd.read_parquet(columns_path)))
    np.testing.assert_array_equal(region.column_indices, expected_cols)
    assert region.num_columns == len(expected_cols)

    # Every neuron's column id is a valid position on the column axis.
    col_vals = region.column_indices_per_neuron
    col_vals = col_vals[~np.isnan(col_vals)]
    assert ((col_vals >= 0) & (col_vals < region.num_columns)).all()


@pytest.mark.parametrize("side", ["left", "both"])
def test_optic_lobe_rejects_non_right_side(side):
    """Only the right optic lobe is supported (right-eye column geometry)."""
    with pytest.raises(ValueError, match="supports only side='right'"):
        build_model_from_test_config(readouts=[], side=side)


def test_build_region_specs_defaults_and_unknown_names():
    """Unknown region names raise; side defaults to the region class default."""
    with pytest.raises(ValueError, match="optic_lobe"):
        regions_mod.build_region_specs({"optic_lobes": {}})
    specs = regions_mod.build_region_specs({"optic_lobe": {}})
    assert specs["optic_lobe"]["side"] == "right"
    assert specs["optic_lobe"]["neuropils"] == ["ME", "LO", "LOP", "LA"]


def test_optic_lobe_input_mapping_abstains_without_neurons():
    """An empty input type list maps nothing instead of crashing."""
    model, _ = build_model_from_test_config(readouts=[])
    region = model.regions["optic_lobe"]
    neuron_idx, column_idx = region.get_shared_columns([])
    assert len(neuron_idx) == 0 and len(column_idx) == 0
    assert region.build_input_mapping([]) is None


@pytest.mark.parametrize("readout", ["HexGridReadout", "HexGridConv", "Regularizer", "NoSuchReadout"])
def test_optic_lobe_rejects_non_readout_classes(readout):
    """Only concrete readouts and standalone regularizers may be configured."""
    with pytest.raises(ValueError, match="Unknown readout class"):
        build_model_from_test_config(readouts=[readout])


def test_remove_cell_types_keeps_initial_state():
    """A trimmed model starts from the source model's state for kept neurons."""
    model, _ = build_model_from_test_config(readouts=[])
    with torch.no_grad():
        model.steady_state.copy_(torch.linspace(-1, 1, model.num_neurons))
    trimmed = model.remove_cell_types(["Tm9"])

    assert "Tm9" not in trimmed.unique_cell_types
    assert trimmed.num_neurons < model.num_neurons
    position = {int(rid): i for i, rid in enumerate(model.root_ids)}
    expected = model.steady_state[[position[int(rid)] for rid in trimmed.root_ids]]
    torch.testing.assert_close(trimmed.steady_state, expected)


def test_optic_lobe_two_eye_buffers_not_persistent():
    """The two-eye viewing directions follow .to(device) but are not saved."""
    model, _ = build_model_from_test_config(readouts=[])
    buffers = dict(model.named_buffers())
    assert "regions.optic_lobe.column_phi_two_eyes" in buffers
    assert "regions.optic_lobe.column_theta_two_eyes" in buffers
    state = model.state_dict()
    assert "regions.optic_lobe.column_phi_two_eyes" not in state
    assert "regions.optic_lobe.column_theta_two_eyes" not in state


def test_region_specs_round_trip_to_loader():
    """build_region_specs + load_connectome plumbing yields the right neuropil set."""
    model, _ = build_model_from_test_config(readouts=[])
    spec = model.region_specs["optic_lobe"]
    resolved = cl.resolve_sided_neuropils(spec["neuropils"], spec["side"])
    # Right-side optic lobe = right variants of ME/LO/LOP/LA only.
    assert resolved == {"ME_R", "LO_R", "LOP_R", "LA_R"}


def test_optic_lobe_sign_overrides_flip_c2_c3_to_l1_l2():
    """C2/C3 (GABA) -> L1/L2 are flipped to positive by the OL anatomy prior.

    The OpticLobeRegion declares C2/C3 -> L1/L2 in ``sign_overrides`` so a
    GABA pre-synaptic NT (raw sign -1) ends up positive after
    ``_initialize_synapse_signs`` runs.
    """
    model, _ = build_model_from_test_config(readouts=[])

    region = model.regions["optic_lobe"]
    assert ("C2", "L1", -1.0) in [tuple(t) for t in region.sign_overrides]
    assert ("C2", "L2", -1.0) in [tuple(t) for t in region.sign_overrides]
    assert ("C3", "L1", -1.0) in [tuple(t) for t in region.sign_overrides]
    assert ("C3", "L2", -1.0) in [tuple(t) for t in region.sign_overrides]

    sc = model.synapse_counts
    pre_n = sc.col_indices()
    post_n = model._csr_row_indices(sc)
    pre_types = model.neuron_to_type_index[pre_n].cpu().numpy()
    post_types = model.neuron_to_type_index[post_n].cpu().numpy()
    values = sc.values().detach().cpu().numpy()

    def type_idx(name):
        return model._cell_type_name_to_index(name)

    lamina_post = np.isin(post_types, [type_idx("L1"), type_idx("L2")])
    num_checked = 0
    for pre_name in ("C2", "C3"):
        any_pre = pre_types == type_idx(pre_name)
        for post_name in ("L1", "L2"):
            edge_mask = any_pre & (post_types == type_idx(post_name))
            # All edges between this pre/post pair must be non-negative
            # (the GABA -1 flipped by factor -1 -> positive count).
            assert (values[edge_mask] >= 0).all(), (
                f"{pre_name}->{post_name} still has negative weights after "
                f"region sign override"
            )
            num_checked += int(edge_mask.sum())
        # The override is targeted: other C2/C3 (GABA) edges stay inhibitory.
        other = any_pre & ~lamina_post
        assert other.any() and (values[other] < 0).all(), (
            f"{pre_name} edges onto other cell types should stay negative"
        )
    assert num_checked > 0, "no C2/C3 -> L1/L2 edges in the test connectome"


def test_conductance_sign_overrides_route_c2_c3_to_excitatory():
    """ConductanceDynamics carries C2/C3 -> L1/L2 on the ACH conductance.

    Only those contacts move: other GABA edges (C2/C3 onto other cell types)
    stay on the GABA conductance.
    """
    model, _ = build_model_from_test_config(readouts=[], dynamics_type='ConductanceDynamics')
    transmitters = list(model.unique_transmitters)

    def edges(name):
        mat = model.synapse_counts[transmitters.index(name)]
        pre = model.neuron_to_type_index[mat.col_indices()].cpu().numpy()
        post = model.neuron_to_type_index[model._csr_row_indices(mat)].cpu().numpy()
        return pre, post

    lamina = [model._cell_type_name_to_index(n) for n in ("L1", "L2")]
    c2_c3 = [model._cell_type_name_to_index(n) for n in ("C2", "C3")]

    ach_pre, ach_post = edges('ACH')
    gaba_pre, gaba_post = edges('GABA')
    assert (np.isin(ach_pre, c2_c3) & np.isin(ach_post, lamina)).any()
    assert not (np.isin(gaba_pre, c2_c3) & np.isin(gaba_post, lamina)).any()
    assert (np.isin(gaba_pre, c2_c3) & ~np.isin(gaba_post, lamina)).any()


def test_optic_lobe_column_phi_theta_two_eyes_mirrored():
    """Single-side host: column_phi_theta_two_eyes mirrors phi for the fake left eye."""
    model, _ = build_model_from_test_config(readouts=[])
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
    model, _ = build_model_from_test_config(readouts=["FlowConv2Readout"])
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

    # Initial states must be given for both eyes or for neither.
    init_state = model.steady_state.unsqueeze(0).expand(S, -1)
    with pytest.raises(ValueError, match="init_state_right and init_state_left"):
        region.predict_two_eyes(inputs_right, inputs_left,
                                init_state_right=init_state)
