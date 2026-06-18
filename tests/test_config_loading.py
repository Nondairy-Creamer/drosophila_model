from pathlib import Path

import pytest

from drosophila_model.config import load_config


def test_load_config_merges_and_converts_none(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    user_config = tmp_path / "config_test_override.toml"
    user_config.write_text(
        "\n".join(
            [
                "[model]",
                'device = "cpu"',
                "",
                "[model.regions.optic_lobe]",
                "readouts = ['FlowConv2Readout', 'L1Reg']",
                "",
                "[training]",
                "num_steps = 123",
                "alpha = 0.123",
                'clip_norm = "none"',
                "",
                "[stimulus]",
                "stimulus_type = 'edge'",
                "",
            ]
        )
    )

    config = load_config(str(user_config))

    assert config["model"]["device"] == "cpu"
    assert config["model"]["regions"]["optic_lobe"]["readouts"] == ['FlowConv2Readout', 'L1Reg']
    assert config["training"]["num_steps"] == 123
    assert config["training"]["alpha"] == 0.123
    assert config["training"]["loss_components"] == ['flow_loss']
    assert config["training"]["clip_norm"] is None
    assert config["stimulus"]["stimulus_type"] == 'edge'
    assert config["data"]["data_folder"] == "../drosophila_model_data"


def test_load_config_merges_data_config_under_data(tmp_path):
    main_config = tmp_path / "config_main.toml"
    main_config.write_text("")

    data_config = tmp_path / "data_config_override.toml"
    data_config.write_text(
        "\n".join(
            [
                "[shared]",
                "stim_time = 7.5",
                'seed = "none"',
                "",
                "[random_pano]",
                "velocity_amp = 250.0",
                "",
            ]
        )
    )

    config = load_config(str(main_config), str(data_config))

    # Data config nested under config['data'], overlay applied.
    assert config["data"]["shared"]["stim_time"] == 7.5
    assert config["data"]["shared"]["seed"] is None
    assert config["data"]["random_pano"]["velocity_amp"] == 250.0
    # Untouched data defaults preserved.
    assert config["data"]["shared"]["pre_stim_time"] == 0.5
    assert config["data"]["random_pano"]["num_repeats"] == 1
    assert config["data"]["edge"]["edge_velocities"] == [30, 60, 120, 240]
    # data_folder still lives alongside the merged data tables.
    assert config["data"]["data_folder"] == "../drosophila_model_data"


def test_load_config_rejects_unknown_data_keys(tmp_path):
    main_config = tmp_path / "config_main.toml"
    main_config.write_text("")

    bad_data_config = tmp_path / "data_config_bad.toml"
    bad_data_config.write_text("[shared]\nnot_a_real_key = 1\n")

    with pytest.raises(ValueError):
        load_config(str(main_config), str(bad_data_config))
