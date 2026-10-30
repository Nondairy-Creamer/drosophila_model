from pathlib import Path

import pytest

from drosophila_model.config import get_data_folder, load_config, load_run_config, save_config

TEST_CONFIG_PATH = Path(__file__).resolve().parent / "config_test.toml"


def test_load_config_merges_and_converts_none(tmp_path, monkeypatch):
    # Run from an empty folder so a local config.toml / config_data.toml
    # cannot leak into the result.
    monkeypatch.chdir(tmp_path)

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
    assert config["training"]["loss_components"] == ['flow_loss', 'odometry_loss']
    assert config["training"]["clip_norm"] is None
    assert config["stimulus"]["stimulus_type"] == 'edge'
    assert config["data"]["data_folder"] == "../drosophila_model_data"


def test_load_config_merges_data_config_under_data(tmp_path):
    data_folder = (tmp_path / "data").as_posix()
    main_config = tmp_path / "config_main.toml"
    main_config.write_text(f'[model]\ndevice = "cpu"\n\n[data]\ndata_folder = "{data_folder}"\n')

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
    # The configured data_folder survives the merge with the data tables.
    assert config["data"]["data_folder"] == data_folder


def test_load_config_rejects_unknown_data_keys(tmp_path):
    main_config = tmp_path / "config_main.toml"
    main_config.write_text("")

    bad_data_config = tmp_path / "data_config_bad.toml"
    bad_data_config.write_text("[shared]\nnot_a_real_key = 1\n")

    with pytest.raises(ValueError):
        load_config(str(main_config), str(bad_data_config))


def test_test_config_matches_schema(tmp_path, monkeypatch):
    """tests/config_test.toml passes strict validation (catches schema drift
    without needing the connectome data)."""
    monkeypatch.chdir(tmp_path)
    config = load_config(str(TEST_CONFIG_PATH))
    assert config["model"]["use_connectome_subset"]


def test_save_config_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(str(TEST_CONFIG_PATH))
    config["run_info"] = {"seed": 3, "num_eyes": 2, "eye_offset": [0.05, -0.05, 0.013]}
    assert config["training"]["clip_norm"] is None  # stored as "none" in TOML

    run_dir = tmp_path / "models" / "run"
    save_config(config, run_dir / "config.toml")
    # Overwriting an existing file must work too (atomic replace on Windows).
    save_config(config, run_dir / "config.toml")

    assert sorted(p.name for p in run_dir.iterdir()) == ["config.toml"]
    assert load_run_config(run_dir) == config


def _write_saved_run(run_dir):
    """A saved run config with its own data tables."""
    run_dir.mkdir(parents=True)
    (run_dir / "config.toml").write_text(
        "\n".join(
            [
                "[model]",
                'device = "cpu"',
                "dt = 0.02",
                "",
                "[training]",
                "num_steps = 7",
                "",
                "[run_info]",
                "seed = 3",
                "",
                "[data]",
                'data_folder = "/somewhere/on/the/training/machine"',
                "",
                "[data.shared]",
                "stim_time = 2.5",
                "",
                "[data.random_pano]",
                "velocity_amp = 99.0",
                "",
            ]
        )
    )


def test_load_run_config_keeps_saved_tables(tmp_path, monkeypatch):
    run_dir = tmp_path / "models" / "run"
    _write_saved_run(run_dir)

    # This machine's config.toml / config_data.toml.
    local_data = (tmp_path / "local_data").as_posix()
    (tmp_path / "config.toml").write_text(f'[data]\ndata_folder = "{local_data}"\n')
    (tmp_path / "config_data.toml").write_text("[shared]\nstim_time = 9.0\n")
    monkeypatch.chdir(tmp_path)

    config = load_run_config(run_dir)

    # Saved values win over the defaults.
    assert config["training"]["num_steps"] == 7
    assert config["model"]["dt"] == 0.02
    assert config["run_info"]["seed"] == 3
    # Saved data tables are kept (not replaced by the local config_data.toml);
    # keys added since the run get their default.
    assert config["data"]["shared"]["stim_time"] == 2.5
    assert config["data"]["random_pano"]["velocity_amp"] == 99.0
    assert config["data"]["shared"]["pre_stim_time"] == 0.5
    # The data location is a property of this machine.
    assert config["data"]["data_folder"] == local_data
    # The config.toml path itself is accepted too.
    assert load_run_config(run_dir / "config.toml") == config


def test_load_run_config_without_local_data_folder(tmp_path, monkeypatch):
    run_dir = tmp_path / "models" / "run"
    _write_saved_run(run_dir)
    monkeypatch.chdir(tmp_path)  # no local config.toml

    config = load_run_config(run_dir)

    assert config["data"]["data_folder"] == "/somewhere/on/the/training/machine"


def test_get_data_folder(tmp_path):
    with pytest.raises(ValueError, match="data_folder is empty"):
        get_data_folder({"data": {"data_folder": ""}})
    with pytest.raises(ValueError, match="data_folder is empty"):
        get_data_folder({})
    with pytest.raises(FileNotFoundError):
        get_data_folder({"data": {"data_folder": str(tmp_path / "missing")}})

    assert get_data_folder({"data": {"data_folder": str(tmp_path)}}) == tmp_path
