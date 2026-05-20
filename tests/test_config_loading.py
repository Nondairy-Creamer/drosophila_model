from pathlib import Path

import data_utilities as du


def test_load_config_merges_and_converts_none(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    user_config = tmp_path / "config_test_override.toml"
    user_config.write_text(
        "\n".join(
            [
                "[model]",
                'device = "cpu"',
                "readouts = ['FlowConv2Readout']",
                "",
                "[training]",
                "num_steps = 123",
                "alpha = 0.123",
                'clip_norm = "none"',
                "",
                "[stimulus]",
                'seed = "none"',
                "",
            ]
        )
    )

    config = du.load_config(str(user_config))

    assert config["model"]["device"] == "cpu"
    assert config["model"]["readouts"] == ['FlowConv2Readout']
    assert config["training"]["num_steps"] == 123
    assert config["training"]["alpha"] == 0.123
    assert config["training"]["loss_components"] == ['flow_loss']
    assert config["training"]["clip_norm"] is None
    assert config["stimulus"]["seed"] is None
    assert config["data"]["hemisphere_coverage"] == 0.85
