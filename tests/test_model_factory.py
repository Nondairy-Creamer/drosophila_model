from pathlib import Path

import pandas as pd

import sys

# Add parent directory to path for imports


from drosophila_model.config import load_config
from drosophila_model.utilities import connectome_loading as cl
from drosophila_model.utilities import data_utilities as du
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import regions as regions_mod


def _resolve_data_folder(repo_root: Path, data_folder: str) -> Path:
    connectome_path = Path(data_folder)
    if connectome_path.is_absolute():
        return connectome_path
    candidate = (repo_root / connectome_path).resolve()
    if candidate.exists():
        return candidate
    # Fall back to the live config.toml's data folder so dev machines
    # without the in-repo fixture (which only exists on CI) can still run.
    live_path = repo_root / "config.toml"
    if live_path.exists():
        live_cfg = load_config(str(live_path))
        live_folder = Path(live_cfg["data"]["data_folder"])
        if live_folder.is_absolute() and live_folder.exists():
            return live_folder
    return candidate


def load_test_config() -> tuple[dict, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(str(repo_root / "tests" / "config_test.toml"))
    return config, repo_root


def build_model_from_test_config(readouts=None, use_input_conv=True):
    if readouts is None:
        readouts = ['FlowConv2Readout']
    config, repo_root = load_test_config()
    config["model"]["device"] = "cpu"
    config["model"].setdefault("regions", {}).setdefault("optic_lobe", {})
    config["model"]["regions"]["optic_lobe"]["use_input_conv"] = use_input_conv
    config["model"]["regions"]["optic_lobe"]["readouts"] = readouts

    data_folder = _resolve_data_folder(repo_root, config["data"]["data_folder"])
    baked = data_folder / "connectome_data" / "baked"
    cell_info_df = pd.read_parquet(baked / "cell_info.parquet")
    synapses_df = pd.read_parquet(baked / "synapses.parquet")

    input_neurons = ["L1", "L2", "L3"]
    output_neurons = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

    if len(config["model"]["use_connectome_subset"]) > 0:
        cell_info_df, synapses_df = du.get_connectome_subset(
            cell_info_df, synapses_df, config["model"]["use_connectome_subset"]
        )
        input_neurons = [i for i in input_neurons if i in config["model"]["use_connectome_subset"]]
        output_neurons = [i for i in output_neurons if i in config["model"]["use_connectome_subset"]]

    connectome = cl.load_connectome_from_dataframes(
        cell_info_df, synapses_df,
        region_specs=regions_mod.build_region_specs(
            config["model"].get("regions", {})
        ),
        data_folder=str(data_folder),
        count_threshold=config["model"]["synapse_count_threshold"],
    )

    model_class = nm.compose_model_class(
        dynamics_type=config["model"]["dynamics_type"],
        model_type=config["model"]["model_type"],
    )

    model = model_class(
        connectome=connectome,
        nonlinearity=config["model"]["nonlinearity"],
        count_threshold=config["model"]["synapse_count_threshold"],
        device=config["model"]["device"],
        dtype=config["model"]["dtype"],
        region_kwargs=config["model"].get("regions", {}),
        dt=config["model"]["dt"],
        seed=0,
        commit_hash=None,
    )

    return model, config, input_neurons, output_neurons


