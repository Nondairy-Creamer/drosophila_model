from pathlib import Path

import pandas as pd

import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


import connectome_loading as cl
import data_utilities as du
import neural_models as nm


def _resolve_connectome_folder(repo_root: Path, connectome_folder: str) -> Path:
    connectome_path = Path(connectome_folder)
    if connectome_path.is_absolute():
        return connectome_path
    return (repo_root / connectome_path).resolve()


def load_test_config() -> tuple[dict, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    config = du.load_config(str(repo_root / "tests" / "config_test.toml"))
    return config, repo_root


def build_model_from_test_config(readouts=None, use_input_conv=True):
    if readouts is None:
        readouts = ['FlowConv2Readout']
    config, repo_root = load_test_config()
    config["model"]["device"] = "cpu"
    config["model"]["readouts"] = readouts
    config["model"]["use_input_conv"] = use_input_conv

    connectome_folder = _resolve_connectome_folder(repo_root, config["data"]["connectome_folder"])
    cell_info_df = pd.read_parquet(connectome_folder / "optic_lobes_cell_info.parquet")
    synapse_counts_df = pd.read_parquet(connectome_folder / "optic_lobes_synapse_counts.parquet")
    column_info_df = pd.read_parquet(connectome_folder / "optic_lobes_column_info.parquet")

    input_neurons = ["L1", "L2", "L3"]
    output_neurons = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

    if len(config["model"]["use_connectome_subset"]) > 0:
        cell_info_df, synapse_counts_df = du.get_connectome_subset(
            cell_info_df, synapse_counts_df, config["model"]["use_connectome_subset"]
        )
        input_neurons = [i for i in input_neurons if i in config["model"]["use_connectome_subset"]]
        output_neurons = [i for i in output_neurons if i in config["model"]["use_connectome_subset"]]

    connectome = cl.load_connectome_from_dataframes(
        cell_info_df, column_info_df, synapse_counts_df,
        name='optic_lobe_right', is_single_eye=True,
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
        readouts=config["model"]["readouts"],
        use_input_conv=config["model"]["use_input_conv"],
        dt=config["model"]["dt"],
        seed=0,
        commit_hash=None,
    )

    return model, config, input_neurons, output_neurons


