from pathlib import Path

import pandas as pd

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


def build_model_from_test_config(use_decoder: bool = True, use_input_conv: bool = True):
    config, repo_root = load_test_config()
    config["model"]["device"] = "cpu"
    config["model"]["use_decoder"] = use_decoder
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

    model_type = config["model"].get("model_type")
    model_class = getattr(nm, model_type)

    model = model_class(
        synapse_counts_coo=synapse_counts_df[["pre_root_id", "post_root_id", "synapse_counts"]].to_numpy(),
        root_ids=cell_info_df["root_id"].to_numpy(),
        cell_types=cell_info_df["type"].to_numpy(),
        neurotransmitters=cell_info_df["neurotransmitter"].to_numpy(),
        column_indices=cell_info_df["column_index"].to_numpy(),
        column_phi_theta=column_info_df[["phi", "theta"]].to_numpy(),
        nonlinearity=config["model"]["nonlinearity"],
        count_threshold=config["model"]["synapse_count_threshold"],
        device=config["model"]["device"],
        dtype=config["model"]["dtype"],
        use_decoder=config["model"]["use_decoder"],
        use_input_conv=config["model"]["use_input_conv"],
        dt=config["model"]["dt"],
        seed=0,
        commit_hash=None,
    )

    return model, config, input_neurons, output_neurons


