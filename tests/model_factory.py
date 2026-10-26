"""Shared helpers for the tests that build a small connectome model.

Not a test module itself: import it as ``from tests.model_factory import ...``.

The connectome data is located the same way as everywhere else in the repo:
``[data].data_folder`` from the repo-root ``config.toml`` (or the default).
Tests that need it are skipped with a clear reason when the folder or the
baked connectome is missing.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.model_definition import connectome_loading as cl
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import regions as regions_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIG_PATH = REPO_ROOT / "tests" / "config_test.toml"


def data_folder() -> Path:
    """``[data].data_folder`` (repo-root config.toml or default), or skip the test."""
    config_path = REPO_ROOT / "config.toml"
    # An invalid config.toml should fail loudly, not skip: only a missing data
    # location is a reason to skip.
    config = load_config(str(config_path)) if config_path.exists() else load_config()
    try:
        folder = get_data_folder(config)
    except (ValueError, FileNotFoundError) as exc:
        pytest.skip(f"connectome data unavailable: {exc}")
    baked = folder / "connectome_data" / "baked"
    if not (baked / "cell_info.parquet").exists():
        pytest.skip(f"{baked} has no cell_info.parquet; the model tests need the baked connectome")
    return folder.resolve()


def load_test_config() -> dict:
    """tests/config_test.toml merged over the defaults, pointed at the local data."""
    config = load_config(str(TEST_CONFIG_PATH))
    config["data"]["data_folder"] = str(data_folder())
    return config


@lru_cache(maxsize=None)
def _connectome_subset(folder: str, cell_types: tuple[str, ...]):
    """Read the baked connectome and cut it to ``cell_types``, once per test
    session (synapses.parquet is large; the loaders do not modify the frames)."""
    baked = Path(folder) / "connectome_data" / "baked"
    return cl.get_connectome_subset(
        pd.read_parquet(baked / "cell_info.parquet"),
        pd.read_parquet(baked / "synapses.parquet"),
        list(cell_types),
    )


def build_model_from_test_config(readouts=None, use_input_conv=True, dynamics_type=None,
                                 side=None):
    """Build the small test model described by tests/config_test.toml.

    Constructed with the same arguments as ``training/run_model.py``; input
    and output neurons are left to the optic-lobe region defaults. ``side``
    overrides ``[model.regions.optic_lobe].side``. Returns ``(model, config)``.
    """
    if readouts is None:
        readouts = ['FlowConv2Readout']
    config = load_test_config()
    model_cfg = config["model"]
    if dynamics_type is not None:
        model_cfg["dynamics_type"] = dynamics_type
    model_cfg["regions"]["optic_lobe"]["use_input_conv"] = use_input_conv
    model_cfg["regions"]["optic_lobe"]["readouts"] = readouts
    if side is not None:
        model_cfg["regions"]["optic_lobe"]["side"] = side

    # The test config always uses a cell-type subset, which keeps builds fast.
    subset = model_cfg["use_connectome_subset"]
    assert subset, "tests/config_test.toml must set [model].use_connectome_subset"
    cell_info_df, synapses_df = _connectome_subset(config["data"]["data_folder"], tuple(subset))
    connectome = cl.load_connectome_from_dataframes(
        cell_info_df, synapses_df,
        region_specs=regions_mod.build_region_specs(model_cfg["regions"]),
        data_folder=config["data"]["data_folder"],
        count_threshold=model_cfg["synapse_count_threshold"],
    )

    model_class = nm.compose_model_class(
        dynamics_type=model_cfg["dynamics_type"],
        model_type=model_cfg["model_type"],
    )
    model = model_class(
        connectome=connectome,
        nonlinearity=model_cfg["nonlinearity"],
        count_threshold=model_cfg["synapse_count_threshold"],
        use_layer_norm=model_cfg["use_layer_norm"],
        device=model_cfg["device"],
        dtype=model_cfg["dtype"],
        region_kwargs=model_cfg["regions"],
        dt=model_cfg["dt"],
        seed=0,
        commit_hash=None,
        **nm.dynamics_kwargs(model_cfg),
    )
    return model, config
