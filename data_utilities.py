'''
Simple TOML configuration loader for the Drosophila neural model.
'''

from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import pickle

if TYPE_CHECKING:
    import pandas as pd

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    # Python <3.11 needs a writer; on 3.11+ prefer tomllib for reading and tomli_w for writing
    import tomli_w as toml_writer
except ImportError:  # pragma: no cover - tomli_w present per requirements for <3.11
    toml_writer = None

def load_config(config_path: str = 'config.toml') -> dict:
    '''
    Load configuration by merging default config with user config.
    
    Returns:
        Dictionary containing the merged configuration
    '''
    # Load default config
    default_path = Path('config_default.toml')
    if not default_path.exists():
        raise FileNotFoundError(f'Default configuration file not found: {default_path}')
    
    with open(default_path, 'rb') as f:
        config = tomllib.load(f)
    
    # Check for user config and merge if it exists
    user_path = Path(config_path)
    if user_path.exists():
        with open(user_path, 'rb') as f:
            user_config = tomllib.load(f)
        
        # Recursively merge user config into default config
        config = _merge_configs(config, user_config)
    
    # Convert "none" strings to Python None
    config = _convert_none_strings(config)
    
    return config

def _merge_configs(default: dict, user: dict) -> dict:
    '''
    Recursively merge user config into default config.
    
    Args:
        default: Default configuration dictionary
        user: User configuration dictionary
        
    Returns:
        Merged configuration dictionary
    '''
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def _convert_none_strings(config: dict | list | Any) -> dict | list | Any:
    '''
    Recursively convert "none" strings (case-insensitive) to Python None.
    
    Args:
        config: Configuration value (dict, list, or other type)
        
    Returns:
        Configuration with "none" strings converted to None
    '''
    if isinstance(config, dict):
        return {key: _convert_none_strings(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_convert_none_strings(item) for item in config]
    elif isinstance(config, str) and config.lower() == "none":
        return None
    else:
        return config

def _convert_none_to_strings(config: dict | list | Any) -> dict | list | Any:
    '''
    Recursively convert Python None values to "none" strings for TOML serialization.
    
    Args:
        config: Configuration value (dict, list, or other type)
        
    Returns:
        Configuration with None values converted to "none" strings
    '''
    if isinstance(config, dict):
        return {key: _convert_none_to_strings(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_convert_none_to_strings(item) for item in config]
    elif config is None:
        return "none"
    else:
        return config

def save_config(config: Dict[str, Any], path: os.PathLike | str) -> None:
    '''
    Save a configuration dictionary to a TOML file.

    Args:
        config: Configuration dictionary to be written
        path: Output TOML path (default: 'config.toml')

    The write is performed atomically by writing to a temporary file
    in the destination directory and then replacing the target.
    '''
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize using tomli_w where available. For Python 3.11+, tomllib has no writer,
    # so tomli-w remains the recommended writer.
    if toml_writer is None:
        raise RuntimeError(
            "TOML writer not available. Please install 'tomli-w' for writing TOML files."
        )

    # Convert None values to "none" strings for TOML serialization
    config_serializable = _convert_none_to_strings(config)

    # Write atomically
    dir_fd = os.open(str(output_path.parent), os.O_RDONLY)
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, dir=str(output_path.parent), suffix='.tmp') as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(toml_writer.dumps(config_serializable))
            tmp.flush()
            os.fsync(tmp.fileno())
        # Replace target
        os.replace(tmp_path, output_path)
        os.fsync(dir_fd)
    finally:
        try:
            os.close(dir_fd)
        except Exception:
            pass


def get_connectome_subset(
    cell_info_df: pd.DataFrame,
    synapse_counts_df: pd.DataFrame,
    cell_types: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter connectome data to include only specified cell types.
    
    Args:
        cell_info_df: DataFrame with cell information (must have 'type' and 'root_id' columns)
        synapse_counts_df: DataFrame with synapse counts (must have 'pre_root_id' and 'post_root_id' columns)
        column_info_df: DataFrame with column information (currently unused but included for API consistency)
        cell_types: List of cell type names to keep
        
    Returns:
        Tuple of (filtered_cell_info_df, filtered_synapse_counts_df)
    """
    # Filter cell_info_df to only include specified cell types
    filtered_cell_info_df = cell_info_df[cell_info_df['type'].isin(cell_types)].copy().reset_index(drop=True)
    
    # Get the set of root_ids that remain after filtering
    valid_root_ids = set(filtered_cell_info_df['root_id'])
    
    # Filter synapse_counts_df to only include synapses where both pre and post neurons are in the subset
    filtered_synapse_counts_df = synapse_counts_df[
        synapse_counts_df['pre_root_id'].isin(valid_root_ids) &
        synapse_counts_df['post_root_id'].isin(valid_root_ids)
    ].copy().reset_index(drop=True)
    
    return filtered_cell_info_df, filtered_synapse_counts_df
