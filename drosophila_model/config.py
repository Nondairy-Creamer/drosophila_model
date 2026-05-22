"""TOML configuration loader/saver for the Drosophila neural model.

Top-level so it's obvious where to import from:

    from drosophila_model.config import load_config, save_config
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    # Python <3.11 needs a writer; on 3.11+ tomllib reads but doesn't write,
    # so tomli_w is the writer either way.
    import tomli_w as toml_writer
except ImportError:  # pragma: no cover - tomli_w present per requirements for <3.11
    toml_writer = None


def load_config(config_path: str = 'config.toml') -> dict:
    '''
    Load configuration by merging default config with user config.

    The GPU/CPU device for the model is ``config['model']['device']``. A
    ``[run_info]`` table (if present) is only provenance from saved runs and is
    not used to set the model device.

    Returns:
        Dictionary containing the merged configuration
    '''
    default_path = Path('config_default.toml')
    if not default_path.exists():
        raise FileNotFoundError(f'Default configuration file not found: {default_path}')

    with open(default_path, 'rb') as f:
        config = tomllib.load(f)

    user_path = Path(config_path)
    if user_path.exists():
        with open(user_path, 'rb') as f:
            user_config = tomllib.load(f)
        config = _merge_configs(config, user_config)
    elif config_path != 'config.toml':
        warnings.warn(
            f'User config file not found: {user_path.resolve()!s}. '
            'Using only config_default.toml (no overlay). '
            'Did you mean config.toml?',
            stacklevel=2,
        )

    config = _convert_none_strings(config)
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

    if toml_writer is None:
        raise RuntimeError(
            "TOML writer not available. Please install 'tomli-w' for writing TOML files."
        )

    config_serializable = _convert_none_to_strings(config)

    dir_fd = os.open(str(output_path.parent), os.O_RDONLY)
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, dir=str(output_path.parent), suffix='.tmp') as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(toml_writer.dumps(config_serializable))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, output_path)
        os.fsync(dir_fd)
    finally:
        try:
            os.close(dir_fd)
        except Exception:
            pass


def _merge_configs(default: dict, user: dict) -> dict:
    '''Recursively merge user config into default config.'''
    result = default.copy()
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def _convert_none_strings(config: dict | list | Any) -> dict | list | Any:
    '''Recursively convert "none" strings (case-insensitive) to Python None.'''
    if isinstance(config, dict):
        return {key: _convert_none_strings(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_convert_none_strings(item) for item in config]
    elif isinstance(config, str) and config.lower() == "none":
        return None
    else:
        return config


def _convert_none_to_strings(config: dict | list | Any) -> dict | list | Any:
    '''Recursively convert Python None values to "none" strings for TOML serialization.'''
    if isinstance(config, dict):
        return {key: _convert_none_to_strings(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_convert_none_to_strings(item) for item in config]
    elif config is None:
        return "none"
    else:
        return config
