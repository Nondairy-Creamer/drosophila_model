"""TOML configuration loader/saver for the Drosophila neural model.

Top-level so it's obvious where to import from:

    from drosophila_model.config import load_config, get_data_config, save_config
"""

from __future__ import annotations

import os
import tempfile
import warnings
from importlib import resources
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


# Paths under which a user config may introduce keys that aren't in
# config_default.toml. Used by :func:`_validate_user_keys`.
#
#   ('model', 'regions') -- region names are user-defined (the default only
#       ships ``optic_lobe``; configs that introduce another area such as
#       ``antenna_lobe`` add their own sub-table). The keys *inside* each
#       region sub-table are validated at runtime by the corresponding
#       ``RegionView._load_extras`` (see model_definition/regions.py), which
#       raises on unknown region opts. So we stop validating at this depth.
#
#   ('run_info',) -- added programmatically by ``run_model.py`` to record
#       per-run provenance (seed, device, run_id) and persisted into saved
#       configs. Not part of the user-editable schema; allow any contents.
_EXTENSIBLE_PATHS: frozenset[tuple[str, ...]] = frozenset({
    ('model', 'regions'),
    ('run_info',),
})


def load_config(config_path: str = 'config.toml') -> dict:
    '''
    Load configuration by merging default config with user config.

    The default lives inside the installed package as
    ``drosophila_model/config_default.toml`` and is located via
    ``importlib.resources``, so it is findable regardless of CWD. The user
    overlay (``config_path``, default ``config.toml``) is resolved relative to
    CWD (or absolute) and is optional.

    Every key in the user overlay must already exist in the default, except
    for the extensible paths declared in :data:`_EXTENSIBLE_PATHS`. Unknown
    keys raise ``ValueError`` to catch silent typos / dead config.

    The GPU/CPU device for the model is ``config['model']['device']``. A
    ``[run_info]`` table (if present) is only provenance from saved runs and is
    not used to set the model device.

    Returns:
        Dictionary containing the merged configuration
    '''
    default_resource = resources.files('drosophila_model').joinpath('config_default.toml')
    with default_resource.open('rb') as f:
        config = tomllib.load(f)

    user_path = Path(config_path)
    if user_path.exists():
        with open(user_path, 'rb') as f:
            user_config = tomllib.load(f)
        _validate_user_keys(user_config, config, source=str(user_path))
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


def get_data_config(config_path: str = 'config_data.toml') -> dict:
    '''
    Load the stimulus/dataset generation config (default + user overlay).

    Same two-file pattern as :func:`load_config`: the default ships inside the
    package as ``drosophila_model/config_default_data.toml`` (located via
    ``importlib.resources``), and the optional user overlay (``config_path``,
    default ``config_data.toml`` at the repo root) is gitignored and resolved
    relative to CWD.

    The schema is a ``[shared]`` table of parameters common to every dataset
    plus one ``[<stimulus_type>]`` table per generator for its specific
    parameters (see ``config_default_data.toml``). Every key in the overlay
    must already exist in the default, or ``ValueError`` is raised.

    Returns:
        Merged data-config dict (``"none"`` strings converted to ``None``).
    '''
    default_resource = resources.files('drosophila_model').joinpath('config_default_data.toml')
    with default_resource.open('rb') as f:
        config = tomllib.load(f)

    user_path = Path(config_path)
    if user_path.exists():
        with open(user_path, 'rb') as f:
            user_config = tomllib.load(f)
        _validate_user_keys(user_config, config, source=str(user_path))
        config = _merge_configs(config, user_config)
    elif config_path != 'config_data.toml':
        warnings.warn(
            f'Data config file not found: {user_path.resolve()!s}. '
            'Using only config_default_data.toml (no overlay). '
            'Did you mean config_data.toml?',
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


def _validate_user_keys(user: dict, default: dict, *, source: str,
                        path: tuple[str, ...] = ()) -> None:
    '''Strict-validate user config keys against the default schema.

    Every key in ``user`` (recursively) must exist at the same path in
    ``default``. Tables under :data:`_EXTENSIBLE_PATHS` are not recursed into;
    their contents are validated elsewhere (e.g. region opts are validated by
    ``RegionView._load_extras``).

    Raises:
        ValueError: with all unknown key paths joined into one message so the
            user sees every problem at once instead of one-by-one.
    '''
    unknown: list[str] = []
    _collect_unknown_keys(user, default, path, unknown)
    if unknown:
        joined = ', '.join(unknown)
        raise ValueError(
            f"Unknown config key(s) in {source} not present in "
            f"config_default.toml: {joined}. Add them to "
            f"drosophila_model/config_default.toml first if intentional."
        )


def _collect_unknown_keys(user: dict, default: dict, path: tuple[str, ...],
                          unknown: list[str]) -> None:
    '''Walker for :func:`_validate_user_keys`; appends dotted unknown paths.

    A path in :data:`_EXTENSIBLE_PATHS` is treated as a closed leaf for
    validation purposes: it (and its descendants) are accepted without
    further checks. This lets the user introduce new region tables under
    ``[model.regions.<name>]`` (per-region inner keys are validated by
    :class:`drosophila_model.model_definition.regions.RegionView`) and lets
    saved-run configs round-trip a programmatic ``[run_info]`` table.
    '''
    if path in _EXTENSIBLE_PATHS:
        return
    for key, user_val in user.items():
        full = path + (key,)
        if full in _EXTENSIBLE_PATHS:
            continue
        if key not in default:
            unknown.append('.'.join(full))
            continue
        default_val = default[key]
        if isinstance(user_val, dict) and isinstance(default_val, dict):
            _collect_unknown_keys(user_val, default_val, full, unknown)


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
