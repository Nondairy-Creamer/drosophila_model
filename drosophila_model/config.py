"""TOML configuration loader/saver for the Drosophila neural model.

Top-level so it's obvious where to import from:

    from drosophila_model.config import load_config, save_config

Three loaders cover every use:

* :func:`load_config` -- the live config for this machine: the packaged
  defaults (``config_default.toml`` + ``config_default_data.toml``) overlaid
  with the user's gitignored ``config.toml`` / ``config_data.toml``. Strictly
  validated so typos fail loudly.
* :func:`load_run_config` -- the config a trained model was saved with
  (``models/<run>/config.toml``). Its saved data tables are kept as-is; only
  ``[data].data_folder`` is taken from this machine's ``config.toml``.
* :func:`get_data_folder` -- the one way to resolve ``[data].data_folder``
  (default ``../drosophila_model_data``, next to the repo; override it in
  ``config.toml``).
"""

from __future__ import annotations

import os
import tempfile
import tomllib
import warnings
from importlib import resources
from pathlib import Path
from typing import Any, Dict

import tomli_w


# Paths under which a user config may introduce keys that aren't in
# config_default.toml. Used by :func:`_validate_user_keys`.
#
#   ('model', 'regions') -- region names are user-defined (the default only
#       ships ``optic_lobe``; configs that introduce another area such as
#       ``antenna_lobe`` add their own sub-table). Region names are checked
#       against ``regions.REGION_REGISTRY`` and the keys *inside* each region
#       sub-table by the corresponding ``RegionView._load_extras`` (see
#       model_definition/regions.py), so we stop validating at this depth.
#
#   ('run_info',) -- added programmatically by ``run_model.py`` to record
#       per-run provenance (seed, device, run_id, eye geometry) and persisted
#       into saved configs. Not part of the user-editable schema.
#
# ``[data]`` is validated too: ``data_folder`` against config_default.toml
# and the stimulus/dataset tables (``[data.shared]``, ``[data.<type>]``)
# against config_default_data.toml.
_EXTENSIBLE_PATHS: frozenset[tuple[str, ...]] = frozenset({
    ('model', 'regions'),
    ('run_info',),
})


def load_config(config_path: str = 'config.toml',
                data_config_path: str = 'config_data.toml') -> dict:
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

    ``[model].device`` is resolved with :func:`resolve_device`, so a CUDA
    device falls back to CPU (with a warning) on machines without CUDA.

    The stimulus/dataset generation parameters (default
    ``config_default_data.toml`` + optional ``data_config_path`` overlay) are
    loaded and merged into ``config['data']`` alongside ``data_folder``, so
    callers read ``config['data']['shared'][...]`` /
    ``config['data'][<stimulus_type>][...]`` rather than a separate loader.
    ``[data.<table>]`` tables in ``config_path`` are applied too, under the
    ``data_config_path`` overlay (which wins where both set a key).

    Read ``[data].data_folder`` with :func:`get_data_folder`, which raises a
    clear error when the folder does not exist.

    Returns:
        Dictionary containing the merged configuration
    '''
    config = _load_default('config_default.toml')
    data_defaults = _load_default('config_default_data.toml')

    user_path = Path(config_path)
    user_data_tables = {}
    if user_path.exists():
        user_config = _read_toml(user_path)
        # [data] holds data_folder (schema: config_default.toml) plus
        # optional stimulus/dataset tables (schema: config_default_data.toml).
        user_data = user_config.get('data', {})
        user_data_tables = {k: v for k, v in user_data.items() if isinstance(v, dict)}
        _validate_user_keys(user_data_tables, data_defaults, source=str(user_path),
                            default_name='config_default_data.toml', path=('data',))
        if 'data' in user_config:
            user_config['data'] = {k: v for k, v in user_data.items()
                                   if k not in user_data_tables}
        _validate_user_keys(user_config, config, source=str(user_path),
                            default_name='config_default.toml')
        config = _merge_configs(config, user_config)
    elif config_path != 'config.toml':
        warnings.warn(
            f'User config file not found: {user_path.resolve()!s}. '
            'Using only config_default.toml (no overlay). '
            'Did you mean config.toml?',
            stacklevel=2,
        )

    # Fold the stimulus/dataset generation config into config['data'] next to
    # data_folder, so there is a single config dict to pass around.
    data_config = _merge_configs(data_defaults, user_data_tables)
    data_config = _merge_configs(data_config,
                                 _load_data_overlay(data_config_path, data_defaults))
    config['data'] = _merge_configs(config['data'], data_config)

    config = _convert_none_strings(config)
    config['model']['device'] = resolve_device(config['model']['device'])
    return config


def load_run_config(path: os.PathLike | str,
                    local_config_path: str = 'config.toml') -> dict:
    '''
    Load the config a trained model was saved with.

    ``path`` is a model folder (``models/<run>``) or its ``config.toml``. The
    saved file is merged over the packaged defaults. The saved ``[data]``
    generation tables are kept as they were at training time (they are *not*
    replaced by this machine's ``config_data.toml``).

    ``[data].data_folder`` is a property of the machine, not the run, so it is
    taken from the local ``config.toml`` when that file sets it.
    ``[model].device`` is resolved with :func:`resolve_device`.
    '''
    path = Path(path)
    if path.is_dir():
        path = path / 'config.toml'
    config = _load_default('config_default.toml')
    config['data'] = _merge_configs(config.get('data', {}),
                                    _load_default('config_default_data.toml'))
    config = _merge_configs(config, _read_toml(path))

    local_path = Path(local_config_path)
    if local_path.exists():
        local_folder = _read_toml(local_path).get('data', {}).get('data_folder')
        if local_folder:
            config['data']['data_folder'] = local_folder

    config = _convert_none_strings(config)
    config['model']['device'] = resolve_device(config['model']['device'])
    return config


def get_data_folder(config: dict) -> Path:
    '''
    Return ``[data].data_folder`` as a Path, or raise a clear error.

    The default (``../drosophila_model_data``) is relative to the working
    directory, so run scripts from the repo root or set an absolute
    ``data_folder`` in the ``[data]`` table of your (gitignored)
    ``config.toml``.
    '''
    folder = (config.get('data') or {}).get('data_folder')
    if not folder:
        raise ValueError(
            "[data].data_folder is empty. Set it in the [data] table of your "
            "config.toml."
        )
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"[data].data_folder does not exist: {path.resolve()!s}. Put the "
            "data bundle there or set [data].data_folder in config.toml "
            f"(read from the current directory, {Path.cwd()}; run scripts "
            "from the repo root)."
        )
    return path


def resolve_device(device: str) -> str:
    '''
    Return ``device``, falling back to ``'cpu'`` when CUDA is unavailable.

    Lets the same config.toml (e.g. ``device = "cuda:0"``) be used on a GPU
    server and on a laptop.
    '''
    device = str(device)
    if device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            warnings.warn(
                f"Config device '{device}' requested but CUDA is not available; "
                "using 'cpu'.",
                stacklevel=3,
            )
            return 'cpu'
    return device


def _load_data_overlay(config_path: str, data_defaults: dict) -> dict:
    '''
    Read and validate the user's stimulus/dataset generation overlay.

    Same two-file pattern as :func:`load_config`: the default ships inside the
    package as ``drosophila_model/config_default_data.toml`` (``data_defaults``),
    and the optional user overlay (``config_path``, default
    ``config_data.toml`` at the repo root) is gitignored and resolved relative
    to CWD.

    The schema is a ``[shared]`` table of parameters common to every dataset
    plus one ``[<stimulus_type>]`` table per generator for its specific
    parameters (see ``config_default_data.toml``). Every key in the overlay
    must already exist in the default, or ``ValueError`` is raised.

    Internal helper for :func:`load_config`, which merges the result into
    ``config['data']``. Returns ``{}`` when the overlay file does not exist.
    '''
    user_path = Path(config_path)
    if not user_path.exists():
        if config_path != 'config_data.toml':
            warnings.warn(
                f'Data config file not found: {user_path.resolve()!s}. '
                'Using only config_default_data.toml (no overlay). '
                'Did you mean config_data.toml?',
                stacklevel=3,
            )
        return {}
    user_config = _read_toml(user_path)
    _validate_user_keys(user_config, data_defaults, source=str(user_path),
                        default_name='config_default_data.toml')
    return user_config


def save_config(config: Dict[str, Any], path: os.PathLike | str) -> None:
    '''
    Save a configuration dictionary to a TOML file.

    Args:
        config: Configuration dictionary to be written
        path: Output TOML path

    The write is atomic: the TOML is written to a temporary file in the
    destination directory and then moved over the target.
    '''
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = tomli_w.dumps(_convert_none_to_strings(config))

    with tempfile.NamedTemporaryFile('w', delete=False, dir=str(output_path.parent),
                                     suffix='.tmp', encoding='utf-8') as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_default(name: str) -> dict:
    '''Read one of the TOML defaults shipped inside the package.'''
    with resources.files('drosophila_model').joinpath(name).open('rb') as f:
        return tomllib.load(f)


def _read_toml(path: Path) -> dict:
    with open(path, 'rb') as f:
        return tomllib.load(f)


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
                        default_name: str,
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
            f"{default_name}: {joined}. Add them to "
            f"drosophila_model/{default_name} first if intentional."
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
