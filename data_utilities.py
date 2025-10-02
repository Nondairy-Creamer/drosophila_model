"""
Simple TOML configuration loader for the Drosophila neural model.
"""

from pathlib import Path
import tomllib

def load_config() -> dict:
    """
    Load configuration by merging default config with user config.
    
    Returns:
        Dictionary containing the merged configuration
    """
    # Load default config
    default_path = Path("config_default.toml")
    if not default_path.exists():
        raise FileNotFoundError(f"Default configuration file not found: {default_path}")
    
    with open(default_path, "rb") as f:
        config = tomllib.load(f)
    
    # Check for user config and merge if it exists
    user_path = Path("config.toml")
    if user_path.exists():
        with open(user_path, "rb") as f:
            user_config = tomllib.load(f)
        
        # Recursively merge user config into default config
        config = _merge_configs(config, user_config)
    
    return config

def _merge_configs(default: dict, user: dict) -> dict:
    """
    Recursively merge user config into default config.
    
    Args:
        default: Default configuration dictionary
        user: User configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

