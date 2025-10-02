"""
Simple TOML configuration loader for the Drosophila neural model.
"""

from pathlib import Path
import tomllib

def load_config(config_path: str = "config.toml") -> dict:
    """
    Load configuration from TOML file and return as dictionary.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    return config

