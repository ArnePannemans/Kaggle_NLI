# src/utils/config.py

"""
Utility module for loading and managing configuration files.
"""

import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the configuration from the specified file and merge with base configuration.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open('configs/common_config.yaml', 'r') as base_file:
        base_config = yaml.safe_load(base_file)
    with open(config_path, 'r') as file:
        specific_config = yaml.safe_load(file)
    
    # Merge configurations (specific config overrides base config)
    config = {**base_config, **specific_config}

    return config
