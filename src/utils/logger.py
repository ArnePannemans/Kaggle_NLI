# src/utils/logger.py

"""
Utility module for setting up logging.
"""

import logging
import os
from typing import Dict, Any


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    logging_level = config.get('logging', {}).get('logging_level', 'INFO').upper()
    log_format = "%(levelname)s - %(message)s"

    logging.basicConfig(level=logging_level, format=log_format)

    # Create logs directory if it doesn't exist
    log_dir = config.get('logging', {}).get('output_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler
    log_file = os.path.join(log_dir, 'application.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Add the handler to the root logger
    logging.getLogger().addHandler(file_handler)
