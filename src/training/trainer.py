# src/training/trainer.py

"""
Module for managing the training process of models.
"""

import logging
from typing import Any, Dict


class Trainer:
    """
    Trainer class for managing the training process.
    """

    def __init__(self, model: Any, config: Dict[str, Any]):
        """
        Initialize the Trainer.

        Args:
            model (Any): The model to train.
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train(self, train_data: Any, val_data: Any) -> None:
        """
        Train the model using the provided data.

        Args:
            train_data (Any): Training dataset.
            val_data (Any): Validation dataset.
        """
        self.logger.info("Starting training process...")
        self.model.train(train_data, val_data)
        self.logger.info("Training completed.")
