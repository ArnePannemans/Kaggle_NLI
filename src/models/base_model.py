# src/models/base_model.py

"""
Abstract base class defining the interface for all model classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModel(ABC):
    """
    Abstract BaseModel class defining the interface for all models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BaseModel.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and tokenizer.
        """
        pass

    @abstractmethod
    def train(self, train_data: Any, val_data: Any) -> None:
        """
        Train the model.

        Args:
            train_data (Any): Training dataset.
            val_data (Any): Validation dataset.
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: Any) -> Dict[str, Any]:
        """
        Evaluate the model.

        Args:
            test_data (Any): Test dataset.

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        pass

    @abstractmethod
    def predict(self, inputs: List[Dict[str, Any]]) -> List[int]:
        """
        Make predictions using the model.

        Args:
            inputs (List[Dict[str, Any]]): Input data.

        Returns:
            List[int]: List of predictions.
        """
        pass
