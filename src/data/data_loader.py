# src/data/data_loader.py

"""
Module for loading and preprocessing data for the NLI task.
"""

import os
import logging
import pandas as pd
from datasets import Dataset
from typing import Tuple, Dict, Any, Optional, Union


class DataLoader:
    """
    DataLoader class for loading and preprocessing datasets.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataLoader.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self, train=False, val=False, test=False) -> Union[Tuple[Dataset, ...], Dataset]:
        """
        Load and preprocess the specified datasets.

        Args:
            train (bool): Whether to load the training data.
            val (bool): Whether to load the validation data.
            test (bool): Whether to load the test data.

        Returns:
            Union[Tuple[Dataset, ...], Dataset]: Loaded datasets.
        """
        datasets = []
        if train:
            train_dataset = self._load_dataset('train_file')
            datasets.append(train_dataset)
        if val:
            val_dataset = self._load_dataset('val_file')
            datasets.append(val_dataset)
        if test:
            test_dataset = self._load_dataset('test_file')
            datasets.append(test_dataset)

        if len(datasets) == 1:
            return datasets[0]
        return tuple(datasets)

    def _load_dataset(self, split_name: str) -> Dataset:
        """
        Helper function to load a dataset split.

        Args:
            split_name (str): Key for the dataset split in the config.

        Returns:
            Dataset: Loaded and preprocessed dataset.
        """
        file_path = self.config['data'][split_name]
        if not os.path.exists(file_path):
            message = f"Data file {file_path} not found."
            self.logger.error(message)
            raise FileNotFoundError(message)

        df = pd.read_csv(file_path)
        df['label'] = df['label'].astype(int)

        dataset = Dataset.from_pandas(df)
        return dataset
