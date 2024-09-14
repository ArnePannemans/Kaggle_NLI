# src/evaluation/evaluator.py

"""
Module for evaluating models and saving evaluation results.
"""

import os
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Any, Dict


class Evaluator:
    """
    Evaluator class for evaluating models and saving results.
    """

    def __init__(self, model: Any, config: Dict[str, Any]):
        """
        Initialize the Evaluator.

        Args:
            model (Any): The model to evaluate.
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    def evaluate(self, test_data: Any) -> None:
        """
        Evaluate the model and save the results.

        Args:
            test_data (Any): Test dataset.
        """
        self.logger.info("Starting evaluation...")

        # Get evaluation results from the model
        results = self.model.evaluate(test_data)

        # Log results
        self.logger.info(f"Accuracy: {results['accuracy'] * 100:.2f}%")
        self.logger.info(f"F1 Score: {results['f1_score']:.4f}")
        self.logger.info(f"Total Time: {results['total_time']:.2f} seconds")

        # Save results
        self.save_experiment_results(results)

        self.logger.info("Evaluation completed.")

    def save_experiment_results(self, results: Dict[str, Any]) -> None:
        """
        Save the experiment results to a JSON file and log to CSV.

        Args:
            results (Dict[str, Any]): Evaluation results.
        """
        output_dir = self.config.get('logging', {}).get('output_dir', 'results/experiment_logs')
        os.makedirs(output_dir, exist_ok=True)

        model_name = self.config['model']['name']
        model_path = self.config['model'].get('model_path', 'base')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{model_name}_evaluation_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)

        experiment_data = {
            "timestamp": timestamp,
            "model_name": model_name,
            "model_path": model_path,
            "config": self.config,
            "results": results
        }

        with open(json_filepath, 'w') as json_file:
            json.dump(experiment_data, json_file, indent=4)