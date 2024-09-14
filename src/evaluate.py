# src/evaluate.py

"""
Script to evaluate NLI models based on the provided configuration.
"""

import argparse
import logging
from utils.config import load_config
from models.model_factory import get_model
from data.data_loader import DataLoader
from evaluation.evaluator import Evaluator
from utils.logger import setup_logging


def parse_arguments():
    parser = argparse.ArgumentParser(description='NLI Model Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to the evaluation configuration file')
    return parser.parse_args()


def main():
    args = parse_arguments()

    config = load_config(args.config)
    setup_logging(config)
    
    data_loader = DataLoader(config)
    test_dataset = data_loader.load_data(train=False, val=False, test=True)
    
    model = get_model(config)
    model.load_model() 
    
    evaluator = Evaluator(model, config)
    evaluator.evaluate(test_dataset)


if __name__ == '__main__':
    main()
