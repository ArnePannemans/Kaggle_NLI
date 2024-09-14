# src/train.py

"""
Script to train NLI models based on the provided configuration.
"""

import argparse
import logging
from utils.config import load_config
from models.model_factory import get_model
from data.data_loader import DataLoader
from training.trainer import Trainer
from utils.logger import setup_logging


def parse_arguments():
    parser = argparse.ArgumentParser(description='NLI Model Training')
    parser.add_argument('--config', type=str, required=True, help='Path to the training configuration file')
    return parser.parse_args()


def main():
    args = parse_arguments()

    config = load_config(args.config)
    setup_logging(config)
    
    data_loader = DataLoader(config)
    train_dataset, val_dataset = data_loader.load_data(train=True, val=True, test=False)
    
    model = get_model(config)
    model.load_model() 
    
    trainer = Trainer(model, config)
    trainer.train(train_dataset, val_dataset)


if __name__ == '__main__':
    main()
