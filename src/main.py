# src/main.py

import argparse
from utils.config import load_config
from models import get_model
from data.data_loader import DataLoader
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.logger import setup_logging

def main():
    parser = argparse.ArgumentParser(description='NLI Model Training and Evaluation')
    parser.add_argument('--config', type=str, default='configs/phi3_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    data_loader = DataLoader(config)
    train_data, val_data, test_data = data_loader.load_data()

    model = get_model(config)
    model.load_model()

    if config['training']['do_train']:
        trainer = Trainer(model, config)
        trainer.train(train_data, val_data)

    if config['evaluation']['do_eval']:
        evaluator = Evaluator(model, config)
        evaluator.evaluate(test_data)

if __name__ == '__main__':
    main()
