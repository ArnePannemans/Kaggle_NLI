# src/models/model_factory.py

from .phi_model import PhiModel
# from .llama_model import LLamaModel  

def get_model(config):
    """
    Factory function to instantiate models based on the configuration.
    """
    model_type = config['model'].get('type', '').lower()
    if model_type == 'phi':
        return PhiModel(config)
    # elif model_type == 'llama':
    #     return LLamaModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
