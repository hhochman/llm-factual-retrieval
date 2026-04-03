from typing import Tuple, Any

import torch
from nnsight import LanguageModel

def load_model(model_name: str) -> Any:
    """
    Loads the nnsight LanguageModel.
    Example model_names: 'meta-llama/Llama-3.1-8B', 'Qwen/Qwen3-8B'
    """
    print(f"Loading {model_name}...")

    model = LanguageModel(model_name, device_map="auto")
    return model

def get_model_config(model_name: str) -> dict:
    """Returns architecture details (number of layers, attention heads) for the specified model."""
    pass