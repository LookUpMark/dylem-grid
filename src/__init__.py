# DYLEM-GRID Source Package

"""
Modern, modular implementation using PyTorch Lightning.

Modules:
    - data: Data loading and preprocessing with LightningDataModule
    - models: Lightning-based model implementations
    - training: Cross-validation and callbacks
    - optimization: Optuna hyperparameter optimization
    - ablation: Ablation study framework
    - hub: Hugging Face Hub integration
"""

from src.data import GestureDataModule
from src.models import BiLSTMModule, TransformerModule
from src.training import CrossValidator
from src.optimization import OptunaObjective
from src.hub import get_model, save_to_hub, load_from_hub

__all__ = [
    "GestureDataModule",
    "BiLSTMModule",
    "TransformerModule",
    "CrossValidator",
    "OptunaObjective",
    "get_model",
    "save_to_hub",
    "load_from_hub",
]

__version__ = "2.0.0"

