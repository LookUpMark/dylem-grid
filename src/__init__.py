"""
Source module for DYLEM-GRID gesture recognition project.

This package contains:
- data_processing: Data loading, cleaning, and preprocessing functions
- bilstm_model: GestureRNN model definition and training functions
- transformer_model: GestureTransformer model definition and training functions
- plots: Visualization and plotting functions
"""

# Make modules easily importable
from . import data_processing
from . import bilstm_model
from . import transformer_model
from . import plots

__all__ = ['data_processing', 'bilstm_model', 'transformer_model', 'plots']
