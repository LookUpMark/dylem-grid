# Models Directory

This directory contains model-related files for the DYLEM-GRID gesture recognition project.

## 📁 Structure

```
models/
├── architectures/              # Model architecture definitions
│   ├── bilstm_model.py        # BiLSTM + Attention model
│   ├── transformer_model.py   # Transformer encoder model
│   └── __init__.py
│
└── checkpoints/                # Saved model weights
    ├── gesture_rnn_model.pth          # Trained BiLSTM model
    ├── gesture_transformer_model.pth  # Trained Transformer model
    └── .gitkeep
```

## 🏗️ Model Architectures

### BiLSTM + Attention (`bilstm_model.py`)

**Architecture:**
- Bidirectional LSTM layers for sequential processing
- Custom attention mechanism for feature importance
- Batch normalization for training stability
- Dropout for regularization
- Fully connected classification head

**Key Features:**
- Sequential processing (forward and backward)
- Attention weights for interpretability
- Effective for time-series data
- Handles variable-length sequences

### Transformer (`transformer_model.py`)

**Architecture:**
- Input projection layer
- Positional encoding for temporal information
- Multi-head self-attention layers
- Feedforward networks
- Global average pooling
- Classification head

**Key Features:**
- Parallel processing for faster training
- Direct long-range dependencies
- Multi-head attention for diverse patterns
- No vanishing gradient issues

## 💾 Model Checkpoints

Saved models include:
- `model_state_dict`: Trained weights
- `label_encoder`: Class label mappings
- `input_size`, `hidden_size`, etc.: Model configuration
- `best_stats`: Training statistics

### Loading Models

```python
import torch
from models.architectures.bilstm_model import GestureRNN

# Load checkpoint
checkpoint = torch.load('models/checkpoints/gesture_rnn_model.pth')

# Initialize model
model = GestureRNN(
    checkpoint['input_size'],
    checkpoint['hidden_size'],
    checkpoint['num_layers'],
    checkpoint['num_classes']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 🔧 Usage

Models are imported and used by training and inference scripts:

```python
from models.architectures.bilstm_model import GestureRNN, train_model, evaluate_model
from models.architectures.transformer_model import GestureTransformer
```

## 📝 Notes

- Model checkpoints are `.gitignore`d by default (except `.gitkeep`)
- Architectures contain model definitions and training utilities
- All models are PyTorch-based
- Checkpoints include full model state for reproducibility
