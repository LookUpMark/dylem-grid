# DYLEM-GRID: Dynamic Hand Gesture Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-orange.svg)](https://huggingface.co/datasets/LookUpMark/DYLEM-GRID)

Deep learning for dynamic gesture recognition using BiLSTM and Transformer models.

## Quick Start

```bash
git clone https://github.com/LookUpMark/dylem-grid.git
cd dylem-grid
pip install -r requirements.txt
```

Open notebooks in `notebooks/` to train, optimize, or run inference.

## Models

| Model | Architecture | Accuracy |
|-------|--------------|----------|
| **BiLSTM** | Bidirectional LSTM + Attention | 100% |
| **Transformer** | Encoder-only + Self-Attention | 100% |

## Structure

```
dylem-grid/
├── src/                    # Core package
│   ├── data/               # GestureDataModule
│   ├── models/             # BiLSTM, Transformer
│   ├── training/           # CrossValidator
│   ├── optimization/       # Optuna integration
│   ├── ablation/           # Ablation studies
│   └── hub.py              # HuggingFace Hub
├── notebooks/              # Jupyter notebooks
│   ├── train.ipynb         # Training & CV
│   ├── optimize.ipynb      # Hyperparameter search
│   ├── inference.ipynb     # Evaluation
│   └── ablation.ipynb      # Ablation studies
└── models/checkpoints/     # Saved models
```

## Usage

### Python API

```python
from src import GestureDataModule, BiLSTMModule, CrossValidator, get_model

# Data (auto-downloads from HuggingFace)
dm = GestureDataModule()
dm.setup()

# Cross-validation
cv = CrossValidator(BiLSTMModule, dm, n_folds=5)
results = cv.run({"hidden_size": 64})

# Inference
model = get_model("bilstm")  # Local or Hub
```

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `train.ipynb` | Train models with optional cross-validation |
| `optimize.ipynb` | Optuna hyperparameter optimization |
| `inference.ipynb` | Run inference with visualizations |
| `ablation.ipynb` | Systematic ablation studies |

## Dataset

[DYLEM-GRID](https://huggingface.co/datasets/LookUpMark/DYLEM-GRID) contains 400 recordings of 4 gestures:
- Air Quotes, Finger Wagging, Waving, Zoom

Auto-downloads on first use.

## Features

- **PyTorch Lightning** - Modular training
- **K-Fold CV** - Robust evaluation
- **Optuna** - Hyperparameter optimization
- **HuggingFace Hub** - Dataset & model hosting

## License

MIT
