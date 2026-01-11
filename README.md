# DYLEM-GRID: Dynamic Hand Gesture Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange.svg)](https://huggingface.co/datasets/LookUpMark/DYLEM-GRID)

Deep learning for dynamic gesture recognition using BiLSTM and Transformer models.

## Quick Start

```bash
git clone https://github.com/LookUpMark/dylem-grid.git
cd dylem-grid
pip install -r requirements.txt
```

## Notebooks

| Notebook | Purpose | Time |
|----------|---------|------|
| `01_optimization.ipynb` | Optuna hyperparameter search | ~1 hour |
| `02_training.ipynb` | Train models with optimized params | ~30 min |
| `03_inference.ipynb` | Evaluate with confusion matrices | ~2 min |
| `04_ablation.ipynb` | Component analysis | ~2 hours |

**Workflow**: `01_optimization` → `02_training` → `03_inference`

## Models

| Model | Architecture | Accuracy |
|-------|--------------|----------|
| BiLSTM | Bidirectional LSTM + Attention | 97.25% ± 0.94% |
| Transformer | Encoder-only + Self-Attention | 94.75% ± 0.94% |

## Structure

```
dylem-grid/
├── src/                    # Core package
│   ├── data/               # GestureDataModule
│   ├── models/             # BiLSTM, Transformer
│   ├── training/           # CrossValidator
│   ├── optimization/       # Optuna
│   ├── ablation/           # Ablation studies
│   ├── hub.py              # HuggingFace Hub
│   └── visualization.py    # Plot helpers
├── notebooks/              # Jupyter notebooks
└── results/                # Output (params, metrics)
```

## Python API

```python
from src import GestureDataModule, BiLSTMModule, CrossValidator

dm = GestureDataModule()  # Auto-downloads from HuggingFace
dm.setup()

cv = CrossValidator(BiLSTMModule, dm, n_folds=5)
results = cv.run({"hidden_size": 64})
```

## Dataset

[DYLEM-GRID](https://huggingface.co/datasets/LookUpMark/DYLEM-GRID): 400 recordings of 4 gestures. Auto-downloads on first use.

## License

MIT
