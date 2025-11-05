# Scripts Directory

This directory contains all executable scripts for training, inference, and optimization.

## 📁 Structure

```
scripts/
├── train/                  # Training scripts
├── inference/              # Inference scripts  
└── optimization/           # Hyperparameter optimization scripts
```

## 🚀 Usage

All scripts should be run from the **root directory** of the project to ensure correct path resolution:

```bash
# From project root (dylem-grid/)
python scripts/train/train_bilstm.py
python scripts/train/train_transformer.py
python scripts/inference/inference_bilstm.py
python scripts/inference/inference_transformer.py
python scripts/optimization/hyperparameter_optimization_bilstm.py
python scripts/optimization/hyperparameter_optimization_transformer.py
```

## 📝 Script Details

### Training Scripts (`train/`)

- **`train_bilstm.py`**: Train BiLSTM model with optimized hyperparameters
- **`train_transformer.py`**: Train Transformer model with optimized hyperparameters

**Inputs:**
- Dataset from `data/DYLEM-GRID/`

**Outputs:**
- Trained models → `models/checkpoints/`
- Visualization plots → `plots/`

### Inference Scripts (`inference/`)

- **`inference_bilstm.py`**: Load trained BiLSTM model and make predictions
- **`inference_transformer.py`**: Load trained Transformer model and make predictions

**Inputs:**
- Trained models from `models/checkpoints/`
- New data for prediction

**Outputs:**
- Predictions and probabilities

### Optimization Scripts (`optimization/`)

- **`hyperparameter_optimization_bilstm.py`**: Optimize BiLSTM hyperparameters using Optuna
- **`hyperparameter_optimization_transformer.py`**: Optimize Transformer hyperparameters using Optuna

**Inputs:**
- Dataset from `data/DYLEM-GRID/`

**Outputs:**
- Optimization results (JSON files in project root)

## 🔧 Path Configuration

All scripts use **relative paths** from the script location:
- Dataset: `../../data/DYLEM-GRID/`
- Models: `../../models/checkpoints/`
- Plots: `../../plots/`
- Utils: Import from `utils.*`
- Model architectures: Import from `models.architectures.*`

## 📦 Dependencies

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 💡 Tips

1. **Always run from project root** to avoid path issues
2. Check that the dataset is in `data/DYLEM-GRID/` before training
3. Run optimization scripts first to find best hyperparameters
4. Training outputs are saved automatically in designated folders
5. Use notebooks in root directory for interactive exploration
