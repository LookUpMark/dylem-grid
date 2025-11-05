# Utils Directory

Utility modules for data processing, visualization, and training.

## 📁 Structure

```
utils/
├── data_processing.py      # Data loading and preprocessing
├── plots.py                # Visualization utilities
├── training_utils.py       # Training and evaluation functions
└── __init__.py
```

## 📦 Modules

### `data_processing.py`

Functions for loading and preprocessing the DYLEM-GRID dataset.

**Key Functions:**
- `data_loader(data_path, data_type)` - Load CSV files from dataset
- `data_preprocess(data, labels)` - Clean and normalize data
- `apply_pca(data, labels, variance_threshold)` - Dimensionality reduction
- `prepare_data(data, labels)` - Convert to PyTorch tensors

**Pipeline:**
1. Load raw CSV files
2. Handle missing values (backward fill)
3. Remove duplicate columns
4. Remove low-variance features
5. Remove outliers per class
6. Normalize with MinMaxScaler
7. Apply PCA (95% variance)
8. Pad sequences and convert to tensors

### `plots.py`

Comprehensive visualization utilities for model evaluation.

**Key Functions:**
- `plot_training_history()` - Training/validation curves
- `plot_confusion_matrix()` - Detailed confusion matrix
- `plot_pca_boxplot()` - PCA component distribution
- `plot_comprehensive_metrics()` - Full performance dashboard
- `plot_model_weights_heatmap()` - Weight visualization
- `plot_lstm_weights_heatmap()` - LSTM-specific weights

**Output:** High-quality PNG files with detailed metrics

### `training_utils.py`

Shared training and evaluation functions for both models.

**Key Functions:**
- `train_model()` - Training loop with early stopping
- `evaluate_model()` - Model evaluation on dataset

**Features:**
- Early stopping with patience
- Progress tracking with tqdm
- Best model tracking
- Comprehensive metrics logging

## 🔧 Usage

Import utilities in your scripts:

```python
from utils.data_processing import data_loader, data_preprocess, apply_pca
from utils.plots import plot_training_history, plot_confusion_matrix
from utils.training_utils import train_model, evaluate_model
```

## 📊 Data Flow

```
Raw CSV files 
  → data_loader() 
  → data_preprocess() 
  → apply_pca() 
  → prepare_data() 
  → PyTorch Tensors
```

## 🎨 Visualization Features

All plots include:
- Professional styling with seaborn
- Color-coded metrics
- Statistical summaries
- Multiple subplots for comprehensive view
- High DPI (300) for publication quality
- Detailed annotations and legends

## 💡 Best Practices

1. **Data Loading:** Use `data_type='Raw'` for full preprocessing pipeline
2. **PCA:** Default 95% variance retention is recommended
3. **Visualization:** Generate all plots after training for complete analysis
4. **Training:** Use early stopping to prevent overfitting

## 🔄 Dependencies

- NumPy, Pandas - Data manipulation
- Scikit-learn - Preprocessing and PCA
- Matplotlib, Seaborn - Visualization
- PyTorch - Tensor operations
