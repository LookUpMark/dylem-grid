# Changelog

## [2.0.0] - 2024-01-10

### Added
- **PyTorch Lightning integration** - Modular training with `LightningModule` and `LightningDataModule`
- **Hugging Face Hub integration** - Auto-download dataset and models
- **K-Fold Cross-Validation** - `CrossValidator` with aggregated results
- **Ablation Study Framework** - `AblationRunner` for systematic experiments
- **Unified CLI scripts** - `train.py`, `optimize.py`, `inference.py`, `ablation.py`
- **New `src/` package** - Modular, well-organized codebase

### Changed
- Complete codebase refactor for modularity and conciseness
- Models now inherit from `GestureBaseModule` with shared training logic
- Data loading moved to `GestureDataModule` with HF Hub support
- Optuna integration uses `PyTorchLightningPruningCallback`

### Removed
- Legacy `utils/` directory (integrated into `src/`)
- Legacy `models/architectures/` (replaced by `src/models/`)
- Separate train/inference/optimization scripts (unified)
- Redundant model checkpoint files in root

## [1.0.0] - 2024-11-27

### Initial Release
- BiLSTM + Attention model
- Transformer model
- Optuna hyperparameter optimization
- W&B experiment tracking
- 100% accuracy on DYLEM-GRID dataset
