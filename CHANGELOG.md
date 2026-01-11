# Changelog

## [2.0.0] - 2026-01-11

### Added
- **PyTorch Lightning integration** - Modular training with `LightningModule` and `LightningDataModule`
- **Hugging Face Hub integration** - Auto-download dataset and models
- **K-Fold Cross-Validation** - `CrossValidator` with aggregated results (5-fold stratified)
- **Ablation Study Framework** - `AblationRunner` for systematic experiments
- **IEEE Conference Paper** - Complete research paper in `paper/paper.tex`
  - Discussion section with statistical significance analysis
  - Limitations and implications for practitioners
  - 10 academic references including 2023-2024 papers
- **New `src/` package** - Modular, well-organized codebase
- **Jupyter Notebooks** - `01_optimization`, `02_training`, `03_inference`, `04_ablation`

### Changed
- Complete codebase refactor for modularity and conciseness
- Models now inherit from `GestureBaseModule` with shared training logic
- Data loading moved to `GestureDataModule` with HF Hub support
- Optuna integration uses `PyTorchLightningPruningCallback`
- Accuracy results updated: BiLSTM 97.25%, Transformer 94.75%

### Removed
- Legacy `utils/` directory (integrated into `src/`)
- Legacy `models/architectures/` (replaced by `src/models/`)
- Separate train/inference/optimization scripts (unified)
- Deprecated plots and redundant images (~14 MB cleaned)

## [1.0.0] - 2024-11-27

### Initial Release
- BiLSTM + Attention model
- Transformer model
- Optuna hyperparameter optimization
- W&B experiment tracking
- DYLEM-GRID dataset integration

