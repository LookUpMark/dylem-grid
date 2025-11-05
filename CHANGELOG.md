# Changelog

All notable changes to the DYLEM-GRID project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.0.0] - 2025-11-05

### 🎉 Major Repository Refactoring

### Added
- New directory structure with clear separation of concerns
- `models/architectures/` - Model definition modules
- `models/checkpoints/` - Trained model weights storage
- `utils/` - Utility modules (data processing, plots, training)
- `scripts/train/` - Training scripts
- `scripts/inference/` - Inference scripts
- `scripts/optimization/` - Hyperparameter optimization scripts
- `data/` - Dataset location directory
- `plots/` - Generated visualizations
- `checkpoints/` - W&B checkpoints
- `.gitkeep` files in all empty directories
- `MIGRATION_GUIDE.md` - Complete migration documentation
- `REFACTORING_SUMMARY.md` - Refactoring overview
- `CHANGELOG.md` - This file
- `scripts/README.md` - Scripts usage documentation
- `models/README.md` - Models documentation
- `utils/README.md` - Utilities documentation

### Changed
- Moved all model architectures from `src/` to `models/architectures/`
- Moved all utilities from `src/` to `utils/`
- Moved all scripts from root to `scripts/` subdirectories
- Moved model checkpoints from root to `models/checkpoints/`
- Renamed `plots_output/` to `plots/`
- Updated all import statements from `src.*` to `utils.*` and `models.architectures.*`
- Updated all file paths to use relative paths from script locations
- Enhanced `.gitignore` with new directory patterns
- Updated `README.md` with new project structure
- Updated usage examples in documentation

### Removed
- `src/` directory (contents moved to `models/architectures/` and `utils/`)
- Root-level training scripts (`train_bilstm.py`, `train_transformer.py`)
- Root-level inference scripts (`inference_bilstm.py`, `inference_transformer.py`)
- Root-level optimization scripts (`hyperparameter_optimization_*.py`)
- Root-level model files (moved to `models/checkpoints/`)
- `plots_output/` directory (renamed to `plots/`)

### Migration Notes
- **Breaking Change:** Import paths have changed
  - Old: `from src.data_processing import ...`
  - New: `from utils.data_processing import ...`
- **Breaking Change:** Script execution paths have changed
  - Old: `python train_bilstm.py`
  - New: `python scripts/train/train_bilstm.py`
- All scripts must be run from project root directory
- See `MIGRATION_GUIDE.md` for complete migration instructions

### Benefits
- Clearer project organization
- Industry-standard structure
- Easier navigation and maintenance
- Better separation of concerns
- More professional and scalable
- Improved Git management

---

## [1.0.0] - Previous

### Initial Release
- BiLSTM + Attention model implementation
- Transformer encoder model implementation
- Data preprocessing pipeline
- PCA dimensionality reduction
- Training scripts with early stopping
- Hyperparameter optimization with Optuna
- Comprehensive visualization suite
- Kaggle notebook examples
- Complete documentation

### Features
- Dynamic hand gesture recognition
- Two model architectures (BiLSTM and Transformer)
- 100% validation accuracy achieved
- Professional visualizations
- Inference utilities
- Detailed performance metrics

---

## Version History

| Version | Date       | Description                          |
|---------|------------|--------------------------------------|
| 2.0.0   | 2025-11-05 | Major repository refactoring         |
| 1.0.0   | Previous   | Initial release with BiLSTM & Trans. |

---

## Future Plans

### Planned Features
- [ ] Add more model architectures (GRU, TCN)
- [ ] Implement model ensemble methods
- [ ] Add real-time inference API
- [ ] Create Docker containerization
- [ ] Add continuous integration (CI/CD)
- [ ] Implement model quantization
- [ ] Add explainability tools (GradCAM, SHAP)
- [ ] Create web interface for inference

### Improvements
- [ ] Optimize data loading pipeline
- [ ] Add data augmentation
- [ ] Implement cross-validation
- [ ] Add model pruning
- [ ] Improve documentation
- [ ] Add more unit tests
- [ ] Create benchmark suite

---

**Maintained by:** LookUpMark  
**Repository:** [dylem-grid](https://github.com/LookUpMark/dylem-grid)  
**License:** MIT
