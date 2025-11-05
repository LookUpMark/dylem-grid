"""
Training script for gesture recognition with Encoder-only Transformer

This script uses hyperparameters optimized with Optuna (Bayesian optimization).
Run hyperparameter_optimization_transformer.py first to find optimal parameters.

Optimized hyperparameters (from transformer_optuna_results.json):
Update these values after running hyperparameter optimization!
- d_model: 64
- nhead: 4
- num_layers: 1
- dim_feedforward: 64
- dropout: 0.4938307576650347
- learning_rate: 0.001081641230706332
- weight_decay: 1.0301684581532967e-05
- batch_size: 32
- optimizer: Adam
- random_state: 44

Note: Update the hyperparameters below with your optimization results
"""

import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.data_processing import data_loader, data_preprocess, apply_pca, prepare_data

from src.transformer_model import GestureTransformer, train_model, evaluate_model

from src.plots import (
    plot_training_history,
    plot_confusion_matrix,
    plot_comprehensive_metrics,
    plot_pca_boxplot,
)


def main():
    # Header
    print("Gesture recognition — Transformer + Attention")
    print("\n" * 3)

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    print("Random seeds set (seed=42)")
    print("\n" * 3)

    # Load data
    print("LOADING DATA")
    print("\n" * 2)
    data, labels = data_loader("./DYLEM-GRID", "Raw")
    print(f"Loaded {len(data):,} samples")

    # Show label distribution
    label_dist = pd.Series(labels).value_counts().to_dict()
    print("Label distribution:")
    for label, count in label_dist.items():
        print(f"  {label:<18} {count:>4} ({count / len(labels) * 100:>4.1f}%)")
    print("\n" * 3)

    # Preprocess data
    print("PREPROCESSING DATA")
    print("\n" * 2)
    print("Preprocessing...")
    data, labels = data_preprocess(data, labels)
    print(f"Sample shape after preprocessing: {data[0].shape}")
    print("\n" * 3)

    # Apply PCA
    print("DIMENSIONALITY REDUCTION (PCA)")
    print("\n" * 2)
    print("Applying PCA (95% variance)...")
    data, labels = apply_pca(data, labels, variance_threshold=0.95)
    print(f"Sample shape after PCA: {data[0].shape}")

    # Generate PCA boxplot visualization
    print("Creating PCA boxplot visualization...")
    plot_pca_boxplot(data, labels, filename="plots_output/transformer_pca_boxplot.png")
    print("\n" * 3)

    # Prepare data for PyTorch
    print("DATA PREPARATION")
    print("\n" * 2)
    X, y, label_encoder = prepare_data(data, labels)
    print(f"Final data shape: {X.shape}")
    print(f"Final labels shape: {y.shape}")

    # Show classes
    print(
        f"Detected classes: {len(label_encoder.classes_)} -> {', '.join(label_encoder.classes_)}"
    )

    # Split data (using same random_state as optimization for reproducibility)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=44, stratify=y
    )
    print(f"Train/Validation split:")
    print(
        f"  Training set:   {X_train.shape[0]:>4} samples ({X_train.shape[0] / X.shape[0] * 100:>5.1f}%)"
    )
    print(
        f"  Validation set: {X_val.shape[0]:>4} samples ({X_val.shape[0] / X.shape[0] * 100:>5.1f}%)"
    )
    print("\n" * 3)

    # Create data loaders
    print("MODEL SETUP")
    print("\n" * 2)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Hyperparameters optimized with Optuna
    batch_size = 32  # Optimal from Optuna
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_size = X.shape[2]  # Number of features
    d_model = 64  # Optimal from Optuna
    nhead = 4  # Optimal from Optuna
    num_layers = 1  # Optimal from Optuna
    dim_feedforward = 64  # Optimal from Optuna
    num_classes = len(label_encoder.classes_)
    dropout = 0.4938307576650347  # Optimal from Optuna

    print("Initializing model...")
    model = GestureTransformer(
        input_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        num_classes,
        dropout=dropout,
    )
    model.to(device)

    print(
        f"Model: input={input_size}, d_model={d_model}, heads={nhead}, layers={num_layers},"
    )
    print(f"       ff={dim_feedforward}, classes={num_classes}, dropout={dropout:.4f}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("\n" * 3)

    # Training
    print("TRAINING PHASE")
    print("\n" * 2)

    # Optimizer parameters optimized with Optuna (update after optimization)
    learning_rate = 0.001081641230706332  # Optimal from Optuna
    weight_decay = 1.0301684581532967e-05  # Optimal from Optuna
    optimizer_name = "Adam"  # Optimal from Optuna (options: Adam, AdamW, NAdam)

    criterion = nn.CrossEntropyLoss()

    # Select optimizer based on optimization results
    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "NAdam":
        optimizer = optim.NAdam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    print(
        f"Training: CrossEntropyLoss, {optimizer_name}(lr={learning_rate:.6f}, weight_decay={weight_decay:.2e}), early stopping=15"
    )

    (
        model,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        best_stats,
        best_val_preds,
        best_val_targets,
    ) = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=50,
        patience=15,
    )

    print("\n" * 3)
    print("FINAL EVALUATION")
    print("\n" * 2)
    print("Best model results:")

    val_acc = best_stats["val_acc"]
    val_preds = best_val_preds
    val_targets = best_val_targets

    print("BEST MODEL PERFORMANCE:")
    print(
        f"  epoch={best_stats['epoch']}  val_loss={best_stats['val_loss']:.6f}  val_acc={val_acc:.4f}  correct={int(val_acc * len(val_targets))}/{len(val_targets)}"
    )
    print("\n" * 3)

    print("GENERATING VISUALIZATIONS")
    print("\n" * 2)
    print("Creating enhanced training history plot...")
    plot_training_history(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        best_stats,
        filename="plots_output/transformer_training_history.png",
    )

    print("Generating detailed confusion matrix...")
    plot_confusion_matrix(
        val_targets,
        val_preds,
        label_encoder.classes_,
        val_acc,
        best_stats["epoch"],
        filename="plots_output/transformer_confusion_matrix.png",
    )

    print("Building comprehensive metrics dashboard...")
    plot_comprehensive_metrics(
        val_targets,
        val_preds,
        label_encoder.classes_,
        val_acc,
        best_stats["epoch"],
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        best_stats,
        filename="plots_output/transformer_comprehensive_metrics.png",
    )
    print("\n" * 3)

    # Save model
    print("SAVING MODEL")
    print("\n" * 2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_encoder": label_encoder,
            "input_size": input_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "num_classes": num_classes,
            "dropout": dropout,
            "best_stats": best_stats,
            "best_val_preds": best_val_preds,
            "best_val_targets": best_val_targets,
        },
        "gesture_transformer_model.pth",
    )
    print("Model saved as 'gesture_transformer_model.pth'")
    print("Saved components: model weights, label encoder, configuration")
    print("\n" * 3)

    # Final success message
    print("=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Final Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Best Epoch: {best_stats['epoch']}")
    print(
        f"Output plots: plots_output/transformer_pca_boxplot.png, plots_output/transformer_training_history.png,"
    )
    print(
        f"              plots_output/transformer_confusion_matrix.png, plots_output/transformer_comprehensive_metrics.png"
    )
    print(f"Model checkpoint: gesture_transformer_model.pth")


if __name__ == "__main__":
    main()
