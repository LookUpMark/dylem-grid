"""
Main training script for gesture recognition with RNN

This script uses hyperparameters optimized with Optuna (Bayesian optimization).
Run hyperparameter_optimization_bilstm.py first to find optimal parameters.

Optimized hyperparameters (from optuna_results.json):
- hidden_size: 64
- num_layers: 2
- learning_rate: 0.00196
- dropout: 0.156
- optimizer: NAdam
- weight_decay: 3.70e-06
- batch_size: 32
- random_state: 44

Result: 100% validation accuracy
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from utils.data_processing import (
    data_loader,
    data_preprocess,
    apply_pca,
    prepare_data
)

from models.architectures.bilstm_model import (
    GestureRNN,
    train_model,
    evaluate_model
)

from utils.plots import (
    plot_training_history,
    plot_confusion_matrix,
    plot_comprehensive_metrics,
    plot_pca_boxplot,
    plot_model_weights_heatmap,
    plot_lstm_weights_heatmap
)

from utils.display import (
    print_header,
    print_section,
    print_subsection,
    print_success,
    print_info,
    print_metric,
    print_model_summary
)


def main():
    # Header
    print_header("BiLSTM + Attention — Gesture Recognition")

    # Initialize Weights & Biases
    if WANDB_AVAILABLE:
        wandb.init(
            project="dylem-grid-gestures",
            name="bilstm-attention-training",
            config={
                "architecture": "BiLSTM + Attention",
                "dataset": "DYLEM-GRID",
                "hidden_size": 64,
                "num_layers": 2,
                "learning_rate": 0.0019562925936030193,
                "dropout": 0.15578214103568824,
                "optimizer": "NAdam",
                "weight_decay": 3.6959932544718737e-06,
                "batch_size": 32,
                "random_state": 44,
                "pca_variance": 0.95,
                "epochs": 50,
                "patience": 15
            },
            tags=["bilstm", "attention", "gesture-recognition"]
        )
        print_success("Weights & Biases initialized")
    else:
        print_info("Training without W&B tracking (wandb not installed)")

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    print_success("Random seeds initialized (seed=42)")

    # Load data
    print_section("Data Loading")
    data, labels = data_loader('data/DYLEM-GRID', 'Raw')
    print_success(f"Loaded {len(data):,} samples")

    # Show label distribution
    print_subsection("Label Distribution")
    label_dist = pd.Series(labels).value_counts().sort_index().to_dict()
    for label, count in label_dist.items():
        percentage = count/len(labels)*100
        print_info(f"{label:<18} {count:>4} samples ({percentage:>5.1f}%)")

    # Preprocess data
    print_section("Data Preprocessing")
    data, labels = data_preprocess(data, labels)
    print_success(f"Preprocessing complete → shape: {data[0].shape}")

    # Apply PCA
    print_subsection("Dimensionality Reduction (PCA)")
    data, labels = apply_pca(data, labels, variance_threshold=0.95)
    print_success(f"PCA applied (95% variance) → shape: {data[0].shape}")
    
    plot_pca_boxplot(data, labels, filename='plots/bilstm_pca_boxplot.png')
    print_info("PCA boxplot saved to plots/bilstm_pca_boxplot.png")

    # Prepare data for PyTorch
    print_section("Data Preparation")
    X, y, label_encoder = prepare_data(data, labels)
    print_success(f"Data tensors created → X: {X.shape}, y: {y.shape}")
    print_info(f"Classes ({len(label_encoder.classes_)}): {', '.join(label_encoder.classes_)}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=44, stratify=y
    )
    print_subsection("Train/Validation Split")
    print_info(f"Training:   {X_train.shape[0]:>4} samples ({X_train.shape[0]/X.shape[0]*100:>5.1f}%)")
    print_info(f"Validation: {X_val.shape[0]:>4} samples ({X_val.shape[0]/X.shape[0]*100:>5.1f}%)")

    # Create data loaders
    print_section("Model Setup")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Hyperparameters optimized with Optuna (100% validation accuracy)
    batch_size = 32  # Optimal from Optuna
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(f"Device: {device}")

    input_size = X.shape[2]  # Number of features
    hidden_size = 64  # Optimal from Optuna
    num_layers = 2  # Optimal from Optuna
    num_classes = len(label_encoder.classes_)
    dropout = 0.15578214103568824  # Optimal from Optuna

    model = GestureRNN(input_size, hidden_size, num_layers, num_classes, dropout=dropout)
    model.to(device)
    
    # Watch model with wandb
    if WANDB_AVAILABLE:
        wandb.watch(model, log="all", log_freq=10)
    
    model_config = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'dropout': dropout
    }
    print_model_summary("BiLSTM", sum(p.numel() for p in model.parameters()), model_config)

    # Training
    print_section("Training Phase")
    
    # Optimizer parameters optimized with Optuna
    learning_rate = 0.0019562925936030193  # Optimal from Optuna
    weight_decay = 3.6959932544718737e-06  # Optimal from Optuna
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print_subsection("Training Configuration")
    print_info(f"Loss: CrossEntropyLoss")
    print_info(f"Optimizer: NAdam (lr={learning_rate:.6f}, weight_decay={weight_decay:.2e})")
    print_info(f"Epochs: 50 (early stopping patience=15)")

    model, train_losses, val_losses, train_accs, val_accs, best_stats, best_val_preds, best_val_targets = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=50, patience=15
    )

    # Log training curves to wandb
    if WANDB_AVAILABLE:
        for epoch in range(len(train_losses)):
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_losses[epoch],
                "train/accuracy": train_accs[epoch],
                "val/loss": val_losses[epoch],
                "val/accuracy": val_accs[epoch]
            })

    # Final evaluation
    print_section("Final Evaluation")
    val_acc = best_stats['val_acc']
    val_preds = best_val_preds
    val_targets = best_val_targets

    print_subsection("Best Model Performance")
    print_metric("Epoch", best_stats['epoch'])
    print_metric("Validation Loss", f"{best_stats['val_loss']:.6f}")
    print_metric("Validation Accuracy", f"{val_acc:.4f} ({val_acc*100:.2f}%)", good_threshold=0.95)
    print_info(f"Correct predictions: {int(val_acc * len(val_targets))}/{len(val_targets)}")

    # Log final metrics to wandb
    if WANDB_AVAILABLE:
        wandb.log({
            "best/epoch": best_stats['epoch'],
            "best/val_loss": best_stats['val_loss'],
            "best/val_accuracy": val_acc,
            "best/correct_predictions": int(val_acc * len(val_targets)),
            "best/total_samples": len(val_targets)
        })

    # Generate visualizations
    print_section("Generating Visualizations")
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs, best_stats, 
                         filename='plots/bilstm_training_history.png')
    print_success("Training history plot saved")

    plot_confusion_matrix(val_targets, val_preds, label_encoder.classes_, val_acc, best_stats['epoch'],
                         filename='plots/bilstm_confusion_matrix.png')
    print_success("Confusion matrix saved")

    plot_comprehensive_metrics(val_targets, val_preds, label_encoder.classes_, val_acc, best_stats['epoch'],
                              train_losses, val_losses, train_accs, val_accs, best_stats,
                              filename='plots/bilstm_comprehensive_metrics.png')
    print_success("Comprehensive metrics dashboard saved")

    plot_model_weights_heatmap(model, label_encoder.classes_, best_stats['epoch'],
                              filename='plots/bilstm_weights_heatmap.png')
    print_success("Model weights heatmap saved")

    plot_lstm_weights_heatmap(model, best_stats['epoch'],
                             filename='plots/bilstm_lstm_weights_heatmap.png')
    print_success("LSTM weights heatmap saved")

    # Log visualizations to wandb
    if WANDB_AVAILABLE:
        wandb.log({
            "plots/training_history": wandb.Image('plots/bilstm_training_history.png'),
            "plots/confusion_matrix": wandb.Image('plots/bilstm_confusion_matrix.png'),
            "plots/comprehensive_metrics": wandb.Image('plots/bilstm_comprehensive_metrics.png'),
            "plots/model_weights": wandb.Image('plots/bilstm_weights_heatmap.png'),
            "plots/lstm_weights": wandb.Image('plots/bilstm_lstm_weights_heatmap.png'),
            "plots/pca_boxplot": wandb.Image('plots/bilstm_pca_boxplot.png')
        })

    # Save model
    print_section("Saving Model")
    model_path = 'models/checkpoints/gesture_rnn_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'best_stats': best_stats,
        'best_val_preds': best_val_preds,
        'best_val_targets': best_val_targets
    }, model_path)
    print_success(f"Model checkpoint saved to {model_path}")
    print_info("Saved: model weights, label encoder, configuration, best stats")

    # Log model artifact to wandb
    if WANDB_AVAILABLE:
        artifact = wandb.Artifact('bilstm-gesture-model', type='model')
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        wandb.finish()
        print_success("Model artifact logged to W&B")

    # Final success message
    print_header("TRAINING COMPLETED SUCCESSFULLY!", '=')
    print_metric("Final Accuracy", f"{val_acc:.4f} ({val_acc*100:.2f}%)")
    print_metric("Best Epoch", best_stats['epoch'])
    print_subsection("Generated Files")
    print_info("Plots:")
    print_info("  • plots/bilstm_pca_boxplot.png", indent=1)
    print_info("  • plots/bilstm_training_history.png", indent=1)
    print_info("  • plots/bilstm_confusion_matrix.png", indent=1)
    print_info("  • plots/bilstm_comprehensive_metrics.png", indent=1)
    print_info("  • plots/bilstm_weights_heatmap.png", indent=1)
    print_info("  • plots/bilstm_lstm_weights_heatmap.png", indent=1)
    print_info("Model:")
    print_info("  • models/checkpoints/gesture_rnn_model.pth", indent=1)
    print()


if __name__ == "__main__":
    main()
