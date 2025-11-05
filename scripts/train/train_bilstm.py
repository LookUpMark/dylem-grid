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

import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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



def main():
    # Header
    print("Gesture recognition — BiLSTM + Attention")
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
    data, labels = data_loader('../../data/DYLEM-GRID', 'Raw')
    print(f"Loaded {len(data):,} samples")

    # Show label distribution
    label_dist = pd.Series(labels).value_counts().to_dict()
    print("Label distribution:")
    for label, count in label_dist.items():
        print(f"  {label:<18} {count:>4} ({count/len(labels)*100:>4.1f}%)")
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
    plot_pca_boxplot(data, labels, filename='../../plots/bilstm_pca_boxplot.png')
    print("\n" * 3)

    # Prepare data for PyTorch
    print("DATA PREPARATION")
    print("\n" * 2)
    X, y, label_encoder = prepare_data(data, labels)
    print(f"Final data shape: {X.shape}")
    print(f"Final labels shape: {y.shape}")

    # Show classes
    print(f"Detected classes: {len(label_encoder.classes_)} -> {', '.join(label_encoder.classes_)}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=44, stratify=y
    )
    print(f"Train/Validation split:")
    print(f"  Training set:   {X_train.shape[0]:>4} samples ({X_train.shape[0]/X.shape[0]*100:>5.1f}%)")
    print(f"  Validation set: {X_val.shape[0]:>4} samples ({X_val.shape[0]/X.shape[0]*100:>5.1f}%)")
    print("\n" * 3)

    # Create data loaders
    print("MODEL SETUP")
    print("\n" * 2)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Hyperparameters optimized with Optuna (100% validation accuracy)
    batch_size = 32  # Optimal from Optuna
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_size = X.shape[2]  # Number of features
    hidden_size = 64  # Optimal from Optuna
    num_layers = 2  # Optimal from Optuna
    num_classes = len(label_encoder.classes_)
    dropout = 0.15578214103568824  # Optimal from Optuna

    print("Initializing model...")
    model = GestureRNN(input_size, hidden_size, num_layers, num_classes, dropout=dropout)
    model.to(device)

    print(f"Model: input={input_size}, hidden={hidden_size}, layers={num_layers}, classes={num_classes}, dropout={dropout:.4f}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("\n" * 3)

    # Training
    print("TRAINING PHASE")
    print("\n" * 2)
    
    # Optimizer parameters optimized with Optuna
    learning_rate = 0.0019562925936030193  # Optimal from Optuna
    weight_decay = 3.6959932544718737e-06  # Optimal from Optuna
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(f"Training: CrossEntropyLoss, NAdam(lr={learning_rate:.6f}, weight_decay={weight_decay:.2e}), early stopping=15")

    model, train_losses, val_losses, train_accs, val_accs, best_stats, best_val_preds, best_val_targets = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=50, patience=15
    )

    # Final evaluation
    print()
    print("FINAL EVALUATION")
    print()
    print("Best model results:")

    val_acc = best_stats['val_acc']
    val_preds = best_val_preds
    val_targets = best_val_targets

    print("BEST MODEL PERFORMANCE:")
    print(f"  epoch={best_stats['epoch']}  val_loss={best_stats['val_loss']:.6f}  val_acc={val_acc:.4f}  correct={int(val_acc * len(val_targets))}/{len(val_targets)}")
    print("\n" * 3)

    # Generate visualizations
    print("GENERATING VISUALIZATIONS")
    print("\n" * 2)
    print("Creating enhanced training history plot...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, best_stats, 
                                                  filename='../../plots/bilstm_training_history.png')

    print("Generating detailed confusion matrix...")
    plot_confusion_matrix(val_targets, val_preds, label_encoder.classes_, val_acc, best_stats['epoch'],
                         filename='../../plots/bilstm_confusion_matrix.png')

    print("Building comprehensive metrics dashboard...")
    plot_comprehensive_metrics(val_targets, val_preds, label_encoder.classes_, val_acc, best_stats['epoch'],
                             train_losses, val_losses, train_accs, val_accs, best_stats,
                             filename='../../plots/bilstm_comprehensive_metrics.png')

    print("Creating model weights heatmap...")
    plot_model_weights_heatmap(model, label_encoder.classes_, best_stats['epoch'],
                              filename='../../plots/bilstm_weights_heatmap.png')

    print("Creating LSTM weights heatmap...")
    plot_lstm_weights_heatmap(model, best_stats['epoch'],
                             filename='../../plots/bilstm_lstm_weights_heatmap.png')
    print("\n" * 3)

    # Save model
    print("SAVING MODEL")
    print("\n" * 2)
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
    }, '../../models/checkpoints/gesture_rnn_model.pth')
    print("Model saved as '../../models/checkpoints/gesture_rnn_model.pth'")
    print("Saved components: model weights, label encoder, configuration")
    print("\n" * 3)

    # Final success message
    print("=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Final Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Best Epoch: {best_stats['epoch']}")
    print(f"Output plots: ../../plots/bilstm_pca_boxplot.png, ../../plots/bilstm_training_history.png,")
    print(f"              ../../plots/bilstm_confusion_matrix.png, ../../plots/bilstm_comprehensive_metrics.png,")
    print(f"              ../../plots/bilstm_weights_heatmap.png, ../../plots/bilstm_lstm_weights_heatmap.png")
    print(f"Model checkpoint: ../../models/checkpoints/gesture_rnn_model.pth")


if __name__ == "__main__":
    main()
