"""
Shared training and evaluation utilities for gesture recognition models
Contains common functions used by both BiLSTM and Transformer models
"""

import torch
from torch import nn

# Optional progress bar for training (gracefully degrade if tqdm not available)
try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset

    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        criterion: Loss function
        device: Device to run on (CPU or CUDA)

    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
        all_preds: List of all predictions
        all_targets: List of all targets
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_targets


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=50,
    patience=15,
):
    """
    Train a model with early stopping

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (CPU or CUDA)
        num_epochs: Maximum number of epochs
        patience: Early stopping patience

    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        best_stats: Dictionary with best epoch statistics
        best_val_preds: Best validation predictions
        best_val_targets: Best validation targets
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    patience_counter = 0
    best_stats = {}
    best_val_preds = []
    best_val_targets = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # Calculate training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation phase
        val_loss, val_acc, val_preds, val_targets = evaluate_model(
            model, val_loader, criterion, device
        )

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_stats = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
            best_val_preds = val_preds
            best_val_targets = val_targets
            patience_counter = 0

            # Save best model
            # torch.save(
            #     {
            #         "epoch": epoch + 1,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "loss": val_loss,
            #     },
            #     "best_model_checkpoint.pth",
            # )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return (
        model,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        best_stats,
        best_val_preds,
        best_val_targets,
    )
