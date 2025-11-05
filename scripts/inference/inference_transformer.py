"""
Inference script for gesture recognition
Load a trained model and make predictions on new data
"""

import torch
import pandas as pd
import numpy as np
from models.architectures.transformer_model import GestureTransformer
from utils.data_processing import prepare_data


def load_model(model_path="gesture_transformer_model.pth"):
    """
    Load a trained transformer model from file

    Args:
        model_path: Path to the saved model

    Returns:
        model: Loaded PyTorch model
        label_encoder: Fitted LabelEncoder
        model_config: Dictionary with model configuration
    """
    # Load with weights_only=False since we're loading sklearn objects
    # Only do this if you trust the source of the checkpoint
    checkpoint = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=False
    )

    # Extract model configuration
    config = {
        "input_size": checkpoint["input_size"],
        "d_model": checkpoint["d_model"],
        "nhead": checkpoint["nhead"],
        "num_layers": checkpoint["num_layers"],
        "dim_feedforward": checkpoint["dim_feedforward"],
        "num_classes": checkpoint["num_classes"],
        "dropout": checkpoint.get("dropout", 0.1),
    }

    # Initialize model
    model = GestureTransformer(
        input_size=config["input_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    label_encoder = checkpoint["label_encoder"]

    return model, label_encoder, config


def predict_gesture(model, data, label_encoder, device="cpu"):
    """
    Make predictions on new data

    Args:
        model: Trained PyTorch model
        data: Input tensor of shape (batch_size, seq_len, features)
        label_encoder: Fitted LabelEncoder
        device: Device to run on

    Returns:
        predictions: List of predicted class names
        probabilities: Numpy array of class probabilities
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        outputs = model(data)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_indices = torch.max(outputs, 1)

    predictions = label_encoder.inverse_transform(predicted_indices.cpu().numpy())
    probabilities = probabilities.cpu().numpy()

    return predictions, probabilities


def main():
    print("LOADING MODEL")
    print("\n" * 2)
    model, label_encoder, config = load_model("gesture_transformer_model.pth")

    print("\n" * 3)
    print("MODEL CONFIGURATION")
    print("\n" * 2)
    print(f"  Input size: {config['input_size']}")
    print(f"  Model dimension: {config['d_model']}")
    print(f"  Attention heads: {config['nhead']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Feedforward dim: {config['dim_feedforward']}")
    print(f"  Classes: {label_encoder.classes_}")

    print("\n" * 3)
    print("USAGE INSTRUCTIONS")
    print("\n" * 2)
    print("To use this model for inference:")
    print("1. Load your CSV data")
    print("2. Apply the same preprocessing (see utils.data_preprocess)")
    print("3. Apply the same PCA transformation")
    print("4. Pad sequences to the same length as training data")
    print("5. Call predict_gesture() with your data")
    print("\n" * 3)


if __name__ == "__main__":
    main()
