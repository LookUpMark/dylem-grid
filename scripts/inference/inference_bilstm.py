"""
Inference script for gesture recognition
Load a trained model and make predictions on new data
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import pandas as pd
import numpy as np
from models.architectures.bilstm_model import GestureRNN
from utils.data_processing import prepare_data
from utils.display import (
    print_header,
    print_section,
    print_subsection,
    print_success,
    print_info
)


def load_model(model_path='models/checkpoints/gesture_rnn_model.pth'):
    """
    Load a trained model from file
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        model: Loaded PyTorch model
        label_encoder: Fitted LabelEncoder
        model_config: Dictionary with model configuration
    """
    # Load with weights_only=False since we're loading sklearn objects
    # Only do this if you trust the source of the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Extract model configuration
    config = {
        'input_size': checkpoint['input_size'],
        'hidden_size': checkpoint['hidden_size'],
        'num_layers': checkpoint['num_layers'],
        'num_classes': checkpoint['num_classes']
    }
    
    # Initialize model
    model = GestureRNN(
        config['input_size'],
        config['hidden_size'],
        config['num_layers'],
        config['num_classes']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    label_encoder = checkpoint['label_encoder']
    
    return model, label_encoder, config


def predict_gesture(model, data, label_encoder, device='cpu'):
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
    print_header("BiLSTM Inference — Model Loading")
    
    print_section("Loading Model")
    model, label_encoder, config = load_model('models/checkpoints/gesture_rnn_model.pth')
    print_success("Model loaded successfully")
    
    print_section("Model Configuration")
    print_info(f"Input size: {config['input_size']}")
    print_info(f"Hidden size: {config['hidden_size']}")
    print_info(f"Number of layers: {config['num_layers']}")
    print_info(f"Number of classes: {config['num_classes']}")
    print_info(f"Classes: {', '.join(label_encoder.classes_)}")
    
    print_section("Usage Instructions")
    print_subsection("To use this model for inference:")
    print_info("1. Load your CSV data", indent=1)
    print_info("2. Apply the same preprocessing (utils.data_preprocess)", indent=1)
    print_info("3. Apply the same PCA transformation (95% variance)", indent=1)
    print_info("4. Pad sequences to the same length as training data", indent=1)
    print_info("5. Call predict_gesture() with your data", indent=1)
    print()
    print_subsection("Example:")
    print("  from inference_bilstm import load_model, predict_gesture")
    print("  model, label_encoder, config = load_model()")
    print("  predictions, probabilities = predict_gesture(model, your_data, label_encoder)")
    print()


if __name__ == "__main__":
    main()
