"""
Neural network model definition for gesture recognition
Contains GestureRNN model and imports shared training utilities
"""

import torch
from torch import nn
from .training_utils import train_model, evaluate_model


class GestureRNN(nn.Module):
    """
    Bidirectional LSTM with Attention mechanism for gesture recognition
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(GestureRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM for better temporal understanding
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM output: (batch_size, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)

        # Attention weights: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # Apply attention: (batch_size, hidden_size * 2)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected layers
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
