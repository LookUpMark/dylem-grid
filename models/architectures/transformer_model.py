"""
Encoder-only Transformer model for gesture recognition
Simple and efficient architecture using multi-head self-attention
"""

import torch
from torch import nn
import math
from utils.training_utils import train_model, evaluate_model


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class GestureTransformer(nn.Module):
    """
    Encoder-only Transformer for gesture classification
    Uses multi-head self-attention to capture temporal patterns
    """

    def __init__(
        self,
        input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=4,
        dropout=0.1,
    ):
        super(GestureTransformer, self).__init__()

        self.d_model = d_model

        # Project input features to model dimension
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # Project to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Global average pooling over sequence
        x = torch.mean(x, dim=1)  # (batch_size, d_model)

        # Classification
        x = self.classifier(x)  # (batch_size, num_classes)

        return x
