"""BiLSTM with Attention for gesture recognition."""
import torch
import torch.nn as nn
from torch import Tensor
from src.models.base import GestureBaseModule


class BiLSTMModule(GestureBaseModule):
    """Bidirectional LSTM with attention mechanism."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.15, use_attention: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.use_attention = use_attention
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        if use_attention: self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.BatchNorm1d(hidden_size),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, self.hparams.num_classes))

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        if self.use_attention:
            weights = torch.softmax(self.attention(out), dim=1)
            out = (weights * out).sum(dim=1)
        else:
            out = out[:, -1, :]
        return self.classifier(out)


GestureRNN = BiLSTMModule  # Backward compat
