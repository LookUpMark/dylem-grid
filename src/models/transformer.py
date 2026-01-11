"""Transformer encoder for gesture recognition."""
import math
import torch
import torch.nn as nn
from torch import Tensor
from src.models.base import GestureBaseModule


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModule(GestureBaseModule):
    """Encoder-only Transformer with configurable pooling."""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, pooling: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.pooling = pooling
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(dim_feedforward, self.hparams.num_classes))

    def _pool(self, x: Tensor) -> Tensor:
        if self.pooling == "max": return x.max(dim=1)[0]
        elif self.pooling == "cls": return x[:, 0]
        return x.mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_encoder(self.input_proj(x))
        return self.classifier(self._pool(self.encoder(x)))

    def forward_with_attention(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward returning (logits, attention_maps_per_layer)."""
        hidden = self.pos_encoder(self.input_proj(x))
        attention_maps = []
        for layer in self.encoder.layers:
            attn_out, attn_weights = layer.self_attn(hidden, hidden, hidden, need_weights=True, average_attn_weights=False)
            attention_maps.append(attn_weights)
            hidden = layer.norm1(hidden + layer.dropout1(attn_out))
            hidden = layer.norm2(hidden + layer._ff_block(hidden))
        return self.classifier(self._pool(hidden)), attention_maps


GestureTransformer = TransformerModule
