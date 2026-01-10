"""Hyperparameter search spaces for Optuna."""
from dataclasses import dataclass, field
from typing import Any, List, Tuple
from optuna import Trial


@dataclass
class BiLSTMSearchSpace:
    """Search space for BiLSTM hyperparameters."""
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    num_layers_range: Tuple[int, int] = (1, 3)
    lr_range: Tuple[float, float] = (1e-4, 1e-2)
    dropout_range: Tuple[float, float] = (0.1, 0.5)
    weight_decay_range: Tuple[float, float] = (1e-6, 1e-3)
    optimizers: List[str] = field(default_factory=lambda: ["nadam", "adam", "adamw"])
    include_attention: bool = False
    name: str = "bilstm"

    def sample(self, trial: Trial) -> dict:
        p = {"hidden_size": trial.suggest_categorical("hidden_size", self.hidden_sizes),
             "num_layers": trial.suggest_int("num_layers", *self.num_layers_range),
             "learning_rate": trial.suggest_float("learning_rate", *self.lr_range, log=True),
             "dropout": trial.suggest_float("dropout", *self.dropout_range),
             "weight_decay": trial.suggest_float("weight_decay", *self.weight_decay_range, log=True),
             "optimizer": trial.suggest_categorical("optimizer", self.optimizers)}
        if self.include_attention: p["use_attention"] = trial.suggest_categorical("use_attention", [True, False])
        return p


@dataclass
class TransformerSearchSpace:
    """Search space for Transformer hyperparameters."""
    d_models: List[int] = field(default_factory=lambda: [32, 64, 128])
    nheads: List[int] = field(default_factory=lambda: [2, 4, 8])
    num_layers_range: Tuple[int, int] = (1, 4)
    dim_ffs: List[int] = field(default_factory=lambda: [64, 128, 256])
    lr_range: Tuple[float, float] = (1e-4, 1e-2)
    dropout_range: Tuple[float, float] = (0.1, 0.5)
    weight_decay_range: Tuple[float, float] = (1e-6, 1e-3)
    optimizers: List[str] = field(default_factory=lambda: ["nadam", "adam", "adamw"])
    poolings: List[str] = field(default_factory=lambda: ["mean", "max", "cls"])
    include_pooling: bool = False
    name: str = "transformer"

    def sample(self, trial: Trial) -> dict:
        d = trial.suggest_categorical("d_model", self.d_models)
        valid_heads = [h for h in self.nheads if d % h == 0] or [1]
        p = {"d_model": d, "nhead": trial.suggest_categorical("nhead", valid_heads),
             "num_layers": trial.suggest_int("num_layers", *self.num_layers_range),
             "dim_feedforward": trial.suggest_categorical("dim_feedforward", self.dim_ffs),
             "learning_rate": trial.suggest_float("learning_rate", *self.lr_range, log=True),
             "dropout": trial.suggest_float("dropout", *self.dropout_range),
             "weight_decay": trial.suggest_float("weight_decay", *self.weight_decay_range, log=True),
             "optimizer": trial.suggest_categorical("optimizer", self.optimizers)}
        if self.include_pooling: p["pooling"] = trial.suggest_categorical("pooling", self.poolings)
        return p


# Pre-configured spaces
DEFAULT_BILSTM_SPACE = BiLSTMSearchSpace()
DEFAULT_TRANSFORMER_SPACE = TransformerSearchSpace()
COMPACT_BILSTM_SPACE = BiLSTMSearchSpace(hidden_sizes=[32, 64], num_layers_range=(1, 2), optimizers=["nadam"])
COMPACT_TRANSFORMER_SPACE = TransformerSearchSpace(d_models=[32, 64], num_layers_range=(1, 2), optimizers=["nadam"])
