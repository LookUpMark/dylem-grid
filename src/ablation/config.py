"""Ablation study configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal
import yaml


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    name: str
    model: Literal["bilstm", "transformer"]
    base_params: Dict[str, Any]
    ablations: Dict[str, List[Any]]
    training_params: Dict[str, Any] = field(default_factory=lambda: {"max_epochs": 50, "patience": 10, "batch_size": 32})
    use_cv: bool = True
    n_folds: int = 5

    def get_all_configurations(self):
        """Generate all (name, param, value, config) tuples."""
        configs = []
        for param, values in self.ablations.items():
            for val in values:
                cfg = {**self.base_params, param: val}
                configs.append((f"{param}={val}", param, val, cfg))
        return configs

    def get_num_configurations(self): return sum(len(v) for v in self.ablations.values())

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f: return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f: yaml.dump(self.__dict__, f, sort_keys=False)


# Pre-defined configurations
BILSTM_ABLATIONS = AblationConfig(
    name="bilstm_ablation", model="bilstm",
    base_params={"hidden_size": 64, "num_layers": 2, "dropout": 0.15, "learning_rate": 0.002,
                 "weight_decay": 1e-5, "optimizer": "nadam", "use_attention": True},
    ablations={"hidden_size": [32, 64, 128, 256], "num_layers": [1, 2, 3],
               "dropout": [0.0, 0.1, 0.2, 0.3], "use_attention": [True, False]})

TRANSFORMER_ABLATIONS = AblationConfig(
    name="transformer_ablation", model="transformer",
    base_params={"d_model": 64, "nhead": 8, "num_layers": 2, "dim_feedforward": 128,
                 "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 1e-5, "optimizer": "nadam", "pooling": "mean"},
    ablations={"d_model": [32, 64, 128], "num_layers": [1, 2, 3, 4], "pooling": ["mean", "max", "cls"]})

BILSTM_QUICK = AblationConfig(
    name="bilstm_quick", model="bilstm",
    base_params={"hidden_size": 64, "num_layers": 2, "dropout": 0.15, "learning_rate": 0.002,
                 "weight_decay": 1e-5, "optimizer": "nadam", "use_attention": True},
    ablations={"hidden_size": [32, 64], "use_attention": [True, False]},
    training_params={"max_epochs": 30, "patience": 5, "batch_size": 32}, n_folds=3)

TRANSFORMER_QUICK = AblationConfig(
    name="transformer_quick", model="transformer",
    base_params={"d_model": 64, "nhead": 8, "num_layers": 2, "dim_feedforward": 128,
                 "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 1e-5, "optimizer": "nadam", "pooling": "mean"},
    ablations={"d_model": [32, 64], "pooling": ["mean", "max"]},
    training_params={"max_epochs": 30, "patience": 5, "batch_size": 32}, n_folds=3)
