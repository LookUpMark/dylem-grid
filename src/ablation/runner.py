"""Ablation study runner."""
import json, time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from src.ablation.config import AblationConfig
from src.data.datamodule import GestureDataModule
from src.models.bilstm import BiLSTMModule
from src.models.transformer import TransformerModule
from src.training.callbacks import BestModelCallback
from src.training.cross_validation import CrossValidator


@dataclass
class AblationResult:
    config_name: str; ablation_param: str; ablation_value: Any
    mean_accuracy: float; std_accuracy: float; mean_f1: float; std_f1: float
    training_time: float = 0.0
    def to_dict(self): return self.__dict__


@dataclass
class AblationResults:
    config: AblationConfig
    results: List[AblationResult] = field(default_factory=list)
    start_time: str = ""; end_time: str = ""

    def add_result(self, r): self.results.append(r)
    def to_dataframe(self): return pd.DataFrame([r.to_dict() for r in self.results])
    
    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path)/f"{self.config.name}_results.json", "w") as f:
            json.dump({"config": self.config.name, "results": [r.to_dict() for r in self.results]}, f, indent=2, default=str)
        self.to_dataframe().to_csv(Path(path)/f"{self.config.name}.csv", index=False)

    def plot(self, path: Optional[str] = None):
        df = self.to_dataframe()
        params = df["ablation_param"].unique()
        n = len(params); cols = min(3, n); rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        axes = [axes] if n == 1 else axes.flatten()
        
        for i, param in enumerate(params):
            p = df[df["ablation_param"] == param].sort_values("ablation_value", key=lambda x: pd.to_numeric(x, errors='coerce'))
            axes[i].bar(range(len(p)), p["mean_accuracy"], yerr=p["std_accuracy"], capsize=5)
            axes[i].set_xticks(range(len(p))); axes[i].set_xticklabels(p["ablation_value"].astype(str), rotation=45)
            axes[i].set_title(param); axes[i].set_ylim(0, 1.05)
        
        for i in range(n, len(axes)): axes[i].set_visible(False)
        plt.tight_layout()
        if path: fig.savefig(Path(path)/f"{self.config.name}.png", dpi=150); plt.close()
        else: plt.show()

    def print_summary(self):
        print(f"\n{'='*60}\nABLATION: {self.config.name} ({len(self.results)} configs)")
        best = max(self.results, key=lambda r: r.mean_accuracy)
        print(f"Best: {best.config_name} -> {best.mean_accuracy:.4f} ± {best.std_accuracy:.4f}\n{'='*60}")


class AblationRunner:
    """Run ablation study."""
    MODELS = {"bilstm": BiLSTMModule, "transformer": TransformerModule}

    def __init__(self, config: AblationConfig, datamodule: GestureDataModule, trainer_kwargs: Optional[dict] = None):
        self.config, self.dm = config, datamodule
        self.trainer_kwargs = trainer_kwargs or {}
        self.model_class = self.MODELS[config.model]

    def run(self, verbose: bool = True) -> AblationResults:
        results = AblationResults(config=self.config, start_time=datetime.now().isoformat())
        configs = self.config.get_all_configurations()
        if verbose: print(f"\nAblation: {self.config.name} ({len(configs)} configs)\n{'='*50}")
        
        for i, (name, param, val, params) in enumerate(configs):
            if verbose: print(f"[{i+1}/{len(configs)}] {name}")
            t0 = time.time()
            r = self._run_config(name, param, val, params)
            r.training_time = time.time() - t0
            results.add_result(r)
            if verbose: print(f"    -> {r.mean_accuracy:.4f} ± {r.std_accuracy:.4f}")
        
        results.end_time = datetime.now().isoformat()
        if verbose: results.print_summary()
        return results

    def _run_config(self, name, param, val, params) -> AblationResult:
        self.dm.batch_size = self.config.training_params.get("batch_size", 32)
        if self.config.use_cv:
            cv = CrossValidator(self.model_class, self.dm, self.config.n_folds,
                               {"max_epochs": self.config.training_params.get("max_epochs", 50), **self.trainer_kwargs})
            r = cv.run(params, self.config.training_params.get("patience", 10), verbose=False)
            return AblationResult(name, param, val, r.mean_accuracy, r.std_accuracy, r.mean_f1, r.std_f1)
        
        self.dm.setup()
        model = self.model_class(input_size=self.dm.input_size, num_classes=self.dm.num_classes, **params)
        trainer = Trainer(max_epochs=self.config.training_params.get("max_epochs", 50),
                         callbacks=[BestModelCallback(), EarlyStopping("val_acc", mode="max", patience=10, verbose=False)],
                         logger=False, enable_progress_bar=False, enable_model_summary=False, **self.trainer_kwargs)
        trainer.fit(model, self.dm)
        m = trainer.callback_metrics
        return AblationResult(name, param, val, m.get("val_acc", 0).item(), 0, m.get("val_f1", 0).item(), 0)
