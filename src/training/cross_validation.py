"""K-Fold Cross-Validation for Lightning models."""
from dataclasses import dataclass, field
from typing import Any, Optional, Type
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from src.data.datamodule import GestureDataModule
from src.training.callbacks import BestModelCallback


@dataclass
class CVFoldResult:
    fold: int; val_acc: float; val_loss: float; val_f1: float
    val_precision: float; val_recall: float; epochs_trained: int; best_epoch: int
    def to_dict(self): return self.__dict__


@dataclass
class CVResults:
    fold_results: list = field(default_factory=list)
    
    @property
    def n_folds(self): return len(self.fold_results)
    @property
    def mean_accuracy(self): return np.mean([r.val_acc for r in self.fold_results])
    @property
    def std_accuracy(self): return np.std([r.val_acc for r in self.fold_results])
    @property
    def mean_f1(self): return np.mean([r.val_f1 for r in self.fold_results])
    @property
    def std_f1(self): return np.std([r.val_f1 for r in self.fold_results])
    
    def add_fold(self, result): self.fold_results.append(result)
    
    def to_dict(self):
        acc = [r.val_acc for r in self.fold_results]
        f1 = [r.val_f1 for r in self.fold_results]
        return {"n_folds": self.n_folds,
                "accuracy": {"mean": float(np.mean(acc)), "std": float(np.std(acc)), "values": acc},
                "f1": {"mean": float(np.mean(f1)), "std": float(np.std(f1)), "values": f1},
                "fold_details": [r.to_dict() for r in self.fold_results]}
    
    def print_summary(self):
        print(f"\n{'='*60}\nCROSS-VALIDATION: {self.n_folds} folds")
        print(f"Accuracy: {self.mean_accuracy:.4f} ± {self.std_accuracy:.4f}")
        print(f"F1 Score: {self.mean_f1:.4f} ± {self.std_f1:.4f}")
        for r in self.fold_results: print(f"  Fold {r.fold+1}: Acc={r.val_acc:.4f}")
        print("="*60)


class CrossValidator:
    """K-Fold CV wrapper for Lightning models."""
    
    def __init__(self, model_class: Type[LightningModule], datamodule: GestureDataModule,
                 n_folds: int = 5, trainer_kwargs: Optional[dict] = None):
        self.model_class, self.dm, self.n_folds = model_class, datamodule, n_folds
        self.trainer_kwargs = trainer_kwargs or {}

    def run(self, model_kwargs: dict, patience: int = 10, verbose: bool = True) -> CVResults:
        results = CVResults()
        if verbose: print(f"\n{self.n_folds}-Fold Cross-Validation\n{'='*50}")
        
        for fold in range(self.n_folds):
            if verbose: print(f"\n--- Fold {fold+1}/{self.n_folds} ---")
            result = self._train_fold(fold, model_kwargs, patience)
            results.add_fold(result)
            if verbose: print(f"Acc: {result.val_acc:.4f}, F1: {result.val_f1:.4f}")
        
        if verbose: results.print_summary()
        return results

    def _train_fold(self, fold: int, model_kwargs: dict, patience: int) -> CVFoldResult:
        self.dm.cv_fold, self.dm.n_folds = fold, self.n_folds
        self.dm.setup()
        
        kwargs = {"input_size": self.dm.input_size, "num_classes": self.dm.num_classes, **model_kwargs}
        model = self.model_class(**kwargs)
        
        best_cb = BestModelCallback("val_acc", "max")
        early_cb = EarlyStopping("val_acc", mode="max", patience=patience, verbose=False)
        
        trainer_args = {
            "max_epochs": 50,
            "callbacks": [best_cb, early_cb],
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
        trainer_args.update(self.trainer_kwargs)
        
        trainer = Trainer(**trainer_args)
        trainer.fit(model, self.dm)
        
        m = trainer.callback_metrics
        return CVFoldResult(fold=fold, val_acc=m.get("val_acc", torch.tensor(0)).item(),
                           val_loss=m.get("val_loss", torch.tensor(0)).item(),
                           val_f1=m.get("val_f1", torch.tensor(0)).item(),
                           val_precision=m.get("val_precision", torch.tensor(0)).item(),
                           val_recall=m.get("val_recall", torch.tensor(0)).item(),
                           epochs_trained=trainer.current_epoch + 1, best_epoch=best_cb.best_epoch)
