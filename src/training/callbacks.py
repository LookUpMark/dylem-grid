"""Lightning callbacks for training."""
import json
from pathlib import Path
from typing import Any, Optional
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class BestModelCallback(Callback):
    """Track best model during training."""
    def __init__(self, monitor: str = "val_acc", mode: str = "max"):
        self.monitor, self.mode = monitor, mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.best_preds = self.best_targets = None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None: return
        is_better = current > self.best_score if self.mode == "max" else current < self.best_score
        if is_better:
            self.best_score, self.best_epoch = current.item(), trainer.current_epoch + 1
            if hasattr(pl_module, "last_val_preds"):
                self.best_preds = pl_module.last_val_preds.clone()
                self.best_targets = pl_module.last_val_targets.clone()


class MetricsLoggerCallback(Callback):
    """Log metrics to JSON and optionally W&B."""
    def __init__(self, log_dir: str = "logs", experiment_name: str = "gesture", use_wandb: bool = True):
        self.log_dir, self.name, self.use_wandb = Path(log_dir), experiment_name, use_wandb
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.wandb = None
        if use_wandb:
            try: import wandb; self.wandb = wandb
            except: pass

    def on_train_epoch_end(self, trainer, pl_module):
        for k in ["train_loss", "train_acc"]:
            if k in trainer.callback_metrics: self.history[k].append(trainer.callback_metrics[k].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        for k in ["val_loss", "val_acc"]:
            if k in trainer.callback_metrics: self.history[k].append(trainer.callback_metrics[k].item())
        if self.wandb and self.wandb.run:
            self.wandb.log({"epoch": trainer.current_epoch + 1, **{k: v[-1] for k, v in self.history.items() if v}})

    def on_train_end(self, trainer, pl_module):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / f"{self.name}_history.json", "w") as f: json.dump(self.history, f, indent=2)


def get_default_callbacks(log_dir="logs", name="gesture", monitor="val_acc", use_wandb=True):
    """Get standard callback set."""
    return [
        BestModelCallback(monitor, "max"),
        MetricsLoggerCallback(log_dir, name, use_wandb),
        ModelCheckpoint(dirpath=f"{log_dir}/checkpoints", filename=f"{name}_{{epoch:02d}}_{{val_acc:.4f}}",
                        monitor=monitor, mode="max", save_top_k=1, save_last=True)]
