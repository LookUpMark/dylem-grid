"""Base LightningModule for gesture recognition."""
from __future__ import annotations
from typing import Any, Literal
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, Precision, Recall


class GestureBaseModule(LightningModule):
    """Base module with shared training logic. Subclasses implement forward()."""
    
    def __init__(self, num_classes: int = 4, learning_rate: float = 1e-3, weight_decay: float = 1e-5,
                 optimizer: Literal["adam", "adamw", "nadam", "sgd"] = "nadam",
                 scheduler: Literal["none", "cosine", "plateau"] = "none", scheduler_patience: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_preds, self.val_targets = [], []

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.train_acc(out.argmax(1), y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        preds = out.argmax(1)
        self.val_acc(preds, y); self.val_f1(preds, y)
        self.val_precision(preds, y); self.val_recall(preds, y)
        self.val_preds.append(preds.cpu()); self.val_targets.append(y.cpu())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_recall", self.val_recall, on_epoch=True)
        return loss

    def on_validation_epoch_start(self): self.val_preds, self.val_targets = [], []
    
    def on_validation_epoch_end(self):
        if self.val_preds:
            self.last_val_preds = torch.cat(self.val_preds)
            self.last_val_targets = torch.cat(self.val_targets)

    def configure_optimizers(self):
        opt_map = {"adam": Adam, "adamw": AdamW, "sgd": SGD,
                   "nadam": lambda p, **k: torch.optim.NAdam(p, **k)}
        kw = {"lr": self.hparams.learning_rate, "weight_decay": self.hparams.weight_decay}
        if self.hparams.optimizer == "sgd": kw["momentum"] = 0.9
        opt = opt_map[self.hparams.optimizer](self.parameters(), **kw)
        
        if self.hparams.scheduler == "none": return {"optimizer": opt}
        if self.hparams.scheduler == "cosine":
            return {"optimizer": opt, "lr_scheduler": {"scheduler": CosineAnnealingLR(opt, T_max=50), "interval": "epoch"}}
        if self.hparams.scheduler == "plateau":
            return {"optimizer": opt, "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=self.hparams.scheduler_patience),
                "monitor": "val_acc", "interval": "epoch"}}
        return {"optimizer": opt}
