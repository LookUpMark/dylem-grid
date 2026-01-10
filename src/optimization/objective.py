"""Optuna objective with Lightning integration."""
import json
from pathlib import Path
from typing import Any, Optional, Type
import optuna
from optuna import Trial
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from src.data.datamodule import GestureDataModule
from src.training.cross_validation import CrossValidator


class OptunaObjective:
    """Optuna objective with CV support."""
    
    def __init__(self, model_class: Type[LightningModule], datamodule: GestureDataModule,
                 search_space, use_cv: bool = False, n_folds: int = 5, max_epochs: int = 50,
                 patience: int = 10, monitor: str = "val_acc", trainer_kwargs: Optional[dict] = None):
        self.model_class, self.dm, self.space = model_class, datamodule, search_space
        self.use_cv, self.n_folds = use_cv, n_folds
        self.max_epochs, self.patience, self.monitor = max_epochs, patience, monitor
        self.trainer_kwargs = trainer_kwargs or {}

    def __call__(self, trial: Trial) -> float:
        params = self.space.sample(trial)
        if "batch_size" in params: self.dm.batch_size = params.pop("batch_size")
        return self._run_cv(trial, params) if self.use_cv else self._run_single(trial, params)

    def _run_single(self, trial: Trial, params: dict) -> float:
        self.dm.setup()
        kwargs = {"input_size": self.dm.input_size, "num_classes": self.dm.num_classes, **params}
        model = self.model_class(**kwargs)
        trial.set_user_attr("num_params", sum(p.numel() for p in model.parameters()))
        
        callbacks = [PyTorchLightningPruningCallback(trial, self.monitor),
                     EarlyStopping(self.monitor, mode="max", patience=self.patience, verbose=False)]
        trainer = Trainer(max_epochs=self.max_epochs, callbacks=callbacks, logger=False,
                          enable_progress_bar=False, enable_model_summary=False, **self.trainer_kwargs)
        trainer.fit(model, self.dm)
        return trainer.callback_metrics.get(self.monitor, 0.0).item()

    def _run_cv(self, trial: Trial, params: dict) -> float:
        cv = CrossValidator(self.model_class, self.dm, self.n_folds,
                           {"max_epochs": self.max_epochs, **self.trainer_kwargs})
        results = cv.run(params, self.patience, verbose=False)
        trial.set_user_attr("cv_results", results.to_dict())
        trial.set_user_attr("cv_std", results.std_accuracy)
        return results.mean_accuracy


def create_study(name: str, direction: str = "maximize", storage: Optional[str] = None,
                 pruner_type: str = "median") -> optuna.Study:
    """Create Optuna study with pruner."""
    pruner = {"median": optuna.pruners.MedianPruner(n_startup_trials=5),
              "hyperband": optuna.pruners.HyperbandPruner(),
              "none": optuna.pruners.NopPruner()}.get(pruner_type, optuna.pruners.NopPruner())
    return optuna.create_study(study_name=name, direction=direction, storage=storage,
                               load_if_exists=True, pruner=pruner)


def run_optimization(model_class, datamodule, search_space, n_trials: int = 100, use_cv: bool = False,
                     n_folds: int = 5, save_dir: str = "results/optuna") -> tuple:
    """Run optimization and save results."""
    datamodule.setup()
    study = create_study(search_space.name)
    objective = OptunaObjective(model_class, datamodule, search_space, use_cv, n_folds)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Save results
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / f"{search_space.name}_results.json", "w") as f:
        json.dump({"best": study.best_trial.value, "params": study.best_trial.params}, f, indent=2)
    
    print(f"\nBest: {study.best_trial.value:.4f}\nParams: {study.best_trial.params}")
    return study, study.best_trial.params
