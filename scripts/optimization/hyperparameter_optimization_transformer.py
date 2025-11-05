"""
Hyperparameter Optimization for Transformer using Optuna
Modern Bayesian optimization with automatic pruning

Install: pip install optuna optuna-dashboard
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from utils.data_processing import data_loader, data_preprocess, apply_pca, prepare_data
from models.architectures.transformer_model import GestureTransformer, evaluate_model
from utils.display import (
    print_header,
    print_section,
    print_subsection,
    print_success,
    print_info,
    print_metric,
    print_warning,
    print_separator
)


def set_seed(seed=44):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def objective(trial, X, y, device):
    """
    Optuna objective function to optimize

    Args:
        trial: Optuna trial object
        X: Input data
        y: Labels
        device: Torch device

    Returns:
        Best validation accuracy
    """

    # Hyperparameters to optimize
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "NAdam"])

    # Ensure d_model is divisible by nhead
    if d_model % nhead != 0:
        return 0.0  # Invalid configuration

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=44, stratify=y
    )

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    input_size = X.shape[2]
    num_classes = len(torch.unique(y))

    model = GestureTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("num_params", num_params)

    # Initialize optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:  # NAdam
        optimizer = optim.NAdam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    criterion = nn.CrossEntropyLoss()

    # Training loop with pruning
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10

    for epoch in range(30):
        # Training phase
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation phase
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        # Log to W&B if available
        if WANDB_AVAILABLE:
            wandb.log({
                'trial': trial.number,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': learning_rate,
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dropout': dropout,
                'batch_size': batch_size,
            })

        # Report intermediate value for pruning
        trial.report(val_acc, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            break

    return best_val_acc


def main():
    """Main function for hyperparameter optimization with Optuna"""

    print_header("Transformer Hyperparameter Optimization — Optuna")

    # Initialize W&B for optimization tracking
    if WANDB_AVAILABLE:
        wandb.init(
            project="dylem-grid-gestures",
            name="transformer-optuna-optimization",
            config={
                "optimization_method": "optuna-tpe",
                "n_trials": 100,
                "pruner": "median",
                "objective": "maximize_val_accuracy"
            },
            tags=["optimization", "transformer", "optuna"]
        )
        print_info("Weights & Biases initialized")

    # Set random seed
    set_seed(44)
    print_success("Random seeds initialized (seed=44)")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Device: {device}")

    # Load and preprocess data
    print_section("Data Loading & Preprocessing")

    data, labels = data_loader("data/DYLEM-GRID", "Raw")
    print_success(f"Loaded {len(data)} samples")

    data, labels = data_preprocess(data, labels)
    print_success("Data preprocessed")

    data, labels = apply_pca(data, labels, variance_threshold=0.95)
    print_success(f"PCA applied → shape: {data[0].shape}")

    X, y, label_encoder = prepare_data(data, labels)
    print_success(f"Data prepared → X: {X.shape}, y: {y.shape}")
    print_info(f"Classes ({len(label_encoder.classes_)}): {', '.join(list(label_encoder.classes_))}")

    # Create Optuna study
    print_section("Optuna Configuration")

    print_subsection("Optimization Settings")
    print_info("Algorithm: TPE (Tree-structured Parzen Estimator)")
    print_info("Pruner: Median Pruner (early stopping for unpromising trials)")
    print_info("Trials: 100")
    print_info("Objective: Maximize validation accuracy")

    # Create study with pruning
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=1
        ),
        study_name="transformer_optimization",
    )

    print_section("Running Optimization")
    print_info("Starting Bayesian optimization with Optuna...")
    print_separator()

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, device),
        n_trials=100,  # Number of trials to run
        timeout=None,
        show_progress_bar=True,
    )

    print_separator()

    # Find best trial considering accuracy first, then parameter count
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print_warning("No completed trials found!")
        return

    # Sort by accuracy (descending), then by parameter count (ascending)
    best_accuracy = max(t.value for t in completed_trials)
    best_trials_at_max_acc = [
        t for t in completed_trials if abs(t.value - best_accuracy) < 1e-6
    ]

    # Among trials with best accuracy, choose the one with fewest parameters
    best_trial = min(
        best_trials_at_max_acc,
        key=lambda t: t.user_attrs.get("num_params", float("inf")),
    )

    # Print results
    print_section("Optimization Results")

    print_subsection("Best Trial")
    print_metric("Trial number", best_trial.number)
    print_metric("Validation accuracy", f"{best_trial.value:.4f} ({best_trial.value*100:.2f}%)")
    print_metric("Model parameters", f"{best_trial.user_attrs.get('num_params', 'N/A'):,}")

    if len(best_trials_at_max_acc) > 1:
        print()
        print_info(f"Note: {len(best_trials_at_max_acc)} trials achieved the same accuracy.")
        print_info(f"Selected trial {best_trial.number} with fewest parameters.")

    print_subsection("Best Hyperparameters")
    for key, value in best_trial.params.items():
        print_info(f"{key}: {value}")

    print_subsection("Statistics")
    print_info(f"Total trials: {len(study.trials)}")
    print_info(f"Completed: {len(completed_trials)}")
    print_info(f"Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    # Save results
    print_section("Saving Results")

    # Save study to a results directory (not in repo root)
    import joblib
    import json
    from datetime import datetime
    import os

    results_dir = os.path.join('results', 'optuna', 'transformer')
    os.makedirs(results_dir, exist_ok=True)

    study_path = os.path.join(results_dir, 'transformer_optuna_study.pkl')
    joblib.dump(study, study_path)
    print_success(f"Study saved to {study_path}")

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_trial": best_trial.number,
        "best_score": float(best_trial.value),
        "best_params": best_trial.params,
        "best_num_params": best_trial.user_attrs.get("num_params", None),
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "all_trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "num_params": trial.user_attrs.get("num_params", None),
            }
            for trial in completed_trials
        ],
    }

    json_path = os.path.join(results_dir, 'transformer_optuna_results.json')
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print_success(f"Results saved to {json_path}")

    # Generate visualizations
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        print_section("Generating Visualizations")

        # 1. Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(results_dir, 'transformer_optimization_history.png'))
        print_success(f"Optimization history saved to {results_dir}")

        # 2. Parameter importances
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(results_dir, 'transformer_param_importances.png'))
        print_success(f"Parameter importances saved to {results_dir}")

        # 3. Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_image(os.path.join(results_dir, 'transformer_parallel_coordinate.png'))
        print_success(f"Parallel coordinate plot saved to {results_dir}")

        # 4. Slice plot
        fig = plot_slice(study)
        fig.write_image(os.path.join(results_dir, 'transformer_slice.png'))
        print_success(f"Slice plot saved to {results_dir}")

        print_info(f"All visualizations saved in {results_dir}")

        # Log visualizations to W&B
        if WANDB_AVAILABLE:
            wandb.log({
                "optimization/history": wandb.Image(os.path.join(results_dir, 'transformer_optimization_history.png')),
                "optimization/param_importances": wandb.Image(os.path.join(results_dir, 'transformer_param_importances.png')),
                "optimization/parallel_coordinate": wandb.Image(os.path.join(results_dir, 'transformer_parallel_coordinate.png')),
                "optimization/slice": wandb.Image(os.path.join(results_dir, 'transformer_slice.png')),
            })

    except (ImportError, Exception) as e:
        print_section("Visualization Generation")
        print_warning("Visualization generation skipped")
        print_info(f"Reason: {type(e).__name__}")
    
    # Log best parameters to W&B
    if WANDB_AVAILABLE:
        wandb.log({
            "best/trial_number": best_trial.number,
            "best/val_accuracy": best_trial.value,
            "best/num_params": best_trial.user_attrs.get('num_params', 0),
        })
        wandb.config.update({"best_params": best_trial.params})
        wandb.finish()
        print_info("Results logged to Weights & Biases")
    
    print_info(f"Results saved in {json_path} and {study_path}")
    print_info(f"View results: optuna-dashboard {study_path}")

    # Final summary
    print_header("OPTIMIZATION COMPLETED!", '=')
    print_subsection("Next Steps")
    print_info("1. Update train_transformer.py with best hyperparameters")
    print_info("2. Train final model on full training set")
    print_info("3. Evaluate on test set")
    print()
    print_subsection("Advanced Options")
    print_info(f"View interactive dashboard: optuna-dashboard {study_path}")
    print_info(f"Load study: study = joblib.load('{study_path}')")
    print()


if __name__ == "__main__":
    main()
