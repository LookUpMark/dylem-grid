"""
Hyperparameter Optimization for Transformer using Optuna
Modern Bayesian optimization with automatic pruning

Install: pip install optuna optuna-dashboard
"""

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

from utils.data_processing import data_loader, data_preprocess, apply_pca, prepare_data
from models.architectures.transformer_model import GestureTransformer, evaluate_model


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

    print("Hyperparameter Optimization with Optuna (Bayesian Optimization)")
    print("\n" * 3)

    # Set random seed
    set_seed(44)
    print("Random seeds set (seed=44)")
    print("\n" * 3)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("\n" * 3)

    # Load and preprocess data
    print("LOADING AND PREPROCESSING DATA")
    print("\n" * 2)

    data, labels = data_loader("DYLEM-GRID", "Raw")
    print(f"Loaded {len(data)} samples")

    data, labels = data_preprocess(data, labels)
    print("Data preprocessed")

    data, labels = apply_pca(data, labels, variance_threshold=0.95)
    print(f"PCA applied, shape: {data[0].shape}")

    X, y, label_encoder = prepare_data(data, labels)
    print(f"Final data shape: {X.shape}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print("\n" * 3)

    # Create Optuna study
    print("RUNNING OPTUNA OPTIMIZATION")
    print("\n" * 2)

    print("Configuration:")
    print("  Algorithm: TPE (Tree-structured Parzen Estimator)")
    print("  Pruner: Median Pruner (stops unpromising trials early)")
    print("  Trials: 100 (configurable)")
    print("  Direction: Maximize validation accuracy")
    print("\n" * 3)

    # Create study with pruning
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=1
        ),
        study_name="transformer_optimization",
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, device),
        n_trials=100,  # Number of trials to run
        timeout=None,
        show_progress_bar=True,
    )

    # Find best trial considering accuracy first, then parameter count
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print("No completed trials found!")
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
    print("\n" * 3)
    print("OPTIMIZATION RESULTS")
    print("\n" * 2)

    print("BEST TRIAL:")
    print(f"  Trial number: {best_trial.number}")
    print(f"  Best validation accuracy: {best_trial.value:.4f}")
    print(f"  Number of parameters: {best_trial.user_attrs.get('num_params', 'N/A'):,}")

    if len(best_trials_at_max_acc) > 1:
        print("\n" * 2)
        print(f"Note: {len(best_trials_at_max_acc)} trials achieved the same accuracy.")
        print(
            f"Selected trial {best_trial.number} with fewest parameters ({best_trial.user_attrs.get('num_params', 'N/A'):,})."
        )

    print("\n" * 3)
    print("BEST HYPERPARAMETERS:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    print("\n" * 3)
    print("Statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len(completed_trials)}")
    print(
        f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )

    # Save results
    print("\n" * 3)
    print("SAVING RESULTS")
    print("\n" * 2)

    # Save study to database (can be loaded later)
    import joblib

    joblib.dump(study, "transformer_optuna_study.pkl")
    print("Study saved to 'transformer_optuna_study.pkl'")

    # Save best parameters to JSON
    import json
    from datetime import datetime

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

    with open("transformer_optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to 'transformer_optuna_results.json'")

    # Generate visualizations
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        print()
        print("GENERATING VISUALIZATIONS")
        print()

        # Create output directory
        import os

        os.makedirs("plots_output", exist_ok=True)

        # 1. Optimization history
        fig = plot_optimization_history(study)
        fig.write_image("plots_output/transformer_optimization_history.png")
        print("Optimization history saved")

        # 2. Parameter importances
        fig = plot_param_importances(study)
        fig.write_image("plots_output/transformer_param_importances.png")
        print("Parameter importances saved")

        # 3. Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_image("plots_output/transformer_parallel_coordinate.png")
        print("Parallel coordinate plot saved")

        # 4. Slice plot
        fig = plot_slice(study)
        fig.write_image("plots_output/transformer_slice.png")
        print("Slice plot saved")

        print()
        print("All visualizations saved in 'plots_output/' directory")

    except (ImportError, Exception) as e:
        print()
        print("Visualization generation skipped")
        print(f"Reason: {type(e).__name__}")
        print(
            "Results are saved in 'transformer_optuna_results.json' and 'transformer_optuna_study.pkl'"
        )
        print(
            "You can view results using: optuna-dashboard transformer_optuna_study.pkl"
        )

    # Final summary
    print("\n" * 3)
    print("=" * 80)
    print("OPTIMIZATION COMPLETED!")
    print("=" * 80)
    print("\n" * 2)
    print("Next Steps:")
    print("  1. Update train_transformer.py with best hyperparameters")
    print("  2. Train final model on full training set")
    print("  3. Evaluate on test set")
    print("\n" * 2)
    print("Advanced:")
    print("  View interactive dashboard: optuna-dashboard transformer_optuna_study.pkl")
    print("  Load study: study = joblib.load('transformer_optuna_study.pkl')")
    print()


if __name__ == "__main__":
    main()
