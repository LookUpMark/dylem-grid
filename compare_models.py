"""
Compare BiLSTM and Transformer models
Loads both trained models and shows their performance metrics
"""

import torch
import json
import os
from src.bilstm_model import GestureRNN
from src.transformer_model import GestureTransformer


def load_model_stats(model_path):
    """Load model and extract statistics"""
    if not os.path.exists(model_path):
        return None

    checkpoint = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=False
    )
    return checkpoint.get("best_stats", None)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 80)
    print("Model Comparison — BiLSTM vs Transformer")
    print("=" * 80)
    print("\n" * 3)

    # Check which models exist
    rnn_exists = os.path.exists("gesture_rnn_model.pth")
    transformer_exists = os.path.exists("gesture_transformer_model.pth")

    if not rnn_exists and not transformer_exists:
        print("\nNo trained models found!")
        print("Train models first:")
        print("  python train_bilstm.py")
        print("  python train_transformer.py")
        return

    print("MODEL AVAILABILITY")
    print("\n" * 2)
    print(f"BiLSTM Model:      {'✓ Available' if rnn_exists else '✗ Not found'}")
    print(
        f"Transformer Model: {'✓ Available' if transformer_exists else '✗ Not found'}"
    )
    print("\n" * 3)

    # Load and compare models
    if rnn_exists:
        print("BiLSTM + Attention Model")
        print("-" * 70)
        print("\n" * 2)

        checkpoint = torch.load(
            "gesture_rnn_model.pth", map_location="cpu", weights_only=False
        )

        # Reconstruct model to count parameters
        model = GestureRNN(
            checkpoint["input_size"],
            checkpoint["hidden_size"],
            checkpoint["num_layers"],
            checkpoint["num_classes"],
        )

        stats = checkpoint.get("best_stats", {})

        print(f"Architecture:")
        print(f"  Input size:    {checkpoint['input_size']}")
        print(f"  Hidden size:   {checkpoint['hidden_size']}")
        print(f"  Num layers:    {checkpoint['num_layers']}")
        print(f"  Num classes:   {checkpoint['num_classes']}")
        print(f"  Parameters:    {count_parameters(model):,}")

        if stats:
            print("\n" * 2)
            print("Performance:")
            print(f"  Best epoch:    {stats['epoch']}")
            print(
                f"  Val accuracy:  {stats['val_acc']:.4f} ({stats['val_acc'] * 100:.2f}%)"
            )
            print(f"  Val loss:      {stats['val_loss']:.6f}")

    if transformer_exists:
        print("\n" * 3)
        print("Encoder-only Transformer Model")
        print("-" * 70)
        print("\n" * 2)

        checkpoint = torch.load(
            "gesture_transformer_model.pth", map_location="cpu", weights_only=False
        )

        # Reconstruct model to count parameters
        model = GestureTransformer(
            checkpoint["input_size"],
            checkpoint["d_model"],
            checkpoint["nhead"],
            checkpoint["num_layers"],
            checkpoint["dim_feedforward"],
            checkpoint["num_classes"],
            checkpoint.get("dropout", 0.1),
        )

        stats = checkpoint.get("best_stats", {})

        print(f"Architecture:")
        print(f"  Input size:    {checkpoint['input_size']}")
        print(f"  Model dim:     {checkpoint['d_model']}")
        print(f"  Attention heads: {checkpoint['nhead']}")
        print(f"  Num layers:    {checkpoint['num_layers']}")
        print(f"  FF dimension:  {checkpoint['dim_feedforward']}")
        print(f"  Num classes:   {checkpoint['num_classes']}")
        print(f"  Parameters:    {count_parameters(model):,}")

        if stats:
            print("\n" * 2)
            print("Performance:")
            print(f"  Best epoch:    {stats['epoch']}")
            print(
                f"  Val accuracy:  {stats['val_acc']:.4f} ({stats['val_acc'] * 100:.2f}%)"
            )
            print(f"  Val loss:      {stats['val_loss']:.6f}")

    print("\n" * 3)
    print("Hyperparameter Optimization Results")
    print("-" * 70)
    print("\n" * 2)

    rnn_optuna = "optuna_results.json"
    transformer_optuna = "transformer_optuna_results.json"

    if os.path.exists(rnn_optuna):
        with open(rnn_optuna, "r") as f:
            results = json.load(f)
        print("BiLSTM Optuna:")
        print(f"  Best trial:    {results['best_trial']}")
        print(f"  Best accuracy: {results['best_value']:.4f}")
    else:
        print("BiLSTM Optuna:   Not found (run hyperparameter_optimization_bilstm.py)")

    if os.path.exists(transformer_optuna):
        with open(transformer_optuna, "r") as f:
            results = json.load(f)
        print("\n" * 2)
        print("Transformer Optuna:")
        print(f"  Best trial:    {results['best_trial']}")
        print(f"  Best accuracy: {results['best_value']:.4f}")
    else:
        print("\n" * 2)
        print(
            "Transformer Optuna: Not found (run hyperparameter_optimization_transformer.py)"
        )

    print("\n" * 3)

    # Comparison summary
    if rnn_exists and transformer_exists:
        print("Quick Comparison")
        print("=" * 70)
        print("\n" * 2)

        rnn_checkpoint = torch.load(
            "gesture_rnn_model.pth", map_location="cpu", weights_only=False
        )
        trans_checkpoint = torch.load(
            "gesture_transformer_model.pth", map_location="cpu", weights_only=False
        )

        rnn_model = GestureRNN(
            rnn_checkpoint["input_size"],
            rnn_checkpoint["hidden_size"],
            rnn_checkpoint["num_layers"],
            rnn_checkpoint["num_classes"],
        )

        trans_model = GestureTransformer(
            trans_checkpoint["input_size"],
            trans_checkpoint["d_model"],
            trans_checkpoint["nhead"],
            trans_checkpoint["num_layers"],
            trans_checkpoint["dim_feedforward"],
            trans_checkpoint["num_classes"],
            trans_checkpoint.get("dropout", 0.1),
        )

        rnn_params = count_parameters(rnn_model)
        trans_params = count_parameters(trans_model)

        rnn_stats = rnn_checkpoint.get("best_stats", {})
        trans_stats = trans_checkpoint.get("best_stats", {})

        print(f"\n{'Metric':<20} {'BiLSTM':>15} {'Transformer':>15}")
        print("-" * 70)
        print(f"{'Parameters':<20} {rnn_params:>15,} {trans_params:>15,}")

        if rnn_stats and trans_stats:
            rnn_acc = rnn_stats["val_acc"]
            trans_acc = trans_stats["val_acc"]
            rnn_loss = rnn_stats["val_loss"]
            trans_loss = trans_stats["val_loss"]

            print(f"{'Val Accuracy':<20} {rnn_acc:>15.4f} {trans_acc:>15.4f}")
            print(f"{'Val Loss':<20} {rnn_loss:>15.6f} {trans_loss:>15.6f}")
            print(
                f"{'Best Epoch':<20} {rnn_stats['epoch']:>15} {trans_stats['epoch']:>15}"
            )

            print("\n" * 2)
            print("-" * 70)
            if rnn_acc > trans_acc:
                print(
                    f"✓ BiLSTM achieved higher accuracy (+{(rnn_acc - trans_acc) * 100:.2f}%)"
                )
            elif trans_acc > rnn_acc:
                print(
                    f"✓ Transformer achieved higher accuracy (+{(trans_acc - rnn_acc) * 100:.2f}%)"
                )
            else:
                print("Both models achieved equal accuracy")

            if rnn_params < trans_params:
                print(
                    f"✓ BiLSTM is more parameter-efficient ({rnn_params:,} vs {trans_params:,})"
                )
            elif trans_params < rnn_params:
                print(
                    f"✓ Transformer is more parameter-efficient ({trans_params:,} vs {rnn_params:,})"
                )

    print("\n" * 3)
    print("=" * 70)


if __name__ == "__main__":
    main()
