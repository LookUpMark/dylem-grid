"""Visualization helpers for notebooks."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Any, Optional


def plot_cv_comparison(results_dict: Dict[str, Any], title: str = "Cross-Validation Comparison"):
    """Plot CV results for multiple models side by side."""
    models = list(results_dict.keys())
    n = len(models)
    
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    
    # Per-model fold results
    for i, (name, results) in enumerate(results_dict.items()):
        accs = [r.val_acc for r in results.fold_results]
        axes[i].bar(range(1, len(accs) + 1), accs, color='steelblue', alpha=0.8)
        axes[i].axhline(results.mean_accuracy, color='red', linestyle='--', 
                        label=f'Mean: {results.mean_accuracy:.4f}')
        axes[i].fill_between([0.5, len(accs) + 0.5], 
                             results.mean_accuracy - results.std_accuracy,
                             results.mean_accuracy + results.std_accuracy,
                             alpha=0.2, color='red')
        axes[i].set_xlabel('Fold')
        axes[i].set_ylabel('Accuracy')
        axes[i].set_title(f'{name.upper()}')
        axes[i].set_ylim(0, 1.05)
        axes[i].legend()
    
    # Comparison bar
    means = [r.mean_accuracy for r in results_dict.values()]
    stds = [r.std_accuracy for r in results_dict.values()]
    bars = axes[-1].bar(models, means, yerr=stds, capsize=10, color=['steelblue', 'coral'][:n])
    axes[-1].set_ylabel('Accuracy')
    axes[-1].set_title('Comparison')
    axes[-1].set_ylim(0, 1.05)
    for bar, m in zip(bars, means):
        axes[-1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{m:.4f}', ha='center')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_confusion_matrices(results_dict: Dict[str, tuple], class_names: List[str]):
    """Plot confusion matrices for multiple models."""
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1: axes = [axes]
    
    for ax, (name, (preds, targets)) in zip(axes, results_dict.items()):
        cm = confusion_matrix(targets, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        acc = (np.array(preds) == np.array(targets)).mean()
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{name.upper()} (Acc: {acc:.2%})')
    
    plt.tight_layout()
    return fig


def plot_training_comparison(histories: Dict[str, Dict[str, List[float]]]):
    """Plot training curves for multiple models."""
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 8))
    if n == 1: axes = axes.reshape(-1, 1)
    
    for i, (name, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist.get('train_loss', [])) + 1)
        
        # Loss
        if 'train_loss' in hist:
            axes[0, i].plot(epochs, hist['train_loss'], label='Train', color='steelblue')
        if 'val_loss' in hist:
            axes[0, i].plot(epochs, hist['val_loss'], label='Val', color='coral')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].set_title(f'{name.upper()} - Loss')
        axes[0, i].legend()
        
        # Accuracy
        if 'train_acc' in hist:
            axes[1, i].plot(epochs, hist['train_acc'], label='Train', color='steelblue')
        if 'val_acc' in hist:
            axes[1, i].plot(epochs, hist['val_acc'], label='Val', color='coral')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy')
        axes[1, i].set_title(f'{name.upper()} - Accuracy')
        axes[1, i].legend()
        axes[1, i].set_ylim(0, 1.05)
    
    plt.tight_layout()
    return fig


def plot_optuna_comparison(studies: Dict[str, Any]):
    """Plot Optuna optimization results for multiple models."""
    n = len(studies)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 10))
    if n == 1: axes = axes.reshape(-1, 1)
    
    for i, (name, study) in enumerate(studies.items()):
        trials = [t for t in study.trials if t.state.name == 'COMPLETE']
        values = [t.value for t in trials]
        
        # Optimization history
        axes[0, i].plot(range(len(values)), values, 'o-', alpha=0.5, markersize=4)
        best_vals = [max(values[:j+1]) for j in range(len(values))]
        axes[0, i].plot(range(len(best_vals)), best_vals, 'r-', linewidth=2, label='Best')
        axes[0, i].set_xlabel('Trial')
        axes[0, i].set_ylabel('Accuracy')
        axes[0, i].set_title(f'{name.upper()} - Optimization History')
        axes[0, i].legend()
        
        # Best params
        params = study.best_trial.params
        param_names = list(params.keys())[:6]  # Top 6
        param_vals = [str(params[p])[:10] for p in param_names]
        axes[1, i].barh(param_names, [1] * len(param_names), color='steelblue', alpha=0.3)
        for j, (pn, pv) in enumerate(zip(param_names, param_vals)):
            axes[1, i].text(0.5, j, f'{pv}', ha='center', va='center', fontsize=10)
        axes[1, i].set_title(f'{name.upper()} Best: {study.best_value:.4f}')
        axes[1, i].set_xlim(0, 1)
    
    plt.tight_layout()
    return fig


def plot_ablation_comparison(results_dict: Dict[str, Any]):
    """Plot ablation results for multiple models."""
    fig, axes = plt.subplots(1, len(results_dict), figsize=(8 * len(results_dict), 6))
    if len(results_dict) == 1: axes = [axes]
    
    for ax, (name, results) in zip(axes, results_dict.items()):
        df = results.to_dataframe()
        params = df['ablation_param'].unique()
        
        x_pos = 0
        x_ticks, x_labels = [], []
        colors = plt.cm.tab10(np.linspace(0, 1, len(params)))
        
        for j, param in enumerate(params):
            p_df = df[df['ablation_param'] == param]
            for _, row in p_df.iterrows():
                ax.bar(x_pos, row['mean_accuracy'], yerr=row['std_accuracy'], 
                       color=colors[j], capsize=3, alpha=0.8)
                x_ticks.append(x_pos)
                x_labels.append(f"{row['ablation_value']}")
                x_pos += 1
            x_pos += 0.5  # Gap between params
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{name.upper()} Ablation')
        ax.set_ylim(0, 1.05)
        
        # Legend
        handles = [plt.Rectangle((0,0),1,1, color=colors[j]) for j in range(len(params))]
        ax.legend(handles, params, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    return fig


def print_results_table(results_dict: Dict[str, Any]):
    """Print formatted results table."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Accuracy':<15} {'F1':<15} {'Epochs'}")
    print("-" * 60)
    for name, r in results_dict.items():
        if hasattr(r, 'mean_accuracy'):
            print(f"{name.upper():<15} {r.mean_accuracy:.4f} ± {r.std_accuracy:.4f}  "
                  f"{r.mean_f1:.4f} ± {r.std_f1:.4f}  {r.n_folds} folds")
        else:
            print(f"{name.upper():<15} {r['accuracy']:.4f}  {r.get('f1', 'N/A')}")
    print("=" * 60)


def save_all_figures(figs: Dict[str, plt.Figure], save_dir: str = "plots"):
    """Save all figures to directory."""
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        fig.savefig(f"{save_dir}/{name}.png", dpi=150, bbox_inches='tight')
    print(f"Saved {len(figs)} figures to {save_dir}/")
