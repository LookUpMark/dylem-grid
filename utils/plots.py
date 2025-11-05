"""
Plotting functions for gesture recognition model visualization
Contains all plotting utilities for training history, confusion matrices, PCA analysis, and comprehensive metrics
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(train_losses, val_losses, train_accs, val_accs, best_stats, filename='training_history.png'):
    """
    Plot training and validation metrics with enhanced visualization

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        best_stats: Dictionary with best epoch statistics
        filename: Output filename for the plot
    """
    # Set style for better aesthetics
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))

    # Create custom color palette
    colors = {
        'train': '#2E86AB',      # Blue
        'val': '#A23B72',        # Purple
        'best': '#F18F01',       # Orange
        'highlight': '#C73E1D'   # Red
    }

    best_epoch = best_stats['epoch'] - 1  # Convert to 0-based indexing
    epochs = range(1, len(train_losses) + 1)

    # Create 2x2 subplot layout
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])  # Loss plot (top, spans both columns)
    ax2 = fig.add_subplot(gs[1, 0])  # Accuracy plot (bottom left)
    ax3 = fig.add_subplot(gs[1, 1])  # Metrics summary (bottom right)

    # Plot 1: Loss curves
    ax1.plot(epochs, train_losses, color=colors['train'], linewidth=2.5,
             label='Training Loss', alpha=0.8, marker='o', markersize=4, markevery=5)
    ax1.plot(epochs, val_losses, color=colors['val'], linewidth=2.5,
             label='Validation Loss', alpha=0.8, marker='s', markersize=4, markevery=5)

    # Highlight best epoch
    ax1.axvline(x=best_stats['epoch'], color=colors['highlight'], linestyle='--',
                linewidth=2, alpha=0.8, label=f'Best Epoch: {best_stats["epoch"]}')

    # Add best points with annotations
    ax1.plot(best_stats['epoch'], best_stats['train_loss'], 'o', color=colors['best'],
             markersize=10, markeredgecolor='black', markeredgewidth=1.5,
             label=f'Best Train Loss: {best_stats["train_loss"]:.4f}')
    ax1.plot(best_stats['epoch'], best_stats['val_loss'], 's', color=colors['best'],
             markersize=10, markeredgecolor='black', markeredgewidth=1.5,
             label=f'Best Val Loss: {best_stats["val_loss"]:.4f}')

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss Progression', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=min(min(train_losses), min(val_losses)) * 0.9)

    # Plot 2: Accuracy curves
    ax2.plot(epochs, train_accs, color=colors['train'], linewidth=2.5,
             label='Training Accuracy', alpha=0.8, marker='o', markersize=4, markevery=5)
    ax2.plot(epochs, val_accs, color=colors['val'], linewidth=2.5,
             label='Validation Accuracy', alpha=0.8, marker='s', markersize=4, markevery=5)

    # Highlight best epoch
    ax2.axvline(x=best_stats['epoch'], color=colors['highlight'], linestyle='--',
                linewidth=2, alpha=0.8, label=f'Best Epoch: {best_stats["epoch"]}')

    # Add best points with annotations
    ax2.plot(best_stats['epoch'], best_stats['train_acc'], 'o', color=colors['best'],
             markersize=10, markeredgecolor='black', markeredgewidth=1.5,
             label=f'Best Train Acc: {best_stats["train_acc"]:.4f}')
    ax2.plot(best_stats['epoch'], best_stats['val_acc'], 's', color=colors['best'],
             markersize=10, markeredgecolor='black', markeredgewidth=1.5,
             label=f'Best Val Acc: {best_stats["val_acc"]:.4f}')

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy Progression', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Plot 3: Metrics Summary Table
    ax3.axis('off')

    # Create summary data
    metrics_data = [
        ['Metric', 'Training', 'Validation', 'Improvement'],
        ['Loss', f'{best_stats["train_loss"]:.4f}', f'{best_stats["val_loss"]:.4f}',
         f'{(best_stats["val_loss"]/best_stats["train_loss"]-1)*100:+.1f}%'],
        ['Accuracy', f'{best_stats["train_acc"]:.4f}', f'{best_stats["val_acc"]:.4f}',
         f'{(best_stats["val_acc"]-best_stats["train_acc"])*100:+.1f}%'],
        ['Epoch', f'{best_stats["epoch"]}', f'{best_stats["epoch"]}', '-'],
        ['Total Epochs', f'{len(train_losses)}', f'{len(train_losses)}', '-']
    ]

    # Create table
    table = ax3.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0.3, 1, 0.7])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Color header row
    for i in range(len(metrics_data[0])):
        table[(0, i)].set_facecolor(colors['highlight'])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color alternating rows
    for i in range(1, len(metrics_data)):
        for j in range(len(metrics_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    # Add title to summary
    ax3.text(0.5, 0.15, 'Best Model Summary', fontsize=14, fontweight='bold',
             ha='center', va='center', transform=ax3.transAxes)

    # Add overfitting indicator
    overfitting_gap = best_stats['train_acc'] - best_stats['val_acc']
    if overfitting_gap > 0.05:
        overfitting_text = f"Overfitting Gap: {overfitting_gap:.3f}"
        overfitting_color = colors['highlight']
    else:
        overfitting_text = f"Generalization Gap: {overfitting_gap:.3f}"
        overfitting_color = 'green'

    ax3.text(0.5, 0.05, overfitting_text, fontsize=11, fontweight='bold',
             ha='center', va='center', color=overfitting_color, transform=ax3.transAxes)

    # Main title
    fig.suptitle(f'Gesture Recognition Model Training History\nBest Model: Epoch {best_stats["epoch"]} | Val Acc: {best_stats["val_acc"]:.4f} ({best_stats["val_acc"]*100:.2f}%)',
                fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save with high quality
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Enhanced training history plot saved as '{filename}' (Best Model: Epoch {best_stats['epoch']})")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, val_accuracy, best_epoch, filename='confusion_matrix.png'):
    """
    Plot enhanced confusion matrix with comprehensive metrics visualization

    Args:
        y_true: True labels from validation set
        y_pred: Predicted labels from best model on validation set
        class_names: List of class names
        val_accuracy: Validation accuracy of best model
        best_epoch: Epoch number of best model
        filename: Output filename for the plot
    """
    # Set style
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 12))

    # Create color scheme
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'error': '#F44336'
    }

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Calculate additional metrics
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    class_precision = [report[class_name]['precision'] for class_name in class_names]
    class_recall = [report[class_name]['recall'] for class_name in class_names]
    class_f1 = [report[class_name]['f1-score'] for class_name in class_names]

    # Create subplot layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
    ax_cm = fig.add_subplot(gs[0, :2])  # Confusion matrix (top left)
    ax_legend = fig.add_subplot(gs[0, 2])  # Legend (top right)
    ax_bar = fig.add_subplot(gs[1, :])  # Bar chart (middle)
    ax_table = fig.add_subplot(gs[2, :])  # Metrics table (bottom)

    # Plot 1: Enhanced Confusion Matrix
    # Normalize for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create custom colormap
    cmap = plt.cm.Blues
    im = ax_cm.imshow(cm_normalized, interpolation='nearest', cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Count', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    # Add text annotations with both counts and percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percentage = cm_normalized[i, j] * 100

            # Choose text color based on background
            text_color = 'white' if cm_normalized[i, j] > 0.6 else 'black'

            if i == j:  # Correct predictions
                weight = 'bold'
                size = 12
            else:  # Incorrect predictions
                weight = 'normal'
                size = 10

            ax_cm.text(j, i, f'{count}\n({percentage:.1f}%)',
                      ha='center', va='center', color=text_color,
                      fontsize=size, fontweight=weight)

    # Set labels and title
    ax_cm.set_title('Confusion Matrix - Best Model Performance', fontsize=16, fontweight='bold', pad=20)
    ax_cm.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax_cm.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax_cm.set_xticks(range(len(class_names)))
    ax_cm.set_yticks(range(len(class_names)))
    ax_cm.set_xticklabels(class_names, fontsize=11, rotation=45, ha='right')
    ax_cm.set_yticklabels(class_names, fontsize=11)

    # Add grid
    ax_cm.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax_cm.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax_cm.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.3)
    ax_cm.tick_params(which='minor', bottom=False, left=False)

    # Plot 2: Legend and Summary Stats
    ax_legend.axis('off')

    # Add summary information
    info_text = f"""Model Performance Summary

Best Epoch: {best_epoch}
Overall Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)
Total Samples: {len(y_true)}
Correct Predictions: {int(val_accuracy * len(y_true))}

Model Quality: {'Excellent' if val_accuracy > 0.95 else 'Good' if val_accuracy > 0.90 else 'Fair' if val_accuracy > 0.80 else 'Poor'}
"""

    ax_legend.text(0.1, 0.9, info_text, transform=ax_legend.transAxes, fontsize=11,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Plot 3: Per-Class Metrics Bar Chart
    x = np.arange(len(class_names))
    width = 0.2

    bars1 = ax_bar.bar(x - width*1.5, class_precision, width, label='Precision', color=colors['primary'], alpha=0.8)
    bars2 = ax_bar.bar(x - width*0.5, class_recall, width, label='Recall', color=colors['secondary'], alpha=0.8)
    bars3 = ax_bar.bar(x + width*0.5, class_f1, width, label='F1-Score', color=colors['accent'], alpha=0.8)
    bars4 = ax_bar.bar(x + width*1.5, class_accuracies, width, label='Accuracy', color=colors['success'], alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax_bar.set_xlabel('Gesture Classes', fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax_bar.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(class_names, fontsize=10, rotation=45, ha='right')
    ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.grid(True, alpha=0.3)

    # Plot 4: Detailed Metrics Table
    ax_table.axis('off')

    # Prepare table data
    table_data = []
    table_data.append(['Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Support'])

    for i, class_name in enumerate(class_names):
        table_data.append([
            class_name,
            f'{class_precision[i]:.3f}',
            f'{class_recall[i]:.3f}',
            f'{class_f1[i]:.3f}',
            f'{class_accuracies[i]:.3f}',
            f'{int(report[class_name]["support"])}'
        ])

    # Add overall averages
    table_data.append([
        'MACRO AVG',
        f'{report["macro avg"]["precision"]:.3f}',
        f'{report["macro avg"]["recall"]:.3f}',
        f'{report["macro avg"]["f1-score"]:.3f}',
        f'{np.mean(class_accuracies):.3f}',
        f'{int(report["macro avg"]["support"])}'
    ])

    # Create table
    table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                          cellLoc='center', loc='center', bbox=[0.1, 0.2, 0.8, 0.7])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor(colors['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color alternating rows and highlight best/worst performers
    best_precision_idx = np.argmax(class_precision)
    worst_precision_idx = np.argmin(class_precision)

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

            # Highlight best and worst precision
            if j == 1 and i-1 == best_precision_idx:
                table[(i, j)].set_facecolor('#d4edda')  # Light green for best
                table[(i, j)].set_text_props(weight='bold')
            elif j == 1 and i-1 == worst_precision_idx:
                table[(i, j)].set_facecolor('#f8d7da')  # Light red for worst

    # Add title
    ax_table.text(0.5, 0.05, 'Detailed Classification Metrics', fontsize=14, fontweight='bold',
                 ha='center', va='center', transform=ax_table.transAxes)

    # Main title
    fig.suptitle(f'Gesture Recognition Model - Confusion Matrix Analysis\nEpoch {best_epoch} | Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)',
                fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Enhanced confusion matrix saved as '{filename}' (Best Model: Epoch {best_epoch})")
    plt.close()

    # Print comprehensive classification report
    print(f"\n{'='*90}")
    print(f"ENHANCED CLASSIFICATION REPORT - BEST MODEL (Epoch {best_epoch}) on VALIDATION SET")
    print(f"{'='*90}")
    print(f"Overall Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"Total Samples: {len(y_true)} | Correct Predictions: {int(val_accuracy * len(y_true))} | Incorrect: {len(y_true) - int(val_accuracy * len(y_true))}")
    print(f"{'='*90}")

    # Best and worst performing classes
    best_class_idx = np.argmax(class_f1)
    worst_class_idx = np.argmin(class_f1)

    print(f"\nBest Performing Class: {class_names[best_class_idx]} (F1: {class_f1[best_class_idx]:.3f})")
    print(f"Worst Performing Class: {class_names[worst_class_idx]} (F1: {class_f1[worst_class_idx]:.3f})")

    print("\nPer-Class Performance:")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10} {'Support':<8}")
    print("-" * 88)

    for i, class_name in enumerate(class_names):
        metrics = report[class_name]
        print(f"{class_name:<20} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {class_accuracies[i]:<10.3f} {int(metrics['support']):<8}")

    # Overall metrics
    print("-" * 88)
    overall = report['macro avg']
    print(f"{'MACRO AVERAGE':<20} {overall['precision']:<10.3f} {overall['recall']:<10.3f} {overall['f1-score']:<10.3f} {np.mean(class_accuracies):<10.3f} {int(overall['support']):<8}")

    overall = report['weighted avg']
    print(f"{'WEIGHTED AVERAGE':<20} {overall['precision']:<10.3f} {overall['recall']:<10.3f} {overall['f1-score']:<10.3f} {val_accuracy:<10.3f} {int(overall['support']):<8}")
    print(f"{'='*90}")

    return report


def plot_pca_boxplot(data, labels, filename='pca_boxplot.png'):
    """
    Create boxplot visualization of PCA-transformed data showing distribution of principal components per class

    Args:
        data: List of PCA-transformed DataFrames
        labels: List of corresponding labels
        filename: Output filename for the plot
    """
    # Concatenate all data and create label array
    concatenated = pd.concat(data, ignore_index=True)
    label_array = []
    for i, (df, label) in enumerate(zip(data, labels)):
        label_array.extend([label] * len(df))

    # Create a copy of the data with labels for plotting
    plot_data = concatenated.copy()
    plot_data['label'] = label_array

    # Set style for better aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 10))

    # Create color palette for classes
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_dict = dict(zip(unique_labels, colors))

    # Create subplots
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])

    # Main boxplot (spans most of the figure)
    ax1 = fig.add_subplot(gs[:, 0])

    # Prepare data for boxplot - reshape for seaborn
    # Melt the dataframe to have 'PC' and 'value' columns
    melted_data = plot_data.melt(id_vars=['label'], var_name='Principal_Component', value_name='Value')

    # Create boxplot with seaborn for better aesthetics
    sns.boxplot(data=melted_data, x='Principal_Component', y='Value',
                hue='label', palette=color_dict, ax=ax1, width=0.8)

    ax1.set_title('Distribution of Principal Components by Gesture Class',
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Principal Components', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Component Values', fontsize=12, fontweight='bold')
    ax1.legend(title='Gesture Classes', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)

    # Statistics summary (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Calculate statistics for each principal component
    stats_text = "PCA COMPONENT STATISTICS\n" + "="*25 + "\n\n"

    for pc in melted_data['Principal_Component'].unique()[:10]:  # Show first 10 PCs
        pc_data = melted_data[melted_data['Principal_Component'] == pc]['Value']
        stats_text += f"{pc}:\n"
        stats_text += f"  Mean: {pc_data.mean():.3f}\n"
        stats_text += f"  Std:  {pc_data.std():.3f}\n"
        stats_text += f"  Min:  {pc_data.min():.3f}\n"
        stats_text += f"  Max:  {pc_data.max():.3f}\n\n"

    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    # Dataset info (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    # Dataset information
    total_samples = len(plot_data)
    n_components = len([col for col in plot_data.columns if col.startswith('PC')])
    class_counts = plot_data['label'].value_counts()

    info_text = f"""DATASET INFORMATION
{"="*25}

Total Samples: {total_samples:,}
Principal Components: {n_components}
Classes: {len(unique_labels)}

Class Distribution:
"""
    for label, count in class_counts.items():
        percentage = count / total_samples * 100
        info_text += f"  {label}: {count} ({percentage:.1f}%)\n"

    info_text += f"\nVariance Explained:\n"
    info_text += f"  (95% variance retained)\n"

    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    # Main title
    fig.suptitle('PCA-Transformed Dataset Analysis\nPost-Processing Feature Distribution',
                fontsize=18, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    # Save with high quality
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"PCA boxplot visualization saved as '{filename}'")
    plt.close()


def plot_comprehensive_metrics(y_true, y_pred, class_names, val_accuracy, best_epoch,
                              train_losses, val_losses, train_accs, val_accs, best_stats,
                              filename='comprehensive_metrics.png'):
    """
    Create comprehensive dashboard with all key metrics and visualizations

    Args:
        y_true: True labels from validation set
        y_pred: Predicted labels from best model on validation set
        class_names: List of class names
        val_accuracy: Validation accuracy of best model
        best_epoch: Epoch number of best model
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        best_stats: Dictionary with best epoch statistics
        filename: Output filename for the plot
    """
    # Set style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 14))

    # Color scheme
    colors = {
        'train': '#2E86AB',
        'val': '#A23B72',
        'best': '#F18F01',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'error': '#F44336'
    }

    # Create 3x3 grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], width_ratios=[1, 1, 1])

    # 1. Training Progress (top, spans all columns)
    ax1 = fig.add_subplot(gs[0, :])

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_accs, color=colors['train'], linewidth=2.5, label='Training Accuracy', alpha=0.8)
    ax1.plot(epochs, val_accs, color=colors['val'], linewidth=2.5, label='Validation Accuracy', alpha=0.8)
    ax1.axvline(x=best_epoch, color=colors['best'], linestyle='--', linewidth=2, alpha=0.8)
    ax1.plot(best_epoch, best_stats['val_acc'], 'o', color=colors['best'], markersize=12,
             markeredgecolor='black', markeredgewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Add annotation for best model
    ax1.annotate(f'Best Model\nEpoch {best_epoch}\nAcc: {best_stats["val_acc"]:.4f}',
                xy=(best_epoch, best_stats['val_acc']), xytext=(best_epoch + len(epochs)*0.1, 0.5),
                arrowprops=dict(arrowstyle='->', color=colors['best'], lw=2),
                fontsize=10, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['best'], alpha=0.3))

    # 2. Key Metrics Overview (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')

    # Calculate key metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    # Metrics text
    metrics_text = f"""KEY METRICS OVERVIEW

Overall Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)
Macro Precision: {macro_precision:.4f}
Macro Recall: {macro_recall:.4f}
Macro F1-Score: {macro_f1:.4f}

Training Statistics:
• Total Epochs: {len(train_losses)}
• Best Epoch: {best_epoch}
• Final Train Loss: {train_losses[-1]:.4f}
• Final Val Loss: {val_losses[-1]:.4f}

Model Quality: {'Excellent' if val_accuracy > 0.95 else 'Good' if val_accuracy > 0.90 else 'Fair' if val_accuracy > 0.80 else 'Poor'}
"""

    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    # 3. Class Distribution (middle center)
    ax3 = fig.add_subplot(gs[1, 1])

    _, counts = np.unique(y_true, return_counts=True)
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

    ax3.pie(counts, labels=class_names, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 9})
    ax3.set_title('Class Distribution in Validation Set', fontsize=12, fontweight='bold')

    # 4. Performance Radar Chart (middle right)
    ax4 = fig.add_subplot(gs[1, 2], projection='polar')

    # Calculate per-class metrics for radar chart
    angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    class_f1_scores = [report[class_name]['f1-score'] for class_name in class_names]
    class_f1_scores += class_f1_scores[:1]  # Complete the circle

    ax4.plot(angles, class_f1_scores, 'o-', linewidth=2, color=colors['train'], markersize=8)
    ax4.fill(angles, class_f1_scores, alpha=0.25, color=colors['train'])
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(class_names, fontsize=9)
    ax4.set_ylim(0, 1)
    ax4.set_title('F1-Score per Class', fontsize=12, fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3)

    # 5. Confusion Matrix (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    cm = confusion_matrix(y_true, y_pred)

    # Normalize for visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax5.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax5.set_title('Confusion Matrix (Normalized)', fontsize=11, fontweight='bold')
    ax5.set_xticks(range(len(class_names)))
    ax5.set_yticks(range(len(class_names)))
    ax5.set_xticklabels(class_names, fontsize=8, rotation=45, ha='right')
    ax5.set_yticklabels(class_names, fontsize=8)
    ax5.set_xlabel('Predicted', fontsize=9)
    ax5.set_ylabel('True', fontsize=9)

    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm_normalized[i, j] * 100
            color = 'white' if cm_normalized[i, j] > 0.6 else 'black'
            ax5.text(j, i, f'{percentage:.0f}%', ha='center', va='center',
                    color=color, fontsize=8, fontweight='bold' if i == j else 'normal')

    # 6. Loss Trends (bottom center)
    ax6 = fig.add_subplot(gs[2, 1])

    # Show last N epochs for better visibility
    last_n = min(30, len(train_losses))
    recent_epochs = range(len(train_losses) - last_n + 1, len(train_losses) + 1)
    recent_train_losses = train_losses[-last_n:]
    recent_val_losses = val_losses[-last_n:]

    ax6.plot(recent_epochs, recent_train_losses, color=colors['train'], linewidth=2,
             label='Train Loss', marker='o', markersize=4)
    ax6.plot(recent_epochs, recent_val_losses, color=colors['val'], linewidth=2,
             label='Val Loss', marker='s', markersize=4)

    if best_epoch in recent_epochs:
        ax6.axvline(x=best_epoch, color=colors['best'], linestyle='--', linewidth=2, alpha=0.8)

    ax6.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Loss', fontsize=10, fontweight='bold')
    ax6.set_title(f'Recent Loss Trends (Last {last_n} Epochs)', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # 7. Learning Curve Analysis (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    # Analyze learning patterns
    overfitting_point = None
    for i in range(1, len(val_accs)):
        if val_accs[i] < val_accs[i-1] and train_accs[i] > train_accs[i-1]:
            overfitting_point = i + 1
            break

    analysis_text = f"""LEARNING ANALYSIS

Convergence Analysis:
• Final Train Acc: {train_accs[-1]:.4f}
• Final Val Acc: {val_accs[-1]:.4f}
• Peak Val Acc: {max(val_accs):.4f}

Overfitting Check:
{"• Signs of overfitting detected" if overfitting_point else "• No clear overfitting signs"}
• Est. overfitting epoch: {overfitting_point if overfitting_point else "N/A"}

Training Efficiency:
• Epochs to 90% accuracy: {next((i+1 for i, acc in enumerate(val_accs) if acc >= 0.9), "N/A")}
• Best improvement: {(max(val_accs) - val_accs[0]):.4f}
"""

    ax7.text(0.1, 0.9, analysis_text, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Main title
    fig.suptitle(f'Gesture Recognition Model - Comprehensive Performance Dashboard\n'
                f'Best Model: Epoch {best_epoch} | Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)',
                fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Comprehensive metrics dashboard saved as '{filename}'")
    plt.close()

    return report


def plot_model_weights_heatmap(model, class_names, best_epoch, filename='weights_heatmap.png'):
    """
    Create heatmap visualizations of model weights for interpretability

    Args:
        model: Trained PyTorch model
        class_names: List of class names
        best_epoch: Epoch number of best model
        filename: Output filename for the plot
    """
    # Set style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))

    # Color scheme
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
    }

    # Create grid layout for multiple weight visualizations
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1.2, 1.2, 0.8])

    # Extract weights from different layers
    with torch.no_grad():
        # 1. Attention weights
        attention_weights = model.attention.weight.cpu().numpy()
        attention_bias = model.attention.bias.cpu().numpy()

        # 2. First fully connected layer
        fc1_weights = model.fc1.weight.cpu().numpy()
        fc1_bias = model.fc1.bias.cpu().numpy()

        # 3. Final classification layer
        fc2_weights = model.fc2.weight.cpu().numpy()
        fc2_bias = model.fc2.bias.cpu().numpy()

    # Plot 1: Attention Layer Weights
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(attention_weights.T, cmap='RdBu_r', aspect='auto', 
                     interpolation='nearest', vmin=-np.abs(attention_weights).max(), 
                     vmax=np.abs(attention_weights).max())
    ax1.set_title('Attention Layer Weights', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Output Features', fontsize=11)
    ax1.set_ylabel('Input Features', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Weight Value', rotation=270, labelpad=15, fontsize=10)

    # Plot 2: FC1 Layer Weights (sampled if too large)
    ax2 = fig.add_subplot(gs[0, 1])
    # Sample weights if too large for visualization
    if fc1_weights.shape[0] > 100 or fc1_weights.shape[1] > 100:
        sample_rows = min(100, fc1_weights.shape[0])
        sample_cols = min(100, fc1_weights.shape[1])
        row_indices = np.linspace(0, fc1_weights.shape[0]-1, sample_rows, dtype=int)
        col_indices = np.linspace(0, fc1_weights.shape[1]-1, sample_cols, dtype=int)
        fc1_sampled = fc1_weights[np.ix_(row_indices, col_indices)]
    else:
        fc1_sampled = fc1_weights

    im2 = ax2.imshow(fc1_sampled, cmap='RdBu_r', aspect='auto', 
                     interpolation='nearest', vmin=-np.abs(fc1_sampled).max(), 
                     vmax=np.abs(fc1_sampled).max())
    ax2.set_title(f'First FC Layer Weights\n(Shape: {fc1_weights.shape})', 
                  fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Input Features', fontsize=11)
    ax2.set_ylabel('Output Features', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Weight Value', rotation=270, labelpad=15, fontsize=10)

    # Plot 3: Final Classification Layer Weights
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(fc2_weights, cmap='RdBu_r', aspect='auto', 
                     interpolation='nearest', vmin=-np.abs(fc2_weights).max(), 
                     vmax=np.abs(fc2_weights).max())
    ax3.set_title('Output Layer Weights', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel('Hidden Features', fontsize=11)
    ax3.set_yticks(range(len(class_names)))
    ax3.set_yticklabels(class_names, fontsize=10)
    ax3.set_ylabel('Classes', fontsize=11)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Weight Value', rotation=270, labelpad=15, fontsize=10)

    # Plot 4: Output Layer Weights per Class (Bar plot)
    ax4 = fig.add_subplot(gs[1, :2])
    class_weight_norms = np.linalg.norm(fc2_weights, axis=1)
    bars = ax4.bar(range(len(class_names)), class_weight_norms, 
                   color=colors['primary'], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Color bars by magnitude
    max_norm = class_weight_norms.max()
    for bar, norm in zip(bars, class_weight_norms):
        if norm == max_norm:
            bar.set_color(colors['accent'])
            bar.set_alpha(0.9)
    
    ax4.set_xlabel('Gesture Classes', fontsize=12, fontweight='bold')
    ax4.set_ylabel('L2 Norm of Weights', fontsize=12, fontweight='bold')
    ax4.set_title('Output Layer Weight Magnitudes per Class', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(class_names)))
    ax4.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, norm) in enumerate(zip(bars, class_weight_norms)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max_norm*0.01,
                f'{norm:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 5: Weight Statistics Summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Calculate statistics
    stats_text = f"""WEIGHT STATISTICS

Attention Layer:
  Shape: {attention_weights.shape}
  Mean: {attention_weights.mean():.4f}
  Std: {attention_weights.std():.4f}
  Min: {attention_weights.min():.4f}
  Max: {attention_weights.max():.4f}

FC1 Layer:
  Shape: {fc1_weights.shape}
  Mean: {fc1_weights.mean():.4f}
  Std: {fc1_weights.std():.4f}
  Min: {fc1_weights.min():.4f}
  Max: {fc1_weights.max():.4f}

Output Layer:
  Shape: {fc2_weights.shape}
  Mean: {fc2_weights.mean():.4f}
  Std: {fc2_weights.std():.4f}
  Min: {fc2_weights.min():.4f}
  Max: {fc2_weights.max():.4f}
"""
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    # Plot 6: Weight Distribution Histogram
    ax6 = fig.add_subplot(gs[2, 0])
    all_weights = np.concatenate([
        attention_weights.flatten(),
        fc1_weights.flatten(),
        fc2_weights.flatten()
    ])
    ax6.hist(all_weights, bins=50, color=colors['primary'], alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero')
    ax6.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('Overall Weight Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Bias Values
    ax7 = fig.add_subplot(gs[2, 1])
    bias_data = {
        'Attention': attention_bias,
        'FC1': fc1_bias[:20] if len(fc1_bias) > 20 else fc1_bias,  # Sample if too large
        'Output': fc2_bias
    }
    
    positions = []
    labels = []
    all_biases = []
    pos = 0
    
    for layer_name, biases in bias_data.items():
        for i, bias in enumerate(biases):
            positions.append(pos)
            all_biases.append(bias)
            pos += 1
        labels.append((layer_name, len(biases)))
    
    ax7.bar(positions, all_biases, color=colors['secondary'], alpha=0.7, edgecolor='black')
    ax7.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax7.set_xlabel('Layer Neurons', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Bias Value', fontsize=11, fontweight='bold')
    ax7.set_title('Bias Values Across Layers (Sample)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add layer separation lines
    cumsum = 0
    for layer_name, count in labels:
        if cumsum > 0:
            ax7.axvline(x=cumsum-0.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
        ax7.text(cumsum + count/2, ax7.get_ylim()[1]*0.9, layer_name, 
                ha='center', fontsize=9, fontweight='bold')
        cumsum += count

    # Plot 8: Layer Comparison
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    layer_info = f"""LAYER ANALYSIS

Most Activated Class:
  {class_names[np.argmax(class_weight_norms)]}

Least Activated Class:
  {class_names[np.argmin(class_weight_norms)]}

Weight Sparsity:
  Near-zero weights: 
  {(np.abs(all_weights) < 0.01).sum()/len(all_weights)*100:.1f}%

Weight Magnitude:
  Largest: {np.abs(all_weights).max():.4f}
  Mean: {np.abs(all_weights).mean():.4f}

Bias Range:
  Output layer:
  [{fc2_bias.min():.3f}, {fc2_bias.max():.3f}]
"""
    
    ax8.text(0.1, 0.9, layer_info, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Main title
    fig.suptitle(f'Gesture Recognition Model - Weight Analysis Heatmaps\n'
                f'Best Model: Epoch {best_epoch}',
                fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Model weights heatmap saved as '{filename}'")
    plt.close()


def plot_lstm_weights_heatmap(model, best_epoch, filename='lstm_weights_heatmap.png'):
    """
    Create detailed heatmap visualizations of LSTM layer weights

    Args:
        model: Trained PyTorch model with LSTM layers
        best_epoch: Epoch number of best model
        filename: Output filename for the plot
    """
    # Set style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 14))

    # Extract LSTM weights
    with torch.no_grad():
        # LSTM has weight_ih (input-hidden) and weight_hh (hidden-hidden) for each layer
        lstm_weights = []
        for name, param in model.lstm.named_parameters():
            if 'weight' in name:
                lstm_weights.append((name, param.cpu().numpy()))

    # Calculate grid size based on number of weight matrices
    n_weights = len(lstm_weights)
    n_cols = 3
    n_rows = (n_weights + n_cols - 1) // n_cols
    
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    # Plot each weight matrix
    for idx, (name, weights) in enumerate(lstm_weights):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Sample if too large
        if weights.shape[0] > 200 or weights.shape[1] > 200:
            sample_rows = min(200, weights.shape[0])
            sample_cols = min(200, weights.shape[1])
            row_indices = np.linspace(0, weights.shape[0]-1, sample_rows, dtype=int)
            col_indices = np.linspace(0, weights.shape[1]-1, sample_cols, dtype=int)
            weights_sampled = weights[np.ix_(row_indices, col_indices)]
            title_suffix = f'\n(Sampled from {weights.shape})'
        else:
            weights_sampled = weights
            title_suffix = f'\n(Shape: {weights.shape})'
        
        # Create heatmap
        im = ax.imshow(weights_sampled, cmap='RdBu_r', aspect='auto', 
                      interpolation='nearest', vmin=-np.abs(weights_sampled).max(), 
                      vmax=np.abs(weights_sampled).max())
        
        # Format layer name
        display_name = name.replace('weight_ih_l', 'Input-Hidden L').replace('weight_hh_l', 'Hidden-Hidden L')
        if '_reverse' in display_name:
            display_name = display_name.replace('_reverse', ' (Reverse)')
        
        ax.set_title(f'{display_name}{title_suffix}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Input Dimension', fontsize=9)
        ax.set_ylabel('Output Dimension', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Weight', rotation=270, labelpad=12, fontsize=9)
        
        # Add statistics text
        stats_text = f'μ={weights.mean():.3f}\nσ={weights.std():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Main title
    fig.suptitle(f'LSTM Layer Weights - Detailed Analysis\n'
                f'Best Model: Epoch {best_epoch}',
                fontsize=18, fontweight='bold', y=0.99)

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"LSTM weights heatmap saved as '{filename}'")
    plt.close()
