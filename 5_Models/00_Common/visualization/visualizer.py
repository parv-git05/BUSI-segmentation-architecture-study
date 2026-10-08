"""
Visualization Utilities for BUSI Phase-1 Segmentation Architecture Study.
Generates publication-quality loss curves, metric progression charts,
qualitative prediction grids, and per-sample metric distribution plots.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive background backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title_suffix: str = ""
) -> None:
    """
    Plots training loss, validation loss, and metric curves over training epochs.
    
    Args:
        history (Dict[str, List[float]]): History dictionary containing epoch-wise metrics.
        save_path (str): Output image file path (.png or .pdf).
        title_suffix (str): Extra text for title.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_style("whitegrid")

    # 1. Loss Subplot
    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title(f"Loss Trajectory {title_suffix}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(loc="upper right")

    # 2. Key Metrics Subplot
    if "val_dice_mean" in history:
        axes[1].plot(epochs, history["val_dice_mean"], label="Val Dice", color="#2ca02c", linewidth=2)
    if "val_iou_mean" in history:
        axes[1].plot(epochs, history["val_iou_mean"], label="Val IoU", color="#d62728", linewidth=2)
    if "val_nsd_1px_mean" in history:
        axes[1].plot(epochs, history["val_nsd_1px_mean"], label="Val NSD @ 1px", color="#9467bd", linewidth=1.5, linestyle=":")
        
    axes[1].set_title(f"Validation Metrics {title_suffix}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Score", fontsize=10)
    axes[1].set_ylim([0.0, 1.05])
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_qualitative_predictions(
    images: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    sample_ids: List[str],
    metrics_list: List[Dict[str, Any]],
    save_path: str,
    max_samples: int = 6
) -> None:
    """
    Renders a grid comparison of Input Image, Ground Truth Mask, Predicted Mask, and Overlay.
    Annotates each sample with its Dice score and HD95 value.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    n = min(len(images), max_samples)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        img = np.squeeze(images[i])
        gt = np.squeeze(gt_masks[i])
        pred = np.squeeze(pred_masks[i])
        m = metrics_list[i] if i < len(metrics_list) else {}
        sid = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"

        dice_val = m.get("dice", 0.0)
        hd95_val = m.get("hd95", np.nan)
        hd95_str = f"{hd95_val:.2f}px" if not np.isnan(hd95_val) else "N/A"

        # Image
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"Image: {sid}", fontsize=10)
        axes[i, 0].axis("off")

        # Ground Truth
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask", fontsize=10)
        axes[i, 1].axis("off")

        # Prediction
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title(f"Prediction (Dice: {dice_val:.4f})", fontsize=10)
        axes[i, 2].axis("off")

        # Overlay (Red GT boundary, Green Pred boundary over Image)
        overlay = np.stack([img, img, img], axis=-1)
        if overlay.max() > 1.0:
            overlay = overlay / 255.0

        # Highlight Pred in Green channel, GT in Red channel
        overlay[gt > 0.5] = overlay[gt > 0.5] * 0.5 + np.array([0.5, 0.0, 0.0])
        overlay[pred > 0.5] = overlay[pred > 0.5] * 0.5 + np.array([0.0, 0.5, 0.0])

        axes[i, 3].imshow(np.clip(overlay, 0, 1))
        axes[i, 3].set_title(f"Overlay (HD95: {hd95_str})", fontsize=10)
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_distributions(
    per_sample_df: pd.DataFrame,
    save_path: str,
    metrics: Optional[List[str]] = None
) -> None:
    """
    Plots distribution histograms and boxplots for per-sample metrics.
    Useful for identifying outliers and model failure cases.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if metrics is None:
        metrics = ["dice", "iou", "precision", "recall", "specificity", "nsd_1px"]

    valid_metrics = [m for m in metrics if m in per_sample_df.columns]
    if not valid_metrics:
        return

    n_metrics = len(valid_metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(4 * n_metrics, 8))

    for idx, metric in enumerate(valid_metrics):
        vals = pd.to_numeric(per_sample_df[metric], errors="coerce").dropna()

        # Histogram / KDE
        sns.histplot(vals, ax=axes[0, idx], kde=True, color="#1f77b4")
        axes[0, idx].set_title(f"{metric.upper()} Distribution", fontsize=11, fontweight="bold")
        axes[0, idx].set_xlabel(metric, fontsize=9)

        # Boxplot
        sns.boxplot(y=vals, ax=axes[1, idx], color="#2ca02c")
        axes[1, idx].set_title(f"{metric.upper()} Boxplot", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
