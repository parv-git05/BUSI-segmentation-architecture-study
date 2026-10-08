"""
Official 100-Epoch Experiment Execution Script for U-Net (EXP-ARCH-01_UNET).
Executes full baseline training protocol without early stopping.
LOCKED TEST SET IS UNTOUCHED.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch

# Ensure 00_Common and 01_UNet are in sys.path
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
common_root = os.path.join(models_root, "00_Common")
unet_root = os.path.join(models_root, "01_UNet")

for p in [common_root, unet_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.experiment_config import ExperimentConfig
from datasets.busi_dataset import create_dataloader, BUSIDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from training.losses import DiceLoss
from training.trainer import SegmentationTrainer
from utils.experiment_tracker import ExperimentTracker
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.seed import seed_everything
from metrics.segmentation_metrics import calculate_batch_metrics, aggregate_metrics
from visualization.visualizer import plot_training_curves, plot_qualitative_predictions, plot_metric_distributions
from model import UNet


def run_official_unet_experiment():
    print("=" * 70)
    print("STARTING OFFICIAL U-NET RESEARCH EXPERIMENT: EXP-ARCH-01_UNET")
    print("=" * 70)

    master_split_dir = os.path.join(models_root, "MASTER_SPLIT")
    exp_parent_dir = os.path.join(unet_root, "experiments")
    exp_dir = os.path.join(exp_parent_dir, "EXP-ARCH-01_UNET")

    # Create directory structure
    subdirs = [
        "logs",
        "checkpoints",
        "metrics",
        "training_curves",
        "qualitative_predictions",
        "reports"
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    # 1. Config Creation
    config = ExperimentConfig(
        experiment_name="EXP-ARCH-01_UNET",
        architecture_name="UNet",
        seed=42,
        train_manifest_path=os.path.join(master_split_dir, "train_70.csv"),
        val_manifest_path=os.path.join(master_split_dir, "val_20.csv"),
        test_manifest_path=os.path.join(master_split_dir, "test_10.csv"),
        input_size=(256, 256),
        img_channels=1,
        batch_size=16,
        epochs=100,
        learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer_name="Adam",
        scheduler_name="ReduceLROnPlateau",
        scheduler_patience=10,
        scheduler_factor=0.5,
        loss_fn_name="DiceLoss",
        dice_smooth=1e-6,
        target_metric="val_dice",
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=exp_parent_dir
    )

    config.save_json(os.path.join(exp_dir, "config.json"))

    # 2. Seed Reproducibility
    seed_everything(config.seed)

    # 3. Create DataLoaders (Train and Val ONLY - NO Test Set!)
    train_transform = get_train_transforms(input_size=config.input_size, augment=True)
    val_transform = get_val_transforms(input_size=config.input_size)

    train_loader = create_dataloader(
        manifest_path=config.train_manifest_path,
        batch_size=config.batch_size,
        shuffle=True,
        transform=train_transform,
        seed=config.seed
    )

    val_loader = create_dataloader(
        manifest_path=config.val_manifest_path,
        batch_size=config.batch_size,
        shuffle=False,
        transform=val_transform,
        seed=config.seed
    )

    print(f"DataLoaders Ready: Train = {len(train_loader.dataset)} samples, Val = {len(val_loader.dataset)} samples.")
    print(f"LOCKED Test Set: {config.test_manifest_path} (NOT loaded/evaluated).")

    # 4. Instantiate Architecture, Optimizer, Loss, Scheduler
    model = UNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = DiceLoss(smooth=config.dice_smooth, from_logits=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        verbose=True
    )

    # 5. Tracker & Logger Setup
    tracker = ExperimentTracker(
        output_root=exp_parent_dir,
        experiment_name="EXP-ARCH-01_UNET"
    )

    # 6. Instantiate and Run Trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        scheduler=scheduler,
        tracker=tracker,
        device=config.device
    )

    print("Beginning 100-epoch training loop...")
    summary = trainer.fit()
    tracker.close()

    print("\nTraining complete. Running final validation evaluation using BEST checkpoint...")

    # 7. Final Validation Evaluation using Best Model Checkpoint
    best_ckpt_path = os.path.join(exp_dir, "checkpoints", "best_model.pth")
    best_model = UNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False).to(config.device)
    ckpt_meta = load_checkpoint(best_ckpt_path, best_model, device=config.device)

    best_model.eval()
    all_val_metrics = []
    val_images, val_gts, val_preds, val_ids = [], [], [], []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            imgs = sample["image"].to(config.device)
            masks = sample["mask"].to(config.device)
            s_ids = sample["sample_id"]

            outputs = best_model(imgs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            batch_m = calculate_batch_metrics(preds, masks, threshold=0.5, sample_ids=s_ids)
            all_val_metrics.extend(batch_m)

            if len(val_images) < 12:
                val_images.extend(imgs.cpu().numpy())
                val_gts.extend(masks.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_ids.extend(s_ids)

    df_val_metrics = pd.DataFrame(all_val_metrics)
    agg_val_metrics = aggregate_metrics(all_val_metrics)

    # Save validation artifacts in EXP-ARCH-01_UNET subfolders
    val_csv_path = os.path.join(exp_dir, "metrics", "val_per_image_metrics.csv")
    df_val_metrics.to_csv(val_csv_path, index=False)

    val_json_path = os.path.join(exp_dir, "metrics", "val_aggregate_metrics.json")
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_val_metrics, f, indent=4)

    # Save Qualitative Prediction Visualizations
    qual_path = os.path.join(exp_dir, "qualitative_predictions", "val_qualitative_predictions.png")
    plot_qualitative_predictions(
        images=val_images[:6],
        gt_masks=val_gts[:6],
        pred_masks=val_preds[:6],
        sample_ids=val_ids[:6],
        metrics_list=all_val_metrics[:6],
        save_path=qual_path,
        max_samples=6
    )

    # Save Metric Distributions Plot
    dist_path = os.path.join(exp_dir, "metrics", "val_metric_distributions.png")
    plot_metric_distributions(df_val_metrics, save_path=dist_path)

    # 8. Save Final Concise Experiment Report
    report_dict = {
        "experiment_id": "EXP-ARCH-01_UNET",
        "architecture": "U-Net (Standard 2D Encoder-Decoder)",
        "parameter_count": model.count_parameters(),
        "dataset_cohort": "BUSI Phase-1 (628 samples)",
        "split_counts": {"train": 440, "val": 125, "test": 63},
        "locked_test_set_evaluated": False,
        "input_shape": [1, 256, 256],
        "augmentation": "Horizontal flip (p=0.5), Vertical flip (p=0.5), Rotation +-15 deg",
        "loss": "Binary Dice Loss (smooth=1e-6)",
        "optimizer": "Adam (lr=1e-4, weight_decay=1e-4)",
        "scheduler": "ReduceLROnPlateau (patience=10, factor=0.5)",
        "total_epochs": 100,
        "early_stopping": "DISABLED",
        "best_epoch": ckpt_meta["epoch"],
        "best_val_loss": ckpt_meta["val_loss"],
        "best_val_dice": agg_val_metrics["dice_mean"],
        "aggregate_validation_metrics": agg_val_metrics,
        "training_duration_seconds": summary["total_time_seconds"],
        "best_checkpoint_path": best_ckpt_path,
        "qualitative_predictions_path": qual_path,
        "hd95_valid_samples": agg_val_metrics.get("hd95_valid_count", 125)
    }

    report_json_path = os.path.join(exp_dir, "reports", "experiment_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    report_md_path = os.path.join(exp_dir, "reports", "EXPERIMENT_REPORT.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"""# Official Experiment Report: EXP-ARCH-01_UNET

- **Experiment ID:** `EXP-ARCH-01_UNET`
- **Architecture:** Standard 2D U-Net (31,036,481 trainable parameters)
- **Dataset:** BUSI Phase-1 (440 Train / 125 Validation / 63 Test)
- **Test Set Status:** **LOCKED & UNTOUCHED (0 test samples loaded/evaluated)**
- **Training Epochs:** 100 / 100 (Early Stopping DISABLED)
- **Best Epoch:** {ckpt_meta["epoch"]}
- **Best Validation Dice:** **{agg_val_metrics["dice_mean"]:.4f} \u00b1 {agg_val_metrics["dice_std"]:.4f}**
- **Validation IoU:** {agg_val_metrics["iou_mean"]:.4f} \u00b1 {agg_val_metrics["iou_std"]:.4f}
- **Validation Precision:** {agg_val_metrics["precision_mean"]:.4f} \u00b1 {agg_val_metrics["precision_std"]:.4f}
- **Validation Recall:** {agg_val_metrics["recall_mean"]:.4f} \u00b1 {agg_val_metrics["recall_std"]:.4f}
- **Validation Specificity:** {agg_val_metrics["specificity_mean"]:.4f} \u00b1 {agg_val_metrics["specificity_std"]:.4f}
- **Validation HD95:** {agg_val_metrics["hd95_mean"]:.2f} \u00b1 {agg_val_metrics["hd95_std"]:.2f} px ({agg_val_metrics.get("hd95_valid_count", 0)}/125 valid cases)
- **Validation NSD @ 1px:** {agg_val_metrics["nsd_1px_mean"]:.4f} \u00b1 {agg_val_metrics["nsd_1px_std"]:.4f}
- **Validation NSD @ 2px:** {agg_val_metrics["nsd_2px_mean"]:.4f} \u00b1 {agg_val_metrics["nsd_2px_std"]:.4f}
- **Validation NSD @ 3px:** {agg_val_metrics["nsd_3px_mean"]:.4f} \u00b1 {agg_val_metrics["nsd_3px_std"]:.4f}
- **Training Duration:** {summary["total_time_seconds"]/60.0:.2f} minutes
- **Best Checkpoint Path:** `{best_ckpt_path}`
""")

    print("\n" + "=" * 70)
    print(f"OFFICIAL U-NET EXPERIMENT COMPLETE! Best Val Dice: {agg_val_metrics['dice_mean']:.4f} at Epoch {ckpt_meta['epoch']}")
    print("=" * 70)


if __name__ == "__main__":
    run_official_unet_experiment()
