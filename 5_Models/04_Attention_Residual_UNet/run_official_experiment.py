"""
Official 100-Epoch Experiment Execution Script for Attention Residual U-Net (EXP-ARCH-04_ATTENTION_RESIDUAL_UNET).
Executes full baseline training protocol matching standard U-Net without early stopping.
LOCKED TEST SET IS UNTOUCHED.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch

# Ensure 00_Common and 04_Attention_Residual_UNet are in sys.path
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
common_root = os.path.join(models_root, "00_Common")
att_res_unet_root = os.path.join(models_root, "04_Attention_Residual_UNet")

for p in [common_root, att_res_unet_root]:
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
from visualization.visualizer import plot_training_curves, plot_qualitative_predictions
from model import AttentionResidualUNet


def run_official_attention_residual_unet_experiment():
    print("=" * 70)
    print("STARTING OFFICIAL ATTENTION RESIDUAL U-NET RESEARCH EXPERIMENT: EXP-ARCH-04_ATTENTION_RESIDUAL_UNET")
    print("=" * 70)

    master_split_dir = os.path.join(models_root, "MASTER_SPLIT")
    exp_parent_dir = os.path.join(att_res_unet_root, "experiments")
    exp_dir = os.path.join(exp_parent_dir, "EXP-ARCH-04_ATTENTION_RESIDUAL_UNET")

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
        experiment_name="EXP-ARCH-04_ATTENTION_RESIDUAL_UNET",
        architecture_name="AttentionResidualUNet",
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

    # 4. Instantiate Model
    model = AttentionResidualUNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = DiceLoss(smooth=config.dice_smooth, from_logits=True)

    tracker = ExperimentTracker(output_root=exp_parent_dir, experiment_name="EXP-ARCH-04_ATTENTION_RESIDUAL_UNET")

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        tracker=tracker,
        device=config.device
    )

    start_time = time.time()
    fit_summary = trainer.fit()
    total_time = time.time() - start_time

    # 5. Load Best Model Checkpoint
    best_ckpt_path = os.path.join(exp_dir, "checkpoints", "best_model.pth")
    checkpoint = load_checkpoint(best_ckpt_path, model, device=config.device)
    best_epoch = checkpoint.get("epoch", fit_summary.get("best_epoch", 0))

    # Evaluate per-sample validation metrics
    val_eval_loader = create_dataloader(
        manifest_path=config.val_manifest_path,
        batch_size=1,
        shuffle=False,
        transform=val_transform,
        seed=config.seed
    )

    model.eval()
    all_val_metrics = []
    qual_images, qual_masks, qual_preds, qual_sids = [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_eval_loader):
            images = batch["image"].to(config.device)
            masks = batch["mask"].to(config.device)
            sample_ids = batch["sample_id"]

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            batch_m = calculate_batch_metrics(outputs, masks, threshold=0.5, sample_ids=sample_ids)
            all_val_metrics.extend(batch_m)

            if len(qual_images) < 6:
                qual_images.append(images[0, 0].cpu().numpy())
                qual_masks.append(masks[0, 0].cpu().numpy())
                qual_preds.append(preds[0, 0].cpu().numpy())
                qual_sids.append(sample_ids[0])

    df_per_image = pd.DataFrame(all_val_metrics)
    per_image_csv_path = os.path.join(exp_dir, "metrics", "val_per_image_metrics.csv")
    df_per_image.to_csv(per_image_csv_path, index=False)

    agg_metrics = aggregate_metrics(all_val_metrics)
    agg_json_path = os.path.join(exp_dir, "metrics", "val_aggregate_metrics.json")
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_metrics, f, indent=4)

    # 6. Generate Official Plots & Reports
    history_csv_path = os.path.join(exp_dir, "training_curves", "epoch_history.csv")
    if os.path.exists(history_csv_path):
        df_hist = pd.read_csv(history_csv_path)
        history_dict = {col: df_hist[col].tolist() for col in df_hist.columns}
        plot_training_curves(history_dict, save_path=os.path.join(exp_dir, "training_curves", "learning_curves.png"))

    plot_qualitative_predictions(
        images=qual_images,
        gt_masks=qual_masks,
        pred_masks=qual_preds,
        sample_ids=qual_sids,
        metrics_list=all_val_metrics[:len(qual_images)],
        save_path=os.path.join(exp_dir, "qualitative_predictions", "val_qualitative_predictions.png")
    )

    summary_report = {
        "experiment_name": "EXP-ARCH-04_ATTENTION_RESIDUAL_UNET",
        "architecture_name": "AttentionResidualUNet",
        "best_epoch": best_epoch,
        "best_target_metric": agg_metrics.get("dice_mean", 0.0),
        "total_epochs_trained": 100,
        "total_time_seconds": total_time,
        "final_val_metrics": agg_metrics
    }
    with open(os.path.join(exp_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)

    exp_report_json = {
        "experiment_id": "EXP-ARCH-04_ATTENTION_RESIDUAL_UNET",
        "architecture": "Attention Residual U-Net",
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
        "best_epoch": best_epoch,
        "best_val_dice": agg_metrics.get("dice_mean", 0.0),
        "aggregate_validation_metrics": agg_metrics,
        "training_duration_seconds": total_time,
        "best_checkpoint_path": best_ckpt_path,
        "qualitative_predictions_path": os.path.join(exp_dir, "qualitative_predictions", "val_qualitative_predictions.png"),
        "hd95_valid_samples": agg_metrics.get("hd95_valid_count", 0)
    }

    with open(os.path.join(exp_dir, "reports", "experiment_report.json"), "w", encoding="utf-8") as f:
        json.dump(exp_report_json, f, indent=4)

    report_md = f"""# Official Experiment Report: EXP-ARCH-04_ATTENTION_RESIDUAL_UNET

- **Experiment ID:** `EXP-ARCH-04_ATTENTION_RESIDUAL_UNET`
- **Architecture:** Attention Residual U-Net ({model.count_parameters():,} trainable parameters)
- **Dataset:** BUSI Phase-1 (440 Train / 125 Validation / 63 Test)
- **Test Set Status:** **LOCKED & UNTOUCHED (0 test samples loaded/evaluated)**
- **Training Epochs:** 100 / 100 (Early Stopping DISABLED)
- **Best Epoch:** {best_epoch}
- **Best Validation Dice:** **{agg_metrics.get('dice_mean', 0.0):.4f} ± {agg_metrics.get('dice_std', 0.0):.4f}**
- **Validation IoU:** {agg_metrics.get('iou_mean', 0.0):.4f} ± {agg_metrics.get('iou_std', 0.0):.4f}
- **Validation Precision:** {agg_metrics.get('precision_mean', 0.0):.4f} ± {agg_metrics.get('precision_std', 0.0):.4f}
- **Validation Recall:** {agg_metrics.get('recall_mean', 0.0):.4f} ± {agg_metrics.get('recall_std', 0.0):.4f}
- **Validation Specificity:** {agg_metrics.get('specificity_mean', 0.0):.4f} ± {agg_metrics.get('specificity_std', 0.0):.4f}
- **Validation HD95:** {agg_metrics.get('hd95_mean', 0.0):.2f} ± {agg_metrics.get('hd95_std', 0.0):.2f} px ({agg_metrics.get('hd95_valid_count', 0)}/125 valid cases)
- **Validation NSD @ 1px:** {agg_metrics.get('nsd_1px_mean', 0.0):.4f} ± {agg_metrics.get('nsd_1px_std', 0.0):.4f}
- **Validation NSD @ 2px:** {agg_metrics.get('nsd_2px_mean', 0.0):.4f} ± {agg_metrics.get('nsd_2px_std', 0.0):.4f}
- **Validation NSD @ 3px:** {agg_metrics.get('nsd_3px_mean', 0.0):.4f} ± {agg_metrics.get('nsd_3px_std', 0.0):.4f}
- **Training Duration:** {total_time / 60.0:.2f} minutes
- **Best Checkpoint Path:** `{best_ckpt_path}`
"""
    with open(os.path.join(exp_dir, "reports", "EXPERIMENT_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    print("=" * 70)
    print("OFFICIAL ATTENTION RESIDUAL U-NET EXPERIMENT COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    run_official_attention_residual_unet_experiment()
