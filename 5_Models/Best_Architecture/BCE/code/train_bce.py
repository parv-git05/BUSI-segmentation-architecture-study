"""
Official Experiment Script: Attention U-Net with Binary Cross-Entropy Loss (BCE).
Architecture: Attention U-Net (Oktay et al., 2018)
Dataset: BUSI Phase-1 Single-Mask Cohort (440 Train / 125 Val / 63 Test)
LOCKED TEST SET IS NEVER ACCESSED OR EVALUATED.
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Paths configuration
code_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.dirname(code_dir)
best_arch_dir = os.path.dirname(exp_dir)
models_root = os.path.dirname(best_arch_dir)
common_root = os.path.join(models_root, "00_Common")
att_unet_root = os.path.join(models_root, "02_Attention_UNet")

for p in [common_root, att_unet_root, models_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets.busi_dataset import create_dataloader, BUSIDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from training.losses import BCELoss
from utils.seed import seed_everything
from utils.checkpoint import save_checkpoint, load_checkpoint
from metrics.segmentation_metrics import calculate_batch_metrics, aggregate_metrics
from visualization.visualizer import plot_training_curves, plot_qualitative_predictions
from model import AttentionUNet


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("BCE_Experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def run_experiment():
    print("=" * 75)
    print("STARTING EXPERIMENT: Attention U-Net + Binary Cross-Entropy Loss (BCE)")
    print("=" * 75)

    # 1. Output directories
    subdirs = ["checkpoints", "metrics", "logs", "curves", "visualizations", "code"]
    for sub in subdirs:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    log_path = os.path.join(exp_dir, "logs", "train.log")
    logger = setup_logger(log_path)

    # 2. Experiment Configuration
    master_split_dir = os.path.join(models_root, "MASTER_SPLIT")
    config = {
        "experiment_name": "BCE",
        "architecture": "AttentionUNet",
        "loss_fn_name": "BCELoss",
        "loss_parameters": {
            "from_logits": True,
            "reduction": "mean"
        },
        "seed": 42,
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": 10,
        "scheduler_factor": 0.5,
        "input_size": [256, 256],
        "img_channels": 1,
        "normalization": [0.0, 1.0],
        "augmentation": {
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.5,
            "rotation_degrees": 15.0
        },
        "train_manifest": os.path.join(master_split_dir, "train_70.csv"),
        "val_manifest": os.path.join(master_split_dir, "val_20.csv"),
        "test_manifest": "LOCKED - NOT ACCESSED",
        "target_metric": "val_dice",
        "early_stopping": "DISABLED"
    }

    config_path = os.path.join(exp_dir, "code", "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # 3. Seed Reproducibility
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 4. DataLoaders (Train & Val ONLY)
    train_transform = get_train_transforms(input_size=tuple(config["input_size"]), augment=True)
    val_transform = get_val_transforms(input_size=tuple(config["input_size"]))

    train_loader = create_dataloader(
        manifest_path=config["train_manifest"],
        batch_size=config["batch_size"],
        shuffle=True,
        transform=train_transform,
        seed=config["seed"]
    )

    val_loader = create_dataloader(
        manifest_path=config["val_manifest"],
        batch_size=config["batch_size"],
        shuffle=False,
        transform=val_transform,
        seed=config["seed"]
    )

    logger.info(f"Loaded Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 5. Architecture, Loss, Optimizer, Scheduler
    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False).to(device)
    criterion = BCELoss(from_logits=True, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config["scheduler_factor"], patience=config["scheduler_patience"]
    )

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info("Starting 100-Epoch Training Protocol...")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice_mean": [],
        "val_iou_mean": [],
        "val_precision_mean": [],
        "val_recall_mean": [],
        "val_specificity_mean": [],
        "val_hd95_mean": [],
        "val_nsd_1px_mean": [],
        "learning_rate": []
    }

    best_val_dice = -float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        # Training Epoch
        model.train()
        train_loss_total = 0.0
        for sample in train_loader:
            images = sample["image"].to(device)
            masks = sample["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / max(1, len(train_loader))

        # Validation Epoch
        model.eval()
        val_loss_total = 0.0
        val_metrics_samples = []

        with torch.no_grad():
            for sample in val_loader:
                images = sample["image"].to(device)
                masks = sample["mask"].to(device)
                sample_ids = sample["sample_id"]

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss_total += loss.item()

                preds_binary = (torch.sigmoid(outputs) > 0.5).float()
                batch_metrics = calculate_batch_metrics(
                    preds=preds_binary,
                    gts=masks,
                    threshold=0.5,
                    sample_ids=sample_ids
                )
                val_metrics_samples.extend(batch_metrics)

        avg_val_loss = val_loss_total / max(1, len(val_loader))
        agg = aggregate_metrics(val_metrics_samples)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)

        epoch_dice = agg.get("dice_mean", 0.0)
        epoch_iou = agg.get("iou_mean", 0.0)
        epoch_hd95 = agg.get("hd95_mean", np.nan)
        hd95_str = f"{epoch_hd95:.2f}px" if not np.isnan(epoch_hd95) else "N/A"

        # Record History
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_dice_mean"].append(epoch_dice)
        history["val_iou_mean"].append(epoch_iou)
        history["val_precision_mean"].append(agg.get("precision_mean", 0.0))
        history["val_recall_mean"].append(agg.get("recall_mean", 0.0))
        history["val_specificity_mean"].append(agg.get("specificity_mean", 0.0))
        history["val_hd95_mean"].append(epoch_hd95)
        history["val_nsd_1px_mean"].append(agg.get("nsd_1px_mean", 0.0))
        history["learning_rate"].append(current_lr)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch [{epoch:03d}/{config['epochs']:03d}] ({epoch_time:.1f}s) - "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {epoch_dice:.4f} | Val IoU: {epoch_iou:.4f} | Val HD95: {hd95_str} | LR: {current_lr:.6f}"
        )

        # Checkpoints
        latest_ckpt_path = os.path.join(exp_dir, "checkpoints", "latest.pth")
        save_checkpoint(
            filepath=latest_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_metrics=agg,
            config=config,
            seed=config["seed"],
            experiment_id="BCE"
        )

        if epoch_dice > best_val_dice:
            best_val_dice = epoch_dice
            best_epoch = epoch
            best_ckpt_path = os.path.join(exp_dir, "checkpoints", "best_model.pth")
            save_checkpoint(
                filepath=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                val_metrics=agg,
                config=config,
                seed=config["seed"],
                experiment_id="BCE"
            )
            logger.info(f"  >>> New Best Model saved! (Val Dice: {best_val_dice:.4f} at epoch {epoch})")

    total_time = time.time() - start_time
    logger.info(f"Training Completed in {total_time/60.0:.2f} mins. Best Epoch: {best_epoch} (Val Dice: {best_val_dice:.4f})")

    # 6. Save Training Curves & History CSV
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(exp_dir, "curves", "epoch_history.csv"), index=False)
    plot_training_curves(
        history=history,
        save_path=os.path.join(exp_dir, "curves", "learning_curves.png"),
        title_suffix="(BCE Loss)"
    )

    # 7. Final Detailed Validation Evaluation using Best Checkpoint
    best_ckpt_path = os.path.join(exp_dir, "checkpoints", "best_model.pth")
    load_checkpoint(best_ckpt_path, model, device=device)
    model.eval()

    val_dataset_eval = BUSIDataset(manifest_path=config["val_manifest"], transform=val_transform)
    val_eval_loader = torch.utils.data.DataLoader(val_dataset_eval, batch_size=1, shuffle=False)

    detailed_metrics = []
    with torch.no_grad():
        for sample in val_eval_loader:
            imgs = sample["image"].to(device)
            msks = sample["mask"].to(device)
            sid = sample["sample_id"][0]

            outs = model(imgs)
            preds_bin = (torch.sigmoid(outs) > 0.5).float()
            m = calculate_batch_metrics(preds=preds_bin, gts=msks, threshold=0.5, sample_ids=[sid])
            detailed_metrics.append(m[0])

    df_detailed = pd.DataFrame(detailed_metrics)
    df_detailed.to_csv(os.path.join(exp_dir, "metrics", "val_per_image_metrics.csv"), index=False)

    final_agg = aggregate_metrics(detailed_metrics)
    with open(os.path.join(exp_dir, "metrics", "val_aggregate_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_agg, f, indent=4)

    # 8. Qualitative Predictions
    qual_indices = [0, 5, 12, 25, 40, 60, 80, 100]
    qual_images, qual_gts, qual_preds, qual_sids, qual_ms = [], [], [], [], []

    for idx in qual_indices:
        if idx < len(val_dataset_eval):
            s = val_dataset_eval[idx]
            img_t = s["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                out_t = model(img_t)
                p_mask = (torch.sigmoid(out_t) > 0.5).float().squeeze(0).cpu().numpy()

            qual_images.append(s["image"].numpy())
            qual_gts.append(s["mask"].numpy())
            qual_preds.append(p_mask)
            qual_sids.append(s["sample_id"])
            qual_ms.append(detailed_metrics[idx])

    plot_qualitative_predictions(
        images=qual_images,
        gt_masks=qual_gts,
        pred_masks=qual_preds,
        sample_ids=qual_sids,
        metrics_list=qual_ms,
        save_path=os.path.join(exp_dir, "visualizations", "val_qualitative_predictions.png"),
        max_samples=8
    )

    # 9. Summary & Experiment Reports
    summary_report = {
        "experiment_name": "BCE",
        "architecture": "AttentionUNet",
        "loss_function": "BCELoss (Binary Cross-Entropy with Logits)",
        "best_epoch": best_epoch,
        "best_val_dice": float(best_val_dice),
        "total_epochs": config["epochs"],
        "training_time_seconds": total_time,
        "validation_metrics": final_agg
    }
    with open(os.path.join(exp_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)

    best_ckpt_file = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
    latest_ckpt_file = os.path.join(exp_dir, 'checkpoints', 'latest.pth')
    d_mean = final_agg.get('dice_mean', 0.0)
    d_std = final_agg.get('dice_std', 0.0)
    i_mean = final_agg.get('iou_mean', 0.0)
    i_std = final_agg.get('iou_std', 0.0)
    p_mean = final_agg.get('precision_mean', 0.0)
    p_std = final_agg.get('precision_std', 0.0)
    r_mean = final_agg.get('recall_mean', 0.0)
    r_std = final_agg.get('recall_std', 0.0)
    s_mean = final_agg.get('specificity_mean', 0.0)
    s_std = final_agg.get('specificity_std', 0.0)
    h_mean = final_agg.get('hd95_mean', 0.0)
    h_std = final_agg.get('hd95_std', 0.0)
    h_valid = final_agg.get('hd95_valid_count', 0)
    n1_mean = final_agg.get('nsd_1px_mean', 0.0)
    n1_std = final_agg.get('nsd_1px_std', 0.0)
    n2_mean = final_agg.get('nsd_2px_mean', 0.0)
    n2_std = final_agg.get('nsd_2px_std', 0.0)
    n3_mean = final_agg.get('nsd_3px_mean', 0.0)
    n3_std = final_agg.get('nsd_3px_std', 0.0)

    report_md = f"""# Experiment Report: BCE (Attention U-Net)

- **Experiment Name:** `BCE`
- **Architecture:** Attention U-Net ({model.count_parameters():,} trainable parameters)
- **Loss Formulation:** Binary Cross-Entropy Loss (with logits)
- **Loss Parameters:**
  - `from_logits`: `True`
  - `reduction`: `mean`
- **Training Configuration:**
  - **Optimizer:** Adam (lr=1e-4, weight_decay=1e-4)
  - **Batch Size:** 16
  - **Epochs:** 100
  - **Early Stopping:** DISABLED
  - **LR Scheduler:** ReduceLROnPlateau (mode=min, factor=0.5, patience=10)
  - **Random Seed:** 42
  - **Input Size:** 256 × 256, Grayscale (1 channel)
  - **Normalization:** [0.0, 1.0]
  - **Data Augmentation:** Horizontal Flip (p=0.5), Vertical Flip (p=0.5), Rotation (±15°)
- **Dataset / Split Information:**
  - **Cohort:** BUSI Phase-1 single-mask dataset (628 samples)
  - **Split:** Master Split (Train: 440, Validation: 125, Test: 63)
  - **Test Set Status:** LOCKED & UNTOUCHED (0 test samples accessed or evaluated)
- **Best Epoch:** {best_epoch}
- **Best Validation Metrics (Mean ± Std):**
  - **Dice:** **{d_mean:.4f} ± {d_std:.4f}**
  - **IoU:** {i_mean:.4f} ± {i_std:.4f}
  - **Precision:** {p_mean:.4f} ± {p_std:.4f}
  - **Recall:** {r_mean:.4f} ± {r_std:.4f}
  - **Specificity:** {s_mean:.4f} ± {s_std:.4f}
  - **HD95:** {h_mean:.2f} ± {h_std:.2f} px ({h_valid}/125 valid cases)
  - **NSD @ 1px:** {n1_mean:.4f} ± {n1_std:.4f}
  - **NSD @ 2px:** {n2_mean:.4f} ± {n2_std:.4f}
  - **NSD @ 3px:** {n3_mean:.4f} ± {n3_std:.4f}
- **Checkpoint Locations:**
  - Best Checkpoint: `{best_ckpt_file}`
  - Latest Checkpoint: `{latest_ckpt_file}`
- **Training Duration:** {total_time / 60.0:.2f} minutes
"""
    with open(os.path.join(exp_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("=" * 75)
    logger.info("BCE EXPERIMENT FINISHED SUCCESSFULLY.")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_experiment()
