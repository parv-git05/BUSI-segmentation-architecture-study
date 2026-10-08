"""
Official Final Held-Out Test Evaluation Script for BUSI Attention U-Net.
Evaluates the locked BCE_Dice Epoch-97 Best-Validation Checkpoint on test_10.csv (63 samples).
Adheres strictly to research integrity standards:
- Frozen architecture, loss, threshold (0.5), preprocessing, and checkpoint.
- First official test set evaluation.
- Preserves NaN handling for HD95.
- Computes macro-averaged Dice, IoU, Precision, Recall, Specificity, HD95, NSD @ 1, 2, 3 px.
- Saves per-image CSV, aggregate JSON, checkpoint metadata JSON, predictions, visualizations, and logs.
"""

import os
import sys
import json
import time
import datetime
import logging
import platform
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Configure paths
ws_root = os.path.abspath("f:/BUSI-segmentation-architecture-study")
models_root = os.path.join(ws_root, "5_Models")
common_root = os.path.join(models_root, "00_Common")
best_arch_root = os.path.join(models_root, "Best_Architecture")
bce_dice_dir = os.path.join(best_arch_root, "BCE_Dice")
att_unet_root = os.path.join(models_root, "02_Attention_UNet")
master_split_dir = os.path.join(models_root, "MASTER_SPLIT")
output_dir = os.path.join(best_arch_root, "FINAL_TEST_EVALUATION")

for p in [common_root, att_unet_root, models_root, best_arch_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import AttentionUNet
from datasets.busi_dataset import BUSIDataset
from datasets.transforms import get_test_transforms
from metrics.segmentation_metrics import calculate_sample_metrics, aggregate_metrics
from visualization.visualizer import plot_qualitative_predictions, plot_metric_distributions


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("FinalTestEvaluation")
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


def run_test_evaluation():
    start_time = time.time()
    eval_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Prepare directory structure
    subdirs = ["test_predictions", "visualizations", "logs"]
    for s in subdirs:
        os.makedirs(os.path.join(output_dir, s), exist_ok=True)

    log_path = os.path.join(output_dir, "logs", "test_eval.log")
    logger = setup_logger(log_path)

    logger.info("=" * 80)
    logger.info("FINAL HELD-OUT TEST EVALUATION — BUSI ATTENTION U-NET")
    logger.info("=" * 80)
    logger.info(f"Evaluation Timestamp: {eval_timestamp}")
    logger.info(f"Output Directory: {output_dir}")

    # 2. Audit Verification Confirmation
    audit_file = os.path.join(ws_root, "scratch", "pre_eval_audit_results.json")
    if not os.path.exists(audit_file):
        raise RuntimeError("Pre-evaluation integrity audit results not found! Must run audit first.")
    with open(audit_file, "r") as f:
        audit_res = json.load(f)
    if not audit_res.get("all_passed", False):
        raise RuntimeError("Integrity audit DID NOT pass all checks! Execution halted.")
    logger.info("PRE-EVALUATION AUDIT STATUS: ALL CHECKS PASSED (A through L)")

    # 3. Model & Checkpoint Loading
    ckpt_path = os.path.join(bce_dice_dir, "checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False)
    param_count = model.count_parameters()
    logger.info(f"Model instantiated: Attention U-Net with {param_count:,} trainable parameters.")
    assert param_count == 31387049, f"Parameter mismatch: expected 31387049, got {param_count}"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch")
    exp_id = ckpt.get("experiment_id")
    logger.info(f"Loaded checkpoint from: {ckpt_path}")
    logger.info(f"Checkpoint Experiment ID: {exp_id} | Epoch: {epoch}")
    assert epoch == 97, f"Expected Epoch 97, got {epoch}"
    assert exp_id == "BCE_Dice", f"Expected experiment BCE_Dice, got {exp_id}"

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    # 4. Dataset & DataLoader (Test Set)
    test_manifest_path = os.path.join(master_split_dir, "test_10.csv")
    test_transforms = get_test_transforms(input_size=(256, 256))
    
    test_dataset = BUSIDataset(
        manifest_path=test_manifest_path,
        transform=test_transforms,
        img_channels=1,
        return_metadata=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_len = len(test_dataset)
    logger.info(f"Loaded Test Dataset from {test_manifest_path} with {test_len} samples.")
    assert test_len == 63, f"Expected exactly 63 test samples, got {test_len}"

    # 5. Inference Loop & Metric Evaluation
    logger.info("Executing frozen test inference (threshold=0.5, no TTA)...")
    per_image_results = []
    qual_images = []
    qual_gts = []
    qual_preds = []
    qual_sids = []
    qual_metrics = []

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            img_tensor = sample["image"].to(device)       # Shape (1, 1, 256, 256)
            gt_tensor = sample["mask"].to(device)         # Shape (1, 1, 256, 256)
            sample_id = sample["sample_id"][0]
            class_name = sample["class_name"][0]
            class_id = int(sample["class_id"][0])
            img_path = sample["image_path"][0]
            mask_path = sample["mask_path"][0]

            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
            preds_bin = (probs > 0.5).float()

            pred_np = preds_bin.squeeze().cpu().numpy().astype(bool)
            gt_np = gt_tensor.squeeze().cpu().numpy().astype(bool)
            prob_np = probs.squeeze().cpu().numpy()

            # Calculate individual sample metrics
            metrics = calculate_sample_metrics(
                pred=pred_np,
                gt=gt_np,
                threshold=0.5,
                sample_id=sample_id
            )

            # Record sample metadata
            record = {
                "sample_id": sample_id,
                "class": class_name,
                "class_id": class_id,
                "image_path": img_path,
                "mask_path": mask_path,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "hd95": metrics["hd95"],
                "nsd_1px": metrics["nsd_1px"],
                "nsd_2px": metrics["nsd_2px"],
                "nsd_3px": metrics["nsd_3px"],
                "gt_pixels": int(gt_np.sum()),
                "pred_pixels": int(pred_np.sum())
            }
            per_image_results.append(record)

            # Save prediction mask image (PNG 0/255)
            pred_img = Image.fromarray((pred_np.astype(np.uint8) * 255))
            pred_filename = f"{os.path.splitext(sample_id)[0]}_pred.png"
            pred_save_path = os.path.join(output_dir, "test_predictions", pred_filename)
            pred_img.save(pred_save_path)

            # Store for qualitative visualizations
            qual_images.append(sample["image"].squeeze().cpu().numpy())
            qual_gts.append(gt_np.astype(np.float32))
            qual_preds.append(pred_np.astype(np.float32))
            qual_sids.append(f"{class_name}/{sample_id}")
            qual_metrics.append(metrics)

            hd95_str = f"{metrics['hd95']:.2f}px" if not np.isnan(metrics['hd95']) else "NaN"
            logger.info(
                f"[{i+1:02d}/63] Sample: {sample_id:<22} | Class: {class_name:<9} | "
                f"Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | "
                f"HD95: {hd95_str:>7} | NSD@1: {metrics['nsd_1px']:.4f}"
            )

    # 6. Aggregate Metrics Calculation
    df_per_image = pd.DataFrame(per_image_results)
    per_image_csv_path = os.path.join(output_dir, "test_per_image_metrics.csv")
    df_per_image.to_csv(per_image_csv_path, index=False)
    logger.info(f"Saved complete per-image metrics to: {per_image_csv_path}")

    agg_test_metrics = aggregate_metrics(per_image_results)
    agg_test_metrics["total_samples"] = test_len
    agg_test_metrics["evaluated_samples"] = len(per_image_results)

    # Breakdown by class
    class_breakdown = {}
    for c_name in df_per_image["class"].unique():
        sub_df = df_per_image[df_per_image["class"] == c_name]
        class_breakdown[c_name] = {
            "count": len(sub_df),
            "dice_mean": float(sub_df["dice"].mean()),
            "dice_std": float(sub_df["dice"].std()),
            "iou_mean": float(sub_df["iou"].mean()),
            "iou_std": float(sub_df["iou"].std()),
            "precision_mean": float(sub_df["precision"].mean()),
            "recall_mean": float(sub_df["recall"].mean()),
            "specificity_mean": float(sub_df["specificity"].mean()),
            "hd95_mean": float(np.nanmean(sub_df["hd95"])),
            "hd95_std": float(np.nanstd(sub_df["hd95"])),
            "hd95_valid_count": int(sub_df["hd95"].notna().sum()),
            "nsd_1px_mean": float(sub_df["nsd_1px"].mean()),
            "nsd_2px_mean": float(sub_df["nsd_2px"].mean()),
            "nsd_3px_mean": float(sub_df["nsd_3px"].mean()),
        }

    agg_output = {
        "overall": agg_test_metrics,
        "by_class": class_breakdown
    }

    test_agg_json_path = os.path.join(output_dir, "test_aggregate_metrics.json")
    with open(test_agg_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_output, f, indent=4)
    logger.info(f"Saved aggregate test metrics to: {test_agg_json_path}")

    # 7. Checkpoint Info & Reproducibility Metadata
    checkpoint_info = {
        "experiment_name": "BCE_Dice",
        "architecture": "Attention U-Net",
        "trainable_parameters": param_count,
        "checkpoint_file": ckpt_path,
        "checkpoint_epoch": epoch,
        "loss_formulation": "Combined Binary Cross-Entropy and Soft Dice Loss (BCEDiceLoss)",
        "loss_hyperparameters": {
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "smooth": 1e-6,
            "from_logits": True,
            "reduction": "mean"
        },
        "training_optimizer": "Adam (lr=1e-4, weight_decay=1e-4)",
        "inference_threshold": 0.5,
        "preprocessing": {
            "mode": "deterministic validation/test preprocessing",
            "channels": 1,
            "color_mode": "grayscale",
            "spatial_size": [256, 256],
            "image_interpolation": "BILINEAR",
            "mask_interpolation": "NEAREST",
            "value_range": [0.0, 1.0],
            "test_time_augmentation": False
        },
        "test_manifest": test_manifest_path,
        "test_sample_count": test_len,
        "evaluation_timestamp": eval_timestamp,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "os": platform.platform()
        },
        "model_selection_policy": {
            "selection_criterion": "Highest validation Dice macro-average",
            "selection_subset": "val_20.csv (125 samples)",
            "test_set_access_status": "LOCKED until final evaluation; 0 test samples accessed during selection or training"
        }
    }
    ckpt_info_json_path = os.path.join(output_dir, "checkpoint_info.json")
    with open(ckpt_info_json_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_info, f, indent=4)
    logger.info(f"Saved checkpoint reproducibility info to: {ckpt_info_json_path}")

    # 8. Qualitative Visualizations
    # Select representative samples: 2 best, 2 median, 2 challenging, plus diverse classes
    sorted_by_dice = sorted(range(len(per_image_results)), key=lambda k: per_image_results[k]["dice"], reverse=True)
    best_indices = sorted_by_dice[:3]
    mid_idx = len(sorted_by_dice) // 2
    median_indices = sorted_by_dice[mid_idx - 1: mid_idx + 2]
    hard_indices = sorted_by_dice[-3:]
    selected_indices = best_indices + median_indices + hard_indices

    plot_qualitative_predictions(
        images=[qual_images[k] for k in selected_indices],
        gt_masks=[qual_gts[k] for k in selected_indices],
        pred_masks=[qual_preds[k] for k in selected_indices],
        sample_ids=[qual_sids[k] for k in selected_indices],
        metrics_list=[qual_metrics[k] for k in selected_indices],
        save_path=os.path.join(output_dir, "visualizations", "test_qualitative_predictions.png"),
        max_samples=len(selected_indices)
    )
    logger.info("Saved representative qualitative grid to visualizations/test_qualitative_predictions.png")

    # Plot metric distributions
    plot_metric_distributions(
        per_sample_df=df_per_image,
        save_path=os.path.join(output_dir, "visualizations", "test_metric_distributions.png"),
        metrics=["dice", "iou", "precision", "recall", "specificity", "nsd_1px"]
    )
    logger.info("Saved metric distributions to visualizations/test_metric_distributions.png")

    # 9. Summary logging
    logger.info("=" * 80)
    logger.info("FINAL TEST SET EVALUATION SUMMARY (N=63)")
    logger.info("=" * 80)
    logger.info(f"Dice Coefficient:       {agg_test_metrics['dice_mean']:.4f} ± {agg_test_metrics['dice_std']:.4f}")
    logger.info(f"Intersection over Union:{agg_test_metrics['iou_mean']:.4f} ± {agg_test_metrics['iou_std']:.4f}")
    logger.info(f"Precision:              {agg_test_metrics['precision_mean']:.4f} ± {agg_test_metrics['precision_std']:.4f}")
    logger.info(f"Recall (Sensitivity):   {agg_test_metrics['recall_mean']:.4f} ± {agg_test_metrics['recall_std']:.4f}")
    logger.info(f"Specificity:            {agg_test_metrics['specificity_mean']:.4f} ± {agg_test_metrics['specificity_std']:.4f}")
    logger.info(f"HD95 (pixels):          {agg_test_metrics['hd95_mean']:.2f} ± {agg_test_metrics['hd95_std']:.2f} px (valid: {agg_test_metrics['hd95_valid_count']}/63)")
    logger.info(f"NSD @ 1px:              {agg_test_metrics['nsd_1px_mean']:.4f} ± {agg_test_metrics['nsd_1px_std']:.4f}")
    logger.info(f"NSD @ 2px:              {agg_test_metrics['nsd_2px_mean']:.4f} ± {agg_test_metrics['nsd_2px_std']:.4f}")
    logger.info(f"NSD @ 3px:              {agg_test_metrics['nsd_3px_mean']:.4f} ± {agg_test_metrics['nsd_3px_std']:.4f}")
    logger.info("=" * 80)

    elapsed_time = time.time() - start_time
    logger.info(f"Test evaluation completed in {elapsed_time:.2f} seconds.")

    return {
        "checkpoint_info": checkpoint_info,
        "aggregate_metrics": agg_output,
        "per_image_df": df_per_image
    }


if __name__ == "__main__":
    run_test_evaluation()
