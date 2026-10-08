"""
Master Terminal-Runnable Sequential Architecture Test Evaluation Script.
Evaluates all five previously trained segmentation architectures:
1. U-Net
2. Attention U-Net
3. Residual U-Net
4. Attention Residual U-Net
5. U-Net++

on the EXACT SAME locked held-out test set (test_10.csv, N=63) using the baseline Dice loss checkpoints.
Strictly adheres to research integrity:
- NO retraining.
- NO threshold tuning (fixed at 0.5).
- NO test-time augmentation.
- NO model selection on the test set.
- Macro-averaged metrics with NaN-aware handling for HD95.
- Sequential evaluation with immediate halt on any failure.
"""

import os
import sys
import json
import time
import argparse
import datetime
import platform
import importlib.util
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn

# Base workspace paths
current_dir = os.path.dirname(os.path.abspath(__file__))
models_root = os.path.dirname(current_dir)
ws_root = os.path.dirname(models_root)
common_root = os.path.join(models_root, "00_Common")
master_split_dir = os.path.join(models_root, "MASTER_SPLIT")
test_manifest_path = os.path.join(master_split_dir, "test_10.csv")

# Ensure common root is in sys.path
if common_root not in sys.path:
    sys.path.insert(0, common_root)

from datasets.busi_dataset import BUSIDataset
from datasets.transforms import get_test_transforms
from metrics.segmentation_metrics import calculate_sample_metrics, aggregate_metrics
from visualization.visualizer import plot_qualitative_predictions

# Default output directory
DEFAULT_OUTPUT_DIR = os.path.join(current_dir, "FINAL_ARCHITECTURE_TEST_EVALUATION")

# Specifications for the 5 architectures in exact required sequence
ARCHITECTURE_SPECS = [
    {
        "id": 1,
        "key": "U_Net",
        "display_name": "U-Net",
        "model_file": os.path.join(models_root, "01_UNet", "model.py"),
        "class_name": "UNet",
        "init_kwargs": {"in_channels": 1, "out_channels": 1, "base_channels": 64, "bilinear": False},
        "ckpt_path": os.path.join(models_root, "01_UNet", "experiments", "EXP-ARCH-01_UNET", "checkpoints", "best_model.pth"),
        "expected_epoch": 93,
        "expected_params": 31036481,
        "loss_fn": "DiceLoss"
    },
    {
        "id": 2,
        "key": "Attention_U_Net",
        "display_name": "Attention U-Net",
        "model_file": os.path.join(models_root, "02_Attention_UNet", "model.py"),
        "class_name": "AttentionUNet",
        "init_kwargs": {"in_channels": 1, "out_channels": 1, "base_channels": 64, "bilinear": False},
        "ckpt_path": os.path.join(models_root, "02_Attention_UNet", "experiments", "EXP-ARCH-02_ATTENTION_UNET", "checkpoints", "best_model.pth"),
        "expected_epoch": 96,
        "expected_params": 31387049,
        "loss_fn": "DiceLoss"
    },
    {
        "id": 3,
        "key": "Residual_U_Net",
        "display_name": "Residual U-Net",
        "model_file": os.path.join(models_root, "03_Residual_UNet", "model.py"),
        "class_name": "ResidualUNet",
        "init_kwargs": {"in_channels": 1, "out_channels": 1, "base_channels": 64, "bilinear": False},
        "ckpt_path": os.path.join(models_root, "03_Residual_UNet", "experiments", "EXP-ARCH-03_RESIDUAL_UNET", "checkpoints", "best_model.pth"),
        "expected_epoch": 89,
        "expected_params": 32435073,
        "loss_fn": "DiceLoss"
    },
    {
        "id": 4,
        "key": "Attention_Residual_U_Net",
        "display_name": "Attention Residual U-Net",
        "model_file": os.path.join(models_root, "04_Attention_Residual_UNet", "model.py"),
        "class_name": "AttentionResidualUNet",
        "init_kwargs": {"in_channels": 1, "out_channels": 1, "base_channels": 64, "bilinear": False},
        "ckpt_path": os.path.join(models_root, "04_Attention_Residual_UNet", "experiments", "EXP-ARCH-04_ATTENTION_RESIDUAL_UNET", "checkpoints", "best_model.pth"),
        "expected_epoch": 93,
        "expected_params": 32785641,
        "loss_fn": "DiceLoss"
    },
    {
        "id": 5,
        "key": "U_Net_PlusPlus",
        "display_name": "U-Net++",
        "model_file": os.path.join(models_root, "05_UNetPP", "model.py"),
        "class_name": "UNetPlusPlus",
        "init_kwargs": {"in_channels": 1, "out_channels": 1, "base_channels": 64, "deep_supervision": False},
        "ckpt_path": os.path.join(models_root, "05_UNetPP", "experiments", "EXP-ARCH-05_UNETPP", "checkpoints", "best_model.pth"),
        "expected_epoch": 88,
        "expected_params": 36164097,
        "loss_fn": "DiceLoss"
    }
]

# Fixed deterministic qualitative indices for visualization across all models (4 benign, 3 malignant)
QUAL_SAMPLE_INDICES = [0, 15, 30, 41, 42, 52, 62]


def build_model_instance(spec: Dict[str, Any]) -> nn.Module:
    """Dynamically loads and instantiates architecture module cleanly."""
    mod_name = f"arch_mod_{spec['key']}"
    module_spec = importlib.util.spec_from_file_location(mod_name, spec["model_file"])
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load module from {spec['model_file']}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    cls = getattr(module, spec["class_name"])
    model = cls(**spec["init_kwargs"])
    return model


def run_dry_run_checks(device: torch.device) -> bool:
    """
    Performs comprehensive pre-flight verification:
    1. Test manifest existence, total sample count (63), class breakdown (42 benign, 21 malignant).
    2. Checkpoint existence, epoch verification, parameter counts, strict loading for all 5 models.
    3. Single-sample forward inference, dimension verification, thresholding, and metric calculation.
    """
    print("\n" + "=" * 80)
    print("STARTING ARCHITECTURE EVALUATION READINESS / DRY-RUN AUDIT")
    print("=" * 80)

    # 1. Test Manifest Verification
    if not os.path.exists(test_manifest_path):
        raise FileNotFoundError(f"Test manifest not found at: {test_manifest_path}")
    test_df = pd.read_csv(test_manifest_path)
    total_samples = len(test_df)
    benign_count = (test_df["class"] == "benign").sum()
    malignant_count = (test_df["class"] == "malignant").sum()
    print(f"[AUDIT] Test manifest loaded: {test_manifest_path}")
    print(f"        Total: {total_samples} samples | Benign: {benign_count} | Malignant: {malignant_count}")

    if total_samples != 63:
        raise ValueError(f"Expected 63 test samples, got {total_samples}")
    if benign_count != 42 or malignant_count != 21:
        raise ValueError(f"Expected 42 benign and 21 malignant, got {benign_count} and {malignant_count}")
    print("[PASS] Test manifest sample counts strictly match (63 total: 42 benign, 21 malignant).")

    # 2. Check each architecture
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    dummy_gt = (torch.rand(1, 1, 256, 256) > 0.5).float().to(device)

    for spec in ARCHITECTURE_SPECS:
        name = spec["display_name"]
        print(f"\n--- Checking Model {spec['id']}/5: {name} ---")
        
        # File existence
        if not os.path.exists(spec["model_file"]):
            raise FileNotFoundError(f"Model file missing for {name}: {spec['model_file']}")
        if not os.path.exists(spec["ckpt_path"]):
            raise FileNotFoundError(f"Checkpoint file missing for {name}: {spec['ckpt_path']}")

        # Model instantiation
        model = build_model_instance(spec)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable parameters: {param_count:,} (expected: {spec['expected_params']:,})")
        if param_count != spec["expected_params"]:
            raise ValueError(f"Parameter count mismatch for {name}: {param_count} != {spec['expected_params']}")

        # Checkpoint loading
        ckpt = torch.load(spec["ckpt_path"], map_location="cpu", weights_only=False)
        ckpt_epoch = ckpt.get("epoch")
        print(f"    Checkpoint epoch: {ckpt_epoch} (expected: {spec['expected_epoch']})")
        if ckpt_epoch != spec["expected_epoch"]:
            raise ValueError(f"Epoch mismatch for {name}: {ckpt_epoch} != {spec['expected_epoch']}")

        load_res = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if len(load_res.missing_keys) > 0 or len(load_res.unexpected_keys) > 0:
            raise ValueError(f"Strict load failed for {name}: missing={load_res.missing_keys}, unexpected={load_res.unexpected_keys}")
        print(f"    Strict checkpoint parameter key match: 100% PASS")

        # Test forward pass on device
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
            if out.shape != (1, 1, 256, 256):
                raise ValueError(f"Unexpected output shape: {out.shape}")
            probs = torch.sigmoid(out)
            preds_bin = (probs >= 0.5).float()
            metrics = calculate_sample_metrics(pred=preds_bin.squeeze().cpu().numpy(), gt=dummy_gt.squeeze().cpu().numpy(), threshold=0.5, sample_id="dryrun_sample")
            if "dice" not in metrics or "hd95" not in metrics:
                raise ValueError(f"Metric calculation failure on dummy forward pass for {name}")

        print(f"    Dummy forward pass & metric computation: PASS (Dice: {metrics['dice']:.4f})")

    # 3. Test Dataset & Transform pipeline
    test_transforms = get_test_transforms(input_size=(256, 256))
    dataset = BUSIDataset(manifest_path=test_manifest_path, transform=test_transforms, img_channels=1, return_metadata=True)
    first_sample = dataset[0]
    print(f"\n[AUDIT] Dataset & transforms verified: sample 0 shape={first_sample['image'].shape}, id={first_sample['sample_id']}")

    print("\n" + "=" * 80)
    print("ALL READINESS AND DRY-RUN CHECKS PASSED PERFECTLY FOR ALL 5 ARCHITECTURES!")
    print("=" * 80 + "\n")
    return True


def evaluate_single_architecture(
    spec: Dict[str, Any],
    dataset: BUSIDataset,
    output_base_dir: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates one architecture deterministically on the full test set:
    - Saves per-image metrics CSV
    - Saves aggregate metrics JSON
    - Saves checkpoint reproducibility info JSON
    - Saves predicted binary mask PNGs
    - Saves qualitative predictions grid PNG
    """
    name = spec["display_name"]
    arch_key = spec["key"]
    arch_out_dir = os.path.join(output_base_dir, arch_key)
    pred_dir = os.path.join(arch_out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"EVALUATING ARCHITECTURE {spec['id']}/5: {name.upper()}")
    print(f"Checkpoint: {spec['ckpt_path']}")
    print(f"Epoch: {spec['expected_epoch']} | Params: {spec['expected_params']:,}")
    print("#" * 80)

    model = build_model_instance(spec)
    ckpt = torch.load(spec["ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    per_image_results = []
    
    qual_images = []
    qual_gts = []
    qual_preds = []
    qual_sids = []
    qual_metrics = []

    start_eval_time = time.time()
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            img_tensor = sample["image"].to(device)       # (1, 1, 256, 256)
            gt_tensor = sample["mask"].to(device)         # (1, 1, 256, 256)
            sample_id = sample["sample_id"][0]
            class_name = sample["class_name"][0]
            class_id = int(sample["class_id"][0])
            img_path = sample["image_path"][0]
            mask_path = sample["mask_path"][0]

            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
            preds_bin = (probs >= 0.5).float()

            pred_np = preds_bin.squeeze().cpu().numpy().astype(bool)
            gt_np = gt_tensor.squeeze().cpu().numpy().astype(bool)

            metrics = calculate_sample_metrics(
                pred=pred_np,
                gt=gt_np,
                threshold=0.5,
                sample_id=sample_id
            )

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

            # Save prediction mask PNG (0/255)
            pred_img = Image.fromarray((pred_np.astype(np.uint8) * 255))
            pred_filename = f"{os.path.splitext(sample_id)[0]}_pred.png"
            pred_img.save(os.path.join(pred_dir, pred_filename))

            # Store qualitative candidates
            if i in QUAL_SAMPLE_INDICES:
                qual_images.append(sample["image"].squeeze().cpu().numpy())
                qual_gts.append(gt_np.astype(np.float32))
                qual_preds.append(pred_np.astype(np.float32))
                qual_sids.append(f"{class_name}/{sample_id}")
                qual_metrics.append(metrics)

            hd95_str = f"{metrics['hd95']:.2f}px" if not np.isnan(metrics['hd95']) else "NaN"
            print(f"[{name}] [{i+1:02d}/63] {sample_id:<22} ({class_name:<9}) | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | HD95: {hd95_str:>7} | NSD@1: {metrics['nsd_1px']:.4f}")

    eval_duration = time.time() - start_eval_time

    # 1. Save per-image metrics CSV
    df_per_image = pd.DataFrame(per_image_results)
    per_image_csv_path = os.path.join(arch_out_dir, "test_per_image_metrics.csv")
    df_per_image.to_csv(per_image_csv_path, index=False)

    # 2. Compute aggregate metrics
    agg_overall = aggregate_metrics(per_image_results)
    agg_overall["total_samples"] = len(per_image_results)

    # By class breakdown
    class_breakdown = {}
    for c_name in ["benign", "malignant"]:
        sub_df = df_per_image[df_per_image["class"] == c_name]
        class_breakdown[c_name] = {
            "count": len(sub_df),
            "dice_mean": float(sub_df["dice"].mean()),
            "dice_std": float(sub_df["dice"].std()),
            "iou_mean": float(sub_df["iou"].mean()),
            "iou_std": float(sub_df["iou"].std()),
            "precision_mean": float(sub_df["precision"].mean()),
            "precision_std": float(sub_df["precision"].std()),
            "recall_mean": float(sub_df["recall"].mean()),
            "recall_std": float(sub_df["recall"].std()),
            "specificity_mean": float(sub_df["specificity"].mean()),
            "specificity_std": float(sub_df["specificity"].std()),
            "hd95_mean": float(np.nanmean(sub_df["hd95"])),
            "hd95_std": float(np.nanstd(sub_df["hd95"])),
            "hd95_valid_count": int(sub_df["hd95"].notna().sum()),
            "nsd_1px_mean": float(sub_df["nsd_1px"].mean()),
            "nsd_1px_std": float(sub_df["nsd_1px"].std()),
            "nsd_2px_mean": float(sub_df["nsd_2px"].mean()),
            "nsd_2px_std": float(sub_df["nsd_2px"].std()),
            "nsd_3px_mean": float(sub_df["nsd_3px"].mean()),
            "nsd_3px_std": float(sub_df["nsd_3px"].std()),
        }

    agg_output = {
        "architecture": name,
        "overall": agg_overall,
        "by_class": class_breakdown
    }
    agg_json_path = os.path.join(arch_out_dir, "test_aggregate_metrics.json")
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_output, f, indent=4)

    # 3. Checkpoint info JSON
    ckpt_info = {
        "architecture_name": name,
        "architecture_key": arch_key,
        "checkpoint_file": spec["ckpt_path"],
        "checkpoint_epoch": spec["expected_epoch"],
        "trainable_parameters": spec["expected_params"],
        "loss_function": "DiceLoss (Baseline)",
        "test_manifest": test_manifest_path,
        "test_sample_count": len(per_image_results),
        "inference_threshold": 0.5,
        "preprocessing": "Grayscale, 256x256, [0,1], no augmentation",
        "evaluation_duration_seconds": eval_duration,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(arch_out_dir, "checkpoint_info.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt_info, f, indent=4)

    # 4. Qualitative Visualization
    plot_qualitative_predictions(
        images=qual_images,
        gt_masks=qual_gts,
        pred_masks=qual_preds,
        sample_ids=qual_sids,
        metrics_list=qual_metrics,
        save_path=os.path.join(arch_out_dir, "qualitative_predictions.png"),
        max_samples=len(qual_images)
    )

    print(f"\n[DONE] {name} Evaluation Summary:")
    print(f"       Dice: {agg_overall['dice_mean']:.4f} ± {agg_overall['dice_std']:.4f}")
    print(f"       IoU:  {agg_overall['iou_mean']:.4f} ± {agg_overall['iou_std']:.4f}")
    print(f"       HD95: {agg_overall['hd95_mean']:.2f} ± {agg_overall['hd95_std']:.2f} px ({agg_overall['hd95_valid_count']}/63 valid)")
    print(f"       Artifacts saved in: {arch_out_dir}")

    return {
        "spec": spec,
        "agg": agg_output,
        "df": df_per_image
    }


def generate_comparison_report(
    eval_results: List[Dict[str, Any]],
    output_base_dir: str
):
    """
    Constructs the consolidated CSV, JSON, and comprehensive Markdown report.
    """
    print("\n" + "=" * 80)
    print("GENERATING CONSOLIDATED ARCHITECTURE COMPARISON ARTIFACTS")
    print("=" * 80)

    rows = []
    for res in eval_results:
        spec = res["spec"]
        agg_overall = res["agg"]["overall"]
        by_class = res["agg"]["by_class"]

        row = {
            "Architecture": spec["display_name"],
            "Checkpoint Epoch": spec["expected_epoch"],
            "Parameter Count": spec["expected_params"],
            "Test N": agg_overall["total_samples"],
            "Benign N": by_class["benign"]["count"],
            "Malignant N": by_class["malignant"]["count"],
            "Dice Mean": agg_overall["dice_mean"],
            "Dice SD": agg_overall["dice_std"],
            "IoU Mean": agg_overall["iou_mean"],
            "IoU SD": agg_overall["iou_std"],
            "Precision Mean": agg_overall["precision_mean"],
            "Precision SD": agg_overall["precision_std"],
            "Recall Mean": agg_overall["recall_mean"],
            "Recall SD": agg_overall["recall_std"],
            "Specificity Mean": agg_overall["specificity_mean"],
            "Specificity SD": agg_overall["specificity_std"],
            "HD95 Mean": agg_overall["hd95_mean"],
            "HD95 SD": agg_overall["hd95_std"],
            "HD95 Valid N": agg_overall["hd95_valid_count"],
            "NSD@1 Mean": agg_overall["nsd_1px_mean"],
            "NSD@1 SD": agg_overall["nsd_1px_std"],
            "NSD@2 Mean": agg_overall["nsd_2px_mean"],
            "NSD@2 SD": agg_overall["nsd_2px_std"],
            "NSD@3 Mean": agg_overall["nsd_3px_mean"],
            "NSD@3 SD": agg_overall["nsd_3px_std"]
        }
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    comp_csv_path = os.path.join(output_base_dir, "test_architecture_comparison.csv")
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"[SAVED] Comparison CSV: {comp_csv_path}")

    # JSON output
    comp_json_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_dataset": "BUSI Phase-1 Single-Mask Cohort (test_10.csv)",
        "sample_counts": {"total": 63, "benign": 42, "malignant": 21},
        "common_loss": "DiceLoss (Baseline, smooth=1e-6, from_logits=True)",
        "inference_threshold": 0.5,
        "models": rows,
        "class_breakdowns": {res["spec"]["display_name"]: res["agg"]["by_class"] for res in eval_results}
    }
    comp_json_path = os.path.join(output_base_dir, "test_architecture_comparison.json")
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comp_json_data, f, indent=4)
    print(f"[SAVED] Comparison JSON: {comp_json_path}")

    # Evaluation Metadata JSON
    eval_meta = {
        "evaluation_timestamp": comp_json_data["timestamp"],
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models_evaluated": [spec["display_name"] for spec in ARCHITECTURE_SPECS],
        "test_manifest": test_manifest_path,
        "input_dimensions": [1, 256, 256],
        "inference_threshold": 0.5,
        "loss_function": "Dice Loss"
    }
    with open(os.path.join(output_base_dir, "evaluation_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(eval_meta, f, indent=4)

    # Markdown Report
    md_content = """# Final Architecture Test Evaluation Report: BUSI Segmentation Benchmark

---

## 1. Study Objective
This report provides the final, unbiased held-out test evaluation comparing all five deep learning segmentation architectures benchmarked under the common baseline Dice loss protocol. The evaluated models represent the official architecture benchmark cohort:
1. Standard U-Net
2. Attention U-Net
3. Residual U-Net
4. Attention Residual U-Net
5. U-Net++

Each architecture was trained using the identical Dice Loss objective on the 70% training cohort (`train_70.csv`, N=440), checkpointed at its optimal validation-Dice epoch on the 20% validation cohort (`val_20.csv`, N=125), and evaluated deterministically here on the locked 10% test set (`test_10.csv`, N=63).

---

## 2. Test Dataset & Isolation
- **Dataset:** BUSI Phase-1 Single-Mask Cohort
- **Manifest:** `5_Models/MASTER_SPLIT/test_10.csv`
- **Total Test Samples:** 63 images
- **Class Breakdown:**
  - **Benign:** 42 images (66.67%)
  - **Malignant:** 21 images (33.33%)
- **Test Isolation Statement:** The test set was locked and untouched throughout model design, hyperparameter selection, loss benchmarking, and checkpoint selection. All five models are evaluated once on the identical test cohort.

---

## 3. Experimental Controls (Frozen Across All Five Architectures)
- **Loss Function:** Dice Loss (`DiceLoss`, `smooth=1e-6`, `from_logits=True`)
- **Optimizer During Training:** Adam (`lr=1e-4`, `weight_decay=1e-4`, `batch_size=16`, `epochs=100`)
- **Preprocessing:**
  - Color mode: 1-channel Grayscale
  - Spatial resolution: 256 × 256 pixels (Image: Bilinear, Mask: Nearest Neighbor)
  - Value range: $[0.0, 1.0]$
  - Augmentation: None (strictly deterministic inference)
- **Inference Threshold:** Fixed at **0.5** applied to sigmoid probabilities: `Prediction = (sigmoid(logits) >= 0.5)`.

---

## 4. Models and Checkpoints Evaluated
| Architecture | Checkpoint File | Best Epoch | Trainable Parameters |
| :--- | :--- | :---: | :---: |
| **U-Net** | `01_UNet/experiments/EXP-ARCH-01_UNET/checkpoints/best_model.pth` | 93 | 31,036,481 |
| **Attention U-Net** | `02_Attention_UNet/experiments/EXP-ARCH-02_ATTENTION_UNET/checkpoints/best_model.pth` | 96 | 31,387,049 |
| **Residual U-Net** | `03_Residual_UNet/experiments/EXP-ARCH-03_RESIDUAL_UNET/checkpoints/best_model.pth` | 89 | 32,435,073 |
| **Attention Residual U-Net** | `04_Attention_Residual_UNet/experiments/EXP-ARCH-04_ATTENTION_RESIDUAL_UNET/checkpoints/best_model.pth` | 93 | 32,785,641 |
| **U-Net++** | `05_UNetPP/experiments/EXP-ARCH-05_UNETPP/checkpoints/best_model.pth` | 88 | 36,164,097 |

---

## 5. Consolidated Test Performance (N=63)
Macro-averaged per-image metrics across all 63 test images:

| Architecture | Dice (Mean ± SD) | IoU (Mean ± SD) | Precision (Mean ± SD) | Recall (Mean ± SD) | Specificity (Mean ± SD) | HD95 (px) (Mean ± SD) | HD95 Valid N | NSD @ 1px | NSD @ 2px | NSD @ 3px |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in rows:
        md_content += f"| **{r['Architecture']}** | **{r['Dice Mean']:.4f} ± {r['Dice SD']:.4f}** | {r['IoU Mean']:.4f} ± {r['IoU SD']:.4f} | {r['Precision Mean']:.4f} ± {r['Precision SD']:.4f} | {r['Recall Mean']:.4f} ± {r['Recall SD']:.4f} | {r['Specificity Mean']:.4f} ± {r['Specificity SD']:.4f} | {r['HD95 Mean']:.2f} ± {r['HD95 SD']:.2f} | {r['HD95 Valid N']}/63 | {r['NSD@1 Mean']:.4f} | {r['NSD@2 Mean']:.4f} | {r['NSD@3 Mean']:.4f} |\n"

    md_content += """
---

## 6. Per-Class Test Performance Breakdown

### A. Benign Lesions (N=42)
| Architecture | Dice (Mean ± SD) | IoU (Mean ± SD) | Precision (Mean ± SD) | Recall (Mean ± SD) | Specificity (Mean ± SD) | HD95 (px, Valid N) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for res in eval_results:
        name = res["spec"]["display_name"]
        b = res["agg"]["by_class"]["benign"]
        md_content += f"| **{name}** | {b['dice_mean']:.4f} ± {b['dice_std']:.4f} | {b['iou_mean']:.4f} ± {b['iou_std']:.4f} | {b['precision_mean']:.4f} ± {b['precision_std']:.4f} | {b['recall_mean']:.4f} ± {b['recall_std']:.4f} | {b['specificity_mean']:.4f} ± {b['specificity_std']:.4f} | {b['hd95_mean']:.2f} px ({b['hd95_valid_count']}/42) |\n"

    md_content += """
### B. Malignant Lesions (N=21)
| Architecture | Dice (Mean ± SD) | IoU (Mean ± SD) | Precision (Mean ± SD) | Recall (Mean ± SD) | Specificity (Mean ± SD) | HD95 (px, Valid N) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for res in eval_results:
        name = res["spec"]["display_name"]
        m = res["agg"]["by_class"]["malignant"]
        md_content += f"| **{name}** | {m['dice_mean']:.4f} ± {m['dice_std']:.4f} | {m['iou_mean']:.4f} ± {m['iou_std']:.4f} | {m['precision_mean']:.4f} ± {m['precision_std']:.4f} | {m['recall_mean']:.4f} ± {m['recall_std']:.4f} | {m['specificity_mean']:.4f} ± {m['specificity_std']:.4f} | {m['hd95_mean']:.2f} px ({m['hd95_valid_count']}/21) |\n"

    md_content += """
---

## 7. Edge-Case Handling & Invalid Metric Observations
- **HD95 Edge Cases:** Under Euclidean distance transform definitions, if either the predicted foreground or the ground truth lesion contains zero pixels, the Hausdorff distance surface contour is mathematically undefined. Consistent with project standards, such cases are assigned `NaN` and excluded from `hd95_mean` without imputing zeros. The exact number of valid cases is reported in the table above.
- **Surface Distances (NSD):** Empty set pairs yield 1.0 (perfect empty match); single empty sets yield 0.0, ensuring 63/63 valid samples for all models.

---

## 8. Note on Loss Comparison Separation
> **Important Scope Note:**
> This evaluation strictly compares the five architectures trained under the baseline **Dice Loss** to isolate architectural impact.
> The separate loss study (evaluating BCE, Focal, Tversky, and BCE + Dice on Attention U-Net) is preserved in `5_Models/Best_Architecture/BCE_Dice/` and `5_Models/Best_Architecture/FINAL_TEST_EVALUATION/`.
"""

    report_md_path = os.path.join(output_base_dir, "ARCHITECTURE_TEST_REPORT.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SAVED] Comprehensive Markdown Report: {report_md_path}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Official Architecture Test Evaluation (5 Models, Locked Test Set)")
    parser.add_argument("--dry-run", action="store_true", help="Perform pre-flight verification only without full test inference.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output root directory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Always perform dry-run checks before full execution
    audit_passed = run_dry_run_checks(device)
    if not audit_passed:
        print("[ERROR] Readiness audit failed! Aborting.")
        sys.exit(1)

    if args.dry_run:
        print("[DRY-RUN COMPLETE] All pre-flight checks passed. Full evaluation was NOT executed.")
        sys.exit(0)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test dataset once for deterministic consistency
    test_transforms = get_test_transforms(input_size=(256, 256))
    dataset = BUSIDataset(manifest_path=test_manifest_path, transform=test_transforms, img_channels=1, return_metadata=True)

    eval_results = []
    print("\n" + "=" * 80)
    print("LAUNCHING SEQUENTIAL TEST EVALUATION FOR ALL 5 ARCHITECTURES")
    print("=" * 80)

    for spec in ARCHITECTURE_SPECS:
        try:
            res = evaluate_single_architecture(
                spec=spec,
                dataset=dataset,
                output_base_dir=args.output_dir,
                device=device
            )
            eval_results.append(res)
        except Exception as e:
            print(f"\n[FATAL ERROR] Architecture evaluation failed on {spec['display_name']}: {e}")
            print("Aborting remaining models to preserve research integrity.")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Generate consolidated outputs
    generate_comparison_report(eval_results, args.output_dir)
    print("ALL 5 ARCHITECTURE TEST EVALUATIONS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
