"""
Sequential Loss Function Benchmark Orchestrator for Attention U-Net.
Executes the four loss experiments in strict sequential order:
1. BCE
2. BCE_Dice
3. Focal
4. Tversky

LOCKED TEST SET IS NEVER ACCESSED OR EVALUATED.
FAILURE POLICY: Halts pipeline immediately on any error or artifact verification failure.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

best_arch_root = os.path.dirname(os.path.abspath(__file__))
models_root = os.path.dirname(best_arch_root)
project_root = os.path.dirname(models_root)

# Sequential experiments list
EXPERIMENTS = [
    {
        "name": "BCE",
        "script": os.path.join(best_arch_root, "BCE", "code", "train_bce.py"),
        "exp_dir": os.path.join(best_arch_root, "BCE")
    },
    {
        "name": "BCE_Dice",
        "script": os.path.join(best_arch_root, "BCE_Dice", "code", "train_bce_dice.py"),
        "exp_dir": os.path.join(best_arch_root, "BCE_Dice")
    },
    {
        "name": "Focal",
        "script": os.path.join(best_arch_root, "Focal", "code", "train_focal.py"),
        "exp_dir": os.path.join(best_arch_root, "Focal")
    },
    {
        "name": "Tversky",
        "script": os.path.join(best_arch_root, "Tversky", "code", "train_tversky.py"),
        "exp_dir": os.path.join(best_arch_root, "Tversky")
    }
]

# Required Artifacts for verification
REQUIRED_ARTIFACTS = [
    os.path.join("checkpoints", "best_model.pth"),
    os.path.join("checkpoints", "latest.pth"),
    os.path.join("metrics", "val_aggregate_metrics.json"),
    os.path.join("metrics", "val_per_image_metrics.csv"),
    os.path.join("curves", "epoch_history.csv"),
    os.path.join("curves", "learning_curves.png"),
    os.path.join("visualizations", "val_qualitative_predictions.png"),
    os.path.join("logs", "train.log"),
    "REPORT.md",
    "summary_report.json",
    "config.json"
]


def log(msg: str):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {msg}", flush=True)


def verify_experiment_artifacts(exp_dir: str) -> tuple[bool, list]:
    missing = []
    for rel_path in REQUIRED_ARTIFACTS:
        full_path = os.path.join(exp_dir, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)
    return len(missing) == 0, missing


def generate_comparison_report():
    log("=" * 80)
    log("GENERATING COMPREHENSIVE LOSS COMPARISON REPORT...")
    log("=" * 80)

    loss_experiments = [
        {"id": "Dice_baseline", "name": "Dice (Baseline)", "dir": os.path.join(best_arch_root, "Dice_baseline")},
        {"id": "BCE", "name": "BCE", "dir": os.path.join(best_arch_root, "BCE")},
        {"id": "BCE_Dice", "name": "BCE + Dice", "dir": os.path.join(best_arch_root, "BCE_Dice")},
        {"id": "Focal", "name": "Focal", "dir": os.path.join(best_arch_root, "Focal")},
        {"id": "Tversky", "name": "Tversky", "dir": os.path.join(best_arch_root, "Tversky")}
    ]

    results = []
    for exp in loss_experiments:
        exp_dir = exp["dir"]
        summary_path = os.path.join(exp_dir, "summary_report.json")
        metrics_path = os.path.join(exp_dir, "metrics", "val_aggregate_metrics.json")
        
        if not os.path.exists(metrics_path):
            log(f"WARNING: Metrics file missing for {exp['id']}: {metrics_path}")
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            agg_m = json.load(f)

        best_epoch = "N/A"
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                sdata = json.load(f)
                best_epoch = sdata.get("best_epoch", "N/A")

        results.append({
            "id": exp["id"],
            "name": exp["name"],
            "best_epoch": best_epoch,
            "dice_mean": agg_m.get("dice_mean", 0.0),
            "dice_std": agg_m.get("dice_std", 0.0),
            "iou_mean": agg_m.get("iou_mean", 0.0),
            "iou_std": agg_m.get("iou_std", 0.0),
            "precision_mean": agg_m.get("precision_mean", 0.0),
            "precision_std": agg_m.get("precision_std", 0.0),
            "recall_mean": agg_m.get("recall_mean", 0.0),
            "recall_std": agg_m.get("recall_std", 0.0),
            "specificity_mean": agg_m.get("specificity_mean", 0.0),
            "specificity_std": agg_m.get("specificity_std", 0.0),
            "hd95_mean": agg_m.get("hd95_mean", 0.0),
            "hd95_std": agg_m.get("hd95_std", 0.0),
            "hd95_valid_count": agg_m.get("hd95_valid_count", 0),
            "nsd_1px_mean": agg_m.get("nsd_1px_mean", 0.0),
            "nsd_1px_std": agg_m.get("nsd_1px_std", 0.0),
            "nsd_2px_mean": agg_m.get("nsd_2px_mean", 0.0),
            "nsd_2px_std": agg_m.get("nsd_2px_std", 0.0),
            "nsd_3px_mean": agg_m.get("nsd_3px_mean", 0.0),
            "nsd_3px_std": agg_m.get("nsd_3px_std", 0.0)
        })

    # Find highest validation dice
    best_loss = max(results, key=lambda x: x["dice_mean"]) if results else None

    # Save summary JSON
    summary_json_path = os.path.join(best_arch_root, "loss_comparison_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "study": "Attention U-Net Loss Function Benchmark",
            "date": datetime.now().isoformat(),
            "best_loss_function": best_loss["name"] if best_loss else "None",
            "results": results
        }, f, indent=4)

    # Build Markdown Comparison Report
    rows = []
    for r in results:
        is_best_mark = " **(Best Val Dice)**" if best_loss and r["id"] == best_loss["id"] else ""
        rows.append(
            f"| `{r['id']}`{is_best_mark} | {r['best_epoch']} | "
            f"**{r['dice_mean']:.4f} ± {r['dice_std']:.4f}** | "
            f"{r['iou_mean']:.4f} ± {r['iou_std']:.4f} | "
            f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f} | "
            f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f} | "
            f"{r['specificity_mean']:.4f} ± {r['specificity_std']:.4f} | "
            f"{r['hd95_mean']:.2f} ± {r['hd95_std']:.2f} ({r['hd95_valid_count']}/125) | "
            f"{r['nsd_1px_mean']:.4f} ± {r['nsd_1px_std']:.4f} | "
            f"{r['nsd_2px_mean']:.4f} ± {r['nsd_2px_std']:.4f} | "
            f"{r['nsd_3px_mean']:.4f} ± {r['nsd_3px_std']:.4f} |"
        )

    table_md = "\n".join(rows)

    best_loss_id = best_loss['id'] if best_loss else 'N/A'
    best_loss_dice_mean = best_loss['dice_mean'] if best_loss else 0.0
    best_loss_dice_std = best_loss['dice_std'] if best_loss else 0.0
    best_loss_epoch = best_loss['best_epoch'] if best_loss else 'N/A'

    report_content = f"""# Loss Function Benchmark Report: Attention U-Net

## 1. Study Overview
- **Objective:** Controlled evaluation of loss functions on the selected optimal architecture (**Attention U-Net**).
- **Architecture Specification:** Attention U-Net (Oktay et al., 2018), 31,387,049 parameters.
- **Dataset:** BUSI Phase-1 Single-Mask Cohort (440 Train / 125 Validation / 63 Test).
- **Test Set Status:** **LOCKED & UNTOUCHED (0 test samples loaded or evaluated). All comparisons use validation set only.**
- **Training Protocol:** 100 Epochs, Adam optimizer (lr=1e-4, weight_decay=1e-4), ReduceLROnPlateau, Batch Size 16, Seed 42. Early stopping DISABLED.

## 2. Validation Metrics Comparison Table

| Loss Function | Best Epoch | Dice | IoU | Precision | Recall | Specificity | HD95 (px) [Valid N] | NSD @ 1px | NSD @ 2px | NSD @ 3px |
|---|---|---|---|---|---|---|---|---|---|---|
{table_md}

## 3. Loss Formulations & Parameters

1. **Dice (Baseline):**
   - Formulation: Binary Soft Dice Loss
   - Parameters: `smooth=1e-6`

2. **Binary Cross-Entropy (BCE):**
   - Formulation: Binary Cross-Entropy with Logits
   - Parameters: `from_logits=True`, `reduction=mean`

3. **Combined BCE + Dice (BCE_Dice):**
   - Formulation: BCE + Dice (bce_weight * BCE + dice_weight * Dice)
   - Parameters: `bce_weight=1.0`, `dice_weight=1.0`, `smooth=1e-6`

4. **Binary Focal Loss (Focal):**
   - Formulation: Binary Focal Loss (gamma=2.0, alpha=0.25)
   - Parameters: `alpha=0.25`, `gamma=2.0`, `from_logits=True`

5. **Tversky Loss (Tversky):**
   - Formulation: Tversky Index Loss (alpha=0.7, beta=0.3)
   - Parameters: `alpha=0.7`, `beta=0.3`, `smooth=1e-6`

## 4. Key Scientific Observations
- **Top Performer by Validation Dice:** `{best_loss_id}` achieved validation Dice of **{best_loss_dice_mean:.4f} ± {best_loss_dice_std:.4f}** (Best Epoch {best_loss_epoch}).
- Complete per-image predictions, learning curves, and model checkpoints are archived in their respective experiment folders.
"""

    report_path = os.path.join(best_arch_root, "LOSS_COMPARISON_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    log(f"Loss comparison report successfully written to: {report_path}")


def main():
    log("=" * 80)
    log("ATTENTION U-NET LOSS FUNCTION BENCHMARK — SEQUENTIAL RUNNER")
    log("Strict Execution Order: BCE -> BCE_Dice -> Focal -> Tversky")
    log("Test Set Status: LOCKED & UNTOUCHED (0 test samples accessed)")
    log("=" * 80)

    total_start = time.time()

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        name = exp["name"]
        script = exp["script"]
        exp_dir = exp["exp_dir"]

        log("")
        log("=" * 80)
        log(f"[{idx}/{len(EXPERIMENTS)}] LAUNCHING EXPERIMENT: {name}")
        log(f"Script: {script}")
        log(f"Output Directory: {exp_dir}")
        log("=" * 80)

        if not os.path.exists(script):
            log(f"CRITICAL ERROR: Experiment script not found at {script}")
            log("HALTING PIPELINE IMMEDIATELY.")
            sys.exit(1)

        exp_start = time.time()

        # Launch process with real-time output streaming
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root
        )

        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()

        process.stdout.close()
        return_code = process.wait()
        exp_duration = time.time() - exp_start

        # FAILURE POLICY ENFORCEMENT
        if return_code != 0:
            log("")
            log("!" * 80)
            log(f"CRITICAL FAILURE IN EXPERIMENT: {name}")
            log(f"Exit Code: {return_code}")
            log(f"Execution Time Before Failure: {exp_duration / 60.0:.2f} mins")
            log(f"Logs: {os.path.join(exp_dir, 'logs', 'train.log')}")
            log(f"Checkpoints: {os.path.join(exp_dir, 'checkpoints')}")
            log("FAILURE POLICY: HALTING SEQUENTIAL RUNNER IMMEDIATELY.")
            log("Subsequent experiments will NOT run until this issue is manually investigated.")
            log("!" * 80)
            sys.exit(return_code)

        # Artifacts Verification
        artifacts_ok, missing = verify_experiment_artifacts(exp_dir)
        if not artifacts_ok:
            log("")
            log("!" * 80)
            log(f"ARTIFACT VERIFICATION FAILED FOR EXPERIMENT: {name}")
            log(f"Missing Artifacts: {missing}")
            log(f"Experiment Directory: {exp_dir}")
            log("FAILURE POLICY: HALTING SEQUENTIAL RUNNER IMMEDIATELY.")
            log("!" * 80)
            sys.exit(1)

        log("")
        log("=" * 80)
        log(f"SUCCESS: {name} completed in {exp_duration / 60.0:.2f} mins with all artifacts verified.")
        log("=" * 80)

    total_duration = time.time() - total_start
    log("")
    log("=" * 80)
    log(f"ALL 4 LOSS EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_duration / 60.0:.2f} MINS.")
    log("=" * 80)

    # Generate the Comparison Report across all 5 loss configurations
    generate_comparison_report()


if __name__ == "__main__":
    main()
