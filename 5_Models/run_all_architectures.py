"""
Final Sequential Official-Training Orchestrator for BUSI Architecture Study.
Executes official 100-epoch experiments for the 4 remaining architectures sequentially:
- EXP-ARCH-02_ATTENTION_UNET (Attention U-Net)
- EXP-ARCH-03_RESIDUAL_UNET (Residual U-Net)
- EXP-ARCH-04_ATTENTION_RESIDUAL_UNET (Attention Residual U-Net)
- EXP-ARCH-05_UNETPP (U-Net++)

NOTE: EXP-ARCH-01_UNET (Standard U-Net) is already officially trained and is skipped.
LOCKED TEST SET (test_10.csv) IS NEVER ACCESSED OR EVALUATED.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

# Setup Root Paths
models_root = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(models_root, "orchestration_logs")
os.makedirs(log_dir, exist_ok=True)
orchestration_log_file = os.path.join(log_dir, "orchestrator.log")

# Architecture Experiment List
ARCHITECTURES = [
    {
        "name": "Attention U-Net",
        "exp_id": "EXP-ARCH-02_ATTENTION_UNET",
        "script": os.path.join(models_root, "02_Attention_UNet", "run_official_experiment.py"),
        "exp_dir": os.path.join(models_root, "02_Attention_UNet", "experiments", "EXP-ARCH-02_ATTENTION_UNET")
    },
    {
        "name": "Residual U-Net",
        "exp_id": "EXP-ARCH-03_RESIDUAL_UNET",
        "script": os.path.join(models_root, "03_Residual_UNet", "run_official_experiment.py"),
        "exp_dir": os.path.join(models_root, "03_Residual_UNet", "experiments", "EXP-ARCH-03_RESIDUAL_UNET")
    },
    {
        "name": "Attention Residual U-Net",
        "exp_id": "EXP-ARCH-04_ATTENTION_RESIDUAL_UNET",
        "script": os.path.join(models_root, "04_Attention_Residual_UNet", "run_official_experiment.py"),
        "exp_dir": os.path.join(models_root, "04_Attention_Residual_UNet", "experiments", "EXP-ARCH-04_ATTENTION_RESIDUAL_UNET")
    },
    {
        "name": "U-Net++",
        "exp_id": "EXP-ARCH-05_UNETPP",
        "script": os.path.join(models_root, "05_UNetPP", "run_official_experiment.py"),
        "exp_dir": os.path.join(models_root, "05_UNetPP", "experiments", "EXP-ARCH-05_UNETPP")
    }
]

# Required Artifacts Checklist
REQUIRED_ARTIFACTS = [
    os.path.join("checkpoints", "best_model.pth"),
    os.path.join("checkpoints", "latest.pth"),
    os.path.join("metrics", "val_aggregate_metrics.json"),
    os.path.join("metrics", "val_per_image_metrics.csv"),
    os.path.join("reports", "experiment_report.json"),
    os.path.join("reports", "EXPERIMENT_REPORT.md"),
    os.path.join("training_curves", "epoch_history.csv"),
    "summary_report.json",
    "config.json"
]


def log_message(msg: str):
    """Outputs timestamped message to both terminal and orchestration log file."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    formatted = f"{timestamp} {msg}"
    print(msg, flush=True)
    with open(orchestration_log_file, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def check_experiment_completion(exp_dir: str) -> str:
    """
    Inspects an experiment directory.
    Returns:
    - 'COMPLETE': All required artifacts exist and 100 epochs trained.
    - 'NOT_STARTED': Directory does not exist, or only contains pre-prepared structure/config.json.
    - 'INCOMPLETE': Training was started (checkpoints/history exist) but missing required final artifacts.
    """
    if not os.path.exists(exp_dir):
        return "NOT_STARTED"

    missing = []
    for rel_path in REQUIRED_ARTIFACTS:
        full_path = os.path.join(exp_dir, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)

    if not missing:
        report_path = os.path.join(exp_dir, "reports", "experiment_report.json")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("total_epochs") == 100 and data.get("locked_test_set_evaluated") is False:
                return "COMPLETE"
        except Exception:
            pass

    # Check if actual training outputs exist (checkpoint or training history)
    best_ckpt = os.path.join(exp_dir, "checkpoints", "best_model.pth")
    history_csv = os.path.join(exp_dir, "training_curves", "epoch_history.csv")
    
    if os.path.exists(best_ckpt) or os.path.exists(history_csv):
        return "INCOMPLETE"
    
    return "NOT_STARTED"


def verify_artifacts(exp_dir: str) -> tuple[bool, list]:
    """Verifies that all required artifacts exist in the experiment directory."""
    missing = []
    for rel_path in REQUIRED_ARTIFACTS:
        full_path = os.path.join(exp_dir, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)
    return (len(missing) == 0, missing)


def main():
    log_message("=" * 70)
    log_message("BUSI LESION SEGMENTATION STUDY — OFFICIAL TRAINING ORCHESTRATOR")
    log_message("Executing 4 remaining experiments sequentially.")
    log_message("Test Set Status: LOCKED & UNTOUCHED")
    log_message("=" * 70)

    orchestration_summary = []

    for arch in ARCHITECTURES:
        name = arch["name"]
        exp_id = arch["exp_id"]
        script = arch["script"]
        exp_dir = arch["exp_dir"]

        log_message("")
        log_message("=" * 70)
        log_message("STARTING OFFICIAL EXPERIMENT")
        log_message(f"Architecture: {name}")
        log_message(f"Experiment: {exp_id}")
        log_message("=" * 70)

        # 1. Check if runner script exists
        if not os.path.exists(script):
            log_message(f"ERROR: Runner script not found at {script}")
            log_message("ORCHESTRATION HALTED.")
            sys.exit(1)

        # 2. Check completion status
        status = check_experiment_completion(exp_dir)

        if status == "COMPLETE":
            log_message(f"SKIPPING — official experiment {exp_id} already complete.")
            log_message(f"============================================================")
            log_message(f"COMPLETED: {name} (SKIPPED - ALREADY DONE)")
            log_message(f"============================================================")
            orchestration_summary.append({
                "architecture": name,
                "experiment_id": exp_id,
                "status": "COMPLETED (SKIPPED)",
                "duration_seconds": 0
            })
            continue
        elif status == "INCOMPLETE":
            log_message(f"INCOMPLETE EXISTING EXPERIMENT — manual review required at: {exp_dir}")
            log_message("ORCHESTRATION HALTED TO PREVENT UNSAFE OVERWRITE.")
            sys.exit(1)

        # 3. Execute Official Training Script with Live Output Streaming
        start_time = time.time()
        log_message(f"Invoking runner: {script}")
        
        project_root = os.path.dirname(models_root)
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
        duration = time.time() - start_time

        # 4. Check Exit Status
        if return_code != 0:
            log_message("")
            log_message("!" * 70)
            log_message(f"FAILURE IN ARCHITECTURE: {name}")
            log_message(f"Exit Code: {return_code}")
            log_message(f"Experiment Directory: {exp_dir}")
            log_message("Manual investigation is required.")
            log_message("!" * 70)
            sys.exit(return_code)

        # 5. Verify Post-Execution Artifacts
        artifacts_ok, missing = verify_artifacts(exp_dir)
        if not artifacts_ok:
            log_message("")
            log_message("!" * 70)
            log_message(f"ARTIFACT VERIFICATION FAILED FOR: {name}")
            log_message(f"Missing required artifacts: {missing}")
            log_message(f"Experiment Directory: {exp_dir}")
            log_message("ORCHESTRATION HALTED.")
            log_message("!" * 70)
            sys.exit(1)

        log_message("")
        log_message("=" * 70)
        log_message(f"COMPLETED: {name} (Duration: {duration / 60.0:.2f} mins)")
        log_message("=" * 70)

        orchestration_summary.append({
            "architecture": name,
            "experiment_id": exp_id,
            "status": "COMPLETED",
            "duration_seconds": duration,
            "artifacts_verified": True
        })

    # Save Orchestration Summary JSON
    summary_json_path = os.path.join(log_dir, "orchestration_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(orchestration_summary, f, indent=4)

    log_message("")
    log_message("=" * 70)
    log_message("ALL OFFICIAL ARCHITECTURE EXPERIMENTS COMPLETED")
    log_message("=" * 70)
    for entry in orchestration_summary:
        log_message(f"{entry['architecture']}: COMPLETED")
    log_message("")
    log_message("Test set: LOCKED / NOT EVALUATED")
    log_message("=" * 70)


if __name__ == "__main__":
    main()
