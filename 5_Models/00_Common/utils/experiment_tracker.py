"""
Experiment Tracking & Workspace Structure Generator.
Standardizes output subdirectories for reproducible model comparisons.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from .logger import setup_logger, close_logger


class ExperimentTracker:
    """
    Manages filesystem directory creation and output artifact preservation for a single experiment run.
    """

    def __init__(self, output_root: str, experiment_name: str):
        """
        Args:
            output_root (str): Parent folder for all experiments (e.g. F:\\...\\5_Models\\experiments).
            experiment_name (str): Identifier for this experiment run.
        """
        self.output_root = os.path.abspath(output_root)
        self.experiment_name = experiment_name
        self.exp_dir = os.path.join(self.output_root, self.experiment_name)

        # Standardized Subdirectory Schema
        self.checkpoints_dir = os.path.join(self.exp_dir, "checkpoints")
        self.training_curves_dir = os.path.join(self.exp_dir, "training_curves")
        self.validation_metrics_dir = os.path.join(self.exp_dir, "validation_metrics")
        self.per_image_metrics_dir = os.path.join(self.exp_dir, "per_image_metrics")
        self.qualitative_dir = os.path.join(self.exp_dir, "qualitative_predictions")

        self._create_directories()

        # Logger Setup
        self.log_file = os.path.join(self.exp_dir, "train.log")
        self.logger = setup_logger(name=experiment_name, log_file=self.log_file)

    def _create_directories(self) -> None:
        """Creates required directory structure."""
        dirs = [
            self.exp_dir,
            self.checkpoints_dir,
            self.training_curves_dir,
            self.validation_metrics_dir,
            self.per_image_metrics_dir,
            self.qualitative_dir
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def save_config(self, config_dict: Dict[str, Any]) -> str:
        """Saves configuration JSON in experiment root."""
        cfg_path = os.path.join(self.exp_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        return cfg_path

    def save_history_csv(self, history: Dict[str, list]) -> str:
        """Saves epoch training/validation history to CSV."""
        df = pd.DataFrame(history)
        path = os.path.join(self.training_curves_dir, "epoch_history.csv")
        df.to_csv(path, index=False)
        return path

    def save_per_sample_metrics(self, df_metrics: pd.DataFrame, prefix: str = "val") -> str:
        """Saves detailed per-sample metric evaluation DataFrame."""
        path = os.path.join(self.per_image_metrics_dir, f"{prefix}_per_image_metrics.csv")
        df_metrics.to_csv(path, index=False)
        return path

    def save_summary_report(self, summary_dict: Dict[str, Any]) -> str:
        """Saves final summary report JSON."""
        path = os.path.join(self.exp_dir, "summary_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=4)
        return path

    def close(self) -> None:
        """Closes file handlers associated with tracker logger."""
        if hasattr(self, "logger"):
            close_logger(self.logger)
