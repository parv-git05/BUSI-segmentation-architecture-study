"""
Centralized configuration mechanism for the BUSI segmentation research framework.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any, Optional


@dataclass
class ExperimentConfig:
    """
    Centralized configuration container for training and evaluating BUSI segmentation models.
    All five model architectures will receive instances of this configuration to ensure
    identical hyperparameter, dataset, loss, and metric settings.
    """
    # Experiment Identification
    experiment_name: str = "default_experiment"
    architecture_name: str = "unspecified_model"
    seed: int = 42
    
    # Dataset Manifest Paths
    train_manifest_path: str = r"F:\Research_BUSI_Segmentation\5_Models\MASTER_SPLIT\train_70.csv"
    val_manifest_path: str = r"F:\Research_BUSI_Segmentation\5_Models\MASTER_SPLIT\val_20.csv"
    test_manifest_path: str = r"F:\Research_BUSI_Segmentation\5_Models\MASTER_SPLIT\test_10.csv"
    dataset_root: Optional[str] = None  # Auto-resolved if None
    
    # Image & Preprocessing Settings
    input_size: Tuple[int, int] = (256, 256)
    img_channels: int = 1
    normalize_range: Tuple[float, float] = (0.0, 1.0)
    
    # Optimization Parameters
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_name: str = "Adam"
    scheduler_name: str = "ReduceLROnPlateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Loss & Metric Settings
    loss_fn_name: str = "DiceLoss"
    dice_smooth: float = 1e-6
    target_metric: str = "val_dice"
    
    # Checkpointing & Validation Execution
    validation_frequency: int = 1
    num_workers: int = 0
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Directory Structure
    output_dir: str = r"F:\Research_BUSI_Segmentation\5_Models\experiments"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Construct ExperimentConfig from dictionary."""
        if "input_size" in data and isinstance(data["input_size"], list):
            data["input_size"] = tuple(data["input_size"])
        if "normalize_range" in data and isinstance(data["normalize_range"], list):
            data["normalize_range"] = tuple(data["normalize_range"])
        return cls(**data)
    
    def save_json(self, filepath: str) -> None:
        """Serialize configuration to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
            
    @classmethod
    def load_json(cls, filepath: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
