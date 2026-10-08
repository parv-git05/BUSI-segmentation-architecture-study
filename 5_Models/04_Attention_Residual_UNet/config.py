"""
Configuration builder for Attention Residual U-Net experiments.
Inherits all common research protocol settings from 00_Common without modification.
"""

import os
import sys
from typing import Optional

# Ensure 00_Common is in sys.path
common_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "00_Common"))
if common_root not in sys.path:
    sys.path.insert(0, common_root)

from configs.experiment_config import ExperimentConfig


def get_attention_residual_unet_config(
    experiment_name: str = "EXP-ARCH-04_ATTENTION_RESIDUAL_UNET",
    output_dir: Optional[str] = None
) -> ExperimentConfig:
    """
    Constructs an ExperimentConfig instance configured specifically for Attention Residual U-Net architecture runs.
    Enforces all locked baseline protocol parameters.
    """
    models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master_split_dir = os.path.join(models_root, "MASTER_SPLIT")
    default_output_dir = os.path.join(models_root, "04_Attention_Residual_UNet", "experiments") if output_dir is None else output_dir

    cfg = ExperimentConfig(
        experiment_name=experiment_name,
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
        validation_frequency=1,
        output_dir=default_output_dir
    )
    return cfg
