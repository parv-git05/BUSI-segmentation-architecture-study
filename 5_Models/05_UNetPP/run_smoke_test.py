"""
Integration & Pipeline Smoke Test Runner for U-Net++.
Executes a 2-epoch smoke test on real BUSI train_70.csv and val_20.csv splits to verify pipeline integrity.
DOES NOT EVALUATE THE LOCKED TEST SET.
DOES NOT RUN THE FULL 100-EPOCH OFFICIAL EXPERIMENT.
"""

import os
import sys
import torch

# Ensure 00_Common and 05_UNetPP are in sys.path
models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
common_root = os.path.join(models_root, "00_Common")
unetpp_root = os.path.join(models_root, "05_UNetPP")

for p in [common_root, unetpp_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.experiment_config import ExperimentConfig
from datasets.busi_dataset import create_dataloader
from datasets.transforms import get_train_transforms, get_val_transforms
from training.losses import DiceLoss
from training.trainer import SegmentationTrainer
from utils.experiment_tracker import ExperimentTracker
from utils.seed import seed_everything
from model import UNetPlusPlus


def run_unetpp_smoke_test():
    """Runs a 2-epoch pipeline smoke test for U-Net++."""
    master_split_dir = os.path.join(models_root, "MASTER_SPLIT")
    smoke_output_dir = os.path.join(unetpp_root, "smoke_test")

    # 1. Smoke Test Configuration (2 Epochs, Batch Size 8)
    config = ExperimentConfig(
        experiment_name="UNetPP_Smoke_Test",
        architecture_name="UNetPlusPlus",
        seed=42,
        train_manifest_path=os.path.join(master_split_dir, "train_70.csv"),
        val_manifest_path=os.path.join(master_split_dir, "val_20.csv"),
        test_manifest_path=os.path.join(master_split_dir, "test_10.csv"),
        input_size=(256, 256),
        img_channels=1,
        batch_size=8,
        epochs=2,
        learning_rate=1e-4,
        optimizer_name="Adam",
        scheduler_name="ReduceLROnPlateau",
        loss_fn_name="DiceLoss",
        dice_smooth=1e-6,
        target_metric="val_dice",
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=smoke_output_dir
    )

    # 2. Seed Everything
    seed_everything(config.seed)

    # 3. Create DataLoaders (Train and Val ONLY)
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

    # 4. Instantiate Model, Optimizer, Criterion
    model = UNetPlusPlus(in_channels=1, out_channels=1, base_channels=64, deep_supervision=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = DiceLoss(smooth=config.dice_smooth, from_logits=True)

    # 5. Tracker
    tracker = ExperimentTracker(
        output_root=smoke_output_dir,
        experiment_name="UNetPP_Pipeline_Smoke"
    )

    # 6. Execute Trainer
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

    summary = trainer.fit()
    tracker.close()
    print("\n[UNET++ SMOKE TEST COMPLETE] Summary:", summary)
    return summary


def prepare_official_experiment_structure():
    """
    Prepares the official U-Net++ experiment directory structure and config.json.
    DOES NOT EXECUTE THE OFFICIAL 100-EPOCH TRAIN RUN.
    """
    official_exp_dir = os.path.join(unetpp_root, "experiments", "EXP-ARCH-05_UNETPP")
    subdirs = [
        "checkpoints",
        "training_curves",
        "validation_metrics",
        "per_image_metrics",
        "qualitative_predictions",
        "reports",
        "logs"
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(official_exp_dir, sub), exist_ok=True)

    from config import get_unetpp_config
    official_cfg = get_unetpp_config(
        experiment_name="EXP-ARCH-05_UNETPP",
        output_dir=os.path.dirname(official_exp_dir)
    )
    official_cfg.save_json(os.path.join(official_exp_dir, "config.json"))
    print(f"\n[OFFICIAL EXPERIMENT STRUCTURE PREPARED] At: {official_exp_dir}")


if __name__ == "__main__":
    run_unetpp_smoke_test()
    prepare_official_experiment_structure()
