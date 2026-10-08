"""
Smoke tests for SegmentationTrainer using a minimal 1-convolution-layer dummy model.
DOES NOT IMPLEMENT OR TRAIN ANY OF THE FIVE TARGET ARCHITECTURES.
"""

import os
import sys
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add 00_Common directory to sys.path
common_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if common_root not in sys.path:
    sys.path.insert(0, common_root)

from configs.experiment_config import ExperimentConfig
from training.losses import DiceLoss
from training.trainer import SegmentationTrainer
from utils.experiment_tracker import ExperimentTracker
from utils.seed import seed_everything
from utils.logger import close_logger


class TinyDummySegmentationModel(nn.Module):
    """
    A minimal 1-layer convolution model used ONLY to smoke-test the training infrastructure.
    This is NOT U-Net, Attention U-Net, ResUNet, AttResUNet, or U-Net++.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TestSegmentationTrainer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExperimentConfig(
            experiment_name="smoke_test_run",
            architecture_name="TinyDummyModel",
            seed=42,
            input_size=(64, 64),
            batch_size=2,
            epochs=2,
            learning_rate=1e-3,
            device="cpu",
            output_dir=self.temp_dir
        )

        # Build synthetic dummy dataset for smoke test (4 samples)
        images = torch.rand(4, 1, 64, 64)
        masks = (torch.rand(4, 1, 64, 64) > 0.5).float()
        
        class SyntheticDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(images)
            def __getitem__(self, idx):
                return {
                    "image": images[idx],
                    "mask": masks[idx],
                    "sample_id": f"synthetic_{idx}",
                    "image_path": f"synthetic_{idx}.png",
                    "mask_path": f"synthetic_{idx}_mask.png",
                    "class_name": "benign",
                    "class_id": 0
                }

        ds = SyntheticDataset()
        self.train_loader = DataLoader(ds, batch_size=2, shuffle=False)
        self.val_loader = DataLoader(ds, batch_size=2, shuffle=False)
        self.tracker = None

    def tearDown(self):
        if self.tracker is not None:
            self.tracker.close()
        # Clean up any leftover loggers
        close_logger(None)
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass  # Fallback for Windows temp dir permissions

    def test_reproducibility_seed(self):
        """Verifies seed_everything produces deterministic tensor initialization."""
        seed_everything(42)
        t1 = torch.randn(5, 5)

        seed_everything(42)
        t2 = torch.randn(5, 5)

        self.assertTrue(torch.equal(t1, t2))

    def test_trainer_smoke_execution(self):
        """Verifies complete training loop execution, metric logging, and checkpoint creation."""
        model = TinyDummySegmentationModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = DiceLoss(smooth=1e-6, from_logits=True)

        self.tracker = ExperimentTracker(
            output_root=self.config.output_dir,
            experiment_name=self.config.experiment_name
        )

        trainer = SegmentationTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=self.config,
            tracker=self.tracker,
            device="cpu"
        )

        summary = trainer.fit()

        # Check summary output
        self.assertEqual(summary["total_epochs_trained"], 2)
        self.assertIn("best_target_metric", summary)

        # Check generated checkpoint files on disk
        ckpt_dir = self.tracker.checkpoints_dir
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "latest.pth")))
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "best_model.pth")))

        # Check training curves plot file
        self.assertTrue(os.path.exists(os.path.join(self.tracker.training_curves_dir, "learning_curves.png")))

        # Check qualitative predictions plot file
        self.assertTrue(os.path.exists(os.path.join(self.tracker.qualitative_dir, "epoch_002_qualitative.png")))


if __name__ == "__main__":
    unittest.main()
