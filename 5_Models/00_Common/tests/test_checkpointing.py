"""
Unit tests for atomic checkpoint saving and loading functionality.
"""

import os
import sys
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn

# Add 00_Common directory to sys.path
common_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if common_root not in sys.path:
    sys.path.insert(0, common_root)

from utils.checkpoint import save_checkpoint, load_checkpoint


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TestCheckpointing(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ckpt_path = os.path.join(self.temp_dir, "test_ckpt.pth")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_checkpoint(self):
        model_orig = DummyModel()
        optimizer_orig = torch.optim.Adam(model_orig.parameters(), lr=1e-3)

        val_metrics = {"dice_mean": 0.85, "iou_mean": 0.75}

        # Save Checkpoint
        save_checkpoint(
            filepath=self.ckpt_path,
            model=model_orig,
            optimizer=optimizer_orig,
            epoch=5,
            train_loss=0.25,
            val_loss=0.30,
            val_metrics=val_metrics,
            seed=42,
            experiment_id="test_exp"
        )

        self.assertTrue(os.path.exists(self.ckpt_path))

        # Load Checkpoint into new model instance
        model_loaded = DummyModel()
        optimizer_loaded = torch.optim.Adam(model_loaded.parameters(), lr=1e-3)

        meta = load_checkpoint(
            filepath=self.ckpt_path,
            model=model_loaded,
            optimizer=optimizer_loaded
        )

        self.assertEqual(meta["epoch"], 5)
        self.assertEqual(meta["train_loss"], 0.25)
        self.assertEqual(meta["val_loss"], 0.30)
        self.assertEqual(meta["val_metrics"]["dice_mean"], 0.85)

        # Verify model weights match exactly
        for p1, p2 in zip(model_orig.parameters(), model_loaded.parameters()):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
