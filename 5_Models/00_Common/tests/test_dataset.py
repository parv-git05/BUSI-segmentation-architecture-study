"""
Unit tests for BUSIDataset and DataLoader construction.
"""

import os
import sys
import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add 00_Common directory to sys.path
common_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if common_root not in sys.path:
    sys.path.insert(0, common_root)

from datasets.busi_dataset import BUSIDataset, create_dataloader
from datasets.transforms import get_val_transforms


class TestBUSIDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Resolve master split paths dynamically
        workspace_root = os.path.abspath(os.path.join(common_root, "..", ".."))
        master_split_dir = os.path.join(workspace_root, "5_Models", "MASTER_SPLIT")
        
        if not os.path.exists(master_split_dir):
            master_split_dir = r"F:\Research_BUSI_Segmentation\5_Models\MASTER_SPLIT"

        cls.train_csv = os.path.join(master_split_dir, "train_70.csv")
        cls.val_csv = os.path.join(master_split_dir, "val_20.csv")
        cls.test_csv = os.path.join(master_split_dir, "test_10.csv")
        cls.workspace_root = workspace_root

    def test_manifest_existence_and_counts(self):
        """Verifies master split manifests exist and sample counts match expectation."""
        self.assertTrue(os.path.exists(self.train_csv), f"Missing: {self.train_csv}")
        self.assertTrue(os.path.exists(self.val_csv), f"Missing: {self.val_csv}")
        self.assertTrue(os.path.exists(self.test_csv), f"Missing: {self.test_csv}")

        df_train = pd.read_csv(self.train_csv)
        df_val = pd.read_csv(self.val_csv)
        df_test = pd.read_csv(self.test_csv)

        self.assertEqual(len(df_train), 440)
        self.assertEqual(len(df_val), 125)
        self.assertEqual(len(df_test), 63)

    def test_no_split_overlap(self):
        """Verifies zero overlap between train, val, and test splits."""
        df_train = pd.read_csv(self.train_csv)
        df_val = pd.read_csv(self.val_csv)
        df_test = pd.read_csv(self.test_csv)

        train_imgs = set(df_train["image_path"].apply(os.path.basename))
        val_imgs = set(df_val["image_path"].apply(os.path.basename))
        test_imgs = set(df_test["image_path"].apply(os.path.basename))

        self.assertEqual(len(train_imgs.intersection(val_imgs)), 0)
        self.assertEqual(len(train_imgs.intersection(test_imgs)), 0)
        self.assertEqual(len(val_imgs.intersection(test_imgs)), 0)

    def test_dataset_loading_and_tensors(self):
        """Verifies BUSIDataset loads samples, resolves paths, and returns valid tensors."""
        transform = get_val_transforms(input_size=(128, 128))
        dataset = BUSIDataset(
            manifest_path=self.train_csv,
            dataset_root=self.workspace_root,
            transform=transform,
            img_channels=1
        )

        self.assertEqual(len(dataset), 440)

        # Inspect first sample
        sample = dataset[0]
        image = sample["image"]
        mask = sample["mask"]

        # Tensor Shape Check
        self.assertEqual(image.shape, (1, 128, 128))
        self.assertEqual(mask.shape, (1, 128, 128))

        # Binary Mask Value Check (Values MUST be strictly {0.0, 1.0})
        unique_mask_vals = torch.unique(mask)
        for val in unique_mask_vals:
            self.assertIn(val.item(), [0.0, 1.0])

        # Image Normalization Check [0.0, 1.0]
        self.assertGreaterEqual(image.min().item(), 0.0)
        self.assertLessEqual(image.max().item(), 1.0)

    def test_dataloader_creation(self):
        """Verifies create_dataloader factory functions properly."""
        loader = create_dataloader(
            manifest_path=self.val_csv,
            batch_size=4,
            shuffle=False,
            transform=get_val_transforms(input_size=(128, 128)),
            dataset_root=self.workspace_root
        )

        self.assertIsInstance(loader, DataLoader)
        batch = next(iter(loader))
        self.assertEqual(batch["image"].shape, (4, 1, 128, 128))
        self.assertEqual(batch["mask"].shape, (4, 1, 128, 128))


if __name__ == "__main__":
    unittest.main()
