"""
Unit tests for Preprocessing & Transformation Pipelines.
"""

import os
import sys
import unittest
import torch
from PIL import Image
import numpy as np

# Add 00_Common directory to sys.path
common_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if common_root not in sys.path:
    sys.path.insert(0, common_root)

from datasets.transforms import (
    BUSITransform,
    get_train_transforms,
    get_val_transforms,
    get_test_transforms
)


class TestBUSITransforms(unittest.TestCase):

    def setUp(self):
        # Create synthetic PIL image and binary mask
        self.img_np = np.random.randint(0, 256, (300, 400), dtype=np.uint8)
        self.mask_np = np.zeros((300, 400), dtype=np.uint8)
        self.mask_np[50:150, 50:150] = 255  # Square lesion

        self.img_pil = Image.fromarray(self.img_np, mode="L")
        self.mask_pil = Image.fromarray(self.mask_np, mode="L")

    def test_val_test_deterministic_preprocessing(self):
        """Verifies val and test transforms produce identical deterministic outputs."""
        transform_val = get_val_transforms(input_size=(256, 256))
        transform_test = get_test_transforms(input_size=(256, 256))

        img_t1, mask_t1 = transform_val(self.img_pil, self.mask_pil)
        img_t2, mask_t2 = transform_test(self.img_pil, self.mask_pil)

        self.assertEqual(img_t1.shape, (1, 256, 256))
        self.assertEqual(mask_t1.shape, (1, 256, 256))

        # Check determinism (val and test must be exactly equal for same input)
        self.assertTrue(torch.allclose(img_t1, img_t2))
        self.assertTrue(torch.allclose(mask_t1, mask_t2))

    def test_mask_binary_discrete_preservation(self):
        """Verifies mask resizing using NEAREST interpolation retains discrete binary values {0.0, 1.0}."""
        transform = get_val_transforms(input_size=(128, 128))
        _, mask_t = transform(self.img_pil, self.mask_pil)

        unique_vals = torch.unique(mask_t).tolist()
        for v in unique_vals:
            self.assertIn(v, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
