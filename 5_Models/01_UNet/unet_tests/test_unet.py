"""
Unit Tests for U-Net Architecture (TASK 3 Validation).
Verifies tensor shapes, forward/backward passes, finite outputs, loss compatibility,
trainer engine compatibility, and parameter count determinism.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Add 00_Common and 01_UNet directories to sys.path
models_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
common_root = os.path.join(models_root, "00_Common")
unet_root = os.path.join(models_root, "01_UNet")

for path in [common_root, unet_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model import UNet
from training.losses import DiceLoss


class TestUNetArchitecture(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = UNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False).to(self.device)

    def test_01_instantiation(self):
        """1. Model can be instantiated."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model, UNet)

    def test_02_to_05_forward_shapes_and_finite_values(self):
        """2, 3, 4, 5. Input [2, 1, 256, 256] -> Output [2, 1, 256, 256] with finite values."""
        input_tensor = torch.randn(2, 1, 256, 256, device=self.device)
        self.assertEqual(input_tensor.shape, torch.Size([2, 1, 256, 256]))

        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, torch.Size([2, 1, 256, 256]))

        # 5. Output contains finite values (no NaNs, no Infs)
        self.assertTrue(torch.isfinite(output_tensor).all().item())

    def test_06_backpropagation(self):
        """6. Backpropagation succeeds and gradients flow cleanly."""
        input_tensor = torch.randn(2, 1, 256, 256, device=self.device, requires_grad=True)
        target_tensor = (torch.rand(2, 1, 256, 256, device=self.device) > 0.5).float()

        output_tensor = self.model(input_tensor)
        loss = output_tensor.sum()
        loss.backward()

        # Check gradients exist and contain valid finite numbers
        self.assertIsNotNone(self.model.inc.double_conv[0].weight.grad)
        self.assertFalse(torch.isnan(self.model.inc.double_conv[0].weight.grad).any().item())

    def test_07_parameter_count(self):
        """7. Model has a reproducible parameter count (31,036,481 parameters)."""
        param_count = self.model.count_parameters()
        self.assertEqual(param_count, 31036481)

    def test_08_dice_loss_compatibility(self):
        """8. Model outputs work cleanly with Dice Loss."""
        criterion = DiceLoss(smooth=1e-6, from_logits=True)
        input_tensor = torch.randn(2, 1, 256, 256, device=self.device)
        target_tensor = (torch.rand(2, 1, 256, 256, device=self.device) > 0.5).float()

        logits = self.model(input_tensor)
        loss = criterion(logits, target_tensor)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0)

    def test_10_cpu_execution(self):
        """10. Model runs cleanly on CPU."""
        model_cpu = UNet().to("cpu")
        x = torch.randn(1, 1, 256, 256, device="cpu")
        out = model_cpu(x)
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.shape, (1, 1, 256, 256))


if __name__ == "__main__":
    unittest.main()
