"""
Unit Tests for Residual U-Net Architecture.
Verifies tensor shapes, residual block functionality, forward/backward passes,
finite outputs, loss compatibility, and parameter count determinism.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Add 00_Common and 03_Residual_UNet directories to sys.path
models_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
common_root = os.path.join(models_root, "00_Common")
res_unet_root = os.path.join(models_root, "03_Residual_UNet")

for path in [common_root, res_unet_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model import ResidualUNet, ResBlock
from training.losses import DiceLoss


class TestResidualUNetArchitecture(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = ResidualUNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False).to(self.device)

    def test_01_instantiation(self):
        """1. Model and ResBlock can be instantiated."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model, ResidualUNet)
        block = ResBlock(64, 128)
        self.assertIsInstance(block, nn.Module)

    def test_02_resblock_functionality(self):
        """2. ResBlock preserves/changes channels correctly with residual shortcuts."""
        # Identity shortcut case (in_channels == out_channels)
        block_same = ResBlock(64, 64).to(self.device)
        x_same = torch.randn(2, 64, 32, 32, device=self.device)
        out_same = block_same(x_same)
        self.assertEqual(out_same.shape, torch.Size([2, 64, 32, 32]))
        self.assertIsInstance(block_same.shortcut, nn.Identity)

        # Projection shortcut case (in_channels != out_channels)
        block_diff = ResBlock(64, 128).to(self.device)
        x_diff = torch.randn(2, 64, 32, 32, device=self.device)
        out_diff = block_diff(x_diff)
        self.assertEqual(out_diff.shape, torch.Size([2, 128, 32, 32]))
        self.assertIsInstance(block_diff.shortcut, nn.Sequential)

    def test_04_forward_pass_shape(self):
        """4. Input [2, 1, 256, 256] -> Output [2, 1, 256, 256]."""
        input_tensor = torch.randn(2, 1, 256, 256, device=self.device)
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, torch.Size([2, 1, 256, 256]))

    def test_05_finite_outputs(self):
        """5. Output contains finite values (no NaNs, no Infs)."""
        input_tensor = torch.randn(2, 1, 256, 256, device=self.device)
        output_tensor = self.model(input_tensor)
        self.assertTrue(torch.isfinite(output_tensor).all().item())

    def test_06_backpropagation(self):
        """6. Backpropagation succeeds and gradients flow to residual shortcuts and convs."""
        input_tensor = torch.randn(2, 1, 256, 256, device=self.device, requires_grad=True)
        target_tensor = (torch.rand(2, 1, 256, 256, device=self.device) > 0.5).float()

        output_tensor = self.model(input_tensor)
        loss = output_tensor.sum()
        loss.backward()

        # Check gradients exist in encoder and residual blocks
        self.assertIsNotNone(self.model.inc.conv1.weight.grad)
        self.assertFalse(torch.isnan(self.model.inc.conv1.weight.grad).any().item())
        self.assertIsNotNone(self.model.inc.shortcut[0].weight.grad)
        self.assertFalse(torch.isnan(self.model.inc.shortcut[0].weight.grad).any().item())

    def test_07_parameter_count(self):
        """7. Model parameter count is reproducible (32,435,073 parameters)."""
        param_count = self.model.count_parameters()
        self.assertEqual(param_count, 32435073)

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


if __name__ == "__main__":
    unittest.main()
