"""
Unit Tests for U-Net++ (Nested U-Net) Architecture.
Verifies tensor shapes, nested pathway tensor compatibility, forward/backward passes,
finite outputs, loss compatibility, and parameter count determinism.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Add 00_Common and 05_UNetPP directories to sys.path
models_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
common_root = os.path.join(models_root, "00_Common")
unetpp_root = os.path.join(models_root, "05_UNetPP")

for path in [common_root, unetpp_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model import UNetPlusPlus
from training.losses import DiceLoss


class TestUNetPlusPlusArchitecture(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = UNetPlusPlus(in_channels=1, out_channels=1, base_channels=64, deep_supervision=False).to(self.device)

    def test_01_instantiation(self):
        """1. Model can be instantiated."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model, UNetPlusPlus)

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
        """6. Backpropagation succeeds and gradients flow cleanly through dense nested pathways."""
        input_tensor = torch.randn(2, 1, 256, 256, device=self.device, requires_grad=True)
        target_tensor = (torch.rand(2, 1, 256, 256, device=self.device) > 0.5).float()

        output_tensor = self.model(input_tensor)
        loss = output_tensor.sum()
        loss.backward()

        self.assertIsNotNone(self.model.conv0_0.double_conv[0].weight.grad)
        self.assertFalse(torch.isnan(self.model.conv0_0.double_conv[0].weight.grad).any().item())
        self.assertIsNotNone(self.model.conv0_4.double_conv[0].weight.grad)
        self.assertFalse(torch.isnan(self.model.conv0_4.double_conv[0].weight.grad).any().item())

    def test_07_parameter_count(self):
        """7. Model parameter count is reproducible (36,164,097 parameters)."""
        param_count = self.model.count_parameters()
        self.assertEqual(param_count, 36164097)

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
