"""
Unit Tests for Attention Residual U-Net Architecture.
Verifies tensor shapes, residual blocks, attention gates, coefficient range, forward/backward passes,
finite outputs, loss compatibility, and parameter count determinism.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Add 00_Common and 04_Attention_Residual_UNet directories to sys.path
models_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
common_root = os.path.join(models_root, "00_Common")
att_res_unet_root = os.path.join(models_root, "04_Attention_Residual_UNet")

for path in [common_root, att_res_unet_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model import AttentionResidualUNet, ResBlock, AttentionGate
from training.losses import DiceLoss


class TestAttentionResidualUNetArchitecture(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = AttentionResidualUNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False).to(self.device)

    def test_01_instantiation(self):
        """1. Model, ResBlock, and AttentionGate can be instantiated."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model, AttentionResidualUNet)
        block = ResBlock(64, 128)
        ag = AttentionGate(512, 512, 256)
        self.assertIsInstance(block, nn.Module)
        self.assertIsInstance(ag, nn.Module)

    def test_02_attention_coefficient_shape_and_range(self):
        """2. Attention coefficients have shape [B, 1, H, W] and range [0, 1]."""
        ag = AttentionGate(in_channels_x=256, in_channels_g=256, inter_channels=128).to(self.device)
        skip_x = torch.randn(2, 256, 64, 64, device=self.device)
        gate_g = torch.randn(2, 256, 64, 64, device=self.device)

        g1 = ag.W_g(gate_g)
        x1 = ag.W_x(skip_x)
        f = ag.relu(g1 + x1)
        alpha = ag.psi(f)

        self.assertEqual(alpha.shape, torch.Size([2, 1, 64, 64]))
        self.assertTrue((alpha >= 0.0).all().item())
        self.assertTrue((alpha <= 1.0).all().item())

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
        """6. Backpropagation succeeds and gradients flow to attention gates, resblocks, and encoder."""
        input_tensor = torch.randn(2, 1, 256, 256, device=self.device, requires_grad=True)
        target_tensor = (torch.rand(2, 1, 256, 256, device=self.device) > 0.5).float()

        output_tensor = self.model(input_tensor)
        loss = output_tensor.sum()
        loss.backward()

        self.assertIsNotNone(self.model.inc.conv1.weight.grad)
        self.assertFalse(torch.isnan(self.model.inc.conv1.weight.grad).any().item())
        self.assertIsNotNone(self.model.up1.ag.W_g[0].weight.grad)
        self.assertFalse(torch.isnan(self.model.up1.ag.W_g[0].weight.grad).any().item())

    def test_07_parameter_count(self):
        """7. Model parameter count is reproducible (32,785,641 parameters)."""
        param_count = self.model.count_parameters()
        self.assertEqual(param_count, 32785641)

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
