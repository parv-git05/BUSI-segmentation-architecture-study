"""
Unit Tests for Attention U-Net Architecture.
Verifies tensor shapes, attention gate output shapes/coefficient ranges, forward/backward passes,
finite outputs, loss compatibility, and parameter count determinism.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Add 00_Common and 02_Attention_UNet directories to sys.path
models_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
common_root = os.path.join(models_root, "00_Common")
att_unet_root = os.path.join(models_root, "02_Attention_UNet")

for path in [common_root, att_unet_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model import AttentionUNet, AttentionGate
from training.losses import DiceLoss


class TestAttentionUNetArchitecture(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = AttentionUNet(in_channels=1, out_channels=1, base_channels=64, bilinear=False).to(self.device)

    def test_01_instantiation(self):
        """1. Model and AttentionGate can be instantiated."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model, AttentionUNet)
        ag = AttentionGate(in_channels_x=512, in_channels_g=512, inter_channels=256)
        self.assertIsInstance(ag, nn.Module)

    def test_02_attention_gate_output_shape(self):
        """2. Attention Gate returns tensor matching skip feature map shape."""
        ag = AttentionGate(in_channels_x=512, in_channels_g=512, inter_channels=256).to(self.device)
        skip_x = torch.randn(2, 512, 32, 32, device=self.device)
        gate_g = torch.randn(2, 512, 32, 32, device=self.device)
        gated_x = ag(skip_x, gate_g)
        self.assertEqual(gated_x.shape, torch.Size([2, 512, 32, 32]))

    def test_03_attention_coefficient_shape_and_range(self):
        """3. Attention coefficients have shape [B, 1, H, W] and values strictly in range [0, 1]."""
        ag = AttentionGate(in_channels_x=256, in_channels_g=256, inter_channels=128).to(self.device)
        skip_x = torch.randn(2, 256, 64, 64, device=self.device)
        gate_g = torch.randn(2, 256, 64, 64, device=self.device)
        
        # Test internal psi activation output
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
        """6. Backpropagation succeeds and gradients flow to attention gates and encoder."""
        input_tensor = torch.randn(2, 1, 256, 256, device=self.device, requires_grad=True)
        target_tensor = (torch.rand(2, 1, 256, 256, device=self.device) > 0.5).float()

        output_tensor = self.model(input_tensor)
        loss = output_tensor.sum()
        loss.backward()

        # Check gradients exist in encoder and attention gates
        self.assertIsNotNone(self.model.inc.double_conv[0].weight.grad)
        self.assertFalse(torch.isnan(self.model.inc.double_conv[0].weight.grad).any().item())
        self.assertIsNotNone(self.model.up1.ag.W_g[0].weight.grad)
        self.assertFalse(torch.isnan(self.model.up1.ag.W_g[0].weight.grad).any().item())

    def test_07_parameter_count(self):
        """7. Model parameter count is reproducible (31,387,049 parameters)."""
        param_count = self.model.count_parameters()
        self.assertEqual(param_count, 31387049)

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
