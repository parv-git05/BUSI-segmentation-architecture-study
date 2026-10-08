"""
Unit tests for all Loss implementations (Dice, BCE, BCE_Dice, Focal, Tversky).
"""

import os
import sys
import unittest
import torch

# Add 00_Common directory to sys.path
common_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if common_root not in sys.path:
    sys.path.insert(0, common_root)

from training.losses import (
    DiceLoss,
    BCELoss,
    BCEDiceLoss,
    FocalLoss,
    TverskyLoss,
    get_loss_function
)


class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_dice_loss(self):
        """Verifies Dice loss calculation and backward pass."""
        criterion = DiceLoss(smooth=1e-6, from_logits=True)
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = (torch.rand(2, 1, 16, 16) > 0.5).float()

        loss = criterion(logits, targets)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_bce_loss(self):
        """Verifies BCE loss calculation and backward pass."""
        criterion = BCELoss(from_logits=True)
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = (torch.rand(2, 1, 16, 16) > 0.5).float()

        loss = criterion(logits, targets)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_bce_dice_loss(self):
        """Verifies combined BCE_Dice loss calculation and backward pass."""
        criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, smooth=1e-6, from_logits=True)
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = (torch.rand(2, 1, 16, 16) > 0.5).float()

        loss = criterion(logits, targets)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_focal_loss(self):
        """Verifies Focal loss calculation and backward pass."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = (torch.rand(2, 1, 16, 16) > 0.5).float()

        loss = criterion(logits, targets)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_tversky_loss(self):
        """Verifies Tversky loss calculation and backward pass."""
        criterion = TverskyLoss(alpha=0.7, beta=0.3, smooth=1e-6, from_logits=True)
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = (torch.rand(2, 1, 16, 16) > 0.5).float()

        loss = criterion(logits, targets)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_factory_function(self):
        """Verifies get_loss_function factory for all 5 loss types."""
        self.assertIsInstance(get_loss_function("Dice"), DiceLoss)
        self.assertIsInstance(get_loss_function("DiceLoss"), DiceLoss)
        self.assertIsInstance(get_loss_function("BCE"), BCELoss)
        self.assertIsInstance(get_loss_function("BCELoss"), BCELoss)
        self.assertIsInstance(get_loss_function("BCE_Dice"), BCEDiceLoss)
        self.assertIsInstance(get_loss_function("BCEDiceLoss"), BCEDiceLoss)
        self.assertIsInstance(get_loss_function("Focal"), FocalLoss)
        self.assertIsInstance(get_loss_function("FocalLoss"), FocalLoss)
        self.assertIsInstance(get_loss_function("Tversky"), TverskyLoss)
        self.assertIsInstance(get_loss_function("TverskyLoss"), TverskyLoss)

        with self.assertRaises(ValueError):
            get_loss_function("UnknownLoss")


if __name__ == "__main__":
    unittest.main()
