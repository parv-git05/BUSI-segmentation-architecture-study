"""
Isolated Loss Functions Module for BUSI Segmentation.
Implements the initial baseline Dice Loss required for 5-architecture benchmark.
Designed with abstract base classes to allow future loss functions to be added seamlessly.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSegmentationLoss(nn.Module):
    """Abstract base class for all segmentation losses in the study."""
    def __init__(self, name: str = "base_loss"):
        super().__init__()
        self.name = name

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward pass.")


class DiceLoss(BaseSegmentationLoss):
    """
    Binary Dice Loss implementation.
    
    Formula:
        Dice = (2 * sum(P * G) + smooth) / (sum(P) + sum(G) + smooth)
        Loss = 1.0 - Dice
    """
    def __init__(
        self,
        smooth: float = 1e-6,
        from_logits: bool = True,
        square_denominator: bool = False,
        reduction: str = "mean"
    ):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
            from_logits (bool): If True, applies sigmoid activation to input predictions.
            square_denominator (bool): If True, uses sum(P^2) + sum(G^2) in denominator.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super().__init__(name="DiceLoss")
        self.smooth = smooth
        self.from_logits = from_logits
        self.square_denominator = square_denominator
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Model output logits or probabilities of shape (B, 1, H, W) or (B, H, W).
            targets (torch.Tensor): Ground-truth binary masks of shape (B, 1, H, W) or (B, H, W).
            
        Returns:
            torch.Tensor: Computed scalar Dice Loss.
        """
        if self.from_logits:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        if probs.shape != targets.shape:
            raise ValueError(f"Shape mismatch in DiceLoss: inputs {probs.shape} != targets {targets.shape}")

        batch_size = probs.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)

        if self.square_denominator:
            denominator = torch.sum(probs_flat ** 2, dim=1) + torch.sum(targets_flat ** 2, dim=1)
        else:
            denominator = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice_score

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        elif self.reduction == "none":
            return dice_loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")


class BCELoss(BaseSegmentationLoss):
    """
    Binary Cross-Entropy Loss for binary segmentation.
    Uses numerically stable BCE with logits when from_logits=True.
    
    Formula:
        BCE = - [G * log(P) + (1 - G) * log(1 - P)]
    """
    def __init__(
        self,
        from_logits: bool = True,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__(name="BCELoss")
        self.from_logits = from_logits
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch in BCELoss: inputs {inputs.shape} != targets {targets.shape}")

        targets = targets.float()
        if self.from_logits:
            return F.binary_cross_entropy_with_logits(
                inputs, targets, reduction=self.reduction, pos_weight=self.pos_weight
            )
        else:
            return F.binary_cross_entropy(
                inputs, targets, reduction=self.reduction
            )


class BCEDiceLoss(BaseSegmentationLoss):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    Combines pixel-level classification (BCE) with region-level overlap (Dice).
    
    Formula:
        Loss = bce_weight * BCE + dice_weight * DiceLoss
    """
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
        from_logits: bool = True,
        reduction: str = "mean"
    ):
        super().__init__(name="BCEDiceLoss")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = BCELoss(from_logits=from_logits, reduction=reduction)
        self.dice_loss = DiceLoss(smooth=smooth, from_logits=from_logits, reduction=reduction)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch in BCEDiceLoss: inputs {inputs.shape} != targets {targets.shape}")

        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalLoss(BaseSegmentationLoss):
    """
    Binary Focal Loss (Lin et al., 2017).
    Downweights easy examples and focuses training on hard foreground/background pixels.
    
    Formula:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
        where p_t = p if y=1 else (1 - p), and alpha_t = alpha if y=1 else (1 - alpha)
    """
    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        from_logits: bool = True,
        reduction: str = "mean"
    ):
        super().__init__(name="FocalLoss")
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch in FocalLoss: inputs {inputs.shape} != targets {targets.shape}")

        targets = targets.float()

        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            probs = torch.sigmoid(inputs)
        else:
            probs = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)
            bce_loss = F.binary_cross_entropy(probs, targets, reduction="none")

        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        modulating_factor = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            focal_loss = alpha_t * modulating_factor * bce_loss
        else:
            focal_loss = modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")


class TverskyLoss(BaseSegmentationLoss):
    """
    Tversky Loss for image segmentation (Salehi et al., 2017).
    Generalizes Dice coefficient by introducing asymmetric weights (alpha, beta) for False Negatives and False Positives.
    Setting alpha > beta penalizes False Negatives more heavily to improve recall on small/sparse lesions.
    
    Formula:
        TI = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
        Loss = 1.0 - TI
    """
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-6,
        from_logits: bool = True,
        reduction: str = "mean"
    ):
        super().__init__(name="TverskyLoss")
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        if probs.shape != targets.shape:
            raise ValueError(f"Shape mismatch in TverskyLoss: inputs {probs.shape} != targets {targets.shape}")

        batch_size = probs.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        # True Positives, False Positives, False Negatives
        tp = torch.sum(probs_flat * targets_flat, dim=1)
        fp = torch.sum(probs_flat * (1.0 - targets_flat), dim=1)
        fn = torch.sum((1.0 - probs_flat) * targets_flat, dim=1)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = 1.0 - tversky_index

        if self.reduction == "mean":
            return tversky_loss.mean()
        elif self.reduction == "sum":
            return tversky_loss.sum()
        elif self.reduction == "none":
            return tversky_loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")


def get_loss_function(name: str = "DiceLoss", **kwargs) -> BaseSegmentationLoss:
    """
    Factory function for loss functions.
    Supported losses: 'DiceLoss' ('Dice'), 'BCELoss' ('BCE'), 'BCEDiceLoss' ('BCE_Dice'),
                      'FocalLoss' ('Focal'), 'TverskyLoss' ('Tversky').
    """
    name_clean = name.lower().replace("-", "_").replace(" ", "_")
    if name_clean in ["diceloss", "dice"]:
        return DiceLoss(**kwargs)
    elif name_clean in ["bceloss", "bce"]:
        return BCELoss(**kwargs)
    elif name_clean in ["bcediceloss", "bce_dice", "bce+dice", "bcedice"]:
        return BCEDiceLoss(**kwargs)
    elif name_clean in ["focalloss", "focal"]:
        return FocalLoss(**kwargs)
    elif name_clean in ["tverskyloss", "tversky"]:
        return TverskyLoss(**kwargs)
    else:
        raise ValueError(
            f"Loss function '{name}' is not supported. "
            f"Supported losses: DiceLoss, BCELoss, BCEDiceLoss, FocalLoss, TverskyLoss."
        )
