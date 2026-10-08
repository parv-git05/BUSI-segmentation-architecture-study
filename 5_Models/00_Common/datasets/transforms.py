"""
Common Preprocessing and Augmentation Pipeline for BUSI Segmentation.
Strictly separates Training (optional augmentations) from Validation and Testing (deterministic preprocessing only).
"""

import random
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


class BUSITransform:
    """
    Unified transformation pipeline for image and mask pairs.
    Handles resizing, augmentations, and tensor conversion synchronously.
    """
    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        is_train: bool = False,
        augment: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotation_degrees: float = 15.0,
        normalize_mean: Optional[Tuple[float, ...]] = None,
        normalize_std: Optional[Tuple[float, ...]] = None,
    ):
        """
        Args:
            input_size (Tuple[int, int]): Target (height, width) for resizing.
            is_train (bool): If True, allows data augmentation if augment=True.
            augment (bool): Enable geometric random data augmentations.
            hflip_prob (float): Probability of horizontal flip during training.
            vflip_prob (float): Probability of vertical flip during training.
            rotation_degrees (float): Maximum random rotation angle in degrees.
            normalize_mean (Optional[Tuple[float, ...]]): Mean for normalization (optional).
            normalize_std (Optional[Tuple[float, ...]]): Std for normalization (optional).
        """
        self.input_size = input_size  # (H, W)
        self.is_train = is_train
        self.augment = augment and is_train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_degrees = rotation_degrees
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies transformations to paired Image and Mask.
        
        Args:
            image (PIL.Image.Image): Input grayscale ultrasound image.
            mask (PIL.Image.Image): Input binary lesion ground-truth mask.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image_tensor: FloatTensor of shape (C, H, W) in [0.0, 1.0]
                - mask_tensor: FloatTensor of shape (1, H, W) in {0.0, 1.0}
        """
        # 1. Spatial Resizing
        # Image uses BILINEAR interpolation; Mask MUST use NEAREST to preserve discrete labels
        target_h, target_w = self.input_size
        image = image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
        mask = mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)

        # 2. Geometric Augmentation (TRAIN ONLY)
        if self.augment:
            # Random Horizontal Flip
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random Vertical Flip
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random Rotation
            if self.rotation_degrees > 0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 3. Convert to PyTorch Tensors
        image_tensor = TF.to_tensor(image)  # Shape (C, H, W), float32, range [0.0, 1.0]
        mask_tensor = TF.to_tensor(mask)    # Shape (1, H, W), float32

        # 4. Enforce strict binary representation for mask {0.0, 1.0}
        mask_tensor = (mask_tensor > 0.5).to(torch.float32)

        # 5. Optional Normalization
        if self.normalize_mean is not None and self.normalize_std is not None:
            image_tensor = TF.normalize(image_tensor, self.normalize_mean, self.normalize_std)

        return image_tensor, mask_tensor


def get_train_transforms(
    input_size: Tuple[int, int] = (256, 256),
    augment: bool = True,
    **kwargs
) -> BUSITransform:
    """Factory function for training transformations."""
    return BUSITransform(input_size=input_size, is_train=True, augment=augment, **kwargs)


def get_val_transforms(
    input_size: Tuple[int, int] = (256, 256),
    **kwargs
) -> BUSITransform:
    """Factory function for validation transformations (preprocessing only)."""
    return BUSITransform(input_size=input_size, is_train=False, augment=False, **kwargs)


def get_test_transforms(
    input_size: Tuple[int, int] = (256, 256),
    **kwargs
) -> BUSITransform:
    """Factory function for test transformations (preprocessing only, no TTA)."""
    return BUSITransform(input_size=input_size, is_train=False, augment=False, **kwargs)
