"""
Reusable PyTorch Dataset and DataLoader factory for the BUSI Phase-1 Dataset.
Reads exclusively from pre-generated MASTER_SPLIT CSV manifest files.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import (
    BUSITransform,
    get_train_transforms,
    get_val_transforms,
    get_test_transforms
)


class BUSIDataset(Dataset):
    """
    PyTorch Dataset implementation for BUSI Phase-1 Breast Ultrasound Segmentation.
    Loads single-mask lesion images strictly based on MASTER_SPLIT CSV manifests.
    """

    def __init__(
        self,
        manifest_path: str,
        dataset_root: Optional[str] = None,
        transform: Optional[BUSITransform] = None,
        img_channels: int = 1,
        return_metadata: bool = True
    ):
        """
        Args:
            manifest_path (str): Path to master split CSV (e.g., train_70.csv, val_20.csv, test_10.csv).
            dataset_root (Optional[str]): Root directory where images reside if paths need rebasing.
                                          Defaults to workspace directory if None.
            transform (Optional[BUSITransform]): Paired transformation pipeline.
            img_channels (int): 1 for grayscale ('L'), 3 for RGB ('RGB'). Default is 1.
            return_metadata (bool): If True, __getitem__ returns dictionary with path & metadata.
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

        self.manifest_path = os.path.abspath(manifest_path)
        self.df = pd.read_csv(self.manifest_path)
        self.transform = transform
        self.img_channels = img_channels
        self.return_metadata = return_metadata

        # Determine workspace / dataset root for path resolution
        if dataset_root is not None:
            self.dataset_root = os.path.abspath(dataset_root)
        else:
            # Search candidate roots for the directory containing 'benign' folder
            candidates = [
                os.path.abspath(os.getcwd()),
                os.path.dirname(os.path.abspath(self.manifest_path)),
                os.path.dirname(os.path.dirname(os.path.abspath(self.manifest_path))),
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(self.manifest_path))))
            ]
            self.dataset_root = candidates[0]
            for cand in candidates:
                if os.path.exists(os.path.join(cand, "benign")):
                    self.dataset_root = cand
                    break

        # Validate that manifest contains required columns
        required_cols = {"image_path", "mask_path", "class", "class_id"}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Manifest CSV missing required columns: {missing_cols}")

        # Resolve all file paths immediately during initialization to guarantee integrity
        self.resolved_image_paths = []
        self.resolved_mask_paths = []
        self._verify_and_resolve_paths()

    def _resolve_path(self, raw_path: str) -> str:
        """
        Resolves stale absolute prefixes (e.g. D:\\Dataset_BUSI_with_GT or F:\\Research_BUSI_Segmentation)
        to valid existing local file paths on disk.
        """
        if os.path.exists(raw_path):
            return os.path.abspath(raw_path)

        # Normalize slashes
        norm_path = raw_path.replace("\\", "/")
        parts = norm_path.split("/")
        
        # Extract relative path starting from class subfolder ('benign', 'malignant', 'normal')
        rel_subpath = None
        for i, part in enumerate(parts):
            if part.lower() in ["benign", "malignant", "normal"]:
                rel_subpath = "/".join(parts[i:])
                break

        if rel_subpath is None:
            rel_subpath = parts[-1]  # fallback to filename

        resolved_candidate = os.path.abspath(os.path.join(self.dataset_root, rel_subpath))
        if os.path.exists(resolved_candidate):
            return resolved_candidate

        # Raise explicit error if resolution fails
        raise FileNotFoundError(
            f"Unable to resolve valid file path for manifest entry: '{raw_path}'. "
            f"Attempted resolution at: '{resolved_candidate}'"
        )

    def _verify_and_resolve_paths(self) -> None:
        """Audits every row in the manifest for path existence."""
        for idx, row in self.df.iterrows():
            img_path = self._resolve_path(row["image_path"])
            mask_path = self._resolve_path(row["mask_path"])
            self.resolved_image_paths.append(img_path)
            self.resolved_mask_paths.append(mask_path)

    def __len__(self) -> int:
        """Returns the exact number of samples. No samples are silently discarded."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads image-mask pair at index `idx` and returns formatted PyTorch tensors.
        """
        img_path = self.resolved_image_paths[idx]
        mask_path = self.resolved_mask_paths[idx]
        row = self.df.iloc[idx]

        # 1. Load Image
        try:
            image = Image.open(img_path)
            if self.img_channels == 1:
                image = image.convert("L")
            else:
                image = image.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image file at {img_path}: {e}")

        # 2. Load Ground-Truth Mask
        try:
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            raise IOError(f"Failed to load mask file at {mask_path}: {e}")

        # 3. Validate Spatial Dimensions
        if image.size != mask.size:  # PIL size is (width, height)
            raise ValueError(
                f"Dimension mismatch at index {idx} ({os.path.basename(img_path)}): "
                f"Image size {image.size} != Mask size {mask.size}"
            )

        # 4. Apply Synchronous Preprocessing & Augmentation Pipeline
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            # Fallback default conversion if no transform provided
            image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
            if self.img_channels == 1:
                image_tensor = image_tensor.unsqueeze(0)  # (1, H, W)
            else:
                image_tensor = image_tensor.permute(2, 0, 1)  # (3, H, W)

            mask_arr = np.array(mask, dtype=np.float32)
            mask_tensor = torch.from_numpy((mask_arr > 127.5).astype(np.float32)).unsqueeze(0)  # (1, H, W)

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": img_path,
            "mask_path": mask_path,
            "class_name": str(row["class"]),
            "class_id": int(row["class_id"]),
            "sample_id": os.path.basename(img_path),
            "index": idx
        }

        return sample


def create_dataloader(
    manifest_path: str,
    batch_size: int = 16,
    shuffle: bool = False,
    transform: Optional[BUSITransform] = None,
    num_workers: int = 0,
    seed: int = 42,
    dataset_root: Optional[str] = None,
    img_channels: int = 1
) -> DataLoader:
    """
    Factory function to construct a PyTorch DataLoader from a BUSI split manifest.
    
    Args:
        manifest_path (str): CSV manifest path.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle dataset samples.
        transform (Optional[BUSITransform]): Transform pipeline.
        num_workers (int): Number of worker processes.
        seed (int): Random seed for reproducibility.
        dataset_root (Optional[str]): Root directory of BUSI dataset images.
        img_channels (int): Number of image channels (1 or 3).
        
    Returns:
        DataLoader: Configured PyTorch DataLoader instance.
    """
    dataset = BUSIDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        transform=transform,
        img_channels=img_channels
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader
