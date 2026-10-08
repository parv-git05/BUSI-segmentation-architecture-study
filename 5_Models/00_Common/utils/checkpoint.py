"""
Checkpoint saving and loading utilities for reproducible model evaluation.
"""

import os
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    val_metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    experiment_id: str = "unspecified_exp"
) -> None:
    """
    Saves a comprehensive experiment checkpoint atomically.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Extract clean model state dict (unwrapping DataParallel if present)
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    checkpoint_state = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics or {},
        "config": config or {},
        "seed": seed,
        "experiment_id": experiment_id
    }
    
    # Write atomically via temp file to avoid file corruption
    tmp_filepath = f"{filepath}.tmp"
    torch.save(checkpoint_state, tmp_filepath)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp_filepath, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """
    Loads checkpoint state into model, optimizer, and scheduler.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    # Load scheduler state if provided
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    return checkpoint
