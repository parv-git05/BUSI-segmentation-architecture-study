"""
Reproducibility utilities for seed control.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Enforce global reproducibility by seeding random, numpy, and torch RNGs.
    
    Args:
        seed (int): The integer seed to set across all RNGs. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
