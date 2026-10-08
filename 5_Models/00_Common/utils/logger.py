"""
Logging utilities for standardized experiment console and file outputs.
"""

import os
import sys
import logging
from typing import Optional


def setup_logger(
    name: str = "BUSI_Experiment",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a logger with standard console output and optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Close and clear existing handlers if re-initializing
    close_logger(logger)
        
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Closes all file handlers attached to logger to release Windows file locks."""
    if logger is None:
        return
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
