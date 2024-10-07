"""
This module provides utility functions for managing device compatibility, memory usage, 
and cleanup operations.
"""

import torch
import psutil
import logging
import gc
from config import setup_logging

# Ensure logging is set up
setup_logging()


def check_gpu_compatibility() -> bool:
    """
    Checks if a compatible GPU is available and logs GPU properties.

    Returns:
        True if a GPU is available, False otherwise.
    """
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        total_memory = gpu_properties.total_memory / (1024**3)  # Convert to GB
        logging.info(
            f"GPU detected: {gpu_properties.name} with {total_memory:.2f} GB VRAM"
        )
        return True
    else:
        # logging.warning("No GPU detected")
        logging.critical("NO GPU AVAILABLE!!!")  # TODO CPU ONLY NOT SUPPORTED
        return False


def get_available_memory() -> float:
    """
    Returns the available memory (in GB) on the device.

    Returns:
        Available memory in GB. GPU memory is returned if available, otherwise system memory.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )  # Convert to GB
    else:
        return psutil.virtual_memory().total / (1024**3)  # Convert to GB


def get_device(min_memory_gb: float = 14) -> torch.device:
    """
    Determines the appropriate device (GPU or CPU) based on availability and memory constraints.

    Args:
        min_memory_gb: Minimum GPU memory in GB required to use the GPU. Defaults to 14 GB.

    Returns:
        The appropriate device (CUDA if available and sufficient memory, otherwise CPU).
    """
    if check_gpu_compatibility() and get_available_memory() >= min_memory_gb:
        return torch.device("cuda")
    else:
        logging.warning(f"Insufficient GPU memory. Using CPU.")
        return torch.device("cpu")


def cleanup_memory() -> None:
    """
    Cleans up unused memory on the device by emptying CUDA cache and running garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
