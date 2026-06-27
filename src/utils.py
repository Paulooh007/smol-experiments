"""Shared utilities: hardware detection, dtype selection, and seeding.

Every script in this repo must run on CPU, Apple Silicon Macs, free-tier Colab
T4s (no bf16 support), and Ampere+ GPUs (bf16). The helpers here centralize
that logic so no script ever hardcodes "cuda" or a dtype.
"""

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the best available local device.

    CUDA wins when present because it supports the mixed-precision paths used
    by the custom training loops. On Apple Silicon, MPS is the next-best local
    target; it runs the repo in float32 without AMP.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_dtype() -> torch.dtype:
    """Auto-detect the best precision for the current hardware.

    - Ampere+ GPUs (compute capability >= 8.0): bfloat16
    - Older GPUs such as the Colab T4: float16 (T4 does NOT support bf16)
    - Apple MPS / CPU: float32 (AMP is not used for these devices in this repo)
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    return torch.float32


def amp_enabled(device: torch.device) -> bool:
    """Mixed-precision autocast is only used when running on CUDA."""
    return device.type == "cuda"


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (incl. CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Total (or trainable) parameter count of a model."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
