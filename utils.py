"""
utils.py
--------
Utilitas kecil yang dipakai train.py dan evaluate.py: seed global & pemilihan device.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seed untuk Python, NumPy, dan PyTorch (CPU + semua GPU).

    deterministic=True mematikan cudnn.benchmark dan mengaktifkan cudnn.deterministic
    agar hasil lebih reproducible (sedikit lebih lambat). Catatan: reproducibility
    bit-per-bit antar mesin/GPU/versi PyTorch yang berbeda tidak dijamin.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """auto -> cuda jika ada, lalu mps (Apple Silicon), lalu cpu."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB), CUDA {torch.version.cuda}"
    if device.type == "mps":
        return "GPU: Apple Silicon (MPS)"
    return "CPU"
