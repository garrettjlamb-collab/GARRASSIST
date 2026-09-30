from __future__ import annotations

import os


def pick_device(prefer: str | None = None) -> str:
    """Torch device to use: an explicit preference, else FILM_DEVICE, else mps > cuda > cpu."""
    choice = prefer or os.environ.get("FILM_DEVICE")
    if choice:
        return choice
    try:
        import torch
    except ImportError:
        return "cpu"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
