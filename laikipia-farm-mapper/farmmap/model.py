"""U-Net (segmentation_models_pytorch) with a pretrained encoder: 3 classes background / interior / boundary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NIR_MEAN, NIR_STD = 0.5, 0.25


def build(arch: str, encoder: str, in_channels: int, encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    return smp.create_model(arch, encoder_name=encoder, encoder_weights=encoder_weights,
                            in_channels=in_channels, classes=3)


def normalize(img_uint8: np.ndarray) -> np.ndarray:
    """(C,H,W) uint8 -> float32; RGB with ImageNet stats (pretrained encoder), NIR with a fixed stat."""
    x = img_uint8.astype(np.float32) / 255.0
    c = x.shape[0]
    mean = np.concatenate([IMAGENET_MEAN, np.full(max(0, c - 3), NIR_MEAN, np.float32)])[:c]
    std = np.concatenate([IMAGENET_STD, np.full(max(0, c - 3), NIR_STD, np.float32)])[:c]
    return (x - mean[:, None, None]) / std[:, None, None]


def save(path: Path, model, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: Path, device: str = "cpu") -> dict:
    """Checkpoints written by this package. PyTorch >= 2.6 loads with weights_only=True by default, which rejects
    the metadata saved by older versions (config objects, numpy numbers); these files are our own, so allow it."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location=device)


def load(path: Path, device: str = "cpu"):
    ckpt = load_checkpoint(path, device)
    m = ckpt["meta"]
    model = build(m["arch"], m["encoder"], m["in_channels"], encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), m


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
