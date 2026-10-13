"""U-Net model: 3-class semantic segmentation (background / farm interior / farm boundary).

Predicting the *boundary* as its own class is what lets us split touching farms into
separate polygons afterwards (see polygonize.py).
"""
from __future__ import annotations

import warnings
from pathlib import Path

CLASSES = ("background", "interior", "boundary")


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(encoder: str, in_channels: int, encoder_weights=None):
    import segmentation_models_pytorch as smp

    try:
        return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                        in_channels=in_channels, classes=len(CLASSES))
    except Exception as exc:  # offline: pretrained weights cannot be downloaded
        if encoder_weights is None:
            raise
        warnings.warn(f"Could not load '{encoder_weights}' encoder weights ({exc}); training from scratch.")
        return smp.Unet(encoder_name=encoder, encoder_weights=None,
                        in_channels=in_channels, classes=len(CLASSES))


def save_checkpoint(path, model, cfg, norm, epoch, metrics):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "encoder": cfg.model.encoder, "norm": norm,
                "epoch": epoch, "metrics": metrics}, path)


def load_checkpoint(path, device="cpu"):
    """-> (model in eval mode, norm dict, checkpoint dict)"""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n_ch = ckpt["state_dict"]["encoder.conv1.weight"].shape[1]
    model = build_model(ckpt["encoder"], n_ch, encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["norm"], ckpt
