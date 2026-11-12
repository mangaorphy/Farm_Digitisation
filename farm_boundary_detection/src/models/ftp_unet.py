"""Fields of the Planet (FTP) U-Net as a drop-in model for this pipeline.

FTP-PRUE+ ("Fields of the Planet: Field Boundary Mapping Beyond 10m", arXiv:2607.04449, 2026) is a
U-Net with an EfficientNet-B3 encoder trained on 3 m PlanetScope surface reflectance and the 1.6 million field
polygons of Fields of The World (24 countries, Kenya included). It is the same sensor, resolution and task as
this project, so it is a far better starting point than ImageNet weights. Weights: Hugging Face
``taylor-geospatial/ftp-b3`` (CC-BY-NC-4.0: research / non-commercial use).

What the released network expects, and how this wrapper provides it:

* 8 channels = two seasonal images x 4 bands (blue, green, red, NIR), reflectance = DN / 10000.
  -> the wrapper picks blue, green, red, NIR from the pipeline's bands (8-band SuperDove or 4-band) and gives
  the same image as both seasons (the network was trained with one season randomly blanked or swapped, so it
  tolerates this, and fine-tuning adapts it). Use ``normalization.mode: scale`` so the input is DN / 10000.
* 3 exclusive classes: background, field interior, field boundary.
  -> converted exactly to this pipeline's two logits: P(farm) = P(interior) + P(boundary) (our interior mask
  includes the boundary pixels) and P(boundary). Loss, metrics, prediction and polygon building are unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
HF_ID = "taylor-geospatial/ftp-b3"
HF_ID_ALT = "fieldsoftheworld/ftp-b3"


def bgrn_indices(in_channels: int, band_names: list[str] | None = None) -> list[int]:
    """0-based positions of blue, green, red, NIR among the pipeline's bands."""
    if band_names:
        names = [str(n).lower() for n in band_names]
        if all(n in names for n in ("blue", "green", "red", "nir")):
            return [names.index(n) for n in ("blue", "green", "red", "nir")]
    if in_channels == 8:      # SuperDove: coastal blue, blue, green I, green, yellow, red, red edge, NIR
        return [1, 3, 5, 7]
    if in_channels == 4:      # Dove: blue, green, red, NIR
        return [0, 1, 2, 3]
    raise ValueError(f"cannot find blue/green/red/NIR among {in_channels} bands {band_names}")


class FTPUnet(nn.Module):
    has_distance = False

    def __init__(self, in_channels: int, band_idx: list[int] | None = None, pretrained: bool = True,
                 encoder: str = "efficientnet-b3", hf_id: str = HF_ID, second_window: str = "copy",
                 input_upsample: int = 1):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:  # pragma: no cover
            raise ImportError("model.arch=ftp_unet needs: pip install segmentation-models-pytorch") from e
        self.in_channels = int(in_channels)
        self.band_idx = list(band_idx) if band_idx is not None else bgrn_indices(in_channels)
        if second_window not in ("copy", "zeros"):
            raise ValueError("model.second_window must be copy or zeros")
        self.second_window = second_window
        self.input_upsample = int(input_upsample or 1)   # 2 = image enlarged x2, outputs averaged back
        if pretrained:
            try:
                self.net = smp.from_pretrained(hf_id)
            except Exception:  # noqa: BLE001 - the repository has moved between organisations
                hf_id = HF_ID_ALT
                self.net = smp.from_pretrained(hf_id)
            log.info("loaded pretrained Fields of the Planet weights (%s)", hf_id)
        else:
            self.net = smp.Unet(encoder, encoder_weights=None, in_channels=8, classes=3)
        # the EfficientNet encoder's batch norms track statistics at 1% per step: far too slow to adapt to new
        # scenes during a short fine-tune. 10% (PyTorch's default) adapts within an epoch.
        for mod in self.net.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.momentum = 0.1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        x = x[:, self.band_idx]
        x = torch.cat([x, x if self.second_window == "copy" else torch.zeros_like(x)], 1)
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect" if (ph < h and pw < w) else "replicate")
        s = self.input_upsample
        if s > 1:
            x = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        z = self.net(x).float()                       # (B, 3, H, W): background, interior, boundary
        if s > 1:   # average class probabilities back to the input grid (stays a valid distribution)
            z = torch.log(F.avg_pool2d(z.softmax(1), s).clamp_min(1e-7))
        z = z[..., :h, :w]
        farm = torch.logsumexp(z[:, 1:3], 1, keepdim=True) - z[:, 0:1]        # logit of P(interior or boundary)
        boundary = z[:, 2:3] - torch.logsumexp(z[:, 0:2], 1, keepdim=True)    # logit of P(boundary)
        return farm, boundary


def save_pretrained_checkpoint(cfg, out_path: Path) -> Path:
    """The released FTP model, untouched, as a checkpoint of this project: lets it be scored and used for mapping
    like any trained model ("zero-shot" baseline). Needs ``metadata/normalization.json`` (for the band list)."""
    from ..data.dataset import load_norm_stats
    from ..utils.helpers import to_plain

    stats = load_norm_stats(cfg, mode="scale")
    n = len(stats["bands"])
    idx = bgrn_indices(n, stats.get("band_names"))
    model = FTPUnet(n, idx, pretrained=True, second_window=cfg.model.get("second_window", "copy"),
                    input_upsample=int(cfg.model.get("input_upsample", 1) or 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": -1, "metrics": {},
                "model_kwargs": {"arch": "ftp_unet", "in_channels": n, "band_idx": idx, "pretrained": False,
                                 "second_window": model.second_window, "input_upsample": model.input_upsample},
                "normalization": stats, "config": to_plain(cfg), "bands": stats["bands"],
                "gsd_m": stats.get("gsd_m")}, out_path)
    return out_path
