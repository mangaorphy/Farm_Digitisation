"""Task 5: multi-task ResNet U-Net with an interior (extent) head and a boundary head.

Inspired by ResUNet-a (Diakogiannis et al. 2020) - it borrows these ideas, it is NOT a reproduction:

* residual convolution blocks in the decoder;
* pyramid scene pooling (PSP) at the bridge for multi-scale context;
* conditioned multi-task heads: the boundary (and, optionally, the distance) is predicted first and its
  probability is fed to the interior head, as ResUNet-a conditions extent on boundary and distance;
* optional distance head (``distance_head``): for every farm pixel, its distance to the farm's own outline,
  normalised to 1 at the farm's centre. Its maxima are clean seeds for separating touching farms.

Not taken from ResUNet-a: its atrous residual encoder (a pretrained torchvision ResNet is used instead, which
matters with a few thousand labelled farms), the colour-reconstruction head and the Tanimoto loss.

One addition for 3 m imagery: the ResNet stem works at 1/2 and then 1/4 resolution, which blurs 1-2 px
boundaries. A light convolutional branch on the full-resolution input is concatenated in the last decoder
stage (``full_resolution_branch``).

``forward`` returns ``(interior_logits, boundary_logits)``, each (B, 1, H, W), plus ``distance_logits`` as a third
element when the distance head is on. No sigmoid: apply it at inference.
Any H, W work (inputs are padded to a multiple of 32 internally and outputs cropped back).
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

log = logging.getLogger(__name__)

ENCODERS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}


def adapt_first_conv(conv: nn.Conv2d, in_channels: int, rgb_indices: list[int] | None) -> nn.Conv2d:
    """A copy of ``conv`` that takes ``in_channels`` bands.

    Pretrained RGB filters go to the input channels that ARE red, green and blue (``rgb_indices``, 0-based);
    every other band (NIR, red edge, coastal...) starts from the mean RGB filter. Weights are scaled by
    3 / in_channels so the first layer's activations keep the magnitude the rest of the encoder expects.
    """
    w = conv.weight.data  # (64, 3, k, k), ImageNet order R, G, B
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    mean = w.mean(dim=1, keepdim=True)
    nw = mean.repeat(1, in_channels, 1, 1).clone()
    if rgb_indices:
        for src, dst in enumerate(rgb_indices[:3]):
            if 0 <= dst < in_channels:
                nw[:, dst] = w[:, src]
    new.weight.data = nw * (3.0 / in_channels)
    return new


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResidualBlock(nn.Module):
    """conv-BN-ReLU-conv-BN + (1x1 projected) shortcut, ReLU."""

    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.short = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False),
                                                                     nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.short(x))


class PSPPooling(nn.Module):
    """Average-pool to 1, 2, 4 and 8 bins, project, upsample and fuse with the input (multi-scale context)."""

    def __init__(self, cin: int, cout: int, bins=(1, 2, 4, 8)):
        super().__init__()
        branch = max(cin // len(bins), 16)
        # GroupNorm: batch norm fails on the 1x1 bin with batch size 1
        self.stages = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(cin, branch, 1, bias=False),
                                                  nn.GroupNorm(1, branch), nn.ReLU(inplace=True)) for b in bins)
        self.fuse = ConvBNReLU(cin + branch * len(bins), cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [x] + [F.interpolate(s(x), size=(h, w), mode="bilinear", align_corners=False) for s in self.stages]
        return self.fuse(torch.cat(feats, 1))


class DecoderStage(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int, dropout: float):
        super().__init__()
        self.block = ResidualBlock(cin + cskip, cout, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        size = skip.shape[-2:] if skip is not None else tuple(s * 2 for s in x.shape[-2:])
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.block(x)


class Head(nn.Module):
    def __init__(self, cin: int, mid: int):
        super().__init__()
        self.body = nn.Sequential(ConvBNReLU(cin, mid), nn.Conv2d(mid, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MultiTaskResUNet(nn.Module):
    def __init__(self, in_channels: int, encoder: str = "resnet34", pretrained: bool = True,
                 decoder_channels=(256, 128, 64, 32), full_resolution_branch: bool = True, psp_pooling: bool = True,
                 condition_interior_on_boundary: bool = True, dropout: float = 0.0,
                 rgb_indices: list[int] | None = None, distance_head: bool = False):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {sorted(ENCODERS)}, got {encoder!r}")
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels needs 4 values (1/16, 1/8, 1/4, 1/2 resolution)")
        self.in_channels = in_channels
        self.condition = condition_interior_on_boundary
        self.has_distance = bool(distance_head)
        weights = "DEFAULT" if pretrained else None
        try:
            net = getattr(torchvision.models, encoder)(weights=weights)
        except Exception as e:  # noqa: BLE001 - no internet / no cached weights
            if not pretrained:
                raise
            log.warning("could not load ImageNet weights for %s (%s): training from scratch", encoder, e)
            net = getattr(torchvision.models, encoder)(weights=None)
        if in_channels != 3 or rgb_indices not in (None, [0, 1, 2]):
            net.conv1 = adapt_first_conv(net.conv1, in_channels, rgb_indices)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)          # 1/2, 64
        self.pool = net.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = net.layer1, net.layer2, net.layer3, net.layer4
        mult = 4 if encoder in ("resnet50", "resnet101") else 1
        skips = [64 * mult, 128 * mult, 256 * mult]                       # 1/4, 1/8, 1/16
        c4 = ENCODERS[encoder]                                            # 1/32
        d1, d2, d3, d4 = decoder_channels
        self.bridge = PSPPooling(c4, d1) if psp_pooling else ConvBNReLU(c4, d1, 1)
        self.dec16 = DecoderStage(d1, skips[2], d1, dropout)
        self.dec8 = DecoderStage(d1, skips[1], d2, dropout)
        self.dec4 = DecoderStage(d2, skips[0], d3, dropout)
        self.dec2 = DecoderStage(d3, 64, d4, 0.0)
        self.full_res = nn.Sequential(ConvBNReLU(in_channels, 32), ConvBNReLU(32, 32)) if full_resolution_branch else None
        self.dec1 = ResidualBlock(d4 + (32 if full_resolution_branch else 0), d4)
        self.boundary_head = Head(d4, d4)
        self.distance_head = Head(d4, d4) if self.has_distance else None
        n_cond = (1 + int(self.has_distance)) if self.condition else 0
        self.interior_head = Head(d4 + n_cond, d4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h, w = x.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:  # reflect-pad to a multiple of 32, crop the outputs back
            mode = "reflect" if (ph < h and pw < w) else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        s2 = self.stem(x)
        s4 = self.enc1(self.pool(s2))
        s8 = self.enc2(s4)
        s16 = self.enc3(s8)
        s32 = self.enc4(s16)
        d = self.bridge(s32)
        d = self.dec16(d, s16)
        d = self.dec8(d, s8)
        d = self.dec4(d, s4)
        d = self.dec2(d, s2)
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.full_res is not None:
            d = torch.cat([d, self.full_res(x)], 1)
        d = self.dec1(d)
        boundary_logits = self.boundary_head(d)
        distance_logits = self.distance_head(d) if self.distance_head is not None else None
        if self.condition:
            cond = [torch.sigmoid(boundary_logits)] + ([torch.sigmoid(distance_logits)] if distance_logits is not None else [])
            inp = torch.cat([d, *cond], 1)
        else:
            inp = d
        interior_logits = self.interior_head(inp)
        out = (interior_logits[..., :h, :w], boundary_logits[..., :h, :w])
        if distance_logits is not None:
            out = out + (distance_logits[..., :h, :w],)
        return out


def build_model(cfg, in_channels: int, rgb_indices: list[int] | None = None,
                pretrained: bool | None = None) -> MultiTaskResUNet:
    mc = cfg.model
    return MultiTaskResUNet(
        in_channels=in_channels, encoder=mc.encoder, pretrained=mc.pretrained if pretrained is None else pretrained,
        decoder_channels=tuple(mc.decoder_channels), full_resolution_branch=mc.full_resolution_branch,
        psp_pooling=mc.psp_pooling, condition_interior_on_boundary=mc.condition_interior_on_boundary,
        dropout=float(mc.get("dropout", 0.0)), rgb_indices=rgb_indices, distance_head=bool(mc.get("distance_head", False)))


def model_kwargs(cfg, in_channels: int, rgb: list[int] | None) -> dict:
    """Everything needed to rebuild the network from a checkpoint (no download: pretrained=False)."""
    mc = cfg.model
    return {"in_channels": in_channels, "encoder": mc.encoder, "pretrained": False,
            "decoder_channels": list(mc.decoder_channels), "full_resolution_branch": bool(mc.full_resolution_branch),
            "psp_pooling": bool(mc.psp_pooling), "condition_interior_on_boundary": bool(mc.condition_interior_on_boundary),
            "dropout": float(mc.get("dropout", 0.0)), "rgb_indices": rgb,
            "distance_head": bool(mc.get("distance_head", False))}


def load_checkpoint(path, device="cpu") -> dict:
    """Our own checkpoints hold metadata (dicts, lists); PyTorch >= 2.6 needs weights_only=False for them."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # very old torch
        return torch.load(path, map_location=device)


def load_model(path, device="cpu") -> tuple[MultiTaskResUNet, dict]:
    ck = load_checkpoint(path, device)
    if "model_kwargs" not in ck:
        raise ValueError(f"{path} is not a checkpoint written by this project")
    model = MultiTaskResUNet(**ck["model_kwargs"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
