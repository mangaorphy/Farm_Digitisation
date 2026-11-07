"""Task 6: multi-task loss.

    total = lambda_interior * interior_loss + lambda_boundary * boundary_loss
    interior_loss = a * BCE(weighted by the boundary-aware map) + b * Dice
    boundary_loss = a * BCE(pos_weight) + b * Dice + c * Tversky + d * Focal

Every term is computed only over valid pixels (``valid_mask`` = 1): nodata, cloud, QA-ignored farms and
unlabelled land never contribute. All terms work on logits (sigmoid inside), in float32 even under AMP.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1.0


def masked_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor | None = None,
               pos_weight: float | None = None) -> torch.Tensor:
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype) if pos_weight else None
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    w = valid if weight is None else valid * weight
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def soft_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits) * valid
    t = target * valid
    return 1.0 - (2.0 * (p * t).sum() + EPS) / (p.sum() + t.sum() + EPS)


def tversky(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """alpha weighs false positives, beta false negatives (beta > alpha favours recall of thin boundaries)."""
    p = torch.sigmoid(logits) * valid
    t = target * valid
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    return 1.0 - (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)


def focal(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = torch.exp(-bce)
    return ((1 - p_t) ** gamma * bce * valid).sum() / valid.sum().clamp_min(1.0)


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        lc = cfg.loss
        self.li, self.lb = float(lc.lambda_interior), float(lc.lambda_boundary)
        self.wi = {k: float(v) for k, v in (lc.interior or {}).items()}
        self.wb = {k: float(v) for k, v in (lc.boundary or {}).items()}
        self.pos_weight = float(lc.get("boundary_pos_weight") or 0) or None
        self.alpha, self.beta = float(lc.get("tversky_alpha", 0.3)), float(lc.get("tversky_beta", 0.7))
        self.gamma = float(lc.get("focal_gamma", 2.0))
        self.use_weight_map = bool((lc.get("boundary_aware") or {}).get("w0", 0))

    def forward(self, interior_logits: torch.Tensor, boundary_logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
        il, bl = interior_logits.float(), boundary_logits.float()
        ti, tb, v = batch["interior_mask"], batch["boundary_mask"], batch["valid_mask"]
        wmap = batch.get("weight_map") if self.use_weight_map else None
        parts: dict[str, torch.Tensor] = {}
        interior = il.new_zeros(())
        if self.wi.get("bce"):
            parts["interior_bce"] = masked_bce(il, ti, v, wmap)
            interior = interior + self.wi["bce"] * parts["interior_bce"]
        if self.wi.get("dice"):
            parts["interior_dice"] = soft_dice(il, ti, v)
            interior = interior + self.wi["dice"] * parts["interior_dice"]
        boundary = bl.new_zeros(())
        if self.wb.get("bce"):
            parts["boundary_bce"] = masked_bce(bl, tb, v, None, self.pos_weight)
            boundary = boundary + self.wb["bce"] * parts["boundary_bce"]
        if self.wb.get("dice"):
            parts["boundary_dice"] = soft_dice(bl, tb, v)
            boundary = boundary + self.wb["dice"] * parts["boundary_dice"]
        if self.wb.get("tversky"):
            parts["boundary_tversky"] = tversky(bl, tb, v, self.alpha, self.beta)
            boundary = boundary + self.wb["tversky"] * parts["boundary_tversky"]
        if self.wb.get("focal"):
            parts["boundary_focal"] = focal(bl, tb, v, self.gamma)
            boundary = boundary + self.wb["focal"] * parts["boundary_focal"]
        total = self.li * interior + self.lb * boundary
        parts.update({"interior_loss": interior, "boundary_loss": boundary, "loss": total})
        return total, {k: float(x.detach()) for k, x in parts.items()}
