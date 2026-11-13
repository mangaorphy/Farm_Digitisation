"""Task 6: multi-task loss.

    total = lambda_interior * interior_loss + lambda_boundary * boundary_loss [+ lambda_distance * distance_loss]
    interior_loss = a * BCE(weighted by the boundary-aware map) + b * Dice
    boundary_loss = a * BCE(pos_weight) + b * Dice + c * Tversky + d * Focal
    distance_loss = BCE with soft targets + soft Dice (only with the distance head; targets in [0, 1])

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


def logcosh_dice(class_logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                 class_weights: torch.Tensor) -> torch.Tensor:
    """Class-weighted log-cosh soft Dice over exclusive classes (PRUE, CVPR 2026); target holds class indices."""
    p = torch.softmax(class_logits, 1) * valid
    t = F.one_hot(target, class_logits.shape[1]).permute(0, 3, 1, 2).float() * valid
    dims = (0, 2, 3)
    dice = 1.0 - (2.0 * (p * t).sum(dims) + EPS) / (p.sum(dims) + t.sum(dims) + EPS)
    return (class_weights * torch.log(torch.cosh(dice))).sum() / class_weights.sum()


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
        sep = lc.get("separation") or {}
        self.use_weight_map = bool((lc.get("boundary_aware") or {}).get("w0", 0)) or \
            bool(sep.get("enabled", False) and sep.get("weight", 0))
        self.ld = float(lc.get("lambda_distance", 1.0))
        self.head3 = str(cfg.model.get("head", "two")) == "softmax3"
        s3 = lc.get("softmax3") or {}
        self.w3 = {"ce": float(s3.get("ce", 1.0)), "logcosh_dice": float(s3.get("logcosh_dice", 1.0))}
        self.register_buffer("class_w", torch.tensor([float(x) for x in s3.get("class_weights", [0.05, 0.2, 0.75])]))

    def forward(self, interior_logits: torch.Tensor, boundary_logits: torch.Tensor, batch: dict,
                extra: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """``extra``: the distance logits (``model.distance_head``) or the 3-class logits (``model.head: softmax3``)."""
        if self.head3:
            return self._softmax3(extra, batch)
        distance_logits = extra
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
        if distance_logits is not None and "distance_mask" in batch:
            dl, td = distance_logits.float(), batch["distance_mask"]
            distance = masked_bce(dl, td, v) + soft_dice(dl, td, v)
            parts["distance_loss"] = distance
            total = total + self.ld * distance
        parts.update({"interior_loss": interior, "boundary_loss": boundary, "loss": total})
        return total, {k: float(x.detach()) for k, x in parts.items()}

    def _softmax3(self, z: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
        """Exclusive classes: 2 = boundary (outlines, and the strips between farms with ``loss.separation``),
        1 = rest of a farm, 0 = background. Pixel weights (boundary-aware / separation) apply to the CE term."""
        if z is None:
            raise ValueError("model.head: softmax3 but the network returned no class logits")
        z = z.float()
        v = batch["valid_mask"]
        b = batch["boundary_mask"][:, 0] > 0.5
        target = torch.where(b, 2, (batch["interior_mask"][:, 0] > 0.5).long())
        w = batch.get("weight_map") if self.use_weight_map else None
        ce = F.cross_entropy(z, target, reduction="none")
        pw = v[:, 0] * (w[:, 0] if w is not None else 1.0) * self.class_w.to(z.device)[target]   # class x pixel weight
        parts = {"class_ce": (ce * pw).sum() / pw.sum().clamp_min(1e-6),
                 "class_logcosh_dice": logcosh_dice(z, target, v, self.class_w.to(z.device))}
        total = self.w3["ce"] * parts["class_ce"] + self.w3["logcosh_dice"] * parts["class_logcosh_dice"]
        parts["loss"] = total
        return total, {k: float(x.detach()) for k, x in parts.items()}
