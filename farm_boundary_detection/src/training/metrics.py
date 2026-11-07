"""Metrics for the two heads (pixel level) and for individual farms (object level). Only valid pixels count.

Interior head: IoU, Dice, precision, recall.
Boundary head: boundary IoU (strict pixel overlap), plus precision / recall / F1 with a distance tolerance:
a predicted boundary pixel is correct if a true boundary pixel lies within ``tolerance`` px, and a true one is
found if a predicted one lies within ``tolerance`` px. With 1-pixel targets a one-pixel shift is not an error
worth counting, so the tolerant F1 is the headline boundary metric.
Object level (``instance_scores``): a true farm is found if exactly one predicted region overlaps it with
IoU >= threshold (one-to-one, best first); merged neighbours and split farms are counted separately.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * r + 1, stride=1, padding=r)


class SegmentationMetrics:
    """Accumulates counts over batches; ``compute()`` gives the scores. Tensors (B, 1, H, W)."""

    def __init__(self, threshold: float = 0.5, tolerance: int = 1):
        self.thr, self.tol = threshold, int(tolerance)
        self.c = {k: 0.0 for k in ("i_tp", "i_fp", "i_fn", "b_tp", "b_fp", "b_fn", "bt_pred_ok", "bt_pred",
                                   "bt_gt_ok", "bt_gt")}

    @torch.no_grad()
    def update(self, interior_logits: torch.Tensor, boundary_logits: torch.Tensor, batch: dict) -> None:
        v = batch["valid_mask"] > 0.5
        pi = (torch.sigmoid(interior_logits.float()) > self.thr) & v
        gi = (batch["interior_mask"] > 0.5) & v
        pb = (torch.sigmoid(boundary_logits.float()) > self.thr) & v
        gb = (batch["boundary_mask"] > 0.5) & v
        c = self.c
        c["i_tp"] += float((pi & gi).sum())
        c["i_fp"] += float((pi & ~gi).sum())
        c["i_fn"] += float((~pi & gi).sum())
        c["b_tp"] += float((pb & gb).sum())
        c["b_fp"] += float((pb & ~gb).sum())
        c["b_fn"] += float((~pb & gb).sum())
        gb_d = _dilate(gb.float(), self.tol) > 0.5
        pb_d = _dilate(pb.float(), self.tol) > 0.5
        c["bt_pred_ok"] += float((pb & gb_d).sum())
        c["bt_pred"] += float(pb.sum())
        c["bt_gt_ok"] += float((gb & pb_d).sum())
        c["bt_gt"] += float(gb.sum())

    def compute(self) -> dict:
        c = self.c

        def div(a, b):
            return a / b if b > 0 else 0.0

        ip, ir = div(c["i_tp"], c["i_tp"] + c["i_fp"]), div(c["i_tp"], c["i_tp"] + c["i_fn"])
        bp, br = div(c["bt_pred_ok"], c["bt_pred"]), div(c["bt_gt_ok"], c["bt_gt"])
        bsp, bsr = div(c["b_tp"], c["b_tp"] + c["b_fp"]), div(c["b_tp"], c["b_tp"] + c["b_fn"])
        out = {
            "interior_iou": div(c["i_tp"], c["i_tp"] + c["i_fp"] + c["i_fn"]),
            "interior_dice": div(2 * c["i_tp"], 2 * c["i_tp"] + c["i_fp"] + c["i_fn"]),
            "interior_precision": ip, "interior_recall": ir,
            "boundary_iou": div(c["b_tp"], c["b_tp"] + c["b_fp"] + c["b_fn"]),
            "boundary_precision": bp, "boundary_recall": br, "boundary_f1": div(2 * bp * br, bp + br),
            "boundary_f1_strict": div(2 * bsp * bsr, bsp + bsr),
            "boundary_tolerance_px": self.tol,
        }
        out["val_score"] = 0.5 * (out["interior_iou"] + out["boundary_f1"])
        return out


def instance_scores(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, iou_threshold: float = 0.5,
                    min_valid_share: float = 0.5) -> dict:
    """Object-level matching of instance label images (0 = none).

    Only true farms with at least ``min_valid_share`` of their pixels valid are scored, and only predicted
    regions that lie mostly in valid pixels count (a "false positive" in unlabelled land may be a real farm).
    Returns counts; combine several images with ``combine_instance_scores``.
    """
    valid = valid.astype(bool)
    g = gt.astype(np.int64).ravel()
    p = pred.astype(np.int64).ravel()
    v = valid.ravel()
    g_ids, g_total = np.unique(g[g > 0], return_counts=True)
    g_valid = dict(zip(*np.unique(g[(g > 0) & v], return_counts=True)))
    keep_g = {i for i, n in zip(g_ids, g_total) if g_valid.get(i, 0) >= min_valid_share * n}
    p_ids, p_total = np.unique(p[p > 0], return_counts=True)
    p_valid = dict(zip(*np.unique(p[(p > 0) & v], return_counts=True)))
    keep_p = {i for i, n in zip(p_ids, p_total) if p_valid.get(i, 0) >= min_valid_share * n}
    area_g = dict(zip(g_ids, g_total))
    area_p = dict(zip(p_ids, p_total))
    both = (g > 0) & (p > 0)
    pairs, inter = np.unique(np.stack([g[both], p[both]]), axis=1, return_counts=True) if both.any() else \
        (np.zeros((2, 0), np.int64), np.zeros(0, np.int64))
    rows = []
    for (gi, pi), n in zip(pairs.T, inter):
        if gi in keep_g:
            rows.append((n / (area_g[gi] + area_p[pi] - n), int(gi), int(pi), n))
    rows.sort(reverse=True)
    used_g, used_p, matched_iou = set(), set(), []
    for iou, gi, pi, _ in rows:
        if iou >= iou_threshold and gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matched_iou.append(iou)
    # merged: one prediction covering >= 50% of two or more true farms; split: one true farm >= 2 predictions
    # each covering >= 20% of it
    cover_g: dict[int, int] = {}
    cover_p: dict[int, int] = {}
    for iou, gi, pi, n in rows:
        if n >= 0.5 * area_g[gi]:
            cover_p[pi] = cover_p.get(pi, 0) + 1
        if n >= 0.2 * area_g[gi]:
            cover_g[gi] = cover_g.get(gi, 0) + 1
    return {"gt": len(keep_g), "pred": len(keep_p), "matched": len(used_g),
            "merged_predictions": sum(1 for k, c in cover_p.items() if c >= 2),
            "split_farms": sum(1 for k, c in cover_g.items() if c >= 2),
            "sum_matched_iou": float(np.sum(matched_iou))}


def combine_instance_scores(items: list[dict]) -> dict:
    t = {k: sum(i[k] for i in items) for k in ("gt", "pred", "matched", "merged_predictions", "split_farms",
                                                "sum_matched_iou")}
    prec = t["matched"] / t["pred"] if t["pred"] else 0.0
    rec = t["matched"] / t["gt"] if t["gt"] else 0.0
    return {"farms_true": t["gt"], "farms_predicted": t["pred"], "farms_matched": t["matched"],
            "object_precision": prec, "object_recall": rec,
            "object_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "mean_matched_iou": t["sum_matched_iou"] / t["matched"] if t["matched"] else 0.0,
            "merged_predictions": t["merged_predictions"], "split_farms": t["split_farms"]}
