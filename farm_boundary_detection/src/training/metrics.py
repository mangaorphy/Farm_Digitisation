"""Metrics for the heads (pixel level) and for individual farms (object level). Only valid pixels count.

Interior head: IoU, Dice, precision, recall (+ confusion matrix).
Boundary head: boundary IoU (strict pixel overlap), plus precision / recall / F1 with a distance tolerance:
a predicted boundary pixel is correct if a true boundary pixel lies within ``tolerance`` px, and a true one is
found if a predicted one lies within ``tolerance`` px. With 1-pixel targets a one-pixel shift is not an error
worth counting, so the tolerant F1 is the headline boundary metric.
Object level (``instance_scores``): a true farm is found if a predicted region overlaps it with IoU >= threshold
(one-to-one, best first); merged neighbours and split farms are counted separately; with ``details=True`` the
per-farm records are returned for scores by farm size.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SIZE_BINS_HA = [0, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf]
SIZE_NAMES = ["<0.1 ha", "0.1-0.25 ha", "0.25-0.5 ha", "0.5-1 ha", "1-2 ha", ">2 ha"]


def _dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * r + 1, stride=1, padding=r)


class SegmentationMetrics:
    """Accumulates counts over batches; ``compute()`` gives the scores, ``confusion()`` the 2x2 matrices.
    Tensors (B, 1, H, W); the heads are given as logits."""

    KEYS = ("i_tp", "i_fp", "i_fn", "i_tn", "b_tp", "b_fp", "b_fn", "b_tn", "bt_pred_ok", "bt_pred", "bt_gt_ok",
            "bt_gt")

    def __init__(self, threshold: float = 0.5, tolerance: int = 1):
        self.thr, self.tol = threshold, int(tolerance)
        self.c = {k: 0.0 for k in self.KEYS}

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
        c["i_fn"] += float((~pi & gi & v).sum())
        c["i_tn"] += float((~pi & ~gi & v).sum())
        c["b_tp"] += float((pb & gb).sum())
        c["b_fp"] += float((pb & ~gb).sum())
        c["b_fn"] += float((~pb & gb & v).sum())
        c["b_tn"] += float((~pb & ~gb & v).sum())
        gb_d = _dilate(gb.float(), self.tol) > 0.5
        pb_d = _dilate(pb.float(), self.tol) > 0.5
        c["bt_pred_ok"] += float((pb & gb_d).sum())
        c["bt_pred"] += float(pb.sum())
        c["bt_gt_ok"] += float((gb & pb_d).sum())
        c["bt_gt"] += float(gb.sum())

    def confusion(self) -> dict:
        """{'interior': [[TN, FP], [FN, TP]], 'boundary': [[TN, FP], [FN, TP]]} (rows = truth, cols = prediction)."""
        c = self.c
        return {"interior": [[c["i_tn"], c["i_fp"]], [c["i_fn"], c["i_tp"]]],
                "boundary": [[c["b_tn"], c["b_fp"]], [c["b_fn"], c["b_tp"]]]}

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
                    min_valid_share: float = 0.5, details: bool = False) -> dict:
    """Object-level matching of instance label images (0 = none).

    Only true farms with at least ``min_valid_share`` of their pixels valid are scored, and only predicted
    regions that lie mostly in valid pixels count (a "false positive" in unlabelled land may be a real farm).
    With ``details``, adds ``gt_records`` (id, pixels, matched, best_iou) and ``pred_records`` (pixels, matched).
    Combine several images with ``combine_instance_scores``.
    """
    valid = valid.astype(bool)
    g = gt.astype(np.int64).ravel()
    p = pred.astype(np.int64).ravel()
    v = valid.ravel()
    g_ids, g_total = np.unique(g[g > 0], return_counts=True)
    g_valid = dict(zip(*np.unique(g[(g > 0) & v], return_counts=True)))
    keep_g = {int(i) for i, n in zip(g_ids, g_total) if g_valid.get(i, 0) >= min_valid_share * n}
    p_ids, p_total = np.unique(p[p > 0], return_counts=True)
    p_valid = dict(zip(*np.unique(p[(p > 0) & v], return_counts=True)))
    keep_p = {int(i) for i, n in zip(p_ids, p_total) if p_valid.get(i, 0) >= min_valid_share * n}
    area_g = {int(k): int(n) for k, n in zip(g_ids, g_total)}
    area_p = {int(k): int(n) for k, n in zip(p_ids, p_total)}
    both = (g > 0) & (p > 0)
    if both.any():
        pairs, inter = np.unique(np.stack([g[both], p[both]]), axis=1, return_counts=True)
    else:
        pairs, inter = np.zeros((2, 0), np.int64), np.zeros(0, np.int64)
    rows = []
    for (gi, pi), n in zip(pairs.T, inter):
        gi, pi = int(gi), int(pi)
        if gi in keep_g:
            rows.append((n / (area_g[gi] + area_p[pi] - n), gi, pi, int(n)))
    rows.sort(reverse=True)
    used_g, used_p, matched_iou = set(), set(), []
    best_iou: dict[int, float] = {}
    for iou, gi, pi, _ in rows:
        best_iou[gi] = max(best_iou.get(gi, 0.0), iou)
        if iou >= iou_threshold and gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matched_iou.append(iou)
    # merged: one prediction covering >= 50% of two or more true farms; split: one true farm with >= 2
    # predictions each covering >= 20% of it
    cover_g: dict[int, int] = {}
    cover_p: dict[int, int] = {}
    for iou, gi, pi, n in rows:
        if n >= 0.5 * area_g[gi]:
            cover_p[pi] = cover_p.get(pi, 0) + 1
        if n >= 0.2 * area_g[gi]:
            cover_g[gi] = cover_g.get(gi, 0) + 1
    out = {"gt": len(keep_g), "pred": len(keep_p), "matched": len(used_g),
           "merged_predictions": sum(1 for c in cover_p.values() if c >= 2),
           "split_farms": sum(1 for c in cover_g.values() if c >= 2),
           "sum_matched_iou": float(np.sum(matched_iou))}
    if details:
        out["gt_records"] = [(gi, area_g[gi], gi in used_g, round(best_iou.get(gi, 0.0), 4)) for gi in keep_g]
        out["pred_records"] = [(area_p[pi], pi in used_p) for pi in keep_p]
    return out


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


def scores_by_size(items: list[dict], gsd: float, true_area_m2: dict[int, float] | None = None) -> "pd.DataFrame":
    """Recall (true farms) and precision (predicted farms) per farm-size class, from ``details=True`` records.

    True farm sizes come from ``true_area_m2`` (polygon areas by instance id) when given, else from their
    pixels inside the patch (an underestimate for farms cut by the patch edge)."""
    import pandas as pd

    px_m2 = gsd * gsd
    g = [(true_area_m2.get(i, n * px_m2) if true_area_m2 else n * px_m2, m, iou)
         for it in items for i, n, m, iou in it.get("gt_records", [])]
    p = [(n * px_m2, m) for it in items for n, m in it.get("pred_records", [])]
    gdf = pd.DataFrame(g, columns=["area_m2", "matched", "best_iou"])
    pdf = pd.DataFrame(p, columns=["area_m2", "matched"])
    gdf["size"] = pd.cut(gdf.area_m2 / 1e4, SIZE_BINS_HA, labels=SIZE_NAMES, right=False)
    pdf["size"] = pd.cut(pdf.area_m2 / 1e4, SIZE_BINS_HA, labels=SIZE_NAMES, right=False)
    rows = []
    for name in SIZE_NAMES:
        gs, ps = gdf[gdf["size"] == name], pdf[pdf["size"] == name]
        rows.append({"size": name, "true_farms": len(gs), "recall": gs.matched.mean() if len(gs) else np.nan,
                     "median_best_iou": gs.best_iou.median() if len(gs) else np.nan,
                     "predicted_farms": len(ps), "precision": ps.matched.mean() if len(ps) else np.nan})
    return pd.DataFrame(rows).set_index("size")
