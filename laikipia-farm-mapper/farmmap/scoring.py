"""One scorer for every method (U-Net, Delineate Anything, any polygon file).

Reference farms are matched one-to-one with predicted polygons by IoU. A farm counts as found
when IoU >= 0.5. Only polygons whose representative point lies in the evaluation region AND
within `reference.coverage_buffer_m` of a reference farm are counted, so predicted farms in
areas you have not digitised are not punished as false positives.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree

SIZE_BINS_HA = [(0.0, 0.25), (0.25, 1.0), (1.0, 4.0), (4.0, float("inf"))]


def match_report(pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, thr: float = 0.5):
    """Greedy one-to-one matching, highest IoU first. Returns (best_iou[n_ref], matched[n_ref] bool)."""
    n = len(ref)
    best, matched = np.zeros(n), np.zeros(n, bool)
    if len(pred) == 0 or n == 0:
        return best, matched
    pg = list(pred.geometry)
    tree = STRtree(pg)
    pairs = []
    for i, g in enumerate(ref.geometry):
        for j in tree.query(g):
            inter = g.intersection(pg[j]).area
            if inter > 0:
                pairs.append((inter / (g.area + pg[j].area - inter), i, int(j)))
    pairs.sort(reverse=True)
    used_i, used_j = set(), set()
    for iou, i, j in pairs:
        best[i] = max(best[i], iou)
        if iou >= thr and i not in used_i and j not in used_j:
            used_i.add(i); used_j.add(j); matched[i] = True
    return best, matched


class Scorer:
    """Accumulate matches over several regions/scenes, then summarise."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.best, self.matched, self.area_ha = [], [], []
        self.n_pred = 0

    def add(self, pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, region_geom):
        ref = ref[ref["ref_use"]] if "ref_use" in ref.columns else ref
        cover = ref.geometry.buffer(self.cfg.reference.coverage_buffer_m).union_all().intersection(region_geom)
        if cover.is_empty:
            return
        r = ref[ref.geometry.centroid.within(region_geom)]
        p = pred[pred.geometry.representative_point().within(cover)] if len(pred) else pred
        best, matched = match_report(p, r)
        self.best += best.tolist(); self.matched += matched.tolist()
        self.area_ha += (r.area / 1e4).tolist()
        self.n_pred += len(p)

    def summary(self) -> dict:
        best, matched, area = np.array(self.best), np.array(self.matched, bool), np.array(self.area_ha)
        n_ref, m = len(best), int(matched.sum())
        prec = m / self.n_pred if self.n_pred else 0.0
        rec = m / n_ref if n_ref else 0.0
        out = {"n_reference": n_ref, "n_predicted": self.n_pred, "matched_iou50": m,
               "precision": round(prec, 3), "recall": round(rec, 3),
               "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
               "mean_best_iou": round(float(best.mean()), 3) if n_ref else 0.0}
        for lo, hi in SIZE_BINS_HA:
            sel = (area >= lo) & (area < hi)
            label = f"recall_{lo:g}-{hi:g}ha" if np.isfinite(hi) else f"recall_{lo:g}+ha"
            out[label] = round(float(matched[sel].mean()), 3) if sel.any() else None
            out[label.replace("recall", "n")] = int(sel.sum())
        return out
