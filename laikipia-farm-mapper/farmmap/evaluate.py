"""Honest evaluation on geographically held-out *test* blocks the model never saw.

Reports (a) pixel IoU per class, (b) boundary precision/recall within a tolerance, and
(c) polygon-level match against your manual polygons (IoU >= 0.5).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy import ndimage as ndi
from shapely.geometry import box
from shapely.ops import unary_union

from .data import block_window, load_or_create_split, split_blocks
from .labels import IGNORE
from .polygonize import polygonize_probs
from .predict import predict_raster
from .reference import load_reference
from .train import confusion, iou_from_confusion


def _match(pred, ref, thr=0.5):
    """Greedy one-to-one matching by IoU. Returns (n_matched, mean IoU of reference farms)."""
    if len(pred) == 0 or len(ref) == 0:
        return 0, 0.0
    tree = pred.sindex
    used, ious, matched = set(), [], 0
    for g in ref.geometry:
        best, best_j = 0.0, None
        for j in tree.query(g, predicate="intersects"):
            if j in used:
                continue
            p = pred.geometry.iloc[j]
            iou = g.intersection(p).area / g.union(p).area
            if iou > best:
                best, best_j = iou, j
        ious.append(best)
        if best >= thr and best_j is not None:
            used.add(best_j)
            matched += 1
    return matched, float(np.mean(ious))


def evaluate(cfg, scenes=None, checkpoint=None, which="test", progress=print):
    """Score a checkpoint on the held-out blocks of every scene. Pass the same `which` for every
    model you want to compare."""
    from .pipeline import list_scenes

    scenes = scenes or list_scenes(cfg)
    split = load_or_create_split(cfg, scenes, progress)
    allb = split_blocks(split, which)
    if not allb:
        raise RuntimeError(f"No '{which}' blocks. The labelled area is too small to hold any out.")
    cm = np.zeros((3, 3), np.int64)
    tol = 2  # px
    tp_p = n_p = tp_r = n_r = matched = n_pred = n_ref = 0
    ious = []
    for scene in scenes:
        blocks = [b for b in allb if b["scene"] == scene["name"]]
        if not blocks:
            continue
        regions = [block_window(b) for b in blocks]
        with tempfile.TemporaryDirectory() as tmp:
            probs_path = predict_raster(cfg, scene["imagery"], checkpoint, Path(tmp) / "probs.tif", regions,
                                        lambda *_: None)
            with rasterio.open(probs_path) as ps, rasterio.open(scene["labels"]) as ls:
                crs, tr = ps.crs, ps.transform
                for w in regions:
                    pred = ps.read(window=w).argmax(0).astype(np.int64)
                    lab = ls.read(1, window=w).astype(np.int64)
                    cm += confusion(pred, lab)
                    keep = lab != IGNORE
                    pb, gb = (pred == 2) & keep, (lab == 2)
                    if gb.any() and pb.any():
                        near_g = ndi.distance_transform_edt(~gb) <= tol
                        near_p = ndi.distance_transform_edt(~pb) <= tol
                        tp_p += int((pb & near_g).sum()); n_p += int(pb.sum())
                        tp_r += int((gb & near_p).sum()); n_r += int(gb.sum())

            pred_poly = polygonize_probs(cfg, probs_path, progress=lambda *_: None)
            region_geom = unary_union([box(*rasterio.windows.bounds(w, tr)) for w in regions])
            ref = load_reference(cfg, crs=crs)
            ref = ref.assign(geometry=ref.intersection(region_geom))
            ref = ref[~ref.geometry.is_empty & (ref.area / 1e4 >= cfg.polygonize.min_area_ha)].reset_index(drop=True)
            pred_poly = gpd.GeoDataFrame(geometry=pred_poly.geometry.intersection(region_geom), crs=crs)
            pred_poly = pred_poly[~pred_poly.geometry.is_empty].reset_index(drop=True)
            m, mean_iou = _match(pred_poly, ref)
            matched += m; n_pred += len(pred_poly); n_ref += len(ref)
            ious += [mean_iou] * len(ref)

    iou = iou_from_confusion(cm)
    bp = tp_p / n_p if n_p else 0.0
    br = tp_r / n_r if n_r else 0.0
    fp = matched / n_pred if n_pred else 0.0
    fr = matched / n_ref if n_ref else 0.0
    return {
        "split": which, "n_blocks": len(allb), "checkpoint": str(checkpoint or cfg.path(cfg.model.checkpoint)),
        "iou_background": float(iou[0]), "iou_interior": float(iou[1]), "iou_boundary": float(iou[2]),
        "boundary_precision": bp, "boundary_recall": br,
        "boundary_f1": 2 * bp * br / (bp + br) if bp + br else 0.0,
        "boundary_tolerance_m": tol * cfg.project.resolution_m,
        "n_reference_farms": n_ref, "n_predicted_farms": n_pred, "matched_iou50": matched,
        "farm_precision": fp, "farm_recall": fr, "farm_f1": 2 * fp * fr / (fp + fr) if fp + fr else 0.0,
        "mean_best_iou_per_reference": float(np.mean(ious)) if ious else 0.0,
    }
