"""Score predicted farm polygons against your parcels, only inside held-out (test) map blocks.

A predicted farm matches a parcel when IoU >= evaluate.iou_match (one-to-one, best IoU first).
Only predictions inside the labelled context count, because your file does not cover every farm:
a "false positive" far from any mapped parcel may simply be an unmapped farm.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .labels import load_labels, matches
from .tiles import blocks_gdf

SIZE_BINS = [0, 0.5, 2, 10, np.inf]
SIZE_NAMES = ["<0.5 ha", "0.5-2 ha", "2-10 ha", ">10 ha"]


def match(pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, thr: float) -> pd.DataFrame:
    """Rows (ref_idx, pred_idx, iou) for every overlapping pair, plus a `matched` flag (greedy one-to-one)."""
    if pred.empty or ref.empty:
        return pd.DataFrame(columns=["ref", "pred", "iou", "matched"])
    ri, pi = pred.sindex.query(ref.geometry, predicate="intersects")
    a, b = ref.geometry.values[ri], pred.geometry.values[pi]
    inter = a.intersection(b).area
    iou = inter / (a.area + b.area - inter)
    pairs = pd.DataFrame({"ref": ref.index.values[ri], "pred": pred.index.values[pi], "iou": iou,
                          "ref_share": inter / a.area, "pred_share": inter / b.area})
    pairs = pairs.sort_values("iou", ascending=False)
    used_r, used_p, flag = set(), set(), []
    for r, p, v in zip(pairs.ref, pairs.pred, pairs.iou):
        ok = v >= thr and r not in used_r and p not in used_p
        if ok:
            used_r.add(r)
            used_p.add(p)
        flag.append(ok)
    pairs["matched"] = flag
    return pairs


def evaluate(cfg, pred_path: Path, split: str = "test", out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or cfg.paths.outputs)
    pred = gpd.read_file(pred_path).to_crs(cfg.aoi.crs)
    labels = load_labels(cfg)
    blocks = blocks_gdf(labels.total_bounds, cfg)
    region = blocks[blocks.split == split].union_all() if split != "all" else blocks.union_all()
    context = labels.geometry.buffer(float(cfg.labels.context_m)).union_all().intersection(region)
    pred_extent = pred.union_all().convex_hull if len(pred) else None

    trusted = matches(labels, cfg.evaluate.get("reference")) if cfg.evaluate.get("reference") else labels.train
    ref = labels[trusted & labels.representative_point().within(region)]
    if pred_extent is not None:  # only parcels inside the area that was actually predicted
        ref = ref[ref.within(pred_extent.buffer(1))]
    p = pred[pred.representative_point().within(context)].reset_index(drop=True)
    ref = ref.reset_index(drop=True)
    # predictions that mostly cover a parcel outside the reference (QA-flagged, scheme layer...) are neither
    # right nor wrong
    ign = labels[~np.asarray(trusted)]
    if len(ign) and len(p):
        ii, pp = ign.sindex.query(p.geometry, predicate="intersects")
        cover = pd.Series(p.geometry.values[ii].intersection(ign.geometry.values[pp]).area / p.area.values[ii]) \
            .groupby(ii).sum()
        p = p.drop(index=cover[cover > 0.5].index).reset_index(drop=True)

    pairs = match(p, ref, float(cfg.evaluate.iou_match))
    tp = int(pairs.matched.sum()) if len(pairs) else 0
    prec = tp / max(len(p), 1)
    rec = tp / max(len(ref), 1)
    best_iou = pairs.groupby("ref").iou.max().reindex(ref.index, fill_value=0) if len(pairs) else \
        pd.Series(0.0, index=ref.index)
    # over-segmentation: a parcel split into >1 predictions each covering >=20% of it; under: one prediction
    # covering >=20% of several parcels (merged neighbours)
    over = int((pairs[pairs.ref_share >= 0.2].groupby("ref").size() > 1).sum()) if len(pairs) else 0
    under = int((pairs[pairs.pred_share >= 0.2].groupby("pred").size() > 1).sum()) if len(pairs) else 0

    ref_area = ref.area / 1e4
    ref_bin = pd.cut(ref_area, SIZE_BINS, labels=SIZE_NAMES)
    matched_refs = set(pairs.loc[pairs.matched, "ref"]) if len(pairs) else set()
    by_size = {str(k): {"parcels": int((ref_bin == k).sum()),
                        "recall": round(float(np.mean([i in matched_refs for i in ref.index[ref_bin == k]]))
                                        if (ref_bin == k).any() else float("nan"), 3)}
               for k in SIZE_NAMES}
    result = {"split": split, "predictions": len(p), "parcels": len(ref), "matched": tp,
              "precision": round(prec, 3), "recall": round(rec, 3),
              "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
              "median_best_iou": round(float(best_iou.median()), 3) if len(best_iou) else None,
              "over_segmented_parcels": over, "merged_predictions": under,
              "recall_by_size": by_size, "iou_threshold": float(cfg.evaluate.iou_match),
              "prediction_file": str(pred_path)}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pred_path).stem
    (out_dir / f"evaluation_{stem}_{split}.json").write_text(json.dumps(result, indent=2))
    ref.assign(best_iou=best_iou.values, matched=[i in matched_refs for i in ref.index]).to_file(
        out_dir / f"evaluation_{stem}_{split}.gpkg", layer="parcels", driver="GPKG")
    print(json.dumps(result, indent=2))
    return result
