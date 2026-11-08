"""Evaluate trained models on held-out patches.

* pixel level: both heads (interior IoU / Dice / precision / recall; boundary IoU / F1 / precision / recall) and
  their confusion matrices;
* object level: individual farms from the same post-processing as inference (``segment``) matched one-to-one
  against the true instance masks (IoU >= ``evaluation.instance_iou``), merged and split counts, and recall /
  precision by farm size (true polygon areas);
* figures: GT vs predicted interior and boundary, overlays and predicted farms.

The probabilities of a split are computed once (``predict_patches``; several models = ensemble, optional flip
TTA) so post-processing can be tuned on validation patches (``tune_postprocess``) without the network.

Outputs: ``<out_dir>/metrics.json``, ``per_patch.csv``, ``by_size.csv``, ``confusion.png``, ``by_size.png``,
``visuals/*.png``.
"""
from __future__ import annotations

import copy
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

from ..data.dataset import Normalizer, load_patch_table
from ..gis.polygon_reconstruction import segment
from ..models.multi_task_unet import load_model
from ..utils.helpers import Config, get_device, rgb_indices, write_json
from ..utils.visualization import plot_confusion, plot_prediction_comparison, plot_size_scores
from .metrics import SegmentationMetrics, combine_instance_scores, instance_scores, scores_by_size

log = logging.getLogger(__name__)


@torch.no_grad()
def predict_patches(cfg, model_path, split: str = "val", batch_size: int = 8, tta: bool = False) -> dict:
    """Run one model (or a list = ensemble, probabilities averaged) on every patch of ``split`` in
    ``cfg.paths.processed_dir``. Each model's own normalisation is used, so models can be scored on patches
    prepared with other label settings (a common reference)."""
    device = get_device()
    paths = model_path if isinstance(model_path, (list, tuple)) else [model_path]
    models = [load_model(p, device) for p in paths]
    norms = [Normalizer(ck["normalization"]) for _, ck in models]
    with_dist = all(m.has_distance for m, _ in models)
    table = load_patch_table(cfg)
    rows = table[table["split"] == split]
    if rows.empty:
        raise ValueError(f"no {split} patches in {cfg.paths.processed_dir}")
    pdir = Path(cfg.paths.processed_dir)
    rgb = rgb_indices(norms[0].n_bands, norms[0].stats.get("band_names"), cfg.imagery.get("rgb_bands"))
    flips = [(), (-1,), (-2,), (-1, -2)] if tta else [()]
    items: list[dict] = []
    ids = rows["patch_id"].tolist()
    for i in range(0, len(ids), batch_size):
        chunk = []
        for pid in ids[i:i + batch_size]:
            def read(sub, pid=pid):
                with rasterio.open(pdir / sub / f"{pid}.tif") as s:
                    return s.read()
            raw = read("images")
            ok = norms[0].data_mask(raw)
            chunk.append({"patch_id": pid, "raw": raw, "ok": ok, "rgb": raw[rgb],
                          "gt_int": read("interior_masks")[0], "gt_bnd": read("boundary_masks")[0],
                          "valid": read("valid_masks")[0].astype(bool) & ok, "gt_inst": read("instance_masks")[0]})
        acc, n = None, 0
        for (model, _), norm in zip(models, norms):
            x = torch.from_numpy(np.stack([norm(c["raw"])[0] for c in chunk])).to(device)
            for f in flips:
                out = model(torch.flip(x, f) if f else x)
                p = torch.cat([torch.sigmoid(o) for o in out[: 3 if with_dist else 2]], 1)
                p = torch.flip(p, f) if f else p
                acc = p if acc is None else acc + p
                n += 1
        probs = (acc / n).float().cpu().numpy()
        for c, pr in zip(chunk, probs):
            c.pop("raw")
            ok = c["ok"]
            pr[0][~ok], pr[1][~ok] = 0, 1
            c["p_int"], c["p_bnd"] = pr[0].astype(np.float16), pr[1].astype(np.float16)
            if with_dist:
                pr[2][~ok] = 0
                c["p_dist"] = pr[2].astype(np.float16)
            items.append(c)
    return {"split": split, "model": [str(p) for p in paths], "items": items, "tta": tta,
            "gsd": float(norms[0].stats.get("gsd_m") or 3.0), "distance": with_dist}


def ensemble_caches(caches: list[dict]) -> dict:
    """Average the probabilities of several ``predict_patches`` results on the same patches."""
    base = caches[0]
    ids = [it["patch_id"] for it in base["items"]]
    for c in caches[1:]:
        if [it["patch_id"] for it in c["items"]] != ids:
            raise ValueError("caches cover different patches")
    with_dist = all(c["distance"] for c in caches)
    items = []
    for k, it in enumerate(base["items"]):
        new = {key: v for key, v in it.items() if key not in ("p_int", "p_bnd", "p_dist", "pred_inst")}
        for key in ("p_int", "p_bnd") + (("p_dist",) if with_dist else ()):
            new[key] = np.mean([c["items"][k][key].astype(np.float32) for c in caches], axis=0).astype(np.float16)
        items.append(new)
    return {**base, "items": items, "distance": with_dist, "model": sum((c["model"] for c in caches), [])}


def _logit(p: np.ndarray) -> torch.Tensor:
    p = np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6)
    return torch.from_numpy(np.log(p / (1 - p)))[None, None]


def pixel_metrics(cfg, cache: dict) -> tuple[dict, dict]:
    """(scores, confusion matrices) of both heads over valid pixels."""
    m = SegmentationMetrics(float(cfg.evaluation.threshold), int(cfg.train.boundary_tolerance_px))
    for it in cache["items"]:
        def t(a):
            return torch.from_numpy(np.asarray(a, np.float32))[None, None]
        m.update(_logit(it["p_int"]), _logit(it["p_bnd"]),
                 {"interior_mask": t(it["gt_int"]), "boundary_mask": t(it["gt_bnd"]), "valid_mask": t(it["valid"])})
    return m.compute(), m.confusion()


def object_metrics(cache: dict, pc, iou: float, details: bool = False) -> tuple[dict, list[dict]]:
    per = []
    for it in cache["items"]:
        pd_ = it.get("p_dist")
        inst = segment(it["p_int"].astype(np.float32), it["p_bnd"].astype(np.float32), pc, cache["gsd"],
                       pd_.astype(np.float32) if pd_ is not None else None)
        it["pred_inst"] = inst
        per.append({"patch_id": it["patch_id"],
                    **instance_scores(it["gt_inst"], inst, it["valid"], iou, details=details)})
    return combine_instance_scores(per), per


_POOL_CACHE: dict | None = None  # shared with forked workers (no copy of the probabilities per task)


def _score_combo(args) -> dict:
    pc, iou, combo = args
    light = {"items": [{k: it[k] for k in ("patch_id", "p_int", "p_bnd", "p_dist", "gt_inst", "valid") if k in it}
                       for it in _POOL_CACHE["items"]], "gsd": _POOL_CACHE["gsd"]}
    res, _ = object_metrics(light, Config(pc), iou)
    return {**combo, **res}


def _grid(cfg, cache, base, grid: dict) -> pd.DataFrame:
    """Score every combination; in parallel over CPU cores where processes can be forked (Linux, Kaggle)."""
    import multiprocessing as mp
    import os

    global _POOL_CACHE
    keys = list(grid)
    iou = float(cfg.evaluation.instance_iou)
    tasks = []
    for values in itertools.product(*(grid[k] for k in keys)):
        pc = dict(copy.deepcopy(base))
        combo = dict(zip(keys, values))
        pc.update(combo)
        tasks.append((pc, iou, combo))
    workers = min(len(tasks), os.cpu_count() or 1)
    import sys
    if workers > 1 and sys.platform.startswith("linux"):  # fork is unsafe with torch on macOS
        _POOL_CACHE = cache
        try:
            with mp.get_context("fork").Pool(workers) as pool:
                rows = pool.map(_score_combo, tasks, chunksize=1)
        finally:
            _POOL_CACHE = None
    else:
        rows = []
        for pc, _, combo in tasks:
            res, _ = object_metrics(cache, Config(pc), iou)
            rows.append({**combo, **res})
    return pd.DataFrame(rows).sort_values(["object_f1", "mean_matched_iou"], ascending=False)


def tune_postprocess(cfg, cache: dict, stage1: dict | None = None, stage2: dict | None = None) -> tuple[Config, pd.DataFrame]:
    """Two-stage search of the post-processing for the best object F1 (use VALIDATION patches only).

    Stage 1: seed source x seed threshold x minimum seed size. Stage 2 (around the stage-1 winner): extent
    threshold x boundary threshold x minimum farm area. Ranges are wide on purpose: the first round of tuning
    ended on the edges of a narrower grid."""
    sources = ["boundary", "distance", "combined"] if cache.get("distance") else ["boundary"]
    stage1 = stage1 or {"seed_source": sources, "seed_threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        "min_seed_px": [10, 25, 50, 100]}
    stage2 = stage2 or {"interior_threshold": [0.25, 0.3, 0.35, 0.4, 0.5], "boundary_threshold": [0.3, 0.5, 0.7],
                        "min_area_m2": [100, 200, 400]}
    base = copy.deepcopy(cfg.postprocess)
    base["interior_threshold"] = 0.35
    t1 = _grid(cfg, cache, base, stage1)
    for k in stage1:
        base[k] = t1.iloc[0][k]
    t2 = _grid(cfg, cache, base, stage2)
    best = copy.deepcopy(base)
    for k in stage2:
        best[k] = t2.iloc[0][k]
    for k in ("min_seed_px",):
        best[k] = int(best[k])
    for k in ("seed_threshold", "interior_threshold", "boundary_threshold", "min_area_m2"):
        best[k] = float(best[k])
    best["seed_source"] = str(best.get("seed_source", "boundary"))
    table = pd.concat([t1.assign(stage=1), t2.assign(stage=2)], ignore_index=True)
    at_edge = [k for k, g in {**stage1, **stage2}.items()
               if k != "seed_source" and best[k] in (min(g), max(g))]
    log.info("best post-processing on %s: %s -> object F1 %.3f%s", cache["split"],
             {k: best[k] for k in list(stage1) + list(stage2)}, t2.iloc[0]["object_f1"],
             f" (on the grid edge: {at_edge})" if at_edge else "")
    return best, table


def true_areas(cfg) -> dict[int, float]:
    """Polygon area (m2) per farm instance id, for scores by farm size."""
    from ..data.ingestion import load_farm_polygons
    labels, _ = load_farm_polygons(cfg)
    return dict(zip(labels["instance_id"].astype(int), labels.area.astype(float)))


def evaluate(cfg, model_path, split: str = "test", out_dir: Path | None = None, postprocess=None,
             cache: dict | None = None, areas: dict | None = None, title: str = "") -> dict:
    """Pixel + object metrics, scores by size, confusion matrices and figures for one split."""
    cache = cache or predict_patches(cfg, model_path, split)
    pc = postprocess or cfg.postprocess
    ec = cfg.evaluation
    out = Path(out_dir or Path(cfg.paths.outputs_dir) / "evaluation" / split)
    (out / "visuals").mkdir(parents=True, exist_ok=True)
    pix, conf = pixel_metrics(cfg, cache)
    obj, per = object_metrics(cache, pc, float(ec.instance_iou), details=True)
    by_size = scores_by_size(per, cache["gsd"], areas)
    keys = ("interior_threshold", "seed_threshold", "boundary_threshold", "min_seed_px", "min_area_m2", "seed_source")
    result = {"split": split, "model": cache["model"], "tta": cache.get("tta", False), "patches": len(cache["items"]),
              **pix, **obj, "confusion": conf, "by_size": by_size.reset_index().to_dict("records"),
              "postprocess": {k: pc.get(k) for k in keys}}
    write_json(result, out / "metrics.json")
    pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_records")} for r in per]).to_csv(
        out / "per_patch.csv", index=False)
    by_size.to_csv(out / "by_size.csv")
    plot_confusion(conf, out / "confusion.png", f"{title} {split}: pixel confusion matrices (valid pixels)")
    plot_size_scores(by_size, out / "by_size.png", f"{title} {split}: individual farms by size")
    n_vis = int(ec.get("n_visuals", 12))
    order = sorted(range(len(per)), key=lambda i: -per[i]["gt"])
    picks = order[: n_vis // 2] + list(np.random.default_rng(0).permutation(len(per))[: n_vis - n_vis // 2])
    for i in dict.fromkeys(picks):
        it, s = cache["items"][i], per[i]
        plot_prediction_comparison(it["rgb"], it["gt_int"], it["p_int"].astype(np.float32), it["gt_bnd"],
                                   it["p_bnd"].astype(np.float32), it["valid"].astype(np.uint8),
                                   out / "visuals" / f"{it['patch_id']}.png",
                                   f"{it['patch_id']}: {s['matched']}/{s['gt']} farms matched", it.get("pred_inst"))
    log.info("%s: interior IoU %.3f Dice %.3f P %.3f R %.3f | boundary IoU %.3f F1 %.3f P %.3f R %.3f | farms: "
             "F1 %.3f (P %.3f R %.3f), %d merged, %d split", split, result["interior_iou"], result["interior_dice"],
             result["interior_precision"], result["interior_recall"], result["boundary_iou"], result["boundary_f1"],
             result["boundary_precision"], result["boundary_recall"], result["object_f1"],
             result["object_precision"], result["object_recall"], result["merged_predictions"], result["split_farms"])
    return result
