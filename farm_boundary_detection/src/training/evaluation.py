"""Evaluate a trained model on held-out patches.

* pixel level: both heads (interior IoU / Dice / precision / recall; boundary IoU / F1 / precision / recall);
* object level: individual farms from the same post-processing as inference (``segment``) matched one-to-one
  against the true instance masks (IoU >= ``evaluation.instance_iou``), plus merged and split counts;
* figures: GT vs predicted interior and boundary, overlays and predicted farms.

The probabilities of a split are computed once (``predict_patches``) so that post-processing thresholds can be
tuned on validation patches (``tune_postprocess``) without running the network again.

Outputs: ``<outputs_dir>/evaluation/<split>/metrics.json``, ``per_patch.csv``, ``visuals/*.png``.
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
from ..utils.visualization import plot_prediction_comparison
from .metrics import SegmentationMetrics, combine_instance_scores, instance_scores

log = logging.getLogger(__name__)


@torch.no_grad()
def predict_patches(cfg, model_path: Path, split: str = "val", batch_size: int = 8) -> dict:
    """Run the model on every patch of ``split`` in ``cfg.paths.processed_dir``. The model's own normalisation
    is used, so a model can be scored on patches prepared with other label settings (a common reference)."""
    device = get_device()
    model, ck = load_model(model_path, device)
    norm = Normalizer(ck["normalization"])
    table = load_patch_table(cfg)
    rows = table[table["split"] == split]
    if rows.empty:
        raise ValueError(f"no {split} patches in {cfg.paths.processed_dir}")
    pdir = Path(cfg.paths.processed_dir)
    rgb = rgb_indices(norm.n_bands, norm.stats.get("band_names"), cfg.imagery.get("rgb_bands"))
    items: list[dict] = []
    ids = rows["patch_id"].tolist()
    for i in range(0, len(ids), batch_size):
        chunk = []
        for pid in ids[i:i + batch_size]:
            def read(sub, pid=pid):
                with rasterio.open(pdir / sub / f"{pid}.tif") as s:
                    return s.read()
            raw = read("images")
            x, ok = norm(raw)
            chunk.append({"patch_id": pid, "x": x, "ok": ok, "rgb": raw[rgb],
                          "gt_int": read("interior_masks")[0], "gt_bnd": read("boundary_masks")[0],
                          "valid": read("valid_masks")[0].astype(bool) & ok, "gt_inst": read("instance_masks")[0]})
        il, bl = model(torch.from_numpy(np.stack([c["x"] for c in chunk])).to(device))
        pi, pb = torch.sigmoid(il)[:, 0].float().cpu().numpy(), torch.sigmoid(bl)[:, 0].float().cpu().numpy()
        for c, a, b in zip(chunk, pi, pb):
            c.pop("x")
            a[~c["ok"]], b[~c["ok"]] = 0, 1
            c["p_int"], c["p_bnd"] = a.astype(np.float16), b.astype(np.float16)
            items.append(c)
    return {"split": split, "model": str(model_path), "items": items, "gsd": float(norm.stats.get("gsd_m") or 3.0)}


def pixel_metrics(cfg, cache: dict) -> dict:
    m = SegmentationMetrics(float(cfg.evaluation.threshold), int(cfg.train.boundary_tolerance_px))
    for it in cache["items"]:
        def t(a):
            return torch.from_numpy(np.asarray(a, np.float32))[None, None]
        # metrics take logits: pass logit(p) so that sigmoid gives back p
        p_i = np.clip(it["p_int"].astype(np.float32), 1e-6, 1 - 1e-6)
        p_b = np.clip(it["p_bnd"].astype(np.float32), 1e-6, 1 - 1e-6)
        m.update(t(np.log(p_i / (1 - p_i))), t(np.log(p_b / (1 - p_b))),
                 {"interior_mask": t(it["gt_int"]), "boundary_mask": t(it["gt_bnd"]), "valid_mask": t(it["valid"])})
    return m.compute()


def object_metrics(cache: dict, pc, iou: float) -> tuple[dict, list[dict]]:
    per = []
    for it in cache["items"]:
        inst = segment(it["p_int"].astype(np.float32), it["p_bnd"].astype(np.float32), pc, cache["gsd"])
        it["pred_inst"] = inst
        per.append({"patch_id": it["patch_id"], **instance_scores(it["gt_inst"], inst, it["valid"], iou)})
    return combine_instance_scores(per), per


def tune_postprocess(cfg, cache: dict, grid: dict | None = None) -> tuple[Config, pd.DataFrame]:
    """Grid-search post-processing thresholds for the best object F1 (use VALIDATION patches only)."""
    grid = grid or {"interior_threshold": [0.4, 0.5, 0.6], "seed_threshold": [0.4, 0.5, 0.6, 0.7],
                    "min_seed_px": [10, 25]}
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        pc = copy.deepcopy(cfg.postprocess)
        for k, v in zip(keys, values):
            pc[k] = v
        res, _ = object_metrics(cache, pc, float(cfg.evaluation.instance_iou))
        rows.append({**dict(zip(keys, values)), **res})
    table = pd.DataFrame(rows).sort_values(["object_f1", "mean_matched_iou"], ascending=False)
    best = copy.deepcopy(cfg.postprocess)
    for k in keys:
        v = table.iloc[0][k]
        best[k] = int(v) if k == "min_seed_px" else float(v)
    log.info("best post-processing on %s: %s -> object F1 %.3f", cache["split"], {k: best[k] for k in keys},
             table.iloc[0]["object_f1"])
    return best, table


def evaluate(cfg, model_path: Path, split: str = "test", out_dir: Path | None = None, postprocess=None,
             cache: dict | None = None) -> dict:
    """Pixel + object metrics and figures for one split. ``postprocess`` overrides ``cfg.postprocess``."""
    cache = cache or predict_patches(cfg, model_path, split)
    pc = postprocess or cfg.postprocess
    ec = cfg.evaluation
    out = Path(out_dir or Path(cfg.paths.outputs_dir) / "evaluation" / split)
    (out / "visuals").mkdir(parents=True, exist_ok=True)
    pix = pixel_metrics(cfg, cache)
    obj, per = object_metrics(cache, pc, float(ec.instance_iou))
    result = {"split": split, "model": str(model_path), "patches": len(cache["items"]), **pix, **obj,
              "postprocess": {k: pc[k] for k in ("interior_threshold", "seed_threshold", "boundary_threshold",
                                                 "min_seed_px")}}
    write_json(result, out / "metrics.json")
    pd.DataFrame(per).to_csv(out / "per_patch.csv", index=False)
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
