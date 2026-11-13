"""Spatial K-fold cross-validation: score a model choice on EVERY digitised farm of an area, not a few patches.

A single train / validation / test split of one area can leave validation with a handful of patches (the Salama
round-3 run had 4), so the chosen model can be the wrong one. Here the area's blocks (``split.block_size_m``) are
dealt into K folds holding similar numbers of farms. For fold k the model trains on the other folds and predicts
fold k, so every farm is predicted exactly once by a model that never saw it ("out-of-fold"):

* every candidate is trained K times with a fixed number of epochs (no early stopping on the fold it is scored
  on) and the LAST epoch is used, so the out-of-fold score is not optimistically selected;
* the polygon post-processing of each candidate is tuned on its pooled out-of-fold predictions (the same tuning
  for every candidate, so the comparison stays fair; the absolute score is slightly optimistic);
* candidates are compared fold by fold (a better recipe should win in most folds, not only on average);
* the K models of the chosen recipe are used together (ensemble) for mapping: all K share one normalisation.

Patches are prepared once per fold: masks of fold 0 are re-used by the other folds (linked, not rebuilt).
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..data.splits import GeoSplit, split_path
from ..utils.helpers import write_json

log = logging.getLogger(__name__)


def assign_folds(labels, crs: str, block_m: float, k: int, seed: int = 42) -> dict[tuple[int, int], int]:
    """Blocks holding farms -> fold 0..k-1, largest blocks first to the fold with the fewest farms."""
    pts = labels.to_crs(crs).geometry.representative_point()
    keys = list(zip(np.floor(pts.x / block_m).astype(int), np.floor(pts.y / block_m).astype(int)))
    counts: dict[tuple[int, int], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    if len(counts) < k:
        raise ValueError(f"only {len(counts)} blocks of {block_m:.0f} m hold farms: fewer than {k} folds "
                         "(lower the block size or the number of folds)")
    order = list(counts)
    np.random.default_rng(seed).shuffle(order)
    order.sort(key=lambda b: -counts[b])            # stable: ties keep the seeded order
    load, out = [0] * k, {}
    for b in order:
        f = int(np.argmin(load))
        out[b] = f
        load[f] += counts[b]
    log.info("%d farm blocks of %.0f m in %d folds; farms per fold %s", len(counts), block_m, k, load)
    return out


def _write_fold_split(cfg, crs: str, block_m: float, assignment: dict, fold: int) -> None:
    """Fold ``fold`` = validation, every other fold = training (``build_split`` re-uses this file)."""
    sc = cfg.split
    g = GeoSplit(crs, block_m, dict(sc.fractions), int(sc.seed),
                 {b: ("val" if f == fold else "train") for b, f in assignment.items()})
    path = split_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.save(path)


def prepare_folds(cfg_for_fold: Callable[[int], object], k: int, seed: int = 42) -> pd.DataFrame:
    """Masks once, then patches + checks for every fold. Returns patches / farms per fold and split."""
    from ..data.ingestion import run_ingestion
    from ..pipeline import prepare

    cfg0 = cfg_for_fold(0)
    labels, _, _ = run_ingestion(cfg0, write_report=False)
    crs = cfg0.split.get("crs") or str(labels.crs)
    block_m = float(cfg0.split.block_size_m)
    assignment = assign_folds(labels[labels["train"]] if "train" in labels else labels, crs, block_m, k, seed)
    dir0 = Path(cfg0.paths.processed_dir)
    rows = []
    for f in range(k):
        cfg = cfg_for_fold(f)
        pdir = Path(cfg.paths.processed_dir)
        meta = pdir / "metadata"
        if not (meta / "patches.csv").exists():
            _write_fold_split(cfg, crs, block_m, assignment, f)
            if f == 0:
                prepare(cfg, ("masks", "patches", "check"))
            else:
                if not (pdir / "masks").exists():
                    os.symlink(dir0 / "masks", pdir / "masks")
                prepare(cfg, ("patches", "check"))
            if f > 0:   # one normalisation for all folds, so the K models can be averaged at mapping time
                shutil.copy(dir0 / "metadata" / "normalization.json", meta / "normalization.json")
        t = pd.read_csv(meta / "patches.csv")
        for s, g in t.groupby("split"):
            rows.append({"fold": f, "split": s, "patches": len(g), "farms": int(g["n_farms"].sum())})
    return pd.DataFrame(rows).pivot_table(index="fold", columns="split", values=["patches", "farms"], fill_value=0)


def train_fold(cfg) -> Path:
    """Train with a fixed number of epochs; returns the LAST epoch (not the epoch best on the scored fold)."""
    from ..training.trainer import train

    train(cfg, resume=True)
    return Path(cfg.paths.models_dir) / "latest_checkpoint.pth"


def out_of_fold(fold_cfgs: list, models: list, tta: bool) -> dict:
    """Each fold's model on its own validation patches, pooled into one cache (``fold`` tagged on items)."""
    from ..training.evaluation import predict_patches

    items, gsd, dist = [], 3.0, True
    for f, (cfg, m) in enumerate(zip(fold_cfgs, models)):
        c = predict_patches(cfg, m, "val", tta=tta)
        for it in c["items"]:
            it["fold"] = f
            it["patch_id"] = f"f{f}_{it['patch_id']}"
        items += c["items"]
        gsd, dist = c["gsd"], dist and c["distance"]
    return {"split": "out-of-fold", "model": [str(m) for m in models], "items": items, "tta": tta, "gsd": gsd,
            "distance": dist}


def per_fold_scores(cache: dict, pc, iou: float) -> pd.DataFrame:
    """Object F1 / precision / recall / merged / split per fold for one post-processing."""
    from ..training.evaluation import object_metrics

    rows = []
    for f in sorted({it["fold"] for it in cache["items"]}):
        sub = {**cache, "items": [it for it in cache["items"] if it["fold"] == f]}
        r, _ = object_metrics(sub, pc, iou)
        rows.append({"fold": f, **{k: r[k] for k in ("object_f1", "object_precision", "object_recall",
                                                      "merged_predictions", "split_farms", "farms_true")}})
    return pd.DataFrame(rows).set_index("fold")


def choose(table: pd.DataFrame, folds: pd.DataFrame, baseline: str, min_gain: float = 0.01,
           min_fold_wins: float = 0.75) -> str:
    """The best candidate by pooled out-of-fold object F1, if it beats ``baseline`` by ``min_gain`` AND wins in at
    least ``min_fold_wins`` of the folds; otherwise the baseline (a gain that does not hold fold by fold is noise)."""
    best = table["object_f1"].astype(float).idxmax()
    if best == baseline:
        return baseline
    gain = float(table.loc[best, "object_f1"]) - float(table.loc[baseline, "object_f1"])
    wins = (folds[best] > folds[baseline]).mean()
    if gain >= min_gain and wins >= min_fold_wins:
        return best
    log.info("%s is %.3f better than %s but wins only %.0f%% of folds: keeping %s", best, gain, baseline,
             100 * wins, baseline)
    return baseline


def save_summary(out: Path, table: pd.DataFrame, folds: pd.DataFrame, chosen: str) -> None:
    write_json({"chosen": chosen, "candidates": table.reset_index().to_dict("records"),
                "per_fold_object_f1": folds.reset_index().to_dict("records")}, Path(out) / "crossval.json")

