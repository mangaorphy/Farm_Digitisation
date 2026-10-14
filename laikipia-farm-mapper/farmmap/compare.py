"""Score U-Net and Delineate Anything (several upsampling factors) on the SAME held-out blocks."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union

from .data import block_window, load_or_create_split, split_blocks
from .reference import load_reference
from .scoring import Scorer


def _regions(cfg, scenes, which, progress):
    split = load_or_create_split(cfg, scenes, progress)
    blocks = split_blocks(split, which)
    if not blocks:
        raise RuntimeError(f"No '{which}' blocks to compare on.")
    out = {}
    for s in scenes:
        w = [block_window(b) for b in blocks if b["scene"] == s["name"]]
        if w:
            out[s["name"]] = (s, w)
    return out


def compare_engines(cfg, scenes=None, which="test", unet_checkpoint=None, da_upsamples=(1, 2, 4),
                    polygon_files=(), only=None, progress=print) -> pd.DataFrame:
    """Rows = methods, columns = F1 / precision / recall / mean IoU and recall by farm size.

    polygon_files: extra polygon files (e.g. output of the official Delineate Anything script) scored
    on the same blocks.
    only: {"SOURCE": ["Digitized"]} scores only reference farms with those attribute values. Use it to
    judge the method on the kind of farm you still have to digitise by hand.
    """
    from .pipeline import list_scenes
    from .reference import read_vector

    scenes = scenes or list_scenes(cfg)
    per_scene = _regions(cfg, scenes, which, progress)
    ckpt = Path(unet_checkpoint or cfg.path(cfg.model.checkpoint))
    methods = []
    if ckpt.exists():
        methods.append(("unet", None))
    for up in da_upsamples:
        methods.append((f"delineate_anything_x{up}", up))
    for f in polygon_files:
        methods.append((f"file:{Path(f).name}", f))

    rows = {}
    for name, arg in methods:
        scorer = Scorer(cfg)
        progress(f"== {name}")
        try:
            for sname, (s, wins) in per_scene.items():
                with rasterio.open(s["imagery"]) as src:
                    crs, tr = src.crs, src.transform
                ref = load_reference(cfg, crs=crs)
                for col, vals in (only or {}).items():
                    ref = ref[ref[col].astype(str).isin([str(v) for v in vals])]
                region = unary_union([box(*rasterio.windows.bounds(w, tr)) for w in wins])
                if name == "unet":
                    from .polygonize import polygonize_probs
                    from .predict import predict_raster

                    with tempfile.TemporaryDirectory() as tmp:
                        pr = predict_raster(cfg, s["imagery"], ckpt, Path(tmp) / "p.tif", wins, lambda *_: None)
                        pred = polygonize_probs(cfg, pr, progress=lambda *_: None)
                elif name.startswith("delineate_anything"):
                    from .delineate import delineate

                    pred = delineate(cfg, s["imagery"], regions=wins, upsample=arg, progress=lambda *_: None)
                else:
                    pred = read_vector(arg)
                    pred = pred.set_crs("EPSG:4326") if pred.crs is None else pred
                    pred = pred.to_crs(crs)
                scorer.add(pred, ref, region)
            rows[name] = scorer.summary()
        except ImportError as exc:  # optional engine not installed: report and continue
            progress(f"   skipped: {exc}")
    return pd.DataFrame(rows).T
