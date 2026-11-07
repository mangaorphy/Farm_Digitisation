"""Task 3: cut aligned image / mask patches with rasterio windows.

For each image, candidate windows of ``patches.size`` px are laid on two grids: the training grid (step
``patches.stride``, overlapping) and the evaluation grid (step ``patches.eval_stride``). The last window of a
row/column is shifted back to end exactly at the image edge, so no strip of the image is lost when the size
is not a multiple of the patch size; images smaller than one patch are padded with nodata (and marked
invalid). A window's split comes from the geographic split (see ``splits.py``): training-grid windows are kept
only in train blocks, evaluation-grid windows only in val / test blocks, and windows straddling two splits are
dropped.

Accepted patches: at least ``min_farm_pixels`` labelled farm pixels, at least ``min_valid_fraction`` usable
pixels and at most ``max_nodata_fraction`` image nodata. Of the remaining (background) candidates with enough
usable pixels, a random ``background_keep_fraction`` is kept, so the model also sees non-farm land.

Output (``paths.processed_dir``)::

    images/<id>.tif            selected bands, original dtype and nodata, georeferenced
    interior_masks/<id>.tif    uint8 0/1
    boundary_masks/<id>.tif    uint8 0/1
    valid_masks/<id>.tif       uint8 0/1   (1 = used in loss / metrics)
    instance_masks/<id>.tif    uint32 farm instance ids
    metadata/patches.csv       one row per patch (window, transform, CRS, pixel counts, farm ids, split)
    metadata/patches.geojson   patch footprints (EPSG:4326) for QGIS
    metadata/normalization.json  per-band statistics of the training patches
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import Polygon
from tqdm import tqdm

from ..utils.helpers import stable_seed, write_json
from .ingestion import RasterInfo, nodata_value, select_bands
from .splits import GeoSplit

log = logging.getLogger(__name__)
PATCH_DIRS = ("images", "interior_masks", "boundary_masks", "valid_masks", "instance_masks")


def window_starts(length: int, size: int, stride: int) -> list[int]:
    """Start offsets covering [0, length) with windows of ``size``; the last one ends at ``length``."""
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts


def _footprint(win: Window, transform, to_split: Transformer) -> Polygon:
    c0, r0, w, h = win.col_off, win.row_off, win.width, win.height
    corners = [transform * (c0, r0), transform * (c0 + w, r0), transform * (c0 + w, r0 + h), transform * (c0, r0 + h)]
    xs, ys = to_split.transform([p[0] for p in corners], [p[1] for p in corners])
    return Polygon(zip(xs, ys))


def _write(path: Path, arr: np.ndarray, transform, crs, dtype: str, nodata=None, descriptions=None) -> None:
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata, compress="deflate") as d:
        d.write(arr.astype(dtype))
        if descriptions:
            d.descriptions = tuple(descriptions)


def extract_patches(cfg, labels: gpd.GeoDataFrame, rasters: list[RasterInfo], split: GeoSplit,
                    mask_root: Path | None = None) -> pd.DataFrame:
    """Cut patches from every image (masks must exist in ``mask_root/<scene_id>/``)."""
    pc = cfg.patches
    out = Path(cfg.paths.processed_dir)
    mask_root = Path(mask_root or out / "masks")
    if pc.get("overwrite", True):
        for d in PATCH_DIRS:
            shutil.rmtree(out / d, ignore_errors=True)
    for d in PATCH_DIRS:
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    size, stride, eval_stride = int(pc.size), int(pc.stride), int(pc.get("eval_stride", pc.size))
    uid_of = dict(zip(labels["instance_id"].astype(int), labels["farm_uid"]))
    rows = []
    for info in rasters:
        mdir = mask_root / info.scene_id
        if not (mdir / "interior.tif").exists():
            raise FileNotFoundError(f"masks for {info.scene_id} missing in {mdir}: run the masks step first")
        bands = select_bands(info, cfg.imagery.get("bands"))
        nodata = nodata_value(info, cfg)
        to_split = Transformer.from_crs(info.crs, split.crs, always_xy=True)
        rng = np.random.default_rng(stable_seed(cfg.project.seed, info.scene_id))
        cand: dict[tuple[int, int], str] = {}
        for grid_stride, allowed in ((stride, {"train"}), (eval_stride, {"val", "test"})):
            for r in window_starts(info.height, size, grid_stride):
                for c in window_starts(info.width, size, grid_stride):
                    s = split.split_for_footprint(_footprint(Window(c, r, size, size), info.affine, to_split))
                    if s in allowed:
                        cand[(r, c)] = s
        kept = n_bg_seen = 0
        desc = [info.descriptions[b - 1] if info.descriptions else f"band_{b}" for b in bands]
        with rasterio.open(info.path) as src, \
                rasterio.open(mdir / "interior.tif") as mi, rasterio.open(mdir / "boundary.tif") as mb, \
                rasterio.open(mdir / "valid.tif") as mv, rasterio.open(mdir / "instance.tif") as mn:
            for (r, c), s in tqdm(sorted(cand.items()), desc=f"patches {info.scene_id}", unit="win", leave=False):
                win = Window(c, r, size, size)
                interior = mi.read(1, window=win, boundless=True, fill_value=0)
                valid = mv.read(1, window=win, boundless=True, fill_value=0)
                n_valid = int(valid.sum())
                if n_valid == 0:
                    continue
                farm_px = int((interior.astype(bool) & valid.astype(bool)).sum())
                valid_frac = n_valid / (size * size)
                img = src.read(bands, window=win, boundless=True, fill_value=nodata)
                nodata_px = np.all(img == nodata, axis=0)
                nodata_frac = float(nodata_px.mean())
                if valid_frac < float(pc.min_valid_fraction) or nodata_frac > float(pc.max_nodata_fraction):
                    continue
                is_bg = farm_px < int(pc.min_farm_pixels)
                if is_bg:
                    n_bg_seen += 1
                    if rng.random() >= float(pc.background_keep_fraction):
                        continue
                boundary = mb.read(1, window=win, boundless=True, fill_value=0)
                instance = mn.read(1, window=win, boundless=True, fill_value=0)
                valid = (valid.astype(bool) & ~nodata_px).astype(np.uint8)  # padding / nodata never valid
                pid = f"{info.scene_id}_r{r:05d}_c{c:05d}"
                wt = src.window_transform(win)
                _write(out / "images" / f"{pid}.tif", img, wt, info.crs, str(img.dtype), nodata, desc)
                _write(out / "interior_masks" / f"{pid}.tif", interior, wt, info.crs, "uint8")
                _write(out / "boundary_masks" / f"{pid}.tif", boundary, wt, info.crs, "uint8")
                _write(out / "valid_masks" / f"{pid}.tif", valid, wt, info.crs, "uint8")
                _write(out / "instance_masks" / f"{pid}.tif", instance, wt, info.crs, "uint32")
                ids = [int(i) for i in np.unique(instance[valid.astype(bool)]) if i]
                fp = Polygon([wt * (0, 0), wt * (size, 0), wt * (size, size), wt * (0, size)])
                rows.append({
                    "patch_id": pid, "split": s, "source_image": info.path.name, "scene_id": info.scene_id,
                    "crs": info.crs, "transform": json.dumps([round(v, 6) for v in tuple(wt)[:6]]),
                    "col_off": c, "row_off": r, "width": size, "height": size,
                    "farm_pixels": farm_px, "boundary_pixels": int((boundary.astype(bool) & valid.astype(bool)).sum()),
                    "valid_pixels": int(valid.sum()), "nodata_pixels": int(nodata_px.sum()),
                    "n_farms": len(ids), "farm_ids": ";".join(uid_of.get(i, str(i)) for i in ids),
                    "is_background": bool(is_bg), "bands": json.dumps(bands), "geometry_wkt": fp.wkt,
                })
                kept += 1
        log.info("%s: %d candidate windows -> %d patches (%d background candidates seen)", info.scene_id,
                 len(cand), kept, n_bg_seen)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no patches were accepted: check the masks figures, patches.min_farm_pixels and "
                           "patches.min_valid_fraction")
    df.to_csv(out / "metadata" / "patches.csv", index=False)
    footprints = [gpd.GeoSeries.from_wkt([w], crs=c).to_crs("EPSG:4326").iloc[0]
                  for w, c in zip(df["geometry_wkt"], df["crs"])]
    gpd.GeoDataFrame(df.drop(columns=["geometry_wkt"]), geometry=footprints, crs="EPSG:4326").to_file(
        out / "metadata" / "patches.geojson", driver="GeoJSON")
    summary = df.groupby("split").agg(patches=("patch_id", "size"), background=("is_background", "sum"),
                                      farm_pixels=("farm_pixels", "sum"), boundary_pixels=("boundary_pixels", "sum"))
    log.info("patches by split:\n%s", summary.to_string())
    return df


def patch_footprints(df: pd.DataFrame, crs: str) -> gpd.GeoDataFrame:
    """Patch footprints in one CRS (patches from different images may be in different CRSs)."""
    parts = []
    for c, g in df.groupby("crs"):
        parts.append(gpd.GeoDataFrame(g, geometry=gpd.GeoSeries.from_wkt(g["geometry_wkt"]).values, crs=c).to_crs(crs))
    return gpd.GeoDataFrame(pd.concat(parts), geometry="geometry", crs=crs)


def compute_normalization(cfg, df: pd.DataFrame) -> dict:
    """Per-band clip limits (percentiles) and mean / std of the clipped values, from training patches only
    (so validation / test data never leak into preprocessing). Nodata pixels are excluded."""
    nc = cfg.normalization
    out = Path(cfg.paths.processed_dir)
    train = df[df["split"] == "train"]
    if train.empty:
        raise RuntimeError("no training patches: cannot compute normalisation")
    ids = train.sample(min(len(train), int(nc.sample_patches)), random_state=int(cfg.project.seed))["patch_id"]
    samples: list[np.ndarray] = []
    nodata = None
    for pid in ids:
        with rasterio.open(out / "images" / f"{pid}.tif") as s:
            img = s.read().astype(np.float64)
            nodata = s.nodata if s.nodata is not None else 0
            descriptions = list(s.descriptions)
        ok = ~np.all(img == nodata, axis=0)
        if ok.sum() == 0:
            continue
        px = img[:, ok]
        take = np.random.default_rng(0).choice(px.shape[1], min(px.shape[1], 20000), replace=False)
        samples.append(px[:, take])
    if not samples:
        raise RuntimeError("training patches contain no data pixels")
    allpx = np.concatenate(samples, axis=1)
    lo_p, hi_p = nc.clip_percentiles
    lo = np.percentile(allpx, lo_p, axis=1)
    hi = np.percentile(allpx, hi_p, axis=1)
    clipped = np.clip(allpx, lo[:, None], hi[:, None])
    stats = {"bands": json.loads(train["bands"].iloc[0]), "band_names": descriptions, "nodata": nodata,
             "clip_low": lo.tolist(), "clip_high": hi.tolist(), "mean": clipped.mean(axis=1).tolist(),
             "std": np.maximum(clipped.std(axis=1), 1e-6).tolist(), "n_pixels": int(allpx.shape[1]),
             "gsd_m": None}
    first = train["transform"].iloc[0]
    stats["gsd_m"] = abs(json.loads(first)[0])
    write_json(stats, out / "metadata" / "normalization.json")
    log.info("normalisation from %d training pixels: mean %s", allpx.shape[1], np.round(stats["mean"], 1).tolist())
    return stats
