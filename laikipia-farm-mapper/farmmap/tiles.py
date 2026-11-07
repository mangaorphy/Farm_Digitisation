"""Cut training chips from the mosaic and draw 3-class masks from your parcels.

Mask values: 0 background, 1 farm interior, 2 farm boundary, 255 ignore (not learned, not scored).
Ignore = further than labels.context_m from any parcel, parcels failing labels.keep, and image nodata.
The split is by map block (tiles.block_m), so val/test chips come from areas the model never saw.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from .labels import load_labels

BG, INTERIOR, BOUNDARY, IGNORE = 0, 1, 2, 255


def block_split(x: float, y: float, cfg) -> str:
    """Deterministic train/val/test assignment of the map block containing (x, y)."""
    bm = float(cfg.tiles.block_m)
    bx, by = math.floor(x / bm), math.floor(y / bm)
    u = int(hashlib.sha1(f"{bx}_{by}_{cfg.tiles.seed}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    s = cfg.tiles.split
    return "train" if u < s["train"] else ("val" if u < s["train"] + s["val"] else "test")


def blocks_gdf(bounds, cfg) -> gpd.GeoDataFrame:
    """All split blocks covering `bounds` (working CRS), e.g. to evaluate only on test blocks."""
    bm = float(cfg.tiles.block_m)
    rows = []
    for bx in range(math.floor(bounds[0] / bm), math.ceil(bounds[2] / bm)):
        for by in range(math.floor(bounds[1] / bm), math.ceil(bounds[3] / bm)):
            x, y = bx * bm, by * bm
            rows.append({"block": f"{bx}_{by}", "split": block_split(x + bm / 2, y + bm / 2, cfg),
                         "geometry": box(x, y, x + bm, y + bm)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=cfg.aoi.crs)


def make_mask(window_transform, shape, parcels: gpd.GeoDataFrame, context, nodata_mask, cfg) -> np.ndarray:
    """Rasterize one chip's mask. `parcels` already clipped to the chip area (with a margin)."""
    mask = np.full(shape, IGNORE, dtype=np.uint8)
    if context is not None and not context.is_empty:
        mask = rasterize([(context, BG)], out_shape=shape, transform=window_transform, fill=IGNORE, dtype="uint8")
    good, bad = parcels[parcels["train"]], parcels[~parcels["train"]]
    if len(good):
        mask = rasterize([(g, INTERIOR) for g in good.geometry], out=mask, transform=window_transform)
        half = float(cfg.tiles.boundary_width_m) / 2
        edges = good.geometry.boundary.buffer(half, cap_style=2, join_style=2)
        mask = rasterize([(g, BOUNDARY) for g in edges if not g.is_empty], out=mask, transform=window_transform)
    if len(bad):
        mask = rasterize([(g, IGNORE) for g in bad.geometry.buffer(float(cfg.tiles.boundary_width_m))],
                         out=mask, transform=window_transform)
    mask[nodata_mask] = IGNORE
    return mask


def make_tiles(cfg, mosaic: Path | None = None, out_dir: Path | None = None, labels=None) -> pd.DataFrame:
    mosaic = Path(mosaic or cfg.paths.mosaic)
    out_dir = Path(out_dir or cfg.paths.tiles)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    gdf = labels if labels is not None else load_labels(cfg)
    if not len(gdf):
        raise ValueError("No parcels inside the AOI")
    context = gdf.geometry.buffer(float(cfg.labels.context_m)).union_all()
    sindex = gdf.sindex
    size = int(cfg.tiles.size_px)
    rows = []
    with rasterio.open(mosaic) as src:
        gsd = src.transform.a
        tile_m = size * gsd
        L, B, R, T = src.bounds
        cb = context.bounds
        x_start = max(L, math.floor(cb[0] / tile_m) * tile_m)
        y_start = min(T, math.ceil(cb[3] / tile_m) * tile_m)
        xs = np.arange(x_start, min(R, cb[2]), tile_m)
        ys = np.arange(y_start, max(B, cb[1]), -tile_m)
        for y in ys:
            for x in xs:
                tb = box(x, y - tile_m, x + tile_m, y)
                ctx = context.intersection(tb)
                if ctx.area < float(cfg.tiles.min_label_fraction) * tb.area:
                    continue
                win = from_bounds(x, y - tile_m, x + tile_m, y, src.transform).round_offsets().round_lengths()
                win = Window(win.col_off, win.row_off, size, size)
                img = src.read(window=win, boundless=True, fill_value=0)
                nodata = (img == 0).all(axis=0)
                if nodata.mean() > float(cfg.tiles.max_nodata_fraction):
                    continue
                wt = src.window_transform(win)
                near = gdf.iloc[sindex.query(tb.buffer(5), predicate="intersects")]
                mask = make_mask(wt, (size, size), near, ctx, nodata, cfg)
                labelled = float((mask != IGNORE).mean())
                if labelled < float(cfg.tiles.min_label_fraction):
                    continue
                tid = f"t_{int(x)}_{int(y)}"
                split = block_split(x + tile_m / 2, y - tile_m / 2, cfg)
                prof = dict(driver="GTiff", width=size, height=size, crs=src.crs, transform=wt,
                            compress="deflate", predictor=2)
                with rasterio.open(out_dir / "images" / f"{tid}.tif", "w", count=img.shape[0], dtype="uint8",
                                   nodata=0, **prof) as d:
                    d.write(img)
                    d.descriptions = tuple(src.descriptions)
                with rasterio.open(out_dir / "masks" / f"{tid}.tif", "w", count=1, dtype="uint8", nodata=IGNORE,
                                   **prof) as d:
                    d.write(mask, 1)
                rows.append({"id": tid, "split": split, "x": x, "y": y, "labelled": round(labelled, 3),
                             "interior": round(float((mask == INTERIOR).mean()), 3),
                             "boundary": round(float((mask == BOUNDARY).mean()), 3)})
        bands = list(src.descriptions)
        mosaic_bounds = src.bounds
    index = pd.DataFrame(rows)
    if index.empty:
        raise ValueError("No chips were written: does the mosaic overlap your parcels?")
    index.to_csv(out_dir / "index.csv", index=False)
    blocks = blocks_gdf(mosaic_bounds, cfg)
    blocks.to_file(out_dir / "blocks.gpkg", driver="GPKG")
    (out_dir / "meta.yaml").write_text(
        f"gsd_m: {gsd}\nsize_px: {size}\nbands: {bands}\nclasses: [background, interior, boundary]\nignore: 255\n")
    print(f"{len(index)} chips -> {out_dir}  {index.split.value_counts().to_dict()}")
    return index
