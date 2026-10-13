"""Rasterise the reference polygons onto the imagery grid.

Label values: 0 background, 1 farm interior, 2 farm boundary, 255 ignore
(no imagery, further than `coverage_buffer_m` from any usable polygon, or a polygon excluded by
`reference.filters` such as a QA-flagged parcel).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

from .imagery import chunk_windows
from .reference import load_reference

IGNORE = 255


def raster_resolution_m(src) -> float:
    """Pixel size in metres. Imagery must be in a projected CRS (e.g. UTM), not lon/lat."""
    if src.crs is None or not src.crs.is_projected:
        raise ValueError(f"{src.name}: imagery must be in a projected CRS in metres (e.g. EPSG:32637). "
                         "Reproject it first, e.g. with `gdalwarp -t_srs EPSG:32637`.")
    return abs(src.transform.a)


def build_labels(cfg, imagery_path, out_path=None, progress=print) -> Path:
    out_path = Path(out_path) if out_path else cfg.path(cfg.labels.dir) / f"{Path(imagery_path).stem}_labels.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(imagery_path) as src:
        crs, transform, width, height = src.crs, src.transform, src.width, src.height
        res = raster_resolution_m(src)
        half_band = cfg.labels.boundary_width_px * res / 2
        ref = load_reference(cfg, crs=crs)
        good = ref[ref["ref_use"]]
        skip = ref[~ref["ref_use"]]
        # draw order: interior, then boundaries, then excluded parcels (ignored)
        interior = list(good.geometry)
        bnd = good.geometry.boundary.buffer(half_band)
        coverage = good.geometry.buffer(cfg.reference.coverage_buffer_m)
        tree_i, tree_b, tree_c = good.geometry.sindex, bnd.sindex, coverage.sindex
        bnd, coverage = list(bnd), list(coverage)
        skip_geoms = list(skip.geometry)
        tree_s = skip.geometry.sindex if len(skip) else None

        profile = dict(driver="GTiff", width=width, height=height, count=1, dtype="uint8", crs=crs,
                       transform=transform, nodata=IGNORE, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate")
        wins = list(chunk_windows(width, height, 2048))
        with rasterio.open(out_path, "w", **profile) as dst:
            for n, win in enumerate(wins):
                wt = rasterio.windows.transform(win, transform)
                shape = (int(win.height), int(win.width))
                region = box(*rasterio.windows.bounds(win, transform))
                valid = (src.read(window=win) > 0).any(0)

                lab = np.full(shape, IGNORE, np.uint8)
                idx = tree_c.query(region, predicate="intersects")
                if len(idx):
                    cov = rasterize([(coverage[i], 1) for i in idx], shape, transform=wt, fill=0, dtype="uint8")
                    lab[cov == 1] = 0
                idx = tree_i.query(region, predicate="intersects")
                if len(idx):
                    ins = rasterize([(interior[i], 1) for i in idx], shape, transform=wt, fill=0, dtype="uint8")
                    lab[ins == 1] = 1
                idx = tree_b.query(region, predicate="intersects")
                if len(idx):
                    edge = rasterize([(bnd[i], 1) for i in idx], shape, transform=wt, fill=0, dtype="uint8")
                    lab[edge == 1] = 2
                if tree_s is not None:
                    idx = tree_s.query(region, predicate="intersects")
                    if len(idx):
                        sk = rasterize([(skip_geoms[i], 1) for i in idx], shape, transform=wt, fill=0, dtype="uint8")
                        lab[sk == 1] = IGNORE
                lab[~valid] = IGNORE
                dst.write(lab, 1, window=win)
                if (n + 1) % 10 == 0 or n + 1 == len(wins):
                    progress(f"  labels {n + 1}/{len(wins)}")
    return out_path
