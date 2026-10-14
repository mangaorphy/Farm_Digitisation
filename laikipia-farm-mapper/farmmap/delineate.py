"""Delineate Anything (YOLO instance segmentation, trained on 0.25-10 m imagery) as a polygon engine.

Model: https://github.com/Lavreniuk/Delineate-Anything  (code and weights AGPL-3.0)
Install the extras first:  pip install -r requirements-da.txt

It takes RGB and returns one polygon per detected field directly. For Sentinel-2 the image is
resampled up before detection (`delineate.upsample`); the best factor is data dependent, so
`python -m farmmap compare` scores several on your own test blocks.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .imagery import chunk_windows
from .labels import raster_resolution_m
from .model import pick_device
from .scale import to_unit


def rgb_positions(cfg):
    """0-based raster positions of (R, G, B)."""
    names = cfg.imagery.bands if cfg.imagery.source == "planetary_computer" else cfg.imagery.get("local_bands")
    if not names:
        raise ValueError("Band names unknown: set imagery.local_bands (e.g. [R, G, B, NIR]) in config.yaml.")
    names = [str(n).upper() for n in names]
    for trio in (("B04", "B03", "B02"), ("R", "G", "B"), ("RED", "GREEN", "BLUE")):
        if all(t in names for t in trio):
            return [names.index(t) for t in trio]
    raise ValueError(f"Could not find red/green/blue among bands {names}.")


def ensure_weights(cfg, progress=print) -> Path:
    dc = cfg.delineate
    path = cfg.path(dc.weights)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Downloading Delineate Anything weights (~125 MB) to {path} ...")
    tmp = path.with_suffix(".part")
    req = urllib.request.Request(dc.weights_url, headers={"User-Agent": "laikipia-farm-mapper/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def make_predictor(cfg, progress=print):
    """Returns f(img_rgb_uint8[H,W,3]) -> [(xy[N,2] in image pixels, confidence)]."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Delineate Anything needs ultralytics: pip install -r requirements-da.txt") from exc
    dc = cfg.delineate
    model = YOLO(str(ensure_weights(cfg, progress)))
    dev = pick_device(cfg.train.device)
    device = 0 if dev == "cuda" else dev

    def predict(img_rgb):
        res = model.predict(np.ascontiguousarray(img_rgb[..., ::-1]),  # ultralytics expects BGR arrays
                            imgsz=dc.imgsz, conf=dc.conf, iou=dc.iou, retina_masks=True,
                            device=device, verbose=False)[0]
        if res.masks is None:
            return []
        confs = res.boxes.conf.cpu().numpy()
        return [(np.asarray(xy), float(c)) for xy, c in zip(res.masks.xy, confs) if len(xy) >= 3]

    return predict


def _stretch(src, idx, pct, sample_px=2_000_000):
    import math

    from rasterio.enums import Resampling

    s = max(1, int(math.sqrt(src.width * src.height / sample_px)))
    d = to_unit(src.read([i + 1 for i in idx], out_shape=(3, max(1, src.height // s), max(1, src.width // s)),
                         resampling=Resampling.nearest))
    v = d[:, (d > 0).any(0)]
    lo, hi = np.percentile(v, pct)
    return float(lo), float(max(hi, lo + 1e-3))


def delineate(cfg, imagery_path, regions=None, upsample=None, predictor=None, progress=print) -> gpd.GeoDataFrame:
    """Polygons for the imagery (or only for `regions`, a list of Windows), in the imagery CRS."""
    dc = cfg.delineate
    up = int(upsample or dc.upsample)
    idx = rgb_positions(cfg)
    predictor = predictor or make_predictor(cfg, progress)
    pad = int(dc.overlap_px) // 2

    geoms, confs = [], []
    with rasterio.open(imagery_path) as src:
        res = raster_resolution_m(src)
        lo, hi = _stretch(src, idx, dc.stretch_percentiles)
        regions = regions or [Window(0, 0, src.width, src.height)]
        tiles = []
        for r in regions:
            for w in chunk_windows(int(r.width), int(r.height), int(dc.tile_px)):
                tiles.append(Window(r.col_off + w.col_off, r.row_off + w.row_off, w.width, w.height))
        t = src.transform
        coeffs = [t.a, t.b, t.d, t.e, t.c, t.f]
        for n, w in enumerate(tiles):
            big = Window(w.col_off - pad, w.row_off - pad, w.width + 2 * pad, w.height + 2 * pad)
            arr = to_unit(src.read([i + 1 for i in idx], window=big, boundless=True, fill_value=0))
            if (arr > 0).any(0).mean() < 0.05:
                continue
            img = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
            if up > 1:
                img = np.clip(ndi.zoom(img.astype("float32"), (up, up, 1), order=3), 0, 255).astype(np.uint8)
            for xy, conf in predictor(img):
                pts = xy / up + np.array([big.col_off, big.row_off])  # native pixel coordinates
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue
                c = poly.centroid
                # a tile owns only polygons whose centre is in its core, so overlaps do not duplicate
                if not (w.col_off <= c.x < w.col_off + w.width and w.row_off <= c.y < w.row_off + w.height):
                    continue
                geoms.append(affine_transform(poly, coeffs))
                confs.append(conf)
            if (n + 1) % 25 == 0 or n + 1 == len(tiles):
                progress(f"  Delineate Anything: {n + 1}/{len(tiles)} tiles, {len(geoms)} polygons")
        crs = src.crs

    frame = gpd.GeoDataFrame({"confidence": confs}, geometry=geoms, crs=crs)
    if len(frame):
        frame = _dedupe(frame)
        frame["geometry"] = frame.geometry.simplify(res * 0.25, preserve_topology=True)
        frame["area_ha"] = (frame.area / 1e4).round(3)
        frame = frame[frame.area_ha >= float(dc.get("min_area_ha", cfg.polygonize.min_area_ha))]
    frame = frame.reset_index(drop=True)
    frame.insert(0, "farm_id", np.arange(1, len(frame) + 1))
    return frame[["farm_id", "area_ha", "confidence", "geometry"]] if len(frame) else frame


def _dedupe(frame: gpd.GeoDataFrame, overlap=0.5) -> gpd.GeoDataFrame:
    """Drop a polygon when >50% of it lies inside a more confident one."""
    frame = frame.sort_values("confidence", ascending=False).reset_index(drop=True)
    g = list(frame.geometry)
    tree = STRtree(g)
    drop = set()
    for i, p in enumerate(g):
        if i in drop:
            continue
        for j in tree.query(p):
            j = int(j)
            if j > i and j not in drop and p.intersection(g[j]).area > overlap * g[j].area:
                drop.add(j)
    return frame.drop(index=list(drop)).reset_index(drop=True)
