"""Turn class probabilities into one polygon per farm.

1. seeds  = pixels that are confidently farm interior and not boundary
2. each connected seed region becomes one farm (the boundary class keeps neighbours apart)
3. polygons are vectorised, grown back by half the boundary width, de-overlapped and cleaned
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.strtree import STRtree

import geopandas as gpd


def _fill_small_holes(geom, min_area):
    def fix(p: Polygon):
        keep = [r for r in p.interiors if Polygon(r).area >= min_area]
        return Polygon(p.exterior, keep)

    if isinstance(geom, MultiPolygon):
        return MultiPolygon([fix(p) for p in geom.geoms])
    return fix(geom)


def polygonize_probs(cfg, probs_path, progress=print) -> gpd.GeoDataFrame:
    pz = cfg.polygonize
    with rasterio.open(probs_path) as src:
        p = src.read().astype(np.uint8)  # (3,H,W): bg, interior, boundary  (x255)
        transform, crs = src.transform, src.crs
        res = abs(transform.a)
    interior, boundary = p[1], p[2]
    seeds = (interior >= pz.interior_threshold * 255) & (boundary < pz.boundary_threshold * 255)
    seeds = ndi.binary_opening(seeds, structure=np.ones((3, 3), bool))
    labels, n = ndi.label(seeds)
    del seeds
    if n == 0:
        return gpd.GeoDataFrame({"farm_id": [], "area_ha": [], "confidence": []}, geometry=[], crs=crs)

    min_seed_px = max(4, int(0.3 * pz.min_area_ha * 1e4 / res**2))
    sizes = np.bincount(labels.ravel())
    small = sizes < min_seed_px
    small[0] = True
    labels[small[labels]] = 0
    keep_ids = np.flatnonzero(~small)
    progress(f"  {len(keep_ids)} candidate farms from {n} seed regions")
    conf = dict(zip(keep_ids, ndi.mean(interior, labels, keep_ids) / 255.0))

    grow = cfg.labels.boundary_width_px * res / 2  # undo the boundary band that separated seeds
    recs = []
    for geom, val in shapes(labels.astype(np.int32), mask=labels > 0, transform=transform):
        g = shape(geom).buffer(grow, join_style="round")
        if g.is_empty:
            continue
        recs.append((int(val), g))
    del labels

    # de-overlap: larger farms keep contested ground
    recs.sort(key=lambda r: -r[1].area)
    tree = STRtree([g for _, g in recs])
    placed: list = []
    out = []
    for i, (val, g) in enumerate(recs):
        for j in tree.query(g):
            if j < i and placed[j] is not None:
                g = g.difference(placed[j])
        if g.is_empty:
            placed.append(None)
            continue
        placed.append(g)
        out.append((val, g))
    frame = gpd.GeoDataFrame({"seed": [v for v, _ in out]}, geometry=[g for _, g in out], crs=crs)

    frame["geometry"] = frame.geometry.simplify(pz.simplify_m, preserve_topology=True)
    frame["geometry"] = frame.geometry.apply(lambda g: _fill_small_holes(g, pz.fill_holes_ha * 1e4))
    frame = frame.explode(index_parts=False)  # a farm split by a de-overlap cut -> keep largest piece
    frame["area_ha"] = frame.area / 1e4
    frame = frame.sort_values("area_ha", ascending=False).drop_duplicates("seed")
    frame = frame[frame.area_ha >= pz.min_area_ha].copy()
    frame["confidence"] = frame["seed"].map(conf).round(3)
    frame = frame.sort_values(["area_ha"], ascending=False).reset_index(drop=True)
    frame["farm_id"] = np.arange(1, len(frame) + 1)
    frame["area_ha"] = frame["area_ha"].round(2)
    return frame[["farm_id", "area_ha", "confidence", "geometry"]]
