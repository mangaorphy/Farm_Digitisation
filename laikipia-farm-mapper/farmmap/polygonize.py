"""Probabilities → one polygon per farm.

Seeds = connected areas where P(interior) − P(boundary) is high; each seed grows over "inside some farm"
pixels (P(interior)+P(boundary)) by watershed on P(boundary), so neighbouring farms split along the
predicted boundary. Runs in blocks with a halo; farms cut by a block edge are merged afterwards.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from scipy import ndimage as ndi
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from skimage.segmentation import watershed
from tqdm import tqdm


def _fill_small_holes(geom, max_area):
    def fix(p: Polygon):
        return Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area >= max_area])
    if geom.geom_type == "Polygon":
        return fix(geom)
    return unary_union([fix(p) for p in geom.geoms])


def segment(p_int: np.ndarray, p_bnd: np.ndarray, pc, gsd: float) -> np.ndarray:
    """Instance label image (0 = background) from interior/boundary probabilities in [0, 1]."""
    min_px = max(4, int(float(pc.min_area_m2) / gsd ** 2) // 4)
    markers, _ = ndi.label((p_int - p_bnd) > float(pc.marker_threshold))
    sizes = np.bincount(markers.ravel())
    small = sizes < min_px
    small[0] = False
    markers[small[markers]] = 0  # tiny seeds would split one farm into many
    mask = (p_int + p_bnd) > float(pc.mask_threshold)
    labels = watershed(p_bnd, markers, mask=mask)
    return labels


def polygonize(cfg, prob_path: Path, out_path: Path | None = None) -> gpd.GeoDataFrame:
    pc = cfg.polygonize
    out_path = Path(out_path or Path(cfg.paths.outputs) / "farms.gpkg")
    block, halo = int(pc.block_px), int(pc.halo_px)
    recs = []
    with rasterio.open(prob_path) as src:
        gsd = src.transform.a
        jobs = [(r, c) for r in range(0, src.height, block) for c in range(0, src.width, block)]
        for r, c in tqdm(jobs, desc="polygonize", unit="block"):
            h, w = min(block, src.height - r), min(block, src.width - c)
            win = Window(c - halo, r - halo, w + 2 * halo, h + 2 * halo).intersection(
                Window(0, 0, src.width, src.height))
            arr = src.read([2, 3], window=win)
            if not arr.any():
                continue
            valid = arr[0] > 0
            p_int = np.where(valid, (arr[0].astype(np.float32) - 1) / 254, 0)
            p_bnd = np.where(valid, (arr[1].astype(np.float32) - 1) / 254, 0)
            lab = segment(p_int, p_bnd, pc, gsd)
            if lab.max() == 0:
                continue
            wt = src.window_transform(win)
            core = box(*rasterio.windows.bounds(Window(c, r, w, h), src.transform))
            # a farm touching an edge of the read window that is not the raster's own edge may be cut off
            inner_edges = box(*rasterio.windows.bounds(win, src.transform)).boundary.difference(
                box(*src.bounds).boundary.buffer(gsd))
            counts = np.bincount(lab.ravel())
            scores = np.bincount(lab.ravel(), weights=p_int.ravel()) / np.maximum(counts, 1)
            for geom, val in shapes(lab.astype(np.int32), mask=lab > 0, transform=wt):
                g = shape(geom)
                if not g.intersects(core):
                    continue  # entirely in the halo: its own block reports it
                margin = g.distance(inner_edges) if not inner_edges.is_empty else np.inf
                recs.append({"geometry": g, "score": float(scores[int(val)]), "margin": margin,
                             "cut": bool(margin <= gsd)})
        crs = src.crs
    gdf = gpd.GeoDataFrame(_dedupe(recs) or {"geometry": [], "score": []}, geometry="geometry", crs=crs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        empty = gpd.GeoDataFrame({"farm_id": [], "score": [], "area_ha": []}, geometry=[], crs=crs)
        empty.to_file(out_path, driver="GPKG", layer="farms")  # written anyway so evaluate/export still run
        print(f"No farms found -> empty {out_path}")
        return empty
    gdf["geometry"] = gdf.geometry.buffer(0).simplify(float(pc.simplify_m), preserve_topology=True)
    gdf["geometry"] = gdf.geometry.apply(lambda g: _fill_small_holes(g, float(pc.hole_area_m2)))
    gdf = gdf[gdf.area >= float(pc.min_area_m2)].copy()
    gdf = context_filter(cfg, gdf)
    gdf = gdf.sort_values("score", ascending=False).reset_index(drop=True)
    gdf.insert(0, "farm_id", [f"AUTO-{i + 1:06d}" for i in range(len(gdf))])
    gdf["area_ha"] = (gdf.area / 1e4).round(3)
    gdf["score"] = gdf.score.round(3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer="farms")
    print(f"{len(gdf)} farm polygons -> {out_path}")
    return gdf


def context_filter(cfg, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Optional: drop farms whose mean of a context band is below a minimum, e.g. not cropland at all.

    polygonize.context_filter: {band: "ctx:dea:crop_prob", min: 10}   (physical units; null = off)
    Farms with no context value (outside the context grid) are kept.
    """
    cf = cfg.polygonize.get("context_filter")
    if not cf or gdf.empty:
        return gdf
    from . import context as CTX
    source = cf["band"].split(":")[1]
    vals = CTX.zonal_mean(gdf, [Path(cfg.paths.context) / f"{source}.tif"], cf["band"])
    keep = ~(vals < float(cf["min"]))
    print(f"context filter {cf['band']} >= {cf['min']}: dropped {int((~keep).sum())} of {len(gdf)} polygons")
    return gdf.assign(ctx_value=np.round(vals, 3))[keep].copy()


def _dedupe(recs: list[dict], overlap: float = 0.3) -> list[dict]:
    """The same farm is seen by every block whose window reaches it; keep one copy.

    Copies are taken greedily, most-centred first (largest distance to a cut window edge = most context).
    A window that cut a farm's seed off floods that farm into a neighbour, giving inflated pieces at the
    edge; those come last and are dropped when they overlap an accepted farm. Only when a farm was cut in
    every window (bigger than block + halo) are its cut pieces unioned.
    """
    if not recs:
        return []
    g = gpd.GeoDataFrame(recs, geometry="geometry")
    g["area"] = g.area
    order = g.sort_values(["margin", "area"], ascending=False).index
    sindex = g.sindex
    accepted: dict[int, dict] = {}  # rec index -> output record
    owner = {}  # rec index -> accepted index it was merged into
    for i in order:
        geom = g.geometry[i]
        hits = [owner[j] for j in sindex.query(geom, predicate="intersects") if j in owner]
        clash = None
        for a in dict.fromkeys(hits):
            other = accepted[a]["geometry"]
            if geom.intersection(other).area > overlap * min(geom.area, other.area):
                clash = a
                break
        if clash is None:
            accepted[i] = {"geometry": geom, "score": g.score[i], "cut": bool(g.cut[i])}
            owner[i] = i
        elif g.cut[i] and accepted[clash]["cut"]:
            acc = accepted[clash]
            acc["geometry"] = unary_union([acc["geometry"], geom])
            owner[i] = clash
    return [{"geometry": r["geometry"], "score": r["score"]} for r in accepted.values()]
