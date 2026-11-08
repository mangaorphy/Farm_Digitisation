"""Task 8 (GIS part): interior + boundary probabilities -> one polygon per farm.

Connected components of the interior mask alone merge neighbouring farms wherever a boundary is thin or
faint. Here the boundary prediction is a separation constraint:

1. extent  = P(interior) > interior_threshold                       (where farms are)
2. seeds   = P(interior) * (1 - P(boundary)) > seed_threshold and P(boundary) < boundary_threshold,
             labelled with 4-connectivity (a diagonal 1-pixel boundary still separates two seeds),
             seeds smaller than min_seed_px dropped (they would split one farm in two)
3. watershed on the boundary probability, from the seeds, inside the extent: every extent pixel joins the
   seed it reaches without crossing a boundary ridge, so farms split along the predicted boundary
4. contours of each region -> polygons (rasterio.features.shapes), simplified, small holes filled,
   tiny polygons dropped; area and perimeter in a projected CRS.

Large scenes are processed in blocks with a halo; a farm seen by several blocks is kept once.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.features import shapes
from rasterio.windows import Window
from scipy import ndimage as ndi
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from skimage.segmentation import watershed
from tqdm import tqdm

log = logging.getLogger(__name__)
FOUR = ndi.generate_binary_structure(2, 1)


def segment(p_int: np.ndarray, p_bnd: np.ndarray, pc, gsd: float, p_dist: np.ndarray | None = None) -> np.ndarray:
    """Instance label image (0 = no farm) from interior / boundary (/ distance) probabilities in [0, 1].

    ``postprocess.seed_source``: ``boundary`` = P(int)*(1-P(bnd)); ``distance`` = predicted distance to the
    farm outline (needs a distance head); ``combined`` = mean of the two. Regions smaller than
    ``postprocess.min_area_m2`` are removed, so patch-level evaluation and full-scene mapping agree."""
    extent = p_int > float(pc.interior_threshold)
    source = pc.get("seed_source", "boundary") if p_dist is not None else "boundary"
    boundary_score = p_int * (1.0 - p_bnd)
    if source == "distance":
        seed_score = p_dist
    elif source == "combined":
        seed_score = 0.5 * (boundary_score + p_dist)
    else:
        seed_score = boundary_score
    seeds = (seed_score > float(pc.seed_threshold)) & (p_bnd < float(pc.boundary_threshold)) & extent
    markers, n = ndi.label(seeds, structure=FOUR)
    if n == 0:
        return np.zeros(p_int.shape, np.int32)
    sizes = np.bincount(markers.ravel())
    small = sizes < int(pc.min_seed_px)
    small[0] = False
    markers[small[markers]] = 0
    # flood over boundary ridges; with a distance map, also downhill from farm centres
    elevation = p_bnd if p_dist is None or source == "boundary" else 0.5 * (p_bnd + (1.0 - p_dist))
    labels = watershed(elevation, markers, mask=extent, connectivity=1).astype(np.int32)
    min_px = int(float(pc.get("min_area_m2", 0)) / (gsd * gsd))
    if min_px > 1 and labels.max():
        counts = np.bincount(labels.ravel())
        tiny = counts < min_px
        tiny[0] = False
        labels[tiny[labels]] = 0
    return labels


def _fill_small_holes(geom, max_area: float):
    def fix(p: Polygon) -> Polygon:
        return Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area >= max_area])
    if geom.geom_type == "Polygon":
        return fix(geom)
    return unary_union([fix(p) for p in geom.geoms])


def _dedupe(recs: list[dict], overlap: float = 0.3) -> list[dict]:
    """Keep one copy of each farm seen by several blocks: most-centred copy first (furthest from a cut window
    edge); later copies overlapping an accepted one are dropped, except that pieces of a farm cut in EVERY
    window it appears in are unioned."""
    if not recs:
        return []
    g = gpd.GeoDataFrame(recs, geometry="geometry")
    g["area"] = g.area
    order = g.sort_values(["margin", "area"], ascending=False).index
    sindex = g.sindex
    accepted: dict[int, dict] = {}
    owner: dict[int, int] = {}
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
            accepted[i] = {k: g.at[i, k] for k in g.columns if k not in ("area", "margin")}
            owner[i] = i
        elif g.at[i, "cut"] and accepted[clash]["cut"]:
            acc = accepted[clash]
            acc["geometry"] = unary_union([acc["geometry"], geom])
            owner[i] = clash
    return [{k: v for k, v in r.items() if k != "cut"} for r in accepted.values()]


def polygonize(cfg, prob_path: Path, source_name: str | None = None) -> gpd.GeoDataFrame:
    """Probability GeoTIFF (band 1 interior, band 2 boundary, optional band 3 distance; uint8 1..255,
    0 = nodata) -> farm polygons
    in the raster CRS, with farm_id, score, boundary_score, area_m2, area_ha, perimeter_m, source_image."""
    pc = cfg.postprocess
    block, halo = int(pc.block_px), int(pc.halo_px)
    recs: list[dict] = []
    with rasterio.open(prob_path) as src:
        if src.count < 2:
            raise ValueError(f"{prob_path}: expected 2 bands (interior, boundary), found {src.count}")
        gsd = abs(src.transform.a)
        crs = src.crs
        full = Window(0, 0, src.width, src.height)
        jobs = [(r, c) for r in range(0, src.height, block) for c in range(0, src.width, block)]
        for r, c in tqdm(jobs, desc="polygonize", unit="block", leave=False):
            h, w = min(block, src.height - r), min(block, src.width - c)
            win = Window(c - halo, r - halo, w + 2 * halo, h + 2 * halo).intersection(full)
            arr = src.read(list(range(1, min(src.count, 3) + 1)), window=win)
            if not arr.any():
                continue
            ok = arr[0] > 0
            p_int = np.where(ok, (arr[0].astype(np.float32) - 1) / 254, 0)
            p_bnd = np.where(ok, (arr[1].astype(np.float32) - 1) / 254, 1)
            p_dist = np.where(ok, (arr[2].astype(np.float32) - 1) / 254, 0) if arr.shape[0] > 2 else None
            lab = segment(p_int, p_bnd, pc, gsd, p_dist)
            if lab.max() == 0:
                continue
            wt = src.window_transform(win)
            core = box(*rasterio.windows.bounds(Window(c, r, w, h), src.transform))
            inner_edges = box(*rasterio.windows.bounds(win, src.transform)).boundary.difference(
                box(*src.bounds).boundary.buffer(gsd))
            counts = np.bincount(lab.ravel())
            s_int = np.bincount(lab.ravel(), weights=p_int.ravel()) / np.maximum(counts, 1)
            # mean boundary probability on each region's own outline pixels: how clearly it is separated
            edge = (lab != ndi.grey_erosion(lab, footprint=FOUR)) | (lab != ndi.grey_dilation(lab, footprint=FOUR))
            s_bnd = np.bincount(lab[edge].ravel(), weights=p_bnd[edge].ravel(), minlength=len(counts)) / \
                np.maximum(np.bincount(lab[edge].ravel(), minlength=len(counts)), 1)
            for geom, val in shapes(lab, mask=lab > 0, transform=wt, connectivity=4):
                g = shape(geom)
                if not g.intersects(core):
                    continue
                margin = g.distance(inner_edges) if not inner_edges.is_empty else np.inf
                recs.append({"geometry": g, "score": float(s_int[int(val)]),
                             "boundary_score": float(s_bnd[int(val)]), "margin": margin, "cut": bool(margin <= gsd)})
    kept = _dedupe(recs)
    cols = ["farm_id", "score", "boundary_score", "area_m2", "area_ha", "perimeter_m", "source_image"]
    if not kept:
        log.warning("%s: no farms found", prob_path)
        return gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame(kept, geometry="geometry", crs=crs)
    gdf["geometry"] = gdf.geometry.buffer(0).simplify(float(pc.simplify_m), preserve_topology=True)
    gdf["geometry"] = gdf.geometry.apply(lambda g: _fill_small_holes(g, float(pc.hole_area_m2)))
    metric = gdf if CRS.from_user_input(crs).is_projected else gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area_m2"] = metric.area.round(1).values
    gdf["perimeter_m"] = metric.length.round(1).values
    gdf = gdf[gdf["area_m2"] >= float(pc.min_area_m2)].copy()
    gdf = gdf.sort_values("score", ascending=False).reset_index(drop=True)
    gdf["area_ha"] = (gdf["area_m2"] / 1e4).round(4)
    gdf["score"] = gdf["score"].round(3)
    gdf["boundary_score"] = gdf["boundary_score"].round(3)
    stem = source_name or Path(prob_path).stem.replace("_prob", "")
    gdf["source_image"] = stem
    gdf.insert(0, "farm_id", [f"AUTO-{i + 1:06d}" for i in range(len(gdf))])
    log.info("%s: %d farm polygons", stem, len(gdf))
    return gdf[cols + ["geometry"]]


def merge_scenes(frames: list[gpd.GeoDataFrame], overlap: float = 0.3) -> gpd.GeoDataFrame:
    """Farms from several (possibly overlapping) scenes -> one layer in a metric CRS. Where scenes overlap, a farm
    found in both is kept once: higher score first, a later polygon overlapping a kept one by more than
    ``overlap`` of the smaller area is dropped. Farm ids are renumbered."""
    import pandas as pd

    frames = [f for f in frames if len(f)]
    if not frames:
        return gpd.GeoDataFrame({"farm_id": []}, geometry=[], crs="EPSG:4326")
    g = gpd.GeoDataFrame(pd.concat([f.to_crs("EPSG:4326") for f in frames], ignore_index=True), crs="EPSG:4326")
    g = g.to_crs(g.estimate_utm_crs()).sort_values("score", ascending=False).reset_index(drop=True)
    keep, sindex = np.ones(len(g), bool), g.sindex
    for i, geom in enumerate(g.geometry):
        if not keep[i]:
            continue
        for j in sindex.query(geom, predicate="intersects"):
            if j > i and keep[j]:
                other = g.geometry.iloc[j]
                if geom.intersection(other).area > overlap * min(geom.area, other.area):
                    keep[j] = False
    out = g[keep].reset_index(drop=True)
    out["farm_id"] = [f"AUTO-{i + 1:06d}" for i in range(len(out))]
    log.info("merged %d scenes: %d polygons -> %d after removing duplicates in overlaps", len(frames), len(g), len(out))
    return out


def export_polygons(gdf: gpd.GeoDataFrame, outputs_dir: Path, name: str) -> dict[str, Path]:
    """GeoJSON (EPSG:4326, RFC 7946), Shapefile and GeoPackage (projected CRS, for area work in QGIS) and KMZ
    (Google Earth Pro)."""
    from .kmz import to_kmz

    outputs_dir = Path(outputs_dir)
    paths = {"geojson": outputs_dir / "geojson" / f"{name}.geojson",
             "shapefile": outputs_dir / "shapefiles" / f"{name}.shp",
             "gpkg": outputs_dir / "gpkg" / f"{name}.gpkg",
             "kmz": outputs_dir / "kmz" / f"{name}.kmz"}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    projected = gdf if gdf.crs is not None and CRS.from_user_input(gdf.crs).is_projected else \
        gdf.to_crs(gdf.estimate_utm_crs()) if len(gdf) else gdf
    wgs = gdf.to_crs("EPSG:4326") if len(gdf) else gdf.set_crs("EPSG:4326", allow_override=True)
    if len(wgs):
        c = projected.geometry.centroid.to_crs("EPSG:4326")
        wgs = wgs.assign(lon=c.x.round(6).values, lat=c.y.round(6).values)
    wgs.to_file(paths["geojson"], driver="GeoJSON")
    # shapefile field names are limited to 10 characters
    projected.rename(columns={"boundary_score": "bnd_score", "perimeter_m": "perim_m", "source_image": "source"}) \
        .to_file(paths["shapefile"], driver="ESRI Shapefile")
    projected.to_file(paths["gpkg"], driver="GPKG", layer="farms")
    to_kmz(gdf, paths["kmz"], name=f"Auto-detected farms - {name}")
    log.info("exported %d polygons -> %s", len(gdf), {k: str(v) for k, v in paths.items()})
    return paths
