"""Automated checks run before training (``python -m src check``; also at the start of ``train``).

Each check returns ok / message / details; a failed check raises ``ValidationError`` (unless asked not to).

1. dimensions   image and every mask have the same size, transform and CRS (full scenes and patches)
2. crs          polygons land on their own interior pixels after reprojection (a wrong CRS or transform puts
                representative points outside the farms)
3. overlap      at least one image is covered by training farms, and every image's report is recorded
4. boundary     outlines of training farms fall on boundary pixels (sampled points on each ring)
5. alignment    in patches, boundary pixels sit on farm edges of the instance mask and vice versa
6. split        train / val / test patch footprints never overlap; patch ids are unique
7. nodata       every image-nodata pixel has valid = 0 (so it is excluded from loss and metrics)
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from shapely.geometry import LineString, box

from ..utils.helpers import write_json
from .ingestion import RasterInfo, nodata_value, select_bands

log = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    pass


def _result(name: str, ok: bool, message: str, **details) -> dict:
    (log.info if ok else log.error)("check %-10s %s  %s", name, "OK  " if ok else "FAIL", message)
    return {"check": name, "ok": bool(ok), "message": message, **details}


def check_dimensions(cfg, rasters: list[RasterInfo], table: pd.DataFrame) -> dict:
    out = Path(cfg.paths.processed_dir)
    bad = []
    for info in rasters:
        for n in ("interior", "boundary", "instance", "valid"):
            p = out / "masks" / info.scene_id / f"{n}.tif"
            if not p.exists():
                bad.append(f"{info.scene_id}: {n}.tif missing")
                continue
            with rasterio.open(p) as m:
                if (m.width, m.height) != (info.width, info.height):
                    bad.append(f"{info.scene_id}/{n}: size {m.width}x{m.height} != image {info.width}x{info.height}")
                if not np.allclose(tuple(m.transform)[:6], info.transform[:6]):
                    bad.append(f"{info.scene_id}/{n}: transform differs from the image")
                if m.crs != rasterio.crs.CRS.from_string(info.crs):
                    bad.append(f"{info.scene_id}/{n}: CRS differs from the image")
    sample = table.sample(min(len(table), 200), random_state=0) if len(table) else table
    for pid in sample["patch_id"]:
        ref = None
        for sub in ("images", "interior_masks", "boundary_masks", "valid_masks", "instance_masks"):
            with rasterio.open(out / sub / f"{pid}.tif") as s:
                key = (s.width, s.height, tuple(round(v, 6) for v in tuple(s.transform)[:6]), s.crs)
            if ref is None:
                ref = key
            elif key != ref:
                bad.append(f"patch {pid}: {sub} grid differs from its image")
    return _result("dimensions", not bad, f"{len(rasters)} scenes, {len(sample)} patches compared"
                   + (f"; {len(bad)} problems" if bad else ""), problems=bad[:50])


def check_crs_alignment(cfg, rasters: list[RasterInfo], labels: gpd.GeoDataFrame, n: int = 300) -> dict:
    out = Path(cfg.paths.processed_dir)
    hits = total = 0
    per_scene = {}
    for info in rasters:
        g = labels[labels["train"]].to_crs(info.crs)
        g = g[g.within(box(*info.bounds))]
        if g.empty:
            continue
        g = g.sample(min(len(g), n), random_state=0)
        pts = g.geometry.representative_point()
        with rasterio.open(out / "masks" / info.scene_id / "instance.tif") as m:
            vals = np.array([v[0] for v in m.sample(list(zip(pts.x, pts.y)))])
        # a point may land on a smaller farm drawn on top (overlapping polygons): count any farm pixel
        ok = vals > 0
        hits += int(ok.sum())
        total += len(ok)
        per_scene[info.scene_id] = round(float(ok.mean()), 4)
    share = hits / total if total else 0.0
    return _result("crs", total > 0 and share >= 0.98,
                   f"{share:.1%} of {total} sampled farm points fall on farm pixels after reprojection",
                   per_scene=per_scene)


def check_overlap(report: dict) -> dict:
    rs = report.get("rasters", [])
    with_farms = [r["scene_id"] for r in rs if r.get("train_polygons_intersecting", 0) > 0]
    without = [r["scene_id"] for r in rs if r.get("train_polygons_intersecting", 0) == 0]
    return _result("overlap", bool(with_farms), f"{len(with_farms)} of {len(rs)} images contain training farms"
                   + (f" (none in: {', '.join(without[:5])})" if without else ""), images_without_farms=without)


def check_boundary_presence(cfg, rasters: list[RasterInfo], labels: gpd.GeoDataFrame, n_polys: int = 200,
                            pts_per_ring: int = 12) -> dict:
    out = Path(cfg.paths.processed_dir)
    hits = total = 0
    empty_scenes = []
    for info in rasters:
        g = labels[labels["train"]].to_crs(info.crs)
        g = g[g.intersects(box(*info.bounds))]
        if g.empty:
            continue
        g = g.sample(min(len(g), n_polys), random_state=1)
        pts = []
        for geom in g.geometry:
            for p in getattr(geom, "geoms", [geom]):
                ring = LineString(p.exterior.coords)
                pts += [ring.interpolate(t, normalized=True) for t in np.linspace(0, 1, pts_per_ring, endpoint=False)]
        x0, y0, x1, y1 = info.bounds
        pts = [p for p in pts if x0 < p.x < x1 and y0 < p.y < y1]
        if not pts:
            continue
        gsd = abs(info.res[0])
        with rasterio.open(out / "masks" / info.scene_id / "boundary.tif") as m:
            # allow the outline to fall within one pixel of a boundary pixel (rasterisation rounding)
            near = []
            for p in pts:
                win = rasterio.windows.from_bounds(p.x - gsd, p.y - gsd, p.x + gsd, p.y + gsd, m.transform)
                arr = m.read(1, window=win.round_offsets().round_lengths(), boundless=True, fill_value=0)
                near.append(arr.any())
        near = np.array(near)
        if not near.any():
            empty_scenes.append(info.scene_id)
        hits += int(near.sum())
        total += len(near)
    share = hits / total if total else 0.0
    return _result("boundary", total > 0 and share >= 0.95 and not empty_scenes,
                   f"{share:.1%} of {total} points sampled on farm outlines lie on boundary pixels",
                   scenes_without_boundaries=empty_scenes)


def check_mask_alignment(cfg, table: pd.DataFrame, n: int = 100) -> dict:
    out = Path(cfg.paths.processed_dir)
    farm = table[table["farm_pixels"] > 0]
    if farm.empty:
        return _result("alignment", False, "no patches with farm pixels")
    on_edge = bnd_total = edge_cov = edge_total = 0
    missing_boundary = []
    for pid in farm.sample(min(len(farm), n), random_state=2)["patch_id"]:
        with rasterio.open(out / "boundary_masks" / f"{pid}.tif") as s:
            bnd = s.read(1).astype(bool)
        with rasterio.open(out / "instance_masks" / f"{pid}.tif") as s:
            inst = s.read(1).astype(np.int64)
            gsd = abs(s.transform.a)
        edges = (inst != ndi.grey_erosion(inst, size=3)) | (inst != ndi.grey_dilation(inst, size=3))
        edges &= ndi.binary_dilation(inst > 0, iterations=1)
        r =max(1, int(np.ceil(float(cfg.masks.boundary_width_m) / 2 / gsd)) + 1)
        near_edge = ndi.binary_dilation(edges, iterations=r)
        near_bnd = ndi.binary_dilation(bnd, iterations=r)
        on_edge += int((bnd & near_edge).sum())
        bnd_total += int(bnd.sum())
        # instance edges at the patch border are artefacts of cutting, not outlines
        core = np.zeros_like(edges)
        core[2:-2, 2:-2] = True
        e = edges & core
        edge_cov += int((e & near_bnd).sum())
        edge_total += int(e.sum())
        if len(np.unique(inst)) > 2 and not bnd.any():
            missing_boundary.append(pid)
    p1 = on_edge / bnd_total if bnd_total else 0.0
    p2 = edge_cov / edge_total if edge_total else 0.0
    return _result("alignment", p1 >= 0.9 and p2 >= 0.9 and not missing_boundary,
                   f"{p1:.1%} of boundary pixels lie on farm edges; {p2:.1%} of farm edges are covered by boundary",
                   patches_with_farms_but_no_boundary=missing_boundary[:20])


def check_split(cfg, table: pd.DataFrame, split_crs: str) -> dict:
    from .patch_extraction import patch_footprints

    if table["patch_id"].duplicated().any():
        return _result("split", False, "duplicate patch ids")
    fp = patch_footprints(table, split_crs)
    regions = {s: g.union_all() for s, g in fp.groupby("split")}
    clashes = {}
    names = sorted(regions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            area = regions[a].intersection(regions[b]).area
            if area > 1.0:
                clashes[f"{a}/{b}"] = round(area, 1)
    counts = table["split"].value_counts().to_dict()
    ok = not clashes and counts.get("train", 0) > 0 and counts.get("val", 0) > 0
    msg = f"patches {counts}; overlapping area between splits: {clashes or 'none'}"
    if counts.get("val", 0) == 0:
        msg += " - no validation patches (too few farm blocks? lower split.block_size_m)"
    return _result("split", ok, msg, overlaps_m2=clashes, patches=counts)


def check_nodata(cfg, table: pd.DataFrame, n: int = 200) -> dict:
    out = Path(cfg.paths.processed_dir)
    leaks = []
    sample = table.sample(min(len(table), n), random_state=3)
    for pid in sample["patch_id"]:
        with rasterio.open(out / "images" / f"{pid}.tif") as s:
            img = s.read()
            nd = s.nodata if s.nodata is not None else 0
        with rasterio.open(out / "valid_masks" / f"{pid}.tif") as s:
            valid = s.read(1).astype(bool)
        nodata = np.all(img == nd, axis=0)
        if np.issubdtype(img.dtype, np.floating):
            nodata |= ~np.all(np.isfinite(img), axis=0)
        if (nodata & valid).any():
            leaks.append(pid)
    return _result("nodata", not leaks, f"{len(sample)} patches: nodata pixels marked valid in {len(leaks)}",
                   patches=leaks[:20])


def run_checks(cfg, labels: gpd.GeoDataFrame, rasters: list[RasterInfo], report: dict, table: pd.DataFrame,
               split_crs: str, raise_on_failure: bool = True) -> list[dict]:
    results = [
        check_dimensions(cfg, rasters, table),
        check_crs_alignment(cfg, rasters, labels),
        check_overlap(report),
        check_boundary_presence(cfg, rasters, labels),
        check_mask_alignment(cfg, table),
        check_split(cfg, table, split_crs),
        check_nodata(cfg, table),
    ]
    path = write_json(results, Path(cfg.paths.processed_dir) / "metadata" / "validation_report.json")
    failed = [r["check"] for r in results if not r["ok"]]
    if failed:
        msg = f"pre-training checks failed: {', '.join(failed)} (details in {path})"
        if raise_on_failure:
            raise ValidationError(msg)
        log.error(msg)
    else:
        log.info("all %d pre-training checks passed (%s)", len(results), path)
    return results
