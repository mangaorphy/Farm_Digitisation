"""Task 2: target masks on the exact grid (CRS, transform, size) of every image.

Four single-band GeoTIFFs per image, written block by block (no full-scene arrays in memory):

* ``interior.tif``  uint8  1 = inside a training farm polygon (farm extent), 0 = not.
* ``boundary.tif``  uint8  1 = on a farm outline (exterior and, optionally, hole rings), 0 = not.
* ``instance.tif``  uint32 farm ``instance_id`` (0 = none): the identity of every farm pixel, for polygon
  reconstruction checks and object-level evaluation. Smaller farms are drawn last so they stay visible.
* ``valid.tif``     uint8  1 = pixel may be used in the loss and metrics. 0 for image nodata, cloud/haze
  (UDM2 not "clear"), QA-flagged (ignored) farms, and - with ``masks.label_region: buffer`` - pixels further
  than ``label_region_buffer_m`` from any digitised farm, where "no polygon" does not mean "no farm".

Interior and boundary are rasterised INDEPENDENTLY from the polygons. Nothing is eroded: a boundary shared by
two adjacent farms is drawn from both outlines and the interior mask keeps every farm pixel, so the two heads
are trained as overlapping (multi-label) targets.

Boundary width. ``masks.boundary_width_m`` is the TOTAL width of the band; the buffer RADIUS applied to the
outline is half of it. At 3 m GSD, 3 m total = 1 pixel, 6 m total = 2 pixels. When the total width is at most
one pixel the outline itself is rasterised with ``all_touched`` ("line" mode), which gives a continuous 1-pixel
line. Buffering a line by 1.5 m and sampling pixel centres would leave gaps wherever the line passes between
pixel centres.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString, box

from .ingestion import RasterInfo, nodata_value, select_bands

log = logging.getLogger(__name__)

MASK_NAMES = ("interior", "boundary", "instance", "valid")


@dataclass
class BoundarySpec:
    """How outlines become boundary pixels. ``total_width_m`` = 2 x ``buffer_radius_m``."""
    total_width_m: float
    buffer_radius_m: float
    width_px: float
    mode: str  # "line" or "buffer"

    @classmethod
    def from_config(cls, width_m: float, pixel_size_m: float, mode: str = "auto") -> "BoundarySpec":
        width_m = float(width_m)
        if width_m <= 0:
            raise ValueError("masks.boundary_width_m must be > 0")
        if mode == "auto":
            mode = "line" if width_m <= pixel_size_m * 1.01 else "buffer"
        if mode not in ("line", "buffer"):
            raise ValueError(f"masks.boundary_mode must be auto, line or buffer, not {mode!r}")
        return cls(total_width_m=width_m, buffer_radius_m=width_m / 2.0, width_px=width_m / pixel_size_m, mode=mode)


def outline(geom, include_holes: bool = True):
    """Exterior (and hole) rings of a Polygon / MultiPolygon as lines."""
    lines = []
    for p in getattr(geom, "geoms", [geom]):
        lines.append(LineString(p.exterior.coords))
        if include_holes:
            lines.extend(LineString(r.coords) for r in p.interiors)
    return lines[0] if len(lines) == 1 else MultiLineString(lines)


def boundary_shapes(geoms, spec: BoundarySpec, include_holes: bool = True) -> list:
    """Geometries to burn as boundary pixels: the outlines ("line") or the outlines buffered by the radius."""
    lines = [outline(g, include_holes) for g in geoms]
    if spec.mode == "line":
        return lines
    # flat caps / mitred joins keep the band exactly radius wide on both sides of the outline
    return [ln.buffer(spec.buffer_radius_m, cap_style="flat", join_style="mitre") for ln in lines]


def _read_valid_data(src, udm, window: Window, bands: list[int], nodata: float) -> np.ndarray:
    """True where the image has data in the selected bands (and UDM2 says clear)."""
    img = src.read(bands, window=window, boundless=True, fill_value=nodata)
    ok = ~np.all(img == nodata, axis=0)
    if np.issubdtype(img.dtype, np.floating):
        ok &= np.all(np.isfinite(img), axis=0)
    if udm is not None:
        clear = udm.read(1, window=window, boundless=True, fill_value=0)
        ok &= clear == 1
    return ok


def _complete_areas(cfg, crs) -> gpd.GeoSeries | None:
    path = cfg.paths.get("complete_areas")
    if not path:
        return None
    if not Path(path).exists():
        raise FileNotFoundError(f"paths.complete_areas not found: {path}")
    from .ingestion import read_vector_file
    g, _ = read_vector_file(Path(path))
    return g.to_crs(crs).geometry


def build_masks(cfg, info: RasterInfo, polys: gpd.GeoDataFrame, out_dir: Path) -> dict:
    """Write the four masks for one image. ``polys`` must already be in the image CRS (see
    ``ingestion.polygons_for_raster``). Returns statistics; mask paths are ``out_dir/<name>.tif``."""
    mc = cfg.masks
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gsd = float(abs(info.res[0]))
    spec = BoundarySpec.from_config(mc.boundary_width_m, gsd, mc.get("boundary_mode", "auto"))
    bands = select_bands(info, cfg.imagery.get("bands"))
    nodata = nodata_value(info, cfg)

    train = polys[polys["train"]].copy()
    ignored = polys[~polys["train"]].copy()
    # big first so that small farms are drawn on top of any overlapping big one in the instance mask
    train = train.assign(_area=train.area).sort_values("_area", ascending=False)
    train_bnd = gpd.GeoDataFrame({"geometry": boundary_shapes(train.geometry, spec, mc.get("include_holes", True))},
                                 geometry="geometry", crs=polys.crs) if len(train) else train
    region_mode = mc.get("label_region", "buffer")
    if region_mode not in ("buffer", "all"):
        raise ValueError(f"masks.label_region must be buffer or all, not {region_mode!r}")
    region = gpd.GeoDataFrame(geometry=polys.geometry.buffer(float(mc.label_region_buffer_m)), crs=polys.crs) \
        if region_mode == "buffer" else None
    complete = _complete_areas(cfg, polys.crs)
    ign = gpd.GeoDataFrame(geometry=ignored.geometry.buffer(float(mc.get("ignore_buffer_m", 0))), crs=polys.crs)

    profile = dict(driver="GTiff", width=info.width, height=info.height, count=1, crs=info.crs,
                   transform=info.affine, tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                   BIGTIFF="IF_SAFER")
    paths = {n: out_dir / f"{n}.tif" for n in MASK_NAMES}
    dtypes = {"interior": "uint8", "boundary": "uint8", "instance": "uint32", "valid": "uint8"}
    stats = {k: 0 for k in ("interior_px", "boundary_px", "boundary_inside_interior_px", "valid_px",
                            "image_nodata_px", "ignored_px", "total_px")}
    block = int(mc.get("block_px", 2048))
    dsts = {n: rasterio.open(paths[n], "w", dtype=dtypes[n], nodata=None, **profile) for n in MASK_NAMES}
    try:
        with rasterio.open(info.path) as src:
            udm = rasterio.open(info.udm2_path) if info.udm2_path else None
            try:
                for r in range(0, info.height, block):
                    for c in range(0, info.width, block):
                        win = Window(c, r, min(block, info.width - c), min(block, info.height - r))
                        shape = (int(win.height), int(win.width))
                        wt = src.window_transform(win)
                        bb = box(*rasterio.windows.bounds(win, info.affine))
                        pad = bb.buffer(spec.total_width_m + gsd * 2)

                        def pick(frame):
                            if frame is None or not len(frame):
                                return []
                            return list(frame.geometry.values[frame.sindex.query(pad, predicate="intersects")])

                        def pick_rows(frame):
                            return frame.iloc[frame.sindex.query(pad, predicate="intersects")] if len(frame) else frame

                        t_rows = pick_rows(train)
                        interior = rasterize([(g, 1) for g in t_rows.geometry], out_shape=shape, transform=wt,
                                             fill=0, dtype="uint8") if len(t_rows) else np.zeros(shape, np.uint8)
                        instance = rasterize(list(zip(t_rows.geometry, t_rows["instance_id"].astype(int))),
                                             out_shape=shape, transform=wt, fill=0, dtype="uint32") \
                            if len(t_rows) else np.zeros(shape, np.uint32)
                        b_geoms = pick(train_bnd)
                        boundary = rasterize([(g, 1) for g in b_geoms], out_shape=shape, transform=wt, fill=0,
                                             dtype="uint8", all_touched=spec.mode == "line") \
                            if b_geoms else np.zeros(shape, np.uint8)

                        data_ok = _read_valid_data(src, udm, win, bands, nodata)
                        if region is None:
                            in_region = np.ones(shape, bool)
                        else:
                            rg = pick(region)
                            in_region = rasterize([(g, 1) for g in rg], out_shape=shape, transform=wt, fill=0,
                                                  dtype="uint8").astype(bool) if rg else np.zeros(shape, bool)
                        if complete is not None and len(complete):
                            in_region |= rasterize([(g, 1) for g in complete if g.intersects(bb)], out_shape=shape,
                                                   transform=wt, fill=0, dtype="uint8").astype(bool) \
                                if any(g.intersects(bb) for g in complete) else False
                        ig = pick(ign)
                        ignored_px = rasterize([(g, 1) for g in ig], out_shape=shape, transform=wt, fill=0,
                                               dtype="uint8").astype(bool) if ig else np.zeros(shape, bool)
                        valid = data_ok & in_region & ~ignored_px

                        dsts["interior"].write(interior, 1, window=win)
                        dsts["boundary"].write(boundary, 1, window=win)
                        dsts["instance"].write(instance, 1, window=win)
                        dsts["valid"].write(valid.astype(np.uint8), 1, window=win)
                        stats["interior_px"] += int(interior.sum())
                        stats["boundary_px"] += int(boundary.sum())
                        stats["boundary_inside_interior_px"] += int((boundary & interior).sum())
                        stats["valid_px"] += int(valid.sum())
                        stats["image_nodata_px"] += int((~data_ok).sum())
                        stats["ignored_px"] += int(ignored_px.sum())
                        stats["total_px"] += shape[0] * shape[1]
            finally:
                if udm is not None:
                    udm.close()
    finally:
        for d in dsts.values():
            d.close()
    for n in MASK_NAMES:
        with rasterio.open(paths[n], "r+") as d:
            d.descriptions = (n,)

    # farms so small/narrow that the boundary band covers most of them
    if len(train):
        band_area = train.length * spec.total_width_m
        mostly_boundary = int((band_area > 0.5 * train.area).sum())
    else:
        mostly_boundary = 0
    stats.update({
        "scene_id": info.scene_id, "boundary_mode": spec.mode, "boundary_total_width_m": spec.total_width_m,
        "boundary_buffer_radius_m": spec.buffer_radius_m, "boundary_width_px": round(spec.width_px, 2),
        "train_polygons": len(train), "ignored_polygons": len(ignored),
        "farms_mostly_boundary": mostly_boundary,
        "boundary_share_of_interior": round(stats["boundary_inside_interior_px"] / max(stats["interior_px"], 1), 3),
        "masks": {n: str(p) for n, p in paths.items()},
    })
    if mostly_boundary:
        log.warning("%s: %d farms are so narrow that the %.1f m boundary band covers > 50%% of them",
                    info.scene_id, mostly_boundary, spec.total_width_m)
    log.info("%s masks: %d interior px, %d boundary px (%s mode, total width %.1f m = %.2f px, radius %.2f m), "
             "%.1f%% valid", info.scene_id, stats["interior_px"], stats["boundary_px"], spec.mode,
             spec.total_width_m, spec.width_px, spec.buffer_radius_m, 100 * stats["valid_px"] / max(stats["total_px"], 1))
    return stats


def densest_window(polys: gpd.GeoDataFrame, info: RasterInfo, size: int) -> Window:
    """A ``size`` x ``size`` window over the part of the image with the most training farms."""
    size = min(size, info.width, info.height)
    t = polys[polys["train"]] if "train" in polys else polys
    if t.empty:
        return Window((info.width - size) // 2, (info.height - size) // 2, size, size)
    pts = t.geometry.representative_point()
    inv = ~info.affine
    cols, rows = inv * (pts.x.to_numpy(), pts.y.to_numpy())
    cols, rows = np.asarray(cols), np.asarray(rows)
    ok = (cols >= 0) & (cols < info.width) & (rows >= 0) & (rows < info.height)
    if not ok.any():
        return Window((info.width - size) // 2, (info.height - size) // 2, size, size)
    cell = max(size // 2, 1)
    keys = (rows[ok] // cell).astype(int) * 100000 + (cols[ok] // cell).astype(int)
    vals, counts = np.unique(keys, return_counts=True)
    best = vals[np.argmax(counts)]
    r0, c0 = (best // 100000) * cell + cell // 2 - size // 2, (best % 100000) * cell + cell // 2 - size // 2
    c0 = int(np.clip(c0, 0, info.width - size))
    r0 = int(np.clip(r0, 0, info.height - size))
    return Window(c0, r0, size, size)


def mask_validation_figure(cfg, info: RasterInfo, polys: gpd.GeoDataFrame, mask_dir: Path, out_path: Path) -> Path:
    """Task 2C: image, interior mask, boundary mask and boundary overlay for the densest crop of one image."""
    from ..utils.helpers import rgb_indices
    from ..utils.visualization import plot_mask_validation

    win = densest_window(polys, info, int(cfg.masks.get("visual_px", 512)))
    bands = select_bands(info, cfg.imagery.get("bands"))
    rgb = rgb_indices(len(bands), [info.descriptions[b - 1] if info.descriptions else "" for b in bands],
                      cfg.imagery.get("rgb_bands"))
    with rasterio.open(info.path) as src:
        img = src.read([bands[i] for i in rgb], window=win, boundless=True, fill_value=nodata_value(info, cfg))
    masks = {}
    for n in ("interior", "boundary", "valid"):
        with rasterio.open(Path(mask_dir) / f"{n}.tif") as m:
            masks[n] = m.read(1, window=win)
    title = (f"{info.scene_id}  window col {int(win.col_off)} row {int(win.row_off)}  "
             f"({int(win.width)}x{int(win.height)} px, {abs(info.res[0]):.1f} m)")
    return plot_mask_validation(img, masks["interior"], masks["boundary"], masks["valid"],
                                nodata_value(info, cfg), out_path, title)
