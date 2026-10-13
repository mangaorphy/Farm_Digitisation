"""Cloud-free Sentinel-2 composites from the free Microsoft Planetary Computer STAC API.

The output is one tiled GeoTIFF (uint16 reflectance x 10000, 0 = no data) on a fixed
10 m UTM grid. Work is done in chunks and is resumable.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from .reference import load_reference, read_vector

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# SCL classes treated as unusable: no data, saturated, cloud shadow, cloud (med/high), cirrus, snow
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)
BOA_OFFSET = 1000  # processing baseline >= 04.00 (all scenes since 2022-01-25)


def chunk_windows(width: int, height: int, size: int):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def grid_for_bounds(bounds, res: float):
    """Grid snapped to multiples of `res` so rasters from different runs line up."""
    minx, miny, maxx, maxy = bounds
    x0 = math.floor(minx / res) * res
    y1 = math.ceil(maxy / res) * res
    return from_origin(x0, y1, res, res), math.ceil((maxx - x0) / res), math.ceil((y1 - miny) / res)


def aoi_bounds(cfg, aoi_path=None, bbox=None):
    """Bounds (project CRS) of what to download: bbox arg > aoi file > config bbox > reference."""
    crs = cfg.project.crs
    bbox = bbox or (None if aoi_path else cfg.imagery.get("bbox"))
    if bbox:
        return transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    if aoi_path:
        df = read_vector(aoi_path)
        df = df.set_crs("EPSG:4326") if df.crs is None else df
        return tuple(df.to_crs(crs).total_bounds)
    return tuple(load_reference(cfg).total_bounds)


def band_positions(cfg):
    """(red_idx, nir_idx) positions in the raster, or (None, None) if unknown."""
    names = cfg.imagery.bands if cfg.imagery.source == "planetary_computer" else cfg.imagery.get("local_bands")
    if not names:
        return None, None
    names = [str(n).upper() for n in names]
    for red, nir in (("B04", "B08"), ("R", "NIR"), ("RED", "NIR")):
        if red in names and nir in names:
            return names.index(red), names.index(nir)
    return None, None


def estimate(cfg, bounds=None):
    bounds = bounds or aoi_bounds(cfg)
    _, w, h = grid_for_bounds(bounds, cfg.project.resolution_m)
    c = cfg.imagery.chunk_px
    n_chunks = math.ceil(w / c) * math.ceil(h / c)
    return {"width_px": w, "height_px": h, "chunks": n_chunks, "area_km2": round(w * h * (cfg.project.resolution_m / 1000) ** 2)}


def _chunk_composite(catalog, cfg, window_bounds, shape, transform, crs):
    """Median composite of the least-cloudy scenes for one chunk -> (bands, h, w) uint16 or None."""
    import odc.stac
    from odc.geo.geobox import GeoBox

    ic = cfg.imagery
    lonlat = transform_bounds(crs, "EPSG:4326", *window_bounds, densify_pts=11)
    items = list(
        catalog.search(
            collections=[ic.collection],
            bbox=lonlat,
            datetime=f"{ic.date_range[0]}/{ic.date_range[1]}",
            query={"eo:cloud_cover": {"lt": ic.max_cloud_cover}},
        ).items()
    )
    if not items:
        return None
    by_tile: dict[str, list] = {}
    for it in items:
        by_tile.setdefault(it.properties.get("s2:mgrs_tile", it.id[:12]), []).append(it)
    chosen = []
    for group in by_tile.values():
        group.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
        chosen += group[: ic.max_scenes_per_tile]

    geobox = GeoBox(shape, transform, crs)
    ds = odc.stac.load(
        chosen,
        bands=[*ic.bands, "SCL"],
        geobox=geobox,
        resampling={"SCL": "nearest", "*": "bilinear"},
        groupby="solar_day",
    )
    bad = np.isin(ds["SCL"].values, SCL_BAD)
    layers = []
    any_valid = np.zeros(shape, bool)
    for b in ic.bands:
        raw = ds[b].values
        if np.issubdtype(raw.dtype, np.integer):  # raw digital numbers: apply BOA offset
            refl = (raw.astype(np.float32) - BOA_OFFSET) / 10000.0
            invalid = raw == 0
        else:  # odc-stac already scaled to reflectance
            refl = raw.astype(np.float32)
            invalid = ~np.isfinite(refl)
        refl[bad | invalid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(refl, axis=0)
        valid = np.isfinite(med)
        any_valid |= valid
        layers.append(np.where(valid, np.clip(med, 0, 1), 0.0))
    out = np.maximum(np.round(np.stack(layers) * 10000), 1).astype(np.uint16)  # 0 is reserved for nodata
    out[:, ~any_valid] = 0
    return out


def fetch_composite(cfg, out_path=None, aoi_path=None, bbox=None, progress=print):
    """Download/compose imagery for the AOI. Resumes if interrupted. Returns the GeoTIFF path."""
    import planetary_computer
    import pystac_client

    ic = cfg.imagery
    out_path = Path(out_path) if out_path else cfg.path(ic.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crs, res = cfg.project.crs, cfg.project.resolution_m
    bounds = aoi_bounds(cfg, aoi_path, bbox)
    transform, width, height = grid_for_bounds(bounds, res)
    nb = len(ic.bands)
    state_file = out_path.with_suffix(".progress.json")

    done: set[int] = set()
    if out_path.exists() and state_file.exists():
        done = set(json.loads(state_file.read_text())["done"])
        mode = "r+"
    else:
        mode = "w"
    kwargs = {}
    if mode == "w":
        kwargs = dict(driver="GTiff", width=width, height=height, count=nb, dtype="uint16", crs=crs,
                      transform=transform, nodata=0, tiled=True, blockxsize=256, blockysize=256,
                      compress="deflate", predictor=2, BIGTIFF="IF_SAFER")

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    wins = list(chunk_windows(width, height, ic.chunk_px))
    progress(f"Composite grid {width}x{height}px, {len(wins)} chunks ({len(done)} already done)")
    with rasterio.open(out_path, mode, **kwargs) as dst:
        if mode == "w":
            dst.update_tags(bands=",".join(ic.bands), source=ic.collection,
                            date_range=f"{ic.date_range[0]}/{ic.date_range[1]}", scale="reflectance*10000")
        for i, win in enumerate(wins):
            if i in done:
                continue
            wb = rasterio.windows.bounds(win, transform)
            wt = rasterio.windows.transform(win, transform)
            try:
                arr = _chunk_composite(catalog, cfg, wb, (int(win.height), int(win.width)), wt, crs)
            except Exception as exc:  # network hiccup: leave chunk for the next resume
                progress(f"  chunk {i + 1}/{len(wins)} failed ({type(exc).__name__}: {exc}); will retry on resume")
                continue
            if arr is not None:
                dst.write(arr, window=win)
            done.add(i)
            state_file.write_text(json.dumps({"done": sorted(done)}))
            progress(f"  chunk {i + 1}/{len(wins)} {'written' if arr is not None else 'no scenes'}")
    if len(done) == len(wins):
        state_file.unlink(missing_ok=True)
    else:
        progress(f"{len(wins) - len(done)} chunk(s) incomplete: rerun fetch to resume.")
    return out_path
