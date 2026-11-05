"""Context layers: coarse (10 m) multi-date information stacked onto the high-resolution chips as extra bands.

The 0.5 m image shows *where* edges are; it is one date, so it cannot tell a ploughed field from bare ground
or a crop from a grass paddock. Context layers add *what happens through the year*, from free sources:

  source `dea` (Digital Earth Africa, open data, no account):
    crop_prob   DE Africa cropland probability (2019, 10 m)
    ndvi_h1     NDVI of the Jan-Jun cloud-free GeoMAD composite of `context.year`
    ndvi_h2     NDVI of the Jul-Dec composite (falls back to the previous year if not published yet)
    bcmad       annual GeoMAD Bray-Curtis MAD: how much the surface changes through the year (fields change a lot)
  source `gee` (Google Earth Engine, needs your account; see farmmap/gee.py):
    emb_pc1..k  Google Satellite Embedding (AlphaEarth, 64-D, annual) reduced to k principal components
    dw_crops, dw_trees, dw_built, dw_bare   Dynamic World mean class probabilities over the year
    nicfi_ndvi_<month>  NDVI of Planet NICFI monthly basemaps (only if `context.gee.nicfi_months` is set)

Each source is written once to data/context/<source>.tif on a 10 m UTM grid (uint8, 0 = nodata, values scaled
with the fixed ranges in SCALES so they mean the same everywhere). `context.use` picks which sources are stacked
onto chips, so experiments differ only in that list. Bands are named "ctx:<source>:<layer>"; the model checkpoint
records them and prediction refuses a context stack that differs from training.
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from shapely.geometry import box, shape

DEA_STAC = "https://explorer.digitalearth.africa/stac/search"
DEA_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="5",
               GDAL_HTTP_RETRY_DELAY="3", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", VSI_CACHE="TRUE")

# physical value range -> uint8 1..255 (0 = nodata). Fixed, not per-area, so a model works on any new area.
SCALES = {"crop_prob": (0, 100), "ndvi": (-0.2, 0.9), "bcmad": (0.0, 0.25), "dw": (0.0, 1.0), "emb_pc": (-3.0, 3.0)}


def scale_for(layer: str) -> tuple[float, float]:
    for key, rng in SCALES.items():
        if layer.startswith(key) or (key == "ndvi" and "ndvi" in layer):
            return rng
    raise KeyError(f"no scale for context layer {layer}")


def encode(values: np.ndarray, layer: str) -> np.ndarray:
    lo, hi = scale_for(layer)
    out = np.clip(np.round(1 + 254 * (values - lo) / (hi - lo)), 1, 255)
    return np.where(np.isfinite(values), out, 0).astype(np.uint8)


def decode(codes: np.ndarray, layer: str) -> np.ndarray:
    lo, hi = scale_for(layer)
    v = lo + (codes.astype(np.float32) - 1) / 254 * (hi - lo)
    return np.where(codes > 0, v, np.nan)


# ---------------------------------------------------------------------------------------------- grid / AOI


def grid(cfg) -> tuple[rasterio.Affine, int, int]:
    """Context grid: working CRS, context.gsd_m, covering the AOI, aligned to multiples of the pixel size."""
    from .labels import aoi_bbox
    gsd = float(cfg.context.gsd_m)
    bb = aoi_bbox(cfg)
    if bb is None:
        raise ValueError("no AOI: run `python -m farmmap aoi` or set aoi.bbox")
    b = transform_bounds("EPSG:4326", cfg.aoi.crs, *bb)
    x0, y1 = math.floor(b[0] / gsd) * gsd, math.ceil(b[3] / gsd) * gsd
    x1, y0 = math.ceil(b[2] / gsd) * gsd, math.floor(b[1] / gsd) * gsd
    return from_origin(x0, y1, gsd, gsd), int(round((x1 - x0) / gsd)), int(round((y1 - y0) / gsd))


def aoi_geometry(cfg):
    """AOI polygon in the working CRS (the AOI file if present, else the bbox)."""
    from .labels import aoi_bbox
    path = cfg.paths.get("aoi")
    if path is not None and Path(path).exists():
        return gpd.read_file(path).to_crs(cfg.aoi.crs).union_all()
    return box(*transform_bounds("EPSG:4326", cfg.aoi.crs, *aoi_bbox(cfg)))


def blocks(cfg, block_px: int):
    """(window, window_transform, shape, bounds) of grid blocks that touch the AOI."""
    transform, W, H = grid(cfg)
    aoi = aoi_geometry(cfg).buffer(2 * float(cfg.context.gsd_m))
    for r in range(0, H, block_px):
        for c in range(0, W, block_px):
            win = Window(c, r, min(block_px, W - c), min(block_px, H - r))
            bnd = rasterio.windows.bounds(win, transform)
            if aoi.intersects(box(*bnd)):
                yield win, rasterio.windows.transform(win, transform), (int(win.height), int(win.width)), bnd


def write_source(cfg, name: str, layers: list[str], block_fn, block_px: int = 1024, tags: dict | None = None) -> Path:
    """Run block_fn(window_transform, shape, bounds) -> {layer: float array (nan = nodata)} over the AOI and write
    data/context/<name>.tif. Blocks already written (a .done list next to the file) are skipped on a re-run."""
    out_dir = Path(cfg.paths.context)
    out_dir.mkdir(parents=True, exist_ok=True)
    path, done_path = out_dir / f"{name}.tif", out_dir / f"{name}.done.json"
    transform, W, H = grid(cfg)
    done = set(json.loads(done_path.read_text())) if done_path.exists() and path.exists() else set()
    if not done:
        profile = dict(driver="GTiff", width=W, height=H, count=len(layers), dtype="uint8", nodata=0,
                       crs=cfg.aoi.crs, transform=transform, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
        with rasterio.open(path, "w", **profile) as dst:
            dst.descriptions = tuple(f"ctx:{name}:{lay}" for lay in layers)
            dst.update_tags(source=name, gsd_m=str(cfg.context.gsd_m), year=str(cfg.context.year),
                            scales=json.dumps({lay: scale_for(lay) for lay in layers}), **(tags or {}))
    todo = [b for b in blocks(cfg, block_px) if f"{b[0].row_off}_{b[0].col_off}" not in done]
    print(f"{name}: {len(todo)} block(s) of {block_px} px to fetch -> {path}")
    with rasterio.open(path, "r+") as dst:
        for i, (win, wt, shp, bnd) in enumerate(todo, 1):
            t0 = time.time()
            vals = block_fn(wt, shp, bnd)
            dst.write(np.stack([encode(vals[lay], lay) if lay in vals else np.zeros(shp, np.uint8)
                                for lay in layers]), window=win)
            done.add(f"{win.row_off}_{win.col_off}")
            done_path.write_text(json.dumps(sorted(done)))
            print(f"  block {i}/{len(todo)} ({time.time() - t0:.0f} s)")
    return path


# ---------------------------------------------------------------------------------------------- Digital Earth Africa


def _https(href: str) -> str:
    """s3://bucket/key -> public HTTPS URL (DE Africa buckets are in af-south-1, open, no credentials)."""
    if href.startswith("s3://"):
        bucket, _, key = href[5:].partition("/")
        return f"/vsicurl/https://{bucket}.s3.af-south-1.amazonaws.com/{key}"
    return "/vsicurl/" + href if href.startswith("http") else href


def stac_search(collection: str, bbox, datetime: str | None = None, limit: int = 100) -> list[dict]:
    """All items of a DE Africa STAC collection over bbox (lon/lat), following `next` links."""
    params = {"collections": collection, "bbox": ",".join(f"{v:.5f}" for v in bbox), "limit": limit}
    if datetime:
        params["datetime"] = datetime
    url, items = f"{DEA_STAC}?{urllib.parse.urlencode(params)}", []
    while url:
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    page = json.load(r)
                break
            except Exception:  # noqa: BLE001 - the explorer occasionally returns 5xx; retry
                if attempt == 4:
                    raise
                time.sleep(3 * (attempt + 1))
        items += page.get("features", [])
        url = next((ln["href"] for ln in page.get("links", []) if ln.get("rel") == "next"), None)
    return items


def _item_start(it) -> str:
    p = it["properties"]
    return (p.get("start_datetime") or p.get("datetime") or "")[:10]


def _read_asset(items: list[dict], asset: str, wt, shp, crs, resampling=Resampling.bilinear) -> np.ndarray:
    """Mosaic one asset of several items onto the block grid (float32, nan = nodata)."""
    out = np.full(shp, np.nan, np.float32)
    left, top = wt.c, wt.f
    blk = box(left, top + shp[0] * wt.e, left + shp[1] * wt.a, top)
    for it in items:
        b = transform_bounds("EPSG:4326", crs, *shape(it["geometry"]).bounds)
        if not blk.intersects(box(*b)) or asset not in it["assets"]:
            continue
        with rasterio.Env(**DEA_ENV), rasterio.open(_https(it["assets"][asset]["href"])) as src:
            nod = src.nodata if src.nodata is not None else 0  # DE Africa integer products use 0 = no data
            with WarpedVRT(src, crs=crs, transform=wt, width=shp[1], height=shp[0], resampling=resampling,
                           src_nodata=nod, nodata=nod, dtype="float32") as vrt:
                a = vrt.read(1).astype(np.float32)
        bad = (a == nod) | ~np.isfinite(a)
        fill = np.isnan(out) & ~bad
        out[fill] = a[fill]
    return out


class DEA:
    """Fetches the `dea` layers block by block from public Cloud-Optimised GeoTIFFs (only the needed bytes)."""

    layers = ["crop_prob", "ndvi_h1", "ndvi_h2", "bcmad"]

    def __init__(self, cfg):
        from .labels import aoi_bbox
        self.cfg, y = cfg, int(cfg.context.year)
        bb = aoi_bbox(cfg)
        self.crop = stac_search("crop_mask", bb)
        semi = stac_search("gm_s2_semiannual", bb, f"{y - 1}-01-01/{y}-12-31")
        annual = stac_search("gm_s2_annual", bb, f"{y - 1}-01-01/{y}-12-31")
        if not self.crop:
            raise ValueError("DE Africa crop_mask has no tiles over the AOI")

        def pick(items, start):
            return [it for it in items if _item_start(it) == start]
        # latest available half-years for `year` (H2 of the year may not be published yet -> previous year)
        self.h1 = pick(semi, f"{y}-01-01") or pick(semi, f"{y - 1}-01-01")
        self.h2 = pick(semi, f"{y}-07-01") or pick(semi, f"{y - 1}-07-01")
        self.annual = pick(annual, f"{y}-01-01") or pick(annual, f"{y - 1}-01-01")
        latest_crop = max(_item_start(it) for it in self.crop)
        self.crop = [it for it in self.crop if _item_start(it) == latest_crop]
        self.periods = {"crop_prob": latest_crop, "ndvi_h1": _item_start(self.h1[0]) if self.h1 else "missing",
                        "ndvi_h2": _item_start(self.h2[0]) if self.h2 else "missing",
                        "bcmad": _item_start(self.annual[0]) if self.annual else "missing"}
        print("DE Africa periods used:", self.periods)

    def __call__(self, wt, shp, bnd) -> dict:
        crs = self.cfg.aoi.crs
        out = {"crop_prob": _read_asset(self.crop, "prob", wt, shp, crs)}
        for name, items in (("ndvi_h1", self.h1), ("ndvi_h2", self.h2)):
            if items:
                red = _read_asset(items, "B04", wt, shp, crs)
                nir = _read_asset(items, "B08", wt, shp, crs)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[name] = (nir - red) / (nir + red)
        if self.annual:
            out["bcmad"] = _read_asset(self.annual, "BCMAD", wt, shp, crs)
        return out


S2_BANDS = [("B02", "blue"), ("B03", "green"), ("B04", "red"), ("B08", "nir")]


def export_s2(cfg, period: str, out_dir: Path | None = None, gsd: float = 10.0, block_px: int = 1024) -> Path:
    """Cloud-free Sentinel-2 GeoMAD composite from DE Africa as the *main image* (when no high-resolution imagery
    is available). period = "2025" (annual), "2025-H1" (Jan-Jun) or "2025-H2" (Jul-Dec). Writes one 4-band uint16
    GeoTIFF (blue, green, red, nir; surface reflectance x 10000) into the raw imagery folder, so `ingest` treats it
    like any other scene. Open data, no account. Interrupted runs resume."""
    from .labels import aoi_bbox
    year, _, half = period.upper().partition("-")
    if not year.isdigit() or half not in ("", "H1", "H2"):
        raise ValueError(f"period {period!r}: use YYYY, YYYY-H1 or YYYY-H2")
    coll, start = (("gm_s2_annual", f"{year}-01-01") if not half else
                   ("gm_s2_semiannual", f"{year}-{'01' if half == 'H1' else '07'}-01"))
    items = [it for it in stac_search(coll, aoi_bbox(cfg), f"{year}-01-01/{year}-12-31") if _item_start(it) == start]
    if not items:
        raise ValueError(f"DE Africa {coll} has no composite starting {start} over the AOI (not published yet?)")
    out_dir = Path(out_dir or cfg.paths.imagery_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"s2_geomedian_{start.replace('-', '')}_{half or 'year'}.tif"
    done_path = path.with_suffix(".done.json")
    b = transform_bounds("EPSG:4326", cfg.aoi.crs, *aoi_bbox(cfg))
    x0, y1 = math.floor(b[0] / gsd) * gsd, math.ceil(b[3] / gsd) * gsd
    W, H = int(math.ceil((b[2] - x0) / gsd)), int(math.ceil((y1 - b[1]) / gsd))
    transform = from_origin(x0, y1, gsd, gsd)
    done = set(json.loads(done_path.read_text())) if done_path.exists() and path.exists() else set()
    if not done:
        prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint16", nodata=0, crs=cfg.aoi.crs,
                    transform=transform, tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                    predictor=2, BIGTIFF="IF_SAFER")
        with rasterio.open(path, "w", **prof) as d:
            d.descriptions = tuple(n for _, n in S2_BANDS)
            d.update_tags(source=f"DE Africa {coll}", period=period, start=start)
    aoi = aoi_geometry(cfg).buffer(2 * gsd)
    jobs = []
    for r in range(0, H, block_px):
        for c in range(0, W, block_px):
            win = Window(c, r, min(block_px, W - c), min(block_px, H - r))
            if f"{r}_{c}" not in done and aoi.intersects(box(*rasterio.windows.bounds(win, transform))):
                jobs.append(win)
    print(f"Sentinel-2 {coll} {start}: {len(items)} tile(s), {len(jobs)} block(s) -> {path}")
    with rasterio.open(path, "r+") as d:
        for i, win in enumerate(jobs, 1):
            t0 = time.time()
            wt, shp = rasterio.windows.transform(win, transform), (int(win.height), int(win.width))
            arr = np.stack([_read_asset(items, a, wt, shp, cfg.aoi.crs) for a, _ in S2_BANDS])
            bad = ~np.isfinite(arr).all(axis=0)
            out = np.clip(np.nan_to_num(arr), 1, 65535).astype(np.uint16)
            out[:, bad] = 0
            d.write(out, window=win)
            done.add(f"{win.row_off}_{win.col_off}")
            done_path.write_text(json.dumps(sorted(done)))
            print(f"  block {i}/{len(jobs)} ({time.time() - t0:.0f} s)")
    return path


def build(cfg, sources: list[str]) -> list[Path]:
    paths = []
    for s in sources:
        if s == "dea":
            fetch = DEA(cfg)
            paths.append(write_source(cfg, "dea", DEA.layers, fetch, int(cfg.context.dea_block_px),
                                      tags={"periods": json.dumps(fetch.periods)}))
        elif s == "gee":
            from .gee import GEEContext
            fetch = GEEContext(cfg)
            paths.append(write_source(cfg, "gee", fetch.layers, fetch, int(cfg.context.gee.block_px),
                                      tags=fetch.tags))
        else:
            raise ValueError(f"unknown context source {s!r} (dea, gee)")
    return paths


# ---------------------------------------------------------------------------------------------- reading


def sources(cfg) -> list[Path]:
    """Context files selected by context.use, in order (empty list = no context)."""
    use = cfg.get("context", {}).get("use") or []
    files = [Path(cfg.paths.context) / f"{s}.tif" for s in use]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"context.use={use} but {missing} not built: run `python -m farmmap context`")
    return files


def band_names(files: list[Path]) -> list[str]:
    names = []
    for f in files:
        with rasterio.open(f) as s:
            names += list(s.descriptions)
    return names


def _outer(win: Window, pad: int = 0) -> Window:
    """Smallest whole-pixel window containing `win` (rasterio 1.4 dropped the floor/ceil rounding options)."""
    c0, r0 = math.floor(win.col_off) - pad, math.floor(win.row_off) - pad
    c1, r1 = math.ceil(win.col_off + win.width) + pad, math.ceil(win.row_off + win.height) + pad
    return Window(c0, r0, c1 - c0, r1 - r0)


def read(files: list[Path], dst_crs, dst_transform, shape: tuple[int, int]) -> np.ndarray:
    """All context bands resampled (bilinear) onto a destination grid, e.g. one 0.5 m chip. uint8, 0 = nodata."""
    if not files:
        return np.zeros((0, *shape), np.uint8)
    parts = []
    for f in files:
        with rasterio.open(f) as src:
            # read only the source window around the destination (plus a margin for interpolation)
            b = rasterio.transform.array_bounds(shape[0], shape[1], dst_transform)
            b = transform_bounds(dst_crs, src.crs, *b)
            m = 2 * src.transform.a
            win = _outer(rasterio.windows.from_bounds(b[0] - m, b[1] - m, b[2] + m, b[3] + m, src.transform))
            arr = src.read(window=win, boundless=True, fill_value=0).astype(np.float32)
            arr[arr == 0] = np.nan  # so bilinear does not blend nodata into valid pixels
            out = np.full((src.count, *shape), np.nan, np.float32)
            reproject(arr, out, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            parts.append(np.where(np.isfinite(out), np.clip(np.round(out), 1, 255), 0).astype(np.uint8))
    return np.concatenate(parts)


def zonal_mean(gdf: gpd.GeoDataFrame, files: list[Path], band: str) -> np.ndarray:
    """Mean physical value of one context band inside each polygon (nan if no valid pixel)."""
    for f in files:
        with rasterio.open(f) as src:
            if band not in src.descriptions:
                continue
            k = src.descriptions.index(band) + 1
            g = gdf.to_crs(src.crs)
            win = _outer(rasterio.windows.from_bounds(*g.total_bounds, src.transform), pad=1)
            codes = src.read(k, window=win, boundless=True, fill_value=0)
            wt = src.window_transform(win)
            ids = rasterize([(geom, i + 1) for i, geom in enumerate(g.geometry)], out_shape=codes.shape,
                            transform=wt, fill=0, dtype="int32", all_touched=True)
            vals = decode(codes, band.split(":")[-1])
            ok = (ids > 0) & np.isfinite(vals)
            s = np.bincount(ids[ok], weights=vals[ok], minlength=len(g) + 1)
            n = np.bincount(ids[ok], minlength=len(g) + 1)
            return np.where(n[1:] > 0, s[1:] / np.maximum(n[1:], 1), np.nan)
    raise KeyError(f"context band {band} not found in {files}")
