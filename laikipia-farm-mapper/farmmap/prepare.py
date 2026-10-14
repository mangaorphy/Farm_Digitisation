"""Make any GeoTIFF usable: metric projected CRS, chosen bands (e.g. drop an alpha band), optional resampling.

Streams window by window, so large orthomosaics are fine.
"""
from __future__ import annotations

import math
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window


def prepare_imagery(src_path, dst_path, crs="EPSG:32637", resolution=None, bands=None,
                    resampling="bilinear", progress=print) -> Path:
    """Write `dst_path`: reprojected to `crs`, at `resolution` metres (default: keep ~native size),
    keeping only `bands` (1-based, e.g. [1, 2, 3, 4] to drop an alpha band). 0 is nodata."""
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        bands = list(bands) if bands else [i for i in range(1, src.count + 1)]
        if max(bands) > src.count or min(bands) < 1:
            raise ValueError(f"{src_path} has {src.count} bands; asked for {bands}")
        if src.crs is None:
            raise ValueError(f"{src_path} has no CRS. Georeference it first.")
        # explicit target grid (WarpedVRT ignores `resolution` on its own), snapped to whole metres
        t, w, h = calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds,
                                              resolution=float(resolution) if resolution else None)
        if resolution:
            res = float(resolution)
            x0 = math.floor(t.c / res) * res
            y1 = math.ceil(t.f / res) * res
            x1 = t.c + t.a * w
            y0 = t.f + t.e * h
            w, h = math.ceil((x1 - x0) / res), math.ceil((y1 - y0) / res)
            t = from_origin(x0, y1, res, res)
        kw = dict(crs=crs, transform=t, width=w, height=h, resampling=Resampling[resampling], nodata=0)
        with WarpedVRT(src, **kw) as vrt:
            profile = dict(driver="GTiff", width=vrt.width, height=vrt.height, count=len(bands),
                           dtype=vrt.dtypes[0], crs=vrt.crs, transform=vrt.transform, nodata=0,
                           tiled=True, blockxsize=256, blockysize=256, compress="deflate", BIGTIFF="IF_SAFER")
            n = 0
            with rasterio.open(dst_path, "w", **profile) as dst:
                step = 2048
                for row in range(0, vrt.height, step):
                    for col in range(0, vrt.width, step):
                        win = Window(col, row, min(step, vrt.width - col), min(step, vrt.height - row))
                        dst.write(vrt.read(bands, window=win), window=win)
                    n += 1
                    if n % 5 == 0:
                        progress(f"  {min(row + step, vrt.height)}/{vrt.height} rows")
            progress(f"{dst_path}: {vrt.width}x{vrt.height}px, {len(bands)} bands, "
                     f"{abs(vrt.transform.a):g} m pixels, {vrt.crs}")
    return dst_path
