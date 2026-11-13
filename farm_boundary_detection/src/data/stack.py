"""Two-date (multi-date) input: stack the scenes of several dates into one image.

Neighbouring smallholder plots often look the same on one date but differ in crop or planting time, so a
boundary invisible in July can show in September (U-TAE, ICCV 2021; Fields of the Planet and PRUE feed two seasons).
Averaging two single-date predictions (what the zone map did so far) cannot use that; a network that sees both
dates at once can.

``stack_dates`` groups the scenes by acquisition date (the YYYYMMDD at the start of Planet file names), mosaics
each date onto one common grid (the first scene's CRS, all scenes' extent, the imagery's pixel size), and writes
one GeoTIFF with the dates' bands one after the other (date 1 bands 1..n, date 2 bands n+1..2n, ...), plus a
matching UDM2-style mask (band 1 = 1 where EVERY date is clear). A pixel missing or cloudy on any date is nodata
in all bands. The file names keep the pipeline's scene-key convention, so the stack and its mask pair up.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from .ingestion import scene_key

log = logging.getLogger(__name__)
_DATE = re.compile(r"^(\d{8})_")


def scene_date(path: Path) -> str:
    m = _DATE.match(Path(path).name)
    return m.group(1) if m else Path(path).stem


def _find_udm2(image: Path) -> Path | None:
    key = scene_key(image)
    for p in Path(image).parent.glob("*"):
        if "udm2" in p.name.lower() and p.suffix.lower() in (".tif", ".tiff") and scene_key(p) == key:
            return p
    return None


def stack_dates(images: list[Path], out_dir: Path, name: str, max_dates: int | None = None) -> Path:
    """-> ``out_dir/<name>_3B_AnalyticMS_SR_clip.tif`` (+ ``<name>_3B_udm2_clip.tif``). Dates sorted in time;
    ``max_dates`` keeps the dates with the most scenes (ties: the latest)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        groups[scene_date(p)].append(Path(p))
    dates = sorted(groups)
    if len(dates) < 2:
        raise ValueError(f"need scenes from at least 2 dates to stack, got {dates}")
    if max_dates and len(dates) > max_dates:
        dates = sorted(sorted(dates, key=lambda d: (len(groups[d]), d))[-max_dates:])
    with rasterio.open(groups[dates[0]][0]) as ref:
        crs, res, n_bands, desc = ref.crs, ref.res, ref.count, list(ref.descriptions)
        ox, oy = ref.transform.c, ref.transform.f          # the reference scene's pixel grid
    bounds = None
    for d in dates:
        for p in groups[d]:
            with rasterio.open(p) as s:
                if s.count != n_bands:
                    raise ValueError(f"{p.name} has {s.count} bands, {groups[dates[0]][0].name} has {n_bands}")
                b = transform_bounds(s.crs, crs, *s.bounds)
            bounds = b if bounds is None else (min(bounds[0], b[0]), min(bounds[1], b[1]),
                                               max(bounds[2], b[2]), max(bounds[3], b[3]))
    x0 = ox + np.floor((bounds[0] - ox) / res[0] + 1e-9) * res[0]      # same pixel edges as the reference scene
    y1 = oy + np.ceil((bounds[3] - oy) / res[1] - 1e-9) * res[1]
    W = int(np.ceil((bounds[2] - x0) / res[0]))
    H = int(np.ceil((y1 - bounds[1]) / res[1]))
    transform = rasterio.transform.from_origin(x0, y1, res[0], res[1])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{name}_3B_AnalyticMS_SR_clip.tif"
    udm_path = out_dir / f"{name}_3B_udm2_clip.tif"
    clear_all = np.ones((H, W), bool)
    data = np.zeros((len(dates) * n_bands, H, W), np.uint16)
    for k, d in enumerate(dates):
        filled = np.zeros((H, W), bool)
        clear = np.zeros((H, W), bool)
        for p in groups[d]:                       # mosaic: the first scene with data wins each pixel
            with rasterio.open(p) as s, WarpedVRT(s, crs=crs, transform=transform, width=W, height=H,
                                                   resampling=Resampling.nearest, nodata=0) as v:
                arr = v.read()
            has = (arr > 0).any(0) & ~filled
            data[k * n_bands:(k + 1) * n_bands][:, has] = arr[:, has]
            u = _find_udm2(p)
            if u is not None:
                with rasterio.open(u) as su, WarpedVRT(su, crs=crs, transform=transform, width=W, height=H,
                                                        resampling=Resampling.nearest, nodata=0) as vu:
                    clear |= has & (vu.read(1) == 1)
            else:
                clear |= has
            filled |= has
        clear_all &= clear & filled
    data[:, ~clear_all] = 0                         # missing or cloudy on any date: nodata everywhere
    names = [f"{b}" if k == 0 else f"{b}_d{k + 1}" for k in range(len(dates))
             for b in (desc if all(desc) else [f"band_{i + 1}" for i in range(n_bands)])]
    profile = dict(driver="GTiff", width=W, height=H, count=data.shape[0], dtype="uint16", crs=crs,
                   transform=transform, nodata=0, tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                   BIGTIFF="IF_SAFER")
    with rasterio.open(img_path, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = tuple(names)
        dst.update_tags(dates=",".join(dates), bands_per_date=n_bands)
    with rasterio.open(udm_path, "w", **{**profile, "count": 1, "dtype": "uint8"}) as dst:
        dst.write(clear_all.astype(np.uint8)[None])
    log.info("stacked %d dates %s (%d bands) on a %dx%d grid -> %s; clear on every date: %.0f%%", len(dates), dates,
             data.shape[0], W, H, img_path, 100 * clear_all.mean())
    return img_path
