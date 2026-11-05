"""Synthetic stand-in for an SDX download, to test the whole pipeline offline.

Writes a Maxar/Vantor-style pair (PAN 0.25 m single band + MS 1.0 m BGRN, uint16, with P1BS/M1BS names)
of Voronoi "fields" separated by dark fence lines, and the fields as a KMZ in the Earth Pro schema.
Numbers produced on it say nothing about accuracy in Laikipia.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import MultiPoint, box
from shapely.ops import voronoi_diagram

from .export import to_kmz


def make_scene(out_dir: Path, size_m: float = 800, n_fields: int = 60, seed: int = 0,
               origin=(270000.0, 12000.0), crs="EPSG:32637") -> tuple[Path, Path]:
    """Returns (raw imagery folder, labels folder)."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    raw, lab = out_dir / "raw", out_dir / "labels"
    raw.mkdir(parents=True, exist_ok=True)
    lab.mkdir(parents=True, exist_ok=True)
    x0, y0 = origin
    area = box(x0, y0, x0 + size_m, y0 + size_m)
    pts = MultiPoint(list(zip(rng.uniform(x0, x0 + size_m, n_fields), rng.uniform(y0, y0 + size_m, n_fields))))
    cells = [c.intersection(area) for c in voronoi_diagram(pts, envelope=area).geoms]
    fields = gpd.GeoDataFrame({"geometry": [c for c in cells if c.area > 50]}, crs=crs)

    pan_gsd, ms_gsd = 0.25, 1.0
    n = int(size_m / pan_gsd)
    t_pan = from_origin(x0, y0 + size_m, pan_gsd, pan_gsd)
    ids = rasterize([(g, i + 1) for i, g in enumerate(fields.geometry)], out_shape=(n, n), transform=t_pan,
                    dtype="int32")
    refl = rng.uniform([300, 400, 300, 1500], [700, 900, 1000, 4000], (len(fields) + 1, 4)).astype(np.float32)
    bands = refl[ids].transpose(2, 0, 1)  # (4, n, n) B, G, R, N
    fences = rasterize([(g, 1) for g in fields.geometry.boundary.buffer(0.6)], out_shape=(n, n), transform=t_pan,
                       dtype="uint8").astype(bool)
    bands[:, fences] *= 0.45
    bands += rng.normal(0, 25, bands.shape).astype(np.float32)
    pan = bands[:3].mean(axis=0) * 1.7 + 50  # different DN scale from MS, like real PAN
    f = int(ms_gsd / pan_gsd)
    ms = bands.reshape(4, n // f, f, n // f, f).mean(axis=(2, 4))

    base = "23JAN15081234-{}-SYNTH_P001"
    common = dict(driver="GTiff", dtype="uint16", crs=crs, nodata=0)
    pan_path, ms_path = raw / f"{base.format('P1BS')}.TIF", raw / f"{base.format('M1BS')}.TIF"
    with rasterio.open(pan_path, "w", width=n, height=n, count=1, transform=t_pan, **common) as d:
        d.write(np.clip(pan, 1, 65535).astype(np.uint16), 1)
    with rasterio.open(ms_path, "w", width=n // f, height=n // f, count=4,
                       transform=from_origin(x0, y0 + size_m, ms_gsd, ms_gsd), **common) as d:
        d.write(np.clip(ms, 1, 65535).astype(np.uint16))
        d.descriptions = ("Blue", "Green", "Red", "NIR")

    fields["FARM_ID"] = [f"SYN-{i:04d}" for i in range(len(fields))]
    fields["SOURCE"] = np.where(rng.random(len(fields)) < 0.5, "Digitized", "Scheme layer")
    fields["QA_FLAG"] = np.where(rng.random(len(fields)) < 0.05, "OVERLAP", "OK")
    fields["SCHEME"] = "Synthetic"
    to_kmz(fields, lab / "synthetic_farms.kmz", name="Synthetic farms")
    return raw, lab
