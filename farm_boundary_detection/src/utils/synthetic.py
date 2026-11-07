"""A synthetic PlanetScope-like scene and labels, to test the whole pipeline offline.

* 8-band uint16 surface reflectance at 3 m in UTM 37N, Planet file names, a UDM2 file with a "cloud" disc,
  a nodata strip along one edge (as in clipped scenes), and bush (no farms) on the east side;
* Voronoi "farms" with their own spectra; most shared edges carry a 1-pixel darker line (a track or hedge),
  some carry none (neighbouring fields told apart only by colour);
* labels split over a KMZ (EPSG:4326, Earth Pro style ExtendedData), a KML holding a duplicate, a self-
  intersecting (invalid) polygon and a polygon outside the image, and a GeoJSON in EPSG:3857.

Numbers obtained on it say nothing about accuracy on real imagery.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import voronoi_diagram

from ..gis.kmz import to_kmz

SCENE = "20260615_073012_12_24a3"


def _kml(gdf: gpd.GeoDataFrame) -> str:
    g = gdf.to_crs("EPSG:4326")
    marks = []
    for _, r in g.iterrows():
        ring = " ".join(f"{x:.8f},{y:.8f},0" for x, y in r.geometry.exterior.coords)
        marks.append(f"<Placemark><name>{r['FARM_ID']}</name><ExtendedData><SchemaData schemaUrl='#s'>"
                     f"<SimpleData name='FARM_ID'>{r['FARM_ID']}</SimpleData>"
                     f"<SimpleData name='QA_FLAG'>{r['QA_FLAG']}</SimpleData></SchemaData></ExtendedData>"
                     f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing>"
                     f"</outerBoundaryIs></Polygon></Placemark>")
    return ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            "<Folder><name>extra</name>" + "".join(marks) +
            "<Placemark><name>fence</name><LineString><coordinates>36.9,0.1 36.91,0.11</coordinates></LineString>"
            "</Placemark></Folder></Document></kml>")


def make_scene(out_dir: Path, size_px: int = 900, n_farms: int = 70, seed: int = 0, gsd: float = 3.0,
               origin=(300000.0, 12000.0), crs: str = "EPSG:32637") -> tuple[Path, Path]:
    """Writes ``out_dir/imagery`` and ``out_dir/labels``; returns both folders."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    img_dir, lab_dir = out_dir / "imagery", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    x0, y1 = origin
    size_m = size_px * gsd
    farmland = box(x0, y1 - size_m, x0 + 0.72 * size_m, y1)
    pts = MultiPoint(list(zip(rng.uniform(x0, x0 + 0.72 * size_m, n_farms), rng.uniform(y1 - size_m, y1, n_farms))))
    cells = [c.intersection(farmland) for c in voronoi_diagram(pts, envelope=farmland).geoms]
    farms = gpd.GeoDataFrame({"geometry": [c for c in cells if c.area > 2500]}, crs=crs)
    farms["FARM_ID"] = [f"SYN-{i:04d}" for i in range(len(farms))]
    farms["QA_FLAG"] = np.where(rng.random(len(farms)) < 0.06, "OVERLAP", "OK")

    t = from_origin(x0, y1, gsd, gsd)
    shape = (size_px, size_px)
    ids = rasterize([(g, i + 1) for i, g in enumerate(farms.geometry)], out_shape=shape, transform=t, dtype="int32")
    base = np.array([500, 600, 800, 900, 1100, 1000, 2200, 2800], np.float32)
    spectra = base * rng.uniform(0.6, 1.5, (len(farms) + 1, 8)).astype(np.float32)
    spectra[0] = base * np.array([1.1, 1.1, 1.0, 1.0, 1.0, 1.1, 1.0, 0.9])  # bush
    img = spectra[ids].transpose(2, 0, 1)
    img *= rng.normal(1.0, 0.04, img.shape).astype(np.float32)
    bush = ids == 0
    img[:, bush] *= (1 + 0.25 * rng.standard_normal(bush.sum()).astype(np.float32))
    visible = [g.boundary for g, v in zip(farms.geometry, rng.random(len(farms))) if v < 0.8]
    tracks = rasterize([(g, 1) for g in visible], out_shape=shape, transform=t, dtype="uint8", all_touched=True)
    img[:, tracks.astype(bool)] *= 0.55
    img[:, :, :12] = 0  # clipped-scene nodata strip on the west edge
    yy, xx = np.mgrid[0:size_px, 0:size_px]
    cloud = (yy - 0.8 * size_px) ** 2 + (xx - 0.15 * size_px) ** 2 < (0.08 * size_px) ** 2
    img[:, cloud] = 6000
    img = np.clip(img, 0, 65535).astype(np.uint16)
    names = ("coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir")
    common = dict(driver="GTiff", width=size_px, height=size_px, crs=crs, transform=t)
    with rasterio.open(img_dir / f"{SCENE}_3B_AnalyticMS_SR_8b_clip.tif", "w", count=8, dtype="uint16", nodata=0,
                       **common) as d:
        d.write(img)
        d.descriptions = names
    udm = np.zeros((8, size_px, size_px), np.uint8)
    udm[0] = (~cloud & (img[0] > 0)).astype(np.uint8)  # band 1 = clear
    udm[5] = cloud
    with rasterio.open(img_dir / f"{SCENE}_3B_udm2_clip.tif", "w", count=8, dtype="uint8", **common) as d:
        d.write(udm)

    # labels: 80% KMZ (4326), 20% GeoJSON (3857), extras in a KML
    pick = rng.random(len(farms)) < 0.8
    to_kmz(farms[pick], lab_dir / "farms_main.kmz", name="synthetic farms")
    farms[~pick].to_crs("EPSG:3857").to_file(lab_dir / "farms_extra_3857.geojson", driver="GeoJSON")
    dup = farms.iloc[[0]].copy()
    bowtie = Polygon([(x0 + 100, y1 - 100), (x0 + 200, y1 - 200), (x0 + 200, y1 - 100), (x0 + 100, y1 - 200)])
    outside = box(x0 - 5000, y1 + 5000, x0 - 4800, y1 + 5200)
    extra = gpd.GeoDataFrame({"FARM_ID": [dup.FARM_ID.iloc[0], "BOWTIE", "OUTSIDE"],
                              "QA_FLAG": ["OK", "OK", "OK"]},
                             geometry=[dup.geometry.iloc[0], bowtie, outside], crs=crs)
    (lab_dir / "extra.kml").write_text(_kml(extra))
    return img_dir, lab_dir
