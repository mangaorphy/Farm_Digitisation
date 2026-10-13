"""Load the manually digitised reference polygons (shapefile, GeoPackage, GeoJSON, KML, KMZ)."""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

VECTOR_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json", ".kml", ".kmz"}


def read_vector(path) -> gpd.GeoDataFrame:
    """Read any supported vector file. KMZ is unzipped; multi-layer KML is concatenated."""
    path = Path(path)
    if path.suffix.lower() == ".kmz":
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as z:
            z.extractall(tmp)
            kmls = sorted(Path(tmp).rglob("*.kml"))
            if not kmls:
                raise ValueError(f"No .kml inside {path}")
            return _read_kml(kmls[0])
    if path.suffix.lower() == ".kml":
        return _read_kml(path)
    return gpd.read_file(path)


def _read_kml(path: Path) -> gpd.GeoDataFrame:
    import pyogrio

    frames = []
    for layer, _ in pyogrio.list_layers(path):
        df = gpd.read_file(path, layer=layer, driver="KML")
        if not df.empty:
            frames.append(df)
    if not frames:
        raise ValueError(f"No features in {path}")
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs or "EPSG:4326")


def _clean(df: gpd.GeoDataFrame, name_field: str, fallback: str, crs) -> gpd.GeoDataFrame:
    df = df[df.geometry.notna() & ~df.geometry.is_empty].copy()
    df = df[df.geom_type.isin(["Polygon", "MultiPolygon"])]
    if df.crs is None:
        df = df.set_crs("EPSG:4326")  # KML and most hand-digitised files are lon/lat
    df = df.to_crs(crs)
    df["geometry"] = shapely.force_2d(df.geometry.values)
    invalid = ~df.geometry.is_valid
    if invalid.any():
        df.loc[invalid, "geometry"] = df.loc[invalid, "geometry"].buffer(0)
    names = None
    for col in (name_field, "Name", "name"):
        if col in df.columns:
            names = df[col].astype(str)
            break
    df["ref_name"] = names if names is not None else fallback
    df["ref_source"] = fallback
    return df[["ref_name", "ref_source", "geometry"]]


def load_reference(cfg, crs=None) -> gpd.GeoDataFrame:
    """All reference polygons in `crs` (default project CRS). Later sources are drawn on top."""
    crs = crs or cfg.project.crs
    files = [cfg.path(p) for p in cfg.reference.polygons]
    for d in cfg.reference.get("extra_dirs", []) or []:
        folder = cfg.path(d)
        if folder.is_dir():
            files += sorted(p for p in folder.iterdir() if p.suffix.lower() in VECTOR_SUFFIXES)
    frames = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Reference polygons not found: {f}")
        frames.append(_clean(read_vector(f), cfg.reference.name_field, f.stem, crs))
    if not frames:
        raise ValueError("No reference polygons configured.")
    # A later (manual correction) layer REPLACES earlier polygons that it covers by >= 50% of their area,
    # so a corrected farm does not leave its stale original underneath. Larger earlier polygons that
    # merely contain the new ones (e.g. a ranch containing a digitised plot) are kept.
    merged = frames[0]
    for new in frames[1:]:
        union = new.geometry.union_all()
        covered = merged.geometry.intersection(union).area / merged.geometry.area.clip(lower=1e-9)
        merged = pd.concat([merged[covered < 0.5], new], ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=crs).reset_index(drop=True)
