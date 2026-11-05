"""Read your farm polygons (Google Earth Pro KMZ/KML, or GPKG/SHP/GeoJSON) and mark which ones to train on.

The Earth Pro export has ~70 folders; GDAL's KML driver re-parses the whole 17 MB file per folder (minutes),
so KML/KMZ is parsed here in one pass. Lines (e.g. the fence reference layer) are skipped.
"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

KML_NS = "{http://www.opengis.net/kml/2.2}"
VECTOR_EXT = {".kml", ".kmz", ".gpkg", ".shp", ".geojson", ".json"}


def _coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in text.split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def _polygon(el) -> Polygon | None:
    outer = el.find(f"{KML_NS}outerBoundaryIs/{KML_NS}LinearRing/{KML_NS}coordinates")
    if outer is None or not outer.text:
        return None
    shell = _coords(outer.text)
    holes = [_coords(c.text) for c in el.findall(f"{KML_NS}innerBoundaryIs/{KML_NS}LinearRing/{KML_NS}coordinates")
             if c.text]
    if len(shell) < 4:
        return None
    return Polygon(shell, [h for h in holes if len(h) >= 4])


def _read_kml_bytes(data: bytes, source_file: str) -> gpd.GeoDataFrame:
    rows = []

    def walk(node, folder):
        for child in node:
            tag = child.tag.replace(KML_NS, "")
            if tag in ("Folder", "Document"):
                name = child.find(f"{KML_NS}name")
                walk(child, name.text if name is not None and tag == "Folder" else folder)
            elif tag == "Placemark":
                polys = [p for p in (_polygon(e) for e in child.iter(f"{KML_NS}Polygon")) if p is not None]
                if not polys:
                    continue  # points / lines (fence reference) are not farms
                attrs = {sd.get("name"): (sd.text or "") for sd in child.iter(f"{KML_NS}SimpleData")}
                attrs.update({d.get("name"): (d.findtext(f"{KML_NS}value") or "") for d in child.iter(f"{KML_NS}Data")})
                name = child.findtext(f"{KML_NS}name") or ""
                attrs.setdefault("FARM_ID", name)
                attrs["name"] = name
                attrs["folder"] = folder
                attrs["file"] = source_file
                attrs["geometry"] = polys[0] if len(polys) == 1 else MultiPolygon(polys)
                rows.append(attrs)

    walk(ET.parse(io.BytesIO(data)).getroot(), None)
    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def read_vector(path: Path) -> gpd.GeoDataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".kmz":
        with zipfile.ZipFile(path) as z:
            kmls = [n for n in z.namelist() if n.lower().endswith(".kml")]
            frames = [_read_kml_bytes(z.read(n), path.name) for n in kmls]
        gdf = pd.concat(frames, ignore_index=True) if frames else _read_kml_bytes(b"<kml/>", path.name)
        return gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    if suffix == ".kml":
        return _read_kml_bytes(path.read_bytes(), path.name)
    gdf = gpd.read_file(path)
    gdf["file"] = path.name
    return gdf.to_crs("EPSG:4326")


def _clean(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        geom = MultiPolygon([p for g in polys for p in getattr(g, "geoms", [g])]) if polys else None
    if geom is None or geom.geom_type not in ("Polygon", "MultiPolygon") or geom.area == 0:
        return None
    return geom


def load_labels(cfg, bbox: tuple | None = "aoi", apply_shifts: bool = True) -> gpd.GeoDataFrame:
    """All farm polygons in the working CRS, with a boolean `train` column.

    train=True  -> learned as farm (interior + boundary) and scored.
    train=False -> real farm that fails cfg.labels.keep (QA flag, source); its pixels are ignored.
    """
    folder = Path(cfg.paths.labels)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VECTOR_EXT) if folder.is_dir() else [folder]
    if not files:
        raise FileNotFoundError(f"No label files (KMZ/KML/GPKG/SHP/GeoJSON) in {folder}")
    gdf = pd.concat([read_vector(f) for f in files], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    gdf["geometry"] = gdf.geometry.apply(_clean)
    gdf = gdf[gdf.geometry.notna()].copy()
    # files traced in the "New farms" folder carry no SOURCE/QA_FLAG: give them the configured defaults
    for col, value in (cfg.labels.get("defaults") or {}).items():
        if col not in gdf.columns:
            gdf[col] = value
        else:
            gdf[col] = gdf[col].replace("", np.nan).fillna(value)

    if bbox == "aoi":
        bbox = aoi_bbox(cfg)
    if bbox is not None:
        gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()

    gdf = gdf.to_crs(cfg.aoi.crs)
    gdf["train"] = matches(gdf, cfg.labels.get("keep")) & ~matches_any(gdf, cfg.labels.get("drop"))

    if apply_shifts and cfg.labels.get("shifts"):
        gdf = apply_label_shifts(gdf, cfg.labels.shift_by, cfg.labels.shifts)
    return gdf.reset_index(drop=True)


def aoi_bbox(cfg) -> list | None:
    """lon/lat W S E N of the working area: the AOI polygon file if it exists, else aoi.bbox."""
    path = cfg.paths.get("aoi")
    if path is not None and Path(path).exists():
        return list(gpd.read_file(path).to_crs("EPSG:4326").total_bounds)
    return cfg.aoi.get("bbox")


def matches_any(gdf, filters: dict | None) -> np.ndarray:
    """Rows where any listed column takes one of the listed values (for labels.drop)."""
    hit = np.zeros(len(gdf), dtype=bool)
    for col, values in (filters or {}).items():
        if col in gdf.columns:
            hit |= gdf[col].astype(str).isin([str(v) for v in values]).to_numpy()
    return hit


def matches(gdf, filters: dict | None) -> np.ndarray:
    """Rows whose columns take one of the allowed values (columns missing from the data are not filtered)."""
    keep = np.ones(len(gdf), dtype=bool)
    for col, allowed in (filters or {}).items():
        if col in gdf.columns:
            keep &= gdf[col].astype(str).isin([str(a) for a in allowed]).to_numpy()
    return keep


def apply_label_shifts(gdf, column: str, shifts: dict) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    for group, (dx, dy) in shifts.items():
        sel = gdf[column].astype(str) == str(group)
        if sel.any():
            gdf.loc[sel, "geometry"] = gdf.loc[sel].geometry.translate(xoff=float(dx), yoff=float(dy))
    return gdf


def summary(gdf: gpd.GeoDataFrame, gsd_m: float) -> pd.DataFrame:
    """Per SOURCE: counts, sizes, and how many pixels wide parcels are at the training resolution."""
    g = gdf.copy()
    g["area_ha"] = g.area / 1e4
    # minimum rotated rectangle short side ≈ parcel width
    widths = []
    for geom in g.geometry:
        r = geom.minimum_rotated_rectangle
        xy = np.asarray(r.exterior.coords)
        sides = np.hypot(*np.diff(xy, axis=0).T)
        widths.append(sides.min() if len(sides) else 0.0)
    g["width_px"] = np.asarray(widths) / gsd_m
    by = "SOURCE" if "SOURCE" in g.columns else "file"
    return g.groupby([by, "train"]).agg(
        parcels=("area_ha", "size"),
        median_ha=("area_ha", "median"),
        median_width_px=("width_px", "median"),
        narrower_than_20px=("width_px", lambda s: float((s < 20).mean())),
    ).round(3)
