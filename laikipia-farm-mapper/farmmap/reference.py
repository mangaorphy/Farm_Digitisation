"""Load your digitised farm polygons (KML, KMZ, shapefile, GeoPackage, GeoJSON).

KML/KMZ is parsed directly (standard library XML) so that Google Earth Pro exports with
attribute tables (<SimpleData>/<Data>), folders and holes are read reliably.
"""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Polygon

VECTOR_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json", ".kml", ".kmz"}


# ------------------------------------------------------------------ KML parsing


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _coords(text: str):
    pts = []
    for tok in (text or "").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def _polygons(el):
    """All <Polygon> elements below `el` as shapely polygons (holes supported)."""
    out = []
    for poly in el.iter():
        if _local(poly.tag) != "Polygon":
            continue
        outer, holes = None, []
        for child in poly:
            kind = _local(child.tag)
            if kind not in ("outerBoundaryIs", "innerBoundaryIs"):
                continue
            for c in child.iter():
                if _local(c.tag) == "coordinates":
                    ring = _coords(c.text)
                    if len(ring) >= 4:
                        if kind == "outerBoundaryIs":
                            outer = ring
                        else:
                            holes.append(ring)
        if outer:
            out.append(Polygon(outer, holes))
    return out


def _read_kml(path: Path) -> gpd.GeoDataFrame:
    rows, geoms = [], []

    def walk(el, folder):
        for child in el:
            kind = _local(child.tag)
            if kind in ("Folder", "Document"):
                name = next((c.text for c in child if _local(c.tag) == "name"), "") or ""
                walk(child, f"{folder}/{name}" if folder else name)
            elif kind == "Placemark":
                polys = _polygons(child)
                if not polys:
                    continue  # lines (e.g. fence references) and points are not farms
                attrs = {"Name": next((c.text for c in child if _local(c.tag) == "name"), None),
                         "FOLDER": folder}
                for d in child.iter():
                    k = _local(d.tag)
                    if k == "SimpleData" and d.get("name"):
                        attrs[d.get("name")] = d.text
                    elif k == "Data" and d.get("name"):
                        attrs[d.get("name")] = next((c.text for c in d if _local(c.tag) == "value"), None)
                rows.append(attrs)
                geoms.append(polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys))

    walk(ET.parse(path).getroot(), "")
    if not geoms:
        raise ValueError(f"No polygons found in {path}")
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs="EPSG:4326")


def read_vector(path) -> gpd.GeoDataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".kmz":
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as z:
            kmls = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kmls:
                raise ValueError(f"No .kml inside {path}")
            target = Path(tmp) / "doc.kml"
            target.write_bytes(z.read("doc.kml" if "doc.kml" in kmls else kmls[0]))
            return _read_kml(target)
    if suffix == ".kml":
        return _read_kml(path)
    return gpd.read_file(path)


# ------------------------------------------------------------------ cleaning / filtering


def _clean(df: gpd.GeoDataFrame, name_field: str, fallback: str, crs, filters) -> gpd.GeoDataFrame:
    df = df[df.geometry.notna() & ~df.geometry.is_empty].copy()
    df = df[df.geom_type.isin(["Polygon", "MultiPolygon"])]
    if df.crs is None:
        df = df.set_crs("EPSG:4326")  # KML and most hand-digitised files are lon/lat
    df = df.to_crs(crs)
    df["geometry"] = shapely.force_2d(df.geometry.values)
    invalid = ~df.geometry.is_valid
    if invalid.any():
        df.loc[invalid, "geometry"] = df.loc[invalid, "geometry"].buffer(0)
    df = df[~df.geometry.is_empty].copy()

    names = None
    for col in (name_field, "Name", "name"):
        if col in df.columns:
            names = df[col].astype(str)
            break
    use = pd.Series(True, index=df.index)
    for col, allowed in (filters or {}).items():
        if col in df.columns:
            use &= df[col].astype(str).isin([str(a) for a in allowed])
    df["ref_name"] = names if names is not None else fallback
    df["ref_source"] = fallback
    df["ref_use"] = use
    keep = [c for c in df.columns if c != "geometry"]
    return df[keep + ["geometry"]]


def load_reference(cfg, crs=None) -> gpd.GeoDataFrame:
    """All farm polygons in `crs` (default project CRS), with boolean column `ref_use`.

    Polygons failing `reference.filters` (e.g. QA_FLAG != OK) have ref_use=False: they are painted as
    IGNORE in the label masks and left out of scoring, never learned as background.
    """
    crs = crs or cfg.project.crs
    filters = cfg.reference.get("filters") or {}
    files = [cfg.path(p) for p in (cfg.reference.get("polygons") or [])]
    for d in cfg.reference.get("extra_dirs", []) or []:
        folder = cfg.path(d)
        if folder.is_dir():
            files += sorted(p for p in folder.iterdir() if p.suffix.lower() in VECTOR_SUFFIXES)
    frames = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Reference polygons not found: {f}")
        frames.append(_clean(read_vector(f), cfg.reference.name_field, f.stem, crs, filters))
    if not frames:
        raise ValueError(
            "No farm polygons found. Training needs YOUR digitised farms: export them from Google Earth "
            "Pro as KML/KMZ and put the files in data/labels/manual/ (or list them under "
            "reference.polygons in config.yaml).")
    # A later (manual correction) layer REPLACES earlier polygons that it covers by >= 50% of their area,
    # so a corrected farm does not leave its stale original underneath. Larger earlier polygons that
    # merely contain the new ones are kept.
    merged = frames[0]
    for new in frames[1:]:
        union = new.geometry.union_all()
        covered = merged.geometry.intersection(union).area / merged.geometry.area.clip(lower=1e-9)
        merged = pd.concat([merged[covered < 0.5], new], ignore_index=True)
    out = gpd.GeoDataFrame(merged, geometry="geometry", crs=crs).reset_index(drop=True)
    out["ref_use"] = out["ref_use"].fillna(True).astype(bool)
    if not out["ref_use"].any():
        raise ValueError("All farm polygons were excluded by reference.filters. Check the filter values "
                         f"{filters} against the attributes in your file.")
    return out


# ------------------------------------------------------------------ reporting


def size_report(ref: gpd.GeoDataFrame, resolution_m: float = 10.0) -> dict:
    """Farm-size summary + whether the imagery resolution can resolve the farms."""
    n_all = len(ref)
    if "ref_use" in ref.columns:
        ref = ref[ref["ref_use"]]
    ha = ref.area / 1e4
    side_px = (ref.area ** 0.5) / resolution_m  # side of an equivalent square, in pixels
    return {
        "n_polygons_total": int(n_all), "n_polygons_used": int(len(ref)),
        "area_ha_p10": round(float(ha.quantile(0.1)), 2), "area_ha_median": round(float(ha.median()), 2),
        "area_ha_p90": round(float(ha.quantile(0.9)), 2),
        "share_under_5_px_wide": round(float((side_px < 5).mean()), 2),
        "share_under_10_px_wide": round(float((side_px < 10).mean()), 2),
        "median_side_px": round(float(side_px.median()), 1),
    }


def resolution_warning(report: dict, resolution_m: float):
    """A message if many farms are too small for this resolution, else None."""
    if report["share_under_10_px_wide"] >= 0.3:
        return (f"{report['share_under_10_px_wide']:.0%} of your farms are narrower than 10 pixels at "
                f"{resolution_m:g} m ({report['share_under_5_px_wide']:.0%} narrower than 5 px; median "
                f"~{report['median_side_px']} px wide). Boundaries between such small neighbouring "
                "parcels are not resolvable in this imagery, so expect merged parcels. Higher-resolution "
                "imagery you are licensed to use (imagery.source: local) is needed for them.")
    return None
