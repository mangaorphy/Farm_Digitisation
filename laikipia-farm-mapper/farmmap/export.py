"""Export polygons to GeoPackage (QGIS) and KMZ (Google Earth Pro)."""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd


def _ring(coords) -> str:
    return " ".join(f"{x:.7f},{y:.7f},0" for x, y, *_ in coords)


def _polygon_kml(poly) -> str:
    s = f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{_ring(poly.exterior.coords)}</coordinates></LinearRing></outerBoundaryIs>"
    for hole in poly.interiors:
        s += f"<innerBoundaryIs><LinearRing><coordinates>{_ring(hole.coords)}</coordinates></LinearRing></innerBoundaryIs>"
    return s + "</Polygon>"


def write_kmz(frame: gpd.GeoDataFrame, path, title="Farm polygons") -> Path:
    wgs = frame.to_crs("EPSG:4326")
    cols = [c for c in wgs.columns if c != "geometry"]
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             f"<name>{escape(title)}</name>",
             '<Style id="farm"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle>'
             '<PolyStyle><color>2200ffff</color></PolyStyle></Style>']
    for _, row in wgs.iterrows():
        g = row.geometry
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        geom_xml = "".join(_polygon_kml(p) for p in polys)
        if len(polys) > 1:
            geom_xml = f"<MultiGeometry>{geom_xml}</MultiGeometry>"
        data = "".join(f'<Data name="{escape(str(c))}"><value>{escape(str(row[c]))}</value></Data>' for c in cols)
        label = row["farm_id"] if "farm_id" in row else row.name
        parts.append(f"<Placemark><name>Farm {label}</name><styleUrl>#farm</styleUrl>"
                     f"<ExtendedData>{data}</ExtendedData>{geom_xml}</Placemark>")
    parts.append("</Document></kml>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", "\n".join(parts))
    return path


def export_polygons(frame: gpd.GeoDataFrame, out_dir, stem: str):
    """Write <stem>.gpkg (project CRS) and <stem>.kmz (WGS84). Returns (gpkg, kmz)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / f"{stem}.gpkg"
    frame.to_file(gpkg, layer="farm_polygons", driver="GPKG")
    kmz = write_kmz(frame, out_dir / f"{stem}.kmz", title=stem)
    return gpkg, kmz
