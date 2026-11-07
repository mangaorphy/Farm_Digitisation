"""KMZ writer for Google Earth Pro: one Placemark per farm, attributes in ExtendedData.
Green outline = confident, orange = score below ``low_score`` (review those first)."""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd

STYLE = ('<Style id="auto"><LineStyle><color>ff00ff00</color><width>1.5</width></LineStyle>'
         '<PolyStyle><color>2200ff00</color></PolyStyle></Style>'
         '<Style id="low"><LineStyle><color>ff0080ff</color><width>1.5</width></LineStyle>'
         '<PolyStyle><color>220080ff</color></PolyStyle></Style>')


def _ring(coords) -> str:
    return " ".join(f"{x:.7f},{y:.7f},0" for x, y, *_ in coords)


def _polygon_kml(p) -> str:
    inner = "".join(f"<innerBoundaryIs><LinearRing><coordinates>{_ring(r.coords)}</coordinates></LinearRing>"
                    f"</innerBoundaryIs>" for r in p.interiors)
    return (f"<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>"
            f"{_ring(p.exterior.coords)}</coordinates></LinearRing></outerBoundaryIs>{inner}</Polygon>")


def to_kmz(gdf: gpd.GeoDataFrame, out_path: Path, name: str = "Auto-detected farms", low_score: float = 0.6) -> Path:
    g = gdf.to_crs("EPSG:4326") if len(gdf) else gdf
    cols = [c for c in g.columns if c != "geometry"]
    marks = []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        polys = list(getattr(geom, "geoms", [geom]))
        body = _polygon_kml(polys[0]) if len(polys) == 1 else \
            "<MultiGeometry>" + "".join(_polygon_kml(p) for p in polys) + "</MultiGeometry>"
        data = "".join(f'<Data name="{escape(str(c))}"><value>{escape(str(row[c]))}</value></Data>' for c in cols)
        style = "low" if "score" in cols and float(row["score"]) < low_score else "auto"
        label = row["farm_id"] if "farm_id" in cols else ""
        desc = f"{row['area_ha']} ha, score {row['score']}" if {"area_ha", "score"} <= set(cols) else ""
        marks.append(f"<Placemark><name>{escape(str(label))}</name><description>{escape(desc)}</description>"
                     f"<styleUrl>#{style}</styleUrl><ExtendedData>{data}</ExtendedData>{body}</Placemark>")
    kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f"<name>{escape(name)}</name>{STYLE}<Folder><name>{escape(name)} ({len(marks)})</name>"
           + "\n".join(marks) + "</Folder></Document></kml>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
    return out_path
