"""The area to map: Laikipia County boundary from open data (no account needed).

Sources (tried in order):
  1. geoBoundaries gbOpen, Kenya ADM1 (counties). CC BY 4.0, https://www.geoboundaries.org
  2. OpenStreetMap via Nominatim. ODbL, (c) OpenStreetMap contributors
Standard library only, so it works before the heavy dependencies are installed.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "laikipia-farm-mapper/1.0 (research use)"}
GEOBOUNDARIES = "https://www.geoboundaries.org/api/current/gbOpen/KEN/ADM1/"


def _get_json(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _from_geoboundaries(name: str):
    meta = _get_json(GEOBOUNDARIES)
    meta = meta[0] if isinstance(meta, list) else meta
    data = _get_json(meta["gjDownloadURL"], timeout=300)
    for feat in data["features"]:
        props = feat.get("properties", {})
        if name.lower() in str(props.get("shapeName", "")).lower():
            return feat, "geoBoundaries gbOpen (CC BY 4.0)"
    raise LookupError(f"'{name}' not found in geoBoundaries Kenya ADM1")


def _from_osm(name: str):
    q = urllib.parse.urlencode({"q": f"{name} County, Kenya", "format": "geojson",
                                "polygon_geojson": 1, "limit": 1})
    data = _get_json("https://nominatim.openstreetmap.org/search?" + q)
    feats = [f for f in data.get("features", []) if f["geometry"]["type"] in ("Polygon", "MultiPolygon")]
    if not feats:
        raise LookupError(f"No polygon for '{name} County' in OpenStreetMap")
    return feats[0], "OpenStreetMap contributors (ODbL)"


def download_county_boundary(out_path, name: str = "Laikipia") -> Path:
    out_path = Path(out_path)
    errors = []
    for fn in (_from_geoboundaries, _from_osm):
        try:
            feat, source = fn(name)
            break
        except Exception as exc:  # network or lookup failure: try the next source
            errors.append(f"{fn.__name__}: {exc}")
    else:
        raise RuntimeError("Could not download the county boundary:\n  " + "\n  ".join(errors)
                           + f"\nDownload it manually and save it as {out_path} (GeoJSON, WGS84).")
    out = {"type": "FeatureCollection", "name": f"{name}_county",
           "features": [{"type": "Feature", "properties": {"name": name, "source": source},
                         "geometry": feat["geometry"]}]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    return out_path


def county_path(cfg, download: bool = True) -> Path:
    """Path to the county boundary file, downloading it on first use."""
    p = cfg.path(cfg.aoi.county_boundary)
    if not p.exists() and download:
        download_county_boundary(p, cfg.aoi.county_name)
    return p


def load_county(cfg, crs=None):
    """County boundary as a one-row GeoDataFrame in `crs` (default: project CRS)."""
    import geopandas as gpd

    df = gpd.read_file(county_path(cfg))
    df = df.set_crs("EPSG:4326") if df.crs is None else df
    union = df.geometry.union_all()
    return gpd.GeoDataFrame({"name": [cfg.aoi.county_name]}, geometry=[union], crs=df.crs).to_crs(crs or cfg.project.crs)
