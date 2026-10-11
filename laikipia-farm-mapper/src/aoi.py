from __future__ import annotations
from pathlib import Path
from .common import UserFacingError, require

def load_aoi(config: dict):
    """Load an AOI from bbox or a vector file and return WGS84 GeoDataFrame."""
    gpd = require("geopandas", "pip install geopandas pyogrio")
    geom_mod = require("shapely.geometry", "pip install shapely")
    spec = config.get("aoi", {}); source = spec.get("source", "bbox")
    if source == "bbox":
        bbox = spec.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise UserFacingError("aoi.bbox must be [min_lon, min_lat, max_lon, max_lat].")
        minx, miny, maxx, maxy = map(float, bbox)
        if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
            raise UserFacingError("AOI bbox has invalid longitude/latitude bounds.")
        frame = gpd.GeoDataFrame({"name":["illustrative_pilot_aoi"]}, geometry=[geom_mod.box(minx,miny,maxx,maxy)], crs="EPSG:4326")
    else:
        path = Path(spec.get("path", ""))
        if not path.exists(): raise UserFacingError(f"AOI file not found: {path}")
        frame = gpd.read_file(path)
        if frame.empty: raise UserFacingError("AOI file contains no features.")
        if frame.crs is None: raise UserFacingError("AOI has no CRS. Define it before use.")
        frame = frame.to_crs("EPSG:4326")
    geom = frame.geometry.union_all()
    if geom.is_empty or not geom.is_valid: raise UserFacingError("AOI geometry is empty or invalid; repair it in QGIS first.")
    return gpd.GeoDataFrame({"name":["aoi"]}, geometry=[geom], crs="EPSG:4326")
