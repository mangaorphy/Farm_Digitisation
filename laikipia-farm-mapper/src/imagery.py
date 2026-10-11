"""Earth Engine Sentinel-2 acquisition and honest imagery-suitability assessment."""
from __future__ import annotations
from pathlib import Path
from .common import UserFacingError, require, write_json

def suitability(config: dict) -> dict:
    i = config.get("imagery", {}); scale = float(i.get("scale_m", 10)); width = float(i.get("expected_min_farm_width_m", 30))
    pixels = width / scale; minimum = float(i.get("suitability_min_short_side_pixels", 5))
    return {"pixel_resolution_m": scale, "expected_min_farm_width_m": width, "short_side_pixels": pixels,
            "minimum_required_pixels": minimum, "suitable_for_candidate_boundaries": pixels >= minimum,
            "warning": None if pixels >= minimum else "Expected farms are too small for Sentinel-2 resolution; do not present polygons as individual farm boundaries."}

def acquire(aoi, config: dict, out_path="data/imagery/sentinel2_composite.tif") -> Path:
    """Export a cloud-masked median composite through Earth Engine (interactive auth required)."""
    try: ee = __import__("ee")
    except ImportError as exc: raise UserFacingError("Earth Engine API is missing. Run setup and scripts/authenticate_earth_engine.py.") from exc
    report = suitability(config); write_json("reports/imagery_suitability.json", report)
    if not report["suitable_for_candidate_boundaries"] and not config.get("imagery", {}).get("allow_unsuitable_inference", False):
        raise UserFacingError(report["warning"] + " Set imagery.allow_unsuitable_inference=true only for exploratory agricultural-region output.")
    try: ee.Initialize()
    except Exception as exc: raise UserFacingError("Earth Engine is not authenticated/initialized. Run python scripts/authenticate_earth_engine.py.") from exc
    coords = aoi.geometry.iloc[0].__geo_interface__["coordinates"]
    region = ee.Geometry.Polygon(coords[0])
    def mask(image):
        scl = image.select("SCL")
        good = scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return image.updateMask(good).select(config["imagery"]["bands"]).multiply(0.0001)
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(config["start_date"], config["end_date"]).filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config["max_cloud_cover"]))
    count = collection.size().getInfo()
    if not count: raise UserFacingError("No Sentinel-2 scenes matched dates/cloud threshold. Widen dates or cloud cover.")
    image = collection.map(mask).median().clip(region)
    # geemap is intentionally avoided: EE's direct download gives a reproducible, small-AOI path.
    import requests
    params = {"name":"sentinel2", "scale":config["imagery"]["scale_m"], "crs":config["target_crs"], "region":region, "filePerBand":False, "format":"GEO_TIFF"}
    url = image.getDownloadURL(params); response = requests.get(url, timeout=300); response.raise_for_status()
    target = Path(out_path); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(response.content)
    dates = collection.aggregate_array("system:time_start").getInfo()
    write_json("data/imagery/metadata.json", {"collection":"COPERNICUS/S2_SR_HARMONIZED", "scene_count":count, "acquisition_timestamps_ms":dates, "bands":config["imagery"]["bands"], "cloud_threshold":config["max_cloud_cover"], "date_range":[config["start_date"],config["end_date"]], "crs":config["target_crs"], "pixel_resolution_m":config["imagery"]["scale_m"], **report})
    return target
