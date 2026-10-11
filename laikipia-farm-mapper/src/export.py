from __future__ import annotations
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape
from .common import require

def export_gpkg_and_kmz(frame, config: dict, aoi=None):
    gpd=require("geopandas", "pip install geopandas pyogrio")
    out=Path(config["output_directory"]); out.mkdir(parents=True,exist_ok=True)
    gpkg=out/"laikipia_farm_polygons.gpkg"; frame.to_file(gpkg,layer="farm_polygons",driver="GPKG")
    if aoi is not None and config.get("export",{}).get("include_aoi",True): aoi.to_file(gpkg,layer="aoi",driver="GPKG")
    geo=frame.to_crs("EPSG:4326")
    placemarks=[]
    for _, row in geo.iterrows():
        polys=[row.geometry] if row.geometry.geom_type=="Polygon" else list(row.geometry.geoms)
        rings="".join("<Polygon><outerBoundaryIs><LinearRing><coordinates>"+" ".join(f"{x:.8f},{y:.8f},0" for x,y in p.exterior.coords)+"</coordinates></LinearRing></outerBoundaryIs></Polygon>" for p in polys)
        desc=escape(f"Farm ID: {row.farm_id}; Area: {row.area_ha:.3f} ha; Method: {row.method}; Validation: {row.validation}")
        placemarks.append(f"<Placemark><name>{escape(str(row.farm_id))} ({row.area_ha:.3f} ha)</name><description>{desc}</description><styleUrl>#candidate</styleUrl>{rings}</Placemark>")
    kml='<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><Style id="candidate"><PolyStyle><color>5500ffff</color></PolyStyle><LineStyle><color>ff00aaff</color><width>2</width></LineStyle></Style>'+''.join(placemarks)+'</Document></kml>'
    kmz=out/config["kmz_filename"]
    with zipfile.ZipFile(kmz,"w",zipfile.ZIP_DEFLATED) as z: z.writestr("doc.kml",kml)
    return gpkg,kmz
