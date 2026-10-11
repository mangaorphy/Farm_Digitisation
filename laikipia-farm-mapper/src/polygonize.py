from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import require

def probabilities_to_polygons(probability_raster: str, config: dict):
    """Polygonize class-1 interiors, excluding class-2 boundaries and nodata."""
    rasterio=require("rasterio", "pip install rasterio geopandas shapely scikit-image")
    gpd=require("geopandas", "pip install geopandas")
    shapely=require("shapely.geometry", "pip install shapely")
    skmeasure=require("skimage.measure", "pip install scikit-image")
    from rasterio.features import shapes
    with rasterio.open(probability_raster) as src:
        arr=src.read(masked=True); transform,crs=src.transform,src.crs
    if arr.shape[0] >= 3:
        labels=np.argmax(arr.filled(-1),axis=0); interior=(labels==1); boundary=(labels==2)
        # A one-pixel buffer makes explicit boundary predictions separate neighbouring interiors.
        interior &= ~boundary
    else: interior=arr[0].filled(0)>=float(config["probability_threshold"])
    labelled=skmeasure.label(interior, connectivity=2)
    records=[]; target=config["target_crs"]
    for geom,value in shapes(labelled.astype("int32"), mask=labelled>0, transform=transform):
        if not value: continue
        poly=shapely.shape(geom).buffer(0)
        if poly.is_empty: continue
        records.append({"component":int(value),"geometry":poly})
    frame=gpd.GeoDataFrame(records,crs=crs)
    if frame.empty: return frame
    projected=frame.to_crs(target); projected["area_m2"]=projected.area
    projected=projected[projected.area_m2>=float(config["minimum_mapping_unit_m2"])].copy()
    projected["area_ha"]=projected.area_m2/10000; projected["area_km2"]=projected.area_m2/1e6
    projected=projected[projected.area_ha>=float(config["minimum_farm_area_ha"])].copy()
    projected["farm_id"]=[f"LKP-{i:06d}" for i in range(1,len(projected)+1)]
    projected["method"]=config.get("segmentation_model","unknown"); projected["confidence"]=None; projected["validation"]="candidate_unverified"
    return projected[["farm_id","area_m2","area_ha","area_km2","method","confidence","validation","geometry"]]
