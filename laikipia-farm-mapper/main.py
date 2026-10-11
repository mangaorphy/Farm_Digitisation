#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml
from src.common import UserFacingError, ensure_dirs

def config_at(path):
    with open(path) as f: c=yaml.safe_load(f)
    if not isinstance(c,dict): raise UserFacingError("Configuration must be a YAML mapping.")
    required=("start_date","end_date","max_cloud_cover","tile_size","tile_overlap","target_crs","output_directory")
    missing=[k for k in required if k not in c]
    if missing: raise UserFacingError("Missing required configuration key(s): "+", ".join(missing))
    if not 0 <= float(c["max_cloud_cover"]) <= 100: raise UserFacingError("max_cloud_cover must be between 0 and 100.")
    if int(c["tile_size"]) <= 0 or int(c["tile_overlap"]) < 0 or int(c["tile_overlap"]) >= int(c["tile_size"]):
        raise UserFacingError("tile_size must be positive and tile_overlap must be non-negative and smaller than tile_size.")
    ensure_dirs(c); return c

def main():
    p=argparse.ArgumentParser(description="Laikipia Farm Mapper: candidate-boundary workflow")
    p.add_argument("command",choices=["imagery","preprocess","predict","train","polygonize","export","validate","run"]); p.add_argument("--config",default="config.yaml"); p.add_argument("--model",choices=["samgeo","unet"])
    a=p.parse_args(); c=config_at(a.config)
    from src.aoi import load_aoi
    aoi=load_aoi(c)
    imagery=Path("data/imagery/sentinel2_composite.tif"); tiles=list(Path("data/tiles").glob("tile_*.tif")); probs=list(Path("outputs/predictions").glob("*_prob.tif"))+list(Path("outputs/predictions").glob("*_sam.tif")); polygons=Path(c["output_directory"])/"farm_polygons.geojson"
    if a.command in ("imagery","run"):
        from src.imagery import acquire; imagery=acquire(aoi,c)
    if a.command in ("preprocess","run"):
        from src.preprocessing import tile_raster,rgb_preview
        if not imagery.exists(): raise UserFacingError("Imagery missing; run imagery first.")
        tiles=tile_raster(str(imagery),c); rgb_preview(str(imagery))
    if a.command in ("predict","run"):
        if not tiles: raise UserFacingError("No tiles found; run preprocess first.")
        from src.segmentation import predict_samgeo,predict_unet
        model=a.model or c["segmentation_model"]; probs=(predict_samgeo if model=="samgeo" else predict_unet)(tiles,c)
    if a.command=="train":
        from src.training import train; print(train(c)); return
    if a.command in ("polygonize","run"):
        if not probs: raise UserFacingError("No prediction rasters found; run predict first.")
        from src.polygonize import probabilities_to_polygons
        import geopandas as gpd
        import pandas as pd
        frames=[probabilities_to_polygons(str(x),c) for x in probs]
        frame=gpd.GeoDataFrame(pd.concat(frames,ignore_index=True), crs=c["target_crs"]) if frames else gpd.GeoDataFrame()
        frame.to_file(polygons,driver="GeoJSON")
    if a.command in ("export","validate","run"):
        import geopandas as gpd
        if not polygons.exists(): raise UserFacingError("Polygon GeoJSON missing; run polygonize first.")
        frame=gpd.read_file(polygons)
        from src.export import export_gpkg_and_kmz
        gpkg,kmz=export_gpkg_and_kmz(frame,c,aoi)
        from src.validation import validate
        print(validate(frame,c,kmz)); print(f"Created {gpkg} and {kmz}")
if __name__=="__main__":
    try: main()
    except UserFacingError as e: print(f"Error: {e}",file=sys.stderr); sys.exit(2)
