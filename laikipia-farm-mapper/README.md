# Laikipia Farm Mapper

Reproducible Python workflow for producing **candidate** farm-boundary polygons from free Sentinel-2 imagery, with GeoPackage and Google Earth Pro KMZ export. It never treats a contiguous agricultural mask as verified individual farms. Free 10 m Sentinel-2 is useful for regional agriculture but is often inadequate for smallholder parcel boundaries; the pipeline creates a suitability warning and stops by default when expected farms are too small in pixel terms.

## Status and requirements

This repository is implemented but no imagery, model weights, model predictions, GeoPackage, or KMZ are included. Earth Engine access, internet access, and an uploaded AOI are needed for a real run. The supplied `config.yaml` bbox is explicitly an illustrative pilot, not a claimed Laikipia farm boundary.

Use Python 3.11 or 3.12. On macOS, use Conda-forge for Rasterio/GDAL rather than attempting a source build through pip.

```bash
cd laikipia-farm-mapper
bash scripts/setup_environment.sh
conda activate laikipia-farm-mapper
python scripts/authenticate_earth_engine.py
```

If `conda` is not available, install [Miniforge](https://github.com/conda-forge/miniforge), restart Terminal, and rerun the setup script. Do not use the failed `.venv` for this project: it lacks the compiled GDAL dependency required by Rasterio.

Earth Engine authentication opens an interactive flow and requires an Earth Engine-enabled Google account; free data still has Earth Engine access and usage limits.

## Configure an AOI

Replace the pilot bbox in `config.yaml`, or use a GeoJSON, KML, or GeoPackage:

```yaml
aoi:
  source: file
  path: data/aoi/my_authoritative_boundary.geojson
```

Use a valid polygon with a declared CRS. The configured projected CRS is EPSG:32737 (UTM 37S); verify it is appropriate for the actual AOI before using calculated areas.

## Commands

```bash
python main.py imagery --config config.yaml
python main.py preprocess --config config.yaml
python main.py predict --model samgeo --config config.yaml
python main.py train --config config.yaml
python main.py predict --model unet --config config.yaml
python main.py polygonize --config config.yaml
python main.py export --config config.yaml
python main.py validate --config config.yaml
python main.py run --config config.yaml
pytest -q
```

`imagery` queries `COPERNICUS/S2_SR_HARMONIZED`, filters the AOI/date/cloud threshold, masks SCL clouds/shadows, and saves a composite and metadata. It resumes from intermediate files for commands after acquisition. Use `samgeo` only after installing its optional package and supplying compatible local weights; it is visual segmentation, not a farm-boundary model. The U-Net expects manually prepared, geographically aligned four-band tiles under `training_dataset/images` and masks under `training_dataset/masks`, with values 0=background, 1=farm interior, 2=boundary.

## Outputs and review

The successful workflow writes a composite/metadata below `data/imagery/`, tiles and manifest below `data/tiles/`, previews/predictions below `outputs/`, quality reports below `reports/`, and:

- `outputs/laikipia_farm_polygons.gpkg`, layer `farm_polygons`
- `outputs/laikipia_farm_polygons.kmz`, ZIP archive with valid `doc.kml`

Open the KMZ in Google Earth Pro; placemarks provide farm ID, area, method, and validation state. Open the GeoPackage in QGIS for imagery-overlay inspection, geometry editing, and a reviewed/verified status update. Areas are computed only in the configured projected CRS.

## Limitations and troubleshooting

Do not publish automatic candidates as cadastral or verified farm boundaries. Adjacent fields can merge without predicted boundaries; SAM may segment roads, trees, or vegetation patches. Confidence is not calibrated unless calibration is carried out against independent reference data. Evaluation metrics (IoU, boundary precision/recall/F1 and count error) require geographically independent manually digitised references and are intentionally not reported otherwise.

If no scenes are found, widen the date range or cloud threshold. If authentication fails, rerun the auth script. If memory is tight, reduce tile size. If an optional dependency is missing, the CLI names the package/command to install. For genuinely small parcels, acquire legally licensed higher-resolution imagery or manually digitise—do not upscale Sentinel-2 or scrape Google imagery.
