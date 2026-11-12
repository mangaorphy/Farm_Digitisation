# Farm boundary detection from PlanetScope imagery

Individual farm polygons from **PlanetScope (3 m) imagery** and farm polygons digitised in **Google Earth Pro**
(KML/KMZ), with a multi-task network that predicts farm **interiors** and the thin **boundaries** between
neighbouring farms. The boundaries are then used to split farms apart when building polygons.

```
Planet E&R order (8-band SR + UDM2, clipped to the farm AOI)
        │ ingest: KML/KMZ → validated, de-duplicated, id'd polygons; rasters discovered, UDM2 paired
        ▼
masks per scene (exact grid): interior · boundary (1–2 px) · instance ids · valid (nodata / cloud / ignored)
        │ geographic split (one ground grid for all scenes) → 512 px patches → normalisation → checks
        ▼
ResNet U-Net, two heads (interior, boundary)  ← BCE + Dice (+ Tversky) with boundary-aware weights
        │ sliding-window inference with blending
        ▼
seeds = interior × (1 − boundary) → watershed on boundary → polygons → GeoJSON · Shapefile · GPKG · KMZ
```

## Quick start

**Laptop** (download the imagery):

```bash
cd farm_boundary_detection
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src aoi                 # training AOI from your farms -> data/aoi/training_aoi.geojson (+ .kmz)
python -m src planet-search       # scenes + estimated quota km² (uses NO quota)
python -m src planet-order --confirm
python -m src planet-download --order-id <id printed by planet-order>
```

**Kaggle** (preprocess, train, compare, map): upload the downloaded order folder
(`data/imagery/raw/<order id>/`) as a **private** Kaggle dataset (here: `laikipia-planetscope`), then open
[`notebooks/kaggle_round3_pretrained.ipynb`](notebooks/kaggle_round3_pretrained.ipynb), turn on GPU and Internet,
attach the dataset and **Save & Run All**. Results: `best/map/kmz/farms_digitised_plus_new.kmz` and the files
next to it.

**Everything also runs from the command line:**

```bash
python -m src prepare             # ingest + masks + figures + split + patches + checks
python -m src train               # resumes from models/latest_checkpoint.pth; --fresh to restart
python -m src evaluate --split test
python -m src run                 # predict every scene + polygons (GeoJSON, Shapefile, GPKG, KMZ)
python -m src demo                # whole pipeline on a synthetic scene (installation check)
pytest -q
```

Any setting can be changed for one run: `python -m src train --set train.batch_size=4 model.encoder=resnet50`.

## Planet: Education & Research plan

- 3,000 km²/month of PlanetScope, counted on the **clipped** area of each scene (overlaps count twice).
  Orders are clipped to the AOI around your farms, and `planet-order` refuses more than `planet.max_order_km2`.
- What the account can download (checked Nov 2026): PSScene 4- and 8-band surface reflectance + UDM2, roughly
  up to a month before today. SkySat (0.5 m) is **not** included.
- `planet-search` searches only scenes the account can download and picks the fewest clear scenes that cover the
  AOI. The candidates and the selection are written to `data/imagery/orders/` so you can look at them in QGIS.
- The key is read from `planet.env_file` (`api_key=...`) or `PL_API_KEY`, and never printed.
- The imagery is licensed: `.gitignore` keeps it out of git, and Kaggle datasets must stay private.

## Design decisions

| Decision | Why |
|---|---|
| **Interior and boundary rasterised independently**, no erosion | A boundary shared by two farms is drawn from both outlines. The interior mask keeps every farm pixel, so the two heads are trained as overlapping targets. |
| **Boundary width = total width**; buffer radius = width / 2 | `masks.boundary_width_m: 3` = 1 px, `6` = 2 px. At ≤ 1 px the outline itself is rasterised with `all_touched` ("line" mode). That gives a continuous line; buffering by 1.5 m and sampling pixel centres would leave gaps. |
| **Instance mask** (`farm instance_id` per pixel) | Keeps farm identities for object-level evaluation and checks. Each patch's metadata lists its farm ids. |
| **Valid mask**: unlabelled land is *ignored*, not background | The KMZ files do not contain every farm. Training "no polygon = not a farm" far from mapped farms would teach the model to miss farms. Background is trusted within `label_region_buffer_m` (60 m) of digitised farms, or inside `paths.complete_areas`. Nodata, UDM2 non-clear pixels and QA-flagged farms are also excluded from loss and metrics. |
| **Geographic split before patching**, one ground grid for all scenes | A patch is kept only if it lies entirely in blocks of one split. Overlapping patches, neighbouring patches and the same place seen in two scenes can't leak between train, validation and test. The split is saved and re-used when labels or scenes are added. |
| **Full-resolution branch** in the decoder | The ResNet stem downsamples ×4 before the first skip connection, which blurs 1–2 px boundaries. |
| ResUNet-a ideas: residual decoder blocks, PSP pooling, **conditioned heads** (interior sees the boundary prediction) | It is not a ResUNet-a reproduction: there is no atrous encoder, no distance or colour heads and no Tanimoto loss. The encoder is a pretrained torchvision ResNet. Its first conv is adapted to 4/8 bands (RGB filters on the R, G, B bands, the mean filter elsewhere). |
| Loss: BCE + Dice (interior), BCE(pos_weight) + Dice + Tversky (boundary), λ_boundary = 2 | Boundary pixels are rare, and a missed boundary merges two farms, so Tversky β > α penalises misses. Interior BCE pixels near true boundaries are up-weighted: w = 1 + w0·exp(−d²/2σ²). |
| **Distance head** (`model.distance_head`) | Predicts each pixel's distance to its farm's own outline, peaking once per farm. Its maxima seed one region per farm (`postprocess.seed_source: distance / combined`), which targets round 1's main error, farms split into pieces. |
| **TTA and ensembles** | `inference.tta` averages over flips. `Predictor([model_a, model_b])` and `ensemble_caches` average several models. The notebook keeps an ensemble only if it beats the best single model on validation. |
| **Model selection on individual farms** | Pixel accuracy is dominated by easy interior pixels. Experiments are compared on object F1 (one-to-one matching, IoU ≥ 0.5) against the hand-traced farms, with post-processing tuned on validation. Test is used once. |
| Polygons: seeds = P(int)·(1−P(bnd)), 4-connected, watershed on P(bnd) | Connected components alone merge neighbours where a boundary is faint. 4-connectivity keeps a diagonal 1-px boundary separating. Blocks with halos keep large scenes in bounded memory. |

Libraries: `opencv-python`, `scikit-learn` and `fiona` are not used. scipy and scikit-image cover the
morphology and watershed. The split is geographic, so scikit-learn's random splitters don't apply. KML/KMZ is
parsed directly, which is one pass instead of GDAL re-reading Earth Pro's ~70 folders.

## Round 3: a network pretrained on PlanetScope fields; polygons that cannot cross

[`notebooks/kaggle_round3_pretrained.ipynb`](notebooks/kaggle_round3_pretrained.ipynb)

| Change | Why |
|---|---|
| `model.arch: ftp_unet` ([`src/models/ftp_unet.py`](src/models/ftp_unet.py)) | The **Fields of the Planet** U-Net (EfficientNet-B3; arXiv:2607.04449, July 2026) is trained on 3 m PlanetScope and the 1.6 million field polygons of Fields of The World (24 countries, Kenya included), for field interior + boundary: the same sensor, resolution and task as here. It replaces ImageNet weights as the starting point. Weights: Hugging Face `taylor-geospatial/ftp-b3`, CC-BY-NC-4.0 (research use). |
| How it is plugged in | It takes 2 seasons × 4 bands (blue, green, red, NIR) as reflectance (`normalization.mode: scale`) and predicts 3 exclusive classes. The wrapper picks the 4 bands from the 8-band scenes, gives the same image as both seasons, and converts the classes exactly to this pipeline's two outputs (P(farm) = P(interior) + P(boundary); P(boundary)). Loss, metrics, prediction and polygon building are unchanged. |
| `postprocess.seed_source: edt` | Each seed blob is split at its narrow necks, so two farms joined through a gap in a faint boundary are still separated. Included in the threshold search. |
| `postprocess.coverage_simplify` | Outlines are simplified **together** (`shapely.coverage_simplify`): a boundary shared by two farms stays one line, so neighbouring polygons cannot cross. Before, each polygon was simplified alone and neighbours crossed. |
| Final map `farms_digitised_plus_new` | Digitised farms kept as they are + model farms only where they are new; model polygons that are very large or ragged are dropped. |

The notebook trains under a time budget, compares every model on the hand-traced validation farms with a
two-stage threshold search, tests the winner once, maps every scene and writes `best/report.html` (data checks,
training curves, comparison, confusion matrices, scores by farm size, map preview). Run it with **Save & Run All
(Commit)**: an interactive session's output is lost when it stops. It scores the round-2 model and the released model (not fine-tuned) on the same hand-traced farms as
the fine-tuned ones. Other published options looked at: *Delineate Anything* (YOLO instance segmentation,
resolution-agnostic; the Fields of the Planet paper reports it well below their model on PlanetScope) and
Fields of The World's Sentinel-2 models (10 m: too coarse for plots under 0.5 ha).

## Zone notebook: one digitising area, fast

[`notebooks/kaggle_zone.ipynb`](notebooks/kaggle_zone.ipynb) maps one area drawn in Google Earth Pro, in about
1–3 hours. Zones live in [`data/zones/`](data/zones/): `salama_area.kmz` (137 km², private dataset
`salama-area-planetscope`) and `mutara_path.kmz` (Mutara – Aberdares path, dataset `mutara-zone-planetscope`).
A zone file without work tiles gets a 3 km grid of tiles (T01, T02, ...). Set `ZONE_FILE`, `ZONE_NAME` and
`DATASET_NAME` at the top, attach the dataset (zone scenes + the round-2 model) and run. The notebook:

1. keeps only the farms in the zone and splits them into train / validation / test ground blocks (the block size
   is chosen automatically so that validation and test both get farms);
2. fine-tunes the round-2 network (`train.init_from`) at 3 m and with the image enlarged ×2
   (`model.input_upsample: 2`), and the Fields of the Planet network; batch-norm statistics are re-estimated on
   the zone's scenes first (`train.bn_recalibrate`), without which a pretrained network can predict nothing;
3. compares them with the round-2 model as it is, on held-out zone farms, and keeps a fine-tuned model only if it
   is clearly better (`SELECT_MARGIN`);
4. maps the zone (`src/gis/zone.py`): every date is predicted, the probabilities are averaged, polygons are built
   once, and the farms already digitised are kept, with model farms added only where they are new. The output is
   `zone/kmz/<zone>_farms.kmz`, with one folder per tile: magenta = digitised, green = model (likely),
   orange = model (check).

**Straight, separated outlines** (`postprocess.regularize`, [`src/gis/regularize.py`](src/gis/regularize.py)): in the
final map every model polygon takes the field-grid direction of the farms around it (digitised farms count most),
becomes a rectangle when one of the same area fits it, otherwise gets edges snapped to that direction and square
corners; overlaps go to the more confident polygon, and neighbours keep a 1 m gap (`gap_m`) so every farm has its
own outline in Google Earth Pro. Digitised farms are never changed. To apply it to a result you already have:
`python tools/straighten_result.py <zone>_farms.gpkg data/zones/<zone>.kmz <out_dir>`.

From the command line: `python -m src zone --zone data/zones/salama_area.kmz --model models/best_model.pth`.

Merged maps have no overlapping polygons: a farm found in two scenes is kept once, and smaller overlaps,
including slivers from simplifying each outline separately, are cut off the lower-scoring polygon
(`remove_overlaps`).

## Pre-training checks (`python -m src check`, also run by `prepare`)

dimensions / transform / CRS of image vs masks (scenes and patches) · farm points land on farm pixels after
reprojection · imagery overlaps the farms · sampled outline points lie on boundary pixels · boundary pixels sit on
instance edges and vice versa · train/val/test footprints do not overlap · every nodata pixel is invalid.
Results: `processed_data/metadata/validation_report.json`.

## Outputs

```
processed_data/  masks/<scene>/{interior,boundary,instance,valid}.tif, images/ interior_masks/ boundary_masks/
                 valid_masks/ instance_masks/, metadata/{preprocessing_report.json, patches.csv,
                 patches.geojson, split_blocks.json, normalization.json, validation_report.json}, visualizations/
models/          best_model.pth, latest_checkpoint.pth, training_history.csv (+ .png)
outputs/         prediction_masks/<scene>_prob.tif (band 1 interior, band 2 boundary, 1..255, 0 = nodata),
                 geojson/ shapefiles/ gpkg/ kmz/, evaluation/<split>/{metrics.json, per_patch.csv, visuals/}
```

Polygon attributes: `farm_id`, `score` (mean interior probability), `boundary_score` (mean boundary probability on
the outline), `area_m2`, `area_ha`, `perimeter_m` (computed in a projected CRS), `source_image`, and in GeoJSON
`lon`/`lat`.

## Main settings (`configs/config.yaml`)

| Setting | Default | Notes |
|---|---|---|
| `masks.boundary_width_m` | 3.0 | total width; 6.0 = 2 px |
| `masks.label_region` / `label_region_buffer_m` | buffer / 60 | where background is trusted |
| `labels.keep` | QA_FLAG OK | other polygons are ignored, not background |
| `split.block_size_m` | 6000 | larger = fewer patches lost on split edges, coarser balance |
| `patches.size` / `stride` / `eval_stride` | 512 / 256 / 512 | |
| `loss.lambda_boundary` | 2.0 | |
| `model.encoder` | resnet34 | resnet18/50/101 |
| `model.arch` | resunet | `ftp_unet` = pretrained Fields of the Planet U-Net (with `normalization.mode: scale`) |
| `model.input_upsample` | 1 | 2 = image enlarged ×2 inside the network (small farms, 4× compute) |
| `train.init_from` | null | start from an earlier `best_model.pth` (fine-tuning) |
| `postprocess.seed_threshold` | 0.5 | higher = fewer, larger farms; tuned on validation in the notebook |
