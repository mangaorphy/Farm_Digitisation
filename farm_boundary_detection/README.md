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
(`data/imagery/raw/<order id>/`) as a **private** Kaggle dataset named `laikipia-planetscope`, then open
[`notebooks/kaggle_end_to_end.ipynb`](notebooks/kaggle_end_to_end.ipynb), turn on GPU and Internet, attach the
dataset and **Save & Run All**. Results: `best/map/kmz/all_farms.kmz` and the files next to it.

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

## Kaggle notebook (round 2)

[`notebooks/kaggle_end_to_end.ipynb`](notebooks/kaggle_end_to_end.ipynb) trains the experiments under a time
budget (and resumes later if the budget runs out). It backs up the models after each experiment: a zip in the
output, plus a private dataset if the `KAGGLE_API_TOKEN` secret is set. It then:

1. compares every model and an ensemble on the validation hand-traced farms, using a two-stage threshold search;
2. tests the winner once;
3. maps every scene;
4. writes `best/report.html` with the data checks, training curves, comparison, pixel confusion matrices, object
   outcomes, scores by farm size, test visuals and a map preview.

**Run it with Save & Run All (Commit)**: an interactive session's `/kaggle/working` is lost when it stops.

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
| `postprocess.seed_threshold` | 0.5 | higher = fewer, larger farms; tuned on validation in the notebook |
