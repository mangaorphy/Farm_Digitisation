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
ResNet U-Net: interior + boundary heads (or one 3-class softmax)  ← weighted BCE / Dice / Tversky
        │ sliding-window inference with blending
        ▼
seeds = interior × (1 − boundary) → watershed on boundary → polygons → straight edges, 1 m gaps
        → GeoJSON · Shapefile · GPKG · KMZ (digitised farms kept, model farms added where new)
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

**Kaggle** (compare recipes, map an area): upload the downloaded order folder (`data/imagery/raw/<order id>/`)
together with the round-2 model as a **private** Kaggle dataset, draw the area in Google Earth Pro and save it in
[`data/zones/`](data/zones/), then run [`notebooks/kaggle_zone.ipynb`](notebooks/kaggle_zone.ipynb) (see below).
Results: `zone/kmz/<area>_farms.kmz` and `report.html`.

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
| `postprocess.seed_source: edt` (in the threshold search) | Each seed blob is split at its narrow necks, so two farms joined through a gap in a faint boundary are still separated. |
| `postprocess.coverage_simplify` | Outlines are simplified together (`shapely.coverage_simplify`): a boundary shared by two farms stays one line and neighbours cannot cross. |

Libraries: `opencv-python`, `scikit-learn` and `fiona` are not used. scipy and scikit-image cover the
morphology and watershed. The split is geographic, so scikit-learn's random splitters don't apply. KML/KMZ is
parsed directly, which is one pass instead of GDAL re-reading Earth Pro's ~70 folders.

## Area notebook: recipes compared on every farm, then the map

[`notebooks/kaggle_zone.ipynb`](notebooks/kaggle_zone.ipynb) (generated by
[`tools/make_zone_notebook.py`](tools/make_zone_notebook.py)) works on one area drawn in Google Earth Pro. Areas live in
[`data/zones/`](data/zones/): `salama_area.kmz` (137 km², private dataset `salama-area-planetscope`) and
`mutara_path.kmz` (Mutara – Aberdares path, dataset `mutara-zone-planetscope`). An area without work tiles gets a 3 km
grid of tiles (T01, T02, ...). Set `ZONE_FILE`, `ZONE_NAME` and `DATASET_NAME` at the top and run.

1. **Spatial K-fold cross-validation** ([`src/training/crossval.py`](src/training/crossval.py)). The area's ground
   blocks are dealt into K folds with similar numbers of farms; each recipe is trained K times on K-1 folds and
   predicts the fold it did not see, so every digitised farm is scored once, by a model that never saw it. Fixed
   epochs, last epoch kept (the scored fold never picks the model). A recipe replaces the baseline only if it is
   better overall by `MIN_GAIN` **and** in `MIN_FOLD_WINS` of the folds. (Round 3 chose on 4 validation patches and
   chose wrongly.)
2. **Recipes compared** (each adds one change, all fine-tuned from the round-2 model; batch-norm statistics are
   re-estimated on the area's scenes first, `train.bn_recalibrate`):
   `R2_as_is` (no training) · `C0_baseline` · `C1_separation` (`loss.separation`) · `C2_softmax3`
   (`model.head: softmax3`).
3. **Map**: the K models of the chosen recipe are averaged; every date is predicted and the probabilities averaged;
   digitised farms are kept and model farms added only where new. Output `zone/kmz/<area>_farms.kmz`, one folder per
   tile: magenta = digitised, green = model (likely), orange = model (check).

**Why these two recipes.** Error analysis of the round-3 Salama model on 862 held-out farms: 43% of farms were merged
with a neighbour (54–67% of plots under 0.25 ha), only 12.5% missed. The model painted 55% of the strips between
neighbouring farms (paths, hedges, edges) as farm, although those strips are visible in the imagery (edge strength
1.4–1.6× a farm interior); stricter polygon thresholds made it worse, so the fix belongs in training.

| Setting | What it does | Source |
|---|---|---|
| `loss.separation` | Background pixels with two different farms within 5 px become boundary targets and get +10 weight in the interior loss: filling the gap between neighbours is costly. | U-Net weight maps for touching objects (Ronneberger et al. 2015) |
| `model.head: softmax3` + `loss.softmax3` | One softmax over background / interior / boundary, class weights 0.05 / 0.2 / 0.75, cross-entropy + log-cosh Dice. Returned as the same two outputs, so prediction and polygons are unchanged. | PRUE, CVPR 2026: boundary weighting and log-cosh Dice were the largest gains |

**Individual, straight, separated farms** (`postprocess.regularize`, [`src/gis/regularize.py`](src/gis/regularize.py)),
applied to every final map: each model polygon takes the field-grid direction of the farms around it (digitised farms
count most) and becomes a rectangle when one of the same area fits it; otherwise its edges are snapped to that
direction with square corners (staircase shapes of more than 10 corners are rejected), and a plot that fits no
right-angled shape gets straight sides at its own angle. Overlaps go to the more confident polygon; a farm notched by a
neighbour becomes the largest rectangle that fits inside what is left. Neighbours keep a 1 m gap (`gap_m`), so every
farm has its own outline in Google Earth Pro. Digitised farms are never changed. On the Salama round-3 map: 10,494
model farms, 0 touching or overlapping, median 5 corners, two thirds rectangles. To apply it to an existing result:
`python tools/straighten_result.py <area>_farms.gpkg data/zones/<area>.kmz <out_dir>`.

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
| `model.head` | two | `softmax3` = 3 exclusive classes (background / interior / boundary) |
| `loss.separation.enabled` | false | strips between neighbouring farms as boundary targets, 10× weight |
| `model.input_upsample` | 1 | 2 = image enlarged ×2 inside the network (small farms, 4× compute) |
| `train.init_from` | null | start from an earlier `best_model.pth` (fine-tuning) |
| `postprocess.seed_threshold` | 0.5 | higher = fewer, larger farms; tuned on validation in the notebook |
