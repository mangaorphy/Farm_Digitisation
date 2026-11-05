# Laikipia Farm Digitisation

Automates what is done by hand in Google Earth Pro: **Vantor Legion imagery → U-Net → one polygon per farm**,
exported as KMZ (Google Earth Pro) and GeoPackage (QGIS).

```
SDX downloads (PAN + MS) ──ingest──► aligned 0.5 m RGB+NIR scenes + mosaic.vrt
Digital Earth Africa / Earth Engine ──context──► 10 m multi-date layers (optional extra bands)
your Earth Pro parcels (KMZ) ──check──► registration shifts per scheme
                                   │
                tiles: chips + masks (background / interior / boundary / ignore)
                                   │
                    Kaggle GPU: train U-Net (pretrained ResNet-34 encoder)
                                   │
   predict ► probabilities ► polygonize (watershed) ► farms.gpkg / .kmz ► evaluate on test blocks
```

Step-by-step run instructions: [RUN_STEPS.md](RUN_STEPS.md). **No SDX/Legion imagery?** That file runs the
whole pipeline on a free 10 m Sentinel-2 composite (`python -m farmmap sentinel2`, `config_s2.yaml`).

## Read this first

- **Labels** are every KMZ/KML in `data/labels/manual/`: the v3 file (12,448 parcels) and `New farms - manga.kmz`
  (2,447 traced farms). Polygons without `SOURCE`/`QA_FLAG` count as `Digitized` / `OK`. Parcels whose `QA_FLAG`
  is not `OK` are *ignored* (not farm, not background, not scored).
- **Working area** = West Laikipia Fence + 5 km, plus the areas of your traced farms (3,222 km²,
  `python -m farmmap aoi`). The fence layers it uses are in `data/aoi/fences/`.
- **The 3,536 traced (`Digitized`) parcels are the reference** every model is scored against, whatever it was
  trained on.
- **Run the registration check before training.** At 0.5 m a 5 m offset between outlines and image is 10 pixels.
  Scheme-layer parcels come from plans and may be shifted.
- **A boundary must be visible to be found** (fence, hedge, track, change of crop). Output = candidates to review.
- **Accuracy numbers are only real from `evaluate`** on test blocks (2 km map blocks never used in training).
- **Licence.** CSDA imagery is for approved research use: never commit it (`.gitignore` blocks `*.tif`), keep
  Kaggle datasets **private**, check the terms before sharing derived products or model weights.

## 1. Get the imagery from SDX

SDX (<https://csdap.earthdata.nasa.gov>) is a browser app with no API, so this is done by hand. It searches by
bounding box only, and **each file can be downloaded only once per user**: back the downloads up.

1. `python -m farmmap aoi` writes `data/aoi/sdx_search_boxes.csv` / `.kmz`: 19 boxes of 20 km covering the
   corridor (B01–B06 hold your traced farms: download those first). Per-box GeoJSONs are in `data/aoi/sdx_upload/`.
2. Per box: *Edit AOI bounds* → NE/SW corners from the CSV → product **Vantor Legion**, last 1–2 years,
   cloud ≤ 10 % → keep scenes that cover the AOI → **Download Inventory** (CSV) → download the chosen scenes.
3. Put the files, unchanged, in `data/imagery/raw/` (any sub-folders). PAN + MS pairs are pansharpened by
   `ingest`; RPC-only files are orthorectified with `imagery.dem` or `imagery.rpc_height_m`.

Expect a few hundred GB for the whole corridor, inside the 1 TB quota. Check the quota after B01–B06.

## 2. Setup (Windows laptop, Python 3.11)

```powershell
cd path\to\laikipia-farm-mapper
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m farmmap demo              # synthetic end-to-end run, ~5 min on CPU: must end with "Demo OK"
pytest -q                           # 11 tests, ~3 min
```

## 3. Workflow

| Step | Where | Command | Output |
|---|---|---|---|
| Parcel summary | laptop | `python -m farmmap labels` | counts, sizes, widths in pixels |
| Working area + SDX boxes | laptop | `python -m farmmap aoi` | `data/aoi/` |
| Ingest imagery | laptop | `python -m farmmap ingest` | `data/imagery/scenes/*.tif`, `mosaic.vrt` |
| Registration check | laptop | `python -m farmmap check` | `outputs/registration_check.csv`, `registration_shifts.yaml` |
| Context layers (optional) | laptop | `python -m farmmap context --sources dea gee` | `data/context/dea.tif`, `gee.tif` |
| Training chips | laptop | `python -m farmmap tiles` | `data/tiles/` (zip it for Kaggle) |
| Train | **Kaggle GPU** | `notebooks/kaggle_train.ipynb` | `best.pt`, `history.csv` |
| Predict test blocks + polygons + KMZ | Kaggle or laptop | `python -m farmmap run --split test` | `outputs/farms_test.gpkg/.kmz` |
| Score | laptop | `python -m farmmap evaluate --pred outputs/farms_test.gpkg` | precision / recall / F1, recall by farm size |
| Whole area | Kaggle or laptop | `python -m farmmap run` | `outputs/farms_all.gpkg/.kmz` |

Any setting can be overridden for one run; options go **after** the command:
`python -m farmmap tiles --set tiles.size_px=384 imagery.gsd_m=0.6`.

**Registration check.** One row per `SOURCE` and per `SCHEME` (≥ 15 parcels): the shift at which image edges are
strongest along the outlines. `OK` = on visible edges. `SHIFTED` = paste the values from
`outputs/registration_shifts.yaml` under `labels:` in `config.yaml`, re-run `check`, then `tiles`. `UNCLEAR` =
weak or inconsistent edges: exclude that scheme (`labels.keep.SCHEME`) or re-trace a sample.

**Training on Kaggle.** Zip `data/tiles` → upload as a **private** dataset → push this repo to GitHub (the
notebook clones it) → open `notebooks/kaggle_train.ipynb`, GPU and Internet on, add the dataset, **Save & Run
All**. `last.pt` is saved every epoch; attach the output to a new version to resume. Download `best.pt` into
`data/models/`. Step 8 of the notebook predicts on the GPU if you also upload the scenes (and `data/context/`).

**Output.** One polygon per farm with `farm_id` (AUTO-000001…), `area_ha` and `score` (mean interior
probability). In the KMZ, green = confident, orange = score < 0.6 (review first).

## 4. Pilot: which labels to train on

Same imagery, test blocks and reference; only the training labels change (PowerShell quoting):

| Exp. | Trained on | Options for `tiles` |
|---|---|---|
| A | v3 file only (scheme layer + old traced) | `--set 'labels.keep.file=["Digitized - Laikipia_Farms_v3_2026-10-05.kmz"]' paths.tiles=data/tiles_A` |
| B | v3 + new farms | `--set paths.tiles=data/tiles_B` |
| C | traced only (no scheme layer) | `--set 'labels.keep.SOURCE=["Digitized"]' paths.tiles=data/tiles_C` |

Train each on Kaggle, then e.g. `python -m farmmap run --split test --model data/models_B/best.pt --out
outputs/farms_B.gpkg` and `python -m farmmap evaluate --pred outputs/farms_B.gpkg`. Compare F1 and
`recall_by_size`. Parcels left out of training are *ignored* pixels, never background.

## 5. Context layers: Digital Earth Africa and Google Earth Engine

Neither source is sharper than Legion. What they add is **time**: one Legion image cannot tell a ploughed field
from bare ground, or a crop from a grass paddock; a year of Sentinel-2 can. The 10 m layers are stacked onto
every chip as extra bands.

| Source | Layer | What it tells the model | Account |
|---|---|---|---|
| Digital Earth Africa (`dea`) | `crop_prob` | cropland probability (DE Africa crop mask, 2019) | none |
| | `ndvi_h1`, `ndvi_h2` | greenness, Jan–Jun and Jul–Dec composites of `context.year` | |
| | `bcmad` | change through the year: cultivated plots stand out | |
| Earth Engine (`gee`) | `emb_pc1..6` | Google Satellite Embedding (AlphaEarth), 64-D → 6 principal components | Earth Engine |
| | `dw_crops/trees/built/bare` | Dynamic World mean class probability (built = homesteads) | |
| | `nicfi_ndvi_<month>` | Planet NICFI monthly NDVI, if `context.gee.nicfi_months` is set | NICFI sign-up |

```powershell
python -m farmmap context --sources dea          # open data, no login
pip install earthengine-api; earthengine authenticate      # once
python -m farmmap context --sources gee --set gee.project=YOUR-CLOUD-PROJECT
```

Interrupted runs resume. Set `context.year` to the year of your Legion scenes. `context.use` (`[]`, `[dea]`,
`[dea, gee]`) picks what `tiles` stacks onto chips; the model records its context bands and `predict` re-attaches
them automatically.

**As a filter, no retraining:** `polygonize.context_filter: {band: "ctx:dea:crop_prob", min: 10}` drops predicted
farms with mean cropland probability under 10 %. Check with `evaluate`; the 2019 crop mask misses some smallholdings.

**NICFI as the main image** (4.77 m, for areas without Legion): `config_nicfi.yaml` overrides only resolution,
chip size, boundary width and folders.

```powershell
python -m farmmap nicfi --month 2025-06 --config config_nicfi.yaml
python -m farmmap ingest --config config_nicfi.yaml
python -m farmmap tiles  --config config_nicfi.yaml
```

NICFI needs a free sign-up with Planet's NICFI programme and is **non-commercial use only**.

## 6. Which data makes the best model

After section 4 has fixed the labels. Same test blocks and reference, so F1 compares across rows:

| Exp. | Image | Context | Options for `tiles` |
|---|---|---|---|
| L0 | Legion 0.5 m | none | `--set paths.tiles=data/tiles_L0` |
| L1 | Legion 0.5 m | DE Africa | `--set 'context.use=["dea"]' paths.tiles=data/tiles_L1` |
| L2 | Legion 0.5 m | DE Africa + Earth Engine | `--set 'context.use=["dea","gee"]' paths.tiles=data/tiles_L2` |
| N0 | NICFI 4.77 m | none | `--config config_nicfi.yaml` |
| N2 | NICFI 4.77 m | DE Africa + Earth Engine | `--config config_nicfi.yaml --set 'context.use=["dea","gee"]' paths.tiles=data/tiles_nicfi_N2` |

Score as in section 4, then try `polygonize.context_filter` on the best.

- **L1/L2 vs L0:** does time help? Watch recall for `<0.5 ha` and `merged_predictions`.
- **N0/N2 vs L0:** what is lost at 4.77 m. If N2 is close for farms > 0.5 ha, NICFI can cover areas without Legion.
- Keep the simplest setup within ~0.02 F1 of the best.

## 7. Settings that matter most (`config.yaml`)

| Setting | Default | Notes |
|---|---|---|
| `imagery.gsd_m` | 0.5 | 0.3 m native is ~3x slower; 0.5 m keeps fences visible |
| `labels.keep` | QA OK, both sources | `SOURCE: [Digitized]` if the scheme layer can't be fixed |
| `labels.context_m` | 30 | background is trusted only this close to mapped farms |
| `tiles.boundary_width_m` | 1.5 | width of the boundary class |
| `context.use` | `[]` | context layers stacked onto chips at `tiles` |
| `context.year` | 2025 | year the context layers describe: match the Legion scenes |
| `model.encoder` | resnet34 | try `resnet50` / `efficientnet-b4` after a first baseline |
| `polygonize.marker_threshold` | 0.55 | higher = fewer, larger farms (less splitting) |
| `polygonize.min_area_m2` | 300 | smallest farm kept |

## 8. Code map

```
farmmap/
  labels.py      KMZ/KML / GPKG / SHP → parcels, QA filters, defaults, shifts
  aoi.py         fence corridor + training area → AOI, SDX search boxes
  imagery.py     SDX files → PAN/MS pairing, RPC ortho, pansharpen, 8-bit stretch, VRT mosaic
  register.py    edge-alignment registration check
  context.py     10 m context layers: DE Africa, stacking onto chips, zonal filter
  gee.py         Earth Engine: embeddings PCA, Dynamic World, NICFI
  tiles.py       chips + masks, block-based train/val/test split
  model.py       U-Net (segmentation_models_pytorch), normalisation, checkpoints
  train.py       training loop (AMP, cosine LR, resume), CE + Dice loss
  predict.py     chunked sliding-window inference → probability GeoTIFF
  polygonize.py  watershed instances → polygons, cross-block de-duplication
  evaluate.py    object matching (IoU ≥ 0.5) on test blocks, recall by size
  export.py      KMZ writer
  synthetic.py   synthetic SDX-style scene for demo/tests
```

## Test status

- **Verified (macOS, Python 3.11, CPU):** the KMZ parser on your real files; the AOI and SDX boxes on your
  fence layers; on a synthetic PAN+MS scene: ingest, registration check, tiles, training, prediction,
  polygonize, evaluate, KMZ export, and context bands end to end. `context --sources dea` and `gee` were run on
  real data over a 5 km box in the pilot area.
- **Not run yet:** anything on real Legion data (not downloaded yet), training at full scale, the NICFI export
  (no NICFI access on the Earth Engine account yet).
