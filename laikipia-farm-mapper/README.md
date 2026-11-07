# Laikipia Farm Digitisation

Automates what is done by hand in Google Earth Pro: **Vantor Legion imagery → U-Net → one polygon per farm**,
exported as KMZ (Google Earth Pro) and GeoPackage (QGIS).

```
SDX downloads (PAN + MS) ──ingest──► aligned 0.5 m RGB+NIR scenes + mosaic.vrt
your Earth Pro parcels (KMZ) ──check──► registration shifts per scheme
                                   │
                tiles: chips + masks (background / interior / boundary / ignore)
                                   │
                    Kaggle GPU: train U-Net (pretrained ResNet-34 encoder)
                                   │
   predict ► probabilities ► polygonize (watershed) ► farms.gpkg / .kmz ► evaluate on test blocks
```

**Status (Nov 2026):** the Sentinel-2 / Digital Earth Africa / Earth Engine route was tried and dropped (the
model could not separate farms at 10 m) and removed from the code. Next imagery source: **Planet** (to be added).
The pipeline below (ingest → tiles → train → predict → polygonize → evaluate → KMZ) is imagery-agnostic.

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
pytest -q                           # 10 tests, ~3 min
```

## 3. Workflow

| Step | Where | Command | Output |
|---|---|---|---|
| Parcel summary | laptop | `python -m farmmap labels` | counts, sizes, widths in pixels |
| Working area + SDX boxes | laptop | `python -m farmmap aoi` | `data/aoi/` |
| Ingest imagery | laptop | `python -m farmmap ingest` | `data/imagery/scenes/*.tif`, `mosaic.vrt` |
| Registration check | laptop | `python -m farmmap check` | `outputs/registration_check.csv`, `registration_shifts.yaml` |
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
`data/models/`. Step 8 of the notebook predicts on the GPU if you also upload the scenes.

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

## 5. Settings that matter most (`config.yaml`)

| Setting | Default | Notes |
|---|---|---|
| `imagery.gsd_m` | 0.5 | 0.3 m native is ~3x slower; 0.5 m keeps fences visible |
| `labels.keep` | QA OK, both sources | `SOURCE: [Digitized]` if the scheme layer can't be fixed |
| `labels.context_m` | 30 | background is trusted only this close to mapped farms |
| `tiles.boundary_width_m` | 1.5 | width of the boundary class |
| `model.encoder` | resnet34 | try `resnet50` / `efficientnet-b4` after a first baseline |
| `polygonize.marker_threshold` | 0.55 | higher = fewer, larger farms (less splitting) |
| `polygonize.min_area_m2` | 300 | smallest farm kept |

## 6. Code map

```
farmmap/
  labels.py      KMZ/KML / GPKG / SHP → parcels, QA filters, defaults, shifts
  aoi.py         fence corridor + training area → AOI, SDX search boxes
  imagery.py     SDX files → PAN/MS pairing, RPC ortho, pansharpen, 8-bit stretch, VRT mosaic
  register.py    edge-alignment registration check
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
  polygonize, evaluate and KMZ export end to end.
- **Not run yet:** anything on Planet imagery.
