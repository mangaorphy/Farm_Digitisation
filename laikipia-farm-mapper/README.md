# Laikipia Farm Digitisation

Automates what you do by hand in Google Earth Pro: **satellite image → trained model → one polygon per farm**, exported as KMZ (opens in Google Earth Pro) and GeoPackage (QGIS). The model can be **retrained** with new images and your corrections.

```
Laikipia County boundary (open data) ─ defines the area
Sentinel-2 (free) → cloud-free composite → U-Net: background / farm interior / farm boundary
                                                  ▲
       YOUR farm polygons from Google Earth Pro ──┘ = training labels
                                                  │
                    probabilities → one polygon per farm → .kmz / .gpkg
```

## Read this first

- **Two different inputs.**
  - **Area to map** = the Laikipia County boundary, downloaded automatically (geoBoundaries, CC BY 4.0; OpenStreetMap fallback) to `data/aoi/laikipia_county.geojson`. The old `data/aoi/property_boundaries/` shapefile (ranch/conservancy units) is **not used**.
  - **Training labels** = your Google Earth Pro file, `data/labels/manual/Digitized - Laikipia_Farms_v3_2026-10-05.kmz` (12,448 parcels: 11,290 "Scheme layer" + 1,158 "Digitized"). Any other KML/KMZ/SHP/GPKG/GeoJSON you add to that folder is used too.
- **Your file's quality flags are respected.** Parcels with `QA_FLAG` other than `OK` (73: duplicate labels, overlaps, label fixes) are *ignored*: not learned as farms, not learned as background, not scored. Change this under `reference.filters` in `config.yaml`. The "fence" reference lines are skipped.
- **Resolution is the main risk, and your data shows it.** Measured on your file at 10 m Sentinel-2: **74% of parcels are narrower than 10 pixels, 28% narrower than 5 pixels**, median ≈ 7 px wide (median 0.46 ha). Neighbouring parcels that small cannot be reliably separated at 10 m. Realistic outcomes: larger or isolated parcels and fenced blocks are outlined; clusters of small adjacent plots tend to merge. For a model that separates small plots, you need higher-resolution imagery that you are licensed to use (a commercial provider, or aerial/drone orthomosaics) supplied as GeoTIFF with `imagery.source: local`; the same code trains on it (pixel size is read from the file). Imagery is **not** taken from Google Earth (terms of service).
- **Scheme-layer parcels may not be registered to the imagery.** Outlines taken from scheme plans are often shifted from the visible fence lines. Run `python -m farmmap check` (or notebook step A4b) after the imagery download: it measures edge strength along each source's outlines at shifts of up to ±40 m. If the scheme layer prefers another shift, train on the traced parcels only: `reference.filters: {QA_FLAG: [OK], SOURCE: [Digitized]}`. Do not skip this check: misregistered labels cap accuracy.
- **A boundary can only be found if something visible marks it** (fence, hedge, road, river, change of crop). Treat output as **candidates to review**.
- **Do not trust any accuracy number until `evaluate` has run on real imagery.** It scores only on map blocks the model never trained on.
- **Test status.** Verified by running: KMZ import of your real file, label building on the pilot area, the registration check, the county boundary, and the config logic. **Not yet run by the author: model training and prediction** (PyTorch could not be installed in the authoring environment). `python -m farmmap demo` and `pytest -q` on your laptop are the first thing to run.

### How the pilot area was chosen

`imagery.bbox` is set to lon 36.45–36.75, lat 0.00–0.30, a 33 × 33 km box holding about 8,200 (two thirds) of your parcels, so a first run downloads and trains on the area where you have most data. Widen it later, or use `null` for the whole county.

## Setup on Windows (Intel Core i7)

1. **Install Python 3.11** (64-bit) from <https://www.python.org/downloads/windows/>. Tick **“Add python.exe to PATH”**. Python 3.10–3.12 work; 3.13 is not yet supported by all dependencies.
2. **Copy this whole `laikipia-farm-mapper` folder** to the laptop (USB or cloud). It already contains your KMZ in `data/labels/manual/`. (Delete the duplicate `Digitized - Laikipia_Farms_v3…kmz` in the project root if you like; only the copy in `data/labels/manual/` is used.)
3. Open **PowerShell** in the folder and run:

```powershell
cd path\to\laikipia-farm-mapper
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

   - If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.
   - The torch download is large (a few hundred MB). A CPU-only build is the default on Windows, which is what you want without an NVIDIA GPU.

4. **Check the install** (offline, uses a synthetic scene, a few minutes):

```powershell
python -m farmmap demo
pytest -q
```

   `demo` ending with `Demo OK: … polygons` means everything works. Its accuracy figures are meaningless for Laikipia. If a step fails, send the error text.

5. **Start the notebook:**

```powershell
jupyter lab
```

   Open `notebooks/farm_digitisation.ipynb` and run the cells from the top.

> Tip: the first full run downloads imagery (the default pilot area is ~33 × 33 km) and trains on CPU. Run the notebook with `cfg.train.epochs = 3` first to see timings, then raise it.

## The notebook (recommended)

`notebooks/farm_digitisation.ipynb` goes step by step with plots:

| Part | What it does |
|---|---|
| A | See the county and your farms, check farm sizes vs resolution, download Sentinel-2, **registration check**, build labels, see the train/val/test split |
| B | The model and training, with loss and IoU curves |
| C | Evaluation on unseen blocks, plus a visual model-vs-yours comparison |
| D | Predict an area → KMZ / GeoPackage |
| E | **Retraining** with new images and your corrections, comparing old vs new before adopting |

## Command line (same steps)

Run from the project folder with the venv active. Settings are in `config.yaml`.

| Command | Purpose |
|---|---|
| `python -m farmmap county` | Download the Laikipia boundary (automatic if missing) |
| `python -m farmmap info [--county]` | Download size for the configured area (only chunks inside the county count) |
| `python -m farmmap fetch` | Cloud-free composite → `data/imagery/laikipia_s2.tif` (resumable) |
| `python -m farmmap check` | Registration check: do each source's outlines sit on image edges? (run after `fetch`) |
| `python -m farmmap labels` | Rebuild label masks from `data/labels/manual/` for every scene in `data/imagery/` |
| `python -m farmmap train` | Train from scratch → `data/models/unet_best.pt` |
| `python -m farmmap evaluate` | Metrics on held-out test blocks |
| `python -m farmmap predict --county` | Image → polygons for the **whole county** (large download) |
| `python -m farmmap predict --bbox 36.8 0.2 36.9 0.3 --name test1` | Image → polygons for a box (`outputs/test1/`) |
| `python -m farmmap predict --aoi area.kml --name area1` | Same, for a KML/KMZ/GeoJSON/SHP area |
| `python -m farmmap predict --imagery my.tif --name x` | Same, using your own GeoTIFF |

## Retraining with new images (how the model gets better)

The model trains on **every GeoTIFF in `data/imagery/`** together with your polygons. To improve it:

1. **Extract new images** for areas you want to add or where the model is weak:
   ```powershell
   python -m farmmap fetch --bbox 36.95 -0.05 37.10 0.10 --out data/imagery/area_2.tif
   ```
   (or `--aoi my_area.kml`). Each file in `data/imagery/` becomes an extra training scene.
2. **Predict** there and open the `.kmz` in Google Earth Pro on top of the imagery:
   ```powershell
   python -m farmmap predict --imagery data/imagery/area_2.tif --name area_2
   ```
3. **Correct or add polygons** as you already do. Save as KML/KMZ into `data/labels/manual/` (use the "New farms" folder of your file, or a new file). (`.shp`, `.gpkg`, `.geojson` also work.)
   - A polygon you draw **replaces** any earlier polygon it covers by more than half. Files are applied in alphabetical order, later files on top.
   - Areas without polygons nearby are **ignored**, not learned as “no farms”.
   - Digitise whole farms, and be consistent about what counts as a boundary.
4. **Rebuild labels and fine-tune** from the current model (keeps what it learned):
   ```powershell
   python -m farmmap labels
   python -m farmmap train --init data/models/unet_best.pt --epochs 15 --out data/models/unet_v2.pt
   ```
5. **Compare on the same test blocks and keep the better one:**
   ```powershell
   python -m farmmap evaluate --checkpoint data/models/unet_best.pt
   python -m farmmap evaluate --checkpoint data/models/unet_v2.pt
   ```
   If v2 is better, replace `data/models/unet_best.pt` with it (keep a backup). The notebook (part E) does steps 4–5 for you.

How the split protects you: whole 5 km blocks are assigned to train / validation / test and saved in `data/labels/split.json`. Test blocks never move when you add data, so v1 and v2 are scored on the same ground. New scenes contribute their own new test blocks, so re-run `evaluate` for *both* models after adding scenes.

## Outputs

| File | Use |
|---|---|
| `outputs/<name>/laikipia_farm_polygons_<name>.kmz` | Google Earth Pro. Placemarks carry `farm_id`, `area_ha`, `confidence` |
| `outputs/<name>/laikipia_farm_polygons_<name>.gpkg` | QGIS, layer `farm_polygons`, EPSG:32637 (UTM 37N) |
| `outputs/<name>/<name>_probs.tif` | Probability raster (background / interior / boundary × 255) for finding uncertain areas |

## How polygons are made

`farmmap/polygonize.py`: pixels that are confidently interior and not boundary become seeds; each connected seed is one farm (the boundary class keeps neighbours apart); seeds are vectorised, grown back by half the boundary width, de-overlapped, simplified, and filtered by `min_area_ha`. In `config.yaml`, raising `polygonize.boundary_threshold` merges more farms, lowering it splits more.

## Project layout

```
config.yaml                          all settings
farmmap/                             package: imagery, labels, data, model, train, predict, polygonize, evaluate, export
notebooks/farm_digitisation.ipynb    guided train + retrain notebook
data/aoi/laikipia_county.geojson     area to map (downloaded)
data/aoi/property_boundaries/        old ranch-unit layer, unused (kept for context)
data/imagery/                        downloaded scenes (generated)
data/labels/manual/                  YOUR farm polygons (KML/KMZ from Google Earth Pro) go here
data/models/                         trained checkpoints (generated)
outputs/                             polygons and probability rasters (generated)
tests/                               pytest -q  (offline, synthetic)
```

## Troubleshooting

- **No scenes / empty composite**: widen `imagery.date_range` or raise `max_cloud_cover` (Laikipia is clearest Jan–Mar and Jul–Sep).
- **Download interrupted**: run `fetch` again; finished chunks are skipped.
- **Out of memory**: lower `train.batch_size` and `train.crop_px`, or use a smaller `imagery.bbox`.
- **Pretrained weights fail to download (offline)**: set `model.encoder_weights: null`.
- **Training too slow on CPU**: lower `train.epochs` / `train.steps_per_epoch`, keep `encoder: resnet18`, shrink the `bbox`.
- **`pip install` fails on rasterio/geopandas**: confirm Python is 3.10–3.12 64-bit (`python --version`).
- **"No farm polygons found"**: export your Google Earth Pro digitising to KML/KMZ and put it in `data/labels/manual/`.
- **County download fails**: save a Laikipia County GeoJSON (WGS84) as `data/aoi/laikipia_county.geojson` manually.
- **Whole county**: set `imagery.bbox: null` (or use `predict --county`). The county is roughly 13,000 × 12,000 px: many GB of download and a long run. Run `python -m farmmap info --county` first.
