# Laikipia Farm Digitisation

Automates what you do by hand in Google Earth Pro: **satellite image → trained model → one polygon per farm**, exported as KMZ (opens in Google Earth Pro) and GeoPackage (QGIS). The model can be **retrained** with new images and your corrections.

```
Sentinel-2 (free) → cloud-free composite → U-Net: background / farm interior / farm boundary
                                                  ▲
          your manual polygons (315 properties) ──┘ = training labels
                                                  │
                    probabilities → one polygon per farm → .kmz / .gpkg
```

## Read this first: what to expect

- **Your polygons are the training data.** `data/aoi/property_boundaries/Property boundaries.shp` (315 properties; median ≈ 940 ha, 10th percentile ≈ 48 ha) is turned into masks. The model learns what a boundary looks like from *your* digitising.
- **Free 10 m Sentinel-2 is adequate at this property size** (a median property is ≈ 300 × 300 px). It is *not* adequate for plots under ~5 ha. For those, use higher-resolution imagery you are licensed to use (`imagery.source: local`).
- **A boundary can only be found if something visible marks it**: a fence line, road, river, or change of land use or vegetation. Adjacent properties with identical savannah on both sides have no boundary in the image, and no model can find it. Expect merged polygons there. Treat the output as **candidates to review**; correcting is much faster than drawing from scratch.
- **Do not trust any accuracy number until `evaluate` has been run on real imagery.** It scores only on map blocks the model never trained on.
- Imagery is **not** taken from Google Earth (terms of service). Sentinel-2 comes from Microsoft Planetary Computer: free, no account or API key.
- The code was written and syntax-checked but **has not been run end-to-end by its author on real imagery**. Step 3 below (`demo` and `pytest`) exists so you find out immediately, offline, whether the install works.

## Setup on Windows (Intel Core i7)

1. **Install Python 3.11** (64-bit) from <https://www.python.org/downloads/windows/>. Tick **“Add python.exe to PATH”**. Python 3.10–3.12 work; 3.13 is not yet supported by all dependencies.
2. **Copy this whole `laikipia-farm-mapper` folder** to the laptop (USB or cloud). Include `data/aoi/property_boundaries/`: it is not committed to git, so a `git clone` alone would miss it.
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
| A | Look at your polygons, download Sentinel-2, build labels, see the train/val/test split |
| B | The model and training, with loss and IoU curves |
| C | Evaluation on unseen blocks, plus a visual model-vs-yours comparison |
| D | Predict an area → KMZ / GeoPackage |
| E | **Retraining** with new images and your corrections, comparing old vs new before adopting |

## Command line (same steps)

Run from the project folder with the venv active. Settings are in `config.yaml`.

| Command | Purpose |
|---|---|
| `python -m farmmap info` | Download size for the configured area |
| `python -m farmmap fetch` | Cloud-free composite → `data/imagery/laikipia_s2.tif` (resumable) |
| `python -m farmmap labels` | Rebuild label masks for every scene in `data/imagery/` |
| `python -m farmmap train` | Train from scratch → `data/models/unet_best.pt` |
| `python -m farmmap evaluate` | Metrics on held-out test blocks |
| `python -m farmmap predict --bbox 36.8 0.2 36.9 0.3 --name test1` | Image → polygons (`outputs/test1/`) |
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
3. **Correct or add polygons** as you already do. Save as KML/KMZ into `data/labels/manual/`. (`.shp`, `.gpkg`, `.geojson` also work.)
   - A polygon you draw **replaces** any earlier polygon it covers by more than half.
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
data/aoi/property_boundaries/        your reference polygons
data/imagery/                        downloaded scenes (generated)
data/labels/manual/                  put corrections / new polygons here
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
- **Whole county**: set `imagery.bbox: null`. This is roughly 13,000 × 12,000 px: many GB of download and a long run. Run `python -m farmmap info` first.
