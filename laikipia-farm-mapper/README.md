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

## Small plots: use higher-resolution imagery (the real fix)

At 10 m, plots under about 0.5 ha are 3–7 pixels wide and cannot be separated by any model. Roughly half of your parcels are in that range. With imagery of **about 3 m or finer** the same pipeline works on them. Everything (labels, training, `check`, `diagnose`, `compare`, retraining) already accepts your own GeoTIFF; the steps below are the whole procedure.

### 1. Get imagery you are licensed to use

Any of these works as long as you have it as a GeoTIFF **and the licence allows analysis and training a model on it**:

| Source | Typical pixel size | Notes |
|---|---|---|
| Drone orthomosaic (you or a partner fly it) | 3–10 cm | Best detail, no licence worries, limited area. Resample to 0.5–1 m for speed |
| Commercial satellite (Planet, Maxar, Airbus or a reseller) | 0.3–3 m | Paid; ask for an *orthorectified, 3-band or 4-band GeoTIFF* and confirm the licence covers ML/derived polygons |
| Aerial survey / government or NGO orthophotos | 0.1–1 m | Check who owns it and what you may do with it |

Do **not** screenshot or export imagery from Google Earth Pro or Google Maps for this: their terms of service do not allow it.

Cover the area where your parcels are (the notebook map shows it). A pilot of 5 × 5 km is plenty to start, and keeps files small.

### 2. Prepare the file (CRS, bands, size)

The pipeline needs a **projected CRS in metres** (EPSG:32637), no alpha band, and a sensible size. This command does all three and works on large files in chunks:

```bash
# keep bands 1-3 (drops an alpha band), reproject to UTM 37N, resample to 2 m pixels
python -m farmmap prepare-imagery path/to/your_ortho.tif data/imagery_hr/ortho.tif --bands 1 2 3 --resolution 2
```

- Typical band orders: RGB = `1 2 3`; RGB + NIR = `1 2 3 4` (check with your provider).
- **Choosing the pixel size:** 1–3 m is the sweet spot. Finer than 0.5 m makes files huge and slows CPU training with no benefit for parcels of 30–100 m. 8-bit (0–255) and 16-bit files are both fine.
- Make one file per area. All files you use together must have the **same pixel size and number of bands**.

### 3. Configure

`config_highres.yaml` is a ready template in separate folders, so it never mixes with the Sentinel-2 project. Open it and edit the lines marked `<-- EDIT`:

| Setting | 3 m pixels | 1 m | 0.5 m |
|---|---|---|---|
| `project.resolution_m` | 3 | 1 | 0.5 |
| `labels.boundary_width_px` | 2 | 5 | 10 |
| `split.block_px` (about 3 km blocks) | 1024 | 3000 | 6000 |
| `polygonize.simplify_m` | 1.5 | 1 | 0.5 |

Also set `imagery.local_path` to your prepared file and `imagery.local_bands` to its band order, for example `[R, G, B]` or `[R, G, B, NIR]` (with NIR the model gets an NDVI channel).

### 4. Run it

Add `--config config_highres.yaml` **before** the command name. In a terminal:

```bash
python -m farmmap --config config_highres.yaml labels      # your polygons -> masks on this imagery
python -m farmmap --config config_highres.yaml check       # are your outlines registered to THIS imagery?
python -m farmmap --config config_highres.yaml diagnose    # imagery stats + baseline + memorise-one-batch
python -m farmmap --config config_highres.yaml train       # train the U-Net
python -m farmmap --config config_highres.yaml evaluate
python -m farmmap --config config_highres.yaml compare     # U-Net vs Delineate Anything (needs requirements-da.txt)
python -m farmmap --config config_highres.yaml predict --imagery data/imagery_hr/ortho.tif --name hr_pilot
```

**In the notebook:** change the first cell that loads the config to

```python
cfg = load_config("config_highres.yaml")
```

and run from the top. Section A4 (download) is not needed: your file is the imagery. Start at **A1**, then **A4b**, **A5**, **A6**, **B0**, and on.

### 5. What to check, in order

1. **`check`**: with sharp imagery the outline-to-edge match is much more telling than at 10 m. If scheme-layer outlines are shifted by several metres, set `reference.filters` to `SOURCE: [Digitized]` for this config (traced on imagery) and re-run `labels`.
2. **`diagnose`**: must say the data carries signal before you train.
3. **`compare`**: at 1–3 m, Delineate Anything with `upsample: 1` is a serious contender and needs no training. Look at the recall-by-size columns.
4. **Retrain/fine-tune** as in the retraining section, adding more areas as scenes in `data/imagery_hr/`.

### Practical limits

- Your parcels outside the area you have imagery for are simply not used (areas without imagery are ignored).
- Very large rasters make `predict` and polygon extraction memory hungry (they hold a probability map in RAM). For areas over about 20,000 × 20,000 pixels, split the imagery into several files, or resample coarser.
- Plot-boundary fences, hedges and tracks are visible at 1–3 m; boundaries that exist only on paper (same crop on both sides, no fence) are still not detectable at any resolution.

## Engines: your U-Net vs Delineate Anything

Two ways to turn imagery into polygons are built in, and `compare` scores them on the **same held-out test blocks with the same scorer**, so you choose with evidence:

| Engine | What it is | Needs training? |
|---|---|---|
| `unet` | The U-Net in this project, learned from your parcels | Yes |
| `delineate_anything` | [Delineate Anything v2](https://github.com/Lavreniuk/Delineate-Anything): a pretrained field-instance model (YOLO segmentation) trained on 0.25–10 m imagery. Gives one polygon per field directly | No (zero-shot) |

```powershell
pip install -r requirements-da.txt              # adds ultralytics; weights (~125 MB) download on first use
python -m farmmap da-predict --name da_pilot     # polygons + KMZ from the composite
python -m farmmap compare                        # U-Net and DA at upsample x1, x2, x4 on the test blocks
python -m farmmap compare --only SOURCE=Digitized   # judge on the kind of farm you still draw by hand
```

`compare` writes `outputs/compare.csv` with F1, precision, recall, mean IoU and **recall by farm size** (<0.25 ha, 0.25–1, 1–4, >4 ha). Read the size columns: they show exactly where each engine stops working.

Notes:
- Delineate Anything's code and weights are **AGPL-3.0** (fine for internal analysis; read the licence before distributing a service built on it).
- Sentinel-2 is resampled up before detection (`delineate.upsample`); the right factor is data dependent, which is why `compare` tries several.
- The official script (`delineate.py` in its repo) can be run separately; score its GeoPackage with `compare --polygons theirs.gpkg`.

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
config.yaml                          all settings (Sentinel-2 project)
config_highres.yaml                  template for your own higher-resolution imagery
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

- **Training loss flat (~1.2) and IoU 0.000 for interior/boundary**: the model learned only class frequencies. Run `python -m farmmap diagnose` (notebook step B0). It separates *degenerate imagery*, *a code bug*, and *no usable signal at 10 m* and prints a verdict. Do not just train longer.

- **No scenes / empty composite**: widen `imagery.date_range` or raise `max_cloud_cover` (Laikipia is clearest Jan–Mar and Jul–Sep).
- **Download interrupted**: run `fetch` again; finished chunks are skipped.
- **Out of memory**: lower `train.batch_size` and `train.crop_px`, or use a smaller `imagery.bbox`.
- **Pretrained weights fail to download (offline)**: set `model.encoder_weights: null`.
- **Training too slow on CPU**: lower `train.epochs` / `train.steps_per_epoch`, keep `encoder: resnet18`, shrink the `bbox`.
- **`pip install` fails on rasterio/geopandas**: confirm Python is 3.10–3.12 64-bit (`python --version`).
- **"No farm polygons found"**: export your Google Earth Pro digitising to KML/KMZ and put it in `data/labels/manual/`.
- **County download fails**: save a Laikipia County GeoJSON (WGS84) as `data/aoi/laikipia_county.geojson` manually.
- **Whole county**: set `imagery.bbox: null` (or use `predict --county`). The county is roughly 13,000 × 12,000 px: many GB of download and a long run. Run `python -m farmmap info --county` first.
