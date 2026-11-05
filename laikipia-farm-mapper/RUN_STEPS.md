# Running the pipeline, step by step (Sentinel-2, no SDX)

This route needs no SDX and no Vantor imagery. The main image is a **cloud-free Sentinel-2 composite** (10 m)
from Digital Earth Africa: free, no account, downloaded by the pipeline itself.

Every command uses `--config config_s2.yaml`, which keeps this route's files separate (`data/imagery/raw_s2`,
`data/tiles_s2`, `data/models_s2`, `outputs/s2`). Run everything from the `laikipia-farm-mapper/` folder.

**What to expect at 10 m:** a 1 ha field is about 100 pixels, but a fence or hedge between two plots is usually
not visible. The model will do well on larger fields and blocks of cultivation, and will miss or merge small
adjacent plots. The many-date context layers (step 4) help it tell fields apart by how they change through the
year.

## Step 0: Setup (once)

```bash
cd ~/Desktop/Farm_Digitisation/laikipia-farm-mapper
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m farmmap demo                # must end with "Demo OK"
```

## Step 1: Check the labels and build the working area

```bash
python -m farmmap labels --config config_s2.yaml   # parcel counts and sizes in pixels at 5 m
python -m farmmap aoi                              # working area -> data/aoi/
```

## Step 2: Download the Sentinel-2 image (automatic)

```bash
python -m farmmap sentinel2 --period 2025-H1 --config config_s2.yaml
```

- `--period`: `2025-H1` (Jan–Jun), `2025-H2` (Jul–Dec) or `2025` (whole year). Use the season when fields
  stand out best. In Laikipia that is usually during or just after the long rains (H1).
- It writes `data/imagery/raw_s2/s2_geomedian_20250101_H1.tif`, about 30 s per 10 km block. If it stops, run
  the same command again and it continues where it stopped.

## Step 3: Prepare the image

```bash
python -m farmmap ingest --config config_s2.yaml   # -> data/imagery/scenes_s2/ and mosaic_s2.vrt (5 m)
python -m farmmap check  --config config_s2.yaml   # are the parcel outlines on visible edges?
```

The check gives each scheme one of three results:

- **OK:** carry on.
- **SHIFTED:** copy the values from `outputs/s2/registration_shifts.yaml` under `labels:` in `config.yaml`,
  then run `check` again.
- **UNCLEAR:** normal at 10 m for schemes of small plots. Carry on, or leave that scheme out of training by
  adding it to `labels.drop.SCHEME` in `config.yaml`.

Shifts under 10 m (one Sentinel-2 pixel) can't be measured reliably at this resolution, so treat them as OK.
Don't apply a suggested shift of 40 m or more: that is the edge of the search range, so it isn't a real
measurement. Drop that scheme instead.

## Step 4: Context layers (the time information)

```bash
python -m farmmap context --sources dea --config config_s2.yaml
```

This writes `data/context/dea.tif`: Jan–Jun and Jul–Dec greenness, change through the year, and cropland
probability. Optional, with Earth Engine:

```bash
earthengine authenticate          # once
python -m farmmap context --sources gee --config config_s2.yaml --set gee.project=venture-intern-africa
```

## Step 5: Make the training chips

Three experiments, the same labels, each in its own folder:

```bash
python -m farmmap tiles --config config_s2.yaml --set paths.tiles=data/tiles_s2_E0
python -m farmmap tiles --config config_s2.yaml --set 'context.use=["dea"]' paths.tiles=data/tiles_s2_E1
python -m farmmap tiles --config config_s2.yaml --set 'context.use=["dea","gee"]' paths.tiles=data/tiles_s2_E2
```

| Exp. | Model sees |
|---|---|
| E0 | Sentinel-2 image only |
| E1 | image + DE Africa context |
| E2 | image + DE Africa + Earth Engine context (only if you ran the `gee` step) |

## Step 6: Train on Kaggle (GPU)

1. Zip a tiles folder, for example `data/tiles_s2_E1`, and upload it to Kaggle as a **private** dataset.
2. Push the repo to GitHub, because the notebook clones it.
3. Open `notebooks/kaggle_train.ipynb` and turn **GPU and Internet on**.
4. In the config cell, set `CONFIG = "config_s2.yaml"`. Add the dataset, then **Save & Run All**.
5. Download `best.pt` into a matching folder, for example `data/models_s2_E1/best.pt`.

Repeat for E0 and E2.

## Step 7: Predict and score on the test blocks (laptop)

```bash
python -m farmmap run --config config_s2.yaml --split test --model data/models_s2_E1/best.pt --out outputs/s2/farms_E1.gpkg
python -m farmmap evaluate --config config_s2.yaml --pred outputs/s2/farms_E1.gpkg
```

Compare the F1 score and recall by farm size of E0, E1 and E2. Keep the simplest setup that scores within about
0.02 F1 of the best.

Optional: drop predicted "farms" that are not cropland, without retraining:

```bash
python -m farmmap run --config config_s2.yaml --split test --model data/models_s2_E1/best.pt \
  --out outputs/s2/farms_E1f.gpkg --set 'polygonize.context_filter={band: "ctx:dea:crop_prob", min: 10}'
python -m farmmap evaluate --config config_s2.yaml --pred outputs/s2/farms_E1f.gpkg
```

## Step 8: Map the whole area

```bash
python -m farmmap run --config config_s2.yaml --model data/models_s2_<best>/best.pt
```

This writes `outputs/s2/farms_all.gpkg` for QGIS and `outputs/s2/farms_all.kmz` for Google Earth Pro. Review the
orange polygons first: their score is below 0.6.

---

## Getting sharper imagery later

Nothing above has to be redone. The same steps work with a sharper image; only the config changes:

- **Planet NICFI (4.77 m):** free sign-up with Planet's NICFI programme, non-commercial use. Once your Earth
  Engine account has access: `python -m farmmap nicfi --month 2025-06 --config config_nicfi.yaml`, then steps
  3–8 with `--config config_nicfi.yaml`.
- **Legion or other 0.3–0.5 m imagery:** put the files in `data/imagery/raw/` and run steps 3–8 with the
  default `config.yaml` (see the README).
