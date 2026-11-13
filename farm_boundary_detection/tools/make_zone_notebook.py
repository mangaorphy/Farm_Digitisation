"""Writes notebooks/kaggle_zone.ipynb (run: python tools/make_zone_notebook.py)."""
import json
import sys
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# One digitising area: model comparison by cross-validation, then the farm map (Kaggle)

Give it an area drawn in Google Earth Pro (a KMZ in `data/zones/`) and the private dataset with the Planet scenes over
it (+ the round-2 model). It compares training recipes on **every** digitised farm of the area, picks the best one
only if the gain is real, and maps the area.

**What is compared** (all start from the round-2 model; the round-2 model itself is the one to beat):

| Candidate | Input | Change | Why |
|---|---|---|---|
| `R2_as_is` | 1 date | round-2 model, no training | best so far (round 4: object F1 0.334 on 1,128 Salama farms) |
| `C1_separation` | 1 date | fine-tuned with the separation target | round 4: fewer merged farms than plain fine-tuning (0.325) |
| `D1_two_dates` | **2 dates** (16 bands) | July + September stacked as one input, + separation | neighbouring plots that look alike on one date often differ in crop / planting time (U-TAE, PRUE, Fields of the Planet use two seasons) |
| `D2_two_dates_small` | **2 dates** | + small farms weigh up to 4x in the loss | 45% of Salama farms are under 0.25 ha, and only 1-10% of them were found |

The two-date models start from the round-2 model with its first-layer filters copied to both dates, and are trained
with the dates randomly swapped or one blanked, so they use both.

**How they are compared: spatial K-fold cross-validation.** The area's ground blocks are dealt into K folds with
similar numbers of farms. Each candidate is trained K times, each time on K-1 folds, and predicts the fold it did not
see: every farm is scored once, by a model that never saw it. (Round 3 picked its model on 4 validation patches, and
picked the wrong one.) Training uses a fixed number of epochs and the last epoch, so the scored fold never chooses the
model. A candidate replaces the baseline only if it is better overall **and** in most folds.

**Map:** the K models of the chosen recipe are averaged; every date is predicted and averaged; polygons get straight
edges and a 1 m gap; digitised farms are kept as they are and model farms are added only where they are new.

**Inputs:** *Add Input → Your Work → Datasets →* the area's dataset (e.g. `salama-area-planetscope`).
**Settings:** *GPU T4 x2*, *Internet On*. Run with **Save Version → Save & Run All (Commit)**.
""")

md("## 1. Settings")
code(r'''
from pathlib import Path

DATASET_NAME = "salama-area-planetscope"   # area imagery + round-2 model (name, user/name or full path)
REPO = "https://github.com/mangaorphy/Farm_Digitisation.git"
BRANCH = "main"
PROJECT_SUBDIR = "farm_boundary_detection"
ZONE_FILE = "data/zones/salama_area.kmz"   # the area polygon (+ optional work tiles T01..)
ZONE_NAME = "salama"                       # used in output file names

QUICK_TEST = False        # True = a short dry run (2 folds, 1 epoch, a few patches, one tile)
TIME_BUDGET_H = 7.0       # whole notebook
K_FOLDS = 4
BLOCK_SIZES_M = [2000, 1500, 1000]   # the first that gives >= 3 farm blocks per fold is used
EPOCHS = 15               # fixed (no early stopping): the last epoch is scored
USE_TTA = True            # flip averaging when scoring and mapping
MAX_DATES = 2             # two-date candidates: the dates with the most scenes (scenes of one date are mosaicked)
MIN_GAIN = 0.01           # a candidate must beat the baseline by this object F1 ...
MIN_FOLD_WINS = 0.75      # ... and in at least this share of the folds
MAP_TILES = None          # None = every tile; or e.g. ["T04", "T11"]
MIN_COMPACT = 0.15        # model polygons less compact than this (ragged blobs, slivers) are dropped
MAX_FARM_HA = None        # model polygons larger than this are dropped; None = 99th percentile of the area's farms

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
import time
T0 = time.time()
''')

md("## 2. Install and get the code")
code(r'''
import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio", "geopandas", "pyogrio", "scikit-image",
                "shapely>=2.1"], check=True)
code_dir = WORK / "Farm_Digitisation"
if not code_dir.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, str(code_dir)], check=True)
else:
    subprocess.run(["git", "-C", str(code_dir), "pull", "--ff-only"], check=False)
PROJECT = code_dir / PROJECT_SUBDIR
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)
import torch, src
print("torch", torch.__version__, "| GPUs:",
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "NONE - turn on the GPU!")
''')

md(r"""
## 3. Area imagery, the area's farms and configuration

Only the farms within the area (+ 500 m) are used, written to a small labels file.
""")
code(r'''
import json, shutil, logging, zipfile
import numpy as np, pandas as pd, geopandas as gpd
from IPython.display import Image, display
from src.utils.helpers import load_config, setup_logging
from src.models.multi_task_unet import load_checkpoint
from src.data.ingestion import load_farm_polygons
from src.gis.zone import read_zone

def find_dataset(name):
    p = Path(name)
    if p.is_absolute() and p.is_dir():
        return p
    hits = sorted((d for d in INPUT.rglob("*") if d.is_dir() and d.name == p.name), key=lambda d: len(d.parts))
    if hits:
        return hits[0]
    sr = sorted(INPUT.rglob("*AnalyticMS_SR*.tif"))
    return sr[0].parent if sr else None

DATA = find_dataset(DATASET_NAME)
assert DATA, f"dataset {DATASET_NAME!r} not found under {INPUT}: attach it with Add Input"
tifs = sorted(p for p in DATA.rglob("*.tif") if "udm" not in p.name.lower())
models_found = sorted(DATA.rglob("*E1_r50*best_model.pth")) or sorted(INPUT.rglob("*best_model.pth"))
assert tifs, f"no scenes in {DATA}"
assert models_found, "the round-2 model (*best_model.pth) is not in the dataset"
BASE_MODEL = models_found[0]
mk = load_checkpoint(BASE_MODEL, "cpu")["model_kwargs"]
print(f"{len(tifs)} scenes:", *[t.name for t in tifs], sep="\n  ")
print("round-2 model:", BASE_MODEL.name, {k: mk[k] for k in ("encoder", "decoder_channels")})

SCRATCH = next((Path(d) / "farmseg" for d in ("/kaggle/temp", "/kaggle/tmp") if Path(d).is_dir()), WORK / "scratch")
setup_logging(WORK / "logs", "INFO", "zone")
REPORT = WORK / "report"; REPORT.mkdir(parents=True, exist_ok=True)

zone, tiles = read_zone(PROJECT / ZONE_FILE)
utm = zone.estimate_utm_crs()
labels_all, _ = load_farm_polygons(load_config(PROJECT / "configs" / "config.yaml"))
zl = labels_all.to_crs(utm)
zl = zl[zl.intersects(zone.to_crs(utm).buffer(500).iloc[0])]
keep_cols = [c for c in ("FARM_ID", "SOURCE", "QA_FLAG", "SCHEME", "name") if c in zl]
ZONE_LABELS = WORK / "zone_labels"; shutil.rmtree(ZONE_LABELS, ignore_errors=True); ZONE_LABELS.mkdir()
zl[keep_cols + ["geometry"]].to_crs(4326).to_file(ZONE_LABELS / "zone_farms.gpkg", driver="GPKG")
if MAX_FARM_HA is None:
    MAX_FARM_HA = round(max(5.0, float(zl.area.quantile(0.99)) / 1e4), 1)
print(f"{len(zl)} digitised farms in the area (+500 m), median {zl.area.median() / 1e4:.2f} ha; model polygons above "
      f"{MAX_FARM_HA} ha or below compactness {MIN_COMPACT} will be dropped")

if QUICK_TEST:
    K_FOLDS, EPOCHS = 2, 1
pts = zl.representative_point()
def n_blocks(b):
    return len(set(zip(np.floor(pts.x / b).astype(int), np.floor(pts.y / b).astype(int))))
BLOCK_M = next((b for b in BLOCK_SIZES_M if n_blocks(b) >= 3 * K_FOLDS), BLOCK_SIZES_M[-1])
print(f"cross-validation: {K_FOLDS} folds of {BLOCK_M} m blocks ({n_blocks(BLOCK_M)} blocks with farms)")

# optional: polygons over land with NO farms (or where every farm is digitised): there, "no polygon" is trusted
# as "not a farm", which teaches the model to leave bush alone (see data/complete_areas/README.md)
NO_FARM = next(iter(sorted((PROJECT / "data" / "complete_areas").glob("*.km[lz]"))), None)
NEG = [f"paths.complete_areas={NO_FARM}", "patches.background_keep_fraction=0.5"] if NO_FARM else []
print("no-farm / complete areas:", NO_FARM.name if NO_FARM else "none")

QUICK = ["train.max_train_patches=16", "evaluation.n_visuals=2"] if QUICK_TEST else []
BASE = [f"paths.labels_dir={ZONE_LABELS}", f"split.block_size_m={BLOCK_M}",
        "split.fractions={train: 0.75, val: 0.25, test: 0.0}",
        "patches.size=256", "patches.stride=128", "patches.eval_stride=256", "patches.min_farm_pixels=150",
        "inference.patch_size=512", "inference.overlap=128", "inference.batch_size=4",
        f"model.encoder={mk['encoder']}", f"model.decoder_channels={list(mk['decoder_channels'])}",
        f"model.full_resolution_branch={str(bool(mk['full_resolution_branch'])).lower()}",
        f"model.psp_pooling={str(bool(mk['psp_pooling'])).lower()}",
        f"model.condition_interior_on_boundary={str(bool(mk['condition_interior_on_boundary'])).lower()}",
        "train.num_workers=4", f"train.epochs={EPOCHS}", "train.patience=null", "train.run_checks=false", *NEG]
FT = [f"train.init_from={BASE_MODEL}", "train.lr=0.0001", "train.warmup_epochs=1", "train.batch_size=16",
      "loss.separation.enabled=true"]
TWO = ["augmentation.date_bands=8", "augmentation.swap_dates=0.5", "augmentation.date_dropout=0.15"]
# candidate -> (input, overrides); None = the round-2 model as it is
CANDIDATES = {
    "R2_as_is":           ("single", None),
    "C1_separation":      ("single", FT),
    "D1_two_dates":       ("stack", FT + TWO),
    "D2_two_dates_small": ("stack", FT + TWO + ["loss.small_farm.enabled=true"]),
}
BASELINE = "R2_as_is"          # the model to beat
STACK_DIR = SCRATCH / "stack"  # the two-date image (built in step 4)

def make_cfg(fold, cand=None, extra=(), prep=None):
    prep = prep or (CANDIDATES[cand][0] if cand else "single")
    run = WORK / "experiments" / (cand or f"_{prep}") / f"fold{fold}"
    over = BASE + [f"paths.imagery_dir={DATA if prep == 'single' else STACK_DIR}",
                   f"paths.processed_dir={SCRATCH / 'prep' / prep / f'fold{fold}'}", f"paths.models_dir={run / 'models'}",
                   f"paths.outputs_dir={run / 'outputs'}", f"paths.logs_dir={WORK / 'logs'}",
                   *((CANDIDATES[cand][1] or []) if cand else []), *extra, *QUICK]
    return load_config(PROJECT / "configs" / "config.yaml", over)

def hours_left():
    return TIME_BUDGET_H - (time.time() - T0) / 3600
''')

md(r"""
## 4. The two-date image, then the folds

The scenes are grouped by date and stacked into one 16-band image (date 1 bands, then date 2 bands; a pixel cloudy
or missing on either date is left out). Masks are built once per input; then for each fold the patches are cut so
training and validation never share ground. Both inputs use the same folds, so every candidate is scored on the same
farms.
""")
code(r'''
from src.data.stack import stack_dates
from src.training.crossval import prepare_folds
PREPS = sorted({CANDIDATES[c][0] for c in CANDIDATES})
STACK = None
if "stack" in PREPS:
    STACK = stack_dates(tifs, STACK_DIR, f"{ZONE_NAME}2d", max_dates=MAX_DATES)
fold_tables = {prep: prepare_folds(lambda f, prep=prep: make_cfg(f, prep=prep), K_FOLDS) for prep in PREPS}
for prep, t in fold_tables.items():
    print(prep); display(t)
meta0 = SCRATCH / "prep" / "single" / "fold0" / "metadata"
checks = pd.DataFrame(json.loads((meta0 / "validation_report.json").read_text()))[["check", "ok", "message"]]
display(checks)
vis = SCRATCH / "prep" / "single" / "fold0" / "visualizations"
mask_figs = [shutil.copy(f, REPORT / f.name) for f in sorted(vis.glob("*_masks.png"))[:2]]
for f in mask_figs:
    display(Image(filename=str(f), width=1300))
fold_table = pd.concat(fold_tables, names=["input"])
print(f"{hours_left():.1f} h left")
''')

md("## 5. Train every candidate on every fold (fixed epochs, last epoch kept)")
code(r'''
from src.training.crossval import train_fold
from src.utils.visualization import plot_history, plot_training_curves

models = {}
runs = [(c, f) for c in CANDIDATES if CANDIDATES[c][1] is not None for f in range(K_FOLDS)]
for i, (cand, f) in enumerate(runs):
    budget = (hours_left() - 1.5) / (len(runs) - i)            # keep 1.5 h for scoring and the map
    if budget < 0.05 and not QUICK_TEST:
        print(f"{cand} fold {f}: skipped - no time left"); continue
    cfg = make_cfg(f, cand, [f"train.max_hours={max(budget, 0.05):.2f}"])
    print(f"\n===== {cand} fold {f} ({budget * 60:.0f} min max) =====")
    models.setdefault(cand, [None] * K_FOLDS)[f] = train_fold(cfg)
    plot_history(Path(cfg.paths.models_dir) / "training_history.csv", Path(cfg.paths.models_dir) / "training_history.png")
models = {c: m for c, m in models.items() if all(m)}         # only candidates trained on every fold are compared
models = {"R2_as_is": [BASE_MODEL] * K_FOLDS, **models}
hist = {c: WORK / "experiments" / c / "fold0" / "models" / "training_history.csv" for c in models if c != "R2_as_is"}
curves_png = plot_training_curves(hist, REPORT / "training_curves.png")
display(Image(filename=str(curves_png), width=1400))
print("trained on every fold:", list(models), f"| {hours_left():.1f} h left")
''')

md(r"""
## 6. Compare on every farm (out-of-fold)

Each candidate's K models predict the fold they did not see (with TTA); its polygon post-processing is tuned on the
pooled predictions, and it is scored on **individual farms**: a farm counts only if one polygon matches it with
IoU ≥ 0.5, so merged and split farms are both errors.
""")
code(r'''
from src.training.crossval import out_of_fold, per_fold_scores, choose, save_summary
from src.training.evaluation import tune_postprocess, evaluate, true_areas
from src.utils.visualization import plot_comparison

fold_cfgs = {prep: [make_cfg(f, prep=prep) for f in range(K_FOLDS)] for prep in PREPS}
AREAS = true_areas(fold_cfgs["single"][0])
SMALL = dict(stage1={"seed_source": ["boundary", "edt"], "seed_threshold": [0.5], "min_seed_px": [25]},
             stage2={"interior_threshold": [0.4], "boundary_threshold": [0.5], "min_area_m2": [200]}) if QUICK_TEST else {}
rows, tuned, fold_f1 = {}, {}, {}
for cand, mlist in models.items():
    fc = fold_cfgs[CANDIDATES[cand][0]]
    oof = out_of_fold(fc, mlist, USE_TTA)
    pc, _ = tune_postprocess(fc[0], oof, SMALL.get("stage1"), SMALL.get("stage2"))
    res = evaluate(fc[0], None, "out-of-fold", WORK / "evaluation" / cand, postprocess=pc, cache=oof,
                   areas=AREAS, title=cand)
    folds = per_fold_scores(oof, pc, float(fc[0].evaluation.instance_iou))
    small = res["by_size"]
    rows_small = {r["size"]: r["recall"] for r in small} if small and "size" in small[0] else {}
    tuned[cand], fold_f1[cand] = pc, folds["object_f1"]
    rows[cand] = {**{k: res[k] for k in ("object_f1", "object_precision", "object_recall", "mean_matched_iou",
                                          "merged_predictions", "split_farms", "interior_iou", "boundary_f1",
                                          "farms_true")},
                  "f1_fold_min": folds["object_f1"].min(), "f1_fold_max": folds["object_f1"].max(),
                  "recall_<0.1ha": rows_small.get("<0.1 ha"), "recall_0.1-0.25ha": rows_small.get("0.1-0.25 ha"),
                  **{f"pp_{k}": pc[k] for k in ("seed_source", "seed_threshold", "min_seed_px", "interior_threshold",
                                                "min_area_m2")}}
    oof["items"].clear()
comparison = pd.DataFrame(rows).T
folds_f1 = pd.DataFrame(fold_f1)
CHOSEN = choose(comparison, folds_f1, BASELINE, MIN_GAIN, MIN_FOLD_WINS)
save_summary(WORK, comparison, folds_f1, CHOSEN)
comparison.to_csv(WORK / "comparison_out_of_fold.csv")
comp_png = plot_comparison(comparison, REPORT / "comparison.png")
print(f"CHOSEN = {CHOSEN} | out-of-fold object F1 {float(comparison.loc[CHOSEN, 'object_f1']):.3f}",
      {c: round(float(v), 3) for c, v in comparison["object_f1"].items()})
display(comparison, folds_f1.round(3).rename_axis("object F1 per fold"))
display(Image(filename=str(comp_png), width=1100))
cdir = WORK / "evaluation" / CHOSEN
display(pd.read_csv(cdir / "by_size.csv").set_index("size"))
for f in (cdir / "by_size.png", cdir / "confusion.png"):
    display(Image(filename=str(f), width=1100))
''')

md(r"""
## 7. Map the area with the chosen recipe

The K models of the chosen recipe are averaged (for `R2_as_is`: the round-2 model). **Magenta** = digitised (kept
as is), **green** = model, likely farm, **orange** = model, check first.
""")
code(r'''
from src.gis.zone import run_zone
pp = tuned[CHOSEN]
cfg_map = make_cfg(0, CHOSEN)
cfg_map.postprocess = pp
cfg_map.inference.tta = USE_TTA
MAP_MODELS = models[CHOSEN] if CHOSEN != "R2_as_is" else BASE_MODEL
MAP_IMAGES = [STACK] if CANDIDATES[CHOSEN][0] == "stack" else tifs
zres = run_zone(cfg_map, MAP_MODELS, PROJECT / ZONE_FILE, MAP_IMAGES, WORK / "zone", ZONE_NAME, digitised=labels_all,
                only_tiles=(tiles.tile.iloc[:1].tolist() if QUICK_TEST else MAP_TILES),
                max_area_ha=MAX_FARM_HA, min_compact=MIN_COMPACT)
farms = zres["farms"]
per_tile = zres["counts"]
display(per_tile)
model_f = farms[farms.source == "Model"]
print(f"{len(farms)} farms: {int((farms.source == 'Digitized').sum())} digitised (kept) + {len(model_f)} from the model "
      f"({int((model_f.review == 'likely farm').sum())} likely, {int((model_f.review == 'check').sum())} to check)")
print({k: str(v) for k, v in zres["paths"].items()})
''')

code(r'''
import rasterio, matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from src.utils.visualization import stretch
preview_png = REPORT / "map_preview.png"
busy = farms.tile.value_counts()
busy = busy[busy.index.str.match(r"T\d+")]
if len(busy):
    t_name = busy.index[0]
    best = None                                          # the scene with most data around the tile centre
    for t in tifs:
        with rasterio.open(t) as s:
            c = tiles[tiles.tile == t_name].to_crs(s.crs).geometry.iloc[0].centroid
            b = (c.x - 1000, c.y - 1000, c.x + 1000, c.y + 1000)
            arr = s.read([6, 4, 2] if s.count == 8 else [3, 2, 1], window=from_bounds(*b, s.transform),
                         boundless=True, fill_value=0)
            valid = float((arr.max(0) > 0).mean())
            if best is None or valid > best[0]:
                best = (valid, arr, b, s.crs, t.name)
    valid, rgb, b, crs, img_name = best
    f = farms.to_crs(crs).cx[b[0]:b[2], b[1]:b[3]]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(stretch(rgb), extent=(b[0], b[2], b[1], b[3]))
    for src_, rev, col in (("Digitized", "existing", "magenta"), ("Model", "likely farm", "lime"), ("Model", "check", "orange")):
        g = f[(f.source == src_) & (f.review == rev)]
        if len(g):
            g.boundary.plot(ax=ax, color=col, linewidth=0.9)
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3]); ax.axis("off")
    ax.set_title(f"{t_name}, 2 x 2 km on {img_name[:8]}: magenta digitised, green model likely, orange model check")
    fig.savefig(preview_png, dpi=100, bbox_inches="tight"); plt.show()
''')

md("## 8. Report and download")
code(r'''
from src.utils.report import build_report
for i, m in enumerate(MAP_MODELS if isinstance(MAP_MODELS, list) else [MAP_MODELS]):
    shutil.copy(m, WORK / "zone" / f"{CHOSEN}_model{i}.pth")
summary = {"chosen": CHOSEN, "k_folds": K_FOLDS, "block_m": BLOCK_M, "epochs": EPOCHS, "tta": USE_TTA,
           "postprocess": dict(pp), "out_of_fold": comparison.to_dict("index"),
           "object_f1_per_fold": folds_f1.to_dict(), "images": zres["images"], "farms": int(len(farms)),
           "model_farms": int(len(model_f))}
(WORK / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
c0 = float(comparison.loc[BASELINE, "object_f1"]) if BASELINE in comparison.index else float("nan")
report = build_report(WORK / "report.html", f"{ZONE_NAME}: recipe comparison and farm map", [
    ("Summary", [f"Chosen: <b>{CHOSEN}</b>. Out-of-fold object F1 <b>{float(comparison.loc[CHOSEN, 'object_f1']):.3f}</b> "
                 f"(baseline {c0:.3f}), scored on {int(comparison.loc[CHOSEN, 'farms_true'])} farms, each by a model that "
                 f"never saw it. Merged farms: {int(comparison.loc[CHOSEN, 'merged_predictions'])} "
                 f"(baseline {int(comparison.loc[BASELINE, 'merged_predictions']) if BASELINE in comparison.index else '-'}). "
                 f"Map: {len(farms)} farms = {int((farms.source == 'Digitized').sum())} digitised + {len(model_f)} "
                 f"from the model.", comparison]),
    ("Object F1 per fold (a real gain holds in most folds)", [folds_f1.round(3)]),
    ("Data", [fold_table, checks] + [Path(p) for p in mask_figs]),
    ("Training curves (fold 0)", [curves_png]),
    ("Comparison", [comp_png]),
    ("Chosen recipe: by farm size and confusion matrix", [pd.read_csv(cdir / "by_size.csv").set_index("size"),
                                                          cdir / "by_size.png", cdir / "confusion.png"]
                                                         + sorted((cdir / "visuals").glob("*.png"))[:3]),
    ("Map: farms per tile", [per_tile] + ([preview_png] if preview_png.exists() else [])),
])
with zipfile.ZipFile(WORK / f"{ZONE_NAME}_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in [WORK / "report.html", WORK / "summary.json", WORK / "comparison_out_of_fold.csv", WORK / "crossval.json"] + \
             [p for p in (WORK / "zone").rglob("*") if p.is_file() and "prediction_masks" not in p.parts]:
        z.write(f, f.relative_to(WORK))
shutil.rmtree(SCRATCH, ignore_errors=True)
print("report ->", report, f"| zip -> {ZONE_NAME}_results.zip |", f"{hours_left():.1f} h of budget left")
''')

md(r"""
**Download** from the version's *Output* tab: `zone/kmz/<area>_farms.kmz` (Google Earth Pro: a folder per tile;
magenta = already digitised, green = model likely farm, orange = check), `report.html` (comparison of the recipes on
every farm, per fold, by farm size), or everything in `<area>_results.zip` (KMZ, GeoPackage, Shapefile, GeoJSON,
models, summary).
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    src = c["source"].split("\n")
    c["source"] = [line + "\n" for line in src[:-1]] + [src[-1]]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_zone.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("cells:", len(cells), "->", out)
