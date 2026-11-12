import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# Farm boundaries, round 3: a network pretrained on PlanetScope fields, and polygons that cannot cross (Kaggle)

**What is new** (round 2: object F1 0.32 on validation; polygons merged several farms and crossed each other):

| Change | Why |
|---|---|
| **Fields of the Planet model** (U-Net, EfficientNet-B3) as the starting point, fine-tuned on the Laikipia farms | Published July 2026 (arXiv:2607.04449): trained on **3 m PlanetScope** and 1.6 million field polygons in 24 countries (Kenya included) for exactly this task (field interior + boundary). Rounds 1–2 started from ImageNet photos. |
| Its recipe: reflectance input, gain / tone augmentation | their largest gain came from stronger augmentation |
| **×2 enlargement** variant | small plots (most farms here are under 0.5 ha) span more pixels |
| The **round-2 network trained further** (new zone dates added) | a safeguard: the best of all candidates on the hand-traced farms is used |
| **Neck splitting** of seeds (`edt`) in the polygon step | two farms joined through a gap in a faint boundary are still separated |
| **Outlines simplified together** | a boundary shared by two farms is ONE line: neighbouring polygons cannot cross or overlap |
| Final map = **your digitised farms + new model farms only**, large / ragged blobs removed | no duplicates of digitised farms, no bush blobs |

The round-2 model and the released Fields of the Planet model (not fine-tuned) are scored on the same farms, so
the table shows what each step is worth.

**Inputs** (*Add Input → Your Work → Datasets*): `laikipia-planetscope` (training scenes) and, optionally,
`mutara-zone-planetscope` and `salama-area-planetscope` (zone scenes + the round-2 model) and any further private
dataset of Planet scenes you want mapped. Every attached scene is used for training where it has digitised
farms, and every scene is mapped. **Settings:** *GPU T4 x2*, *Internet On*. Run with **Save Version → Save & Run
All (Commit)**. The pretrained weights (CC-BY-NC, research use) are downloaded from Hugging Face.
""")

md("## 1. Settings")
code(r'''
from pathlib import Path

REPO = "https://github.com/mangaorphy/Farm_Digitisation.git"
BRANCH = "main"
PROJECT_SUBDIR = "farm_boundary_detection"
ZONES_DIR = "data/zones"   # every zone file here also gets a per-tile map, if scenes over it are attached

QUICK_TEST = False        # True = short dry run of the whole notebook on a few patches
TIME_BUDGET_H = 11.0      # whole notebook
EVAL_RESERVE_H = 2.0      # kept for comparison, test and mapping
EPOCHS = 60
PATIENCE = 10
USE_TTA = True
ENSEMBLE_TOP = 2          # average of the 2 best fine-tuned models, kept only if clearly better (0 = off)
MAP_SCENES = "all"        # "all" or a list of words that must appear in the scene file name, e.g. ["20260930"]
MIN_COMPACT = 0.15        # model polygons less compact than this are dropped (ragged blobs, slivers)
MAX_FARM_HA = None        # model polygons larger than this are dropped; None = the 99th percentile of the
                          # digitised farms (at least 5 ha). Digitised farms are never dropped.

PREP = {
    "b3_all":    ["masks.boundary_width_m=3.0"],
    "b3_traced": ["masks.boundary_width_m=3.0", "labels.keep={QA_FLAG: [OK], SOURCE: [Digitized]}"],
}
REFERENCE_PREP = "b3_traced"          # hand-traced farms: the common yardstick

FTP = ["model.arch=ftp_unet", "normalization.mode=scale", "train.lr=0.0002", "train.warmup_epochs=1",
       "augmentation.gain=0.3", "augmentation.gamma=0.15", "train.batch_size=16"]
# experiment -> (preprocessing variant, overrides); simplest first (wins ties)
EXPERIMENTS = {
    "F1_ftp":     ("b3_all", FTP),
    "F2_ftp_x2":  ("b3_all", FTP + ["model.input_upsample=2", "train.batch_size=6"]),
    # + "R3_resunet_ft" (added below when the round-2 model is attached): the round-2 network trained further
}

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
import time
T0 = time.time()
''')

md("## 2. Install and get the code")
code(r'''
import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio", "geopandas", "pyogrio", "scikit-image",
                "segmentation-models-pytorch", "shapely>=2.1"], check=True)
code_dir = WORK / "Farm_Digitisation"
if not code_dir.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, str(code_dir)], check=True)
else:
    subprocess.run(["git", "-C", str(code_dir), "pull", "--ff-only"], check=False)
PROJECT = code_dir / PROJECT_SUBDIR
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)
import torch, shapely, src
print("torch", torch.__version__, "| shapely", shapely.__version__, "| GPUs:",
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "NONE - turn on the GPU!")
''')

md("## 3. Scenes, configuration, backups")
code(r'''
import json, shutil, logging, zipfile
import pandas as pd, geopandas as gpd
from IPython.display import Image, display
from src.utils.helpers import load_config, setup_logging

# temporary files (scene links, patches): outside the saved output where Kaggle offers it
SCRATCH = next((Path(d) / "farmseg" for d in ("/kaggle/temp", "/kaggle/tmp") if Path(d).is_dir()), WORK / "scratch")
tifs_in = sorted(p for p in INPUT.rglob("*.tif") if "AnalyticMS_SR" in p.name and "udm" not in p.name.lower())
assert tifs_in, f"no Planet scenes (*AnalyticMS_SR*.tif) under {INPUT}: attach the dataset(s) with Add Input"
IMAGERY = SCRATCH / "imagery"; shutil.rmtree(IMAGERY, ignore_errors=True); IMAGERY.mkdir(parents=True)
seen = set()
for t in tifs_in:                     # one folder with every scene and its UDM2 mask (links, no copy)
    if t.name in seen:
        continue
    seen.add(t.name)
    for f in t.parent.glob(t.name.split("_3B_")[0] + "*.tif"):
        if not (IMAGERY / f.name).exists():
            os.symlink(f, IMAGERY / f.name)
tifs = sorted(p for p in IMAGERY.glob("*.tif") if "udm" not in p.name.lower())
print(f"{len(tifs)} scenes from {sorted({str(t.parent.relative_to(INPUT)) for t in tifs_in})}")
R2_MODEL = next(iter(sorted(INPUT.rglob("*E1_r50*best_model.pth"))), None)
print("round-2 model for comparison:", R2_MODEL)
if R2_MODEL:   # hedge: the round-2 network, continued on all scenes (the new zone dates included)
    EXPERIMENTS["R3_resunet_ft"] = ("b3_all", ["model.encoder=resnet50", "train.batch_size=12",
                                               f"train.init_from={R2_MODEL}", "train.lr=0.0001", "train.warmup_epochs=1"])

setup_logging(WORK / "logs", "INFO", "kaggle")
QUICK = ["train.max_train_patches=32", "train.epochs=2", "evaluation.n_visuals=3"] if QUICK_TEST else []
REPORT = WORK / "best" / "report"; REPORT.mkdir(parents=True, exist_ok=True)

# Optional: polygons drawn over land with NO farms (bush, forest, ranch) or where EVERY farm is digitised.
# Inside them, "no polygon" is trusted as "not a farm": this is what teaches the model to leave bush alone.
NO_FARM = next(iter(sorted((PROJECT / "data" / "complete_areas").glob("*.km[lz]"))), None)
NEG = [f"paths.complete_areas={NO_FARM}", "patches.background_keep_fraction=0.5"] if NO_FARM else []
print("no-farm / complete areas:", NO_FARM or "none (add data/complete_areas/no_farm_areas.kmz to the repository to use them)")

def make_cfg(prep, exp=None, extra=()):
    exp_dir = WORK / "experiments" / (exp or "_none")
    over = NEG + [f"paths.imagery_dir={IMAGERY}", f"paths.processed_dir={SCRATCH / 'prep' / prep}",
            f"paths.models_dir={exp_dir / 'models'}", f"paths.outputs_dir={exp_dir / 'outputs'}",
            f"paths.logs_dir={WORK / 'logs'}", "train.num_workers=4", f"train.epochs={EPOCHS}",
            f"train.patience={PATIENCE}", *PREP[prep], *extra, *QUICK]
    return load_config(PROJECT / "configs" / "config.yaml", over)

def hours_left():
    return TIME_BUDGET_H - (time.time() - T0) / 3600

def backup():
    """Models and histories of every experiment in one zip in the output (kept by a committed run)."""
    with zipfile.ZipFile(WORK / "results_backup.zip", "w") as z:
        for exp in EXPERIMENTS:
            mdir = WORK / "experiments" / exp / "models"
            for f in ("best_model.pth", "training_history.csv"):
                if (mdir / f).exists():
                    z.write(mdir / f, f"{exp}__{f}")

def restore(exp, mdir):
    """Continue an experiment from an earlier version of this notebook attached as input."""
    if (mdir / "latest_checkpoint.pth").exists():
        return
    hits = sorted(INPUT.rglob(f"experiments/{exp}/models/latest_checkpoint.pth"))
    if hits:
        mdir.mkdir(parents=True, exist_ok=True)
        for f in ("best_model.pth", "latest_checkpoint.pth", "training_history.csv"):
            if (hits[0].parent / f).exists():
                shutil.copy(hits[0].parent / f, mdir / f)
        print(f"{exp}: restored from {hits[0].parent}")
''')

md("## 4. Preprocess (masks, geographic split, patches, checks)")
code(r'''
from src.pipeline import prepare

first = next(iter(PREP))
for prep in PREP:
    cfg = make_cfg(prep)
    meta = Path(cfg.paths.processed_dir) / "metadata"
    if (meta / "patches.csv").exists() and (meta / "validation_report.json").exists():
        print(f"{prep}: already prepared"); continue
    if prep != first:
        meta.mkdir(parents=True, exist_ok=True)
        shutil.copy(SCRATCH / "prep" / first / "metadata" / "split_blocks.json", meta / "split_blocks.json")
    print(f"\n========== preprocessing {prep} ==========")
    prepare(cfg)

patch_tables = {p: pd.read_csv(SCRATCH / "prep" / p / "metadata" / "patches.csv") for p in PREP}
patch_summary = pd.concat({p: t.groupby("split").agg(patches=("patch_id", "size"), farms=("n_farms", "sum"),
                                                     farm_px=("farm_pixels", "sum"), boundary_px=("boundary_pixels", "sum"))
                           for p, t in patch_tables.items()})
checks = pd.DataFrame(json.loads((SCRATCH / "prep" / first / "metadata" / "validation_report.json").read_text()))[["check", "ok", "message"]]
display(patch_summary, checks)
vis = SCRATCH / "prep" / first / "visualizations"
mask_figs = []
for f in sorted(vis.glob("*_masks.png"))[:2] + [vis / "split_map.png"]:
    if f.exists():
        mask_figs.append(shutil.copy(f, REPORT / f.name)); display(Image(filename=str(f), width=1300))
print(f"{hours_left():.1f} h left")
''')

md(r"""
## 5. Baselines (no training) and fine-tuning

`P0_ftp_as_released` is the published model untouched (on these farms it finds only about a third of the farm
area: it has to be fine-tuned); `R2_resunet_as_is` is the round-2 winner. Then, under a time budget: the published
model fine-tuned on the Laikipia farms (at 3 m and enlarged ×2), and the round-2 network trained further.
""")
code(r'''
from src.training.trainer import train
from src.models.ftp_unet import save_pretrained_checkpoint
from src.utils.visualization import plot_history, plot_training_curves

best_pt = {"P0_ftp_as_released": save_pretrained_checkpoint(make_cfg(first, "P0_ftp_as_released", FTP),
                                                           WORK / "experiments" / "P0_ftp_as_released" / "models" / "best_model.pth")}
if R2_MODEL:
    best_pt["R2_resunet_as_is"] = R2_MODEL
todo = list(EXPERIMENTS)
for k, exp in enumerate(todo):
    prep, extra = EXPERIMENTS[exp]
    budget = (hours_left() - EVAL_RESERVE_H) / (len(todo) - k)
    if budget < 0.25 and not QUICK_TEST:
        print(f"{exp}: skipped - no time left (attach this version's output next time to continue)"); continue
    cfg = make_cfg(prep, exp, extra + [f"train.max_hours={max(budget, 0.25):.2f}"])
    mdir = Path(cfg.paths.models_dir)
    restore(exp, mdir)
    print(f"\n========== {exp} ({budget:.1f} h budget): {extra} ==========")
    best_pt[exp] = train(cfg)
    plot_history(mdir / "training_history.csv", mdir / "training_history.png")
    backup()
TRAINED = [e for e in EXPERIMENTS if e in best_pt]
hist = {e: WORK / "experiments" / e / "models" / "training_history.csv" for e in TRAINED}
curves_png = plot_training_curves(hist, REPORT / "training_curves.png")
display(Image(filename=str(curves_png), width=1500))
train_summary = pd.DataFrame({e: {"epochs run": len(pd.read_csv(h)), "best epoch": int(pd.read_csv(h).val_score.idxmax()),
                                  "best val_score": pd.read_csv(h).val_score.max(),
                                  "best boundary F1": pd.read_csv(h).boundary_f1.max(),
                                  "best interior IoU": pd.read_csv(h).interior_iou.max(),
                                  "hours": pd.read_csv(h).minutes.sum() / 60} for e, h in hist.items()}).T
display(train_summary)
print(f"{hours_left():.1f} h left")
''')

md(r"""
## 6. Compare on the hand-traced validation farms

Every model predicts the same validation patches (with TTA), its polygon step is tuned (seed source incl. neck
splitting, thresholds, minimum sizes) and it is scored on **individual farms**: a farm counts only if one
predicted polygon matches it with IoU ≥ 0.5, so merged and split farms both count as errors.
""")
code(r'''
from src.training.evaluation import predict_patches, tune_postprocess, evaluate, ensemble_caches, true_areas
from src.utils.visualization import plot_comparison

ref_cfg = make_cfg(REFERENCE_PREP)
AREAS = true_areas(ref_cfg)
SMALL = dict(stage1={"seed_source": ["boundary", "edt"], "seed_threshold": [0.5], "min_seed_px": [25]},
             stage2={"interior_threshold": [0.4], "boundary_threshold": [0.5], "min_area_m2": [200]}) if QUICK_TEST else {}
caches, tuned, rows = {}, {}, {}

def score(name, cache):
    pc, _ = tune_postprocess(ref_cfg, cache, SMALL.get("stage1"), SMALL.get("stage2"))
    tuned[name] = pc
    res = evaluate(ref_cfg, None, "val", WORK / "evaluation" / name / "val_reference", postprocess=pc, cache=cache,
                   areas=AREAS, title=name)
    rows[name] = {k: res[k] for k in ("object_f1", "object_precision", "object_recall", "mean_matched_iou",
                                      "merged_predictions", "split_farms", "interior_iou", "boundary_f1",
                                      "boundary_iou", "farms_true")}
    rows[name].update({f"pp_{k}": pc[k] for k in ("seed_source", "seed_threshold", "min_seed_px",
                                                  "interior_threshold", "boundary_threshold", "min_area_m2")})

for exp, m in best_pt.items():
    caches[exp] = predict_patches(ref_cfg, m, "val", tta=USE_TTA)
    score(exp, caches[exp])
f1 = pd.DataFrame(rows).T["object_f1"].astype(float)
ENS = list(f1[TRAINED].sort_values(ascending=False).index[:ENSEMBLE_TOP]) if ENSEMBLE_TOP > 1 and len(TRAINED) > 1 else []
if ENS:
    score("ENSEMBLE", ensemble_caches([caches[e] for e in ENS]))
comparison = pd.DataFrame(rows).T.loc[[e for e in list(best_pt) + ["ENSEMBLE"] if e in rows]]
comparison.to_csv(WORK / "comparison_val.csv")
f1 = comparison["object_f1"].astype(float)
top = float(f1[list(best_pt)].max())
BEST = next(e for e in best_pt if float(f1[e]) >= top - 0.01)            # simplest within 0.01 of the best
if "ENSEMBLE" in f1 and float(f1["ENSEMBLE"]) > float(f1[BEST]) + 0.005:
    BEST = "ENSEMBLE"
BEST_MODELS = [best_pt[e] for e in ENS] if BEST == "ENSEMBLE" else best_pt[BEST]
print(f"BEST = {BEST}" + (f" {ENS}" if BEST == "ENSEMBLE" else ""), f"| val object F1 {float(f1[BEST]):.3f}",
      "|", {e: round(float(f1[e]), 3) for e in f1.index})
comp_png = plot_comparison(comparison, REPORT / "comparison_val.png")
display(Image(filename=str(comp_png), width=1200))
for c in caches.values():
    c["items"].clear()
vdir = WORK / "evaluation" / BEST / "val_reference"
display(comparison, pd.read_csv(vdir / "by_size.csv").set_index("size"))
display(Image(filename=str(vdir / "by_size.png"), width=1200)); display(Image(filename=str(vdir / "confusion.png"), width=1000))
''')

md("## 7. Test the winner (held-out blocks, reported once)")
code(r'''
from src.utils.visualization import plot_object_outcomes
best_dir = WORK / "best"
pp = tuned[BEST]
test_ref = evaluate(ref_cfg, None, "test", best_dir / "test_reference", postprocess=pp,
                    cache=predict_patches(ref_cfg, BEST_MODELS, "test", tta=USE_TTA), areas=AREAS,
                    title=f"{BEST} (traced farms)")
own_cfg = make_cfg(first)
test_own = evaluate(own_cfg, None, "test", best_dir / "test_own_labels", postprocess=pp,
                    cache=predict_patches(own_cfg, BEST_MODELS, "test", tta=USE_TTA), areas=AREAS,
                    title=f"{BEST} (all labels)")
outcomes_png = plot_object_outcomes(test_ref, REPORT / "test_outcomes.png", f"{BEST}: test, hand-traced farms")
for m in (BEST_MODELS if isinstance(BEST_MODELS, list) else [BEST_MODELS]):
    shutil.copy(m, best_dir / f"{Path(m).parent.parent.name if Path(m).parent.name == 'models' else 'R2'}_best_model.pth")
keys = ["object_f1", "object_precision", "object_recall", "mean_matched_iou", "merged_predictions", "split_farms",
        "interior_iou", "interior_precision", "interior_recall", "boundary_iou", "boundary_f1", "farms_true"]
test_table = pd.DataFrame({"test (hand-traced farms)": {k: test_ref[k] for k in keys},
                           "test (all labels)": {k: test_own[k] for k in keys}})
display(test_table, pd.read_csv(best_dir / "test_reference" / "by_size.csv").set_index("size"))
for f in (best_dir / "test_reference" / "by_size.png", best_dir / "test_reference" / "confusion.png", outcomes_png):
    display(Image(filename=str(f), width=1200))
for f in sorted((best_dir / "test_reference" / "visuals").glob("*.png"))[:4]:
    display(Image(filename=str(f), width=1500))
''')

md(r"""
## 8. Map

Every scene is predicted and turned into polygons whose outlines are simplified together (no crossing). Scenes
are merged without overlaps, and the result is combined with the digitised farms: **magenta** = digitised (kept
as they are), **green** = new model farm (likely), **orange** = new model farm (check first).
""")
code(r'''
import numpy as np, rasterio
from shapely.geometry import box
from shapely.ops import unary_union
from src.inference.predictor import Predictor
from src.data.ingestion import load_farm_polygons
from src.gis.polygon_reconstruction import polygonize, export_polygons, merge_scenes
from src.gis.zone import zone_product, write_zone_kmz, run_zone, read_zone

labels_all, _ = load_farm_polygons(own_cfg)
if MAX_FARM_HA is None:
    MAX_FARM_HA = round(max(5.0, float(labels_all.to_crs(labels_all.estimate_utm_crs()).area.quantile(0.99)) / 1e4), 1)
cfg_map = make_cfg(first)
cfg_map.paths.outputs_dir = best_dir / "map"
cfg_map.postprocess = pp
cfg_map.inference.tta = USE_TTA
cfg_map.inference.batch_size = 4
scenes = [t for t in tifs if MAP_SCENES == "all" or any(w in t.name for w in MAP_SCENES)]
if QUICK_TEST:
    scenes = scenes[:1]
predictor = Predictor(BEST_MODELS, cfg_map)
frames, foot, map_counts = [], [], {}
for img in scenes:
    prob = predictor.predict_raster(img)
    g = polygonize(cfg_map, prob)
    frames.append(g); map_counts[prob.stem.replace("_prob", "")] = len(g)
    with rasterio.open(img) as s:
        foot.append(gpd.GeoSeries([box(*s.bounds)], crs=s.crs).to_crs(4326).iloc[0])
allf = merge_scenes(frames, min_area_m2=float(pp["min_area_m2"]))
export_polygons(allf, cfg_map.paths.outputs_dir, "model_farms_all")
area = gpd.GeoDataFrame({"name": ["mapped area"]}, geometry=[unary_union(foot)], crs=4326)
none = gpd.GeoDataFrame({"tile": [], "name": []}, geometry=[], crs=4326)
farms = zone_product(allf, labels_all, area, none, min_area_m2=float(pp["min_area_m2"]),
                     max_area_ha=MAX_FARM_HA, min_compact=MIN_COMPACT,
                     regularize_cfg=dict(pp.get("regularize") or {}))       # straight edges, 1 m gaps
farms["tile"] = "Laikipia"
paths = export_polygons(farms, cfg_map.paths.outputs_dir, "farms_digitised_plus_new")
paths["kmz"] = write_zone_kmz(farms, none, cfg_map.paths.outputs_dir / "kmz" / "farms_digitised_plus_new.kmz",
                              "Laikipia farms: digitised + new (model)")
new = farms[farms.source == "Model"]
map_counts.update({"model polygons (merged, no overlaps)": len(allf), "digitised farms kept": int((farms.source == "Digitized").sum()),
                   "NEW model farms (likely)": int((new.review == "likely farm").sum()),
                   "NEW model farms (check)": int((new.review == "check").sum())})
ov = farms.sindex.query(farms.geometry, predicate="overlaps")
is_model = (farms.source == "Model").values
pairs = [(i, j) for i, j in zip(*ov) if i < j and farms.geometry.iloc[i].intersection(farms.geometry.iloc[j]).area > 1.0]
n_cross = sum(1 for i, j in pairs if is_model[i] or is_model[j])      # must be 0
map_counts["overlapping pairs involving a model polygon"] = n_cross
map_counts["overlapping pairs among your digitised farms (as drawn)"] = len(pairs) - n_cross
print(pd.Series(map_counts).to_string()); print({k: str(v) for k, v in paths.items()})

zone_counts = {}
for zf in sorted((PROJECT / ZONES_DIR).glob("*.km[lz]")):
    zname = zf.stem.split("_")[0]
    zgeom = read_zone(zf)[0].geometry.iloc[0]
    over = [im for im, fp in zip(scenes, foot) if fp.intersects(zgeom)]     # scenes whose frame touches the zone
    if not over:
        print(f"{zname}: no attached scene covers it - skipped"); continue
    try:
        zres = run_zone(cfg_map, BEST_MODELS, zf, over, best_dir / "zone", zname, digitised=labels_all,
                        max_area_ha=MAX_FARM_HA, min_compact=MIN_COMPACT, only_tiles=["T02"] if QUICK_TEST else None)
        zone_counts[zname] = zres["counts"]; print(zname); display(zres["counts"])
    except ValueError as e:
        print(f"{zname}: zone map skipped:", e)
''')

code(r'''
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from src.utils.visualization import stretch
preview_png = REPORT / "map_preview.png"
try:
    with rasterio.open(scenes[0]) as s:
        f0 = farms.to_crs(s.crs)
        inside = f0[f0.representative_point().within(box(*s.bounds))]
        c = inside[inside.source == "Model"].geometry.centroid if (inside.source == "Model").any() else inside.geometry.centroid
        cell = (np.floor(c.x / 2400) * 2400).astype(int).astype(str) + "_" + (np.floor(c.y / 2400) * 2400).astype(int).astype(str)
        x0, y0 = map(float, cell.value_counts().index[0].split("_")); b = (x0, y0, x0 + 2400, y0 + 2400)
        rgb = s.read([6, 4, 2] if s.count == 8 else [3, 2, 1], window=from_bounds(*b, s.transform), boundless=True, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(stretch(rgb), extent=(b[0], b[2], b[1], b[3]))
    f = f0.cx[b[0]:b[2], b[1]:b[3]]
    for src_, rev, col in (("Digitized", "existing", "magenta"), ("Model", "likely farm", "lime"), ("Model", "check", "orange")):
        g = f[(f.source == src_) & (f.review == rev)]
        if len(g):
            g.boundary.plot(ax=ax, color=col, linewidth=1.0)
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3]); ax.axis("off")
    ax.set_title(f"{BEST}: 2.4 x 2.4 km - magenta digitised, green new (likely), orange new (check)")
    fig.savefig(preview_png, dpi=110, bbox_inches="tight"); plt.show()
except Exception as e:  # the preview is only an illustration: never stop the run for it
    print("no preview:", e)
''')

md("## 9. Report and download")
code(r'''
from src.utils.report import build_report
tt = best_dir / "test_reference"
summary = {"best": BEST, "ensemble_members": ENS if BEST == "ENSEMBLE" else [], "tta": USE_TTA, "postprocess": dict(pp),
           "val_reference": rows[BEST], "val_all_models": {e: rows[e]["object_f1"] for e in rows},
           "test_reference": test_ref, "test_own_labels": test_own, "map": map_counts}
(best_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
r2 = f"{float(f1['R2_resunet_as_is']):.3f}" if "R2_resunet_as_is" in f1 else "0.32 (round 2 report)"
report = build_report(best_dir / "report.html", "Farm boundary detection - round 3 results", [
    ("Summary", [f"Winner: <b>{BEST}</b>" + (f" (ensemble of {', '.join(ENS)})" if BEST == "ENSEMBLE" else "")
                 + f". Validation object F1 <b>{float(f1[BEST]):.3f}</b> (round-2 model on the same farms: {r2}; "
                 f"published model not fine-tuned: {float(f1['P0_ftp_as_released']):.3f}). Test object F1 on "
                 f"hand-traced farms: <b>{test_ref['object_f1']:.3f}</b> (precision {test_ref['object_precision']:.3f}, "
                 f"recall {test_ref['object_recall']:.3f}). Overlapping polygon pairs involving a model polygon: {n_cross}.",
                 test_table]),
    ("Data", [patch_summary, checks] + [Path(p) for p in mask_figs[:2]]),
    ("Training curves", [train_summary, curves_png]),
    ("Validation comparison (hand-traced farms)", [comparison, comp_png]),
    ("Winner on validation: by farm size, confusion matrices", [pd.read_csv(vdir / "by_size.csv").set_index("size"),
                                                                vdir / "by_size.png", vdir / "confusion.png"]),
    ("Winner on test (held out)", [pd.read_csv(tt / "by_size.csv").set_index("size"), tt / "by_size.png",
                                   tt / "confusion.png", outcomes_png] + sorted((tt / "visuals").glob("*.png"))[:3]),
    ("Map", [pd.Series(map_counts, name="count").to_frame()] + list(zone_counts.values())
            + ([preview_png] if preview_png.exists() else [])),
])
backup()
with zipfile.ZipFile(WORK / "best_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in best_dir.rglob("*"):
        if f.is_file() and "prediction_masks" not in f.parts:
            z.write(f, f.relative_to(WORK))
shutil.rmtree(SCRATCH, ignore_errors=True)       # links and patches are not results
print("report ->", report, "| best_results.zip |", f"{hours_left():.1f} h of budget left")
''')

md(r"""
**Download** from the version's *Output* tab:

* `best/map/kmz/farms_digitised_plus_new.kmz` - Google Earth Pro: digitised farms (magenta) + new model farms
  (green likely, orange check); nothing overlaps. Also `geojson/`, `gpkg/`, `shapefiles/` for QGIS;
* `best/zone/kmz/salama_farms.kmz`, `mutara_farms.kmz` - each digitising zone per work tile (if scenes over it
  were attached);
* `best/report.html` - comparison with round 2, training curves, confusion matrices, scores by farm size;
* `best_results.zip` (all of the above + the model), `results_backup.zip` (every trained model).
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    src = c["source"].split("\n")
    c["source"] = [l + "\n" for l in src[:-1]] + [src[-1]]
with open(sys.argv[1], "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("cells:", len(cells))
