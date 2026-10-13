"""High-level steps shared by the CLI and the notebook."""
from __future__ import annotations

from pathlib import Path

from .config import Cfg
from .data import ensure_norm
from .export import export_polygons
from .imagery import fetch_composite
from .labels import build_labels
from .polygonize import polygonize_probs
from .predict import predict_raster

_SKIP_SUFFIXES = ("_probs", "_labels")


def main_imagery(cfg: Cfg) -> Path:
    """The primary training composite."""
    if cfg.imagery.source == "local":
        if not cfg.imagery.get("local_path"):
            raise ValueError("imagery.source is 'local' but imagery.local_path is not set.")
        return cfg.path(cfg.imagery.local_path)
    return cfg.path(cfg.imagery.output)


def labels_path_for(cfg: Cfg, imagery) -> Path:
    return cfg.path(cfg.labels.dir) / f"{Path(imagery).stem}_labels.tif"


def imagery_files(cfg: Cfg) -> list[Path]:
    """The main composite plus every other GeoTIFF in imagery.dir (extra training/prediction scenes)."""
    files = [main_imagery(cfg)]
    folder = cfg.path(cfg.imagery.dir)
    if folder.is_dir():
        for f in sorted(folder.glob("*.tif")):
            if f.resolve() not in {x.resolve() for x in files} and not f.stem.endswith(_SKIP_SUFFIXES):
                files.append(f)
    return [f for f in files if f.exists()]


def list_scenes(cfg: Cfg) -> list[dict]:
    """Scenes that have a label raster: [{'name', 'imagery', 'labels'}]."""
    scenes = []
    for f in imagery_files(cfg):
        lab = labels_path_for(cfg, f)
        if lab.exists():
            scenes.append({"name": f.stem, "imagery": f, "labels": lab})
    return scenes


def prepare_training_data(cfg: Cfg, progress=print, refetch=False, only=None):
    """Download the main composite if missing, compute normalisation, (re)build every label raster.

    Call this again after adding scenes or dropping new polygons into data/labels/manual.
    Returns the list of scenes.
    """
    from .reference import load_reference, resolution_warning, size_report

    # Fail early (before any download) if there are no farm polygons to learn from.
    rep = size_report(load_reference(cfg), cfg.project.resolution_m)
    progress(f"Reference farms: {rep}")
    warn = resolution_warning(rep, cfg.project.resolution_m)
    if warn:
        progress("WARNING: " + warn)

    main = main_imagery(cfg)
    if cfg.imagery.source != "local" and (refetch or not main.exists() or main.with_suffix(".progress.json").exists()):
        progress("Downloading Sentinel-2 composite ...")
        fetch_composite(cfg, progress=progress)
    if not main.exists():
        raise FileNotFoundError(f"Imagery not found: {main}")
    ensure_norm(cfg, main, force=True)
    files = [Path(only)] if only else imagery_files(cfg)
    for f in files:
        progress(f"Rasterising reference polygons for scene '{f.stem}' ...")
        build_labels(cfg, f, labels_path_for(cfg, f), progress=progress)
    return list_scenes(cfg)


def predict_area(cfg: Cfg, imagery, name: str, checkpoint=None, progress=print):
    """imagery GeoTIFF -> probabilities -> farm polygons -> GeoPackage + KMZ."""
    out_dir = cfg.path(cfg.outputs.dir) / name
    probs = predict_raster(cfg, imagery, checkpoint, out_dir / f"{name}_probs.tif", progress=progress)
    progress("Extracting farm polygons ...")
    frame = polygonize_probs(cfg, probs, progress=progress)
    gpkg, kmz = export_polygons(frame, out_dir, f"{cfg.outputs.kmz_name}_{name}")
    return {"probs": probs, "gpkg": gpkg, "kmz": kmz, "frame": frame}
