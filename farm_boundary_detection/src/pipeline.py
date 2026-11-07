"""Preprocessing orchestration: ingestion -> masks (+ figures) -> split -> patches -> normalisation -> checks."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .data.ingestion import polygons_for_raster, run_ingestion
from .data.patch_extraction import compute_normalization, extract_patches, patch_footprints
from .data.rasterization import build_masks, mask_validation_figure
from .data.splits import GeoSplit, build_split, split_path
from .data.validation import run_checks
from .utils.helpers import write_json
from .utils.visualization import plot_split_map

log = logging.getLogger(__name__)


def make_masks(cfg, labels, rasters, report) -> list[dict]:
    out = Path(cfg.paths.processed_dir)
    stats = []
    for info in rasters:
        polys, _ = polygons_for_raster(labels, info)
        mdir = out / "masks" / info.scene_id
        s = build_masks(cfg, info, polys, mdir)
        fig = mask_validation_figure(cfg, info, polys, mdir, out / "visualizations" / f"{info.scene_id}_masks.png")
        s["figure"] = str(fig)
        stats.append(s)
        log.info("mask figure -> %s", fig)
    report["masks"] = stats
    write_json(report, out / "metadata" / "preprocessing_report.json")
    return stats


def make_patches(cfg, labels, rasters) -> tuple[pd.DataFrame, GeoSplit]:
    split = build_split(cfg, labels)
    table = extract_patches(cfg, labels, rasters, split)
    compute_normalization(cfg, table)
    fp = patch_footprints(table, split.crs)
    blocks = split.blocks_gdf(fp.total_bounds)
    blocks.to_crs("EPSG:4326").to_file(Path(cfg.paths.processed_dir) / "metadata" / "split_blocks.geojson",
                                       driver="GeoJSON")
    plot_split_map(fp, blocks, labels[labels["train"]].representative_point().to_frame("geometry").set_geometry(
        "geometry"), Path(cfg.paths.processed_dir) / "visualizations" / "split_map.png")
    return table, split


def prepare(cfg, steps=("masks", "patches", "check")) -> dict:
    """Run the preprocessing steps; masks must exist before patches, patches before checks."""
    labels, rasters, report = run_ingestion(cfg)
    if "masks" in steps:
        make_masks(cfg, labels, rasters, report)
    table, split = None, None
    if "patches" in steps:
        table, split = make_patches(cfg, labels, rasters)
    if "check" in steps:
        if table is None:
            table = pd.read_csv(Path(cfg.paths.processed_dir) / "metadata" / "patches.csv")
            split = GeoSplit.load(split_path(cfg))
        run_checks(cfg, labels, rasters, report, table, split.crs, raise_on_failure=True)
    return {"labels": labels, "rasters": rasters, "report": report, "patches": table, "split": split}


def check(cfg, raise_on_failure: bool = True) -> list[dict]:
    labels, rasters, report = run_ingestion(cfg, write_report=False)
    table = pd.read_csv(Path(cfg.paths.processed_dir) / "metadata" / "patches.csv")
    split = GeoSplit.load(split_path(cfg))
    return run_checks(cfg, labels, rasters, report, table, split.crs, raise_on_failure)
