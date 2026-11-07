"""Command line: python -m src <command> [--config configs/config.yaml] [--set key=value ...]

Imagery (Planet Education & Research plan)
  aoi              training AOI from the farm polygons -> data/aoi/training_aoi.geojson (+ .kmz)
  planet-search    downloadable scenes over the AOI, greedy selection, estimated quota km2 (uses no quota)
  planet-order     order the selected scenes clipped to the AOI (dry run unless --confirm)
  planet-download  wait for an order and download it into data/imagery/raw/<order id>/

Preprocessing
  ingest           read labels + imagery, write the preprocessing report
  masks            interior / boundary / instance / valid masks per image + validation figures
  patches          geographic split, 512 px patches, metadata, normalisation statistics
  check            pre-training validation checks
  prepare          ingest + masks + patches + check

Model
  train            train the multi-task network (resumes from models/latest_checkpoint.pth; --fresh to restart)
  evaluate         metrics + figures on the test (or val) patches
  predict          probability rasters for images (default: every image in paths.imagery_dir)
  polygonize       probability raster(s) -> farm polygons (GeoJSON, Shapefile, GPKG, KMZ)
  run              predict + polygonize
  demo             the whole pipeline on a synthetic scene (installation check)
"""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from .utils.helpers import load_config, setup_logging

log = logging.getLogger("farmseg.cli")


def _images(cfg, given: list[str] | None) -> list[Path]:
    if given:
        return [Path(p) for p in given]
    from .data.ingestion import discover_rasters
    return [r.path for r in discover_rasters(cfg)[0]]


def _predict(cfg, args) -> list[Path]:
    from .inference.predictor import Predictor
    model = Path(args.model or Path(cfg.paths.models_dir) / "best_model.pth")
    if not model.exists():
        raise FileNotFoundError(f"{model} not found: train first or pass --model")
    pred = Predictor(model, cfg)
    return [pred.predict_raster(img) for img in _images(cfg, args.images)]


def _polygonize(cfg, probs: list[Path], name: str | None) -> None:
    from .gis.polygon_reconstruction import export_polygons, merge_scenes, polygonize
    frames = []
    for p in probs:
        g = polygonize(cfg, p)
        export_polygons(g, cfg.paths.outputs_dir, Path(p).stem.replace("_prob", "") + "_farms")
        frames.append(g)
    if len(probs) > 1:
        export_polygons(merge_scenes(frames), cfg.paths.outputs_dir, name or "all_farms")


def demo(args) -> None:
    from .gis.polygon_reconstruction import export_polygons, polygonize
    from .inference.predictor import Predictor
    from .pipeline import prepare
    from .training.evaluation import evaluate
    from .training.trainer import train
    from .utils.synthetic import make_scene

    work = Path(args.workdir or tempfile.mkdtemp(prefix="farmseg_demo_"))
    img, lab = make_scene(work / "input")
    over = [f"paths.imagery_dir={img}", f"paths.labels_dir={lab}", f"paths.processed_dir={work / 'processed_data'}",
            f"paths.models_dir={work / 'models'}", f"paths.outputs_dir={work / 'outputs'}",
            f"paths.logs_dir={work / 'logs'}", f"paths.aoi={work / 'aoi.geojson'}",
            "split.block_size_m=900", "patches.size=128", "patches.stride=64", "patches.eval_stride=128",
            "patches.min_farm_pixels=200", "masks.label_region=all", "model.encoder=resnet18", "model.pretrained=false",
            "model.decoder_channels=[128,64,32,16]", "train.batch_size=4", "train.num_workers=0",
            f"train.epochs={args.epochs}", "train.patience=null", "train.warmup_epochs=0", "train.lr=0.001",
            "inference.patch_size=128", "inference.overlap=32", "inference.chunk_px=512",
            "postprocess.block_px=512", "postprocess.halo_px=128", *(args.set or [])]
    cfg = load_config(args.config, over)
    setup_logging(cfg.paths.logs_dir, cfg.project.log_level, "demo")
    prepare(cfg)
    best = train(cfg, resume=False)
    evaluate(cfg, best, "test")
    prob = Predictor(best, cfg).predict_raster(next(Path(img).glob("*SR*.tif")))
    farms = polygonize(cfg, prob)
    export_polygons(farms, cfg.paths.outputs_dir, "demo_farms")
    log.info("Demo OK: %d farm polygons; everything is in %s", len(farms), work)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m src", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="default: configs/config.yaml")
    common.add_argument("--set", nargs="+", action="extend", metavar="KEY=VALUE",
                        help="override config values, e.g. --set train.batch_size=4 patches.stride=512")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    add("aoi")
    add("planet-search")
    s = add("planet-order")
    s.add_argument("--confirm", action="store_true", help="really place the order (uses quota)")
    s.add_argument("--force", action="store_true", help="allow more than planet.max_order_km2")
    s.add_argument("--name")
    s = add("planet-download")
    s.add_argument("--order-id", required=True)
    for name in ("ingest", "masks", "patches", "prepare"):
        add(name)
    s = add("check")
    s.add_argument("--no-raise", action="store_true", help="report failures without an error exit")
    s = add("train")
    s.add_argument("--fresh", action="store_true", help="ignore latest_checkpoint.pth")
    s = add("evaluate")
    s.add_argument("--model")
    s.add_argument("--split", default="test", choices=["train", "val", "test"])
    for name in ("predict", "run"):
        s = add(name)
        s.add_argument("--model")
        s.add_argument("--images", nargs="+", help="GeoTIFFs (default: all in paths.imagery_dir)")
        s.add_argument("--name", help="name of the merged output when several images are given")
    s = add("polygonize")
    s.add_argument("--prob", nargs="+", required=True, help="probability rasters from predict")
    s.add_argument("--name")
    s = add("demo")
    s.add_argument("--workdir")
    s.add_argument("--epochs", type=int, default=15)
    args = p.parse_args(argv)

    if args.cmd == "demo":
        return demo(args)
    cfg = load_config(args.config, args.set)
    setup_logging(cfg.paths.logs_dir, cfg.project.log_level, args.cmd)

    if args.cmd == "aoi":
        from .data.ingestion import load_farm_polygons
        from .data.planet import build_aoi
        labels, _ = load_farm_polygons(cfg)
        build_aoi(cfg, labels)
    elif args.cmd == "planet-search":
        from .data.planet import plan
        plan(cfg)
    elif args.cmd == "planet-order":
        from .data.planet import place_order
        place_order(cfg, args.confirm, args.force, args.name)
    elif args.cmd == "planet-download":
        from .data.planet import download_order
        download_order(cfg, args.order_id)
    elif args.cmd == "ingest":
        from .data.ingestion import run_ingestion
        run_ingestion(cfg)
    elif args.cmd in ("masks", "patches", "prepare"):
        from .pipeline import prepare
        steps = {"masks": ("masks",), "patches": ("patches",), "prepare": ("masks", "patches", "check")}[args.cmd]
        prepare(cfg, steps)
    elif args.cmd == "check":
        from .pipeline import check
        check(cfg, raise_on_failure=not args.no_raise)
    elif args.cmd == "train":
        from .training.trainer import train
        from .utils.visualization import plot_history
        best = train(cfg, resume=not args.fresh)
        hist = Path(cfg.paths.models_dir) / "training_history.csv"
        if hist.exists():
            plot_history(hist, Path(cfg.paths.models_dir) / "training_history.png")
        log.info("best model: %s", best)
    elif args.cmd == "evaluate":
        from .training.evaluation import evaluate
        evaluate(cfg, Path(args.model or Path(cfg.paths.models_dir) / "best_model.pth"), args.split)
    elif args.cmd == "predict":
        _predict(cfg, args)
    elif args.cmd == "run":
        _polygonize(cfg, _predict(cfg, args), args.name)
    elif args.cmd == "polygonize":
        _polygonize(cfg, [Path(x) for x in args.prob], args.name)


if __name__ == "__main__":
    main()
