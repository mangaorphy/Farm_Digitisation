"""Command line: python -m farmmap <command> [options]   (run from the project folder)"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def _cfg(args):
    from .config import load_config

    return load_config(args.config)


def cmd_info(args):
    from .imagery import aoi_bounds, estimate

    cfg = _cfg(args)
    bounds = aoi_bounds(cfg, bbox=args.bbox)
    print(json.dumps(estimate(cfg, bounds), indent=2))


def cmd_fetch(args):
    from .imagery import fetch_composite

    cfg = _cfg(args)
    out = fetch_composite(cfg, out_path=args.out, aoi_path=args.aoi, bbox=args.bbox)
    print(f"Imagery: {out}")


def cmd_labels(args):
    from .pipeline import prepare_training_data

    scenes = prepare_training_data(_cfg(args), only=args.imagery)
    for sc in scenes:
        print(f"Scene {sc['name']}: {sc['labels']}")


def cmd_train(args):
    from .train import train

    cfg = _cfg(args)
    res = train(cfg, init_checkpoint=args.init, epochs=args.epochs, out_checkpoint=args.out)
    print(f"Best epoch {res['best'].get('epoch')}  score={res['best'].get('score', 0):.3f}  -> {res['checkpoint']}")


def cmd_evaluate(args):
    from .evaluate import evaluate

    cfg = _cfg(args)
    res = evaluate(cfg, checkpoint=args.checkpoint, which=args.split)
    print(json.dumps(res, indent=2))


def cmd_predict(args):
    from .imagery import fetch_composite
    from .pipeline import predict_area

    cfg = _cfg(args)
    if args.imagery:
        img = Path(args.imagery)
    elif args.aoi or args.bbox:
        name = args.name or (Path(args.aoi).stem if args.aoi else "area")
        img = cfg.path(cfg.imagery.dir) / f"{name}.tif"
        if not img.exists() or img.with_suffix(".progress.json").exists():
            fetch_composite(cfg, out_path=img, aoi_path=args.aoi, bbox=args.bbox)
    else:
        sys.exit("Give --aoi FILE, --bbox LON LAT LON LAT or --imagery GEOTIFF.")
    name = args.name or img.stem
    res = predict_area(cfg, img, name, checkpoint=args.checkpoint)
    print(f"{len(res['frame'])} farms -> {res['gpkg']}\n  open in Google Earth Pro: {res['kmz']}")


def cmd_demo(args):
    """Synthetic end-to-end smoke test (about 1-3 min on CPU, no internet)."""
    from .config import load_config
    from .evaluate import evaluate
    from .pipeline import predict_area, prepare_training_data
    from .synthetic import demo_config, make_demo_data
    from .train import train

    base = load_config(args.config)
    out = Path(args.out or tempfile.mkdtemp(prefix="farmmap_demo_"))
    make_demo_data(out)
    cfg = demo_config(base, out, epochs=args.epochs)
    scenes = prepare_training_data(cfg)
    res = train(cfg, scenes)
    print("best:", {k: round(float(v), 3) for k, v in res["best"].items() if k != "epoch"})
    print(json.dumps(evaluate(cfg, scenes), indent=2))
    pred = predict_area(cfg, scenes[0]["imagery"], "demo")
    print(f"Demo OK: {len(pred['frame'])} polygons -> {pred['kmz']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="farmmap", description="Laikipia farm digitisation")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="size of the imagery download for the configured area")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("fetch", help="download a cloud-free Sentinel-2 composite (resumable)")
    s.add_argument("--aoi", help="GeoJSON/KML/KMZ/SHP polygon of the area")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    s.add_argument("--out")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("labels", help="rasterise your manual polygons into training masks (all scenes)")
    s.add_argument("--imagery", help="only this scene")
    s.set_defaults(fn=cmd_labels)

    s = sub.add_parser("train", help="train, or retrain with --init to fine-tune")
    s.add_argument("--init", help="checkpoint to fine-tune from")
    s.add_argument("--epochs", type=int)
    s.add_argument("--out", help="where to save the new checkpoint (default: model.checkpoint)")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="metrics on held-out blocks")
    s.add_argument("--split", default="test", choices=["test", "val", "train"])
    s.add_argument("--checkpoint", help="model to score (default: model.checkpoint)")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("predict", help="imagery -> farm polygons (GeoPackage + KMZ)")
    s.add_argument("--aoi")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    s.add_argument("--imagery", help="use this GeoTIFF instead of downloading")
    s.add_argument("--name")
    s.add_argument("--checkpoint")
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("demo", help="offline synthetic smoke test of the whole pipeline")
    s.add_argument("--out")
    s.add_argument("--epochs", type=int, default=6)
    s.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
