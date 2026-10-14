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
    from .imagery import estimate

    cfg = _cfg(args)
    print(json.dumps(estimate(cfg, _aoi(cfg, args), args.bbox), indent=2))


def _aoi(cfg, args):
    """--county uses the Laikipia boundary file; otherwise --aoi (or None)."""
    if getattr(args, "county", False):
        from .county import county_path

        return county_path(cfg)
    return getattr(args, "aoi", None)


def cmd_county(args):
    from .county import download_county_boundary

    cfg = _cfg(args)
    p = download_county_boundary(cfg.path(cfg.aoi.county_boundary), cfg.aoi.county_name)
    print(f"County boundary saved: {p}")


def cmd_fetch(args):
    from .imagery import fetch_composite

    cfg = _cfg(args)
    out = fetch_composite(cfg, out_path=args.out, aoi_path=_aoi(cfg, args), bbox=args.bbox)
    print(f"Imagery: {out}")


def cmd_labels(args):
    from .pipeline import prepare_training_data

    scenes = prepare_training_data(_cfg(args), only=args.imagery)
    for sc in scenes:
        print(f"Scene {sc['name']}: {sc['labels']}")


def cmd_check(args):
    """Do the polygon outlines sit on visible edges? (per SOURCE)"""
    from .diagnostics import registration_check
    from .pipeline import main_imagery

    cfg = _cfg(args)
    res = registration_check(cfg, args.imagery or main_imagery(cfg), group_by=args.group_by)
    for name, r in res.items():
        print(json.dumps({name: {k: v for k, v in r.items() if k != "grid"}}, indent=2))
    print("Interpretation: best shift should be (0, 0) m. A clearly different best shift or contrast "
          "<= ~1.0 means those outlines are not registered to the imagery (they are not traced on it).")


def cmd_diagnose(args):
    """Why is training not improving? Imagery stats + simple baseline + memorise-one-batch test."""
    from .diagnostics import diagnose

    diagnose(_cfg(args))


def cmd_da_predict(args):
    """Delineate Anything on a GeoTIFF -> GeoPackage + KMZ."""
    from .delineate import delineate
    from .export import export_polygons
    from .pipeline import main_imagery

    cfg = _cfg(args)
    img = Path(args.imagery) if args.imagery else main_imagery(cfg)
    name = args.name or f"{img.stem}_da"
    frame = delineate(cfg, img, upsample=args.upsample)
    gpkg, kmz = export_polygons(frame, cfg.path(cfg.outputs.dir) / name, f"{cfg.outputs.kmz_name}_{name}")
    print(f"{len(frame)} fields -> {gpkg}\n  open in Google Earth Pro: {kmz}")


def cmd_compare(args):
    """Same test blocks, same scorer: U-Net vs Delineate Anything (several upsampling factors)."""
    import pandas as pd

    from .compare import compare_engines

    cfg = _cfg(args)
    only = {}
    for item in args.only or []:
        key, _, vals = item.partition("=")
        only[key] = vals.split(",")
    df = compare_engines(cfg, which=args.split, unet_checkpoint=args.checkpoint,
                         da_upsamples=tuple(args.upsample), polygon_files=args.polygons or (), only=only)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(df.to_string())
    out = cfg.path(cfg.outputs.dir) / "compare.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"saved {out}")


def cmd_prepare(args):
    """Reproject / drop bands / resample any GeoTIFF so the pipeline can use it."""
    from .prepare import prepare_imagery

    cfg = _cfg(args)
    prepare_imagery(args.src, args.dst, crs=args.crs or cfg.project.crs, resolution=args.resolution,
                    bands=args.bands, resampling=args.resampling)


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
    elif args.aoi or args.bbox or args.county:
        aoi = _aoi(cfg, args)
        name = args.name or ("laikipia_county" if args.county else Path(args.aoi).stem if args.aoi else "area")
        img = cfg.path(cfg.imagery.dir) / f"{name}.tif"
        if not img.exists() or img.with_suffix(".progress.json").exists():
            fetch_composite(cfg, out_path=img, aoi_path=aoi, bbox=args.bbox)
    else:
        sys.exit("Give --county, --aoi FILE, --bbox W S E N or --imagery GEOTIFF.")
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

    s = sub.add_parser("county", help="download the Laikipia County boundary (done automatically if missing)")
    s.set_defaults(fn=cmd_county)

    s = sub.add_parser("info", help="size of the imagery download for the configured area")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    s.add_argument("--aoi")
    s.add_argument("--county", action="store_true", help="whole Laikipia County")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("fetch", help="download a cloud-free Sentinel-2 composite (resumable)")
    s.add_argument("--aoi", help="GeoJSON/KML/KMZ/SHP polygon of the area")
    s.add_argument("--county", action="store_true", help="whole Laikipia County")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    s.add_argument("--out")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("labels", help="rasterise your manual polygons into training masks (all scenes)")
    s.add_argument("--imagery", help="only this scene")
    s.set_defaults(fn=cmd_labels)

    s = sub.add_parser("check", help="registration check: do polygon outlines sit on image edges?")
    s.add_argument("--imagery")
    s.add_argument("--group-by", default="SOURCE")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("diagnose", help="sanity checks before/after a failed training run")
    s.set_defaults(fn=cmd_diagnose)

    s = sub.add_parser("da-predict", help="Delineate Anything on imagery -> polygons (needs requirements-da.txt)")
    s.add_argument("--imagery")
    s.add_argument("--name")
    s.add_argument("--upsample", type=int)
    s.set_defaults(fn=cmd_da_predict)

    s = sub.add_parser("compare", help="score U-Net vs Delineate Anything on held-out blocks")
    s.add_argument("--split", default="test", choices=["test", "val"])
    s.add_argument("--checkpoint", help="U-Net checkpoint (default: model.checkpoint if it exists)")
    s.add_argument("--upsample", type=int, nargs="+", default=[1, 2, 4], help="DA upsampling factors")
    s.add_argument("--polygons", nargs="*", help="extra polygon files to score (e.g. official DA output)")
    s.add_argument("--only", nargs="*", metavar="COL=VAL[,VAL]",
                   help="score only reference farms with these attribute values, e.g. SOURCE=Digitized")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("prepare-imagery", help="make your own GeoTIFF usable (CRS, bands, resolution)")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--crs", help="default: project.crs")
    s.add_argument("--resolution", type=float, help="output pixel size in metres, e.g. 2")
    s.add_argument("--bands", type=int, nargs="+", help="1-based bands to keep, e.g. 1 2 3 (drops alpha)")
    s.add_argument("--resampling", default="bilinear", choices=["nearest", "bilinear", "cubic", "average"])
    s.set_defaults(fn=cmd_prepare)

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
    s.add_argument("--county", action="store_true", help="whole Laikipia County")
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
