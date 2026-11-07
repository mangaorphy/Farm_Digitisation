"""Command line: python -m farmmap <command> [--config config.yaml] [--set key=value ...]

  labels       summarise your parcels at the training resolution
  aoi          fence corridor + training-parcel area -> data/aoi/ (AOI, SDX search boxes, KMZ to view)
  ingest      SDX downloads in data/imagery/raw -> aligned 8-bit scenes + mosaic.vrt
  check        registration check: are parcel outlines on visible edges? suggests per-scheme shifts
  tiles        cut training chips + masks (zip data/tiles for Kaggle)
  train        train the U-Net (Kaggle GPU)
  predict      model -> probability raster (whole mosaic, or --split test / --bbox)
  polygonize   probabilities -> farms.gpkg
  evaluate     score farms.gpkg against your parcels on held-out test blocks
  export       farms.gpkg -> farms.kmz for Google Earth Pro
  run          predict + polygonize + export in one go
  demo         full pipeline on a synthetic scene (install check)
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import geopandas as gpd
from rasterio.warp import transform_bounds

from . import config as C


def _bounds(cfg, args):
    if getattr(args, "bbox", None):
        return transform_bounds("EPSG:4326", cfg.aoi.crs, *args.bbox)
    if getattr(args, "split", None):
        from .labels import load_labels
        from .tiles import blocks_gdf
        lab = load_labels(cfg)
        b = blocks_gdf(lab.total_bounds, cfg)
        ctx = lab.geometry.buffer(float(cfg.labels.context_m)).union_all()
        sel = b[(b.split == args.split) & b.intersects(ctx)]
        return tuple(sel.total_bounds)
    return None


def demo(args) -> None:
    from . import evaluate, export, imagery, polygonize, predict, synthetic, tiles, train

    work = Path(args.workdir or tempfile.mkdtemp(prefix="farmmap_demo_"))
    raw, lab = synthetic.make_scene(work)
    cfg = C.load(args.config, [
        f"paths.imagery_raw={raw}", f"paths.labels={lab}", f"paths.scenes={work / 'scenes'}",
        f"paths.mosaic={work / 'mosaic.vrt'}", f"paths.tiles={work / 'tiles'}", f"paths.models={work / 'models'}",
        f"paths.outputs={work / 'outputs'}", f"paths.aoi={work / 'no_aoi.geojson'}", "aoi.bbox=null",
        "tiles.size_px=256", "tiles.block_m=200",
        "tiles.split={train: 0.6, val: 0.2, test: 0.2}", "train.batch_size=4", "train.num_workers=0",
        f"model.encoder={args.encoder}", "model.encoder_weights=null", "predict.tile_px=256",
        "predict.overlap_px=64", "predict.chunk_px=1024", "polygonize.block_px=1024", "polygonize.halo_px=256",
        *(args.set or [])])
    imagery.ingest(cfg)
    tiles.make_tiles(cfg)
    best = train.train(cfg, epochs=args.epochs)
    prob = predict.predict(cfg, best)
    farms = polygonize.polygonize(cfg, prob)
    evaluate.evaluate(cfg, Path(cfg.paths.outputs) / "farms.gpkg", split="all")
    export.to_kmz(farms, Path(cfg.paths.outputs) / "farms.kmz")
    print(f"Demo OK: {len(farms)} polygons in {work}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="farmmap", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml")
    common.add_argument("--set", nargs="+", action="extend", metavar="KEY=VALUE",
                        help="override config after the command, e.g. tiles --set tiles.size_px=384")
    sub = p.add_subparsers(dest="cmd", required=True)
    add = lambda name: sub.add_parser(name, parents=[common])  # noqa: E731
    add("labels")
    add("aoi")
    s = add("ingest")
    s.add_argument("--only", nargs="*", help="scene ids")
    s = add("check")
    s.add_argument("--max-shift", type=float, default=40.0)
    s.add_argument("--per-group", type=int, default=300)
    add("tiles")
    s = add("train")
    s.add_argument("--tiles")
    s.add_argument("--out")
    s.add_argument("--epochs", type=int)
    s.add_argument("--fresh", action="store_true", help="ignore last.pt and start over")
    for name in ("predict", "run"):
        s = add(name)
        s.add_argument("--model", default=None)
        s.add_argument("--raster")
        s.add_argument("--out")
        s.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
        s.add_argument("--split", choices=["train", "val", "test"])
    s = add("polygonize")
    s.add_argument("--prob", required=True)
    s.add_argument("--out")
    s = add("evaluate")
    s.add_argument("--pred", required=True)
    s.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    s = add("export")
    s.add_argument("--pred", required=True)
    s.add_argument("--out")
    s = add("demo")
    s.add_argument("--workdir")
    s.add_argument("--epochs", type=int, default=8)
    s.add_argument("--encoder", default="resnet18")
    args = p.parse_args(argv)

    if args.cmd == "demo":
        return demo(args)
    cfg = C.load(args.config, args.set)
    if args.cmd == "labels":
        from .labels import load_labels, summary
        g = load_labels(cfg)
        print(f"{len(g)} parcels in AOI, {int(g.train.sum())} used for training\n")
        print(summary(g, float(cfg.imagery.gsd_m)).to_string())
    elif args.cmd == "aoi":
        from .aoi import build
        build(cfg)
    elif args.cmd == "ingest":
        from .imagery import ingest
        ingest(cfg, args.only)
    elif args.cmd == "check":
        from .register import check
        check(cfg, args.max_shift, args.per_group)
    elif args.cmd == "tiles":
        from .tiles import make_tiles
        make_tiles(cfg)
    elif args.cmd == "train":
        from .train import train
        train(cfg, args.tiles, args.out, args.epochs, resume=not args.fresh)
    elif args.cmd in ("predict", "run"):
        from .predict import predict
        model = Path(args.model or Path(cfg.paths.models) / "best.pt")
        tag = args.split or ("bbox" if args.bbox else "all")
        prob = predict(cfg, model, args.raster, Path(cfg.paths.outputs) / f"probabilities_{tag}.tif",
                       _bounds(cfg, args))
        if args.cmd == "run":
            from .export import to_kmz
            from .polygonize import polygonize
            out = Path(args.out or Path(cfg.paths.outputs) / f"farms_{tag}.gpkg")
            farms = polygonize(cfg, prob, out)
            to_kmz(farms, out.with_suffix(".kmz"))
    elif args.cmd == "polygonize":
        from .polygonize import polygonize
        polygonize(cfg, Path(args.prob), Path(args.out) if args.out else None)
    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        evaluate(cfg, Path(args.pred), args.split)
    elif args.cmd == "export":
        from .export import to_kmz
        pred = Path(args.pred)
        to_kmz(gpd.read_file(pred), Path(args.out) if args.out else pred.with_suffix(".kmz"))


if __name__ == "__main__":
    main()
