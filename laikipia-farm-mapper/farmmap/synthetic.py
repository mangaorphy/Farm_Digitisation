"""Synthetic 'farms' scene for smoke-testing the whole pipeline offline (no download, no real data).

This only proves the code runs end to end. Its accuracy numbers say nothing about real Laikipia.
"""
from __future__ import annotations

import copy
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from shapely.geometry import shape


def make_demo_data(out_dir, size=1536, n_fields=28, n_bands=6, seed=0, crs="EPSG:32637"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    transform = from_origin(250000, 30000, 10, 10)

    seeds = rng.uniform(0, size, (n_fields, 2))
    yy, xx = np.mgrid[:size, :size]
    _, ids = cKDTree(seeds).query(np.c_[yy.ravel(), xx.ravel()])
    ids = ids.reshape(size, size)

    base = rng.uniform(0.05, 0.35, (n_bands,))
    sig = base + rng.normal(0, 0.05, (n_fields, n_bands))
    img = sig[ids].transpose(2, 0, 1).copy()
    for b in range(n_bands):  # field texture + noise
        img[b] += ndi.gaussian_filter(rng.normal(0, 1, (size, size)), 6) * 0.04
        img[b] += rng.normal(0, 0.01, (size, size))
    edge = (ndi.maximum_filter(ids, 3) != ndi.minimum_filter(ids, 3))
    img[:, edge] += 0.12  # fences / tracks between farms
    arr = np.clip(np.round(img * 10000), 1, 10000).astype(np.uint16)

    img_path = out_dir / "demo_imagery.tif"
    with rasterio.open(img_path, "w", driver="GTiff", width=size, height=size, count=n_bands, dtype="uint16",
                       crs=crs, transform=transform, nodata=0, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate") as dst:
        dst.write(arr)

    polys = [shape(g) for g, v in shapes(ids.astype(np.int32), transform=transform)]
    gpd.GeoDataFrame({"NAME_96": [f"demo farm {i}" for i in range(len(polys))]}, geometry=polys, crs=crs) \
        .to_file(out_dir / "demo_reference.gpkg", driver="GPKG")
    return img_path, out_dir / "demo_reference.gpkg"


def demo_config(cfg, out_dir, epochs=6):
    """A copy of `cfg` pointed at the synthetic data, with a tiny fast model."""
    out_dir = Path(out_dir)
    c = copy.deepcopy(cfg)
    c.reference.polygons = [str(out_dir / "demo_reference.gpkg")]
    c.reference.extra_dirs = []
    c.reference.name_field = "NAME_96"
    c.imagery.bbox = None
    c.imagery.source = "local"
    c.imagery.local_path = str(out_dir / "demo_imagery.tif")
    c.imagery.local_bands = list(cfg.imagery.bands)
    c.imagery.output = str(out_dir / "demo_imagery.tif")
    c.imagery.dir = str(out_dir)
    c.labels.dir = str(out_dir / "labels")
    c.split.file = str(out_dir / "split.json")
    c.split.block_px = 512
    c.model.encoder = "resnet18"
    c.model.encoder_weights = None
    c.model.checkpoint = str(out_dir / "unet_best.pt")
    c.train.epochs, c.train.steps_per_epoch, c.train.batch_size = epochs, 12, 8
    c.train.val_crops, c.train.crop_px, c.train.patience = 32, 128, 99
    c.predict.tile_px, c.predict.pad_px, c.predict.tta = 256, 32, False
    c.polygonize.min_area_ha = 5.0
    c.outputs.dir = str(out_dir / "outputs")
    return c
