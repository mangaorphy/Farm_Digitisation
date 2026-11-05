"""Run the model over a raster (the mosaic or one scene) and write class probabilities as a 3-band uint8 GeoTIFF
(0-255 = 0-100 %; bands: background, interior, boundary). Works in chunks, so any area fits in RAM.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

from . import context as CTX
from . import model as M


def _weights(n: int) -> np.ndarray:
    """2D blending window: down-weights chip edges where the model has less context."""
    w = np.hanning(n + 2)[1:-1]
    return np.maximum(np.outer(w, w), 1e-3).astype(np.float32)


def _infer_chunk(net, img: np.ndarray, tile: int, overlap: int, batch: int, dev: str, n_image: int) -> np.ndarray:
    C, H, W = img.shape
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    acc = np.zeros((3, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    wt = _weights(tile)
    coords = [(y, x) for y in ys for x in xs if (img[:n_image, y:y + tile, x:x + tile] > 0).any()]
    for i in range(0, len(coords), batch):
        part = coords[i:i + batch]
        x = np.stack([M.normalize(img[:, y:y + tile, x:x + tile]) for y, x in part])
        with torch.no_grad():
            p = net(torch.from_numpy(x).to(dev)).softmax(1).float().cpu().numpy()
        for (y, x0), pk in zip(part, p):
            acc[:, y:y + tile, x0:x0 + tile] += pk * wt
            wsum[y:y + tile, x0:x0 + tile] += wt
    return acc / np.maximum(wsum, 1e-6)


def predict(cfg, model_path: Path, raster: Path | None = None, out_path: Path | None = None,
            bounds: tuple | None = None) -> Path:
    raster = Path(raster or cfg.paths.mosaic)
    out_path = Path(out_path or Path(cfg.paths.outputs) / "probabilities.tif")
    dev = M.device()
    if dev == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))
    net, meta = M.load(model_path, dev)
    pc = cfg.predict
    tile, overlap, chunk = int(pc.tile_px), int(pc.overlap_px), int(pc.chunk_px)
    halo = overlap
    # context layers: the same sources the model was trained with (recorded in the checkpoint), from paths.context
    want_ctx = list(meta.get("context_bands", []))
    ctx_files = [Path(cfg.paths.context) / f"{s}.tif" for s in dict.fromkeys(b.split(":")[1] for b in want_ctx)]
    missing = [str(f) for f in ctx_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"model needs context layers {missing}: run `python -m farmmap context` "
                                "or point paths.context at them")
    if CTX.band_names(ctx_files) != want_ctx:
        raise ValueError(f"model was trained with context bands {want_ctx}, but {ctx_files} hold "
                         f"{CTX.band_names(ctx_files)}: rebuild them with the training settings")
    with rasterio.open(raster) as src:
        n_image = src.count
        if abs(src.transform.a - meta["gsd_m"]) > 0.05 * meta["gsd_m"]:
            print(f"WARNING: raster pixel {src.transform.a} m but model trained at {meta['gsd_m']} m")
        if n_image + len(want_ctx) != meta["in_channels"]:
            raise ValueError(f"raster has {n_image} bands (+{len(want_ctx)} context), model expects "
                             f"{meta['in_channels']} {meta['bands']}")
        full = Window(0, 0, src.width, src.height)
        if bounds is not None:
            full = from_bounds(*bounds, src.transform).round_offsets().round_lengths().intersection(full)
        c0, r0 = int(full.col_off), int(full.row_off)
        W, H = int(full.width), int(full.height)
        profile = dict(driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs=src.crs,
                       transform=src.window_transform(full), tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", nodata=0, BIGTIFF="IF_SAFER")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jobs = [(r, c) for r in range(0, H, chunk) for c in range(0, W, chunk)]
        with rasterio.open(out_path, "w", **profile) as dst:
            for r, c in tqdm(jobs, desc="predict", unit="chunk"):
                h, w = min(chunk, H - r), min(chunk, W - c)
                read = Window(c0 + c - halo, r0 + r - halo, w + 2 * halo, h + 2 * halo)
                img = src.read(window=read, boundless=True, fill_value=0)
                if not (img[:, halo:halo + h, halo:halo + w] > 0).any():
                    continue
                nodata = (img[:, halo:halo + h, halo:halo + w] == 0).all(axis=0)
                if ctx_files:
                    img = np.concatenate([img, CTX.read(ctx_files, src.crs, src.window_transform(read),
                                                        (img.shape[1], img.shape[2]))])
                prob = _infer_chunk(net, img, tile, overlap, int(pc.batch_size), dev, n_image)[
                    :, halo:halo + h, halo:halo + w]
                out = np.clip(np.round(prob * 254) + 1, 1, 255).astype(np.uint8)  # 0 stays nodata
                out[:, nodata] = 0
                dst.write(out, window=Window(c, r, w, h))
            dst.descriptions = ("background", "interior", "boundary")
    print(f"Probabilities -> {out_path}")
    return out_path
