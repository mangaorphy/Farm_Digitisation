"""Sliding-window inference -> 3-band uint8 probability GeoTIFF (P*255 for bg / interior / boundary)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .data import to_model_input
from .imagery import chunk_windows
from .model import load_checkpoint, pick_device


def _tiles_in(region: Window, size: int):
    for w in chunk_windows(int(region.width), int(region.height), size):
        yield Window(region.col_off + w.col_off, region.row_off + w.row_off, w.width, w.height)


def predict_raster(cfg, imagery_path, checkpoint=None, out_path=None, regions=None, progress=print):
    """Predict class probabilities. `regions` = list of Windows to restrict to (e.g. test blocks);
    everything else stays 0. Returns the output path."""
    import torch

    pc = cfg.predict
    checkpoint = checkpoint or cfg.path(cfg.model.checkpoint)
    device = pick_device(cfg.train.device)
    model, norm, _ = load_checkpoint(checkpoint, device)
    imagery_path = Path(imagery_path)
    out_path = Path(out_path) if out_path else cfg.path(cfg.outputs.dir) / (imagery_path.stem + "_probs.tif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(imagery_path) as src:
        profile = dict(driver="GTiff", width=src.width, height=src.height, count=3, dtype="uint8",
                       crs=src.crs, transform=src.transform, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", BIGTIFF="IF_SAFER")
        full = Window(0, 0, src.width, src.height)
        regions = regions or [full]
        tiles = [t for r in regions for t in _tiles_in(r, pc.tile_px)]
        pad = pc.pad_px
        with rasterio.open(out_path, "w", **profile) as dst:
            for n, t in enumerate(tiles):
                big = Window(t.col_off - pad, t.row_off - pad, t.width + 2 * pad, t.height + 2 * pad)
                arr = src.read(window=big, boundless=True, fill_value=0)
                x = torch.from_numpy(to_model_input(arr, norm))[None]
                # pad to a multiple of 32 for the encoder
                H, W = x.shape[-2:]
                ph, pw = (-H) % 32, (-W) % 32
                x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect" if min(H, W) > 32 else "constant")
                with torch.no_grad():
                    p = torch.softmax(model(x.to(device)), 1)
                    if pc.tta:
                        for dims in ((-1,), (-2,)):
                            p = p + torch.softmax(model(x.flip(dims).to(device)), 1).flip(dims)
                        p = p / 3
                p = p[0, :, pad:pad + int(t.height), pad:pad + int(t.width)].cpu().numpy()
                valid = (arr[:, pad:pad + int(t.height), pad:pad + int(t.width)] > 0).any(0)
                out = np.round(p * 255).astype(np.uint8)
                out[:, ~valid] = np.array([255, 0, 0], np.uint8)[:, None]
                dst.write(out, window=t)
                if (n + 1) % 20 == 0 or n + 1 == len(tiles):
                    progress(f"  predicted {n + 1}/{len(tiles)} tiles")
    return out_path
