"""Task 8: run a trained model over a whole PlanetScope GeoTIFF.

The scene is processed in chunks (``inference.chunk_px``) read with a halo; inside a chunk, overlapping
``patch_size`` windows are predicted and blended with a 2-D Hann weight (window edges, where the model sees
less context, count less), so the full-scene probability maps have no seams and memory use is bounded.

The input goes through exactly the training preprocessing: the same bands (``bands`` stored in the model),
clipping and z-scoring with the training statistics, nodata (and UDM2 non-clear pixels, if a UDM2 file is
found next to the image) set to 0 and written as nodata in the output.

Several models can be given (an ensemble): their probabilities are averaged. ``inference.tta`` also averages
over flips.

Output: ``outputs/prediction_masks/<scene>_prob.tif``, uint8, band 1 = P(interior), band 2 = P(boundary),
band 3 = predicted distance to the farm outline (only if every model has a distance head),
value = 1 + round(254 * p) (0 = nodata), same CRS / transform / size as the input.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from tqdm import tqdm

from ..data.dataset import Normalizer
from ..data.ingestion import scene_key
from ..models.multi_task_unet import load_model
from ..utils.helpers import get_device

log = logging.getLogger(__name__)


def hann2d(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]
    return np.maximum(np.outer(w, w), 1e-3).astype(np.float32)


def _find_udm2(image: Path) -> Path | None:
    key = scene_key(image)
    for p in Path(image).parent.glob("*"):
        if "udm2" in p.name.lower() and p.suffix.lower() in (".tif", ".tiff") and scene_key(p) == key:
            return p
    return None


class Predictor:
    def __init__(self, model_path: str | Path | list, cfg, device: str | None = None):
        self.cfg = cfg
        self.device = get_device(device)
        paths = model_path if isinstance(model_path, (list, tuple)) else [model_path]
        loaded = [load_model(p, self.device) for p in paths]
        self.models = [m for m, _ in loaded]
        self.ck = loaded[0][1]
        for _, ck in loaded[1:]:
            if ck["bands"] != self.ck["bands"] or ck["normalization"]["mean"] != self.ck["normalization"]["mean"]:
                raise ValueError("ensemble models must share bands and normalisation (train them on the same patches)")
        self.n_out = 3 if all(m.has_distance for m in self.models) else 2
        self.norm = Normalizer(self.ck["normalization"])
        self.bands = list(self.ck["bands"])
        ic = cfg.inference
        self.tile, self.overlap = int(ic.patch_size), int(ic.overlap)
        if self.overlap >= self.tile:
            raise ValueError("inference.overlap must be smaller than inference.patch_size")
        self.batch, self.chunk = int(ic.batch_size), int(ic.chunk_px)
        self.tta = bool(ic.get("tta", False))
        self.weight = hann2d(self.tile)

    @torch.no_grad()
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """(N, C, t, t) normalised -> (N, n_out, t, t) probabilities (interior, boundary[, distance]),
        averaged over models and (with TTA) flips."""
        t = torch.from_numpy(x).to(self.device)
        flips = [(), (-1,), (-2,), (-1, -2)] if self.tta else [()]
        acc, n = None, 0
        for model in self.models:
            for f in flips:
                xi = torch.flip(t, f) if f else t
                out = model(xi)
                p = torch.cat([torch.sigmoid(o) for o in out[: self.n_out]], 1)
                p = torch.flip(p, f) if f else p
                acc = p if acc is None else acc + p
                n += 1
        return (acc / n).float().cpu().numpy()

    def predict_array(self, x: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """Normalised (C, H, W) -> (n_out, H, W) blended probabilities. Edges smaller than a tile are padded."""
        C, H, W = x.shape
        if H < self.tile or W < self.tile:
            pad = np.zeros((C, max(H, self.tile), max(W, self.tile)), np.float32)
            pad[:, :H, :W] = x
            okp = np.zeros(pad.shape[1:], bool)
            okp[:H, :W] = ok
            return self.predict_array(pad, okp)[:, :H, :W]
        step = self.tile - self.overlap
        ys = list(range(0, H - self.tile + 1, step))
        xs = list(range(0, W - self.tile + 1, step))
        if ys[-1] + self.tile < H:
            ys.append(H - self.tile)
        if xs[-1] + self.tile < W:
            xs.append(W - self.tile)
        coords = [(y, x0) for y in ys for x0 in xs if ok[y:y + self.tile, x0:x0 + self.tile].any()]
        acc = np.zeros((self.n_out, H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        for i in range(0, len(coords), self.batch):
            part = coords[i:i + self.batch]
            batch = np.stack([x[:, y:y + self.tile, x0:x0 + self.tile] for y, x0 in part])
            probs = self._forward(batch)
            for (y, x0), p in zip(part, probs):
                acc[:, y:y + self.tile, x0:x0 + self.tile] += p * self.weight
                wsum[y:y + self.tile, x0:x0 + self.tile] += self.weight
        return acc / np.maximum(wsum, 1e-6)

    def predict_raster(self, raster: str | Path, out_path: str | Path | None = None, udm2: Path | None = None,
                       bounds: tuple | None = None) -> Path:
        raster = Path(raster)
        if not raster.exists():
            raise FileNotFoundError(raster)
        out_dir = Path(self.cfg.paths.outputs_dir) / "prediction_masks"
        out_path = Path(out_path or out_dir / f"{scene_key(raster)}_prob.tif")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        udm2 = udm2 if udm2 is not None else (_find_udm2(raster) if self.cfg.imagery.get("use_udm2", True) else None)
        halo = self.overlap
        with rasterio.open(raster) as src:
            if max(self.bands) > src.count:
                raise ValueError(f"{raster.name} has {src.count} bands; the model was trained on bands {self.bands}")
            gsd_model = self.ck.get("gsd_m")
            if gsd_model and abs(abs(src.res[0]) - gsd_model) > 0.1 * gsd_model:
                log.warning("%s: pixel %.2f m but the model was trained at %.2f m", raster.name, src.res[0], gsd_model)
            if src.crs is None:
                raise ValueError(f"{raster.name} has no CRS")
            full = Window(0, 0, src.width, src.height)
            if bounds is not None:
                full = rasterio.windows.from_bounds(*bounds, src.transform).round_offsets().round_lengths() \
                    .intersection(full)
            c0, r0, W, H = int(full.col_off), int(full.row_off), int(full.width), int(full.height)
            profile = dict(driver="GTiff", width=W, height=H, count=self.n_out, dtype="uint8", crs=src.crs,
                           transform=src.window_transform(full), nodata=0, tiled=True, blockxsize=512,
                           blockysize=512, compress="deflate", BIGTIFF="IF_SAFER")
            udm = rasterio.open(udm2) if udm2 else None
            try:
                with rasterio.open(out_path, "w", **profile) as dst:
                    jobs = [(r, c) for r in range(0, H, self.chunk) for c in range(0, W, self.chunk)]
                    for r, c in tqdm(jobs, desc=f"predict {raster.name}", unit="chunk", leave=False):
                        h, w = min(self.chunk, H - r), min(self.chunk, W - c)
                        read = Window(c0 + c - halo, r0 + r - halo, w + 2 * halo, h + 2 * halo)
                        img = src.read(self.bands, window=read, boundless=True, fill_value=self.norm.nodata or 0)
                        x, ok = self.norm(img)
                        if udm is not None:
                            ok &= udm.read(1, window=read, boundless=True, fill_value=0) == 1
                            x[:, ~ok] = 0
                        core_ok = ok[halo:halo + h, halo:halo + w]
                        if not core_ok.any():
                            continue
                        prob = self.predict_array(x, ok)[:, halo:halo + h, halo:halo + w]
                        out = np.clip(np.round(prob * 254) + 1, 1, 255).astype(np.uint8)
                        out[:, ~core_ok] = 0
                        dst.write(out, window=Window(c, r, w, h))
                    dst.descriptions = ("interior_probability", "boundary_probability", "distance")[: self.n_out]
            finally:
                if udm is not None:
                    udm.close()
        log.info("probabilities -> %s", out_path)
        return out_path
