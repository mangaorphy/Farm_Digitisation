"""Task 4: PyTorch Dataset over the saved patches.

Each item is a dict::

    image          float32 (C, H, W)  normalised bands (any number: 4-band, 8-band PlanetScope...)
    interior_mask  float32 (1, H, W)  0/1
    boundary_mask  float32 (1, H, W)  0/1
    valid_mask     float32 (1, H, W)  1 = pixel counts in loss and metrics (0 = nodata / cloud / ignored)
    weight_map     float32 (1, H, W)  boundary-aware pixel weights for the interior loss
    distance_mask  float32 (1, H, W)  (if ``distance=True``) distance to the farm's own outline, 1 at its centre
    patch_id       str

Normalisation (identical at inference): clip each band to the training percentiles, z-score with the training
mean / std, then set nodata pixels to 0 (the mean). Spatial augmentations are applied identically to the image
and all masks (nearest-neighbour for masks, so they stay binary); radiometric ones touch only valid image pixels.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

from ..utils.helpers import read_json, stable_seed

log = logging.getLogger(__name__)


class Normalizer:
    """Band-wise clip + z-score with statistics from ``normalization.json``."""

    def __init__(self, stats: dict):
        self.stats = stats
        self.lo = np.asarray(stats["clip_low"], np.float32)[:, None, None]
        self.hi = np.asarray(stats["clip_high"], np.float32)[:, None, None]
        self.mean = np.asarray(stats["mean"], np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], np.float32)[:, None, None]
        self.nodata = stats.get("nodata", 0)

    @classmethod
    def from_file(cls, path: str | Path) -> "Normalizer":
        return cls(read_json(path))

    @property
    def n_bands(self) -> int:
        return int(self.mean.shape[0])

    def data_mask(self, img: np.ndarray) -> np.ndarray:
        """True where the pixel has data (not nodata in every band, and finite)."""
        ok = ~np.all(img == self.nodata, axis=0) if self.nodata is not None else np.ones(img.shape[1:], bool)
        return ok & np.all(np.isfinite(img), axis=0)

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(C, H, W) raw -> (normalised float32, data mask)."""
        if img.shape[0] != self.n_bands:
            raise ValueError(f"image has {img.shape[0]} bands, normalisation expects {self.n_bands}")
        ok = self.data_mask(img)
        x = np.clip(np.nan_to_num(img.astype(np.float32)), self.lo, self.hi)
        x = (x - self.mean) / self.std
        x[:, ~ok] = 0.0
        return x.astype(np.float32), ok


def load_norm_stats(cfg) -> dict:
    """Normalisation statistics of the training patches (``metadata/normalization.json``)."""
    return read_json(Path(cfg.paths.processed_dir) / "metadata" / "normalization.json")

def boundary_weight_map(boundary: np.ndarray, w0: float, sigma_px: float) -> np.ndarray:
    """w = 1 + w0 * exp(-d^2 / (2 sigma^2)), d = distance (px) to the nearest boundary pixel.

    Up-weights the interior/background pixels next to farm edges, where neighbouring farms touch and the
    model must learn to stop exactly at the boundary (cf. the U-Net separation weights of Ronneberger 2015).
    """
    if w0 <= 0 or not boundary.any():
        return np.ones(boundary.shape, np.float32)
    d = ndi.distance_transform_edt(boundary == 0)
    return (1.0 + w0 * np.exp(-(d ** 2) / (2.0 * sigma_px ** 2))).astype(np.float32)


def instance_distance(instance: np.ndarray) -> np.ndarray:
    """Per farm: Euclidean distance (px) from each pixel to the nearest pixel outside that farm, divided by the
    farm's maximum, so every farm goes from ~0 at its outline to 1 at its centre (0 outside farms).
    Touching farms are separated because a neighbour's pixels count as "outside"."""
    inst = instance.astype(np.int64)
    out = np.zeros(inst.shape, np.float32)
    if not inst.any():
        return out
    # a pixel is on an outline if any 4-neighbour has a different id
    pad = np.pad(inst, 1, mode="edge")
    edge = ((pad[1:-1, 1:-1] != pad[:-2, 1:-1]) | (pad[1:-1, 1:-1] != pad[2:, 1:-1]) |
            (pad[1:-1, 1:-1] != pad[1:-1, :-2]) | (pad[1:-1, 1:-1] != pad[1:-1, 2:]) | (inst == 0))
    d = ndi.distance_transform_edt(~edge).astype(np.float32) + 1.0
    d[inst == 0] = 0
    ids = np.unique(inst[inst > 0])
    mx = ndi.maximum(d, labels=inst, index=ids)
    scale = np.zeros(int(ids.max()) + 1, np.float32)
    scale[ids] = np.asarray(mx, np.float32)
    out = np.where(inst > 0, d / np.maximum(scale[inst], 1e-6), 0).astype(np.float32)
    return out


def separation_strip(instance: np.ndarray, window_px: int = 5) -> np.ndarray:
    """Background pixels lying between two DIFFERENT farms (both within the ``window_px`` square around the pixel):
    the paths, hedges and field edges that keep neighbouring farms apart. A model that paints them as farm merges
    the neighbours into one polygon (round-3 error analysis: 55% of these pixels were predicted as farm, and 43% of
    farms were merged with a neighbour)."""
    inst = instance.astype(np.int64)
    if inst.max() == 0:
        return np.zeros(inst.shape, bool)
    big = inst.max() + 1
    mx = ndi.maximum_filter(inst, size=window_px)
    mn = ndi.minimum_filter(np.where(inst > 0, inst, big), size=window_px)
    return (inst == 0) & (mx > 0) & (mn < big) & (mx != mn)


class FarmPatchDataset(Dataset):
    """Patches of one split. ``augment`` = training augmentations (config ``augmentation``)."""

    def __init__(self, processed_dir: str | Path, patch_ids: list[str], normalizer: Normalizer,
                 augment: dict | None = None, boundary_aware: dict | None = None, seed: int = 0,
                 distance: bool = False, separation: dict | None = None, small_farm: dict | None = None):
        self.dir = Path(processed_dir)
        self.ids = list(patch_ids)
        if not self.ids:
            raise ValueError("FarmPatchDataset needs at least one patch")
        self.norm = normalizer
        self.aug = augment
        self.ba = boundary_aware or {}
        self.seed = seed
        self.distance = distance
        self.sep = separation if separation and separation.get("enabled", False) else None
        self.small = small_farm if small_farm and small_farm.get("enabled", False) else None
        missing = [p for p in self.ids[:20] if not (self.dir / "images" / f"{p}.tif").exists()]
        if missing:
            raise FileNotFoundError(f"patch files missing in {self.dir}/images, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.ids)

    def _read(self, sub: str, pid: str) -> np.ndarray:
        with rasterio.open(self.dir / sub / f"{pid}.tif") as s:
            return s.read()

    def load_raw(self, pid: str) -> dict[str, np.ndarray]:
        raw = {"image": self._read("images", pid), "interior": self._read("interior_masks", pid)[0],
               "boundary": self._read("boundary_masks", pid)[0], "valid": self._read("valid_masks", pid)[0]}
        if self.distance or self.sep or self.small:
            raw["instance"] = self._read("instance_masks", pid)[0]
        return raw

    # ---------------------------------------------------------------- augmentation
    @staticmethod
    def _spatial(arrs: list[np.ndarray], rng: np.random.Generator, a: dict) -> list[np.ndarray]:
        """Same flips / rotations for every array (image (C,H,W), masks (H,W))."""
        def apply(fn):
            return [fn(x) for x in arrs]
        if a.get("rot90", True):
            k = int(rng.integers(4))
            arrs = apply(lambda x: np.rot90(x, k, axes=(-2, -1)))
        if rng.random() < float(a.get("hflip", 0)):
            arrs = apply(lambda x: x[..., :, ::-1])
        if rng.random() < float(a.get("vflip", 0)):
            arrs = apply(lambda x: x[..., ::-1, :])
        if rng.random() < float(a.get("rotate_prob", 0)):
            ang = float(rng.uniform(-a.get("rotate_max_deg", 30), a.get("rotate_max_deg", 30)))
            img, *masks = arrs
            img = ndi.rotate(img, ang, axes=(-1, -2), reshape=False, order=1, mode="constant", cval=0.0)
            masks = [ndi.rotate(m, ang, axes=(-1, -2), reshape=False, order=0, mode="constant", cval=0) for m in masks]
            arrs = [img, *masks]  # corners rotated in from outside have valid = 0
        return [np.ascontiguousarray(x) for x in arrs]

    @staticmethod
    def _dates(x: np.ndarray, rng: np.random.Generator, a: dict) -> np.ndarray:
        """Multi-date input (``date_bands`` bands per date): random order of the dates (``swap_dates``) and,
        sometimes, one date blanked (``date_dropout``), so the network uses every date and still works when one
        is missing (PRUE / Fields of the Planet augmentations)."""
        n = int(a.get("date_bands") or 0)
        if n <= 0 or x.shape[0] % n or x.shape[0] == n:
            return x
        dates = x.reshape(x.shape[0] // n, n, *x.shape[1:])
        if rng.random() < float(a.get("swap_dates", 0)):
            dates = dates[rng.permutation(len(dates))]
        if rng.random() < float(a.get("date_dropout", 0)):
            dates = dates.copy()
            dates[int(rng.integers(len(dates)))] = 0.0          # 0 = the training mean after z-scoring
        return np.ascontiguousarray(dates.reshape(x.shape))

    @staticmethod
    def _radiometric(x: np.ndarray, ok: np.ndarray, rng: np.random.Generator, a: dict) -> np.ndarray:
        """Brightness / contrast / noise per band, on valid pixels only (x is already normalised)."""
        c = x.shape[0]
        b = rng.uniform(-a.get("brightness", 0), a.get("brightness", 0), (c, 1, 1)).astype(np.float32)
        k = rng.uniform(1 - a.get("contrast", 0), 1 + a.get("contrast", 0), (c, 1, 1)).astype(np.float32)
        y = x * k + b
        if a.get("gain", 0):      # one factor for all bands: illumination / atmospheric-correction drift
            y = y * float(rng.uniform(1 - a["gain"], 1 + a["gain"]))
        if a.get("noise_std", 0):
            y = y + rng.normal(0, a["noise_std"], x.shape).astype(np.float32)
        return np.where(ok[None], y, 0.0).astype(np.float32)

    def __getitem__(self, i: int) -> dict:
        pid = self.ids[i]
        raw = self.load_raw(pid)
        x, ok = self.norm(raw["image"])
        interior, boundary = raw["interior"].astype(np.uint8), raw["boundary"].astype(np.uint8)
        valid = (raw["valid"].astype(bool) & ok).astype(np.uint8)
        instance = raw.get("instance", np.zeros(interior.shape, np.uint32)).astype(np.int32)
        if self.aug:
            rng = np.random.default_rng(stable_seed(self.seed, pid, torch.initial_seed(), np.random.randint(1 << 30)))
            x, interior, boundary, valid, data, instance = self._spatial(
                [x, interior, boundary, valid, ok.astype(np.uint8), instance], rng, self.aug)
            # image pixels outside the labelled region are still real context: augment every DATA pixel
            x = self._radiometric(x, data.astype(bool), rng, self.aug)
            x = self._dates(x, rng, self.aug)
            valid = (valid.astype(bool) & data.astype(bool)).astype(np.uint8)
        strip = None
        if self.sep:    # the strips between neighbouring farms: part of the boundary target, and costly to fill
            strip = separation_strip(instance, int(self.sep.get("window_px", 5))) & valid.astype(bool)
            if self.sep.get("as_boundary", True):
                boundary = (boundary.astype(bool) | strip).astype(np.uint8)
        weight = boundary_weight_map(boundary, float(self.ba.get("w0", 0)), float(self.ba.get("sigma_px", 2.0)))
        if strip is not None and float(self.sep.get("weight", 0)) > 0:
            weight = np.where(strip, weight + float(self.sep["weight"]), weight).astype(np.float32)
        if self.small:   # small farms count more: w = clip(sqrt(ref / farm pixels), 1, max_weight)
            gsd = float(self.norm.stats.get("gsd_m") or 3.0)
            ref_px = float(self.small.get("ref_ha", 0.25)) * 1e4 / (gsd * gsd)
            ids, counts = np.unique(instance[instance > 0], return_counts=True)
            if len(ids):
                w_farm = np.clip(np.sqrt(ref_px / counts), 1.0, float(self.small.get("max_weight", 4.0)))
                edge = np.unique(np.concatenate([instance[0], instance[-1], instance[:, 0], instance[:, -1]]))
                w_farm[np.isin(ids, edge)] = 1.0      # cut by the patch edge: not necessarily small
                lut = np.ones(int(instance.max()) + 1, np.float32)
                lut[ids] = w_farm
                weight = (weight * lut[np.clip(instance, 0, None)]).astype(np.float32)
        item = {
            "image": torch.from_numpy(x.astype(np.float32)),
            "interior_mask": torch.from_numpy((interior > 0).astype(np.float32))[None],
            "boundary_mask": torch.from_numpy((boundary > 0).astype(np.float32))[None],
            "valid_mask": torch.from_numpy((valid > 0).astype(np.float32))[None],
            "weight_map": torch.from_numpy(weight)[None],
            "patch_id": pid,
        }
        if self.distance:
            item["distance_mask"] = torch.from_numpy(instance_distance(instance))[None]
        return item


def load_patch_table(cfg) -> pd.DataFrame:
    path = Path(cfg.paths.processed_dir) / "metadata" / "patches.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m src prepare` first")
    return pd.read_csv(path)


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)


def make_dataset(cfg, split: str, augment: bool | None = None, table: pd.DataFrame | None = None,
                 limit: int | None = None) -> FarmPatchDataset:
    table = table if table is not None else load_patch_table(cfg)
    ids = table.loc[table["split"] == split, "patch_id"].tolist()
    if limit:
        ids = ids[: int(limit)]
    if not ids:
        raise ValueError(f"no {split} patches in {cfg.paths.processed_dir}/metadata/patches.csv")
    stats = load_norm_stats(cfg)
    norm = Normalizer(stats)
    aug = dict(cfg.augmentation) if (augment if augment is not None else split == "train") else None
    return FarmPatchDataset(cfg.paths.processed_dir, ids, norm, aug, dict(cfg.loss.get("boundary_aware") or {}),
                            int(cfg.project.seed), distance=bool(cfg.model.get("distance_head", False)),
                            separation=dict(cfg.loss.get("separation") or {}),
                            small_farm=dict(cfg.loss.get("small_farm") or {}))


def make_dataloader(cfg, split: str, shuffle: bool | None = None, augment: bool | None = None,
                    limit: int | None = None, table: pd.DataFrame | None = None) -> DataLoader:
    tc = cfg.train
    ds = make_dataset(cfg, split, augment, table, limit)
    shuffle = split == "train" if shuffle is None else shuffle
    g = torch.Generator()
    g.manual_seed(int(cfg.project.seed))
    nw = int(tc.num_workers)
    return DataLoader(ds, batch_size=int(tc.batch_size), shuffle=shuffle, num_workers=nw,
                      pin_memory=bool(tc.pin_memory) and torch.cuda.is_available(),
                      drop_last=shuffle and len(ds) > int(tc.batch_size), worker_init_fn=_seed_worker, generator=g,
                      persistent_workers=nw > 0)
