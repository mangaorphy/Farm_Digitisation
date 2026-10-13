"""Input normalisation, geographic train/val/test split, and the PyTorch crop dataset."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from .imagery import band_positions
from .labels import IGNORE

# ---------------------------------------------------------------- normalisation


def compute_norm(cfg, imagery_path, sample_px: int = 2_000_000) -> dict:
    """Per-band mean/std of valid pixels (from a decimated read) + red/NIR positions for NDVI."""
    with rasterio.open(imagery_path) as src:
        scale = max(1, int(math.sqrt(src.width * src.height / sample_px)))
        data = src.read(out_shape=(src.count, max(1, src.height // scale), max(1, src.width // scale)),
                        resampling=Resampling.nearest)
    valid = (data > 0).any(0)
    if not valid.any():
        raise ValueError("Imagery contains no valid pixels.")
    refl = data[:, valid].astype(np.float32) / 10000.0
    red, nir = band_positions(cfg)
    if not cfg.model.add_ndvi:
        red = nir = None
    return {"mean": refl.mean(1).tolist(), "std": (refl.std(1) + 1e-6).tolist(), "red": red, "nir": nir}


def ensure_norm(cfg, imagery_path, force=False) -> dict:
    f = Path(imagery_path).with_suffix(".norm.json")
    if f.exists() and not force:
        return json.loads(f.read_text())
    norm = compute_norm(cfg, imagery_path)
    f.write_text(json.dumps(norm))
    return norm


def n_channels(norm: dict) -> int:
    return len(norm["mean"]) + (1 if norm.get("red") is not None else 0)


def to_model_input(arr: np.ndarray, norm: dict) -> np.ndarray:
    """(C,H,W) uint16 reflectance*10000 -> standardised float32 (+NDVI). Nodata becomes 0."""
    nodata = ~(arr > 0).any(0)
    x = arr.astype(np.float32) / 10000.0
    mean = np.asarray(norm["mean"], np.float32)[:, None, None]
    std = np.asarray(norm["std"], np.float32)[:, None, None]
    chans = [(x - mean) / std]
    if norm.get("red") is not None:
        red, nir = x[norm["red"]], x[norm["nir"]]
        chans.append(((nir - red) / (nir + red + 1e-6) * 3.0)[None])
    out = np.concatenate(chans, 0)
    out[:, nodata] = 0.0
    return out


# ---------------------------------------------------------------- spatial split


def block_table(labels_path, block_px: int):
    """Usable blocks: some labelled pixels (>=5%) and at least some farm interior."""
    blocks = []
    with rasterio.open(labels_path) as src:
        step = max(1, block_px // 64)
        for row in range(0, src.height, block_px):
            for col in range(0, src.width, block_px):
                win = Window(col, row, min(block_px, src.width - col), min(block_px, src.height - row))
                if win.width < block_px // 2 or win.height < block_px // 2:
                    continue
                lab = src.read(1, window=win, out_shape=(max(1, int(win.height) // step), max(1, int(win.width) // step)),
                               resampling=Resampling.nearest)
                if (lab != IGNORE).mean() >= 0.05 and (lab == 1).any():
                    blocks.append({"id": f"{col}_{row}", "col": col, "row": row,
                                   "w": int(win.width), "h": int(win.height)})
    return blocks


def load_or_create_split(cfg, scenes, progress=print) -> dict:
    """Assign every usable block of every scene to train / val / test.

    `scenes` = list of {"name", "imagery", "labels"}. Assignments are persisted in split.file and
    never change afterwards, so the test blocks stay fixed when you add scenes/labels and retrain.
    """
    sc = cfg.split
    f = cfg.path(sc.file)
    saved = json.loads(f.read_text()).get("scenes", {}) if f.exists() else {}
    out = {}
    for scene in scenes:
        with rasterio.open(scene["labels"]) as src:
            grid = [src.width, src.height, sc.block_px, [round(v, 3) for v in list(src.transform)[:6]]]
        entry = saved.get(scene["name"])
        if entry and entry["grid"] != grid:
            progress(f"Grid of scene '{scene['name']}' changed: its split is recreated.")
            entry = None
        assignment = dict(entry["assignment"]) if entry else {}
        blocks = block_table(scene["labels"], sc.block_px)
        new = [b for b in blocks if b["id"] not in assignment]
        random.Random(f"{sc.seed}-{scene['name']}").shuffle(new)
        n = len(new)
        n_test = max(1, round(n * sc.test_fraction)) if n >= 3 else 0
        n_val = max(1, round(n * sc.val_fraction)) if n >= 3 else 0
        for i, b in enumerate(new):
            assignment[b["id"]] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        out[scene["name"]] = {"grid": grid, "assignment": assignment,
                              "blocks": {b["id"]: b for b in blocks}}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"scenes": {k: {"grid": v["grid"], "assignment": v["assignment"]}
                                        for k, v in out.items()}}))
    return {"scenes": out}


def split_blocks(split: dict, name: str):
    """Blocks (dicts with a 'scene' key) belonging to 'train' | 'val' | 'test'."""
    res = []
    for scene, e in split["scenes"].items():
        for bid, s in e["assignment"].items():
            if s == name and bid in e["blocks"]:
                res.append({**e["blocks"][bid], "scene": scene})
    return res


def block_window(b) -> Window:
    return Window(b["col"], b["row"], b["w"], b["h"])


# ---------------------------------------------------------------- dataset


class CropDataset:
    """Random (train) or fixed (val) crops from the allowed blocks of one or more scenes.
    Crops never leave their block, so train/val/test imagery does not overlap."""

    def __init__(self, scenes, norm, blocks, crop_px, length, train=True, seed=0):
        self.paths = {s["name"]: (str(s["imagery"]), str(s["labels"])) for s in scenes}
        self.norm, self.blocks, self.crop, self.length, self.train, self.seed = norm, blocks, crop_px, length, train, seed
        self._open: dict = {}
        self.fixed = None
        if not train:
            rng = random.Random(seed)
            cands = [(b, r, c) for b in blocks
                     for r in range(0, b["h"] - crop_px + 1, crop_px)
                     for c in range(0, b["w"] - crop_px + 1, crop_px)]
            rng.shuffle(cands)
            self.fixed = cands[:length]
            self.length = len(self.fixed)

    def __len__(self):
        return self.length

    def _read(self, b, r, c):
        if b["scene"] not in self._open:
            img, lab = self.paths[b["scene"]]
            self._open[b["scene"]] = (rasterio.open(img), rasterio.open(lab))
        img, lab = self._open[b["scene"]]
        win = Window(b["col"] + c, b["row"] + r, self.crop, self.crop)
        return img.read(window=win), lab.read(1, window=win)

    def __getitem__(self, i):
        import torch

        if self.fixed is not None:
            img, lab = self._read(*self.fixed[i])
        else:
            rng = random.Random((self.seed + 1) * 1_000_003 + i + random.getrandbits(32))
            best = None
            for _ in range(12):  # prefer crops that are mostly labelled
                b = rng.choice(self.blocks)
                r = rng.randint(0, max(0, b["h"] - self.crop))
                c = rng.randint(0, max(0, b["w"] - self.crop))
                img, lab = self._read(b, r, c)
                frac = (lab != IGNORE).mean()
                if best is None or frac > best[0]:
                    best = (frac, img, lab)
                if frac >= 0.3:
                    break
            _, img, lab = best
            k = rng.randint(0, 3)
            img, lab = np.rot90(img, k, (1, 2)), np.rot90(lab, k)
            if rng.random() < 0.5:
                img, lab = img[:, :, ::-1], lab[:, ::-1]
        x = to_model_input(np.ascontiguousarray(img), self.norm)
        if self.train:  # mild radiometric jitter
            x[: len(self.norm["mean"])] += np.float32(random.uniform(-0.15, 0.15))
        y = np.ascontiguousarray(lab).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
