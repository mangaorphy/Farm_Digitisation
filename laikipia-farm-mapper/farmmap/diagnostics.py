"""Checks you should run before trusting a training set.

registration_check: do the polygon outlines actually sit on visible edges in the imagery?
Parcels taken from plans/schemes (rather than traced on the imagery) are often shifted by tens
of metres; at 10 m that teaches the model the wrong boundary. We measure image edge strength
along each source's outlines after shifting them by up to +/- max_shift pixels. The shift with
the highest contrast is the best-fitting registration; it should be (0, 0).
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import ndimage as ndi

from .labels import raster_resolution_m
from .reference import load_reference
from .scale import to_unit


def registration_check(cfg, imagery_path, group_by="SOURCE", max_shift_m=40.0, max_px=3000, progress=print):
    with rasterio.open(imagery_path) as src:
        res = raster_resolution_m(src)
        ref = load_reference(cfg, crs=src.crs)
        ref = ref[ref["ref_use"]]
        # window centred on the median parcel position among parcels that fall inside the raster
        c = ref.geometry.centroid
        cols, rows = ~src.transform * (c.x.values, c.y.values)
        inside = (cols >= 0) & (cols < src.width) & (rows >= 0) & (rows < src.height)
        if not inside.any():
            raise ValueError("No reference polygons fall inside this imagery.")
        cc, rr = int(np.median(cols[inside])), int(np.median(rows[inside]))
        col0 = int(np.clip(cc - max_px // 2, 0, max(0, src.width - max_px)))
        row0 = int(np.clip(rr - max_px // 2, 0, max(0, src.height - max_px)))
        win = Window(col0, row0, min(max_px, src.width - col0), min(max_px, src.height - row0))
        data = to_unit(src.read(window=win))
        wt = src.window_transform(win)
    valid = (data > 0).any(0)
    if valid.mean() < 0.2:
        raise ValueError("Too little valid imagery in the sampled window for a registration check.")
    # search a 9 x 9 grid of shifts covering +/- max_shift_m, whatever the pixel size
    step = max(1, int(round(max_shift_m / 4 / res)))
    max_shift = 4
    edge = np.mean([ndi.gaussian_gradient_magnitude(b, 1.0) for b in data], axis=0)
    edge /= max(float(edge[valid].mean()), 1e-9)  # 1.0 = average edge strength of the scene

    col = group_by if group_by in ref.columns else "ref_source"
    out = {}
    for name, grp in ref.groupby(col):
        geoms = [g.boundary.buffer(res * 0.5) for g in grp.geometry]
        mask = rasterize([(g, 1) for g in geoms], edge.shape, transform=wt, fill=0, dtype="uint8").astype(bool)
        mask &= valid
        r, k = np.nonzero(mask)
        if len(r) < 500:
            continue
        grid = np.zeros((2 * max_shift + 1, 2 * max_shift + 1))
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                rr2 = np.clip(r + dy * step, 0, edge.shape[0] - 1)
                kk2 = np.clip(k + dx * step, 0, edge.shape[1] - 1)
                grid[dy + max_shift, dx + max_shift] = edge[rr2, kk2].mean()
        iy, ix = np.unravel_index(int(grid.argmax()), grid.shape)
        out[str(name)] = {
            "n_outline_px": int(len(r)),
            "contrast_at_zero_shift": round(float(grid[max_shift, max_shift]), 3),
            "best_shift_px_east_south": (int((ix - max_shift) * step), int((iy - max_shift) * step)),
            "best_shift_m": (round((ix - max_shift) * step * res), round((iy - max_shift) * step * res)),
            "contrast_at_best_shift": round(float(grid.max()), 3),
            "grid": grid.round(3).tolist(),
        }
        progress(f"  {name}: contrast {out[str(name)]['contrast_at_zero_shift']} at 0 shift, "
                 f"best shift {out[str(name)]['best_shift_m']} m")
    return out


# ====================================================================== training sanity checks


def imagery_stats(imagery_path, sample_px: int = 2_000_000) -> dict:
    """Per-band statistics of valid pixels. Degenerate imagery shows up here immediately."""
    import math

    from rasterio.enums import Resampling

    from .data import imagery_problems

    with rasterio.open(imagery_path) as src:
        scale = max(1, int(math.sqrt(src.width * src.height / sample_px)))
        data = src.read(out_shape=(src.count, max(1, src.height // scale), max(1, src.width // scale)),
                        resampling=Resampling.nearest)
    valid = (data > 0).any(0)
    out = {"valid_fraction": round(float(valid.mean()), 3), "bands": []}
    if not valid.any():
        out["problems"] = ["no valid pixels"]
        return out
    refl = to_unit(data)[:, valid]
    for i, b in enumerate(refl):
        out["bands"].append({"band": i, "p2": round(float(np.percentile(b, 2)), 3),
                             "median": round(float(np.median(b)), 3), "p98": round(float(np.percentile(b, 98)), 3),
                             "std": round(float(b.std()), 4)})
    out["problems"] = imagery_problems(refl, float(valid.mean()))
    return out


def _pixel_features(data: np.ndarray, red: int | None, nir: int | None) -> np.ndarray:
    """(bands,H,W) reflectance -> (H*W, F): bands, NDVI, smoothed bands, local texture, edges."""
    x = data.astype("float32")
    feats = [x]
    if red is not None:
        feats.append(((x[nir] - x[red]) / (x[nir] + x[red] + 1e-6))[None])
    feats.append(np.stack([ndi.gaussian_filter(b, 2.0) for b in x]))
    mean = x.mean(0)
    feats.append(ndi.gaussian_gradient_magnitude(mean, 1.0)[None])
    feats.append(np.sqrt(np.maximum(ndi.uniform_filter(mean**2, 7) - ndi.uniform_filter(mean, 7) ** 2, 0))[None])
    f = np.concatenate(feats, 0)
    return f.reshape(f.shape[0], -1).T


def baseline_check(cfg, scenes, n_blocks=12, px_per_block=6000, seed=0, progress=print) -> dict:
    """Model-free sanity check: can a per-pixel gradient-boosting classifier tell background /
    interior / boundary apart from the imagery? Trained on train blocks, scored on val blocks."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from .data import block_window, load_or_create_split, split_blocks
    from .imagery import band_positions
    from .labels import IGNORE

    rng = np.random.default_rng(seed)
    split = load_or_create_split(cfg, scenes, lambda *_: None)
    red, nir = band_positions(cfg)
    if not cfg.model.add_ndvi:
        red = nir = None
    paths = {s["name"]: s for s in scenes}

    def sample(blocks):
        X, y = [], []
        for b in blocks:
            s = paths[b["scene"]]
            win = block_window(b)
            with rasterio.open(s["imagery"]) as si, rasterio.open(s["labels"]) as sl:
                img = to_unit(si.read(window=win))
                lab = sl.read(1, window=win).ravel()
            feats = _pixel_features(img, red, nir)
            ok = np.flatnonzero((lab != IGNORE) & (img > 0).any(0).ravel())
            if len(ok) == 0:
                continue
            take = rng.choice(ok, size=min(px_per_block, len(ok)), replace=False)
            X.append(feats[take]); y.append(lab[take])
        return (np.concatenate(X), np.concatenate(y)) if X else (None, None)

    tr = split_blocks(split, "train")
    va = split_blocks(split, "val") or split_blocks(split, "test")
    rng.shuffle(tr)
    Xtr, ytr = sample(tr[:n_blocks])
    Xva, yva = sample(va[: max(3, n_blocks // 3)])
    if Xtr is None or Xva is None:
        raise RuntimeError("Not enough labelled blocks for the baseline check.")
    counts = np.bincount(ytr, minlength=3).astype(float)
    w = (len(ytr) / (3 * np.maximum(counts, 1)))[ytr]  # balanced
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(Xtr, ytr, sample_weight=w)
    pred = clf.predict(Xva)
    cm = np.bincount(yva * 3 + pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    out = {"class_share_val": (np.bincount(yva, minlength=3) / len(yva)).round(3).tolist(),
           "iou_bg_interior_boundary": iou.round(3).tolist(),
           "recall_bg_interior_boundary": recall.round(3).tolist(),
           "balanced_accuracy": round(float(recall.mean()), 3)}
    progress(f"  baseline: IoU {out['iou_bg_interior_boundary']}  balanced accuracy {out['balanced_accuracy']}")
    return out


def overfit_test(cfg, scenes, steps=60, progress=print) -> dict:
    """Can the network memorise ONE batch? If not, something in the pipeline is broken."""
    import segmentation_models_pytorch as smp
    import torch

    from .data import CropDataset, ensure_norm, load_or_create_split, n_channels, split_blocks
    from .labels import IGNORE
    from .model import build_model, pick_device

    tc = cfg.train
    norm = ensure_norm(cfg, scenes[0]["imagery"])
    split = load_or_create_split(cfg, scenes, lambda *_: None)
    blocks = [b for b in split_blocks(split, "train") if b["w"] >= tc.crop_px and b["h"] >= tc.crop_px]
    ds = CropDataset(scenes, norm, blocks, tc.crop_px, 8, train=False, seed=1)
    xs, ys = zip(*[ds[i] for i in range(min(8, len(ds)))])
    x, y = torch.stack(xs), torch.stack(ys)
    device = pick_device(tc.device)
    torch.manual_seed(0)
    model = build_model(cfg.model.encoder, n_channels(norm), None).to(device).train()
    ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = x.to(device), y.to(device)
    losses = []
    for _ in range(steps):
        loss = ce(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    out = {"first_loss": round(losses[0], 3), "last_loss": round(losses[-1], 3), "steps": steps}
    progress(f"  overfit test: loss {out['first_loss']} -> {out['last_loss']}")
    return out


def diagnose(cfg, scenes=None, progress=print) -> dict:
    """Run all sanity checks and print a verdict."""
    from .pipeline import list_scenes

    scenes = scenes or list_scenes(cfg)
    res = {}
    progress("1/3 Imagery statistics ...")
    res["imagery"] = imagery_stats(scenes[0]["imagery"])
    progress(f"  valid {res['imagery']['valid_fraction']:.0%}; bands (median): "
             f"{[b['median'] for b in res['imagery'].get('bands', [])]}")
    for p in res["imagery"].get("problems", []):
        progress("  PROBLEM: " + p)
    progress("2/3 Model-free baseline (gradient boosting on pixel features) ...")
    res["baseline"] = baseline_check(cfg, scenes, progress=progress)
    progress("3/3 Can the network memorise one batch? ...")
    res["overfit"] = overfit_test(cfg, scenes, progress=progress)

    imap = res["baseline"]["iou_bg_interior_boundary"]
    bal = res["baseline"]["balanced_accuracy"]
    learns = res["overfit"]["last_loss"] < 0.6 * res["overfit"]["first_loss"]
    if res["imagery"].get("problems"):
        v = "IMAGERY PROBLEM: the composite is degenerate. Re-run fetch; see the problems above."
    elif not learns:
        v = ("PIPELINE BUG: the network cannot even memorise one batch. Send me this output and the "
             "first lines of the training log.")
    elif bal < 0.42 and imap[1] < 0.05:
        v = ("NO USABLE SIGNAL: even a simple per-pixel classifier cannot separate farm interior from "
             "background with this imagery and these labels. Likely causes: parcels too small for 10 m "
             "pixels, outlines not registered to the imagery (run `check`), or too little/poor imagery. "
             "A deep network will not fix this; higher-resolution imagery or a coarser target will.")
    elif imap[1] >= 0.05:
        v = ("The data DOES carry signal and the network can learn. Train longer / tune: more steps per "
             "epoch, a higher learning rate (1e-3), more epochs, and compare with this baseline's IoU.")
    else:
        v = "Weak signal. Training may reach modest scores; check registration and consider a coarser target."
    res["verdict"] = v
    progress("\nVERDICT: " + v)
    return res
