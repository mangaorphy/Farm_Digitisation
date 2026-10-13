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


def registration_check(cfg, imagery_path, group_by="SOURCE", max_shift=4, max_px=3000, progress=print):
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
        data = src.read(window=win).astype("float32") / 10000.0
        wt = src.window_transform(win)
    valid = (data > 0).any(0)
    if valid.mean() < 0.2:
        raise ValueError("Too little valid imagery in the sampled window for a registration check.")
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
                rr2 = np.clip(r + dy, 0, edge.shape[0] - 1)
                kk2 = np.clip(k + dx, 0, edge.shape[1] - 1)
                grid[dy + max_shift, dx + max_shift] = edge[rr2, kk2].mean()
        iy, ix = np.unravel_index(int(grid.argmax()), grid.shape)
        out[str(name)] = {
            "n_outline_px": int(len(r)),
            "contrast_at_zero_shift": round(float(grid[max_shift, max_shift]), 3),
            "best_shift_px_east_south": (int(ix - max_shift), int(iy - max_shift)),
            "best_shift_m": (round((ix - max_shift) * res), round((iy - max_shift) * res)),
            "contrast_at_best_shift": round(float(grid.max()), 3),
            "grid": grid.round(3).tolist(),
        }
        progress(f"  {name}: contrast {out[str(name)]['contrast_at_zero_shift']} at 0 shift, "
                 f"best shift {out[str(name)]['best_shift_m']} m")
    return out
