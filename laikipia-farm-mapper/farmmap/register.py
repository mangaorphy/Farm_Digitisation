"""Registration check: do your parcel outlines sit on visible edges in the imagery, and at what shift?

For each group (SOURCE, and each SCHEME), image edge strength is averaged along parcel outlines shifted
east/north by up to ±max_shift_m. If the strongest edges are at a non-zero shift, the outlines are
offset from the imagery; training on them teaches the model the wrong place. Scheme plans are the usual
suspect. The suggested shifts can be pasted into labels.shifts in config.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy import ndimage as ndi

from .labels import load_labels


def _outline_points(geom, step):
    pts = []
    for ring in getattr(geom, "geoms", [geom]):
        line = ring.exterior
        d = np.arange(0, line.length, step)
        pts.extend((p.x, p.y) for p in (line.interpolate(x) for x in d))
    return np.asarray(pts)


def _edge_window(src, bounds, gsd_work):
    win = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
    factor = max(1, int(round(gsd_work / src.transform.a)))
    h, w = max(1, int(win.height) // factor), max(1, int(win.width) // factor)
    img = src.read(out_shape=(src.count, h, w), window=win, boundless=True, fill_value=0,
                   resampling=Resampling.average).astype(np.float32)
    valid = (img > 0).all(axis=0)
    gray = ndi.gaussian_filter(img[:3].mean(axis=0), 1.0)
    mag = np.hypot(ndi.sobel(gray, 0), ndi.sobel(gray, 1))
    # a fence/hedge is a thin line, whose gradient has two ridges either side of it: smoothing merges them
    # into one peak on the line, so line-type and step-type boundaries both score best when centred
    mag = ndi.gaussian_filter(mag, 1.5)
    mag[~ndi.binary_erosion(valid, iterations=2)] = np.nan
    t = src.window_transform(win) * rasterio.Affine.scale(win.width / w, win.height / h)
    return mag, t


def check(cfg, max_shift_m: float = 40.0, per_group: int = 300, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir or cfg.paths.outputs)
    gdf = load_labels(cfg, apply_shifts=False)
    gdf = gdf[gdf["QA_FLAG"].astype(str).eq("OK")] if "QA_FLAG" in gdf.columns else gdf
    groups = {}
    if "SOURCE" in gdf.columns:
        for s, part in gdf.groupby("SOURCE"):
            groups[f"SOURCE={s}"] = part
    col = cfg.labels.get("shift_by") or "SCHEME"
    if col in gdf.columns:
        for s, part in gdf.groupby(col):
            if len(part) >= 15:
                groups[f"{col}={s}"] = part

    coarse = np.arange(-max_shift_m, max_shift_m + 1e-6, 4.0)
    rng = np.random.default_rng(0)
    rows, shifts = [], {}
    with rasterio.open(cfg.paths.mosaic) as src:
        gsd_work = max(1.0, src.transform.a)
        step = gsd_work
        # cache edge rasters per parcel once; score every shift on them
        cache = {}

        def parcel_data(i, geom):
            if i not in cache:
                b = geom.bounds
                pad = max_shift_m + 5
                mag, t = _edge_window(src, (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), gsd_work)
                cache[i] = (mag, ~t, _outline_points(geom, step))
            return cache[i]

        def score(items, dxs, dys):
            gx, gy = np.meshgrid(dxs, dys)  # (len(dys), len(dxs))
            sx, sy = gx.ravel(), gy.ravel()
            surf = np.zeros(sx.size)
            cnt = np.zeros(sx.size)
            for i, geom in items:
                mag, inv, pts = parcel_data(i, geom)
                if not len(pts) or np.isnan(mag).all():
                    continue
                # all shifts at once: (S, P) sample positions
                col_, row_ = inv * (pts[None, :, 0] + sx[:, None], pts[None, :, 1] + sy[:, None])
                v = ndi.map_coordinates(mag, [row_.ravel() - 0.5, col_.ravel() - 0.5], order=1,
                                        cval=np.nan).reshape(sx.size, len(pts))
                ok = np.isfinite(v).sum(axis=1) > 0.5 * len(pts)
                surf[ok] += np.nanmean(v[ok], axis=1)
                cnt[ok] += 1
            return (surf / np.maximum(cnt, 1)).reshape(gx.shape), int(cnt.max())

        for name, part in groups.items():
            take = part.sample(min(per_group, len(part)), random_state=int(rng.integers(1e9)))
            items = [(int(i), g) for i, g in zip(take.index, take.geometry)]
            s1, n = score(items, coarse, coarse)
            if n == 0:
                rows.append({"group": name, "parcels_scored": 0})
                continue
            a, b = np.unravel_index(np.nanargmax(s1), s1.shape)
            fine_x = np.arange(coarse[b] - 4, coarse[b] + 4.01, 0.5)
            fine_y = np.arange(coarse[a] - 4, coarse[a] + 4.01, 0.5)
            s2, n = score(items, fine_x, fine_y)
            a2, b2 = np.unravel_index(np.nanargmax(s2), s2.shape)
            zero = score(items, np.array([0.0]), np.array([0.0]))[0][0, 0]
            best = s2[a2, b2]
            dx, dy = float(fine_x[b2]), float(fine_y[a2])
            gain = best / max(zero, 1e-6) - 1
            off = np.hypot(dx, dy)
            verdict = "OK" if off <= 2 * src.transform.a or gain < 0.05 else ("SHIFTED" if gain >= 0.15 else "UNCLEAR")
            rows.append({"group": name, "parcels_scored": n, "best_dx_m": dx, "best_dy_m": dy,
                         "offset_m": round(off, 1), "edge_gain_vs_unshifted": round(gain, 3), "verdict": verdict})
            if verdict == "SHIFTED" and name.startswith(f"{col}="):
                shifts[name.split("=", 1)[1]] = [dx, dy]
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "registration_check.csv", index=False)
    (out_dir / "registration_shifts.yaml").write_text(
        "# Paste under labels: in config.yaml if you agree (east, north metres).\n"
        + yaml.safe_dump({"shift_by": col, "shifts": shifts}, sort_keys=True))
    print(df.to_string(index=False))
    print(f"\nWrote {out_dir / 'registration_check.csv'} and registration_shifts.yaml")
    return df
