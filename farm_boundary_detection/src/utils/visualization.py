"""Figures for manual inspection: mask validation, prediction comparisons, split map."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; works on servers and in notebooks
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def stretch(rgb: np.ndarray, nodata: float | None = 0, pct=(2, 98)) -> np.ndarray:
    """(3, H, W) any dtype -> (H, W, 3) float in [0, 1], per-band percentile stretch over data pixels."""
    rgb = rgb.astype(np.float32)
    ok = np.ones(rgb.shape[1:], bool) if nodata is None else ~np.all(rgb == nodata, axis=0)
    out = np.zeros(rgb.shape[1:] + (3,), np.float32)
    for i in range(3):
        band = rgb[i]
        vals = band[ok & np.isfinite(band)]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, pct)
        out[..., i] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    out[~ok] = 0
    return out


def _rgba(mask: np.ndarray, color, alpha: float) -> np.ndarray:
    a = np.zeros(mask.shape + (4,), np.float32)
    a[mask.astype(bool)] = (*color, alpha)
    return a


def plot_mask_validation(rgb: np.ndarray, interior: np.ndarray, boundary: np.ndarray, valid: np.ndarray,
                         nodata: float | None, out_path: Path, title: str = "") -> Path:
    """Four panels: image | interior mask | boundary mask | boundaries over the image (+ ignored area)."""
    img = stretch(rgb, nodata)
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].imshow(img)
    ax[0].set_title("PlanetScope image")
    ax[1].imshow(interior, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    ax[1].set_title(f"interior mask ({int(interior.sum())} px)")
    ax[2].imshow(boundary, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    ax[2].set_title(f"boundary mask ({int(boundary.sum())} px)")
    ax[3].imshow(img)
    ax[3].imshow(_rgba(valid == 0, (0.5, 0.5, 0.5), 0.45), interpolation="nearest")
    ax[3].imshow(_rgba(boundary, (1, 0.1, 0.1), 0.9), interpolation="nearest")
    ax[3].set_title("boundaries (red) over image; grey = ignored")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_prediction_comparison(rgb: np.ndarray, gt_int: np.ndarray, pr_int: np.ndarray, gt_bnd: np.ndarray,
                               pr_bnd: np.ndarray, valid: np.ndarray, out_path: Path, title: str = "",
                               instances: np.ndarray | None = None) -> Path:
    """GT / predicted interior, GT / predicted boundary, predictions over the image, (optional) instances.
    ``pr_*`` are probabilities in [0, 1]."""
    img = stretch(rgb, None) if rgb.dtype != np.float32 or rgb.max() > 1 else np.moveaxis(rgb, 0, -1)
    n = 7 if instances is not None else 6
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 4.4))
    ax[0].imshow(img)
    ax[0].set_title("image")
    ax[1].imshow(np.where(valid > 0, gt_int, 0.5), cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    ax[1].set_title("GT interior (grey = ignored)")
    ax[2].imshow(pr_int, cmap="Greens", vmin=0, vmax=1)
    ax[2].set_title("predicted interior")
    ax[3].imshow(np.where(valid > 0, gt_bnd, 0.5), cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    ax[3].set_title("GT boundary")
    ax[4].imshow(pr_bnd, cmap="Reds", vmin=0, vmax=1)
    ax[4].set_title("predicted boundary")
    ax[5].imshow(img)
    ax[5].imshow(_rgba(gt_bnd, (0.0, 0.9, 1.0), 0.8), interpolation="nearest")
    ax[5].imshow(_rgba(pr_bnd > 0.5, (1.0, 0.9, 0.0), 0.8), interpolation="nearest")
    ax[5].set_title("GT boundary (cyan) vs predicted (yellow)")
    if instances is not None:
        rng = np.random.default_rng(0)
        colors = rng.uniform(0.2, 1.0, (int(instances.max()) + 1, 3))
        colors[0] = 0
        ax[6].imshow(colors[instances])
        ax[6].set_title(f"predicted farms ({len(np.unique(instances)) - 1})")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_split_map(patches_gdf, blocks_gdf, labels_gdf, out_path: Path) -> Path:
    """Split blocks, patch footprints and farms, to check that train / val / test areas are disjoint."""
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(10, 10))
    crs = blocks_gdf.crs
    for s, c in colors.items():
        b = blocks_gdf[blocks_gdf["split"] == s]
        if len(b):
            b.plot(ax=ax, facecolor=c, alpha=0.12, edgecolor=c, linewidth=0.5)
        p = patches_gdf[patches_gdf["split"] == s] if len(patches_gdf) else patches_gdf
        if len(p):
            p.to_crs(crs).boundary.plot(ax=ax, color=c, linewidth=0.6, label=f"{s} patches ({len(p)})")
    if labels_gdf is not None and len(labels_gdf):
        labels_gdf.to_crs(crs).plot(ax=ax, color="k", markersize=1, linewidth=0, alpha=0.5)
    if len(patches_gdf):
        x0, y0, x1, y1 = patches_gdf.to_crs(crs).total_bounds
        pad = 0.05 * max(x1 - x0, y1 - y0, 1)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
    ax.legend(loc="upper right")
    ax.set_title("geographic split: blocks (shaded) and patches (outlines); farms in black")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_history(history_csv: Path, out_path: Path) -> Path:
    import pandas as pd

    h = pd.read_csv(history_csv)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("loss")
    for col in ("interior_iou", "interior_dice"):
        if col in h:
            ax[1].plot(h.epoch, h[col], label=col)
    ax[1].set_title("interior head (val)")
    for col in ("boundary_f1", "boundary_iou", "val_score"):
        if col in h:
            ax[2].plot(h.epoch, h[col], label=col)
    ax[2].set_title("boundary head (val)")
    for a in ax:
        a.legend()
        a.set_xlabel("epoch")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
