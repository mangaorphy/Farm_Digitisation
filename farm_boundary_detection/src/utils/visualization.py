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


def plot_confusion(conf: dict, out_path: Path, title: str = "") -> Path:
    """2x2 confusion matrices (rows = truth, columns = prediction) for each head, as counts and row shares."""
    heads = list(conf)
    fig, ax = plt.subplots(1, len(heads), figsize=(5.5 * len(heads), 4.8))
    ax = np.atleast_1d(ax)
    labels = {"interior": ["not farm", "farm"], "boundary": ["not boundary", "boundary"]}
    for a, h in zip(ax, heads):
        m = np.asarray(conf[h], dtype=float)
        share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        a.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{share[i, j]:.1%}\n({m[i, j]:,.0f} px)", ha="center", va="center",
                       color="white" if share[i, j] > 0.5 else "black", fontsize=10)
        names = labels.get(h, ["0", "1"])
        a.set_xticks([0, 1], [f"pred {n}" for n in names])
        a.set_yticks([0, 1], [f"true {n}" for n in names])
        a.set_title(f"{h} head")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_size_scores(by_size, out_path: Path, title: str = "") -> Path:
    """Recall (true farms found individually) and precision (predicted farms that are right) per size class."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(by_size))
    ax[0].bar(x, by_size["recall"].fillna(0), color="#2a7ab9")
    ax[0].set_title("recall: true farms found as one polygon (IoU >= 0.5)")
    for i, (r, n) in enumerate(zip(by_size["recall"], by_size["true_farms"])):
        ax[0].text(i, (0 if np.isnan(r) else r) + 0.02, f"n={n}", ha="center", fontsize=8)
    ax[1].bar(x, by_size["precision"].fillna(0), color="#e07b39")
    ax[1].set_title("precision: predicted farms that match one true farm")
    for i, (p_, n) in enumerate(zip(by_size["precision"], by_size["predicted_farms"])):
        ax[1].text(i, (0 if np.isnan(p_) else p_) + 0.02, f"n={n}", ha="center", fontsize=8)
    for a in ax:
        a.set_xticks(x, by_size.index, rotation=20)
        a.set_ylim(0, 1.05)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_training_curves(histories: dict, out_path: Path) -> Path:
    """Training curves of several experiments: losses, interior IoU, boundary F1, monitored score, LR."""
    import pandas as pd

    panels = [("train_loss", "val_loss", "loss (solid = train, dashed = val)"), ("interior_iou", None, "interior IoU (val)"),
              ("boundary_f1", None, "boundary F1, 1 px tolerance (val)"), ("val_score", None, "monitored score (val)"),
              ("train_boundary_loss", None, "boundary loss (train)"), ("lr", None, "learning rate")]
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (name, h) in enumerate(histories.items()):
        h = pd.read_csv(h) if not hasattr(h, "columns") else h
        c = colors[k % len(colors)]
        for a, (col, col2, t) in zip(axes.ravel(), panels):
            if col in h:
                a.plot(h.epoch, h[col], color=c, label=name)
            if col2 and col2 in h:
                a.plot(h.epoch, h[col2], color=c, linestyle="--")
            a.set_title(t)
            a.set_xlabel("epoch")
        best = h["val_score"].idxmax() if "val_score" in h else None
        if best is not None:
            axes[0, 2].scatter(h.epoch[best], h.boundary_f1[best], color=c, marker="*", s=120, zorder=5)
            axes[1, 0].scatter(h.epoch[best], h.val_score[best], color=c, marker="*", s=120, zorder=5)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("training curves (star = epoch of the saved best model)", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_comparison(table, out_path: Path, metrics=("object_f1", "object_precision", "object_recall",
                                                     "interior_iou", "boundary_f1")) -> Path:
    """Grouped bars: validation scores of every candidate (experiments, ensembles)."""
    metrics = [m for m in metrics if m in table.columns]
    x = np.arange(len(table))
    w = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(table)), 4.8))
    for k, m in enumerate(metrics):
        ax.bar(x + k * w - 0.4 + w / 2, table[m].astype(float), w, label=m)
    ax.set_xticks(x, table.index, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=len(metrics))
    ax.set_title("validation scores on the common reference (hand-traced farms)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_object_outcomes(result: dict, out_path: Path, title: str = "") -> Path:
    """What happened to the true farms and to the predicted farms."""
    t, p, m = result["farms_true"], result["farms_predicted"], result["farms_matched"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].barh(["found (IoU>=0.5)", "missed / wrong shape", "split into pieces"],
               [m, t - m, result["split_farms"]], color=["#3a9d5d", "#c94c4c", "#e0a030"])
    ax[0].set_title(f"true farms ({t})")
    ax[1].barh(["correct", "wrong / extra", "merging >= 2 farms"], [m, p - m, result["merged_predictions"]],
               color=["#3a9d5d", "#c94c4c", "#e0a030"])
    ax[1].set_title(f"predicted farms ({p})")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
