"""Training and retraining (fine-tuning) of the U-Net."""
from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np

from .data import CropDataset, ensure_norm, load_or_create_split, n_channels, split_blocks
from .labels import IGNORE
from .model import build_model, load_checkpoint, pick_device, save_checkpoint


def confusion(pred, target, n=3):
    keep = target != IGNORE
    idx = target[keep] * n + pred[keep]
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def iou_from_confusion(cm):
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1) - np.diag(cm)
    return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)


def _evaluate(model, loader, loss_fn, device):
    import torch

    model.eval()
    cm, total, n = np.zeros((3, 3), np.int64), 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += loss_fn(logits, y).item() * len(x)
            n += len(x)
            cm += confusion(logits.argmax(1).cpu().numpy(), y.cpu().numpy())
    iou = iou_from_confusion(cm)
    return {"val_loss": total / max(n, 1), "iou_bg": iou[0], "iou_interior": iou[1], "iou_boundary": iou[2],
            "score": float(np.nanmean(iou[1:]))}


def _check_scenes_compatible(scenes):
    import rasterio

    info = []
    for s in scenes:
        with rasterio.open(s["imagery"]) as r:
            info.append((s["name"], r.count, abs(r.transform.a)))
    n0, c0, r0 = info[0]
    for n, c, r in info[1:]:
        if c != c0 or abs(r - r0) / r0 > 0.01:
            raise ValueError(
                f"Scene '{n}' ({c} bands, {r:g} m pixels) does not match '{n0}' ({c0} bands, {r0:g} m). "
                "One model needs one kind of imagery. Keep each imagery type in its own project folder "
                "(copy config.yaml and use --config), or move the odd scene out of data/imagery/.")


def train(cfg, scenes=None, init_checkpoint=None, epochs=None, out_checkpoint=None, progress=print):
    """Train from scratch, or fine-tune from `init_checkpoint` (retraining on new/corrected labels).

    `scenes` = list of {"name", "imagery", "labels"}; default: every scene in data/imagery that has labels.

    Returns {"checkpoint", "history", "best"}. The best epoch (by mean IoU of interior+boundary on
    validation blocks) is saved to cfg.model.checkpoint (or `out_checkpoint`).
    """
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader

    from .pipeline import list_scenes

    tc = cfg.train
    scenes = scenes or list_scenes(cfg)
    if not scenes:
        raise RuntimeError("No scenes with labels found. Run the `fetch` and `labels` steps first.")
    _check_scenes_compatible(scenes)
    out_ckpt = Path(out_checkpoint or cfg.path(cfg.model.checkpoint))
    epochs = epochs or tc.epochs
    random.seed(tc.seed), np.random.seed(tc.seed), torch.manual_seed(tc.seed)
    device = pick_device(tc.device)

    if init_checkpoint:
        model, norm, _ = load_checkpoint(init_checkpoint, device)
        lr = tc.finetune_lr
        progress(f"Fine-tuning from {init_checkpoint} (lr={lr})")
    else:
        norm = ensure_norm(cfg, scenes[0]["imagery"])
        model = build_model(cfg.model.encoder, n_channels(norm), cfg.model.encoder_weights).to(device)
        lr = tc.lr
    model.train()

    split = load_or_create_split(cfg, scenes, progress)
    ok = lambda b: b["w"] >= tc.crop_px and b["h"] >= tc.crop_px  # noqa: E731
    train_b = [b for b in split_blocks(split, "train") if ok(b)]
    val_b = [b for b in split_blocks(split, "val") if ok(b)]
    if not train_b:
        raise RuntimeError("No usable training blocks. Is the labelled area large enough "
                           "(try a larger imagery.bbox or smaller split.block_px)?")
    progress(f"Blocks: train={len(train_b)} val={len(val_b)} test={len(split_blocks(split, 'test'))}; device={device}")

    ds_tr = CropDataset(scenes, norm, train_b, tc.crop_px,
                        epochs * tc.steps_per_epoch * tc.batch_size, train=True, seed=tc.seed)
    ds_va = CropDataset(scenes, norm, val_b or train_b, tc.crop_px, tc.val_crops, train=False, seed=tc.seed)
    loader_va = DataLoader(ds_va, batch_size=tc.batch_size, num_workers=tc.num_workers)

    weights = torch.tensor(tc.class_weights, dtype=torch.float32, device=device)
    ce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE)
    loss_fn = lambda logits, y: ce(logits, y) + 0.5 * dice(logits, y)  # noqa: E731
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history, best, stale = [], {"score": -1.0}, 0
    log_path = out_ckpt.with_suffix(".history.csv")
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    per_epoch = tc.steps_per_epoch * tc.batch_size
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        sub = torch.utils.data.Subset(ds_tr, range((epoch - 1) * per_epoch, epoch * per_epoch))
        loader = DataLoader(sub, batch_size=tc.batch_size, shuffle=False, num_workers=tc.num_workers)
        run, nb = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        sched.step()
        m = _evaluate(model, loader_va, loss_fn, device)
        row = {"epoch": epoch, "train_loss": run / max(nb, 1), **m, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        progress(f"epoch {epoch:02d}/{epochs} train_loss={row['train_loss']:.3f} val_loss={m['val_loss']:.3f} "
                 f"IoU interior={m['iou_interior']:.3f} boundary={m['iou_boundary']:.3f} ({row['seconds']}s)")
        if m["score"] > best["score"]:
            best, stale = {**row}, 0
            save_checkpoint(out_ckpt, model, cfg, norm, epoch, {k: float(v) for k, v in m.items()})
        else:
            stale += 1
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        if stale >= tc.patience:
            progress(f"Early stop: no improvement for {tc.patience} epochs.")
            break
    return {"checkpoint": out_ckpt, "history": history, "best": best, "history_csv": log_path}
