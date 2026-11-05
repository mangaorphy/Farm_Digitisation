"""Train the U-Net on chips from `tiles` (designed for a Kaggle GPU; also runs, slowly, on CPU).

Loss = class-weighted cross-entropy + soft Dice on interior/boundary, both skipping ignore (255) pixels.
Checkpoints: last.pt every epoch (resume after a Kaggle session ends) and best.pt by validation
mean IoU of interior and boundary.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import model as M
from .config import to_plain

IGNORE = 255


class ChipDataset(Dataset):
    def __init__(self, tiles_dir: Path, ids: list[str], augment: bool):
        self.dir, self.ids, self.augment = Path(tiles_dir), list(ids), augment
        with rasterio.open(self.dir / "images" / f"{self.ids[0]}.tif") as s:
            self.n_image = n_image_bands(s.descriptions)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        tid = self.ids[i]
        with rasterio.open(self.dir / "images" / f"{tid}.tif") as s:
            img = s.read()
        with rasterio.open(self.dir / "masks" / f"{tid}.tif") as s:
            mask = s.read(1)
        if self.augment:
            k = np.random.randint(4)
            img, mask = np.rot90(img, k, axes=(1, 2)), np.rot90(mask, k)
            if np.random.rand() < 0.5:
                img, mask = img[:, :, ::-1], mask[:, ::-1]
            # per-band brightness/contrast jitter on the image bands only (scenes differ in sun angle and haze);
            # context bands have fixed physical scales and are left alone
            n = self.n_image
            im = img[:n]
            valid = im > 0
            gain = np.random.uniform(0.85, 1.15, (n, 1, 1))
            bias = np.random.uniform(-12, 12, (n, 1, 1))
            im = np.where(valid, np.clip(im * gain + bias, 1, 255), 0).astype(np.uint8)
            img = np.concatenate([im, img[n:]])
        x = torch.from_numpy(np.ascontiguousarray(M.normalize(img)))
        y = torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64))
        return x, y


def n_image_bands(descriptions) -> int:
    """Image bands come first; context bands are named ctx:<source>:<layer>."""
    return sum(1 for d in descriptions if not (d or "").startswith("ctx:"))


def _seed_worker(_):
    """Each DataLoader worker would otherwise start numpy from the same state and repeat augmentations."""
    np.random.seed(torch.initial_seed() % 2**32)


def dice_loss(logits, target, classes=(1, 2)):
    valid = target != IGNORE
    prob = logits.softmax(1)
    loss = 0.0
    for c in classes:
        p = prob[:, c][valid]
        t = (target[valid] == c).float()
        loss = loss + 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return loss / len(classes)


def confusion(logits, target, n=3):
    pred = logits.argmax(1)
    valid = target != IGNORE
    idx = target[valid] * n + pred[valid]
    return torch.bincount(idx, minlength=n * n).reshape(n, n).cpu().numpy()


def iou_from_confusion(cm):
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    return inter / np.maximum(union, 1)


def train(cfg, tiles_dir: Path | None = None, out_dir: Path | None = None, epochs: int | None = None,
          resume: bool = True, limit: int | None = None) -> Path:
    tiles_dir = Path(tiles_dir or cfg.paths.tiles)
    out_dir = Path(out_dir or cfg.paths.models)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(tiles_dir / "index.csv")
    tr_ids = index[index.split == "train"].id.tolist()
    va_ids = index[index.split == "val"].id.tolist() or index[index.split == "test"].id.tolist()
    if limit:
        tr_ids, va_ids = tr_ids[:limit], va_ids[:max(1, limit // 4)]
    if not tr_ids or not va_ids:
        raise ValueError(f"need train and val chips, got {len(tr_ids)} / {len(va_ids)}")
    with rasterio.open(tiles_dir / "images" / f"{tr_ids[0]}.tif") as s:
        in_ch, gsd, bands = s.count, s.transform.a, list(s.descriptions)

    dev = M.device()
    tc = cfg.train
    epochs = int(epochs or tc.epochs)
    # same seed -> same initial weights and chip order, so experiments differ only in their labels
    seed = int(tc.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = M.build(cfg.model.arch, cfg.model.encoder, in_ch, cfg.model.get("encoder_weights")).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay))
    use_amp = bool(tc.amp) and dev == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    weights = torch.tensor(tc.class_weights, dtype=torch.float32, device=dev)

    nw = int(tc.num_workers)
    tr = DataLoader(ChipDataset(tiles_dir, tr_ids, True), batch_size=int(tc.batch_size), shuffle=True,
                    num_workers=nw, pin_memory=dev == "cuda", drop_last=len(tr_ids) > int(tc.batch_size),
                    generator=torch.Generator().manual_seed(seed), worker_init_fn=_seed_worker)
    va = DataLoader(ChipDataset(tiles_dir, va_ids, False), batch_size=int(tc.batch_size), num_workers=nw)
    steps_total = epochs * len(tr)
    warmup = min(500, max(1, steps_total // 20))

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps_total - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    meta = {"arch": cfg.model.arch, "encoder": cfg.model.encoder, "in_channels": in_ch, "bands": bands,
            "image_bands": n_image_bands(bands), "context_bands": [b for b in bands if (b or "").startswith("ctx:")],
            "gsd_m": gsd, "classes": ["background", "interior", "boundary"], "config": to_plain(cfg)}

    # early stopping: stop when val mIoU has not improved by min_delta for `patience` epochs (null = off)
    patience = tc.get("patience")
    patience = int(patience) if patience else None
    min_delta = float(tc.get("min_delta", 0.0))
    start, best, gain_epoch, history = 0, -1.0, -1, []
    last = out_dir / "last.pt"
    if resume and last.exists():
        ck = M.load_checkpoint(last, dev)
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        gain_epoch = ck.get("gain_epoch", max((r["epoch"] for r in history if r["val_miou"] >= best), default=-1))
        if ck.get("stopped"):
            start = epochs  # finished by early stopping: nothing left to do
        print(f"Resuming from epoch {start} (best val mIoU {best:.3f})")

    print(f"device={dev} train={len(tr_ids)} val={len(va_ids)} chips, {in_ch} bands {bands}, gsd {gsd} m")
    for epoch in range(start, epochs):
        t0 = time.time()
        net.train()
        run = 0.0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = net(x)
                loss = F.cross_entropy(logits, y, weight=weights, ignore_index=IGNORE)
            loss = loss + float(tc.dice_weight) * dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        net.eval()
        cm = np.zeros((3, 3), dtype=np.int64)
        vloss = 0.0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast("cuda", enabled=use_amp):
                    logits = net(x)
                vloss += F.cross_entropy(logits.float(), y, weight=weights, ignore_index=IGNORE).item()
                cm += confusion(logits, y)
        iou = iou_from_confusion(cm)
        miou = float(iou[1:].mean())
        row = {"epoch": epoch, "train_loss": run / len(tr), "val_loss": vloss / len(va), "iou_bg": iou[0],
               "iou_interior": iou[1], "iou_boundary": iou[2], "val_miou": miou,
               "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60}
        row = {k: float(v) if hasattr(v, "item") else v for k, v in row.items()}
        history.append(row)
        print("epoch {epoch:3d}  loss {train_loss:.3f}/{val_loss:.3f}  IoU interior {iou_interior:.3f} "
              "boundary {iou_boundary:.3f}  ({minutes:.1f} min)".format(**row))
        if miou > best + min_delta or gain_epoch < 0:
            gain_epoch = epoch  # last epoch with a real gain: the early-stopping counter restarts here
        if miou > best:
            best = miou
            M.save(out_dir / "best.pt", net, {**meta, "epoch": epoch, "val": row})
        stop = patience is not None and epoch - gain_epoch >= patience
        torch.save({"state_dict": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best": best, "gain_epoch": gain_epoch, "history": history, "meta": meta,
                    "stopped": stop}, last)
        with open(out_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        if stop:
            print(f"Early stop at epoch {epoch}: no val mIoU gain > {min_delta} for {patience} epochs")
            break
    print(f"Best val mIoU (interior+boundary) {best:.3f} -> {out_dir / 'best.pt'}")
    return out_dir / "best.pt"
