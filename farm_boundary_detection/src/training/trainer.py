"""Task 7: training loop.

AdamW, cosine schedule with linear warm-up (or reduce-on-plateau), mixed precision on CUDA, gradient
clipping, early stopping on the monitored validation metric, reproducible seeds, CSV history and log files.

    models/best_model.pth         best validation score (weights + everything needed for inference)
    models/latest_checkpoint.pth  last epoch (weights, optimiser, scheduler, scaler, history) for resuming
    models/training_history.csv   one row per epoch
"""
from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import load_norm_stats, load_patch_table, make_dataloader
from ..models.multi_task_unet import build_model, load_checkpoint, model_kwargs
from ..utils.helpers import get_device, read_json, rgb_indices, seed_everything, to_plain
from .losses import MultiTaskLoss
from .metrics import SegmentationMetrics

log = logging.getLogger(__name__)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)  # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _to(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp: bool, threshold: float, tolerance: int) -> dict:
    model.eval()
    metrics = SegmentationMetrics(threshold, tolerance)
    total, n = 0.0, 0
    for batch in loader:
        batch = _to(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(batch["image"])
        il, bl = out[0], out[1]
        loss, _ = loss_fn(il, bl, batch, out[2] if len(out) > 2 else None)
        total += float(loss) * batch["image"].shape[0]
        n += batch["image"].shape[0]
        metrics.update(il, bl, batch)
    out = metrics.compute()
    out["val_loss"] = total / max(n, 1)
    return out


class Trainer:
    def __init__(self, cfg, resume: bool = True):
        self.cfg = cfg
        tc = cfg.train
        seed_everything(int(cfg.project.seed), bool(tc.get("deterministic", False)))
        self.device = get_device()
        self.out = Path(cfg.paths.models_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        table = load_patch_table(cfg)
        self.norm_stats = load_norm_stats(cfg)
        self.train_loader = make_dataloader(cfg, "train", limit=tc.get("max_train_patches"), table=table)
        self.val_loader = make_dataloader(cfg, "val", shuffle=False, augment=False, table=table)
        in_ch = len(self.norm_stats["bands"])
        self.rgb = rgb_indices(in_ch, self.norm_stats.get("band_names"), cfg.imagery.get("rgb_bands"))
        self.net = build_model(cfg, in_ch, self.rgb).to(self.device)  # the bare network (saved / loaded)
        self.model = self.net
        if self.device.type == "cuda" and torch.cuda.device_count() > 1 and tc.get("multi_gpu", True):
            self.model = torch.nn.DataParallel(self.net)  # e.g. Kaggle "GPU T4 x2"
            log.info("using %d GPUs (DataParallel)", torch.cuda.device_count())
        self.max_hours = float(tc.get("max_hours") or 0)
        self.loss_fn = MultiTaskLoss(cfg)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay))
        self.epochs = int(tc.epochs)
        steps = self.epochs * max(len(self.train_loader), 1)
        warm = int(tc.get("warmup_epochs", 0)) * max(len(self.train_loader), 1)
        self.sched_kind = tc.get("scheduler", "cosine")
        if self.sched_kind == "cosine":
            def lr_at(step: int) -> float:
                if step < warm:
                    return (step + 1) / max(warm, 1)
                return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, steps - warm))))
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_at)
        elif self.sched_kind == "plateau":
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="max", factor=0.5, patience=5)
        else:
            raise ValueError(f"train.scheduler must be cosine or plateau, not {self.sched_kind!r}")
        self.use_amp = bool(tc.amp) and self.device.type == "cuda"
        self.scaler = _grad_scaler(self.use_amp)
        self.monitor = tc.get("monitor", "val_score")
        self.start_epoch, self.best, self.best_epoch, self.history = 0, -float("inf"), -1, []
        self.stopped = False
        last = self.out / "latest_checkpoint.pth"
        if resume and last.exists():
            self._resume(last)
        else:
            if tc.get("init_from"):
                self._init_from(Path(tc.init_from))
            recal = tc.get("bn_recalibrate", "auto")
            if recal is True or (recal == "auto" and tc.get("init_from")):
                self._recalibrate_bn()
        log.info("device %s | %d train / %d val patches | %d bands %s | encoder %s | AMP %s",
                 self.device, len(self.train_loader.dataset), len(self.val_loader.dataset), in_ch,
                 self.norm_stats.get("band_names"),
                 f"{cfg.model.encoder} ({cfg.model.get('head', 'two')} head)", self.use_amp)

    # ------------------------------------------------------------------ checkpoints
    def _meta(self) -> dict:
        return {"model_kwargs": model_kwargs(self.cfg, len(self.norm_stats["bands"]), self.rgb),
                "normalization": self.norm_stats, "config": to_plain(self.cfg),
                "bands": self.norm_stats["bands"], "gsd_m": self.norm_stats.get("gsd_m")}

    def _save_best(self, epoch: int, metrics: dict) -> None:
        torch.save({"state_dict": self.net.state_dict(), "epoch": epoch, "metrics": metrics, **self._meta()},
                   self.out / "best_model.pth")

    def _save_latest(self, epoch: int) -> None:
        torch.save({"state_dict": self.net.state_dict(), "optimizer": self.opt.state_dict(),
                    "scheduler": self.sched.state_dict(), "scaler": self.scaler.state_dict(), "epoch": epoch,
                    "best": self.best, "best_epoch": self.best_epoch, "history": self.history,
                    "stopped": self.stopped, **self._meta()}, self.out / "latest_checkpoint.pth")

    def _resume(self, path: Path) -> None:
        ck = load_checkpoint(path, self.device)
        if ck.get("model_kwargs", {}).get("in_channels") != len(self.norm_stats["bands"]):
            log.warning("latest_checkpoint.pth has a different number of bands: starting fresh")
            return
        self.net.load_state_dict(ck["state_dict"])
        self.opt.load_state_dict(ck["optimizer"])
        self.sched.load_state_dict(ck["scheduler"])
        if ck.get("scaler"):
            self.scaler.load_state_dict(ck["scaler"])
        self.start_epoch = int(ck["epoch"]) + 1
        self.best, self.best_epoch, self.history = ck["best"], ck["best_epoch"], ck["history"]
        self.stopped = bool(ck.get("stopped"))
        log.info("resumed from %s at epoch %d (best %s %.4f at epoch %d)", path, self.start_epoch, self.monitor,
                 self.best, self.best_epoch)

    def _init_from(self, path: Path) -> None:
        """Fine-tuning: copy every weight of an earlier checkpoint whose name and shape match (a new head or a
        different band count stays freshly initialised). Optimiser, schedule and history start fresh."""
        if not path.exists():
            raise FileNotFoundError(f"train.init_from not found: {path}")
        src = load_checkpoint(path, self.device)["state_dict"]
        own = self.net.state_dict()
        ok, tiled = {}, []
        for k, v in src.items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                ok[k] = v
            elif (v.dim() == 4 and own[k].dim() == 4 and own[k].shape[0] == v.shape[0] and
                  own[k].shape[2:] == v.shape[2:] and own[k].shape[1] % v.shape[1] == 0):
                # more input bands (e.g. 2 dates x 8 bands): every date gets the learned filters, divided by
                # the number of dates, so the layer starts out seeing the average of the dates
                n = own[k].shape[1] // v.shape[1]
                ok[k] = v.repeat(1, n, 1, 1) / n
                tiled.append(k)
        self.net.load_state_dict(ok, strict=False)
        log.info("initialised %d / %d tensors from %s (%d skipped: missing or different shape)%s",
                 len(ok), len(own), path, len(own) - len(ok),
                 f"; first-layer filters copied to every date: {tiled}" if tiled else "")

    @torch.no_grad()
    def _recalibrate_bn(self, max_batches: int = 50) -> None:
        """Re-estimate every batch-norm layer's running statistics on this data before fine-tuning (AdaBN).

        A pretrained network's statistics describe its original training images. On other scenes they can be so
        far off that the network, in evaluation mode, predicts nothing at all, and the slow running update during
        a short fine-tune never catches up. Here they are recomputed exactly (cumulative average) from up to
        ``max_batches`` training batches; the weights are not changed."""
        bns = [m for m in self.net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if not bns:
            return
        saved = [m.momentum for m in bns]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None
        self.net.train()
        n = 0
        for batch in self.train_loader:
            if n >= max_batches:
                break
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                self.net(batch["image"].to(self.device, non_blocking=True))
            n += 1
        for m, s in zip(bns, saved):
            m.momentum = s
        log.info("re-estimated %d batch-norm layers on %d training batches (AdaBN)", len(bns), n)

    def _write_history(self) -> None:
        if not self.history:
            return
        keys = list(dict.fromkeys(k for row in self.history for k in row))
        with open(self.out / "training_history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)

    # ------------------------------------------------------------------ loop
    def train_one_epoch(self) -> dict:
        self.model.train()
        clip = float(self.cfg.train.get("grad_clip") or 0)
        sums: dict[str, float] = {}
        n = 0
        for batch in self.train_loader:
            batch = _to(batch, self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(batch["image"])
            loss, parts = self.loss_fn(out[0], out[1], batch, out[2] if len(out) > 2 else None)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {parts}: check the input normalisation")
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if clip:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            if self.sched_kind == "cosine":
                self.sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        return {f"train_{k}": v / max(n, 1) for k, v in sums.items()}

    def fit(self) -> Path:
        tc = self.cfg.train
        patience = int(tc.patience) if tc.get("patience") else None
        min_delta = float(tc.get("min_delta", 0.0))
        if self.stopped:
            log.info("training already finished (early stop); delete latest_checkpoint.pth or use --fresh to retrain")
            return self.out / "best_model.pth"
        t_start = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            if self.max_hours and epoch > self.start_epoch and (time.time() - t_start) / 3600 > self.max_hours:
                log.warning("time budget of %.1f h used: pausing at epoch %d (resume continues from "
                            "latest_checkpoint.pth)", self.max_hours, epoch)
                break
            t0 = time.time()
            tr = self.train_one_epoch()
            va = validate(self.model, self.val_loader, self.loss_fn, self.device, self.use_amp,
                          float(self.cfg.evaluation.threshold), int(tc.boundary_tolerance_px))
            if self.sched_kind == "plateau":
                self.sched.step(va[self.monitor])
            score = float(va[self.monitor])
            row = {"epoch": epoch, "train_loss": tr["train_loss"], **{k: v for k, v in tr.items() if k != "train_loss"},
                   **va, "lr": self.opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60}
            row = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
            self.history.append(row)
            improved = score > self.best + min_delta
            if score > self.best:
                self.best, self.best_epoch = score, epoch
                self._save_best(epoch, va)
            log.info("epoch %3d | loss %.4f / %.4f | interior IoU %.3f | boundary F1 %.3f (IoU %.3f) | %s %.4f%s | "
                     "%.1f min", epoch, row["train_loss"], row["val_loss"], va["interior_iou"], va["boundary_f1"],
                     va["boundary_iou"], self.monitor, score, " *" if epoch == self.best_epoch else "", row["minutes"])
            if not improved and patience is not None and epoch - self._last_gain() >= patience:
                self.stopped = True
            self._save_latest(epoch)
            self._write_history()
            if self.stopped:
                log.info("early stop at epoch %d: no %s gain > %s for %d epochs", epoch, self.monitor, min_delta, patience)
                break
        log.info("best %s %.4f at epoch %d -> %s", self.monitor, self.best, self.best_epoch,
                 self.out / "best_model.pth")
        return self.out / "best_model.pth"

    def _last_gain(self) -> int:
        """Last epoch whose score beat every earlier one by more than min_delta."""
        min_delta = float(self.cfg.train.get("min_delta", 0.0))
        best, last = -float("inf"), 0
        for r in self.history:
            if r[self.monitor] > best + min_delta:
                best, last = r[self.monitor], r["epoch"]
        return last


def train(cfg, resume: bool = True) -> Path:
    return Trainer(cfg, resume=resume).fit()
