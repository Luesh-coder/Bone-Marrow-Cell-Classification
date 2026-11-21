"""
Training engine shared by scripts/train.py, tune.py and evaluate.py.

    build_loss        cross-entropy, class-weighted cross-entropy, or focal loss
    build_optimizer   AdamW or SGD (momentum 0.9, Nesterov) over param groups
    build_scheduler   linear warmup, then cosine annealing or step decay (per step)
    train_one_epoch   bf16 autocast, channels_last, optional gradient clipping
    evaluate          softmax probabilities + labels for a loader
    fit               full run: validate every epoch, early-stop on validation
                      macro-F1, save best/last checkpoints, history and plots
    benchmark         images/s of the data loader alone and loader + training step

Run outputs go to runs/<timestamp>_<name>/:
    config.yaml, history.csv, best.pt, last.pt, val_metrics.json, summary.json,
    curves.png, val_best/ (per-class CSV + confusion matrix of the best epoch)
"""

import csv
import gc
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .data import build_data, check_worker_memory, class_weights, make_loader, make_sampler
from .metrics import compute_metrics, plot_history, save_metrics, summary_line
from .models import build_model, count_params, freeze_bn_stats, param_groups
from .preprocess import normalize


# --------------------------------------------------------------------------- losses

class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): cross-entropy scaled by (1 - p_true) ** gamma,
    so easy, confidently-correct examples count less. Optional per-class weights
    (alpha), averaged like weighted cross-entropy. gamma = 0 is plain (weighted) CE."""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", None if weight is None else torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum()
        return loss.mean()


def build_loss(cfg, train_counts):
    if cfg.loss == "ce":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    w = class_weights(train_counts, cfg.class_weight_scheme)
    if cfg.loss == "weighted_ce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    if cfg.loss == "focal":
        return FocalLoss(weight=None if cfg.class_weight_scheme == "none" else w, gamma=cfg.focal_gamma)
    raise ValueError(f"unknown loss {cfg.loss!r}")


# ------------------------------------------------------------- optimizer / schedule

def build_optimizer(groups, cfg):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9, nesterov=True,
                               weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def lr_factor(step, cfg, steps_per_epoch):
    """LR multiplier at a given optimizer step (warmup, then cosine/step/constant)."""
    warmup = int(cfg.warmup_epochs * steps_per_epoch)
    if step < warmup:
        return (step + 1) / warmup
    if cfg.scheduler == "cosine":
        total = max(cfg.epochs * steps_per_epoch - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(step - warmup, total) / total))
    if cfg.scheduler == "step":
        size = cfg.step_size or max(cfg.epochs // 3, 1)
        return 0.1 ** ((step // steps_per_epoch) // size)
    return 1.0


def build_scheduler(opt, cfg, steps_per_epoch):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, cfg, steps_per_epoch))


# -------------------------------------------------------------------------- helpers

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.bfloat16,
                          enabled=enabled and device.type == "cuda")


def _to_device(x, y, device):
    """uint8 numpy batch from the loader -> normalized float tensor on device
    (preprocessing steps 3-4 run here, on the GPU; see preprocess.normalize)."""
    x = normalize(torch.from_numpy(x).to(device, non_blocking=True))
    return x.contiguous(memory_format=torch.channels_last), torch.from_numpy(y).to(device)


# ---------------------------------------------------------------------- train / eval

def train_one_epoch(model, loader, loss_fn, opt, sched, device, cfg, desc="train"):
    model.train()
    freeze_bn_stats(model)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen, t0 = 0, time.perf_counter()
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for i, (x, y) in enumerate(bar):
        x, y = _to_device(x, y, device)
        with _autocast(device, cfg.amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        loss_sum += loss.detach() * len(y)
        correct += (logits.argmax(1) == y).sum()
        seen += len(y)
        if i % 50 == 0:  # syncs with the GPU, so only now and then
            running = loss_sum.item() / seen
            if not math.isfinite(running):
                raise FloatingPointError(f"training loss became {running} (step {i})")
            bar.set_postfix(loss=f"{running:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
    secs = time.perf_counter() - t0
    train_loss = loss_sum.item() / max(seen, 1)
    if not math.isfinite(train_loss):
        raise FloatingPointError(f"training loss became {train_loss}")
    return {"train_loss": train_loss, "train_acc": correct.item() / max(seen, 1),
            "train_imgs_per_s": seen / secs, "train_secs": secs}


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device, amp=True, desc="eval"):
    """Returns (mean loss, probabilities [N, C], labels [N])."""
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        x, y = _to_device(x, y, device)
        with _autocast(device, amp):
            logits = model(x)
        logits = logits.float()
        if loss_fn is not None:
            loss_sum += loss_fn(logits, y).item() * len(y)
        n += len(y)
        probs.append(logits.softmax(1).cpu())
        labels.append(y.cpu())
    return (loss_sum / max(n, 1), torch.cat(probs).numpy(), torch.cat(labels).numpy())


# ------------------------------------------------------------------------------ fit

@dataclass
class FitResult:
    run_dir: Path
    best_val_macro_f1: float
    best_epoch: int
    history: list = field(default_factory=list)
    stopped_early: bool = False


def make_run_dir(cfg, run_dir=None):
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(cfg.out_dir) / f"{stamp}_{cfg.name}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_history(history, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)


def build_training(cfg, smoke=False, device=None):
    """Data, model, loss, loaders, optimizer and scheduler for a config."""
    device = device or get_device()
    data = build_data(cfg, splits=("train", "validation"), smoke=smoke)
    train, val = data.splits["train"], data.splits["validation"]

    model = build_model(cfg, data.num_classes).to(device, memory_format=torch.channels_last)
    loss_fn = build_loss(cfg, data.train_counts).to(device)
    # Loader workers are cheap (~0.15 GB each, no torch), so validation gets as many
    workers = cfg.num_workers
    val_workers = workers
    check_worker_memory(workers + val_workers)
    sampler = (make_sampler(train.labels, data.num_classes, cfg.sampler_power, cfg.seed)
               if cfg.sampler == "weighted" else None)
    train_loader = make_loader(train.dataset, cfg.batch_size, shuffle=True, sampler=sampler,
                               num_workers=workers, drop_last=True, seed=cfg.seed)
    val_loader = make_loader(val.dataset, cfg.batch_size * 2, shuffle=False, num_workers=val_workers)
    groups = param_groups(model, cfg.model, cfg.lr, cfg.weight_decay,
                          cfg.backbone_lr_mult, cfg.pretrained)
    opt = build_optimizer(groups, cfg)
    sched = build_scheduler(opt, cfg, len(train_loader))
    return dict(device=device, data=data, model=model, loss_fn=loss_fn, opt=opt, sched=sched,
                train_loader=train_loader, val_loader=val_loader)


def fit(cfg, smoke=False, on_epoch_end=None, run_dir=None, resume=False, quiet=False):
    """Train one configuration. `on_epoch_end(epoch, row)` is called after every
    validation (tune.py uses it to report to Optuna and prune; it may raise).
    With resume=True, `run_dir` must hold a last.pt from an earlier run."""
    if smoke:
        cfg.epochs, cfg.num_workers = min(cfg.epochs, 1), min(cfg.num_workers, 2)
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    log = (lambda *a: None) if quiet else print

    run_dir = make_run_dir(cfg, run_dir)
    cfg.save(run_dir / "config.yaml")
    t = build_training(cfg, smoke=smoke)
    device, data, model = t["device"], t["data"], t["model"]
    opt, sched, loss_fn = t["opt"], t["sched"], t["loss_fn"]
    train_loader, val_loader = t["train_loader"], t["val_loader"]
    names = data.class_names

    total, trainable = count_params(model)
    log(f"Run dir: {run_dir}")
    log(f"Model {cfg.model}: {total / 1e6:.2f}M params ({trainable / 1e6:.2f}M trainable) on {device}")
    log(f"Train {len(data.splits['train'].labels):,} images, val {len(data.splits['validation'].labels):,}, "
        f"{data.num_classes} classes, {len(train_loader)} steps/epoch")

    history, best, best_epoch, bad, start = [], -1.0, 0, 0, 0
    if resume:
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        history, best, best_epoch, bad, start = (ck["history"], ck["best"], ck["best_epoch"],
                                                 ck["bad_epochs"], ck["epoch"])
        log(f"Resumed after epoch {start} (best val macro-F1 {best:.4f} at epoch {best_epoch})")

    best_metrics, stopped_early = None, False
    try:
        for epoch in range(start + 1, cfg.epochs + 1):
            t0 = time.perf_counter()
            tr = train_one_epoch(model, train_loader, loss_fn, opt, sched, device, cfg,
                                 desc=f"epoch {epoch}/{cfg.epochs}")
            val_loss, probs, labels = evaluate(model, val_loader, loss_fn, device, cfg.amp, desc="val")
            m = compute_metrics(labels, probs, names)
            row = {"epoch": epoch, **{k: round(v, 6) for k, v in tr.items()},
                   "val_loss": round(val_loss, 6), "val_macro_f1": round(m["macro_f1"], 6),
                   "val_balanced_acc": round(m["balanced_accuracy"], 6),
                   "val_acc": round(m["accuracy"], 6),
                   "head_lr_end": max(g["lr"] for g in opt.param_groups),  # LR after this epoch
                   "epoch_secs": round(time.perf_counter() - t0, 1)}
            history.append(row)
            _write_history(history, run_dir / "history.csv")

            improved = m["macro_f1"] > best + cfg.min_delta
            if improved:
                best, best_epoch, bad, best_metrics = m["macro_f1"], epoch, 0, m
                (run_dir / "val_metrics.json").write_text(json.dumps(m, indent=2))
                if cfg.save_checkpoints:
                    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                                "class_names": names, "epoch": epoch, "val_metrics": m},
                               run_dir / "best.pt")
            else:
                bad += 1
            if cfg.save_checkpoints:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(), "epoch": epoch, "best": best,
                            "best_epoch": best_epoch, "bad_epochs": bad, "history": history},
                           run_dir / "last.pt")

            log(f"epoch {epoch:>3} | train loss {tr['train_loss']:.4f} | val loss {val_loss:.4f} | "
                f"{summary_line(m)} | {row['epoch_secs']:.0f}s{'  *best*' if improved else ''}")
            if on_epoch_end is not None:
                on_epoch_end(epoch, row)
            if bad >= cfg.patience:
                stopped_early = True
                log(f"Early stopping: no val macro-F1 improvement for {cfg.patience} epochs.")
                break
    finally:
        # Stop loader worker processes now (matters when tune.py runs many trials
        # in one process, and when a trial is pruned or crashes mid-run)
        train_loader.close()
        val_loader.close()
        del train_loader, val_loader, t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if history:
        plot_history(history, run_dir / "curves.png")
    if best_metrics is not None:
        save_metrics(best_metrics, run_dir / "val_best")
    summary = {"name": cfg.name, "model": cfg.model, "best_epoch": best_epoch,
               "best_val_macro_f1": best, "epochs_run": len(history), "stopped_early": stopped_early}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Best val macro-F1 {best:.4f} at epoch {best_epoch}. Outputs in {run_dir}")
    return FitResult(run_dir, best, best_epoch, history, stopped_early)


# ------------------------------------------------------------------------ benchmark

def benchmark(cfg, n_batches=50):
    """Throughput of the input pipeline alone and of full training steps, to
    estimate epoch time before committing to a long run."""
    torch.backends.cudnn.benchmark = True
    t = build_training(cfg)
    device, model, loader = t["device"], t["model"], t["train_loader"]
    n_train = len(t["data"].splits["train"].labels)

    # Warm up past worker start-up, cuDNN autotuning and the prefetch buffer
    # (workers x prefetch_factor batches), so both rates are steady-state
    warmup = 5 + cfg.num_workers * 4

    def timed(fn, label):
        it = iter(loader)
        for _ in range(warmup):
            fn(next(it))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0, n = time.perf_counter(), 0
        for _ in range(n_batches):
            x, y = next(it)
            fn((x, y))
            n += len(y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rate = n / (time.perf_counter() - t0)
        print(f"  {label:<26} {rate:8.0f} img/s  -> {n_train / rate / 60:5.1f} min per train epoch")
        return rate

    def step(batch):
        x, y = _to_device(*batch, device)
        with _autocast(device, cfg.amp):
            logits = model(x)
        loss = t["loss_fn"](logits.float(), y)
        t["opt"].zero_grad(set_to_none=True)
        loss.backward()
        t["opt"].step()

    print(f"Benchmark: {cfg.model} @ {cfg.image_size}px, batch {cfg.batch_size}, "
          f"{cfg.num_workers} workers, aug {cfg.aug}, {n_train:,} train images, device {device}")
    model.train()
    freeze_bn_stats(model)
    loader_rate = timed(lambda b: None, "data loader only")
    train_rate = timed(step, "loader + train step")
    if device.type == "cuda":
        print(f"  peak GPU memory            {torch.cuda.max_memory_allocated() / 2**30:8.2f} GB")
    bound = "data loading" if loader_rate < train_rate * 1.15 else "the GPU"
    print(f"  bottleneck looks like: {bound}")
    loader.close()
    t["val_loader"].close()
    del loader, t
    gc.collect()
    return {"loader_imgs_per_s": loader_rate, "train_imgs_per_s": train_rate}
