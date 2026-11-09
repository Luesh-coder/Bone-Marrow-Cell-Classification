"""
Evaluate a trained model (proposal slide 11), and compare runs.

    # Any time, on validation
    python evaluate.py --checkpoint runs/<run>/best.pt --split validation

    # ONCE per final model, on the held-out test set
    python evaluate.py --checkpoint runs/<run>/best.pt --split test

    # Side-by-side table, e.g. baseline vs ResNet-50
    python evaluate.py --compare runs/<baseline_run> runs/<resnet_run> --split test

Writes runs/<run>/eval_<split>/: metrics.json, per_class.csv, confusion_matrix.png
and predictions.npz (probabilities + labels, for ROC curves or error analysis).

The test set is never used for tuning or model selection. After a test
evaluation, a TEST_EVALUATED marker is written and a second test run of the same
checkpoint is refused unless --force is given.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

# Heavy imports (torch, engine, data, ...) live inside functions: on Windows every
# loader worker process re-runs this file's top level, and torch alone costs ~1 GB each.
from cli import run_and_exit
from config import TrainConfig


def run_eval(checkpoint, split, force=False, batch_size=128, num_workers=8, smoke=False):
    import numpy as np
    import torch

    from data import build_data, check_worker_memory, make_loader
    from engine import evaluate, get_device
    from metrics import compute_metrics, save_metrics, summary_line
    from models import build_model

    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent
    out_dir = run_dir / f"eval_{split}{'_smoke' if smoke else ''}"
    marker = run_dir / f"eval_{split}" / "TEST_EVALUATED"
    if split == "test" and not smoke and marker.exists() and not force:
        raise SystemExit(f"{checkpoint} was already evaluated on the test set "
                         f"({marker.read_text().strip()}). Use --force to re-run.")

    device = get_device()
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_dict(ck["config"])
    cfg.pretrained = False                     # weights come from the checkpoint
    model = build_model(cfg, len(ck["class_names"]))
    model.load_state_dict(ck["model"])
    model.to(device, memory_format=torch.channels_last)

    data = build_data(cfg, splits=(split,), smoke=smoke)
    if data.class_names != ck["class_names"]:
        raise SystemExit("class names in the checkpoint don't match the data config")
    check_worker_memory(num_workers)
    loader = make_loader(data.splits[split].dataset, batch_size, num_workers=num_workers)
    try:
        _, probs, labels = evaluate(model, loader, None, device, cfg.amp, desc=split)
    finally:
        loader.close()
    m = compute_metrics(labels, probs, data.class_names)
    m.update({"checkpoint": str(checkpoint), "split": split, "epoch": ck.get("epoch")})

    save_metrics(m, out_dir)
    np.savez_compressed(out_dir / "predictions.npz", probs=probs, labels=labels,
                        class_names=np.array(data.class_names))
    if split == "test" and not smoke:
        marker.write_text(f"evaluated {datetime.now():%Y-%m-%d %H:%M}\n")

    print(f"{cfg.name} on {split} ({m['n']:,} images): {summary_line(m)}")
    print(f"\n{'class':<6} {'prec':>6} {'recall':>6} {'F1':>6} {'AUC':>6} {'n':>6}")
    for r in m["per_class"]:
        auc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "  n/a"
        print(f"{r['class']:<6} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} {auc:>6} {r['support']:6d}")
    print(f"\nSaved to {out_dir}")
    return m


def compare(run_dirs, split):
    rows = []
    for rd in map(Path, run_dirs):
        path = rd / f"eval_{split}" / "metrics.json"
        if not path.exists() and split == "validation":
            path = rd / "val_metrics.json"
        if not path.exists():
            print(f"[skip] {rd}: no {split} metrics (run evaluate.py --split {split} first)")
            continue
        m = json.loads(path.read_text())
        name = rd.name
        cfg_path = rd / "config.yaml"
        if cfg_path.exists():
            import yaml
            c = yaml.safe_load(cfg_path.read_text())
            name = f"{c.get('name', rd.name)} ({c.get('model')})"
        rows.append((name, m))
    if not rows:
        return
    width = max(len(n) for n, _ in rows) + 2
    print(f"\n{split} split\n{'run':<{width}} {'macro-F1':>9} {'bal. acc':>9} {'accuracy':>9} {'ROC-AUC':>9}")
    for name, m in rows:
        auc = f"{m['macro_roc_auc']:.4f}" if m.get("macro_roc_auc") is not None else "n/a"
        print(f"{name:<{width}} {m['macro_f1']:9.4f} {m['balanced_accuracy']:9.4f} "
              f"{m['accuracy']:9.4f} {auc:>9}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="path to best.pt")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--force", action="store_true", help="allow a repeated test evaluation")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="evaluate on ~1k images only (no test marker)")
    p.add_argument("--compare", nargs="+", metavar="RUN_DIR", help="print a comparison table")
    args = p.parse_args()

    if args.compare:
        compare(args.compare, args.split)
    elif args.checkpoint:
        run_eval(args.checkpoint, args.split, args.force, args.batch_size, args.num_workers, args.smoke)
    else:
        p.error("give --checkpoint or --compare")


if __name__ == "__main__":
    run_and_exit(main)   # avoids a segfault at interpreter exit (see cli.py)
