"""
Train one model configuration.

Examples:
    # Baseline CNN from scratch, then fine-tuned ResNet-50 (proposal slide 8)
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/resnet50.yaml

    # Override any config field
    python scripts/train.py --config configs/resnet50.yaml --set lr=1e-4 --set unfreeze=last_block

    # Quick end-to-end check (~2k train / 1k val images, 1 epoch)
    python scripts/train.py --config configs/resnet50.yaml --smoke

    # Measure throughput and estimated epoch time without training
    python scripts/train.py --config configs/resnet50.yaml --benchmark 50

    # Continue an interrupted run from its last.pt
    python scripts/train.py --resume runs/20261109-160000_resnet50

Each run writes runs/<timestamp>_<name>/ (see bone_marrow/engine.py). The model is
selected on validation macro-F1; the test split is only touched by scripts/evaluate.py.
Run from the repo root: configs/ and runs/ paths are relative to it.
"""

import argparse
import sys
from pathlib import Path

# `python scripts/train.py` puts scripts/ on sys.path, not the repo root, so add the
# root to make the bone_marrow package importable (loader workers inherit sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Heavy imports (torch, engine, data, ...) live inside functions: on Windows every
# loader worker process re-runs this file's top level, and torch alone costs ~1 GB each.
from bone_marrow.cli import run_and_exit
from bone_marrow.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config, e.g. configs/resnet50.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config field (repeatable)")
    p.add_argument("--name", help="run name (default: config's name)")
    p.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch: checks the pipeline end to end")
    p.add_argument("--resume", metavar="RUN_DIR", help="continue a run from RUN_DIR/last.pt")
    p.add_argument("--benchmark", type=int, metavar="N", help="time N batches instead of training")
    return p.parse_args()


def main():
    from bone_marrow.engine import benchmark, fit

    args = parse_args()
    if args.resume:
        run_dir = Path(args.resume)
        cfg = load_config(run_dir / "config.yaml", args.set)
        fit(cfg, run_dir=run_dir, resume=True)
        return

    if not args.config:
        raise SystemExit("--config is required (or --resume RUN_DIR)")
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg.name = args.name
    if args.benchmark:
        benchmark(cfg, args.benchmark)
        return
    if args.smoke:
        cfg.name = f"{cfg.name}_smoke"
    fit(cfg, smoke=args.smoke)


if __name__ == "__main__":
    run_and_exit(main)   # avoids a segfault at interpreter exit (see cli.py)
