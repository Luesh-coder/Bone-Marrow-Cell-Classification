"""
Hyperparameter search with Optuna (proposal slide 10).

Plan: a coarse random search on a stratified 20% training subset, full training
only for the top 2-3 configurations, everything chosen by validation macro-F1.
The test set is never used here.

    # 1. Random search: 30 new trials, 8 epochs each, on 20% of train
    python tune.py search --model resnet50 --trials 30
    python tune.py search --model baseline --trials 30 --study baseline_search

    # 2. Rank trials, export a CSV, write the top-k as full-run configs
    python tune.py top --study resnet50_search --k 3

    # 3. Train the top-k configs on the full training set
    python tune.py full --study resnet50_search --k 3
    #    (or run them one by one: python train.py --config configs/tuned/resnet50_search_rank1.yaml)

Search space (slide 10 plus the imbalance options of slide 9):
    lr               1e-4 .. 1e-2 (log)        batch_size     32, 64
    optimizer        adamw, sgd (mom. 0.9)     weight_decay   1e-5 .. 1e-3 (log)
    dropout          0.2 .. 0.5                image_size     128, 224
    unfreeze         head, last_block, all     scheduler      cosine, step
    aug              light, medium, strong     loss           weighted_ce, focal
    sampler          none, weighted            (unfreeze: pretrained models only)

Trials that fall below the median of earlier trials at the same epoch are
stopped early (MedianPruner, after 5 trials and 2 epochs). Progress is stored in
runs/tuning/<study>.db, so an interrupted search resumes when re-run with the same
--study; --trials counts new trials for this call.
"""

import argparse
from pathlib import Path

# Heavy imports (torch, engine, data, ...) live inside functions: on Windows every
# loader worker process re-runs this file's top level, and torch alone costs ~1 GB each.
from cli import run_and_exit
from config import TrainConfig, load_config

TUNING_DIR = Path("runs/tuning")
TUNED_CONFIG_DIR = Path("configs/tuned")


def storage_url(study):
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(TUNING_DIR / f'{study}.db').resolve().as_posix()}"


def suggest_config(trial, base: TrainConfig) -> TrainConfig:
    p = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "sgd"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "image_size": trial.suggest_categorical("image_size", [128, 224]),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "step"]),
        "aug": trial.suggest_categorical("aug", ["light", "medium", "strong"]),
        "loss": trial.suggest_categorical("loss", ["weighted_ce", "focal"]),
        "sampler": trial.suggest_categorical("sampler", ["none", "weighted"]),
    }
    if base.model != "baseline" and base.pretrained:
        p["unfreeze"] = trial.suggest_categorical("unfreeze", ["head", "last_block", "all"])
    return TrainConfig.from_dict({**base.to_dict(), **p})


def export_trials(study, path):
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    if len(df):
        df = df.sort_values("value", ascending=False, na_position="last")
        df.to_csv(path, index=False)
    return df


def cmd_search(args):
    import optuna
    import torch

    from engine import fit

    study_name = args.study or f"{args.model}_search"
    base = load_config(args.base_config or f"configs/{args.model}.yaml", args.set)
    base.subset_fraction = args.subset
    base.epochs = args.epochs
    base.patience = args.patience
    base.save_checkpoints = False
    base.out_dir = str(TUNING_DIR / study_name)

    pruner = (optuna.pruners.NopPruner() if args.no_prune else
              optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
    study = optuna.create_study(
        study_name=study_name, storage=storage_url(study_name), direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=args.seed), pruner=pruner, load_if_exists=True)
    study.set_user_attr("model", args.model)
    study.set_user_attr("subset_fraction", args.subset)
    study.set_user_attr("epochs", args.epochs)

    def objective(trial):
        cfg = suggest_config(trial, base)
        cfg.name = f"trial{trial.number:03d}"
        trial.set_user_attr("config", cfg.to_dict())

        def report(epoch, row):
            trial.report(row["val_macro_f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after epoch {epoch}")

        try:
            res = fit(cfg, smoke=args.smoke, on_epoch_end=report,
                      run_dir=Path(base.out_dir) / f"trial_{trial.number:03d}")
        except FloatingPointError as exc:       # diverged (e.g. lr too high for SGD)
            trial.set_user_attr("failure", str(exc))
            raise optuna.TrialPruned(f"diverged: {exc}") from None
        except torch.cuda.OutOfMemoryError:
            trial.set_user_attr("failure", "CUDA out of memory")
            raise optuna.TrialPruned("CUDA out of memory") from None
        trial.set_user_attr("best_epoch", res.best_epoch)
        trial.set_user_attr("run_dir", str(res.run_dir))
        return res.best_val_macro_f1

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, gc_after_trial=True)
    export_trials(study, TUNING_DIR / f"{study_name}_trials.csv")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nStudy {study_name}: {len(study.trials)} trials, {len(completed)} completed.")
    if completed:
        print(f"Best so far: trial {study.best_trial.number}, val macro-F1 {study.best_value:.4f}")
        print(f"  {study.best_trial.params}")
    print(f"Next: python tune.py top --study {study_name} --k 3")


def cmd_top(args):
    import optuna

    study = optuna.load_study(study_name=args.study, storage=storage_url(args.study))
    df = export_trials(study, TUNING_DIR / f"{args.study}_trials.csv")
    done = sorted((t for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None),
                  key=lambda t: t.value, reverse=True)
    if not done:
        raise SystemExit("no completed trials yet")

    print(f"Study {args.study}: {len(study.trials)} trials, {len(done)} completed. "
          f"All trials -> {TUNING_DIR / f'{args.study}_trials.csv'}")
    try:
        imp = optuna.importance.get_param_importances(study)
        print("\nParameter importance (fANOVA):")
        for k, v in imp.items():
            print(f"  {k:<14} {v:.3f}")
    except Exception as exc:  # needs enough completed trials
        print(f"\n(parameter importance unavailable: {exc})")

    TUNED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nTop {min(args.k, len(done))} configurations (full-run configs written):")
    for rank, t in enumerate(done[:args.k], 1):
        cfg = TrainConfig.from_dict(t.user_attrs["config"])
        full = load_config(f"configs/{cfg.model}.yaml")
        cfg.subset_fraction = 1.0
        cfg.epochs = args.epochs or full.epochs
        cfg.patience = full.patience
        cfg.save_checkpoints = True
        cfg.out_dir = full.out_dir
        cfg.num_workers = full.num_workers
        cfg.name = f"{args.study}_rank{rank}"
        path = TUNED_CONFIG_DIR / f"{args.study}_rank{rank}.yaml"
        cfg.save(path)
        print(f"  #{rank} trial {t.number:>3}  val macro-F1 {t.value:.4f}  -> {path}")
        print(f"       {t.params}")


def cmd_full(args):
    from engine import fit

    paths = sorted(TUNED_CONFIG_DIR.glob(f"{args.study}_rank*.yaml"))[:args.k]
    if not paths:
        raise SystemExit(f"no configs/tuned/{args.study}_rank*.yaml; run `tune.py top` first")
    results = []
    for path in paths:
        cfg = load_config(path, args.set)
        print(f"\n=== {path} ===")
        res = fit(cfg, smoke=args.smoke)
        results.append((path.stem, res))
    print("\nFull runs (validation macro-F1):")
    for name, res in sorted(results, key=lambda r: r[1].best_val_macro_f1, reverse=True):
        print(f"  {name:<32} {res.best_val_macro_f1:.4f}  (epoch {res.best_epoch})  {res.run_dir}")
    print("Evaluate the winner ONCE on test: python evaluate.py --checkpoint <run>/best.pt --split test")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="run random-search trials")
    s.add_argument("--model", choices=("baseline", "resnet50", "efficientnet_b0"), default="resnet50")
    s.add_argument("--study", help="study name (default: <model>_search)")
    s.add_argument("--base-config", help="config the search starts from (default: configs/<model>.yaml)")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a base field")
    s.add_argument("--trials", type=int, default=30, help="number of NEW trials to run")
    s.add_argument("--timeout", type=float, help="stop starting new trials after this many seconds")
    s.add_argument("--subset", type=float, default=0.2, help="fraction of train used per trial")
    s.add_argument("--epochs", type=int, default=8)
    s.add_argument("--patience", type=int, default=3)
    s.add_argument("--seed", type=int, default=0, help="sampler seed")
    s.add_argument("--no-prune", action="store_true")
    s.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch per trial")
    s.set_defaults(fn=cmd_search)

    t = sub.add_parser("top", help="rank trials, export CSV, write top-k full-run configs")
    t.add_argument("--study", required=True)
    t.add_argument("--k", type=int, default=3)
    t.add_argument("--epochs", type=int, help="epochs for the full runs (default: configs/<model>.yaml)")
    t.set_defaults(fn=cmd_top)

    f = sub.add_parser("full", help="train the top-k configs on the full training set")
    f.add_argument("--study", required=True)
    f.add_argument("--k", type=int, default=3)
    f.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    f.add_argument("--smoke", action="store_true")
    f.set_defaults(fn=cmd_full)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    run_and_exit(main)   # avoids a segfault at interpreter exit (see cli.py)
