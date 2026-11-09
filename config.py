"""
Training configuration: one dataclass holding every knob from the proposal's
tuning plan (slide 10) and imbalance options (slide 9), plus run settings.

Configs live in YAML files (configs/*.yaml). Any field can be overridden from the
command line with repeatable `--set key=value` flags, e.g.
    python train.py --config configs/resnet50.yaml --set lr=1e-4 --set unfreeze=last_block

Values are converted to the field's type, so `lr=1e-4` is a float even though
YAML on its own would read "1e-4" as a string.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

CHOICES = {
    "model": ("baseline", "resnet50", "efficientnet_b0"),
    "unfreeze": ("head", "last_block", "all"),
    "optimizer": ("adamw", "sgd"),
    "scheduler": ("cosine", "step", "none"),
    "aug": ("none", "light", "medium", "strong"),
    "rare_aug": (None, "none", "light", "medium", "strong"),
    "loss": ("ce", "weighted_ce", "focal"),
    "class_weight_scheme": ("none", "inverse", "inv_sqrt", "effective_num"),
    "sampler": ("none", "weighted"),
}


@dataclass
class TrainConfig:
    # Run
    name: str = "run"
    seed: int = 42
    out_dir: str = "runs"
    save_checkpoints: bool = True
    num_workers: int = 12                # loader worker processes (no torch, ~0.15 GB each)
    amp: bool = True                    # bf16 autocast on CUDA

    # Model
    model: str = "baseline"             # baseline | resnet50 | efficientnet_b0
    pretrained: bool = True             # ImageNet weights (ignored for baseline)
    unfreeze: str = "all"               # head | last_block | all (pretrained models)
    dropout: float = 0.3                # dropout in the classifier head

    # Input / augmentation
    image_size: int = 224               # 128 or 224
    aug: str = "medium"                 # none | light | medium | strong
    rare_aug: str | None = None         # preset for rare classes (None = same as aug)
    rare_threshold: int = 500           # "rare" = fewer train images than this

    # Optimisation
    epochs: int = 20
    patience: int = 5                   # early stopping on val macro-F1
    min_delta: float = 0.0
    batch_size: int = 64
    optimizer: str = "adamw"            # adamw | sgd (momentum 0.9, nesterov)
    lr: float = 1e-3                    # head LR; pretrained layers get lr * backbone_lr_mult
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    scheduler: str = "cosine"           # cosine | step | none (all with linear warmup)
    warmup_epochs: float = 1.0
    step_size: int = 0                  # step decay every N epochs (0 = epochs // 3)
    grad_clip: float = 0.0              # 0 = off

    # Class imbalance
    loss: str = "weighted_ce"           # ce | weighted_ce | focal
    class_weight_scheme: str = "inv_sqrt"  # none | inverse | inv_sqrt | effective_num
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0        # cross-entropy only
    sampler: str = "none"               # none | weighted (WeightedRandomSampler)
    sampler_power: float = 0.5          # sample weight per class = count ** -power
    merge: dict = field(default_factory=dict)   # e.g. {ABE: EOS} folds ABE into EOS
    drop: list = field(default_factory=list)    # e.g. [KSC] removes KSC everywhere

    # Data subset (tuning uses 0.2)
    subset_fraction: float = 1.0
    subset_min_per_class: int = 50      # keep at least this many of each class (or all)

    def validate(self):
        for key, allowed in CHOICES.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key}={getattr(self, key)!r}; choose from {allowed}")
        if self.rare_aug == "none":
            self.rare_aug = None
        if not 0 < self.subset_fraction <= 1:
            raise ValueError("subset_fraction must be in (0, 1]")
        return self

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, d):
        known = {f.name: f for f in fields(cls)}
        unknown = set(d) - set(known)
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        defaults = cls()
        kwargs = {k: _coerce(v, getattr(defaults, k), k) for k, v in d.items()}
        return cls(**kwargs).validate()


def _coerce(value, default, key):
    """Convert `value` (from YAML or the command line) to the type of `default`."""
    if isinstance(value, str) and not isinstance(default, str):
        if key == "rare_aug":
            return None if value.lower() in ("", "none", "null") else value
        value = yaml.safe_load(value)          # lists, dicts, bools, numbers
    if key == "rare_aug":
        return None if value in (None, "none", "null", "") else str(value)
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, dict):
        return dict(value or {})
    if isinstance(default, list):
        if isinstance(value, str):
            return [v for v in value.split(",") if v]
        return list(value or [])
    return value


def load_config(path=None, overrides=()):
    """Read a YAML config (optional) and apply `key=value` overrides."""
    d = {}
    if path:
        d = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        d[k.strip()] = v.strip()
    return TrainConfig.from_dict(d)
