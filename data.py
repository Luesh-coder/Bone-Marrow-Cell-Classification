"""
Datasets and batch loaders for training, tuning and evaluation.

Builds on preprocess.py (which does the per-image work) and adds:
    - label access straight from the Arrow column (fast, never decodes images)
    - stratified subsets (20% of train for tuning, tiny ones for --smoke runs)
      that keep the rare classes
    - optional class merging / dropping (proposal slide 9, option 4)
    - class weights for the loss and a WeightedRandomSampler (slide 9, options 1-2)
    - batch loaders (loader.py) that run the transform once per batch (HF
      `__getitems__`) in torch-free worker processes and yield uint8 numpy
      batches, normalized later on the GPU

Subsets use `Dataset.select`, which only stores an index mapping. Never call
`Dataset.map` over the images: it would rewrite the ~7 GB dataset to disk.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from datasets import DatasetDict, load_dataset

from loader import BatchLoader, WeightedSampler, collate  # noqa: F401  (collate re-exported)
from preprocess import REPO_ID, with_preprocessing

# Label order of the dataset (index -> abbreviation), as in the dataset card
CLASS_CODES = [
    "ABE", "ART", "BAS", "BLA", "EBO", "EOS", "FGC", "HAC", "KSC", "LYI", "LYT",
    "MMZ", "MON", "MYB", "NGB", "NGS", "NIF", "OTH", "PEB", "PLM", "PMO",
]

SMOKE_SIZES = {"train": 2000, "validation": 1000, "test": 1000}

# Images that cannot be decoded, excluded from every run. Found by check_images.py
# (2026-11-09): train row 83795 (MYB) raises "broken data stream" in PIL. Its bytes
# are identical in the Hugging Face source (train-00007-of-00012.parquet), so the
# dataset itself is corrupt there, not the local cache. Validation and test are clean.
CORRUPT_IMAGES = {"train": (83795,)}


def load_raw() -> DatasetDict:
    """All three splits, untransformed (uses the local Hugging Face cache)."""
    return load_dataset(REPO_ID)


def class_names_of(d) -> list[str]:
    try:
        return list(d.features["label"].names)
    except (AttributeError, KeyError):
        return list(CLASS_CODES)


def labels_of(d) -> np.ndarray:
    """Labels of a split as an int64 array, read from Arrow without touching images
    or the transform. Respects any `select()` index mapping."""
    return np.asarray(d.select_columns(["label"]).with_format("numpy")[:]["label"],
                      dtype=np.int64)


def stratified_indices(labels, fraction, seed=0, min_per_class=50) -> np.ndarray:
    """Random per-class sample of `fraction` of each class, but at least
    min(class size, min_per_class) images, so rare classes survive subsetting."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        k = max(int(round(fraction * len(idx))), min(len(idx), min_per_class))
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=np.int64)


def keep_indices(labels, lut=None, exclude=()):
    """Row indices to keep: not in `exclude` (corrupt images) and not mapped to a
    dropped class (lut value -1). Returns None when every row is kept."""
    keep = np.ones(len(labels), dtype=bool)
    exclude = [i for i in exclude if i < len(labels)]
    keep[exclude] = False
    if lut is not None:
        keep &= lut[labels] >= 0
    return None if keep.all() else np.flatnonzero(keep)


def class_map(class_names, merge=None, drop=None):
    """Lookup table original label -> new label (-1 = dropped) and the new names.

    merge: {"ABE": "EOS"} folds ABE into EOS; drop: ["KSC"] removes KSC.
    Returns (None, class_names) when nothing changes.
    """
    merge, drop = dict(merge or {}), list(drop or [])
    if not merge and not drop:
        return None, list(class_names)
    for c in list(merge) + list(merge.values()) + drop:
        if c not in class_names:
            raise ValueError(f"unknown class {c!r}; classes are {class_names}")
    for src, dst in merge.items():
        if dst in merge or dst in drop:
            raise ValueError(f"merge target {dst!r} is itself merged or dropped")
    new_names = [c for c in class_names if c not in merge and c not in drop]
    lut = np.array([-1 if c in drop else new_names.index(merge.get(c, c))
                    for c in class_names], dtype=np.int64)
    return lut, new_names


def class_weights(counts, scheme="inv_sqrt", beta=0.999) -> torch.Tensor:
    """Per-class loss weights, normalized to mean 1 over classes.

    inverse: 1/n (very aggressive, ~3900:1 here); inv_sqrt: 1/sqrt(n) (default);
    effective_num: (1 - beta) / (1 - beta^n) from Cui et al. (2019); none: all 1.
    """
    n = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    if scheme == "none":
        w = np.ones_like(n)
    elif scheme == "inverse":
        w = 1.0 / n
    elif scheme == "inv_sqrt":
        w = 1.0 / np.sqrt(n)
    elif scheme == "effective_num":
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    else:
        raise ValueError(f"unknown class weight scheme {scheme!r}")
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def make_sampler(labels, num_classes, power=0.5, seed=0) -> WeightedSampler:
    """Sample each image with probability proportional to count(class) ** -power.

    power=1 gives perfectly balanced batches but replays the 6 ABE images
    thousands of times per epoch; 0.5 is a softer middle ground.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    per_class = np.where(counts > 0, np.maximum(counts, 1) ** -power, 0.0)
    return WeightedSampler(per_class[labels], seed=seed)


# Loader workers are separate processes (Windows "spawn") importing numpy, PIL and
# datasets but not torch: ~0.15 GB of committed memory each (measured), versus
# ~1 GB for a torch DataLoader worker. Workers parallelize by being many processes,
# so they get one BLAS/OpenMP thread each (16 OpenBLAS threads reserve ~0.5 GB);
# spawned children inherit these environment variables.
WORKER_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
GB_PER_WORKER = 0.25
GB_MAIN_PROCESS = 4.0      # training process with CUDA context, model and batches in flight


def available_commit_gb():
    """Free commit charge (RAM + page file) in GB on Windows; None elsewhere."""
    if os.name != "nt":
        return None
    import ctypes

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    s = MemStatus()
    s.dwLength = ctypes.sizeof(MemStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        return None
    return s.ullAvailPageFile / 2**30


def check_worker_memory(total_workers):
    """Warn (with a concrete fix) when training is unlikely to fit in free commit
    memory; otherwise it fails part-way with MemoryError."""
    free = available_commit_gb()
    need = GB_MAIN_PROCESS + total_workers * GB_PER_WORKER
    if free is not None and need > free - 2.0:
        print(f"[warn] training needs ~{need:.0f} GB of commit memory (RAM + page file) but only "
              f"~{free:.0f} GB is free. Close other programs, lower num_workers, or enlarge the "
              f"Windows page file.")


def make_loader(d, batch_size, shuffle=False, sampler=None, num_workers=8,
                drop_last=False, seed=0) -> BatchLoader:
    """Loader yielding (uint8 images, int64 labels) as numpy arrays (see loader.py).
    Call .close() when done to stop its worker processes."""
    if num_workers > 0:
        for key in WORKER_THREAD_ENV:
            os.environ.setdefault(key, "1")
    return BatchLoader(d, batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                       drop_last=drop_last, seed=seed)


@dataclass
class SplitData:
    dataset: object          # transformed datasets.Dataset
    labels: np.ndarray       # training-space labels (after merge/drop), aligned with rows


@dataclass
class DataBundle:
    splits: dict             # split name -> SplitData
    class_names: list
    train_counts: np.ndarray | None  # per class, training space

    @property
    def num_classes(self):
        return len(self.class_names)


def build_data(cfg, splits=("train", "validation"), smoke=False, raw=None) -> DataBundle:
    """Prepare the requested splits for a TrainConfig.

    Train gets augmentation, the tuning subset (cfg.subset_fraction) and, if
    cfg.rare_aug is set, the stronger preset for classes with fewer than
    cfg.rare_threshold training images. Other splits are never augmented or
    subset (except in smoke mode).
    """
    raw = load_raw() if raw is None else raw
    original_names = class_names_of(raw["train"])
    lut, names = class_map(original_names, cfg.merge, cfg.drop)

    rare = ()
    if cfg.rare_aug:
        counts = np.bincount(labels_of(raw["train"]), minlength=len(original_names))
        rare = tuple(int(c) for c in np.flatnonzero(counts < cfg.rare_threshold))

    out, train_counts = {}, None
    for split in splits:
        d = raw[split]
        y = labels_of(d)
        keep = keep_indices(y, lut, CORRUPT_IMAGES.get(split, ()))  # corrupt + dropped classes
        if keep is not None:
            d, y = d.select(keep), y[keep]
        if split == "train" and cfg.subset_fraction < 1:
            idx = stratified_indices(y, cfg.subset_fraction, cfg.seed, cfg.subset_min_per_class)
            d, y = d.select(idx), y[idx]
        if smoke:
            n = SMOKE_SIZES.get(split, 1000)
            if len(y) > n:
                idx = stratified_indices(y, n / len(y), cfg.seed, min_per_class=2)
                d, y = d.select(idx), y[idx]

        is_train = split == "train"
        d = with_preprocessing(d, train=is_train, image_size=cfg.image_size,
                               aug=cfg.aug if is_train else "none",
                               rare_classes=rare if is_train else (),
                               rare_aug=cfg.rare_aug if is_train else None,
                               label_lut=lut, output="uint8")
        y = lut[y] if lut is not None else y
        out[split] = SplitData(d, y)
        if is_train:
            train_counts = np.bincount(y, minlength=len(names))
    return DataBundle(out, names, train_counts)
