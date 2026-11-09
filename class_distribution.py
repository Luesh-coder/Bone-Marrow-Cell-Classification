"""
Per-class distribution chart (log scale) for the bone marrow cell dataset.

Dataset: https://huggingface.co/datasets/ekim15/bone_marrow_cell_dataset

Reads ONLY the "label" column from the dataset's Parquet files over HTTP range
requests, so it does not download the ~6.9 GB of images. Falls back to the
`datasets` library (full download) if that fails.

Outputs:
    class_distribution.png   grouped bar chart, train / validation / test, log y-axis
    class_distribution.csv   per-class counts per split, for your slides

Install:
    pip install huggingface_hub pyarrow pandas matplotlib datasets

Run:
    python class_distribution.py
"""

from pathlib import PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ID = "ekim15/bone_marrow_cell_dataset"
SPLITS = ["train", "validation", "test"]

# Label order from the dataset card (index -> abbreviation)
CLASS_CODES = [
    "ABE", "ART", "BAS", "BLA", "EBO", "EOS", "FGC", "HAC", "KSC", "LYI", "LYT",
    "MMZ", "MON", "MYB", "NGB", "NGS", "NIF", "OTH", "PEB", "PLM", "PMO",
]
CLASS_NAMES = {
    "ABE": "Abnormal eosinophil", "ART": "Artefact", "BAS": "Basophil",
    "BLA": "Blast", "EBO": "Erythroblast", "EOS": "Eosinophil",
    "FGC": "Faggot cell", "HAC": "Hairy cell", "KSC": "Smudge cell",
    "LYI": "Immature lymphocyte", "LYT": "Lymphocyte", "MMZ": "Metamyelocyte",
    "MON": "Monocyte", "MYB": "Myelocyte", "NGB": "Band neutrophil",
    "NGS": "Segmented neutrophil", "NIF": "Not identifiable", "OTH": "Other cell",
    "PEB": "Proerythroblast", "PLM": "Plasma cell", "PMO": "Promyelocyte",
}


def label_names_from_hub():
    """Read class names from the dataset's metadata (no image download)."""
    try:
        from datasets import load_dataset_builder

        names = load_dataset_builder(REPO_ID).info.features["label"].names
        if len(names) == len(CLASS_CODES):
            return list(names)
    except Exception as exc:  # metadata lookup is optional
        print(f"[info] Using built-in class list ({exc.__class__.__name__}).")
    return CLASS_CODES


def split_of(path: str):
    """Infer the split from a Parquet file path, e.g. data/train-00000-of-00012.parquet."""
    p = PurePosixPath(path)
    for split in SPLITS:
        if p.name.startswith(split) or split in p.parts:
            return split
    return None


def counts_label_only():
    """Fast path: read just the label column from the Hub Parquet files."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = fs.glob(f"datasets/{REPO_ID}/**/*.parquet")
    if not files:
        raise FileNotFoundError("No Parquet files found in the dataset repo.")

    labels = {s: [] for s in SPLITS}
    for f in files:
        split = split_of(f)
        if split is None:
            continue
        with fs.open(f, "rb") as fh:
            table = pq.read_table(fh, columns=["label"])
        labels[split].append(table.column("label").to_numpy())
        print(f"  read {split:<10} {PurePosixPath(f).name}  ({table.num_rows:,} rows)")

    return {s: np.concatenate(v) for s, v in labels.items() if v}


def counts_full_download():
    """Fallback: load through `datasets` (downloads the whole dataset once)."""
    from datasets import load_dataset

    ds = load_dataset(REPO_ID)
    return {s: np.asarray(ds[s]["label"]) for s in SPLITS if s in ds}


def build_table(labels_by_split, codes):
    n = len(codes)
    df = pd.DataFrame(
        {s: np.bincount(labels_by_split[s], minlength=n) for s in labels_by_split},
        index=codes,
    )
    df["total"] = df.sum(axis=1)
    df.insert(0, "name", [CLASS_NAMES.get(c, c) for c in codes])
    return df.sort_values("total", ascending=False)


def plot(df, out_path="class_distribution.png"):
    splits = [s for s in SPLITS if s in df.columns]
    x = np.arange(len(df))
    width = 0.8 / len(splits)
    colors = {"train": "#2563eb", "validation": "#f59e0b", "test": "#10b981"}

    fig, ax = plt.subplots(figsize=(14, 6.5))
    for i, s in enumerate(splits):
        # log scale cannot show 0, so empty bars are simply not drawn
        vals = df[s].to_numpy().astype(float)
        vals[vals == 0] = np.nan
        ax.bar(x + (i - (len(splits) - 1) / 2) * width, vals, width,
               label=f"{s} ({int(df[s].sum()):,})", color=colors.get(s))

    # total count above each class group
    for xi, total in zip(x, df["total"]):
        ax.text(xi, total * 1.15, f"{total:,}", ha="center", va="bottom",
                fontsize=7.5, rotation=90)

    ax.set_yscale("log")
    ax.set_ylim(0.8, df["total"].max() * 6)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=10)
    ax.set_xlabel("Cell class (sorted by total count)")
    ax.set_ylabel("Number of images (log scale)")
    ratio = df["total"].max() / max(df["total"].min(), 1)
    ax.set_title(
        f"Bone marrow cell dataset: per-class counts, {int(df['total'].sum()):,} images, "
        f"{len(df)} classes (largest:smallest ≈ {ratio:,.0f}:1)",
        fontsize=12,
    )
    ax.grid(axis="y", which="major", alpha=0.3)
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved chart  -> {out_path}")
    plt.show()


def main():
    codes = label_names_from_hub()
    print(f"Counting labels for {REPO_ID} ...")
    try:
        labels = counts_label_only()
    except Exception as exc:
        print(f"[warn] Label-only read failed ({exc}); falling back to full download.")
        labels = counts_full_download()

    df = build_table(labels, codes)
    df.to_csv("class_distribution.csv")
    print("Saved counts -> class_distribution.csv\n")
    with pd.option_context("display.width", 120):
        print(df.to_string())

    # Flag classes that are thin or missing in the held-out splits
    for s in ("validation", "test"):
        if s in df.columns:
            thin = df[df[s] < 5]
            if len(thin):
                print(f"\n[note] Classes with fewer than 5 {s} images: "
                      + ", ".join(f"{c} ({n})" for c, n in thin[s].items()))

    plot(df)


if __name__ == "__main__":
    main()