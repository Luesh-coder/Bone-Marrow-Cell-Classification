"""
Evaluation metrics (proposal slide 11) and plots.

compute_metrics(y_true, y_prob, class_names) returns:
    macro_f1           primary metric: F1 averaged over classes present in y_true,
                       every class weighted equally
    balanced_accuracy  mean recall over classes present in y_true
    accuracy           overall accuracy, for completeness only
    macro_roc_auc      mean one-vs-rest ROC-AUC over classes that can be scored
    per_class          precision / recall / F1 / support / ROC-AUC per class
    confusion_matrix   rows = true class, columns = predicted class (all classes)

Classes absent from y_true (only possible in small subsets) are left out of the
macro averages instead of counting as F1 = 0. On the full validation and test
splits every class is present, though ABE has a single image in each, so its
per-class numbers are very noisy.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

# Reference categorical palette (fixed order) and a single-hue sequential map
SERIES = {"train": "#2a78d6", "validation": "#eb6834", "extra": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def compute_metrics(y_true, y_prob, class_names):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = y_prob.argmax(axis=1)
    k = len(class_names)
    all_labels = np.arange(k)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0)
    present = support > 0

    aucs = np.full(k, np.nan)
    for c in all_labels:
        pos = y_true == c
        if 0 < pos.sum() < len(y_true):
            aucs[c] = roc_auc_score(pos, y_prob[:, c])

    per_class = [
        {"class": class_names[c], "precision": float(prec[c]), "recall": float(rec[c]),
         "f1": float(f1[c]), "support": int(support[c]),
         "roc_auc": None if np.isnan(aucs[c]) else float(aucs[c])}
        for c in all_labels
    ]
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "balanced_accuracy": float(rec[present].mean()) if present.any() else 0.0,
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "macro_roc_auc": float(np.nanmean(aucs)) if np.isfinite(aucs).any() else None,
        "classes_present": int(present.sum()),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels).tolist(),
        "class_names": list(class_names),
    }


def save_metrics(metrics, out_dir):
    """metrics.json + per_class.csv + confusion_matrix.png into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out_dir / "per_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support", "roc_auc"])
        w.writeheader()
        w.writerows(metrics["per_class"])
    plot_confusion(np.array(metrics["confusion_matrix"]), metrics["class_names"],
                   out_dir / "confusion_matrix.png")


def summary_line(m):
    auc = f"{m['macro_roc_auc']:.4f}" if m.get("macro_roc_auc") is not None else "n/a"
    return (f"macro-F1 {m['macro_f1']:.4f} | balanced acc {m['balanced_accuracy']:.4f} | "
            f"acc {m['accuracy']:.4f} | macro ROC-AUC {auc}")


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_confusion(cm, class_names, path, title="Confusion matrix (row-normalized: recall per true class)"):
    """Row-normalized confusion matrix, single-hue sequential scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    k = len(class_names)
    fig, ax = plt.subplots(figsize=(0.5 * k + 3, 0.5 * k + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}"[1:] if norm[i, j] < 1 else "1", ha="center",
                        va="center", fontsize=6.5, color="white" if norm[i, j] > 0.55 else INK)
    labels = [f"{n} ({int(r)})" for n, r in zip(class_names, rows.ravel())]
    ax.set_xticks(range(k), class_names, rotation=90, fontsize=8, color=MUTED)
    ax.set_yticks(range(k), labels, fontsize=8, color=MUTED)
    ax.set_xlabel("Predicted class", color=INK)
    ax.set_ylabel("True class (support)", color=INK)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history, path):
    """Training curves: loss (train vs val) and validation scores, on two panels
    (never two y-scales on one axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in history], color=SERIES["train"], lw=2, marker="o", ms=4, label="train")
    a1.plot(ep, [h["val_loss"] for h in history], color=SERIES["validation"], lw=2, marker="o", ms=4, label="validation")
    a1.set_title("Loss", loc="left", fontsize=11, color=INK)
    a1.set_xlabel("Epoch", color=INK)
    a2.plot(ep, [h["val_macro_f1"] for h in history], color=SERIES["validation"], lw=2, marker="o", ms=4,
            label="val macro-F1")
    a2.plot(ep, [h["val_balanced_acc"] for h in history], color=SERIES["extra"], lw=2, marker="o", ms=4,
            label="val balanced acc")
    best = int(np.argmax([h["val_macro_f1"] for h in history]))
    # Ring the best epoch and label it below the line (balanced acc usually runs above)
    a2.plot(ep[best], history[best]["val_macro_f1"], "o", ms=10, mfc="none", mec=INK, mew=1.2)
    a2.annotate(f"best macro-F1 {history[best]['val_macro_f1']:.3f} (epoch {ep[best]})",
                (ep[best], history[best]["val_macro_f1"]), textcoords="offset points", xytext=(0, -16),
                ha="center", va="top", fontsize=8, color=INK)
    a2.set_title("Validation scores", loc="left", fontsize=11, color=INK)
    a2.set_xlabel("Epoch", color=INK)
    a2.set_ylim(0, 1)
    for ax in (a1, a2):
        _style(ax)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
