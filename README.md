# Bone Marrow Cell Classification with CNNs

CAP 5610 Machine Learning (UCF, Fall 2026) project. It classifies single bone marrow cell images (250×250 RGB, May-Grünwald-Giemsa stain) into 21 cell types. The data is the [bone marrow cell dataset on Hugging Face](https://huggingface.co/datasets/ekim15/bone_marrow_cell_dataset), a copy of Matek et al., 2021. It has 137,093 train, 17,146 validation and 17,135 test images.

The class imbalance is severe: about 3,700:1 between the largest class (NGS) and the smallest (ABE, 6 training images). For that reason the primary metric is **macro-F1**.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Preprocess | `preprocess.py` | Resizes to 224 px (or 128), scales to [0, 1] and applies ImageNet normalization. Train-only augmentation comes in three presets: `light`, `medium` and `strong`. |
| Data | `data.py`, `loader.py` | Stratified subsets, class merge/drop, class weights, weighted sampler; a multiprocess batch loader whose workers never import torch |
| Models | `models.py` | Baseline CNN (from scratch), pretrained ResNet-50, EfficientNet-B0 fallback |
| Training | `engine.py`, `train.py` | Validates every epoch, early-stops on validation macro-F1, saves checkpoints |
| Tuning | `tune.py` | Optuna random search on 20% of train, then full runs for the top 3 configs |
| Evaluation | `evaluate.py`, `metrics.py` | macro-F1, balanced accuracy, per-class P/R/F1, ROC-AUC, confusion matrix |
| EDA | `class_distribution.py` | Per-class counts chart (`class_distribution.png` / `.csv`) |

Settings live in `configs/*.yaml` (see `config.py` for every field). Any field can be overridden with `--set key=value`.

## Setup

These steps use the existing `bmenv` virtual environment (Python 3.15). The RTX 50-series GPU needs the CUDA 13.0 PyTorch wheels.

```powershell
bmenv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
python -m pytest -q
```

The dataset (~6.9 GB) downloads to the Hugging Face cache on first use. The ImageNet weights (~100 MB) download on the first ResNet-50 run.

## Workflow

```powershell
# 0. Sanity checks: preprocessing, and a 1-epoch run on ~2k images
python preprocess.py
python train.py --config configs/resnet50.yaml --smoke

# 1. Baseline CNN, trained from scratch
python train.py --config configs/baseline.yaml

# 2. Fine-tuned ResNet-50
python train.py --config configs/resnet50.yaml --benchmark 50   # estimates epoch time first
python train.py --config configs/resnet50.yaml

# 3. Tuning: random search on 20% of train, then full runs of the top 3
python tune.py search --model resnet50 --trials 30
python tune.py top --study resnet50_search --k 3
python tune.py full --study resnet50_search --k 3

# 4. Final evaluation: run ONCE per final model on the test set
python evaluate.py --checkpoint runs/<best_run>/best.pt --split test
python evaluate.py --compare runs/<baseline_run> runs/<resnet_run> --split test
```

An interrupted run continues with `python train.py --resume runs/<run>`. An interrupted search continues by re-running the same `tune.py search` command, because the study is stored in `runs/tuning/<study>.db`.

### Outputs

Each training run writes `runs/<timestamp>_<name>/`:

| File | Contents |
|---|---|
| `config.yaml` | The exact settings of the run |
| `history.csv`, `curves.png` | Per-epoch loss, macro-F1, balanced accuracy, LR and speed |
| `best.pt` | Best epoch by validation macro-F1 (model, config, class names) |
| `last.pt` | Latest epoch with optimizer state, used for `--resume` |
| `val_metrics.json`, `val_best/` | Validation metrics of the best epoch, per-class CSV, confusion matrix |
| `eval_test/` | Written by `evaluate.py --split test` |

## Imbalance options

These are config fields, and the tuner searches over most of them:

- `loss`: `ce`, `weighted_ce`, `focal` (with `class_weight_scheme`: `inv_sqrt`, `inverse`, `effective_num`, `none`)
- `sampler: weighted` draws each class with probability ∝ count^-`sampler_power`
- `rare_aug: strong` applies stronger augmentation to classes with fewer than `rare_threshold` training images
- `merge: {ABE: EOS}` / `drop: [KSC]` merge or drop classes too small to learn. Report this if you use it.

## Timings (RTX 5070, 12 loader workers, measured)

| Run | Throughput | Time |
|---|---|---|
| Baseline CNN @ 224 px | ~1,470 img/s | ~2 min per epoch with validation; 30 epochs max ≈ 1 h |
| ResNet-50 @ 224 px | ~845 img/s (limited by the GPU, 3 GB memory) | ~3 min per epoch with validation; 20 epochs max ≈ 1 h |
| Tuning trial (20% subset, ≤ 8 epochs) | | ~4–6 min; 30 trials ≈ 2–3 h, less with pruning |

Early stopping usually ends runs sooner. Re-measure with `python train.py --config <cfg> --benchmark 80`.

## Troubleshooting (Windows)

- **Why there's a custom loader (`loader.py`) instead of torch's DataLoader.** On this setup (Windows, torch 2.14, Python 3.15), every DataLoader worker imports torch and its CUDA DLLs, about 1 GB of commit memory each. A full run died with `MemoryError` at epoch 6 once other programs took their share. Workers that returned torch tensors also crashed. The loader's workers import only numpy, PIL and `datasets` (~0.15 GB each) and return uint8 numpy batches. The GPU does the normalization (`preprocess.normalize`), and uint8 is 4× less data to pass between processes.
- **`MemoryError` or `[warn] training needs ~N GB of commit memory`.** RAM plus page file is nearly used up. Close other programs, lower `num_workers`, or enlarge the Windows page file (currently fixed at 16 GB).
- **Exit code 139 (segfault) after a run finishes.** This torch build crashes during Python shutdown once CUDA convolutions have run, after all outputs are written. The scripts avoid it by ending through `cli.run_and_exit`, so they exit with 0 on success and 1 on error. Your own scripts that train models may still show 139 at exit; their results are fine.
- **Entry scripts keep heavy imports inside functions.** On Windows every worker process re-runs the main script's top level, so a top-level `import torch` would cost ~1 GB per worker again. If that happens, the workers print a `[warn]`.

## Notes

- The split is not patient-level, so results may be slightly optimistic. State this in the report.
- One training image is corrupt *in the source dataset*: train row 83795 (MYB) can't be decoded, and the Hugging Face Parquet file has the same broken bytes. It is excluded from every run (`data.CORRUPT_IMAGES`), so training uses 137,092 images. Validation and test are clean. Re-check with `python check_images.py`.
- ABE has only 1 validation and 1 test image, so its per-class scores are very noisy.
