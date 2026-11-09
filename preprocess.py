"""
Preprocessing for the bone marrow cell dataset.

Dataset: https://huggingface.co/datasets/ekim15/bone_marrow_cell_dataset

Training images only are randomly augmented first. A new draw is made every
time an image is read, so each epoch sees different versions:
    - horizontal and vertical flips (p = 0.5 each)
    - rotation by a random angle in [-180, 180] degrees
    - colour jitter: brightness, contrast, saturation +/- 20%, hue +/- 0.02
Validation and test images are never augmented.

Then every image goes through:
    1. convert to RGB
    2. resize to 224 x 224 (source images are 250 x 250, so no distortion)
    3. scale pixel values from [0, 255] to [0, 1]
    4. ImageNet normalization, per channel: (x - mean) / std

The transforms run on the fly via `with_transform`, so nothing extra is written
to disk. Saving the normalized float32 arrays instead would take ~100 GB
(224 * 224 * 3 * 4 bytes per image, ~171k images) versus ~6.9 GB of JPEGs.

Each example becomes:
    pixel_values   float32 array, shape (3, 224, 224), channels first
    label          int class index (0-20)

Use from a training script:
    from preprocess import load_preprocessed
    ds = load_preprocessed()
    x, y = ds["train"][0]["pixel_values"], ds["train"][0]["label"]

Install:
    pip install datasets pillow numpy

Run (downloads the dataset once, then prints sanity checks):
    python preprocess.py
"""

import random
from functools import partial

import numpy as np
from datasets import DatasetDict, load_dataset
from PIL import Image, ImageEnhance

REPO_ID = "ekim15/bone_marrow_cell_dataset"cla
IMAGE_SIZE = 224

# ImageNet channel statistics (RGB), defined for pixels already scaled to [0, 1]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Train-only augmentation strengths (same meaning as torchvision's ColorJitter)
ROTATION_DEGREES = 180  # cells have no natural orientation
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.02              # kept small: large hue shifts would change the stain colours


def augment(img: Image.Image) -> Image.Image:
    """Random flips, rotation and colour jitter, for training images only.

    Uses Python's `random`, which PyTorch's DataLoader reseeds in every worker,
    so parallel workers don't repeat the same augmentations.
    """
    img = img.convert("RGB")
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Fill the corners exposed by rotation with the average border colour, so the
    # model never learns from black corners that val/test images don't have
    arr = np.asarray(img)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    fill = tuple(border.mean(axis=0).round().astype(int).tolist())
    img = img.rotate(random.uniform(-ROTATION_DEGREES, ROTATION_DEGREES),
                     resample=Image.Resampling.BILINEAR, fillcolor=fill)

    img = ImageEnhance.Brightness(img).enhance(random.uniform(1 - BRIGHTNESS, 1 + BRIGHTNESS))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(1 - CONTRAST, 1 + CONTRAST))
    img = ImageEnhance.Color(img).enhance(random.uniform(1 - SATURATION, 1 + SATURATION))

    # PIL stores hue as 0-255 around the colour wheel, so the shift wraps around
    shift = round(random.uniform(-HUE, HUE) * 255)
    if shift:
        h, s, v = img.convert("HSV").split()
        h = h.point(lambda p: (p + shift) % 256)
        img = Image.merge("HSV", (h, s, v)).convert("RGB")
    return img


def preprocess_image(img: Image.Image) -> np.ndarray:
    """PIL image -> normalized float32 array of shape (3, 224, 224)."""
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0  # [0, 255] -> [0, 1]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD          # ImageNet normalization
    return x.transpose(2, 0, 1)                     # HWC -> CHW


def transform(batch, train=False):
    """Batched transform for `Dataset.with_transform`; train=True adds augmentation."""
    images = [augment(img) for img in batch["image"]] if train else batch["image"]
    return {
        "pixel_values": [preprocess_image(img) for img in images],
        "label": batch["label"],
    }


def load_preprocessed():
    """Load all splits, preprocessed lazily on access; only train is augmented."""
    ds = load_dataset(REPO_ID)
    return DatasetDict({
        split: d.with_transform(partial(transform, train=(split == "train")))
        for split, d in ds.items()
    })


def main():
    print(f"Loading {REPO_ID} ...")
    ds = load_preprocessed()
    for split in ds:
        print(f"  {split:<10} {ds[split].num_rows:,} images")

    # The transform only runs on the rows we index, so this stays fast
    rng = np.random.default_rng(0)
    idx = rng.choice(ds["train"].num_rows, size=256, replace=False).tolist()
    x = np.stack(ds["train"][idx]["pixel_values"])

    lo, hi = -IMAGENET_MEAN / IMAGENET_STD, (1 - IMAGENET_MEAN) / IMAGENET_STD
    print(f"\nSample batch: shape {x.shape}, dtype {x.dtype}")
    print(f"Value range:  [{x.min():.3f}, {x.max():.3f}]  "
          f"(possible after normalization: [{lo.min():.3f}, {hi.max():.3f}])")
    print(f"Channel mean: {x.mean(axis=(0, 2, 3)).round(3)}")
    print(f"Channel std:  {x.std(axis=(0, 2, 3)).round(3)}")

    print("\nFirst image of each split, read twice "
          "(only train should differ, because of augmentation):")
    for split in ds:
        a, b = ds[split][0]["pixel_values"], ds[split][0]["pixel_values"]
        print(f"  {split:<10} {'same' if np.array_equal(a, b) else 'different'}")


if __name__ == "__main__":
    main()
