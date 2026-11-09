"""
Preprocessing for the bone marrow cell dataset.

Dataset: https://huggingface.co/datasets/ekim15/bone_marrow_cell_dataset

Training images only are randomly augmented first. A new draw is made every
time an image is read, so each epoch sees different versions. The strength is
chosen by name from AUG_PRESETS (the hyperparameter search tries all three):
    light    flips + random 90-degree turns (no interpolation, no colour change)
    medium   (default) flips (p = 0.5 each), rotation by a random angle in
             [-180, 180] degrees, colour jitter: brightness, contrast,
             saturation +/- 20%, hue +/- 0.02
    strong   medium + jitter +/- 30% (hue +/- 0.04), random zoom 0.9-1.1x and
             an occasional mild blur
Rare classes can get their own, stronger preset (`rare_aug`), which is the
"targeted augmentation" option for class imbalance.
Validation and test images are never augmented.

Then every image goes through:
    1. convert to RGB
    2. resize to image_size x image_size (224 by default; 128 is the fast option).
       Source images are 250 x 250, so there is no distortion.
    3. scale pixel values from [0, 255] to [0, 1]
    4. ImageNet normalization, per channel: (x - mean) / std

For training, data.py asks for output="uint8": the loader workers (loader.py) do steps 1-2
and the GPU does steps 3-4 (`normalize`), because passing float32 batches between
processes is 4x more data and was the bottleneck (~1,300 vs ~6,700 img/s).

The transforms run on the fly via `with_transform`, so nothing extra is written
to disk. Saving the normalized float32 arrays instead would take ~100 GB
(224 * 224 * 3 * 4 bytes per image, ~171k images) versus ~6.9 GB of JPEGs.

Each example becomes:
    pixel_values   float32 array, shape (3, S, S), channels first
    label          int class index (0-20, or fewer if classes are merged/dropped)

Use from a training script (data.py does this for you):
    from preprocess import load_preprocessed
    ds = load_preprocessed()
    x, y = ds["train"][0]["pixel_values"], ds["train"][0]["label"]

Install:
    pip install datasets pillow numpy

Run (downloads the dataset once, then prints sanity checks):
    python preprocess.py
"""

import random
import time
from functools import partial

import numpy as np
from datasets import DatasetDict, load_dataset
from PIL import Image, ImageEnhance, ImageFilter

REPO_ID = "ekim15/bone_marrow_cell_dataset"
IMAGE_SIZE = 224

# ImageNet channel statistics (RGB), defined for pixels already scaled to [0, 1]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Train-only augmentation strengths of the "medium" preset
# (same meaning as torchvision's ColorJitter)
ROTATION_DEGREES = 180  # cells have no natural orientation
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.02              # kept small: large hue shifts would change the stain colours

# Named augmentation strengths. "medium" is the original pipeline; keep it unchanged
# so results stay comparable across runs.
AUG_PRESETS = {
    "none": None,
    "light": dict(rot90=True, rotation=0, brightness=0.0, contrast=0.0,
                  saturation=0.0, hue=0.0, zoom=0.0, blur_p=0.0),
    "medium": dict(rot90=False, rotation=ROTATION_DEGREES, brightness=BRIGHTNESS,
                   contrast=CONTRAST, saturation=SATURATION, hue=HUE,
                   zoom=0.0, blur_p=0.0),
    "strong": dict(rot90=False, rotation=ROTATION_DEGREES, brightness=0.3,
                   contrast=0.3, saturation=0.3, hue=0.04, zoom=0.1, blur_p=0.2),
}


def _border_fill(img: Image.Image) -> tuple:
    """Average colour of the outermost pixels, used to fill corners exposed by
    rotation or zoom-out, so the model never learns from black corners that
    val/test images don't have."""
    arr = np.asarray(img)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return tuple(border.mean(axis=0).round().astype(int).tolist())


def augment(img: Image.Image, preset: str = "medium") -> Image.Image:
    """Random flips, rotation, colour jitter (and zoom/blur for "strong").
    For training images only.

    Uses Python's `random`, which loader.py seeds differently in every worker
    process, so parallel workers don't repeat the same augmentations.
    """
    img = img.convert("RGB")
    p = AUG_PRESETS[preset]
    if p is None:
        return img

    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if p["rot90"]:
        k = random.randrange(4)
        if k:
            img = img.transpose((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                                 Image.Transpose.ROTATE_270)[k - 1])

    fill = None
    if p["rotation"]:
        fill = _border_fill(img)
        img = img.rotate(random.uniform(-p["rotation"], p["rotation"]),
                         resample=Image.Resampling.BILINEAR, fillcolor=fill)

    if p["zoom"]:
        # Scale about the centre: output pixel (x, y) samples input ((x-c)/s + c)
        s = random.uniform(1 - p["zoom"], 1 + p["zoom"])
        cx, cy = img.width / 2, img.height / 2
        fill = fill or _border_fill(img)
        img = img.transform(img.size, Image.Transform.AFFINE,
                            (1 / s, 0, cx - cx / s, 0, 1 / s, cy - cy / s),
                            resample=Image.Resampling.BILINEAR, fillcolor=fill)

    if p["brightness"]:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1 - p["brightness"], 1 + p["brightness"]))
    if p["contrast"]:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1 - p["contrast"], 1 + p["contrast"]))
    if p["saturation"]:
        img = ImageEnhance.Color(img).enhance(random.uniform(1 - p["saturation"], 1 + p["saturation"]))

    # PIL stores hue as 0-255 around the colour wheel, so the shift wraps around
    if p["hue"]:
        shift = round(random.uniform(-p["hue"], p["hue"]) * 255)
        if shift:
            h, s_, v = img.convert("HSV").split()
            h = h.point(lambda q: (q + shift) % 256)
            img = Image.merge("HSV", (h, s_, v)).convert("RGB")

    if p["blur_p"] and random.random() < p["blur_p"]:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
    return img


def resize_uint8(img: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Steps 1-2: PIL image -> RGB, resized, uint8 array of shape (3, size, size)."""
    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))  # HWC -> CHW


def preprocess_image(img: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """PIL image -> normalized float32 array of shape (3, size, size)."""
    x = resize_uint8(img, size).astype(np.float32) / 255.0      # [0, 255] -> [0, 1]
    return (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]  # ImageNet normalization


_NORM_CACHE = {}


def normalize(x):
    """Steps 3-4 for a uint8 torch batch (N, 3, S, S) on any device -> float32,
    identical to preprocess_image. The loader sends uint8 images and the GPU
    does this, because float32 batches are 4x larger to pass between processes."""
    import torch  # imported lazily so preprocess.py itself only needs numpy/PIL

    key = x.device
    if key not in _NORM_CACHE:
        mean = torch.as_tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.as_tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        _NORM_CACHE[key] = (mean, std)
    mean, std = _NORM_CACHE[key]
    return (x.float() / 255.0 - mean) / std


def transform(batch, train=False, size=IMAGE_SIZE, aug="medium",
              rare_classes=frozenset(), rare_aug=None, label_lut=None, output="float"):
    """Batched transform for `Dataset.with_transform`.

    train=True adds augmentation: `aug` for most images, `rare_aug` (if given) for
    images whose original label is in `rare_classes`. `label_lut` (optional tuple)
    maps original label -> training label, which is how classes get merged.
    output="float" gives normalized float32 arrays (steps 1-4); output="uint8"
    stops after the resize (steps 1-2) and leaves steps 3-4 to `normalize`.

    Must stay a module-level function: Windows loader workers are started with
    "spawn" and need to pickle it.
    """
    labels = batch["label"]
    if train:
        images = [augment(img, rare_aug if (rare_aug and y in rare_classes) else aug)
                  for img, y in zip(batch["image"], labels)]
    else:
        images = batch["image"]
    if label_lut is not None:
        labels = [label_lut[y] for y in labels]
    to_array = resize_uint8 if output == "uint8" else preprocess_image
    return {
        "pixel_values": [to_array(img, size) for img in images],
        "label": labels,
    }


def with_preprocessing(d, train=False, image_size=IMAGE_SIZE, aug="medium",
                       rare_classes=(), rare_aug=None, label_lut=None, output="float"):
    """Attach the lazy transform to one split (a `datasets.Dataset`)."""
    if aug not in AUG_PRESETS or (rare_aug is not None and rare_aug not in AUG_PRESETS):
        raise ValueError(f"unknown aug preset; choose from {list(AUG_PRESETS)}")
    if output not in ("float", "uint8"):
        raise ValueError("output must be 'float' or 'uint8'")
    return d.with_transform(partial(
        transform, train=train, size=image_size, aug=aug,
        rare_classes=frozenset(rare_classes), rare_aug=rare_aug,
        label_lut=None if label_lut is None else tuple(int(v) for v in label_lut),
        output=output,
    ))


def load_preprocessed(image_size=IMAGE_SIZE, aug="medium", rare_classes=(),
                      rare_aug=None, label_lut=None, ds=None):
    """Load all splits, preprocessed lazily on access; only train is augmented."""
    ds = load_dataset(REPO_ID) if ds is None else ds
    return DatasetDict({
        split: with_preprocessing(d, train=(split == "train"), image_size=image_size,
                                  aug=aug, rare_classes=rare_classes,
                                  rare_aug=rare_aug, label_lut=label_lut)
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

    # Rough CPU cost per image of each preset (one process; loader workers
    # run this in parallel)
    raw = load_dataset(REPO_ID, split="validation").select(range(64))
    images = [r["image"] for r in raw]
    print("\nCPU time per image (augment + preprocess, 224 px):")
    for name in AUG_PRESETS:
        t = time.perf_counter()
        for img in images:
            preprocess_image(augment(img, name))
        print(f"  {name:<7} {(time.perf_counter() - t) / len(images) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
