"""
Decode every image of every split and list the ones that cannot be decoded.

A single corrupt JPEG crashes a loader worker and with it the whole training
run, so known-bad rows are excluded via bone_marrow.data.CORRUPT_IMAGES. This
script is how that list was produced; re-run it if the dataset revision changes.

Run (~20 s with 8 processes; reads the local Hugging Face cache):
    python tools/check_images.py
"""

import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# `python tools/check_images.py` puts tools/ on sys.path, not the repo root, so add
# the root to make the bone_marrow package importable (workers inherit sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Image as HFImage, load_dataset
from PIL import Image

from bone_marrow.preprocess import REPO_ID

SPLITS = ("train", "validation", "test")
CHUNK = 4000
_cache = {}


def raw_split(split):
    """Split with undecoded image bytes (decode=False is a metadata change, not a rewrite)."""
    if split not in _cache:
        _cache[split] = load_dataset(REPO_ID, split=split).cast_column("image", HFImage(decode=False))
    return _cache[split]


def scan(job):
    split, start, end = job
    rows = raw_split(split)[start:end]
    bad = []
    for i, (img, label) in enumerate(zip(rows["image"], rows["label"])):
        try:
            with Image.open(io.BytesIO(img["bytes"])) as im:
                im.load()
                problem = None if (im.size, im.mode) == ((250, 250), "RGB") else f"unexpected {im.size} {im.mode}"
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
        if problem:
            bad.append((start + i, label, problem))
    return split, bad


def main():
    from bone_marrow.data import CORRUPT_IMAGES   # imports torch; keep out of the workers

    t = time.time()
    jobs = [(s, a, min(a + CHUNK, raw_split(s).num_rows))
            for s in SPLITS for a in range(0, raw_split(s).num_rows, CHUNK)]
    found = {s: [] for s in SPLITS}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for split, bad in ex.map(scan, jobs):
            found[split] += bad

    for split in SPLITS:
        print(f"{split:<10} {len(found[split])} undecodable / unexpected images")
        for idx, label, problem in found[split]:
            print(f"    row {idx:>6}  label {label:>2}  {problem}")
    listed = {s: tuple(sorted(i for i, _, _ in found[s])) for s in SPLITS if found[s]}
    print(f"\nScanned in {time.time() - t:.0f}s. Found: {listed}")
    print("bone_marrow.data.CORRUPT_IMAGES is", "up to date." if listed == CORRUPT_IMAGES else
          f"OUT OF DATE (currently {CORRUPT_IMAGES}); update it to the dict above.")


if __name__ == "__main__":
    main()
