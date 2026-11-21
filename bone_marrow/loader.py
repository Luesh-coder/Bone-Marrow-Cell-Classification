"""
Multiprocess batch loader whose worker processes never import torch.

Why not torch's DataLoader? On this machine (Windows, torch 2.14, Python 3.15):
    - every DataLoader worker imports torch and its CUDA DLLs: ~1 GB of committed
      memory per worker (measured), so 8 train + 4 val workers needed ~12 GB and
      a full run died with MemoryError once other programs took their share;
    - workers that return torch tensors crash (access violation in torch's
      shared-memory pickling).
These workers import only numpy, PIL and `datasets` (~0.15 GB each, measured)
and return numpy arrays; engine.py makes tensors on the main process.

Keep this module free of torch imports. Entry scripts (scripts/*.py,
tools/check_images.py) keep heavy imports inside functions, because on
Windows every worker re-runs the top level of the main script ("spawn").

    loader = BatchLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    for x, y in loader:      # x: uint8 (N, 3, S, S), y: int64 (N,), numpy
        ...
    loader.close()
"""

import multiprocessing as mp
import os
import random
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RESULT_TIMEOUT_S = 600     # a batch takes well under a second; this only catches hangs
_DATASET = None            # set in each worker process by _init_worker


def collate(rows):
    """List of {"pixel_values": (3,S,S) uint8, "label": int} -> (x, y) numpy arrays."""
    x = np.stack([r["pixel_values"] for r in rows])
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    return x, y


def _init_worker(dataset, seed):
    global _DATASET
    _DATASET = dataset
    # Distinct augmentation streams per worker (pid differs per process)
    s = (seed * 1_000_003 + os.getpid()) % 2**32
    random.seed(s)
    np.random.seed(s)
    if "torch" in sys.modules:
        print("[warn] a loader worker imported torch (~1 GB extra per worker). Keep heavy "
              "imports inside main() of the entry script.", file=sys.stderr, flush=True)


def _fetch(indices):
    return collate(_DATASET.__getitems__(indices))


def _worker_has_torch():
    """For tests: True if torch got imported in a worker (it should not)."""
    return "torch" in sys.modules


class WeightedSampler:
    """Draws len(weights) indices with replacement, P(i) proportional to weights[i];
    a new draw every epoch (numpy stand-in for torch's WeightedRandomSampler)."""

    def __init__(self, weights, seed=0):
        w = np.asarray(weights, dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.p)

    def __iter__(self):
        return iter(self.rng.choice(len(self.p), size=len(self.p), replace=True, p=self.p).tolist())


class BatchLoader:
    def __init__(self, dataset, batch_size, shuffle=False, sampler=None, num_workers=8,
                 drop_last=False, seed=0, prefetch=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.seed = seed
        self.max_pending = max(num_workers * prefetch, 1)
        self.rng = np.random.default_rng(seed)
        self._pool = None

    def __len__(self):
        n = len(self.sampler) if self.sampler is not None else len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _batches(self):
        if self.sampler is not None:
            order = np.fromiter(iter(self.sampler), dtype=np.int64)
        elif self.shuffle:
            order = self.rng.permutation(len(self.dataset))
        else:
            order = np.arange(len(self.dataset))
        stop = len(self) * self.batch_size
        return [order[i:i + self.batch_size].tolist() for i in range(0, min(stop, len(order)), self.batch_size)]

    def _executor(self):
        if self._pool is None:   # started once, reused every epoch
            self._pool = ProcessPoolExecutor(
                max_workers=self.num_workers, mp_context=mp.get_context("spawn"),
                initializer=_init_worker, initargs=(self.dataset, self.seed))
        return self._pool

    def __iter__(self):
        batches = self._batches()
        if self.num_workers == 0:
            for b in batches:
                yield collate(self.dataset.__getitems__(b))
            return
        pool, it, pending = self._executor(), iter(batches), deque()
        try:
            for b in it:
                pending.append(pool.submit(_fetch, b))
                if len(pending) >= self.max_pending:
                    break
            while pending:
                result = pending.popleft().result(timeout=RESULT_TIMEOUT_S)
                nxt = next(it, None)
                if nxt is not None:
                    pending.append(pool.submit(_fetch, nxt))
                yield result
        finally:
            for f in pending:      # epoch abandoned early (benchmark, error): drop queued work
                f.cancel()

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
