"""Fast unit tests for the pipeline. No dataset download, no GPU needed.

    python -m pytest -q
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from config import TrainConfig, load_config
from data import (CORRUPT_IMAGES, class_map, class_weights, collate, keep_indices, make_sampler,
                  stratified_indices)
from engine import FocalLoss, lr_factor
from metrics import compute_metrics
from models import build_model, count_params, freeze_bn_stats, param_groups
from preprocess import AUG_PRESETS, augment, normalize, preprocess_image, resize_uint8, transform


@pytest.fixture
def cell_image():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(250, 250, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# --------------------------------------------------------------------- preprocess

@pytest.mark.parametrize("preset", list(AUG_PRESETS))
@pytest.mark.parametrize("size", [128, 224])
def test_aug_presets_output_shape(cell_image, preset, size):
    x = preprocess_image(augment(cell_image, preset), size)
    assert x.shape == (3, size, size) and x.dtype == np.float32
    assert np.isfinite(x).all()


def test_no_aug_is_deterministic(cell_image):
    a = preprocess_image(augment(cell_image, "none"))
    b = preprocess_image(augment(cell_image, "none"))
    assert np.array_equal(a, b)


def test_device_normalize_matches_cpu_preprocessing(cell_image):
    ref = preprocess_image(cell_image, 128)
    u8 = resize_uint8(cell_image, 128)
    assert u8.dtype == np.uint8 and u8.shape == (3, 128, 128)
    got = normalize(torch.from_numpy(u8)[None])[0].numpy()
    assert np.allclose(got, ref, atol=1e-5)


def test_collate_returns_numpy_not_tensors():
    # Tensors from worker processes crash on torch 2.14 + Python 3.15 (Windows)
    rows = [{"pixel_values": np.zeros((3, 8, 8), np.uint8), "label": 3}] * 2
    x, y = collate(rows)
    assert isinstance(x, np.ndarray) and x.dtype == np.uint8 and x.shape == (2, 3, 8, 8)
    assert isinstance(y, np.ndarray) and y.tolist() == [3, 3]


def test_transform_rare_aug_and_label_lut(cell_image):
    batch = {"image": [cell_image, cell_image], "label": [0, 3]}
    out = transform(batch, train=True, size=64, aug="none", rare_classes=frozenset({0}),
                    rare_aug="strong", label_lut=(5, 6, 7, 8))
    assert out["label"] == [5, 8]
    plain = preprocess_image(cell_image, 64)
    assert np.array_equal(out["pixel_values"][1], plain)        # class 3: aug "none"
    assert not np.array_equal(out["pixel_values"][0], plain)    # class 0: rare -> strong


# --------------------------------------------------------------------------- data

def test_stratified_indices_keeps_rare_classes():
    labels = np.array([0] * 6 + [1] * 1000 + [2] * 40)
    idx = stratified_indices(labels, 0.2, seed=0, min_per_class=50)
    counts = np.bincount(labels[idx], minlength=3)
    assert counts.tolist() == [6, 200, 40]
    assert len(np.unique(idx)) == len(idx)


def test_class_map_merge_and_drop():
    names = ["ABE", "EOS", "KSC", "LYT"]
    lut, new = class_map(names, merge={"ABE": "EOS"}, drop=["KSC"])
    assert new == ["EOS", "LYT"]
    assert lut.tolist() == [0, 0, -1, 1]
    assert class_map(names) == (None, names)
    with pytest.raises(ValueError):
        class_map(names, merge={"ABE": "XXX"})


def test_keep_indices_excludes_corrupt_and_dropped():
    labels = np.array([0, 1, 2, 1, 0])
    assert keep_indices(labels) is None
    assert keep_indices(labels, exclude=(3,)).tolist() == [0, 1, 2, 4]
    lut = np.array([0, -1, 1])                       # class 1 dropped
    assert keep_indices(labels, lut, exclude=(0,)).tolist() == [2, 4]
    assert CORRUPT_IMAGES["train"] == (83795,)


def test_class_weights():
    counts = [4, 100, 10000]
    for scheme in ("inverse", "inv_sqrt", "effective_num", "none"):
        w = class_weights(counts, scheme)
        assert w.shape == (3,) and torch.isclose(w.mean(), torch.tensor(1.0))
    w = class_weights(counts, "inv_sqrt")
    assert w[0] > w[1] > w[2]


def test_sampler_oversamples_rare_class():
    labels = np.array([0] * 10 + [1] * 990)
    s = make_sampler(labels, 2, power=1.0, seed=0)
    drawn = labels[list(iter(s))]
    assert 0.35 < (drawn == 0).mean() < 0.65


# ------------------------------------------------------------------------- loader

def _tiny_dataset(n=10, size=40):
    from datasets import ClassLabel, Dataset, Features, Image as HFImage
    rng = np.random.default_rng(1)
    images = [Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)) for _ in range(n)]
    feats = Features({"image": HFImage(), "label": ClassLabel(names=["a", "b", "c"])})
    return Dataset.from_dict({"image": images, "label": [i % 3 for i in range(n)]}, features=feats)


def test_batch_loader_in_process():
    from loader import BatchLoader
    from preprocess import with_preprocessing
    d = with_preprocessing(_tiny_dataset(10), train=False, image_size=16, output="uint8")
    batches = list(BatchLoader(d, batch_size=4, num_workers=0))
    assert [len(y) for _, y in batches] == [4, 4, 2]
    assert batches[0][0].dtype == np.uint8 and batches[0][0].shape == (4, 3, 16, 16)
    loader = BatchLoader(d, batch_size=4, shuffle=True, drop_last=True, num_workers=0, seed=0)
    assert len(loader) == 2 and [len(y) for _, y in loader] == [4, 4]
    seen = np.concatenate([y for _, y in BatchLoader(d, 3, shuffle=True, num_workers=0)])
    assert sorted(seen.tolist()) == sorted([i % 3 for i in range(10)])


def test_batch_loader_workers_match_and_skip_torch():
    from loader import BatchLoader, _worker_has_torch
    from preprocess import with_preprocessing
    d = with_preprocessing(_tiny_dataset(10), train=False, image_size=16, output="uint8")
    ref = list(BatchLoader(d, batch_size=4, num_workers=0))
    loader = BatchLoader(d, batch_size=4, num_workers=2)
    try:
        for epoch in range(2):                          # pool is reused across epochs
            got = list(loader)
            assert len(got) == len(ref)
            for (x1, y1), (x2, y2) in zip(got, ref):
                assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
        assert loader._executor().submit(_worker_has_torch).result(timeout=60) is False
    finally:
        loader.close()


# ------------------------------------------------------------------------- models

@pytest.mark.parametrize("arch", ["baseline", "resnet50", "efficientnet_b0"])
@pytest.mark.parametrize("size", [128, 224])
def test_model_output_shape(arch, size):
    m = build_model(TrainConfig(model=arch, pretrained=False), 21).eval()
    with torch.no_grad():
        assert m(torch.zeros(2, 3, size, size)).shape == (2, 21)


def test_set_trainable_and_param_groups():
    cfg = TrainConfig(model="resnet50", pretrained=False, unfreeze="head")
    m = build_model(cfg, 21)
    _, trainable = count_params(m)
    assert trainable == 2048 * 21 + 21
    m.train()
    freeze_bn_stats(m)
    assert not m.bn1.training and m.fc.training

    cfg.unfreeze = "last_block"
    m = build_model(cfg, 21)
    assert all(p.requires_grad for p in m.layer4.parameters())
    assert not any(p.requires_grad for p in m.layer3.parameters())

    m = build_model(TrainConfig(model="resnet50", pretrained=False, unfreeze="all"), 21)
    groups = param_groups(m, "resnet50", lr=1e-3, weight_decay=1e-4, backbone_lr_mult=0.1, pretrained=True)
    lrs = {g["name"]: g["lr"] for g in groups}
    assert lrs["head"] == pytest.approx(1e-3) and lrs["backbone"] == pytest.approx(1e-4)
    assert {g["name"]: g["weight_decay"] for g in groups}["backbone_no_decay"] == 0.0
    n_params = sum(len(g["params"]) for g in groups)
    assert n_params == len(list(m.parameters()))


# ------------------------------------------------------------------- loss / sched

def test_focal_gamma_zero_equals_cross_entropy():
    torch.manual_seed(0)
    logits, y = torch.randn(32, 5), torch.randint(0, 5, (32,))
    w = torch.rand(5) + 0.5
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, y), F.cross_entropy(logits, y), atol=1e-6)
    assert torch.allclose(FocalLoss(w, gamma=0.0)(logits, y), F.cross_entropy(logits, y, weight=w), atol=1e-6)
    assert FocalLoss(gamma=2.0)(logits, y) < F.cross_entropy(logits, y)


def test_lr_schedule_shapes():
    cfg = TrainConfig(epochs=10, warmup_epochs=1, scheduler="cosine")
    f = [lr_factor(s, cfg, 100) for s in range(1000)]
    assert f[0] == pytest.approx(0.01) and f[99] == pytest.approx(1.0)
    assert f[-1] < 0.01 and all(a >= b for a, b in zip(f[100:], f[101:]))
    cfg = TrainConfig(epochs=9, warmup_epochs=0, scheduler="step")
    assert lr_factor(0, cfg, 10) == 1.0 and lr_factor(35, cfg, 10) == pytest.approx(0.1)


# ------------------------------------------------------------------------ metrics

def test_metrics_with_absent_class():
    names = ["a", "b", "c"]
    y = np.array([0, 0, 1, 1])                       # class "c" absent
    prob = np.array([[.9, .1, 0], [.2, .7, .1], [.1, .8, .1], [.1, .2, .7]])
    m = compute_metrics(y, prob, names)
    assert m["classes_present"] == 2
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert m["per_class"][2]["roc_auc"] is None
    assert np.array(m["confusion_matrix"]).sum() == 4


# ------------------------------------------------------------------------- config

def test_config_overrides_are_typed(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: resnet50\nlr: 1e-4\n")          # YAML alone reads 1e-4 as a string
    cfg = load_config(p, ["batch_size=32", "pretrained=false", "drop=KSC,FGC",
                          "merge={ABE: EOS}", "rare_aug=strong"])
    assert cfg.lr == pytest.approx(1e-4) and cfg.batch_size == 32 and cfg.pretrained is False
    assert cfg.drop == ["KSC", "FGC"] and cfg.merge == {"ABE": "EOS"} and cfg.rare_aug == "strong"
    with pytest.raises(ValueError):
        load_config(None, ["model=vgg"])
    with pytest.raises(KeyError):
        load_config(None, ["learning_rate=1"])
