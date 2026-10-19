import numpy as np
import pytest

from acenet.data import DataLoader, download, load, load_splits
from acenet.shift import make_shifted, rotate


@pytest.fixture(scope="module")
def splits():
    return load_splits(np.random.default_rng(0))


def test_raw_shapes_and_dtypes():
    download()
    Xtr, ytr, Xte, yte = load()
    assert Xtr.shape == (60000, 28, 28) and Xte.shape == (10000, 28, 28)
    assert ytr.shape == (60000,) and yte.shape == (10000,)
    assert all(a.dtype == np.uint8 for a in (Xtr, ytr, Xte, yte))


def test_split_shapes_and_dtypes(splits):
    assert splits["X_train"].shape == (54000, 1, 28, 28)
    assert splits["X_val"].shape == (6000, 1, 28, 28)
    assert splits["X_test"].shape == (10000, 1, 28, 28)
    for k in ("X_train", "X_val", "X_test"):
        assert splits[k].dtype == np.float32
    for k in ("y_train", "y_val", "y_test"):
        assert splits[k].dtype == np.int64
        assert splits[k].min() == 0 and splits[k].max() == 9


def test_flatten_option():
    s = load_splits(np.random.default_rng(0), flatten=True)
    assert s["X_train"].shape == (54000, 784) and s["X_val"].shape == (6000, 784)


def test_val_class_balance(splits):
    counts = np.bincount(splits["y_val"], minlength=10)
    assert counts.min() >= 500 and counts.max() <= 700, counts


def test_no_train_val_overlap(splits):
    tr, va = splits["train_idx"], splits["val_idx"]
    assert len(np.intersect1d(tr, va)) == 0
    assert len(np.union1d(tr, va)) == 60000


def test_split_is_seeded():
    a = load_splits(np.random.default_rng(0))["val_idx"]
    b = load_splits(np.random.default_rng(0))["val_idx"]
    c = load_splits(np.random.default_rng(1))["val_idx"]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_train_standardized(splits):
    X = splits["X_train"].astype(np.float64)
    assert abs(X.mean()) < 1e-4
    assert abs(X.std() - 1) < 1e-4


def test_dataloader():
    X = np.arange(10, dtype=np.float32)[:, None]
    y = np.arange(10)
    dl = DataLoader(X, y, 4, shuffle=False)
    assert [len(b) for b, _ in dl] == [4, 4, 2] and len(dl) == 3
    dl = DataLoader(X, y, 4, shuffle=True, rng=np.random.default_rng(0), drop_last=True)
    batches = list(dl)
    assert len(batches) == len(dl) == 2
    seen = np.concatenate([b[:, 0] for b, _ in batches])
    assert len(np.unique(seen)) == 8
    for xb, yb in batches:
        assert np.array_equal(xb[:, 0], yb)


def test_rotate_matches_rot90():
    img = np.random.default_rng(0).random((2, 28, 28)).astype(np.float32)
    assert np.allclose(rotate(img, 0.0), img)
    assert np.allclose(rotate(img, 90.0), np.rot90(img, k=1, axes=(1, 2)), atol=1e-5)


def test_shifts_stay_in_unit_range():
    rng = np.random.default_rng(0)
    X = rng.random((5, 1, 28, 28)).astype(np.float32)
    for kind, sev in (("rotate", 15.0), ("noise", 0.3), ("brightness", 0.3)):
        out = make_shifted(X, kind, sev, rng)
        assert out.shape == X.shape and out.dtype == np.float32
        assert out.min() >= 0 and out.max() <= 1


def test_brightness_modes():
    X = np.full((2, 28, 28), 0.5, dtype=np.float32)
    out = make_shifted(X, "brightness", [0.2, 0.2], np.random.default_rng(0), mode=[0, 1])
    assert np.allclose(out[0], 0.6) and np.allclose(out[1], 0.7)
