"""Fashion-MNIST download, parsing, splitting, standardization and batching."""

import gzip
import os
import urllib.request

import numpy as np

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(ROOT, "raw")

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049
EXPECTED_COUNT = {"train": 60000, "test": 10000}


def _parse_idx(path):
    """Parse a gzipped idx file. Returns (magic, uint8 array)."""
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic = int.from_bytes(buf[0:4], "big")
    if magic == IMAGE_MAGIC:
        n, rows, cols = (int.from_bytes(buf[i:i + 4], "big") for i in (4, 8, 12))
        arr = np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    elif magic == LABEL_MAGIC:
        n = int.from_bytes(buf[4:8], "big")
        arr = np.frombuffer(buf, dtype=np.uint8, offset=8)
        if arr.shape[0] != n:
            raise ValueError(f"{path}: header says {n} labels, found {arr.shape[0]}")
    else:
        raise ValueError(f"{path}: bad magic number {magic}")
    return magic, arr


def _verify(key, path):
    magic, arr = _parse_idx(path)
    split = key.split("_")[0]
    n = EXPECTED_COUNT[split]
    if key.endswith("images"):
        if magic != IMAGE_MAGIC or arr.shape != (n, 28, 28):
            raise ValueError(f"{path}: expected magic {IMAGE_MAGIC}, shape {(n, 28, 28)}; "
                             f"got {magic}, {arr.shape}")
    else:
        if magic != LABEL_MAGIC or arr.shape != (n,):
            raise ValueError(f"{path}: expected magic {LABEL_MAGIC}, shape {(n,)}; "
                             f"got {magic}, {arr.shape}")
    return arr


def download(raw_dir=RAW_DIR):
    """Fetch the four Fashion-MNIST files into raw_dir (skipping existing ones) and verify them."""
    os.makedirs(raw_dir, exist_ok=True)
    for key, name in FILES.items():
        path = os.path.join(raw_dir, name)
        if not os.path.exists(path):
            tmp = path + ".part"
            urllib.request.urlretrieve(BASE_URL + name, tmp)
            os.replace(tmp, path)
        _verify(key, path)


def load(raw_dir=RAW_DIR):
    """Return (X_train, y_train, X_test, y_test) as uint8 arrays; images are (N, 28, 28)."""
    out = [_verify(key, os.path.join(raw_dir, FILES[key]))
           for key in ("train_images", "train_labels", "test_images", "test_labels")]
    return tuple(out)


def split_train_val(n, n_val, rng):
    """Seeded permutation of range(n) -> (train_idx, val_idx)."""
    perm = rng.permutation(n)
    return perm[n_val:], perm[:n_val]


def to_unit(X):
    """uint8 [0, 255] -> float32 [0, 1]."""
    return X.astype(np.float32) / np.float32(255.0)


def standardize(X01, mean, std, flatten=False):
    """[0, 1] images (N, 28, 28) or (N, 1, 28, 28) -> standardized float32 (N, 1, 28, 28) or (N, 784)."""
    X = ((X01 - np.float32(mean)) / np.float32(std)).astype(np.float32)
    X = X.reshape(X.shape[0], -1) if flatten else X.reshape(X.shape[0], 1, 28, 28)
    return X


def load_splits(rng, n_val=6000, flatten=False, raw_dir=RAW_DIR):
    """Download if needed, split train into train/val, standardize with train-only statistics.

    Returns a dict with X_train, y_train, X_val, y_val, X_test, y_test (float32 / int64),
    the indices into the original 60k train set, and the scalar mean/std used.
    """
    download(raw_dir)
    Xtr_u8, ytr, Xte_u8, yte = load(raw_dir)
    tr_idx, va_idx = split_train_val(Xtr_u8.shape[0], n_val, rng)
    X_train01 = to_unit(Xtr_u8[tr_idx])
    mean = float(X_train01.mean(dtype=np.float64))
    std = float(X_train01.std(dtype=np.float64))
    return {
        "X_train": standardize(X_train01, mean, std, flatten),
        "y_train": ytr[tr_idx].astype(np.int64),
        "X_val": standardize(to_unit(Xtr_u8[va_idx]), mean, std, flatten),
        "y_val": ytr[va_idx].astype(np.int64),
        "X_test": standardize(to_unit(Xte_u8), mean, std, flatten),
        "y_test": yte.astype(np.int64),
        "train_idx": tr_idx,
        "val_idx": va_idx,
        "mean": mean,
        "std": std,
    }


class DataLoader:
    """Iterable over (x, y) minibatches. A fresh permutation is drawn from rng each epoch when shuffling."""

    def __init__(self, X, y, batch_size, shuffle, rng=None, drop_last=False):
        if shuffle and rng is None:
            raise ValueError("shuffle=True needs an explicit np.random.Generator")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different lengths")
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng
        self.drop_last = drop_last

    def __len__(self):
        n = self.X.shape[0]
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = self.X.shape[0]
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            yield self.X[idx], self.y[idx]
