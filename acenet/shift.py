"""Synthetic distribution shift on [0, 1] images: rotation, Gaussian noise, brightness.

Shifts are applied to [0, 1] pixels before standardization. Results are clipped to [0, 1].
"""

import os

import numpy as np

from .data import ROOT, standardize

KINDS = ("rotate", "noise", "brightness")
SHIFTED_VAL_PATH = os.path.join(ROOT, "shifted_val.npz")

# Parameter ranges for the mixed shifted-val set.
ROTATE_MAX_DEG = 15.0
NOISE_SIGMA = (0.1, 0.3)
BRIGHTNESS_MAX = 0.3      # multiplicative: pixel * (1 + s), s in [-0.3, 0.3]
BRIGHTNESS_ADD_MAX = 0.2  # additive: pixel + c, c in [-0.2, 0.2]
BRIGHTNESS_MODES = ("mul", "add")


def rotate(X, degrees):
    """Rotate each (H, W) image counter-clockwise by degrees[i] about its centre.

    Inverse mapping with bilinear interpolation; source pixels outside the image read as 0.
    X: (N, H, W) float. degrees: scalar or (N,).
    """
    N, H, W = X.shape
    theta = np.deg2rad(np.broadcast_to(np.asarray(degrees, dtype=np.float64), (N,)))
    cos = np.cos(theta)[:, None, None]
    sin = np.sin(theta)[:, None, None]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y, x = r - cy, c - cx
    # Output (y, x) samples the source at R(-theta) (y, x); rows grow downward.
    src_x = cos * x - sin * y + cx
    src_y = sin * x + cos * y + cy

    r0 = np.floor(src_y).astype(np.int64)
    c0 = np.floor(src_x).astype(np.int64)
    wy = (src_y - r0).astype(X.dtype)
    wx = (src_x - c0).astype(X.dtype)
    flat = X.reshape(N, H * W)

    def tap(rr, cc):
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        idx = (np.clip(rr, 0, H - 1) * W + np.clip(cc, 0, W - 1)).reshape(N, -1)
        return np.take_along_axis(flat, idx, axis=1).reshape(N, H, W) * valid

    out = ((1 - wy) * (1 - wx) * tap(r0, c0) + (1 - wy) * wx * tap(r0, c0 + 1)
           + wy * (1 - wx) * tap(r0 + 1, c0) + wy * wx * tap(r0 + 1, c0 + 1))
    return out.astype(X.dtype)


def make_shifted(X, kind, severity, rng, mode=0):
    """Apply one shift kind to [0, 1] images X of shape (N, 28, 28) or (N, 1, 28, 28).

    severity (scalar or per-sample (N,)) is the shift parameter itself:
      rotate     -> angle in degrees (signed)
      noise      -> Gaussian sigma on the [0, 1] scale
      brightness -> mode 0 ("mul"): pixel * (1 + severity); mode 1 ("add"): pixel + severity
    mode (scalar or per-sample (N,)) only applies to brightness. rng supplies the noise draws. Returns float32 in [0, 1] with the input's shape.
    """
    shape = X.shape
    X = np.asarray(X, dtype=np.float32).reshape(shape[0], 28, 28)
    sev = np.broadcast_to(np.asarray(severity, dtype=np.float32), (shape[0],))[:, None, None]
    if kind == "rotate":
        out = rotate(X, sev[:, 0, 0])
    elif kind == "noise":
        out = X + sev * rng.standard_normal(X.shape, dtype=np.float32)
    elif kind == "brightness":
        add = np.broadcast_to(np.asarray(mode), (shape[0],))[:, None, None] == 1
        out = np.where(add, X + sev, X * (1 + sev))
    else:
        raise ValueError(f"unknown shift kind {kind!r}; expected one of {KINDS}")
    return np.clip(out, 0.0, 1.0).astype(np.float32).reshape(shape)


def sample_severity(kind, n, rng):
    """Draw n per-sample parameters from the shifted-val ranges for one kind.

    Returns (severity, mode); mode is the brightness mode (0 mul / 1 add, 50/50), else -1.
    """
    mode = np.full(n, -1, dtype=np.int8)
    if kind == "rotate":
        return rng.uniform(-ROTATE_MAX_DEG, ROTATE_MAX_DEG, n), mode
    if kind == "noise":
        return rng.uniform(*NOISE_SIGMA, n), mode
    if kind == "brightness":
        mode = rng.integers(0, 2, n).astype(np.int8)
        sev = np.where(mode == 1, rng.uniform(-BRIGHTNESS_ADD_MAX, BRIGHTNESS_ADD_MAX, n),
                       rng.uniform(-BRIGHTNESS_MAX, BRIGHTNESS_MAX, n))
        return sev, mode
    raise ValueError(f"unknown shift kind {kind!r}")


def build_shifted_val(X_val01, y_val, rng, path=SHIFTED_VAL_PATH):
    """Mixed-kind shifted copy of the [0, 1] validation images, saved to path.

    Each sample gets one kind uniformly at random and a parameter from that kind's range.
    Saves X (float32 [0, 1], (N, 28, 28)), y, kind (index into KINDS), severity, and mode
    (brightness mode index into BRIGHTNESS_MODES; -1 for other kinds).
    """
    N = X_val01.shape[0]
    X_val01 = X_val01.reshape(N, 28, 28)
    kind = rng.integers(0, len(KINDS), N)
    severity = np.zeros(N, dtype=np.float32)
    mode = np.full(N, -1, dtype=np.int8)
    X = np.empty((N, 28, 28), dtype=np.float32)
    for k, name in enumerate(KINDS):
        m = kind == k
        severity[m], mode[m] = sample_severity(name, int(m.sum()), rng)
        X[m] = make_shifted(X_val01[m], name, severity[m], rng, mode[m])
    np.savez_compressed(path, X=X, y=y_val, kind=kind, severity=severity, mode=mode,
                        kinds=np.array(KINDS), brightness_modes=np.array(BRIGHTNESS_MODES))
    return X, kind, severity, mode


def load_shifted_val(mean, std, flatten=False, path=SHIFTED_VAL_PATH):
    """Load shifted val and standardize it with the train-split statistics."""
    d = np.load(path)
    return (standardize(d["X"], mean, std, flatten), d["y"].astype(np.int64),
            d["kind"], d["severity"], d["mode"])
