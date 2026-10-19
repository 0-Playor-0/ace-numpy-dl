"""Build data/shifted_val.npz and save 10x10 sample grids (one row per class) for clean and shifted val."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acenet.data import CLASS_NAMES, load, load_splits, to_unit
from acenet.shift import KINDS, SHIFTED_VAL_PATH, build_shifted_val

SEED = 0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def caption(kind, sev, mode):
    if KINDS[kind] == "rotate":
        return f"rot {sev:+.1f}°"
    if KINDS[kind] == "noise":
        return f"noise σ={sev:.2f}"
    return f"bright ×{1 + sev:.2f}" if mode == 0 else f"bright {sev:+.2f}"


def grid(X, y, rng, title, path, captions=None):
    fig, axes = plt.subplots(10, 10, figsize=(12, 14))
    for c in range(10):
        picks = rng.choice(np.flatnonzero(y == c), 10, replace=False)
        for j, i in enumerate(picks):
            ax = axes[c, j]
            ax.imshow(X[i], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            label = CLASS_NAMES[c] if j == 0 else ""
            if captions is not None:
                label = f"{label}\n{captions[i]}" if label else captions[i]
            ax.set_title(label, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    splits = load_splits(rng)
    X_val01 = to_unit(load()[0][splits["val_idx"]])
    y_val = splits["y_val"]
    X_sh, kind, sev, mode = build_shifted_val(X_val01, y_val, rng)
    print(f"wrote {SHIFTED_VAL_PATH}: {X_sh.shape}, kinds {np.bincount(kind, minlength=3)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    captions = [caption(k, s, m) for k, s, m in zip(kind, sev, mode)]
    grid(X_val01, y_val, rng, "Fashion-MNIST val (clean)", os.path.join(OUT_DIR, "val_clean.png"))
    grid(X_sh, y_val, rng, "Fashion-MNIST val (shifted, mixed kinds)",
         os.path.join(OUT_DIR, "val_shifted.png"), captions)
    print(f"wrote grids to {OUT_DIR}")


if __name__ == "__main__":
    main()
