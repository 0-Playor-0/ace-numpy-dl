"""Evaluate every declared model (float + int8 where applicable, flip TTA) on a hidden Fashion-MNIST-style set.

Models (pre-registered: E2/B/D in commit 693f9c2, S0/S1 int8 in commit ba2ccb9):
  E2, B, D   float32 checkpoint (softmax-mean TTA) and int8 npz (int32 logit-sum TTA)
  S0, S1     int8 compact .ace8 student files (logit-sum TTA)

Input formats (auto-detected):
  .npz          image array under X/x/images/x_test/X_test/data (else the largest >=3-D array),
                optional labels under y/labels/y_test/Y
  idx (.gz ok)  an idx3 image file, or a folder holding one; labels from a sibling idx1 file
  PNG folder    *.png anywhere below the folder; labels from the parent folder name (digit or
                class name), a labels.csv (filename,label), or a trailing _<digit> in the file name
Images of any size / channel count are converted to 28x28 grayscale in [0, 1] (every conversion is
logged): RGB(A) -> luminance (alpha premultiplied), area resize when shrinking, bilinear when
growing, and polarity inversion when the border is bright (FMNIST is light-on-black).
Inputs are standardized with the train-split mean/std from runs/final_model.json (shared by all models).

Usage:
  python -m experiments.hidden_eval PATH [--labels PATH] [--name NAME] [--invert auto|yes|no]
  python -m experiments.hidden_eval --selftest   # shifted_val saved as npz / idx.gz / PNGs
Writes runs/hidden_eval_<name>.json, _per_class.md, _predictions.csv, _confusion.png and _vs_test.png
(clean test vs hidden, per model).
"""

import argparse
import csv
import glob
import gzip
import json
import os
import re
import struct
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs")
FINAL = os.path.join(RUNS_DIR, "final_model.json")
TEST_ONCE = os.path.join(RUNS_DIR, "test_once.json")
TEACHERS = ("E2", "B", "D")
STUDENTS = ("S0", "S1")
SELFTEST_DIR = os.path.join(ROOT, "data", "hidden_selftest")
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
IDX_DTYPES = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4", 0x0D: ">f4", 0x0E: ">f8"}
LOG = []


def log(msg):
    LOG.append(msg)
    print("[hidden_eval]", msg)


# ---------------------------------------------------------------- PNG (stdlib zlib only)

def png_read(path):
    """Decode a non-interlaced PNG (bit depth 8/16; gray, RGB, palette, gray+alpha, RGBA) to an
    (H, W, C) uint8/uint16 array. Other variants fall back to matplotlib.image.imread."""
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos, idat, plte, trns = 8, [], None, None
    while pos < len(buf):
        n, typ = struct.unpack(">I4s", buf[pos:pos + 8])
        data = buf[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", data)
        elif typ == b"IDAT":
            idat.append(data)
        elif typ == b"PLTE":
            plte = np.frombuffer(data, np.uint8).reshape(-1, 3)
        elif typ == b"tRNS":
            trns = np.frombuffer(data, np.uint8)
        elif typ == b"IEND":
            break
        pos += 12 + n
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    if interlace or depth not in (8, 16) or (ctype == 3 and depth != 8):
        import matplotlib.image as mpimg
        img = mpimg.imread(path)
        return np.round(np.atleast_3d(img) * 255).astype(np.uint8)
    bpp = channels * depth // 8
    stride = w * bpp
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, stride + 1)
    out = np.zeros((h, stride), np.uint8)
    prev = np.zeros(stride, np.int32)
    for r in range(h):
        ft, line = raw[r, 0], raw[r, 1:].astype(np.int32)
        if ft == 0:
            cur = line
        elif ft == 2:
            cur = (line + prev) & 255
        elif ft == 1:  # Sub: running sum per byte lane, mod 256
            cur = np.cumsum(line.reshape(-1, bpp), axis=0).reshape(-1) & 255
        else:          # Average / Paeth depend on the reconstructed left byte: sequential
            cur = line.copy()
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                if ft == 3:
                    cur[i] = (cur[i] + ((a + b) >> 1)) & 255
                else:
                    c = prev[i - bpp] if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    cur[i] = (cur[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        out[r] = cur
        prev = cur
    if depth == 16:
        img = out.view(">u2").reshape(h, w, channels).astype(np.uint16)
    else:
        img = out.reshape(h, w, channels)
    if ctype == 3:
        idx = img[..., 0]
        rgb = plte[idx]
        if trns is not None:
            alpha = np.full(len(plte), 255, np.uint8)
            alpha[:len(trns)] = trns
            return np.concatenate([rgb, alpha[idx][..., None]], axis=2)
        return rgb
    return img


def png_write(path, img):
    """Write an (H, W) or (H, W, C) uint8 array (C in 1, 3, 4) as an 8-bit PNG (filter 0)."""
    img = np.atleast_3d(np.asarray(img, np.uint8))
    h, w, c = img.shape
    ctype = {1: 0, 3: 2, 4: 6}[c]
    rows = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, w * c)], axis=1)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- idx

def idx_read(path):
    opener = gzip.open if open(path, "rb").read(2) == b"\x1f\x8b" else open
    with opener(path, "rb") as f:
        buf = f.read()
    if buf[0] != 0 or buf[1] != 0 or buf[2] not in IDX_DTYPES:
        raise ValueError(f"{path}: not an idx file")
    nd = buf[3]
    dims = struct.unpack(">" + "I" * nd, buf[4:4 + 4 * nd])
    return np.frombuffer(buf, IDX_DTYPES[buf[2]], offset=4 + 4 * nd).reshape(dims)


def idx_write(path, arr):
    arr = np.ascontiguousarray(arr, np.uint8)
    head = bytes([0, 0, 0x08, arr.ndim]) + struct.pack(">" + "I" * arr.ndim, *arr.shape)
    with gzip.open(path, "wb") as f:
        f.write(head + arr.tobytes())


def idx_label_partner(img_path):
    d, name = os.path.split(img_path)
    cands = {name}
    for a, b in (("idx3", "idx1"), ("images", "labels"), ("image", "label")):
        cands |= {c.replace(a, b) for c in cands}
    cands.discard(name)
    for c in sorted(cands, key=len):
        cand = os.path.join(d, c)
        if os.path.exists(cand):
            try:
                if idx_read(cand).ndim == 1:
                    return cand
            except (ValueError, OSError, IndexError, struct.error):
                continue
    return None


# ---------------------------------------------------------------- format detection and loading

def detect(path):
    if os.path.isdir(path):
        if glob.glob(os.path.join(path, "**", "*.png"), recursive=True):
            return "png"
        for f in sorted(os.listdir(path)):
            fp = os.path.join(path, f)
            if f.endswith(".npz"):
                return "npz"
            try:
                if idx_read(fp).ndim >= 3:
                    return "idx"
            except (ValueError, OSError, IndexError, struct.error):
                continue
        raise ValueError(f"{path}: no .npz, idx or .png data found")
    if path.endswith(".npz"):
        return "npz"
    if path.lower().endswith(".png"):
        return "png"
    idx_read(path)  # raises if not idx
    return "idx"


def label_from_name(s):
    s = s.strip().lower()
    if re.fullmatch(r"\d", s):
        return int(s)
    names = [n.lower() for n in CLASS_NAMES]
    for i, n in enumerate(names):
        if s in (n, n.replace("/", "_"), n.replace("/", "-"), n.replace(" ", "_"), n.split("/")[0]):
            return i
    return None


def load_npz(path):
    if os.path.isdir(path):
        path = os.path.join(path, sorted(f for f in os.listdir(path) if f.endswith(".npz"))[0])
    with np.load(path, allow_pickle=False) as d:
        keys = d.files
        xk = next((k for k in ("X", "x", "images", "x_test", "X_test", "data") if k in keys), None)
        if xk is None:
            xk = max((k for k in keys if d[k].ndim >= 2 and d[k].shape[0] > 1), key=lambda k: d[k].size)
        yk = next((k for k in ("y", "labels", "y_test", "Y", "label") if k in keys), None)
        X = d[xk]
        y = d[yk].astype(np.int64).reshape(-1) if yk else None
    log(f"npz {os.path.basename(path)}: images '{xk}' {X.shape} {X.dtype}; labels {repr(yk)}")
    return X, y


def load_idx(path):
    if os.path.isdir(path):
        path = next(os.path.join(path, f) for f in sorted(os.listdir(path))
                    if not f.startswith(".") and _is_idx_images(os.path.join(path, f)))
    X = idx_read(path)
    lp = idx_label_partner(path)
    y = idx_read(lp).astype(np.int64) if lp else None
    log(f"idx {os.path.basename(path)}: {X.shape} {X.dtype}; labels {os.path.basename(lp) if lp else None}")
    return X, y


def _is_idx_images(p):
    try:
        return idx_read(p).ndim >= 3
    except (ValueError, OSError, IndexError, struct.error):
        return False


def load_png(path):
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "**", "*.png"), recursive=True))
    csv_labels = {}
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "labels.csv")):
        with open(os.path.join(path, "labels.csv")) as f:
            for row in csv.reader(f):
                if len(row) >= 2 and label_from_name(row[1]) is not None:
                    csv_labels[os.path.basename(row[0])] = label_from_name(row[1])
    imgs, labels, sources = [], [], set()
    for fp in files:
        img = png_read(fp)
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535
        imgs.append(img)
        base = os.path.basename(fp)
        lab, src = csv_labels.get(base), "labels.csv"
        if lab is None and os.path.dirname(fp) != os.path.normpath(path):
            lab, src = label_from_name(os.path.basename(os.path.dirname(fp))), "folder"
        if lab is None:
            m = re.search(r"[_-](\d)\.png$", base, re.IGNORECASE)
            lab, src = (int(m.group(1)), "filename") if m else (None, None)
        labels.append(-1 if lab is None else lab)
        sources.add(src)
    shapes = {im.shape for im in imgs}
    log(f"png: {len(files)} files, shapes {sorted(shapes)[:5]}{' ...' if len(shapes) > 5 else ''}, "
        f"label source {sorted(s for s in sources if s) or None}")
    y = np.array(labels, np.int64)
    if (y < 0).any():
        log(f"{int((y < 0).sum())} PNGs without a label: evaluating predictions only")
        y = None
    return imgs, y


def resize_matrix(n_in, n_out):
    """(n_out, n_in) interpolation matrix: area averaging when shrinking, bilinear when growing."""
    R = np.zeros((n_out, n_in))
    if n_out <= n_in:
        s = n_in / n_out
        for i in range(n_out):
            lo, hi = i * s, (i + 1) * s
            j = np.arange(int(np.floor(lo)), int(np.ceil(hi)))
            R[i, j] = np.minimum(j + 1, hi) - np.maximum(j, lo)
        return R / s
    x = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
    x0 = np.clip(np.floor(x).astype(int), 0, n_in - 1)
    x1 = np.clip(x0 + 1, 0, n_in - 1)
    w = np.clip(x - x0, 0, 1)
    R[np.arange(n_out), x0] += 1 - w
    R[np.arange(n_out), x1] += w
    return R


def to_unit_gray28(imgs, invert="auto"):
    """List or array of images -> float32 (N, 28, 28) in [0, 1], logging every conversion."""
    if isinstance(imgs, np.ndarray):
        X = imgs
        if X.ndim == 2 and X.shape[1] == 784:
            X = X.reshape(-1, 28, 28)
            log("flat 784 vectors reshaped to 28x28")
        if X.ndim == 4 and X.shape[1] in (1, 3, 4) and X.shape[-1] not in (1, 3, 4):
            X = X.transpose(0, 2, 3, 1)
            log("NCHW -> NHWC")
        imgs = list(X) if X.ndim == 4 else list(X[..., None])
    out, notes = [], {}
    for im in imgs:
        im = np.atleast_3d(im)
        if im.dtype == np.uint8:
            im = im.astype(np.float32) / 255
        else:
            im = im.astype(np.float32)
            if im.max() > 1.0 + 1e-6:
                im = im / 255
                notes["float values > 1 treated as 0-255"] = notes.get("float values > 1 treated as 0-255", 0) + 1
        c = im.shape[2]
        if c in (2, 4):
            im = im[..., :c - 1] * im[..., c - 1:]
            notes["alpha premultiplied (composited on black)"] = notes.get("alpha premultiplied (composited on black)", 0) + 1
        if im.shape[2] == 3:
            im = im @ np.array([0.299, 0.587, 0.114], np.float32)
            notes["RGB -> grayscale (BT.601 luma)"] = notes.get("RGB -> grayscale (BT.601 luma)", 0) + 1
        else:
            im = im[..., 0]
        if im.shape != (28, 28):
            key = f"resized {im.shape[0]}x{im.shape[1]} -> 28x28 ({'area' if min(im.shape) >= 28 else 'bilinear'})"
            notes[key] = notes.get(key, 0) + 1
            im = (resize_matrix(im.shape[0], 28) @ im @ resize_matrix(im.shape[1], 28).T).astype(np.float32)
        out.append(im)
    X = np.clip(np.stack(out), 0, 1).astype(np.float32)
    for k, v in notes.items():
        log(f"{k}: {v} images")
    border = np.concatenate([X[:, 0], X[:, -1], X[:, :, 0], X[:, :, -1]], axis=1).mean()
    if invert == "yes" or (invert == "auto" and border > 0.5):
        X = 1 - X
        log(f"inverted polarity (mean border {border:.3f}; FMNIST backgrounds are ~0)")
    else:
        log(f"no inversion (mean border {border:.3f})")
    return X


def load_any(path, labels=None, invert="auto"):
    fmt = detect(path)
    log(f"detected format: {fmt} ({path})")
    imgs, y = {"npz": load_npz, "idx": load_idx, "png": load_png}[fmt](path)
    if labels is not None:
        y = (load_npz(labels)[1] if labels.endswith(".npz") else idx_read(labels)).astype(np.int64)
        log(f"labels from {labels}")
    return fmt, to_unit_gray28(imgs, invert), y


# ---------------------------------------------------------------- evaluation

def model_specs():
    """[(key, kind, loader, clean test acc)] for the declared models, in report order."""
    from acenet.quant import Int8Model
    from experiments.cnn_ablation import entry, load_model
    final = json.load(open(FINAL))
    tests = json.load(open(TEST_ONCE))
    test = {"B": final["test_float_tta"], "B int8": final["test_int8_tta"],
            **{r: tests[f"{r}_float"]["test_acc"] for r in ("E2", "D")},
            **{f"{r} int8": tests[f"{r}_int8"]["test_acc"] for r in ("E2", "D")},
            **{f"{s} int8": tests[f"student_{s}_int8"]["test_acc"] for s in STUDENTS}}
    specs = []
    for rid in TEACHERS:
        specs.append((rid, "float", lambda rid=rid: load_model(entry(rid)), test[rid]))
        specs.append((f"{rid} int8", "int8", lambda rid=rid: Int8Model.load(
            os.path.join(RUNS_DIR, f"cnn_{rid.lower()}_int8.npz")), test[f"{rid} int8"]))
    for sid in STUDENTS:
        specs.append((f"{sid} int8", "int8", lambda sid=sid: Int8Model.load_compact(
            os.path.join(RUNS_DIR, f"student_{sid}_int8.ace8")), test[f"{sid} int8"]))
    return specs


def load_models():
    cfg = json.load(open(FINAL))
    return cfg, [(k, kind, fn(), t) for k, kind, fn, t in model_specs()]


def run_models(X01, cfg, models):
    """{model key: TTA predictions}, {model key: seconds}."""
    from experiments.cnn_ablation import predict
    X = ((X01 - np.float32(cfg["mean"])) / np.float32(cfg["std"])).astype(np.float32)[:, None]
    preds, timing = {}, {}
    for key, kind, m, _ in models:
        t0 = time.perf_counter()
        preds[key] = predict(m, X)[1].argmax(1) if kind == "float" else m.logits(X, flip=True).argmax(1)
        timing[key] = time.perf_counter() - t0
    return preds, timing


def report(pred, y):
    from experiments.cnn_ablation import confusion
    cm = confusion(pred, y)
    return {"acc": float((pred == y).mean()),
            "per_class": {CLASS_NAMES[c]: float(cm[c, c] / max(1, cm[c].sum())) for c in range(10)},
            "confusion": cm.tolist()}


def per_class_table(res):
    keys = list(res["models"])
    lines = ["| class | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for c in CLASS_NAMES:
        lines.append(f"| {c} | " + " | ".join(f"{res['models'][k]['per_class'][c]:.3f}" for k in keys) + " |")
    lines.append("| **all** | " + " | ".join(f"**{res['models'][k]['acc']:.4f}**" for k in keys) + " |")
    lines.append("| clean test | " + " | ".join("–" if res['models'][k]['clean_test'] is None else
                                               f"{res['models'][k]['clean_test']:.4f}" for k in keys) + " |")
    return "\n".join(lines)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(cms, path, title):
    plt = _plt()
    n = len(cms)
    cols = min(n, 4)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.8 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (name, cm) in zip(axes.flat, cms.items()):
        cm = np.array(cm)
        norm = cm / np.maximum(1, cm.sum(1, keepdims=True))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(10), CLASS_NAMES, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name} (TTA)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_vs_test(res, path, name):
    """Grouped bars: clean test accuracy vs hidden-set accuracy for every model (both with flip TTA)."""
    plt = _plt()
    keys = list(res["models"])
    clean = [np.nan if res["models"][k]["clean_test"] is None else res["models"][k]["clean_test"] for k in keys]
    hid = [res["models"][k]["acc"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(1.1 * len(keys) + 2, 4))
    ax.bar(x - 0.2, clean, 0.4, label="clean test (once)", color="#2a78d6")
    ax.bar(x + 0.2, hid, 0.4, label=f"hidden: {name}", color="#eb6834")
    for i in range(len(keys)):
        ax.text(x[i] - 0.2, clean[i] + 0.005, f"{clean[i]:.3f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.2, hid[i] + 0.005, f"{hid[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, keys)
    lo = np.nanmin(clean + hid)
    ax.set_ylim(lo - 0.08 if lo > 0.1 else 0, 1.0)
    ax.set_ylabel("accuracy (flip TTA)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"Clean test vs hidden set ({res['n']} images)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def evaluate_path(path, name, labels=None, invert="auto", write=True, loaded=None):
    LOG.clear()
    cfg, models = loaded or load_models()
    fmt, X01, y = load_any(path, labels, invert)
    preds, timing = run_models(X01, cfg, models)
    res = {"path": path, "format": fmt, "n": int(len(X01)), "tta": "flip",
           "models_declared": [k for k, *_ in models],
           "sources": {"teachers float": "runs/cnn_index.json best.npz", "teachers int8": "runs/cnn_{e2,b,d}_int8.npz",
                       "students int8": "runs/student_S{0,1}_int8.ace8", "clean_test": "runs/test_once.json, runs/final_model.json"},
           "standardize": {"mean": cfg["mean"], "std": cfg["std"]}, "log": list(LOG), "timing_s": timing,
           "pred_counts": {k: np.bincount(p, minlength=10).tolist() for k, p in preds.items()},
           "agree_float_int8": {r: float((preds[r] == preds[f"{r} int8"]).mean()) for r in TEACHERS}}
    if y is not None:
        if len(y) != len(X01):
            raise ValueError(f"{len(X01)} images but {len(y)} labels")
        res["models"] = {k: {**report(preds[k], y), "clean_test": t, "kind": kind} for k, kind, _, t in models}
        res["per_class_table_md"] = per_class_table(res)
        print(f"[{name}] n={len(y)} " + ", ".join(f"{k} {v['acc']:.4f}" for k, v in res["models"].items()))
    else:
        print(f"[{name}] n={len(X01)} unlabeled; float/int8 agreement {res['agree_float_int8']}")
    if write:
        stem = os.path.join(RUNS_DIR, f"hidden_eval_{name}")
        with open(stem + ".json", "w") as f:
            json.dump(res, f, indent=2)
        keys = list(preds)
        with open(stem + "_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index"] + [f"pred_{k.replace(' ', '_')}" for k in keys] + (["label"] if y is not None else []))
            for i in range(len(X01)):
                w.writerow([i] + [int(preds[k][i]) for k in keys] + ([int(y[i])] if y is not None else []))
        if y is not None:
            with open(stem + "_per_class.md", "w") as f:
                f.write(f"# Hidden set `{name}`: per-class accuracy (flip TTA)\n\n{res['per_class_table_md']}\n")
            plot_confusion({k: v["confusion"] for k, v in res["models"].items()}, stem + "_confusion.png", name)
            plot_vs_test(res, stem + "_vs_test.png", name)
    return res, preds


# ---------------------------------------------------------------- self-test

def selftest():
    """Save shifted val (uint8-quantized) as .npz, idx .gz and a PNG folder. Each loader must return
    exactly the reference images and labels (so every model's predictions are identical); npz and PNG are
    also run end to end through all models and checked prediction for prediction. Plus a 56x56 RGB PNG
    variant (resize path), evaluated on all models."""
    from acenet.shift import SHIFTED_VAL_PATH
    d = np.load(SHIFTED_VAL_PATH)
    X8 = np.round(d["X"] * 255).astype(np.uint8)
    y = d["y"].astype(np.int64)
    dirs = {k: os.path.join(SELFTEST_DIR, k) for k in ("npz", "idx", "png", "png_rgb56")}
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    t0 = time.perf_counter()
    np.savez_compressed(os.path.join(dirs["npz"], "shifted_val.npz"), X=X8, y=y)
    idx_write(os.path.join(dirs["idx"], "shifted-images-idx3-ubyte.gz"), X8)
    idx_write(os.path.join(dirs["idx"], "shifted-labels-idx1-ubyte.gz"), y.astype(np.uint8))
    for c in range(10):
        os.makedirs(os.path.join(dirs["png"], str(c)), exist_ok=True)
    for i in range(len(X8)):
        png_write(os.path.join(dirs["png"], str(y[i]), f"{i:05d}.png"), X8[i])
    n_rgb = min(500, len(X8))
    big = np.repeat(np.repeat(X8[:n_rgb], 2, axis=1), 2, axis=2)  # 56x56 nearest upsample
    for i in range(n_rgb):
        png_write(os.path.join(dirs["png_rgb56"], f"{i:05d}_{y[i]}.png"), np.stack([big[i]] * 3, axis=2))
    t_write = time.perf_counter() - t0

    loaded = load_models()
    cfg = loaded[0]
    X_ref = X8.astype(np.float32) / 255
    ref, _ = run_models(X_ref, cfg, loaded[1])
    out = {"source": "data/shifted_val.npz quantized to uint8", "n": int(len(y)), "write_s": t_write,
           "models": list(ref), "reference_acc": {k: float((p == y).mean()) for k, p in ref.items()}, "formats": {}}
    for fmt in ("npz", "idx", "png"):
        t0 = time.perf_counter()
        LOG.clear()
        _, X01, yl = load_any(dirs[fmt])
        order = np.arange(len(y))
        if fmt == "png":  # PNGs are read in path order (class folder, then index): map back to the original order
            files = sorted(glob.glob(os.path.join(dirs["png"], "**", "*.png"), recursive=True))
            order = np.array([int(os.path.basename(f)[:5]) for f in files])
        r = {"detected": fmt, "images_identical": bool(np.array_equal(X01, X_ref[order])),
             "labels_identical": bool(np.array_equal(yl, y[order])), "log": list(LOG)}
        if fmt in ("npz", "png"):
            res, preds = evaluate_path(dirs[fmt], f"selftest_{fmt}", write=(fmt == "png"), loaded=loaded)
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            r["acc"] = {k: v["acc"] for k, v in res["models"].items()}
            r["predictions_identical"] = {k: bool(np.array_equal(p[inv], ref[k])) for k, p in preds.items()}
        r["s"] = time.perf_counter() - t0
        out["formats"][fmt] = r
    res, preds = evaluate_path(dirs["png_rgb56"], "selftest_png_rgb56", write=False, loaded=loaded)
    out["formats"]["png_rgb56"] = {"detected": res["format"], "n": res["n"],
                                   "acc": {k: v["acc"] for k, v in res["models"].items()},
                                   "agree_with_reference": {k: float((p == ref[k][:n_rgb]).mean()) for k, p in preds.items()},
                                   "log": res["log"]}
    ok = all(v["images_identical"] and v["labels_identical"] and all(v.get("predictions_identical", {}).values())
             for k, v in out["formats"].items() if k != "png_rgb56")
    out["pass"] = bool(ok)
    with open(os.path.join(RUNS_DIR, "hidden_eval_selftest.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "log"} for k, v in out["formats"].items()},
                     indent=1))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--labels")
    ap.add_argument("--name", default="hidden")
    ap.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not a.path:
        ap.error("path required (or --selftest)")
    evaluate_path(a.path, a.name, a.labels, a.invert)


if __name__ == "__main__":
    main()
