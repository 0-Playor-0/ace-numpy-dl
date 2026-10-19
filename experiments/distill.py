"""Phase 6b: final selection (E2) + knowledge distillation into a <= 50,000-byte int8 student.

Usage:
  python -m experiments.distill teachers   # test ONCE (flip TTA): E2 float/int8, D float/int8
  python -m experiments.distill cache      # K=4 augmented copies of train + E2 logits -> data/kd_cache/
  python -m experiments.distill candidates # params / projected int8 bytes of the 3 student candidates
  python -m experiments.distill speed      # s/epoch of each candidate (fwd+bwd on cache batches)
  python -m experiments.distill screen     # lr_finder + 10 epochs CE for each candidate; pick by val
  python -m experiments.distill students   # S0 (CE), S1 (KD T=4), S2 (KD T=2), 40 epochs, from cache
  python -m experiments.distill final      # pick S* by val+TTA, quantize, size check, test ONCE
  python -m experiments.distill ablation   # runs/ablation.json + docs/ablation.md

Test-once guard: every test-set number goes through once(key, fn), which stores it in
runs/test_once.json and refuses to call fn again for a key that is already there.
"""

import argparse
import json
import os
import time

import numpy as np

from acenet import nn, ops
from acenet.data import CLASS_NAMES, CacheLoader, DataLoader, train_augment
from acenet.optim import Adam, CosineWarmup, lr_finder, plot_lr_finder
from acenet.quant import (Int8Model, batched, build_qspec, calibrate, compact_bytes, float_forward, fold_bn)
from acenet.shift import load_shifted_val
from acenet.tensor import Tensor, no_grad
from acenet.train import fit, load_checkpoint
from experiments.cnn import BATCH, EVAL_BATCH, ROOT, RUN_SEED, RUNS_DIR, WD, black_fill, load_data
from experiments.cnn_ablation import (ECE_BINS, confusion, ece, entry as run_entry, load_model, per_class_acc,
                                      predict, shift_groups)

TEST_ONCE = os.path.join(RUNS_DIR, "test_once.json")
CACHE_DIR = os.path.join(ROOT, "data", "kd_cache")             # v1: crop+flip+erase (used by the screen only)
CACHE_CF_DIR = os.path.join(ROOT, "data", "kd_cache_cropflip")  # v2: crop+flip only (all S runs)
DISTILL = os.path.join(RUNS_DIR, "distill.json")      # cache / candidates / speed / screen / students / final
K_COPIES = 4
CACHE_SEED = 21         # augmentation stream for the cache (copies drawn in order k = 0..3)
CALIB_SEED, N_CALIB = 4, 2000   # as Phase 4: 2,000 clean TRAIN images
SIZE_BUDGET = 50_000
SCREEN_EPOCHS, STUDENT_EPOCHS, WARMUP_EPOCHS = 10, 40, 1
STUDENTS = {"S0": None, "S1": (4.0, 0.9), "S2": (2.0, 0.9)}   # (T, alpha); None = CE only
# rules added after the screen started (user, Phase 6b; recorded in PROGRESS.md as deviations)
P_TOLERANCE = 0.003        # pick P if its screen val is within 0.3 pt of the best candidate (speed)
SLOW_S_PER_EPOCH, SLOW_EPOCHS = 120.0, 30   # selected student > 120 s/epoch -> 30 epochs for all S runs
CRITERION_PT = 0.02        # final criterion: student int8 test >= E2 float test - 2.0 pt
S2_GATE = 0.9450 - 0.02    # S2 runs only if S1 int8 val+TTA < E2 float val+TTA (0.9450) - 2.0 pt (val only)


def once(key, fn, path=TEST_ONCE):
    """Return the stored test result for key; otherwise run fn() ONCE, store and return it."""
    store = json.load(open(path)) if os.path.exists(path) else {}
    if key in store:
        print(f"[test-once] {key} already evaluated; returning stored result")
        return store[key]
    res = fn()
    store[key] = res
    with open(path, "w") as f:  # persisted immediately: the guard exists before anything else runs
        json.dump(store, f, indent=2)
    print(f"[test-once] {key}: {res}")
    return res


def float_test(rid, data):
    model = load_model(run_entry(rid))
    _, pt = predict(model, data["X_test"])
    y = data["y_test"]
    return {"tta": True, "test_acc": float((pt.argmax(1) == y).mean()),
            "test_nll": float(-np.log(pt[np.arange(len(y)), y]).mean())}


def teachers():
    """Step 1: E2 and D, float (softmax-mean flip TTA) then int8 (Phase 4 pipeline, p99.99,
    train-only calibration, logit-sum flip TTA). Each test number is evaluated once."""
    from experiments import quant
    data = load_data()
    out = {}
    for rid in ("E2", "D"):
        f = once(f"{rid}_float", lambda: float_test(rid, data))
        q = quant.main(rid, float_test=f["test_acc"], calibrators=("p99.99",))
        once(f"{rid}_int8", lambda: q["test"])
        out[rid] = {"float_test_tta": f["test_acc"], "int8_test_tta": q["test"]["test_acc"],
                    "int8_val_tta": q["calibrators"]["p99.99"]["val_acc_tta"],
                    "int8_file_bytes": q["size"]["file_bytes"]}
    print(json.dumps(out, indent=2))


def _load(path=DISTILL):
    return json.load(open(path)) if os.path.exists(path) else {}


def _save(key, value, path=DISTILL):
    d = _load(path)
    d[key] = value
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


# ---------------------------------------------------------------- step 2: teacher logits cache

def cache(cache_dir=CACHE_DIR, erase=True, key="cache"):
    """K augmented copies (crop pad 2 + hflip, + random erasing if erase; NO shift aug) of the
    54k train split, stored float16, and E2's eval-mode logits computed on the stored float16 images.
    v1 (erase=True, data/kd_cache) was used for the screen; v2 (erase=False, same seed and K) for S0-S2."""
    if os.path.exists(os.path.join(cache_dir, "meta.json")):
        print("cache exists:", json.load(open(os.path.join(cache_dir, "meta.json"))))
        return
    os.makedirs(cache_dir, exist_ok=True)
    data = load_data()
    X, y = data["X_train"], data["y_train"]
    fill = black_fill(data)
    teacher = load_model(run_entry("E2"))
    teacher.eval()
    rng = np.random.default_rng(CACHE_SEED)
    t0 = time.perf_counter()
    images = np.empty((K_COPIES,) + X.shape, np.float16)
    logits = np.empty((K_COPIES, len(X), 10), np.float32)
    t_aug = 0.0
    with no_grad():
        for k in range(K_COPIES):
            for i in range(0, len(X), 1000):
                ta = time.perf_counter()
                images[k, i:i + 1000] = train_augment(X[i:i + 1000], rng, fill, erase=erase)
                t_aug += time.perf_counter() - ta
                for j in range(i, min(i + 1000, len(X)), EVAL_BATCH):
                    logits[k, j:j + EVAL_BATCH] = teacher(Tensor(images[k, j:j + EVAL_BATCH].astype(np.float32))).data
            print(f"copy {k} done ({time.perf_counter() - t0:.0f}s)")
    build_s = time.perf_counter() - t0
    np.save(os.path.join(cache_dir, "images.npy"), images)
    np.save(os.path.join(cache_dir, "logits.npy"), logits)
    np.save(os.path.join(cache_dir, "labels.npy"), y)
    acc = [float((logits[k].argmax(1) == y).mean()) for k in range(K_COPIES)]
    files = {f: os.path.getsize(os.path.join(cache_dir, f)) for f in ("images.npy", "logits.npy", "labels.npy")}
    meta = {"K": K_COPIES, "n": int(len(y)), "seed": CACHE_SEED, "teacher": "E2", "teacher_run": run_entry("E2")["run_dir"],
            "augmentation": "crop pad 2 (black fill) + hflip 0.5" + (
                " + random erasing (p 0.5, area 0.02-0.2, N(0,1))" if erase else "") + "; no shift aug",
            "dir": os.path.relpath(cache_dir, ROOT),
            "images_dtype": "float16", "logits_dtype": "float32", "logits_on": "the stored float16 images, eval mode",
            "build_s": build_s, "augment_s": t_aug, "teacher_forward_s": build_s - t_aug,
            "files_bytes": files, "total_mb": sum(files.values()) / 2 ** 20,
            "teacher_train_acc_on_copies": acc}
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    _save(key, meta)
    print(json.dumps(meta, indent=2))


def cache_cropflip():
    cache(CACHE_CF_DIR, erase=False, key="cache_cropflip")


def load_cache(d=CACHE_DIR):
    return (np.load(os.path.join(d, "images.npy")), np.load(os.path.join(d, "labels.npy")),
            np.load(os.path.join(d, "logits.npy")))


# ---------------------------------------------------------------- step 4: student candidates

def conv_bn(cin, cout, k, rng):
    return [nn.Conv2D(cin, cout, k, rng, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]


def ds_bn(cin, cout, rng):
    """Depthwise 3x3 + BN + ReLU, then pointwise 1x1 + BN + ReLU."""
    return [nn.DepthwiseConv2D(cin, 3, rng, padding=1, bias=False), nn.BatchNorm2d(cin), nn.ReLU(),
            *conv_bn(cin, cout, 1, rng)]


def head(cin, rng):
    return [nn.GlobalAvgPool2D(), nn.Linear(cin, 10, rng)]


CANDIDATES = {
    # plain 3x3 convs: 16-24 @28, 48 @14, 64 @7
    "P": ("plain conv 16-24 | 48 | 64, GAP, fc",
          lambda r: nn.Sequential(*conv_bn(1, 16, 3, r), *conv_bn(16, 24, 3, r), nn.MaxPool2D(2),
                                  *conv_bn(24, 48, 3, r), nn.MaxPool2D(2), *conv_bn(48, 64, 3, r), *head(64, r))),
    # depthwise-separable after a 3x3 stem: 32 -> DS48 @28 | DS96, DS96 @14 | DS192 @7
    "DS": ("stem 32, DS 48 | DS 96, DS 96 | DS 192, GAP, fc",
           lambda r: nn.Sequential(*conv_bn(1, 32, 3, r), *ds_bn(32, 48, r), nn.MaxPool2D(2),
                                   *ds_bn(48, 96, r), *ds_bn(96, 96, r), nn.MaxPool2D(2),
                                   *ds_bn(96, 192, r), *head(192, r))),
    # hybrid: plain convs at 28x28, DS at 14 and 7
    "H": ("plain 24-32 @28 | DS 64, DS 128 | DS 160, GAP, fc",
          lambda r: nn.Sequential(*conv_bn(1, 24, 3, r), *conv_bn(24, 32, 3, r), nn.MaxPool2D(2),
                                  *ds_bn(32, 64, r), *ds_bn(64, 128, r), nn.MaxPool2D(2),
                                  *ds_bn(128, 160, r), *head(160, r))),
}


def build_student(cid, seed=RUN_SEED):
    return CANDIDATES[cid][1](np.random.default_rng(seed))


def size_report(model):
    """Trainable params, deployed (BN-folded) params and the exact compact int8 file size (unit scales:
    the byte count does not depend on the values)."""
    model.eval()
    fops = fold_bn(model)
    qspec = build_qspec(fops, [1.0] * sum(o["op"] in ("conv", "dwconv", "linear") for o in fops))
    qm = Int8Model.from_qspec(fops, qspec)
    model.train()
    folded = int(sum(o["Wm"].size + o["b"].size for o in fops if "Wm" in o))
    n_bytes = len(compact_bytes(qm))
    weights = int(sum(L["w"].size for L in qm.layers))
    channels = int(sum(L["b"].size for L in qm.layers))
    return {"params_trainable": int(sum(p.data.size for p in model.parameters())), "params_folded": folded,
            "int8_weights_bytes": weights, "per_channel_bytes": 9 * channels, "header_bytes": n_bytes - weights - 9 * channels,
            "int8_file_bytes": n_bytes, "macs_per_image": macs(model)}


def macs(model):
    h, c, total = 28, 1, 0
    for m in model.layers:
        if isinstance(m, nn.Conv2D):
            c_out, _, k, _ = m.W.shape
            total += h * h * c_out * c * k * k
            c = c_out
        elif isinstance(m, nn.DepthwiseConv2D):
            total += h * h * c * m.W.shape[2] * m.W.shape[3]
        elif isinstance(m, nn.MaxPool2D):
            h //= 2
        elif isinstance(m, nn.Linear):
            total += m.W.shape[0] * m.W.shape[1]
    return int(total)


def candidates():
    out = {cid: {"description": desc, **size_report(build_student(cid))} for cid, (desc, _) in CANDIDATES.items()}
    for cid, r in out.items():
        print(cid, r)
        assert 35_000 <= r["params_folded"] <= 45_000 and r["int8_file_bytes"] <= SIZE_BUDGET, cid
    _save("candidates", out)


def speed(n_timed=5):
    """fwd+bwd ms per batch of 128 cache images and projected s/epoch (incl. val eval), per candidate."""
    images, labels, _ = load_cache()
    data = load_data()
    out = {}
    for cid in CANDIDATES:
        model = build_student(cid)
        xb, yb = images[0, :BATCH].astype(np.float32), labels[:BATCH]
        ts = []
        for i in range(n_timed + 1):
            t0 = time.perf_counter()
            loss = ops.softmax_cross_entropy(model(Tensor(xb)), yb)
            model.zero_grad()
            loss.backward()
            if i:
                ts.append(time.perf_counter() - t0)
        model.eval()
        t0 = time.perf_counter()
        with no_grad():
            for i in range(0, 2000, EVAL_BATCH):
                model(Tensor(data["X_val"][i:i + EVAL_BATCH]))
        ev = (time.perf_counter() - t0) / 2000 * len(data["X_val"])
        ms = 1000 * float(np.median(ts))
        out[cid] = {"ms_per_batch_fwd_bwd": ms, "val_eval_s": ev,
                    "projected_s_per_epoch": -(-len(labels) // BATCH) * ms / 1000 + ev}
        print(cid, out[cid])
    _save("speed", out)


# ---------------------------------------------------------------- training

def _train(cid, name, epochs, lr, kd=None, cache_dir=CACHE_DIR):
    """AdamW (wd 5e-4), batch 128, CosineWarmup (1 ep warmup, cosine to 0), batches from the KD cache.
    kd = (T, alpha) for distillation; None = cross-entropy on the cache images (teacher ignored)."""
    images, labels, logits = load_cache(cache_dir)
    data = load_data()
    rng = np.random.default_rng(RUN_SEED)
    model = build_student(cid)
    opt = Adam(model.parameters(), lr=lr, weight_decay=WD, decoupled=True)
    tr = CacheLoader(images, labels, BATCH, rng, teacher=logits if kd else None)
    va = DataLoader(data["X_val"], data["y_val"], EVAL_BATCH, shuffle=False)
    sched = CosineWarmup(opt, WARMUP_EPOCHS, epochs, len(tr), min_lr=0.0)
    loss_fn = None if kd is None else (lambda z, y, t: ops.kd_loss(z, t, y, T=kd[0], alpha=kd[1]))
    config = {"student": cid, "arch": CANDIDATES[cid][0], **size_report(model), "optimizer": "AdamW", "lr": lr,
              "weight_decay": WD, "batch_size": BATCH, "schedule": f"CosineWarmup {WARMUP_EPOCHS} ep -> 0 at {epochs}",
              "data": f"KD cache {os.path.relpath(cache_dir, ROOT)} ({K_COPIES} copies, one random copy per sample per epoch)",
              "loss": "CE" if kd is None else f"KD T={kd[0]} alpha={kd[1]} + (1-alpha) CE", "teacher": "E2",
              "data_seed": 0}
    res = fit(model, opt, tr, va, epochs, run_name=name, config=config, seed=RUN_SEED, scheduler=sched,
              loss_fn=loss_fn)
    return {k: res[k] for k in ("run_name", "status", "best_epoch", "best_val_acc", "best_val_loss", "wall_time_s")} | {
        "run_dir": os.path.relpath(res["run_dir"], ROOT), "student": cid, "lr": lr, "epochs": epochs,
        "kd": kd, "s_per_epoch": res["wall_time_s"] / epochs}


def finder_for(cid):
    images, labels, _ = load_cache()
    loader = CacheLoader(images, labels, BATCH, np.random.default_rng(11))
    res = lr_finder(lambda: build_student(cid), lambda p, lr: Adam(p, lr=lr, weight_decay=WD, decoupled=True),
                    loader, lr_min=1e-6, lr_max=1.0)
    plot_lr_finder(res, os.path.join(RUNS_DIR, f"lr_finder_student_{cid}.png"), f"LR finder: student {cid}")
    return {k: v for k, v in res.items() if k not in ("lrs", "losses", "smoothed")}


def screen():
    """lr_finder + SCREEN_EPOCHS epochs of CE for each candidate; selection by best val accuracy."""
    out = _load().get("screen", {})
    for cid in CANDIDATES:
        if cid in out:
            continue
        t0 = time.perf_counter()
        f = finder_for(cid)
        f["wall_time_s"] = time.perf_counter() - t0
        r = _train(cid, f"student_screen_{cid}", SCREEN_EPOCHS, f["suggested_min_div10"])
        out[cid] = {"finder": f, **r}
        _save("screen", out)
        print(cid, out[cid])
    best = max(out, key=lambda c: out[c]["best_val_acc"])
    _save("selected_student_arch", {"id": best, "rule": "max best val acc after 10-epoch CE screen",
                                    "val": {c: out[c]["best_val_acc"] for c in out}})
    print("selected", best)


def select_arch():
    """Screen selection (revised rule): P if its val is within P_TOLERANCE of the best candidate, else the
    best by val. Also fixes the S-run epoch count from the selected arch's measured screen s/epoch."""
    d = _load()
    sc = d["screen"]
    val = {c: sc[c]["best_val_acc"] for c in sc}
    best = max(val, key=val.get)
    sel = "P" if val[best] - val["P"] <= P_TOLERANCE + 1e-12 else best
    spe = sc[sel]["s_per_epoch"]
    epochs = SLOW_EPOCHS if spe > SLOW_S_PER_EPOCH else STUDENT_EPOCHS
    out = {"id": sel, "rule": f"P if val within {100 * P_TOLERANCE:.1f} pt of the best, else max val (10-ep CE screen)",
           "val": val, "best_by_val": best, "old_rule_pick": d.get("selected_student_arch", {}).get("id"),
           "screen_s_per_epoch": spe, "student_epochs": epochs,
           "epoch_rule": f"{SLOW_EPOCHS} if screen s/epoch > {SLOW_S_PER_EPOCH:g} else {STUDENT_EPOCHS}"}
    _save("selected_student_arch", out)
    print(out)


def train_student(sid):
    d = _load()
    a = d["selected_student_arch"]
    out = d.get("students", {})
    if sid in out:
        print(f"{sid} already trained; skipping")
        return
    lr = d["screen"][a["id"]]["finder"]["suggested_min_div10"]
    out[sid] = _train(a["id"], f"student_{sid}", a["student_epochs"], lr, kd=STUDENTS[sid], cache_dir=CACHE_CF_DIR)
    _save("students", out)
    print(sid, out[sid])


def students():
    """S0 (CE) and S1 (KD T=4) on the crop+flip cache; S1 is quantized (val only, no test); S2 (KD T=2)
    is trained only if S1's int8 val+TTA < S2_GATE. No student touches test here."""
    train_student("S0")
    train_student("S1")
    r = quantize_student("S1")
    need = r["val_acc_tta"] < S2_GATE
    _save("s2_decision", {"S1_int8_val_tta": r["val_acc_tta"], "gate": S2_GATE, "run_S2": bool(need),
                          "rule": "train S2 iff S1 int8 val+TTA < E2 val+TTA 0.9450 - 2.0 pt"})
    if need:
        train_student("S2")


def load_student(sid):
    r = _load()["students"][sid]
    model = build_student(r["student"])
    ep = load_checkpoint(model, os.path.join(ROOT, r["run_dir"], "best.npz"))
    assert ep == r["best_epoch"]
    return model.eval()


# ---------------------------------------------------------------- evaluation rows

def eval_model_row(rid, model, data, Xs, ys, groups, extra=None):
    Xv, yv = data["X_val"], data["y_val"]
    p, p_tta = predict(model, Xv)
    ps, ps_tta = predict(model, Xs)
    pred, pred_tta = p.argmax(1), p_tta.argmax(1)
    return {"id": rid, **(extra or {}),
            "val_acc": float((pred == yv).mean()), "val_acc_tta": float((pred_tta == yv).mean()),
            "val_nll": float(-np.log(p[np.arange(len(yv)), yv]).mean()),
            "ece": ece(p, yv), "ece_tta": ece(p_tta, yv),
            "per_class_val_acc_tta": per_class_acc(pred_tta, yv),
            "confusion_val_tta": confusion(pred_tta, yv).tolist(),
            "shifted_val_acc": {g: float((ps.argmax(1)[m] == ys[m]).mean()) for g, m in groups.items()},
            "shifted_val_acc_tta": {g: float((ps_tta.argmax(1)[m] == ys[m]).mean()) for g, m in groups.items()},
            "shifted_val_acc_all": float((ps.argmax(1) == ys).mean()),
            "shifted_val_acc_all_tta": float((ps_tta.argmax(1) == ys).mean())}


def student_rows(data, Xs, ys, groups):
    rows = {}
    for sid, r in _load()["students"].items():
        rows[sid] = eval_model_row(sid, load_student(sid), data, Xs, ys, groups,
                                   {"run_dir": r["run_dir"], "kd": r["kd"], "best_epoch": r["best_epoch"],
                                    "wall_time_s": r["wall_time_s"]})
        assert abs(rows[sid]["val_acc"] - r["best_val_acc"]) < 1e-9
        print(sid, {k: rows[sid][k] for k in ("val_acc", "val_acc_tta", "ece", "shifted_val_acc_all")})
    return rows


# ---------------------------------------------------------------- step 6: select, quantize, test once

def flip(X):
    return np.ascontiguousarray(X[..., ::-1])


def acc(z, y):
    return float((z.argmax(1) == y).mean())


def _quantized(sid):
    """(eval model, Int8Model loaded from the deployed compact file) for an already quantized student."""
    r = _load()["quant"][sid]
    return load_student(sid), Int8Model.load_compact(os.path.join(ROOT, r["file"]))


def quantize_student(sid):
    """Quantize student sid (Phase 4 pipeline: BN fold, p99.99 on 2,000 clean TRAIN images, per-channel int8),
    write the compact file, verify size and round trip, int8 val / shifted val. Validation only: no test."""
    d = _load()
    if sid in d.get("quant", {}):
        return d["quant"][sid]
    data = load_data()
    Xv, yv = data["X_val"], data["y_val"]
    Xs, ys, kind, _, mode = load_shifted_val(data["mean"], data["std"])
    groups = shift_groups(kind, mode)
    model = load_student(sid)
    fops = fold_bn(model)
    with no_grad():
        z_ref = np.concatenate([model(Tensor(Xv[i:i + EVAL_BATCH])).data for i in range(0, len(Xv), EVAL_BATCH)])
    z_fold = batched(lambda xb: float_forward(fops, xb), Xv)
    assert (z_fold.argmax(1) == z_ref.argmax(1)).all() and np.abs(z_fold - z_ref).max() < 1e-3
    rng = np.random.default_rng(CALIB_SEED)
    Xc = data["X_train"][rng.choice(len(data["X_train"]), N_CALIB, replace=False)]
    scales = calibrate(fops, Xc, "p99.99")
    qspec = build_qspec(fops, scales)
    path = os.path.join(RUNS_DIR, f"student_{sid}_int8.ace8")
    n_bytes = Int8Model.from_qspec(fops, qspec).save_compact(path)
    assert os.path.getsize(path) == n_bytes <= SIZE_BUDGET, n_bytes
    qm = Int8Model.load_compact(path)  # everything below runs on the deployed file
    zi, zi_f = qm.logits(Xv), qm.logits(flip(Xv))
    zf = batched(lambda xb: float_forward(fops, xb, qspec=qspec), Xv)
    zs, zs_f = qm.logits(Xs), qm.logits(flip(Xs))
    r = {"file": os.path.relpath(path, ROOT), "file_bytes": n_bytes, "budget": SIZE_BUDGET,
         "act_scales": scales, "fold_max_abs_logit_diff": float(np.abs(z_fold - z_ref).max()),
         "val_acc": acc(zi, yv), "val_acc_tta": acc(zi + zi_f, yv),
         "fake_val_acc": acc(zf, yv), "agree_int_vs_fake": float((zi.argmax(1) == zf.argmax(1)).mean()),
         "agree_int_vs_float": float((zi.argmax(1) == z_fold.argmax(1)).mean()),
         "shifted_val_acc": {g: acc(zs[m], ys[m]) for g, m in groups.items()},
         "shifted_val_acc_tta": {g: acc((zs + zs_f)[m], ys[m]) for g, m in groups.items()},
         "shifted_val_acc_all": acc(zs, ys), "shifted_val_acc_all_tta": acc(zs + zs_f, ys)}
    q = _load().get("quant", {})
    q[sid] = r
    _save("quant", q)
    print(sid, {k: v for k, v in r.items() if k != "act_scales"})
    return r


def test_student(sid, data):
    """Test ONCE (flip TTA): float (softmax mean) and int8 (logit sum, on the deployed compact file).
    Called only for the final student, after it is selected on val+TTA."""
    model, qm = _quantized(sid)
    yt = data["y_test"]

    def ft():
        _, pt = predict(model, data["X_test"])
        return {"model": sid, "tta": True, "test_acc": float((pt.argmax(1) == yt).mean())}

    q = _load()["quant"]
    q[sid]["float_test"] = once(f"student_{sid}_float", ft)["test_acc"]
    q[sid]["int8_test"] = once(f"student_{sid}_int8", lambda: {
        "model": sid, "tta": True, "file_bytes": q[sid]["file_bytes"],
        "test_acc": acc(qm.logits(data["X_test"], flip=True), yt)})["test_acc"]
    _save("quant", q)
    return q[sid]


def final():
    """Quantize every trained S run (val only, no test), pick the final student by INT8 val+TTA
    (ties -> S0, S1, S2), then test that one student ONCE (float + int8). No other student touches test."""
    data = load_data()
    Xs, ys, kind, _, mode = load_shifted_val(data["mean"], data["std"])
    rows = student_rows(data, Xs, ys, shift_groups(kind, mode))
    int8_val_tta = {sid: quantize_student(sid)["val_acc_tta"] for sid in rows}
    sel = max(rows, key=lambda s: (int8_val_tta[s], -int(s[1])))
    store = json.load(open(TEST_ONCE))
    tested = [k for k in store if k.startswith("student_") and not k.startswith(f"student_{sel}_")]
    assert not tested, f"another student already touched test: {tested}"
    q = test_student(sel, data)
    store = json.load(open(TEST_ONCE))
    e2f, e2i = store["E2_float"]["test_acc"], store["E2_int8"]["test_acc"]
    res = {"rows": rows, "selected": sel, "int8_val_acc_tta": int8_val_tta,
           "rule": "max INT8 val acc with flip TTA (logit sum) over trained S runs; ties -> S0, S1, S2",
           "int8": q, "test": {"float": q["float_test"], "int8": q["int8_test"], "E2_float": e2f, "E2_int8": e2i,
                               "int8_minus_E2_float_pt": 100 * (q["int8_test"] - e2f),
                               "int8_minus_E2_int8_pt": 100 * (q["int8_test"] - e2i),
                               "criterion": "student int8 test >= E2 float test - 2.0 pt",
                               "pass": bool(q["int8_test"] >= e2f - CRITERION_PT)}}
    _save("final", res)
    print("selected", sel, json.dumps(res["test"], indent=2))


# ---------------------------------------------------------------- step 7: ablation table

ABLATION = os.path.join(RUNS_DIR, "ablation.json")
ABLATION_MD = os.path.join(ROOT, "docs", "ablation.md")
ORDER = ["A", "C", "B", "E1", "E2", "D", "S0", "S1", "S2"]


def _float_params(row):
    cfg = json.load(open(os.path.join(ROOT, row["run_dir"], "config.json")))
    return cfg["params"], cfg.get("buffers", 0)


def ablation():
    """Merge teacher rows (runs/ablation.json, Phase 3b/6a) with student rows and int8 rows into one table."""
    abl = json.load(open(ABLATION))
    d = _load()
    tests = json.load(open(TEST_ONCE))
    rows = {r["id"]: r for r in abl["rows"] if r["id"] in ORDER[:6]}
    rows.update(d["final"]["rows"])
    sel = d["final"]["selected"]
    quant = d["quant"]
    order = [r for r in ORDER if r in rows]
    test = {"A": (json.load(open(os.path.join(RUNS_DIR, "cnn_test.json")))["test_acc"], False),
            "B": (json.load(open(os.path.join(RUNS_DIR, "cnn_ablation.json")))["test"]["test_acc"], True),
            "E2": (tests["E2_float"]["test_acc"], True), "D": (tests["D_float"]["test_acc"], True),
            **{sid: (q["float_test"], True) for sid, q in quant.items() if "float_test" in q}}
    table = []
    for rid in order:
        r = rows[rid]
        if rid.startswith("S"):
            cfg = json.load(open(os.path.join(ROOT, r["run_dir"], "config.json")))
            params, buffers = cfg["params_trainable"], 0
            bn = sum(m.running_mean.size * 2 for m in build_student(cfg["student"]).modules()
                     if isinstance(m, nn._BatchNorm))
            buffers = bn
        else:
            params, buffers = _float_params(r)
        t = test.get(rid)
        table.append({"id": rid, "kind": "float32", "val": r["val_acc"], "val_tta": r["val_acc_tta"],
                      "test": t[0] if t else None, "test_tta": t[1] if t else None, "ece": r["ece"],
                      "shifted_all": r["shifted_val_acc_all"], "noise": r["shifted_val_acc"]["noise"],
                      "params": params, "size_kb": 4 * (params + buffers) / 1024})
        q = None
        if rid in ("B", "E2", "D"):
            qj = json.load(open(os.path.join(RUNS_DIR, f"cnn_{rid.lower()}_int8.json")))
            c = qj["calibrators"]["p99.99"]
            sv = qj["shifted_val"]
            q = {"val": c["val_acc"], "val_tta": c["val_acc_tta"], "test": qj["test"]["test_acc"],
                 "shifted_all": sum(v["n"] * v["int8"] for v in sv.values()) / sum(v["n"] for v in sv.values()),
                 "noise": sv["noise"]["int8"], "size_kb": qj["size"]["file_bytes"] / 1024, "format": "npz"}
        elif rid in quant:
            f8 = quant[rid]
            q = {"val": f8["val_acc"], "val_tta": f8["val_acc_tta"], "test": f8.get("int8_test"),
                 "shifted_all": f8["shifted_val_acc_all"], "noise": f8["shifted_val_acc"]["noise"],
                 "size_kb": f8["file_bytes"] / 1024, "format": "ace8 compact"}
        if q:
            table.append({"id": f"{rid} int8", "kind": "int8", **q, "test_tta": True, "ece": None,
                          "params": table[-1]["params"]})
    by = {t["id"]: t for t in table}

    def dv(a, b, k="val_tta"):
        return 100 * (by[a][k] - by[b][k])

    deltas = {
        "random erasing (B - A)": dv("B", "A"), "label smoothing (C - B)": dv("C", "B"),
        "finder lr, constant (E1 - A)": dv("E1", "A"), "cosine warmup (E2 - E1)": dv("E2", "E1"),
        "shift aug + erase on E2 recipe (D - E2)": dv("D", "E2"),
        "KD T=4 (S1 - S0)": dv("S1", "S0"), **({"KD T=2 (S2 - S0)": dv("S2", "S0")} if "S2" in by else {}),
        "int8 PTQ, E2 (E2 int8 - E2)": dv("E2 int8", "E2"), f"int8 PTQ, student ({sel} int8 - {sel})": dv(f"{sel} int8", sel),
        f"student vs teacher ({sel} - E2)": dv(sel, "E2"),
        "shifted overall, shift aug (D - E2)": dv("D", "E2", "shifted_all"),
        "noise, random erasing (B - A)": dv("B", "A", "noise"),
    }
    out = {"phase": "6b", "columns": "val/val_tta/test are accuracies; test_tta: whether test used flip TTA; "
                                     "shifted_all/noise: shifted val, no TTA; size_kb: float = 4 B x (params + "
                                     "BN buffers), int8 = file size",
           "student_selected": sel, "table": table, "deltas_pt_val_tta": deltas, "teacher_rows": abl["rows"],
           "student_rows": d["final"]["rows"]}
    with open(os.path.join(RUNS_DIR, "ablation_6b.json"), "w") as f:
        json.dump(out, f, indent=2)
    abl["table_6b"], abl["deltas_6b_pt_val_tta"], abl["student_rows"] = table, deltas, d["final"]["rows"]
    with open(ABLATION, "w") as f:
        json.dump(abl, f, indent=2)

    def f(v, d=4):
        return "–" if v is None else f"{v:.{d}f}"

    lines = ["| model | val | val+TTA | test | ECE | shifted all | noise | params | size KB |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        test_s = "–" if t["test"] is None else f"{t['test']:.4f}" + ("" if t["test_tta"] else " (no TTA)")
        lines.append(f"| {t['id']} | {f(t['val'])} | {f(t['val_tta'])} | {test_s} | {f(t['ece'])} | "
                     f"{f(t['shifted_all'], 3)} | {f(t['noise'], 3)} | {t['params']:,} | {t['size_kb']:.1f} |")
    print("\n".join(lines))
    print({k: round(v, 2) for k, v in deltas.items()})
    return lines, deltas


def main():
    ap = argparse.ArgumentParser()
    stages = {"teachers": teachers, "cache": cache, "candidates": candidates, "speed": speed, "screen": screen,
              "cache_cropflip": cache_cropflip, "select_arch": select_arch, "students": students, "final": final,
              "ablation": ablation}
    ap.add_argument("stage", choices=list(stages))
    stages[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
