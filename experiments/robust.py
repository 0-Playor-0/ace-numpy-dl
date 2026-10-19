"""Phase 5+6a: LR finder, cosine-warmup schedule, shift augmentation, robustness evaluation, final selection.

Usage:
  python -m experiments.robust finder    # LR range test on A's architecture + augmentation
  python -m experiments.robust augspeed  # added ms/batch of shift augmentation
  python -m experiments.robust train     # E1 (only if needed), E2, D; 20 epochs, seed of A
  python -m experiments.robust eval      # val metrics for A/B/C/E1/E2/D -> runs/ablation.json,
                                         # noise mechanism -> runs/robustness.json
  python -m experiments.robust plot      # runs/robustness_{noise,shift}.png
  python -m experiments.robust select    # B vs D (rule fixed below); if D: test once (float TTA),
                                         # quantize D (p99.99, train calibration), test int8 once
"""

import argparse
import json
import os
import time

import numpy as np

from acenet import nn
from acenet.data import CLASS_NAMES, DataLoader, train_augment
from acenet.optim import Adam, lr_finder, plot_lr_finder
from acenet.shift import KINDS, SHIFTED_VAL_PATH, load_shifted_val, make_shifted
from experiments.cnn import (BATCH, LR, RUN_SEED, RUNS_DIR, SHIFT_AUG_SEED, WD, ROOT, _index, baseline,
                             black_fill, build_model, load_data, run)
from experiments.cnn_ablation import (ECE_BINS, SHIFT_GROUPS, TAGS, entry, eval_row, load_model, predict,
                                      shift_groups)

FINDER_JSON = os.path.join(RUNS_DIR, "lr_finder.json")
FINDER_PNG = os.path.join(RUNS_DIR, "lr_finder.png")
SPEED_JSON = os.path.join(RUNS_DIR, "shift_aug_speed.json")
ABLATION = os.path.join(RUNS_DIR, "ablation.json")
ROBUST = os.path.join(RUNS_DIR, "robustness.json")
FINAL = os.path.join(RUNS_DIR, "final_model.json")
FINDER_SEED = 11
E1_TOLERANCE = 1.5      # skip E1 if finder lr is within x1.5 of A's 1e-3
WARMUP_EPOCHS = 1
SELECTION_RULE = ("among B and D: higher val accuracy with flip TTA wins; if |diff| <= 0.0015 (0.15 pt), "
                  "the higher overall shifted-val accuracy (no TTA, all 6000 images) wins")
NOISE_BINS = [0.1, 0.15, 0.2, 0.25, 0.3]

DESCRIPTIONS = {
    "A": ("baseline (Phase 3 selected)", []),
    "B": ("A + random erasing", ["random_erase"]),
    "C": ("B + label smoothing 0.1", ["random_erase", "label_smoothing=0.1"]),
    "E1": ("A + finder lr, constant", ["lr=finder"]),
    "E2": ("A + finder lr + CosineWarmup(1 ep)", ["lr=finder", "cosine_warmup"]),
    "D": ("B + shift aug + finder lr + CosineWarmup(1 ep)", ["random_erase", "shift_aug", "lr=finder",
                                                              "cosine_warmup"]),
}


def finder():
    data = load_data()
    fill = black_fill(data)
    rng = np.random.default_rng(FINDER_SEED)
    loader = DataLoader(data["X_train"], data["y_train"], BATCH, shuffle=True, rng=rng,
                        transform=lambda x, r: train_augment(x, r, fill))
    t0 = time.perf_counter()
    res = lr_finder(lambda: build_model(np.random.default_rng(RUN_SEED)),
                    lambda p, lr: Adam(p, lr=lr, weight_decay=WD, decoupled=True), loader,
                    lr_min=1e-6, lr_max=1.0)
    res["wall_time_s"] = time.perf_counter() - t0
    lr = res["suggested_min_div10"]
    res["used"] = "suggested_min_div10"
    res["ratio_vs_1e-3"] = lr / LR
    res["run_e1"] = not (1 / E1_TOLERANCE <= lr / LR <= E1_TOLERANCE)
    res["config"] = {"model": "A (cnn-32-32-64-64-fc128)", "augmentation": "crop pad 2 + hflip",
                     "optimizer": "AdamW wd 5e-4", "batch": BATCH, "lr_range": [1e-6, 1.0],
                     "schedule": "exponential per batch over 1 epoch", "stop": "smoothed loss > 4x min",
                     "seed_loader": FINDER_SEED, "seed_model": RUN_SEED}
    with open(FINDER_JSON, "w") as f:
        json.dump(res, f, indent=2)
    plot_lr_finder(res, FINDER_PNG, "LR finder: CNN A (AdamW, crop+flip), 1e-6 -> 1")
    print({k: v for k, v in res.items() if k not in ("lrs", "losses", "smoothed")})


def finder_lr():
    with open(FINDER_JSON) as f:
        return json.load(f)


def augspeed(n=30):
    data = load_data()
    fill = black_fill(data)
    xb = data["X_train"][:BATCH]
    out = {}
    for name, shift in (("crop_flip_erase", None),
                        ("crop_flip_shift_erase", (np.random.default_rng(SHIFT_AUG_SEED), data["mean"], data["std"]))):
        rng = np.random.default_rng(0)
        train_augment(xb, rng, fill, erase=True, shift=shift)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            train_augment(xb, rng, fill, erase=True, shift=shift)
            ts.append(time.perf_counter() - t0)
        out[name] = 1000 * float(np.median(ts))
    out["added_ms_per_batch"] = out["crop_flip_shift_erase"] - out["crop_flip_erase"]
    out["batch"] = BATCH
    with open(SPEED_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(out)


def train():
    f = finder_lr()
    lr = f["suggested_min_div10"]
    a = baseline()
    done = {e["tag"] for e in _index()}
    plan = []
    if f["run_e1"]:
        plan.append(("E1", dict(lr=lr)))
    plan += [("E2", dict(lr=lr, warmup_epochs=WARMUP_EPOCHS)),
             ("D", dict(lr=lr, warmup_epochs=WARMUP_EPOCHS, erase=True, shift_aug=True))]
    for rid, kw in plan:
        if TAGS[rid] in done:
            print(f"{rid} already trained; skipping")
            continue
        t0 = time.time()
        e = run(TAGS[rid], a["epochs"], **kw)
        print(f"{rid} done: best val {e['best_val_acc']:.4f} ep {e['best_epoch']} ({time.time() - t0:.0f}s)")


def ids_available():
    tags = {e["tag"] for e in _index()}
    return ["A", "B", "C"] + [r for r in ("E1", "E2", "D") if TAGS[r] in tags]


def noise_mechanism(models, data, Xs, ys, kind, sev):
    """Clipped Gaussian noise on [0, 1] images: background lift vs object speckle.

    For the noise subset of shifted val (the exact noisy images), build two counterfactuals from the
    clean val image: noise kept only on background pixels (clean == 0) or only on object pixels.
    Reports accuracy per sigma bin for each model and the mean background lift.
    """
    mean, std = data["mean"], data["std"]
    assert np.array_equal(ys, data["y_val"]), "shifted val must be aligned with the val split"
    m = kind == KINDS.index("noise")
    clean01 = data["X_val"][m, 0] * np.float32(std) + np.float32(mean)
    clean01 = np.clip(np.round(clean01 * 255) / 255, 0, 1)  # undo float error: originals are uint8/255
    noisy01 = Xs[m, 0] * np.float32(std) + np.float32(mean)
    bg = clean01 <= 0.5 / 255
    variants = {"full noise": noisy01,
                "noise on background only": np.where(bg, noisy01, clean01),
                "noise on object only": np.where(bg, clean01, noisy01),
                "full noise, background re-zeroed": np.where(bg, 0.0, noisy01)}
    y, s = ys[m], sev[m]
    bins = list(zip(NOISE_BINS[:-1], NOISE_BINS[1:]))
    sel = [(s >= lo) & (s <= hi) for lo, hi in bins]
    out = {"n": int(m.sum()), "background_fraction": float(bg.mean()),
           "background_mean_after_noise_by_sigma": {}, "expected_lift_sigma_over_sqrt_2pi": {},
           "object_mean_abs_change_by_sigma": {}, "acc": {}}
    for (lo, hi), b in zip(bins, sel):
        key = f"{lo:g}-{hi:g}"
        out["background_mean_after_noise_by_sigma"][key] = float(noisy01[b][bg[b]].mean())
        out["expected_lift_sigma_over_sqrt_2pi"][key] = float(s[b].mean() / np.sqrt(2 * np.pi))
        out["object_mean_abs_change_by_sigma"][key] = float(np.abs(noisy01[b] - clean01[b])[~bg[b]].mean())
    for rid, model in models.items():
        out["acc"][rid] = {}
        for vname, v in variants.items():
            p, p_tta = predict(model, ((v - mean) / std).astype(np.float32)[:, None])
            pred = p.argmax(1)
            out["acc"][rid][vname] = {"all": float((pred == y).mean()),
                                      **{f"{lo:g}-{hi:g}": float((pred[b] == y[b]).mean())
                                         for (lo, hi), b in zip(bins, sel)}}
        print(rid, {k: round(v["all"], 4) for k, v in out["acc"][rid].items()})
    return out


def rotation_by_angle(models, Xs, ys, kind, sev):
    m = kind == KINDS.index("rotate")
    out = {}
    for rid, model in models.items():
        pred = predict(model, Xs[m])[0].argmax(1)
        a = np.abs(sev[m])
        out[rid] = {f"{lo}-{hi}": float((pred[(a >= lo) & (a <= hi)] == ys[m][(a >= lo) & (a <= hi)]).mean())
                    for lo, hi in ((0, 5), (5, 10), (10, 15))}
    return out


def evaluate_all():
    data = load_data()
    Xv, yv = data["X_val"], data["y_val"]
    Xs, ys, kind, sev, mode = load_shifted_val(data["mean"], data["std"])
    groups = shift_groups(kind, mode)
    rows = []
    for rid in ids_available():
        desc, changes = DESCRIPTIONS[rid]
        rows.append(eval_row(rid, desc, changes, entry(rid), Xv, yv, Xs, ys, groups))
    f = finder_lr()
    out = {"phases": "3b + 5/6a", "baseline": "A", "class_names": CLASS_NAMES, "shift_groups": list(SHIFT_GROUPS),
           "ece_bins": ECE_BINS, "tta": "mean of softmax(x) and softmax(hflip(x))",
           "finder": {"suggested_min_div10": f["suggested_min_div10"], "suggested_steepest": f["suggested_steepest"],
                      "e1_run": f["run_e1"],
                      "e1_note": None if f["run_e1"] else
                      f"finder lr within x{E1_TOLERANCE} of 1e-3: E1 skipped, A covers constant lr"},
           "note_shifted_val_for_D": "D's augmentation targets the same shift kinds/ranges as shifted val; "
                                     "shifted val is not an independent check for D",
           "rows": rows}
    with open(ABLATION, "w") as fh:
        json.dump(out, fh, indent=2)
    models = {rid: load_model(entry(rid)) for rid in ("A", "B", "D") if rid in ids_available()}
    rob = {"noise_mechanism": noise_mechanism(models, data, Xs, ys, kind, sev),
           "rotation_by_abs_angle": rotation_by_angle(models, Xs, ys, kind, sev)}
    with open(ROBUST, "w") as fh:
        json.dump(rob, fh, indent=2)
    for r in rows:
        print(f"{r['id']:3s} val {r['val_acc']:.4f} tta {r['val_acc_tta']:.4f} ece {r['ece']:.4f} "
              f"shift {r['shifted_val_acc_all']:.4f} noise {r['shifted_val_acc']['noise']:.4f} "
              f"wall {r['wall_time_s']:.0f}s")


def select():
    with open(ABLATION) as f:
        rows = {r["id"]: r for r in json.load(f)["rows"]}
    b, d = rows["B"], rows["D"]
    diff = d["val_acc_tta"] - b["val_acc_tta"]
    if abs(diff) > 0.0015 + 1e-12:
        winner, why = ("D" if diff > 0 else "B"), f"val+TTA diff D-B = {diff:+.4f} > 0.15 pt"
    else:
        winner = "D" if d["shifted_val_acc_all"] > b["shifted_val_acc_all"] else "B"
        why = (f"val+TTA diff D-B = {diff:+.4f} within 0.15 pt; shifted-val B {b['shifted_val_acc_all']:.4f} "
               f"vs D {d['shifted_val_acc_all']:.4f}")
    previous = json.load(open(FINAL)) if os.path.exists(FINAL) else {}
    data = load_data()
    res = {"rule": SELECTION_RULE, "B": {k: b[k] for k in ("val_acc_tta", "shifted_val_acc_all")},
           "D": {k: d[k] for k in ("val_acc_tta", "shifted_val_acc_all")}, "selected": winner, "reason": why,
           "mean": data["mean"], "std": data["std"], "tta": "flip"}
    print(res)
    if winner == "B":
        res.update(checkpoint=os.path.relpath(os.path.join(ROOT, b["run_dir"], "best.npz"), ROOT),
                   int8="runs/cnn_b_int8.npz", test_float_tta=0.9349, test_int8_tta=0.9344,
                   test_note="B kept: Phase 3b/4 test results stand; test not touched")
        with open(FINAL, "w") as f:
            json.dump(res, f, indent=2)
        return
    res.update(checkpoint=os.path.relpath(os.path.join(ROOT, d["run_dir"], "best.npz"), ROOT),
               int8="runs/cnn_d_int8.npz")
    test = previous.get("test_float")
    if test is not None:
        if previous.get("selected") != "D":
            raise SystemExit("test guard: a different model was already tested")
        print("float test already evaluated; not re-running")
    else:
        model = load_model(entry("D"))
        _, pt = predict(model, data["X_test"])
        test = {"tta": True, "test_acc": float((pt.argmax(1) == data["y_test"]).mean()),
                "test_nll": float(-np.log(pt[np.arange(len(pt)), data["y_test"]]).mean())}
        print("FLOAT TEST (once)", test)
    res["test_float"] = test
    with open(FINAL, "w") as f:  # persist the guard before quantization
        json.dump(res, f, indent=2)
    from experiments import quant
    q = quant.main("D", float_test=test["test_acc"], calibrators=("p99.99",))
    res["test_int8"] = q["test"]
    res["int8_val_acc_tta"] = q["calibrators"]["p99.99"]["val_acc_tta"]
    res["int8_shifted_val"] = q["shifted_val"]
    with open(FINAL, "w") as f:
        json.dump(res, f, indent=2)
    print("INT8 TEST (once)", q["test"])


def plot():
    """runs/robustness_noise.png: accuracy vs noise sigma (full / background-only / object-only) for A, B, D;
    runs/robustness_shift.png: shifted-val accuracy by kind for A, B, E2, D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nm = json.load(open(ROBUST))["noise_mechanism"]
    bins = list(nm["background_mean_after_noise_by_sigma"])
    x = [np.mean([float(v) for v in b.split("-")]) for b in bins]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for a, v in zip(ax, ("full noise", "noise on background only", "noise on object only")):
        for rid, c in (("A", "C0"), ("B", "C1"), ("D", "C2")):
            a.plot(x, [nm["acc"][rid][v][b] for b in bins], "o-", color=c, label=rid)
        a.set_title(v)
        a.set_xlabel("noise sigma (bin centre)")
        a.set_ylim(0, 1)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("accuracy (no TTA)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS_DIR, "robustness_noise.png"), dpi=100)
    plt.close(fig)
    rows = {r["id"]: r for r in json.load(open(ABLATION))["rows"]}
    groups = list(SHIFT_GROUPS) + ["clean"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ids = [i for i in ("A", "B", "E2", "D") if i in rows]
    for k, rid in enumerate(ids):
        vals = [rows[rid]["shifted_val_acc"][g] for g in SHIFT_GROUPS] + [rows[rid]["val_acc"]]
        ax.bar(np.arange(len(groups)) + (k - 1.5) * 0.2, vals, 0.2, label=rid)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("val accuracy (no TTA)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("shifted val by kind (D trained on the same shift kinds)")
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS_DIR, "robustness_shift.png"), dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["finder", "augspeed", "train", "eval", "select", "plot"])
    {"finder": finder, "augspeed": augspeed, "train": train, "eval": evaluate_all,
     "select": select, "plot": plot}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
