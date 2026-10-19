"""Phase 3b evaluation of CNN runs A (baseline), B (A + random erasing), C (B + label smoothing 0.1).

Validation only (clean val + data/shifted_val.npz): accuracy with and without horizontal-flip TTA,
per-class accuracy, confusion matrix, 15-bin ECE, shifted-val accuracy by kind. Selects the best of
A/B/C (with or without TTA) by val accuracy; only if that is not A is it evaluated on test, once.
Writes runs/cnn_ablation.json.

Usage: python -m experiments.cnn_ablation
"""

import json
import os

import numpy as np

from acenet.data import CLASS_NAMES
from acenet.shift import KINDS, load_shifted_val
from acenet.tensor import Tensor, no_grad
from acenet.train import load_checkpoint
from experiments.cnn import EVAL_BATCH, ROOT, RUN_SEED, RUNS_DIR, _index, baseline, build_model, load_data

OUT = os.path.join(RUNS_DIR, "cnn_ablation.json")
ECE_BINS = 15
SHIFT_GROUPS = ("rotate", "noise", "brightness-mul", "brightness-add")


def softmax_probs(model, X, flip=False):
    """Softmax probabilities (float64) in eval mode; flip=True mirrors the inputs left-right."""
    model.eval()
    out = []
    with no_grad():
        for i in range(0, len(X), EVAL_BATCH):
            xb = X[i:i + EVAL_BATCH]
            if flip:
                xb = np.ascontiguousarray(xb[..., ::-1])
            z = model(Tensor(xb)).data.astype(np.float64)
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            out.append(p / p.sum(axis=1, keepdims=True))
    return np.concatenate(out)


def predict(model, X):
    """(plain probs, flip-TTA probs = mean of softmax(x) and softmax(hflip(x)))."""
    p = softmax_probs(model, X)
    return p, 0.5 * (p + softmax_probs(model, X, flip=True))


def ece(probs, y, bins=ECE_BINS):
    """Expected calibration error with equal-width confidence bins on the max probability."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    b = np.minimum((conf * bins).astype(np.int64), bins - 1)
    n = np.bincount(b, minlength=bins)
    acc_b = np.bincount(b, weights=correct, minlength=bins)
    conf_b = np.bincount(b, weights=conf, minlength=bins)
    return float(np.abs(acc_b - conf_b).sum() / len(y))


def confusion(pred, y, k=10):
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)


def per_class_acc(pred, y):
    cm = confusion(pred, y)
    return {CLASS_NAMES[c]: float(cm[c, c] / cm[c].sum()) for c in range(10)}


def shift_groups(kind, mode):
    groups = {}
    for k, name in enumerate(KINDS):
        if name == "brightness":
            groups["brightness-mul"] = (kind == k) & (mode == 0)
            groups["brightness-add"] = (kind == k) & (mode == 1)
        else:
            groups[name] = kind == k
    return {g: groups[g] for g in SHIFT_GROUPS}


def runs():
    idx = _index()
    a = baseline()
    by_tag = {e["tag"]: e for e in idx}
    return [("A", "baseline (Phase 3 selected)", [], a),
            ("B", "A + random erasing", ["random_erase"], by_tag["erase"]),
            ("C", "B + label smoothing 0.1", ["random_erase", "label_smoothing=0.1"], by_tag["erase_ls"])]


def load_model(entry):
    model = build_model(np.random.default_rng(RUN_SEED))
    epoch = load_checkpoint(model, os.path.join(ROOT, entry["run_dir"], "best.npz"))
    assert epoch == entry["best_epoch"]
    return model


def main():
    data = load_data()
    Xv, yv = data["X_val"], data["y_val"]
    Xs, ys, kind, _, mode = load_shifted_val(data["mean"], data["std"])
    groups = shift_groups(kind, mode)
    previous = json.load(open(OUT)) if os.path.exists(OUT) else {}

    rows = []
    for rid, desc, changes, e in runs():
        model = load_model(e)
        p, p_tta = predict(model, Xv)
        ps, ps_tta = predict(model, Xs)
        pred, pred_tta = p.argmax(1), p_tta.argmax(1)
        row = {
            "id": rid, "description": desc, "changes_vs_A": changes,
            "run_name": e["run_name"], "run_dir": e["run_dir"], "epochs": e["epochs"],
            "lr_schedule": e["lr_schedule"], "label_smoothing": e.get("label_smoothing", 0.0),
            "random_erase": bool(e.get("erase", False)),
            "best_epoch": e["best_epoch"], "wall_time_s": e["wall_time_s"],
            "val_acc": float((pred == yv).mean()),
            "val_acc_tta": float((pred_tta == yv).mean()),
            "val_nll": float(-np.log(p[np.arange(len(yv)), yv]).mean()),
            "val_nll_tta": float(-np.log(p_tta[np.arange(len(yv)), yv]).mean()),
            "ece": ece(p, yv), "ece_tta": ece(p_tta, yv),
            "per_class_val_acc": per_class_acc(pred, yv),
            "per_class_val_acc_tta": per_class_acc(pred_tta, yv),
            "confusion_val": confusion(pred, yv).tolist(),
            "confusion_val_tta": confusion(pred_tta, yv).tolist(),
            "shifted_val_acc": {g: float((ps.argmax(1)[m] == ys[m]).mean()) for g, m in groups.items()},
            "shifted_val_acc_tta": {g: float((ps_tta.argmax(1)[m] == ys[m]).mean()) for g, m in groups.items()},
            "shifted_val_acc_all": float((ps.argmax(1) == ys).mean()),
            "shifted_val_n": {g: int(m.sum()) for g, m in groups.items()},
        }
        assert abs(row["val_acc"] - e["best_val_acc"]) < 1e-9, "checkpoint does not reproduce logged val acc"
        rows.append(row)
        print(f"{rid}: val {row['val_acc']:.4f} tta {row['val_acc_tta']:.4f} ece {row['ece']:.4f} "
              f"shift {row['shifted_val_acc']}")

    # Selection by val accuracy over (run, TTA on/off). Ties go to the earlier run and to no-TTA.
    cands = [(r["val_acc"], -i, 0, r["id"], False) for i, r in enumerate(rows)]
    cands += [(r["val_acc_tta"], -i, -1, r["id"], True) for i, r in enumerate(rows)]
    best = max(cands)
    sel = {"id": best[3], "tta": best[4], "val_acc": best[0],
           "rule": "max val accuracy over A/B/C x {no TTA, flip TTA}; ties -> earlier run, then no TTA"}

    test = previous.get("test")
    if sel["id"] == "A":
        test = None
        sel["test_policy"] = "selected model is A: test not touched again (A's Phase 3 test stands)"
    elif test is not None:
        if (test["id"], test["tta"]) != (sel["id"], sel["tta"]):
            raise SystemExit(f"test already evaluated for {test['id']} (tta={test['tta']}); "
                             f"selection changed to {sel['id']} - refusing to touch test again")
        print("test already evaluated; not re-running")
    else:
        e = next(e for rid, _, _, e in runs() if rid == sel["id"])
        model = load_model(e)
        pt, pt_tta = predict(model, data["X_test"])
        probs = pt_tta if sel["tta"] else pt
        test = {"id": sel["id"], "tta": sel["tta"], "run_name": e["run_name"],
                "test_acc": float((probs.argmax(1) == data["y_test"]).mean()),
                "test_nll": float(-np.log(probs[np.arange(len(data["y_test"])), data["y_test"]]).mean())}
        sel["test_policy"] = "selected model is not A: evaluated on test once as the new final CNN"
        print(f"TEST (once): {test}")

    out = {"phase": "3b", "baseline": "A", "class_names": CLASS_NAMES, "shift_groups": list(SHIFT_GROUPS),
           "ece_bins": ECE_BINS, "tta": "mean of softmax(x) and softmax(hflip(x))",
           "rows": rows, "selected": sel, "test": test}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"selected: {sel}")


if __name__ == "__main__":
    main()
