"""Phase 4: post-training int8 quantization of the final CNN B (runs/cnn_ablation.json, id B).

Steps: BN folding check (val) -> calibrate activation scales on 2,000 TRAIN images with min/max and
99.99th-percentile |x| -> integer engine + fake-quant on val (both calibrators, with/without flip
TTA) -> pick calibrator by int8 val acc with TTA (the deployed config) -> per-layer sensitivity
(fake-quant, one layer at a time) -> shifted val -> test ONCE (guarded) -> serialize to
runs/cnn_b_int8.npz -> time 10k images. Writes runs/cnn_b_int8.json.

Phase 6a reuses the same pipeline for model D: main("D") -> runs/cnn_d_int8.{json,npz}.

Usage: python -m experiments.quant [B|D]
"""

import json
import os
import sys
import time

import numpy as np

from acenet.quant import CALIBRATORS, Int8Model, batched, build_qspec, calibrate, float_forward, fold_bn
from acenet.shift import load_shifted_val
from acenet.tensor import Tensor, no_grad
from experiments.cnn import EVAL_BATCH, RUNS_DIR, load_data
from experiments.cnn_ablation import entry as run_entry, load_model, shift_groups

OUT = os.path.join(RUNS_DIR, "cnn_b_int8.json")
NPZ = os.path.join(RUNS_DIR, "cnn_b_int8.npz")
CALIB_SEED = 4   # picks the 2,000 calibration images from the train split
N_CALIB = 2000
N_TIME = 10_000
FLOAT_B_TEST = 0.9349


def acc(z, y):
    return float((z.argmax(1) == y).mean())


def flip(X):
    return np.ascontiguousarray(X[..., ::-1])


def paths(model_id):
    if model_id == "B":
        return OUT, NPZ
    stem = os.path.join(RUNS_DIR, f"cnn_{model_id.lower()}_int8")
    return stem + ".json", stem + ".npz"


def main(model_id="B", float_test=FLOAT_B_TEST, calibrators=CALIBRATORS):
    """float_test: the model's float flip-TTA test accuracy (already evaluated once) for the drop.
    calibrators: candidates; Phase 6a passes ("p99.99",) only (fixed from Phase 4)."""
    out_path, npz_path = paths(model_id)
    previous = json.load(open(out_path)) if os.path.exists(out_path) else {}
    data = load_data()
    Xv, yv = data["X_val"], data["y_val"]
    entry = run_entry(model_id)
    model = load_model(entry)
    model.eval()
    abl_file = "cnn_ablation.json" if model_id == "B" else "ablation.json"
    abl = next(r for r in json.load(open(os.path.join(RUNS_DIR, abl_file)))["rows"] if r["id"] == model_id)
    res = {"phase": 4 if model_id == "B" else "6a", "model": model_id, "run_dir": entry["run_dir"],
           "calib_seed": CALIB_SEED, "n_calib": N_CALIB, "calibrators_tried": list(calibrators)}

    # 1. BN folding
    ops = fold_bn(model)
    with no_grad():
        z_ref = np.concatenate([model(Tensor(Xv[i:i + EVAL_BATCH])).data for i in range(0, len(Xv), EVAL_BATCH)])
    z_fold = batched(lambda xb: float_forward(ops, xb), Xv)
    z_fold_f = batched(lambda xb: float_forward(ops, xb), flip(Xv))
    res["folding"] = {"max_abs_logit_diff": float(np.abs(z_fold - z_ref).max()),
                      "pred_identical": bool((z_fold.argmax(1) == z_ref.argmax(1)).all()),
                      "val_acc_original": acc(z_ref, yv), "val_acc_folded": acc(z_fold, yv)}
    assert res["folding"]["pred_identical"] and res["folding"]["max_abs_logit_diff"] < 1e-4, res["folding"]
    res["float_val"] = {"acc": acc(z_fold, yv), "acc_tta_logit_sum": acc(z_fold + z_fold_f, yv),
                        "acc_tta_softmax_mean_phase3b": abl["val_acc_tta"]}
    print("folding", res["folding"], "float", res["float_val"])

    # 2-4. calibrate, build, evaluate integer engine and fake-quant on val
    rng = np.random.default_rng(CALIB_SEED)
    Xc = data["X_train"][rng.choice(len(data["X_train"]), N_CALIB, replace=False)]
    res["calibrators"], specs, engines, val_logits = {}, {}, {}, {}
    for method in calibrators:
        scales = calibrate(ops, Xc, method)
        qspec = build_qspec(ops, scales)
        qm = Int8Model.from_qspec(ops, qspec)
        t0 = time.perf_counter()
        zi, zi_f = qm.logits(Xv), qm.logits(flip(Xv))
        t_val = time.perf_counter() - t0
        zf = batched(lambda xb: float_forward(ops, xb, qspec=qspec), Xv)
        zf_f = batched(lambda xb: float_forward(ops, xb, qspec=qspec), flip(Xv))
        r = {"act_scales": scales,
             "val_acc": acc(zi, yv), "val_acc_tta": acc(zi + zi_f, yv),
             "fake_val_acc": acc(zf, yv), "fake_val_acc_tta": acc(zf + zf_f, yv),
             "agree_int_vs_fake": float((zi.argmax(1) == zf.argmax(1)).mean()),
             "agree_int_vs_fake_tta": float(((zi + zi_f).argmax(1) == (zf + zf_f).argmax(1)).mean()),
             "agree_int_vs_float": float((zi.argmax(1) == z_fold.argmax(1)).mean()),
             "multiplier_range": [float(min(q["M"].min() for q in qspec)), float(max(q["M"].max() for q in qspec))],
             "engine_s_12k_val_images": t_val}
        res["calibrators"][method] = r
        specs[method], engines[method], val_logits[method] = qspec, qm, (zi, zi_f)
        print(method, {k: v for k, v in r.items() if k != "act_scales"})

    # selection on val: int8 acc with TTA, then without, ties -> minmax (CALIBRATORS order)
    sel = max(calibrators, key=lambda m: (res["calibrators"][m]["val_acc_tta"], res["calibrators"][m]["val_acc"],
                                          -calibrators.index(m)))
    res["selected_calibrator"] = sel
    qspec, qm = specs[sel], engines[sel]

    # 5. per-layer sensitivity (fake-quant, one compute layer quantized, rest float)
    names = ["conv1", "conv2", "conv3", "conv4", "fc1", "fc2"]
    base = res["float_val"]["acc"]
    res["sensitivity"] = {}
    for i, name in enumerate(names):
        a = acc(batched(lambda xb: float_forward(ops, xb, qspec=qspec, quantized={i}), Xv), yv)
        res["sensitivity"][name] = {"val_acc": a, "drop": base - a}
    print("sensitivity", res["sensitivity"])

    # 6b. shifted val by kind (int8 selected vs float B from Phase 3b)
    Xs, ys, kind, _, mode = load_shifted_val(data["mean"], data["std"])
    zs, zs_f = qm.logits(Xs), qm.logits(flip(Xs))
    res["shifted_val"] = {g: {"n": int(m.sum()), "int8": acc(zs[m], ys[m]), "int8_tta": acc((zs + zs_f)[m], ys[m]),
                              "float": abl["shifted_val_acc"][g], "float_tta": abl["shifted_val_acc_tta"][g]}
                          for g, m in shift_groups(kind, mode).items()}
    print("shifted", res["shifted_val"])

    # 6c. test ONCE: selected calibrator, flip TTA
    test = previous.get("test")
    if test is not None:
        if test["calibrator"] != sel:
            raise SystemExit(f"test already evaluated with {test['calibrator']}; selection now {sel} - refusing")
        print("test already evaluated; not re-running")
    else:
        zt = qm.logits(data["X_test"], flip=True)
        ta = acc(zt, data["y_test"])
        test = {"calibrator": sel, "tta": True, "test_acc": ta, "float_test_tta": float_test,
                "drop": float_test - ta}
        print("TEST (once)", test)
    res["test"] = test
    with open(out_path, "w") as f:  # persist the test result (and its guard) immediately
        json.dump(res, f, indent=2)

    # 7. serialization and size
    qm.save(npz_path)
    arrays = qm.arrays()
    layers = {}
    for i, name in enumerate(names):
        b = {k: int(arrays[f"l{i}_{k}"].nbytes) for k in ("w", "b", "M0", "shift")}
        n_params = int(qspec[i]["w"].size + qspec[i]["b"].size)
        layers[name] = {**b, "total": sum(b.values()), "float32_params_bytes": 4 * n_params}
    payload = sum(v["total"] for v in layers.values()) + 4
    file_bytes = os.path.getsize(npz_path)
    res["size"] = {"file_bytes": file_bytes, "file_kib": file_bytes / 1024, "payload_bytes": payload,
                   "npz_overhead_bytes": file_bytes - payload, "layers": layers,
                   "float32_kib": 1828.7, "ratio_vs_float32": 1828.7 * 1024 / file_bytes}
    qm2 = Int8Model.load(npz_path)
    assert np.array_equal(qm2.logits(Xv[:1000]), val_logits[sel][0][:1000]), "npz round trip mismatch"

    # timing: 10k TRAIN images (test is not touched for timing), no TTA, includes input quantization
    Xt = data["X_train"][:N_TIME]
    qm2.logits(Xt[:200])
    t0 = time.perf_counter()
    qm2.logits(Xt)
    t = time.perf_counter() - t0
    t0 = time.perf_counter()
    batched(lambda xb: float_forward(ops, xb), Xt)
    tf = time.perf_counter() - t0
    res["timing"] = {"n": N_TIME, "batch": 200, "int8_s": t, "int8_ms_per_image": 1000 * t / N_TIME,
                     "float32_folded_numpy_s": tf}
    print("size", res["size"], "timing", res["timing"])
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["B"]))
