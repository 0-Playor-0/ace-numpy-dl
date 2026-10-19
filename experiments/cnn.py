"""CNN on Fashion-MNIST with BN, dropout and crop/flip augmentation.

Usage:
  python -m experiments.cnn speed       # time forward+backward of one batch; project s/epoch
  python -m experiments.cnn train       # AdamW 1e-3, wd 5e-4, 20 epochs, constant lr
  python -m experiments.cnn retry       # only if best val < 92%: 25 epochs, lr x0.1 at epoch 15
  python -m experiments.cnn test        # pick best-val CNN run, evaluate test ONCE
  python -m experiments.cnn shift       # selected model on data/shifted_val.npz, by shift kind
  python -m experiments.cnn ablation    # Phase 3b: train B (A + erase) and C (B + label smoothing)
Every run is appended to runs/cnn_index.json.
"""

import argparse
import json
import os
import time

import numpy as np

from acenet import nn
from acenet.data import SHIFT_AUG, DataLoader, augment_batch, load_splits, train_augment
from acenet.optim import Adam, CosineWarmup
from acenet.shift import KINDS, load_shifted_val
from acenet.tensor import Tensor
from acenet import ops
from acenet.train import RUNS_DIR, evaluate, fit, load_checkpoint

DATA_SEED = 0   # fixes the train/val split (same split as data/shifted_val.npz)
RUN_SEED = 1    # init, dropout masks, batch order, augmentation
SHIFT_AUG_SEED = 7  # own stream for shift augmentation (shifted_val was built from seed 0)
BATCH = 128
EVAL_BATCH = 200
LR, WD = 1e-3, 5e-4
VAL_TARGET = 0.92
ROOT = os.path.dirname(RUNS_DIR)
INDEX = os.path.join(RUNS_DIR, "cnn_index.json")
TEST_RESULTS = os.path.join(RUNS_DIR, "cnn_test.json")
SHIFT_RESULTS = os.path.join(RUNS_DIR, "cnn_shift.json")
MAX_EPOCH_SECONDS = 180


def conv_bn_relu(cin, cout, rng):
    # Conv bias omitted: BN subtracts the batch mean, so a bias there has zero gradient.
    return [nn.Conv2D(cin, cout, 3, rng, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]


def build_model(rng):
    return nn.Sequential(
        *conv_bn_relu(1, 32, rng), *conv_bn_relu(32, 32, rng), nn.MaxPool2D(2),
        *conv_bn_relu(32, 64, rng), *conv_bn_relu(64, 64, rng), nn.MaxPool2D(2),
        nn.Flatten(), nn.Dropout(0.3, rng),
        nn.Linear(3136, 128, rng, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Dropout(0.3, rng), nn.Linear(128, 10, rng),
    )


def param_count(model):
    n = sum(p.data.size for p in model.parameters())
    buffers = sum(getattr(m, name).size for _, m, name in model.named_buffers())
    return n, buffers


def load_data():
    return load_splits(np.random.default_rng(DATA_SEED))


def black_fill(data):
    """Standardized value of a black (0) pixel: the padding value for random crops."""
    return float((0.0 - data["mean"]) / data["std"])


def speed(n_warmup=1, n_timed=5):
    data = load_data()
    rng = np.random.default_rng(RUN_SEED)
    model = build_model(rng)
    xb, yb = data["X_train"][:BATCH], data["y_train"][:BATCH]
    times = []
    for i in range(n_warmup + n_timed):
        t0 = time.perf_counter()
        loss = ops.softmax_cross_entropy(model(Tensor(xb)), yb)
        model.zero_grad()
        loss.backward()
        if i >= n_warmup:
            times.append(time.perf_counter() - t0)
    model.eval()
    va = DataLoader(data["X_val"][:2000], data["y_val"][:2000], EVAL_BATCH, shuffle=False)
    t0 = time.perf_counter()
    evaluate(model, va, track_relu=True)
    eval_per_sample = (time.perf_counter() - t0) / 2000
    t_aug = time.perf_counter()
    for _ in range(10):
        augment_batch(xb, rng, fill=black_fill(data))
    aug = (time.perf_counter() - t_aug) / 10
    ms = 1000 * float(np.median(times))
    steps = -(-len(data["y_train"]) // BATCH)
    projected = steps * (ms / 1000 + aug) + eval_per_sample * len(data["y_val"])
    n_params, n_buffers = param_count(model)
    res = {"ms_per_batch_fwd_bwd": ms, "ms_per_batch_all": [1000 * t for t in times],
           "ms_augment_per_batch": 1000 * aug, "steps_per_epoch": steps,
           "val_eval_s": eval_per_sample * len(data["y_val"]),
           "projected_s_per_epoch": projected, "params": n_params, "buffers": n_buffers}
    print(json.dumps(res, indent=2))
    with open(os.path.join(RUNS_DIR, "cnn_speed.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def _index():
    if os.path.exists(INDEX):
        with open(INDEX) as f:
            return json.load(f)
    return []


SCHEDULES = {
    "constant": None,
    "step: lr 1e-3 for epochs 1-14, 1e-4 from epoch 15": lambda ep: LR if ep < 15 else LR * 0.1,
}


def run(tag, epochs, lr_schedule=None, schedule_desc=None, erase=False, label_smoothing=0.0,
        lr=LR, warmup_epochs=None, shift_aug=False):
    """warmup_epochs (not None) switches to a per-batch CosineWarmup schedule to 0 over all epochs.
    shift_aug adds data.shift_augment with its own generator (SHIFT_AUG_SEED)."""
    data = load_data()
    rng = np.random.default_rng(RUN_SEED)
    model = build_model(rng)
    opt = Adam(model.parameters(), lr=lr, weight_decay=WD, decoupled=True)
    fill = black_fill(data)
    shift = (np.random.default_rng(SHIFT_AUG_SEED), data["mean"], data["std"]) if shift_aug else None
    tr = DataLoader(data["X_train"], data["y_train"], BATCH, shuffle=True, rng=rng,
                    transform=lambda x, r: train_augment(x, r, fill, erase=erase, shift=shift))
    scheduler = None
    if warmup_epochs is not None:
        scheduler = CosineWarmup(opt, warmup_epochs, epochs, len(tr), min_lr=0.0)
        schedule_desc = f"per-batch linear warmup {warmup_epochs} ep to {lr:g}, cosine to 0 at epoch {epochs}"
    va = DataLoader(data["X_val"], data["y_val"], EVAL_BATCH, shuffle=False)
    n_params, n_buffers = param_count(model)
    config = {"model": "cnn-32-32-64-64-fc128", "optimizer": "AdamW", "lr": lr, "weight_decay": WD,
              "adam_betas": [0.9, 0.999], "batch_size": BATCH, "dropout": 0.3,
              "augmentation": {"random_crop_pad": 2, "pad_fill_standardized": fill, "hflip_p": 0.5,
                               "random_erase": {"p": 0.5, "area": [0.02, 0.2], "aspect": [0.3, 3.3],
                                                "fill": "N(0,1)"} if erase else None,
                               "shift": dict(SHIFT_AUG, seed=SHIFT_AUG_SEED, order="rotate, noise, brightness",
                                             brightness_modes="50/50 mul/add") if shift_aug else None},
              "lr_schedule": schedule_desc, "data_seed": DATA_SEED, "tag": tag,
              "params": n_params, "buffers": n_buffers}
    name = f"cnn_{tag}"
    res = fit(model, opt, tr, va, epochs, lr_schedule=lr_schedule, run_name=name, config=config,
              seed=RUN_SEED, label_smoothing=label_smoothing, scheduler=scheduler)
    entry = {k: res[k] for k in ("run_name", "status", "epochs_run", "best_epoch",
                                 "best_val_acc", "best_val_loss", "wall_time_s")}
    entry.update(run_dir=os.path.relpath(res["run_dir"], ROOT), tag=tag, epochs=epochs,
                 lr_schedule=schedule_desc, erase=erase, label_smoothing=label_smoothing,
                 lr=lr, warmup_epochs=warmup_epochs, shift_aug=shift_aug)
    idx = _index()
    idx.append(entry)
    with open(INDEX, "w") as f:
        json.dump(idx, f, indent=2)
    return entry


def train():
    return run("const", 20, None, "constant")


def retry():
    base = next((e for e in _index() if e["tag"] == "const"), None)
    if base is None:
        raise SystemExit("run `train` first")
    if base["best_val_acc"] >= VAL_TARGET:
        print(f"best val {base['best_val_acc']:.4f} >= {VAL_TARGET}; no retry needed")
        return None
    desc = "step: lr 1e-3 for epochs 1-14, 1e-4 from epoch 15"
    return run("step", 25, SCHEDULES[desc], desc)


def baseline():
    """Baseline A for the Phase 3b ablation: the run selected (and test-evaluated) in Phase 3."""
    with open(TEST_RESULTS) as f:
        name = json.load(f)["run_name"]
    return next(e for e in _index() if e["run_name"] == name)


def ablation():
    """B = A + random erasing; C = B + label smoothing 0.1. Same epochs/schedule/seed as A."""
    a = baseline()
    done = {e["tag"] for e in _index()}
    for tag, ls in (("erase", 0.0), ("erase_ls", 0.1)):
        if tag in done:
            print(f"{tag} already trained; skipping")
            continue
        run(tag, a["epochs"], SCHEDULES[a["lr_schedule"]], a["lr_schedule"], erase=True, label_smoothing=ls)


def selected():
    candidates = [e for e in _index() if e["status"] == "ok" and e["tag"] in ("const", "step")]
    best = max(candidates, key=lambda e: e["best_val_acc"])
    model = build_model(np.random.default_rng(RUN_SEED))
    epoch = load_checkpoint(model, os.path.join(ROOT, best["run_dir"], "best.npz"))
    assert epoch == best["best_epoch"]
    return best, model


def test():
    """Evaluate the best-val CNN run's checkpoint on the test set, once."""
    if os.path.exists(TEST_RESULTS):
        with open(TEST_RESULTS) as f:
            print("test set already evaluated; not re-running:\n" + f.read())
        return
    data = load_data()
    best, model = selected()
    te = DataLoader(data["X_test"], data["y_test"], EVAL_BATCH, shuffle=False)
    res = {**best, **{f"test_{k}": v for k, v in evaluate(model, te).items()}}
    print(f"{best['run_name']} epoch {best['best_epoch']} val {best['best_val_acc']:.4f} "
          f"test {res['test_acc']:.4f}")
    with open(TEST_RESULTS, "w") as f:
        json.dump(res, f, indent=2)


def shift():
    """Selected model on clean val and on the shifted val set, broken down by shift kind."""
    data = load_data()
    best, model = selected()
    X, y, kind, sev, mode = load_shifted_val(data["mean"], data["std"])
    groups = {"clean": (data["X_val"], data["y_val"])}
    for k, name in enumerate(KINDS):
        if name == "brightness":
            for m, mname in ((0, "brightness-mul"), (1, "brightness-add")):
                sel = (kind == k) & (mode == m)
                groups[mname] = (X[sel], y[sel])
        else:
            groups[name] = (X[kind == k], y[kind == k])
    groups["shifted (all)"] = (X, y)
    res = {"model_run": best["run_name"], "model_epoch": best["best_epoch"], "groups": {}}
    for name, (Xg, yg) in groups.items():
        r = evaluate(model, DataLoader(Xg, yg, EVAL_BATCH, shuffle=False))
        res["groups"][name] = {"n": int(len(yg)), "acc": r["acc"], "loss": r["loss"]}
        print(f"{name:16s} n={len(yg):5d} acc {r['acc']:.4f} loss {r['loss']:.4f}")
    # rotation accuracy by |angle| bucket, noise by sigma bucket (finer view, still val data)
    for name, k, edges in (("rotate", 0, [0, 5, 10, 15]), ("noise", 1, [0.1, 0.15, 0.2, 0.25, 0.3])):
        mag = np.abs(sev)
        out = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (kind == k) & (mag >= lo) & (mag <= hi)
            r = evaluate(model, DataLoader(X[sel], y[sel], EVAL_BATCH, shuffle=False))
            out[f"{lo:g}-{hi:g}"] = {"n": int(sel.sum()), "acc": r["acc"]}
        res[f"{name}_by_severity"] = out
        print(name, {b: round(v["acc"], 4) for b, v in out.items()})
    with open(SHIFT_RESULTS, "w") as f:
        json.dump(res, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["speed", "train", "retry", "test", "shift", "ablation"])
    stage = ap.parse_args().stage
    if stage == "speed":
        speed()
    elif stage == "train":
        if speed()["projected_s_per_epoch"] > MAX_EPOCH_SECONDS:
            raise SystemExit("projected epoch time over budget; not training")
        train()
    elif stage == "retry":
        retry()
    elif stage == "test":
        test()
    elif stage == "ablation":
        ablation()
    else:
        shift()


if __name__ == "__main__":
    main()
