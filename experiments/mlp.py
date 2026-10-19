"""784-256-128-10 MLP on Fashion-MNIST: lr sweeps for SGD-momentum and Adam, optional regularization,
one-shot test evaluation of the chosen configs, and two deliberate failure runs.

Usage:
  python -m experiments.mlp sweep                      # 2 optimizers x 3 lrs, 40 epochs
  python -m experiments.mlp run --opt sgd --lr 0.03 --dropout 0.2 --wd 5e-4   # extra (regularized) run
  python -m experiments.mlp run --tag fix ...            # failure-fix run, never a test candidate
  python -m experiments.mlp failures                   # 60-epoch unregularized run + SGD lr=1.0
  python -m experiments.mlp test                       # pick best-val config per optimizer, test ONCE
Every run is appended to runs/mlp_index.json.
"""

import argparse
import json
import os

import numpy as np

from acenet import nn
from acenet.data import DataLoader, load_splits
from acenet.optim import SGD, Adam
from acenet.train import RUNS_DIR, evaluate, fit, load_checkpoint

DATA_SEED = 0   # fixes the train/val split (same split as data/shifted_val.npz)
RUN_SEED = 1    # model init, dropout masks, batch order
BATCH = 128
EPOCHS = 40
LR_GRID = {"sgd": [0.01, 0.03, 0.1], "adam": [3e-4, 1e-3, 3e-3]}
INDEX = os.path.join(RUNS_DIR, "mlp_index.json")
TEST_RESULTS = os.path.join(RUNS_DIR, "mlp_test.json")


def build_model(rng, dropout=0.0):
    layers = [nn.Linear(784, 256, rng), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout, rng))
    layers += [nn.Linear(256, 128, rng), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout, rng))
    layers.append(nn.Linear(128, 10, rng))
    return nn.Sequential(*layers)


def build_opt(name, params, lr, wd):
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=wd, decoupled=True)
    raise ValueError(name)


def _index():
    if os.path.exists(INDEX):
        with open(INDEX) as f:
            return json.load(f)
    return []


def run(opt_name, lr, epochs=EPOCHS, dropout=0.0, wd=0.0, tag="sweep", data=None):
    data = data or load_splits(np.random.default_rng(DATA_SEED), flatten=True)
    rng = np.random.default_rng(RUN_SEED)
    model = build_model(rng, dropout)
    opt = build_opt(opt_name, model.parameters(), lr, wd)
    config = {"model": "mlp-784-256-128-10", "optimizer": opt_name, "lr": lr,
              "momentum": 0.9 if opt_name == "sgd" else None,
              "adam_betas": [0.9, 0.999] if opt_name == "adam" else None,
              "weight_decay": wd, "weight_decay_kind": ("L2" if opt_name == "sgd" else "decoupled (AdamW)") if wd else None,
              "dropout": dropout, "batch_size": BATCH, "data_seed": DATA_SEED, "tag": tag,
              "lr_schedule": None}
    name = f"mlp_{tag}_{opt_name}_lr{lr:g}" + (f"_do{dropout:g}" if dropout else "") + (f"_wd{wd:g}" if wd else "")
    tr = DataLoader(data["X_train"], data["y_train"], BATCH, shuffle=True, rng=rng)
    va = DataLoader(data["X_val"], data["y_val"], 1000, shuffle=False)
    res = fit(model, opt, tr, va, epochs, run_name=name, config=config, seed=RUN_SEED)
    entry = {k: res[k] for k in ("run_name", "run_dir", "status", "epochs_run", "best_epoch",
                                 "best_val_acc", "best_val_loss", "wall_time_s")}
    entry["run_dir"] = os.path.relpath(res["run_dir"], os.path.dirname(RUNS_DIR))
    entry.update(optimizer=opt_name, lr=lr, dropout=dropout, weight_decay=wd, tag=tag, epochs=epochs)
    idx = _index()
    idx.append(entry)
    with open(INDEX, "w") as f:
        json.dump(idx, f, indent=2)
    return entry


def sweep():
    data = load_splits(np.random.default_rng(DATA_SEED), flatten=True)
    for opt_name, lrs in LR_GRID.items():
        for lr in lrs:
            run(opt_name, lr, data=data)


def failures():
    data = load_splits(np.random.default_rng(DATA_SEED), flatten=True)
    sweep_adam = [e for e in _index() if e["tag"] == "sweep" and e["optimizer"] == "adam"]
    best_lr = max(sweep_adam, key=lambda e: e["best_val_acc"])["lr"]
    run("adam", best_lr, epochs=60, tag="overfit", data=data)
    run("sgd", 1.0, epochs=EPOCHS, tag="diverge", data=data)


def test():
    """Evaluate the best-val config of each optimizer on the test set, once."""
    if os.path.exists(TEST_RESULTS):
        with open(TEST_RESULTS) as f:
            print("test set already evaluated; not re-running:\n" + f.read())
        return
    data = load_splits(np.random.default_rng(DATA_SEED), flatten=True)
    te = DataLoader(data["X_test"], data["y_test"], 1000, shuffle=False)
    candidates = [e for e in _index() if e["tag"] in ("sweep", "reg") and e["status"] == "ok"]
    results = {}
    for opt_name in LR_GRID:
        best = max((e for e in candidates if e["optimizer"] == opt_name), key=lambda e: e["best_val_acc"])
        model = build_model(np.random.default_rng(RUN_SEED), best["dropout"])
        epoch = load_checkpoint(model, os.path.join(os.path.dirname(RUNS_DIR), best["run_dir"], "best.npz"))
        assert epoch == best["best_epoch"]
        results[opt_name] = {**best, **{f"test_{k}": v for k, v in evaluate(model, te).items()}}
        print(f"{opt_name}: {best['run_name']} epoch {epoch} val {best['best_val_acc']:.4f} "
              f"test {results[opt_name]['test_acc']:.4f}")
    with open(TEST_RESULTS, "w") as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "run", "failures", "test"])
    ap.add_argument("--opt", choices=list(LR_GRID))
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--tag", choices=["reg", "fix"], default="reg",
                    help="reg runs are test-selection candidates; fix runs (failure write-up) are not")
    args = ap.parse_args()
    if args.stage == "sweep":
        sweep()
    elif args.stage == "run":
        run(args.opt, args.lr, args.epochs, args.dropout, args.wd, tag=args.tag)
    elif args.stage == "failures":
        failures()
    else:
        test()


if __name__ == "__main__":
    main()
