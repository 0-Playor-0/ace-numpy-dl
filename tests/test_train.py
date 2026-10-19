import json
import os

import numpy as np

from acenet import nn
from acenet.data import DataLoader
from acenet.optim import SGD
from acenet.train import evaluate, fit, load_checkpoint


def _setup(rng):
    X = rng.standard_normal((256, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    model = nn.Sequential(nn.Linear(10, 16, rng), nn.ReLU(), nn.Linear(16, 2, rng))
    tr = DataLoader(X[:192], y[:192], 32, shuffle=True, rng=rng)
    va = DataLoader(X[192:], y[192:], 64, shuffle=False)
    return model, tr, va


def test_fit_logs_and_restores_best(tmp_path):
    rng = np.random.default_rng(0)
    model, tr, va = _setup(rng)
    res = fit(model, SGD(model.parameters(), lr=0.1, momentum=0.9), tr, va, epochs=5,
              run_name="toy", seed=0, runs_dir=str(tmp_path), verbose=False)
    files = set(os.listdir(res["run_dir"]))
    assert {"config.json", "seed.txt", "metrics.csv", "curves.png", "best.npz", "summary.json"} <= files
    assert res["status"] == "ok" and len(res["rows"]) == 5
    assert json.load(open(os.path.join(res["run_dir"], "config.json")))["seed"] == 0
    assert evaluate(model, va)["acc"] == res["best_val_acc"]
    assert 0 <= res["rows"][-1]["dead_relu"] <= 1
    assert load_checkpoint(model, os.path.join(res["run_dir"], "best.npz")) == res["best_epoch"]


def test_fit_detects_divergence(tmp_path):
    rng = np.random.default_rng(0)
    model, tr, va = _setup(rng)
    res = fit(model, SGD(model.parameters(), lr=1e4, momentum=0.9), tr, va, epochs=20,
              run_name="boom", seed=0, runs_dir=str(tmp_path), verbose=False)
    assert res["status"] == "diverged" and res["rows"][-1]["status"] == "diverged"
