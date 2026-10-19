"""Training loop, evaluation and run logging.

Each fit() writes runs/<name>_<timestamp>/ with config.json (including the seed), seed.txt,
metrics.csv (one row per epoch), curves.png, best.npz (best-val checkpoint), last.npz (final
weights, when the run did not diverge) and summary.json.
"""

import csv
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import ops
from .nn import ReLU
from .tensor import Tensor, no_grad

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def evaluate(model, loader, track_relu=False):
    """Mean loss and accuracy over loader under no_grad + eval().

    With track_relu, also reports the fraction of hidden ReLU units whose activation was zero
    for every sample in loader: overall ("dead_relu") and per layer ("dead_relu_layers").
    """
    was_training = model.training
    relus = [m for m in model.modules() if isinstance(m, ReLU)]
    for r in relus:
        r.tracking, r.alive = track_relu, None
    model.eval()
    total, loss_sum, correct = 0, 0.0, 0
    try:
        with no_grad(), np.errstate(over="ignore", invalid="ignore"):
            for xb, yb in loader:
                logits = model(Tensor(xb))
                n = len(yb)
                total += n
                loss_sum += float(ops.softmax_cross_entropy(logits, yb).data) * n
                correct += int((logits.data.argmax(axis=1) == yb).sum())
    finally:
        model.train(was_training)
    out = {"loss": loss_sum / total, "acc": correct / total}
    if track_relu:
        dead = [int((~r.alive).sum()) for r in relus]
        widths = [r.alive.size for r in relus]
        out["dead_relu"] = sum(dead) / sum(widths) if relus else 0.0
        out["dead_relu_layers"] = [d / w for d, w in zip(dead, widths)]
        for r in relus:
            r.tracking, r.alive = False, None
    return out


def grad_global_norm(params):
    return math.sqrt(sum(float(np.sum(p.grad.astype(np.float64) ** 2))
                         for p in params if p.grad is not None))


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_curves(rows, path, title):
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].plot(ep, [r["train_loss"] for r in rows], label="train")
    ax[0].plot(ep, [r["val_loss"] for r in rows], label="val")
    ax[0].set_title("loss")
    ax[1].plot(ep, [r["train_acc"] for r in rows], label="train")
    ax[1].plot(ep, [r["val_acc"] for r in rows], label="val")
    ax[1].set_title("accuracy")
    ax[2].plot(ep, [r["dead_relu"] for r in rows])
    ax[2].set_title("dead-ReLU fraction (val)")
    ax[3].plot(ep, [r["grad_norm_mean"] for r in rows], label="mean")
    ax[3].plot(ep, [r["grad_norm_max"] for r in rows], label="max")
    ax[3].set_yscale("log")
    ax[3].set_title("grad global norm")
    for a in ax:
        a.set_xlabel("epoch")
    for a in (ax[0], ax[1], ax[3]):
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fit(model, opt, train_loader, val_loader, epochs, lr_schedule=None, run_name="run",
        config=None, seed=None, runs_dir=RUNS_DIR, verbose=True):
    """Train with softmax cross-entropy; validate after every epoch.

    lr_schedule: optional callable epoch (1-based) -> lr, applied at the start of each epoch.
    A non-finite training loss aborts the run with status "diverged".
    On return the model holds the best-val-accuracy weights (also saved to best.npz).
    """
    run_dir = os.path.join(runs_dir, f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir)
    config = dict(config or {}, run_name=run_name, seed=seed, epochs=epochs)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "seed.txt"), "w") as f:
        f.write(f"{seed}\n")

    params = model.parameters()
    rows, best, status, diverged_at = [], None, "ok", None
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        if lr_schedule is not None:
            opt.lr = lr_schedule(epoch)
        model.train()
        n_seen, loss_sum, correct, norms = 0, 0.0, 0, []
        with np.errstate(over="ignore", invalid="ignore"):
            for step, (xb, yb) in enumerate(train_loader, 1):
                logits = model(Tensor(xb))
                loss = ops.softmax_cross_entropy(logits, yb)
                lv = float(loss.data)
                if not math.isfinite(lv):
                    status, diverged_at = "diverged", {"epoch": epoch, "step": step, "loss": lv}
                    break
                opt.zero_grad()
                loss.backward()
                norms.append(grad_global_norm(params))
                opt.step()
                n = len(yb)
                n_seen += n
                loss_sum += lv * n
                correct += int((logits.data.argmax(axis=1) == yb).sum())
        if status == "diverged":
            rows.append({"epoch": epoch, "lr": opt.lr, "train_loss": float("nan"),
                         "train_acc": float("nan"), "val_loss": float("nan"),
                         "val_acc": float("nan"), "dead_relu": float("nan"),
                         "dead_relu_layers": "", "grad_norm_mean": float(np.mean(norms)) if norms else float("nan"),
                         "grad_norm_max": max(norms) if norms else float("nan"),
                         "epoch_time": time.perf_counter() - t0, "status": "diverged"})
            if verbose:
                print(f"[{run_name}] diverged at epoch {epoch} step {step}: loss={lv}")
            break
        val = evaluate(model, val_loader, track_relu=True)
        row = {"epoch": epoch, "lr": opt.lr, "train_loss": loss_sum / n_seen,
               "train_acc": correct / n_seen, "val_loss": val["loss"], "val_acc": val["acc"],
               "dead_relu": val["dead_relu"],
               "dead_relu_layers": ";".join(f"{d:.4f}" for d in val["dead_relu_layers"]),
               "grad_norm_mean": float(np.mean(norms)), "grad_norm_max": max(norms),
               "epoch_time": time.perf_counter() - t0, "status": "ok"}
        rows.append(row)
        if not math.isfinite(val["loss"]):
            status, diverged_at = "diverged", {"epoch": epoch, "step": None, "loss": val["loss"]}
            row["status"] = "diverged"
            break
        if best is None or val["acc"] > best["val_acc"]:
            best = {"epoch": epoch, "val_acc": val["acc"], "val_loss": val["loss"]}
            state = model.state_dict()
            np.savez(os.path.join(run_dir, "best.npz"), epoch=epoch, **state)
            best_state = state
        if verbose:
            print(f"[{run_name}] ep {epoch:3d} lr {opt.lr:.2e} train {row['train_loss']:.4f}/"
                  f"{row['train_acc']:.4f} val {val['loss']:.4f}/{val['acc']:.4f} "
                  f"dead {val['dead_relu']:.3f} |g| {row['grad_norm_mean']:.3f} "
                  f"({row['epoch_time']:.1f}s)")
        _write_csv(os.path.join(run_dir, "metrics.csv"), rows)
    wall = time.perf_counter() - t_start

    _write_csv(os.path.join(run_dir, "metrics.csv"), rows)
    plot_curves(rows, os.path.join(run_dir, "curves.png"), f"{run_name} ({status})")
    if status == "ok":
        np.savez(os.path.join(run_dir, "last.npz"), epoch=len(rows), **model.state_dict())
    if best is not None:
        model.load_state_dict(best_state)
    summary = {"run_name": run_name, "run_dir": os.path.relpath(run_dir, os.path.dirname(runs_dir)),
               "status": status, "diverged_at": diverged_at,
               "epochs_run": len(rows), "best_epoch": best and best["epoch"],
               "best_val_acc": best and best["val_acc"], "best_val_loss": best and best["val_loss"],
               "wall_time_s": wall}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return {**summary, "rows": rows, "run_dir": run_dir}


def load_checkpoint(model, path):
    with np.load(path) as d:
        model.load_state_dict({k: d[k] for k in d.files if k != "epoch"})
        return int(d["epoch"])
