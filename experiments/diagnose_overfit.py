"""Why does val loss climb while val accuracy stays flat? Re-run the overfit and fix configs
(identical seeds), check they reproduce the logged metrics, then compare the best-val and final
checkpoints on the validation set. Writes runs/mlp_overfit_diagnosis.json. Val only; no test data.
"""

import csv
import json
import os

import numpy as np

from acenet import ops
from acenet.data import load_splits
from acenet.tensor import Tensor, no_grad
from acenet.train import RUNS_DIR, load_checkpoint
from experiments.mlp import DATA_SEED, RUN_SEED, _index, build_model, run

ROOT = os.path.dirname(RUNS_DIR)


def val_stats(model, X, y):
    model.eval()
    with no_grad():
        logits = model(Tensor(X)).data.astype(np.float64)
    logp = ops.log_softmax(Tensor(logits)).data
    p = np.exp(logp)
    nll = -logp[np.arange(len(y)), y]
    pred = logits.argmax(1)
    wrong = pred != y
    conf = p.max(1)
    weights = [v for k, v in model.state_dict().items() if k.endswith(".W")]
    return {
        "val_acc": float((~wrong).mean()),
        "val_loss": float(nll.mean()),
        "n_wrong": int(wrong.sum()),
        "loss_share_from_wrong": float(nll[wrong].sum() / nll.sum()),
        "mean_nll_correct": float(nll[~wrong].mean()),
        "mean_nll_wrong": float(nll[wrong].mean()),
        "mean_conf_wrong": float(conf[wrong].mean()),
        "frac_wrong_conf_gt_0.99": float((conf[wrong] > 0.99).mean()),
        "mean_logit_range": float((logits.max(1) - logits.min(1)).mean()),
        "weight_l2_norm": float(np.sqrt(sum((w.astype(np.float64) ** 2).sum() for w in weights))),
    }


def metrics(run_dir):
    with open(os.path.join(ROOT, run_dir, "metrics.csv")) as f:
        return [(float(r["train_loss"]), float(r["val_loss"]), float(r["val_acc"])) for r in csv.DictReader(f)]


def main():
    data = load_splits(np.random.default_rng(DATA_SEED), flatten=True)
    idx = _index()
    out = {}
    for tag in ("overfit", "fix"):
        orig = next(e for e in idx if e["tag"] == tag)
        new = run(orig["optimizer"], orig["lr"], orig["epochs"], orig["dropout"], orig["weight_decay"],
                  tag=f"diag-{tag}", data=data)
        reproduced = metrics(orig["run_dir"]) == metrics(new["run_dir"])
        res = {"original_run": orig["run_dir"], "rerun": new["run_dir"], "metrics_reproduced_exactly": reproduced}
        for ckpt in ("best", "last"):
            model = build_model(np.random.default_rng(RUN_SEED), orig["dropout"])
            epoch = load_checkpoint(model, os.path.join(ROOT, new["run_dir"], f"{ckpt}.npz"))
            res[ckpt] = {"epoch": epoch, **val_stats(model, data["X_val"], data["y_val"])}
        out[tag] = res
        print(json.dumps(res, indent=2))
    with open(os.path.join(RUNS_DIR, "mlp_overfit_diagnosis.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
