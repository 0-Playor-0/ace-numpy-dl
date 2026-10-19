"""Phase 2 figures from runs/mlp_index.json and each run's metrics.csv -> experiments/figures/."""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "experiments", "figures")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OPT_COLOR = {"sgd": SERIES[0], "adam": SERIES[1]}
OPT_LABEL = {"sgd": "SGD momentum 0.9", "adam": "Adam"}

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "font.size": 10, "figure.facecolor": "white",
})


def load_index():
    with open(os.path.join(ROOT, "runs", "mlp_index.json")) as f:
        return json.load(f)


def metrics(entry):
    with open(os.path.join(ROOT, entry["run_dir"], "metrics.csv")) as f:
        return [{k: (float(v) if k not in ("status", "dead_relu_layers") and v != "" else v)
                 for k, v in row.items()} for row in csv.DictReader(f)]


def col(rows, key):
    return [r["epoch"] for r in rows], [r[key] for r in rows]


def chosen_runs(index):
    with open(os.path.join(ROOT, "runs", "mlp_test.json")) as f:
        test = json.load(f)
    by_name = {e["run_name"]: e for e in index}
    return {opt: (by_name[r["run_name"]], r) for opt, r in test.items()}


def fig_sgd_vs_adam(index):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for opt, (entry, res) in chosen_runs(index).items():
        rows = metrics(entry)
        c = OPT_COLOR[opt]
        name = f"{OPT_LABEL[opt]}, lr {entry['lr']:g}"
        for ax, key in zip(axes, ("loss", "acc")):
            ax.plot(*col(rows, f"train_{key}"), color=c, linestyle="--", alpha=0.85,
                    label=f"{name} (train)")
            ax.plot(*col(rows, f"val_{key}"), color=c, label=f"{name} (val)")
        axes[1].plot([entry["best_epoch"]], [entry["best_val_acc"]], "o", ms=8, color=c,
                     markeredgecolor="white", markeredgewidth=2)
        axes[1].annotate(f"best val {entry['best_val_acc']:.4f}\n(ep {entry['best_epoch']}, "
                         f"test {res['test_acc']:.4f})", (entry["best_epoch"], entry["best_val_acc"]),
                         xytext=(0, -38 if opt == "sgd" else 14), textcoords="offset points",
                         ha="center", fontsize=8, color=MUTED)
    axes[0].set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0.84, 1.0))
    axes[0].legend(fontsize=8)
    fig.suptitle("Selected MLP configs: SGD vs Adam (solid = val, dashed = train)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "mlp_sgd_vs_adam.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_overfit(index):
    entry = next(e for e in index if e["tag"] == "overfit")
    rows = metrics(entry)
    best = entry["best_epoch"]
    ep, vl = col(rows, "val_loss")
    min_ep = int(ep[vl.index(min(vl))])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key in zip(axes, ("loss", "acc")):
        ax.plot(*col(rows, f"train_{key}"), color=SERIES[0], label="train")
        ax.plot(*col(rows, f"val_{key}"), color=SERIES[1], label="val")
        ax.axvline(best, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlabel("epoch")
    axes[0].axvline(min_ep, color=MUTED, linewidth=1, linestyle=":")
    axes[0].text(min_ep + 0.6, 0.55, f"min val loss\nep {min_ep} ({min(vl):.3f})", fontsize=8, color=MUTED)
    axes[0].text(best + 0.6, 0.35, f"best-val-acc\ncheckpoint ep {best}", fontsize=8, color=MUTED)
    axes[1].text(best + 0.6, 0.86, f"checkpoint ep {best}\nval acc {entry['best_val_acc']:.4f}",
                 fontsize=8, color=MUTED)
    axes[0].set(title="Loss", ylabel="cross-entropy")
    axes[1].set(title="Accuracy", ylabel="accuracy", ylim=(0.8, 1.0))
    axes[0].legend()
    fig.suptitle(f"Overfitting run: Adam lr {entry['lr']:g}, no regularization, {entry['epochs']} epochs")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "mlp_overfit.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_overfit_fix(index):
    before = next(e for e in index if e["tag"] == "overfit")
    after = next(e for e in index if e["tag"] == "fix")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for e, c, name in ((before, SERIES[0], "no regularization"),
                       (after, SERIES[1], f"dropout {after['dropout']:g} + AdamW wd {after['weight_decay']:g}")):
        rows = metrics(e)
        for ax, key in zip(axes, ("loss", "acc")):
            ax.plot(*col(rows, f"train_{key}"), color=c, linestyle="--", alpha=0.85, label=f"{name} (train)")
            ax.plot(*col(rows, f"val_{key}"), color=c, label=f"{name} (val)")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0.8, 1.0))
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Overfitting fix, Adam lr {before['lr']:g}, 60 epochs (solid = val, dashed = train)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "mlp_overfit_fix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_dead_relu(index):
    panels = [("SGD momentum lr sweep", [e for e in index if e["tag"] == "sweep" and e["optimizer"] == "sgd"]),
              ("Adam lr sweep", [e for e in index if e["tag"] == "sweep" and e["optimizer"] == "adam"]),
              ("Failure / fix runs", [e for e in index if e["tag"] in ("overfit", "diverge", "fix", "reg")])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (title, entries) in zip(axes, panels):
        for i, e in enumerate(entries):
            rows = metrics(e)
            label = f"lr {e['lr']:g}" if e["tag"] == "sweep" else f"{e['tag']}: {e['optimizer']} lr {e['lr']:g}"
            if e["status"] == "diverged":
                ax.plot([], [], " ", label=f"{label} — diverged ep 1, no val pass")
                continue
            ax.plot(*col(rows, "dead_relu"), color=SERIES[i % 3], label=label)
        ax.set(title=title, xlabel="epoch")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fraction of hidden units dead on all of val")
    fig.suptitle("Dead-ReLU fraction (units with zero activation on the entire val set), both hidden layers")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "mlp_dead_relu.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    index = load_index()
    for fn in (fig_sgd_vs_adam, fig_overfit, fig_overfit_fix, fig_dead_relu):
        print("wrote", os.path.relpath(fn(index), ROOT))


if __name__ == "__main__":
    main()
