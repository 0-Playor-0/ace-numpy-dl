"""Figures from runs/{mlp,cnn}_index.json and each run's metrics.csv -> experiments/figures/.

Usage: python -m experiments.plots [mlp|cnn|ablation|all]
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def load_index(name="mlp"):
    with open(os.path.join(ROOT, "runs", f"{name}_index.json")) as f:
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


def fig_cnn_curves(index):
    with open(os.path.join(ROOT, "runs", "cnn_test.json")) as f:
        chosen = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, e in enumerate(index):
        rows = metrics(e)
        c = SERIES[i]
        name = f"{e['tag']} ({e['lr_schedule']})"
        for ax, key in zip(axes, ("loss", "acc")):
            ax.plot(*col(rows, f"train_{key}"), color=c, linestyle="--", alpha=0.85, label=f"{name}, train")
            ax.plot(*col(rows, f"val_{key}"), color=c, label=f"{name}, val")
        if e["run_name"] == chosen["run_name"]:
            axes[1].plot([e["best_epoch"]], [e["best_val_acc"]], "o", ms=8, color=c,
                         markeredgecolor="white", markeredgewidth=2)
            axes[1].annotate(f"selected: ep {e['best_epoch']}, val {e['best_val_acc']:.4f}, "
                             f"test {chosen['test_acc']:.4f}", (e["best_epoch"], e["best_val_acc"]),
                             xytext=(0, -60), textcoords="offset points", ha="right", fontsize=8, color=MUTED,
                             arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    axes[0].set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle("CNN, AdamW lr 1e-3 wd 5e-4 (train = running avg with augmentation + dropout)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cnn_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _bars(ax, labels, accs, ns, title, color):
    xs = range(len(labels))
    ax.bar(xs, accs, width=0.6, color=color, edgecolor="white", linewidth=2)
    for x, a, n in zip(xs, accs, ns):
        ax.text(x, a + 0.008, f"{a:.3f}", ha="center", fontsize=9, color=INK)
        ax.text(x, 0.015, f"n={n}", ha="center", fontsize=8, color="white")
    ax.set_xticks(list(xs), labels, fontsize=9)
    ax.set(title=title, ylim=(0, 1.05))
    ax.grid(axis="x", visible=False)


def fig_cnn_shift(_index=None):
    with open(os.path.join(ROOT, "runs", "cnn_shift.json")) as f:
        res = json.load(f)
    g = res["groups"]
    kinds = ["clean", "rotate", "noise", "brightness-mul", "brightness-add"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [5, 3, 4]})
    _bars(axes[0], ["clean val", "rotate\n±15°", "noise\nσ 0.1–0.3", "brightness\n×(1±0.3)",
                    "brightness\n±0.2"], [g[k]["acc"] for k in kinds], [g[k]["n"] for k in kinds],
          "By shift kind", SERIES[0])
    axes[0].axhline(g["clean"]["acc"], color=MUTED, linewidth=1, linestyle="--")
    axes[0].set_ylabel("accuracy")
    rot, noi = res["rotate_by_severity"], res["noise_by_severity"]
    _bars(axes[1], [f"|θ| {b}°" for b in rot], [v["acc"] for v in rot.values()],
          [v["n"] for v in rot.values()], "Rotation by |angle|", SERIES[0])
    _bars(axes[2], [f"σ {b}" for b in noi], [v["acc"] for v in noi.values()],
          [v["n"] for v in noi.values()], "Gaussian noise by σ", SERIES[0])
    for ax in axes[1:]:
        ax.axhline(g["clean"]["acc"], color=MUTED, linewidth=1, linestyle="--")
    fig.suptitle(f"CNN ({res['model_run']}, epoch {res['model_epoch']}) on shifted val "
                 f"(dashed = clean val {g['clean']['acc']:.3f})")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cnn_shift_accuracy.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


ABL_COLOR = {"A": SERIES[0], "B": SERIES[1], "C": SERIES[2]}
HIGHLIGHT = ("T-shirt/top", "Pullover", "Coat", "Shirt")


def load_ablation():
    with open(os.path.join(ROOT, "runs", "cnn_ablation.json")) as f:
        return json.load(f)


def fig_ablation_curves(abl):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in abl["rows"]:
        rows = metrics(r)
        c = ABL_COLOR[r["id"]]
        name = f"{r['id']}: {r['description']}"
        for ax, key in zip(axes, ("loss", "acc")):
            ax.plot(*col(rows, f"train_{key}"), color=c, linestyle="--", alpha=0.85, label=f"{name}, train")
            ax.plot(*col(rows, f"val_{key}"), color=c, label=f"{name}, val")
        axes[1].plot([r["best_epoch"]], [r["val_acc"]], "o", ms=8, color=c, markeredgecolor="white",
                     markeredgewidth=2)
    axes[0].set(title="Cross-entropy loss (C's train loss includes label smoothing)", xlabel="epoch",
                ylabel="loss")
    axes[1].set(title="Accuracy (dots = best-val checkpoint)", xlabel="epoch", ylabel="accuracy")
    axes[0].legend(fontsize=7)
    fig.suptitle("CNN ablation: A baseline, B + random erasing, C + label smoothing (solid = val, dashed = train)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cnn_ablation_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _grouped(ax, groups, series, ylim):
    """series: list of (label, color, [values per group])."""
    n = len(series)
    width = 0.8 / n
    for k, (label, color, vals) in enumerate(series):
        xs = [i + (k - (n - 1) / 2) * width for i in range(len(groups))]
        ax.bar(xs, vals, width=width, color=color, edgecolor="white", linewidth=2, label=label)
    ax.set_xticks(range(len(groups)), groups, fontsize=9)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.06))


def fig_ablation_per_class(abl):
    classes = abl["class_names"]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    series = [(f"{r['id']}: {r['description']}", ABL_COLOR[r["id"]],
               [r["per_class_val_acc"][c] for c in classes]) for r in abl["rows"]]
    _grouped(ax, classes, series, (0.6, 1.02))
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in HIGHLIGHT:
            lbl.set_fontweight("bold")
    ax.set_title("Per-class val accuracy (bold = the four upper-body classes that get confused)", pad=34)
    ax.set_ylabel("accuracy")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cnn_ablation_per_class.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_ablation_shift(abl):
    groups = ["clean"] + abl["shift_groups"]
    fig, ax = plt.subplots(figsize=(11, 5))
    series = [(f"{r['id']}: {r['description']}", ABL_COLOR[r["id"]],
               [r["val_acc"]] + [r["shifted_val_acc"][g] for g in abl["shift_groups"]]) for r in abl["rows"]]
    _grouped(ax, groups, series, (0.0, 1.05))
    for k, (_, _, vals) in enumerate(series):
        for i, v in enumerate(vals):
            ax.text(i + (k - 1) * 0.8 / 3, v + 0.005, f"{v:.3f}", ha="center", fontsize=7, color=INK)
    n = abl["rows"][0]["shifted_val_n"]
    ax.set_xticks(range(len(groups)), ["clean val\nn=6000"] + [f"{g}\nn={n[g]}" for g in abl["shift_groups"]])
    ax.set_title("Val accuracy by shift kind (no TTA)", pad=34)
    ax.set_ylabel("accuracy")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cnn_ablation_shift.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_confusion(row, classes, path, tta=False):
    cm = np.array(row["confusion_val_tta" if tta else "confusion_val"])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(10), classes, rotation=45, ha="right")
    ax.set_yticks(range(10), classes)
    ax.set(xlabel="predicted", ylabel="true")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")
    acc = row["val_acc_tta" if tta else "val_acc"]
    ax.set_title(f"{row['id']}: {row['description']}{' + flip TTA' if tta else ''}\nval confusion "
                 f"(counts; colour = row-normalized), acc {acc:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def ablation_figures():
    abl = load_ablation()
    paths = [fig_ablation_curves(abl), fig_ablation_per_class(abl), fig_ablation_shift(abl)]
    for r in abl["rows"]:
        paths.append(fig_confusion(r, abl["class_names"], os.path.join(OUT_DIR, f"cnn_confusion_{r['id']}.png")))
    sel = abl["selected"]
    row = next(r for r in abl["rows"] if r["id"] == sel["id"])
    paths.append(fig_confusion(row, abl["class_names"], os.path.join(OUT_DIR, "cnn_confusion_selected.png"),
                               tta=sel["tta"]))
    return paths


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT_DIR, exist_ok=True)
    if which in ("mlp", "all"):
        index = load_index("mlp")
        for fn in (fig_sgd_vs_adam, fig_overfit, fig_overfit_fix, fig_dead_relu):
            print("wrote", os.path.relpath(fn(index), ROOT))
    if which in ("cnn", "all"):
        index = [e for e in load_index("cnn") if e["tag"] in ("const", "step")]
        for fn in (fig_cnn_curves, fig_cnn_shift):
            print("wrote", os.path.relpath(fn(index), ROOT))
    if which in ("ablation", "all"):
        for path in ablation_figures():
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
