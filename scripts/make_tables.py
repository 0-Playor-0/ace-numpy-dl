"""Regenerate every results table in README.md from runs/*.json (single source of truth).

README.md marks each table as
    <!-- table:NAME -->
    ...generated...
    <!-- /table:NAME -->
and this script rewrites the body between the markers. It fails loudly (exit 1) when README references a
table it does not know, or a table needs a run file / key that does not exist.

Usage:
  python -m scripts.make_tables           # rewrite README.md in place
  python -m scripts.make_tables --check   # exit 1 if README.md is out of date (nothing written)
  python -m scripts.make_tables --smoke   # smoke runs: a missing source renders as MISSING and is listed, exit 0
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
README = os.path.join(ROOT, "README.md")
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
_CACHE = {}


class Missing(Exception):
    pass


def J(name, *keys):
    """runs/<name>.json, then keys in order; raises Missing naming the file and key path."""
    path = os.path.join(RUNS, name + ".json")
    if path not in _CACHE:
        if not os.path.exists(path):
            raise Missing(f"runs/{name}.json does not exist")
        with open(path) as f:
            _CACHE[path] = json.load(f)
    v = _CACHE[path]
    for i, k in enumerate(keys):
        try:
            v = v[k]
        except (KeyError, IndexError, TypeError):
            raise Missing(f"runs/{name}.json: key {'.'.join(map(str, keys[:i + 1]))} not found") from None
    return v


def exists(name):
    return os.path.exists(os.path.join(RUNS, name + ".json"))


def f4(v):
    return "–" if v is None else f"{v:.4f}"


def pct(v, d=1):
    return f"{100 * v:.{d}f}"


def pt(v):
    return f"{100 * v:+.2f}"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def hms(s):
    if s is None:
        return "not logged"
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min" if s < 7200 else f"{s / 3600:.1f} h"


def abl(rid):
    return next(t for t in J("ablation_6b", "table") if t["id"] == rid)


def teacher_row(rid):
    return next(r for r in J("ablation_6b", "teacher_rows") if r["id"] == rid)


def student_row(sid):
    return J("ablation_6b", "student_rows", sid)


def cnn_run(tag):
    return next(e for e in J("cnn_index") if e["tag"] == tag)


# ---------------------------------------------------------------- tables

def t_checklist():
    tests = J("test_once")
    gc = J("gradcheck")
    worst = max(gc["max_rel_err"].values())
    e2i, s0i, s1i = tests["E2_int8"]["test_acc"], tests["student_S0_int8"]["test_acc"], tests["student_S1_int8"]["test_acc"]
    e2f = tests["E2_float"]["test_acc"]
    b8 = J("cnn_b_int8")
    rows = [
        ("Core", "Tensor + reverse-mode autograd (graph, topo sort, broadcasting)",
         f"{len(gc['max_rel_err'])} ops gradchecked, worst rel. err {worst:.1e} (tol {gc['tolerance']:g}); "
         f"{gc['pytest']['passed']} tests pass", "`acenet/tensor.py`, `tests/test_autograd.py`, `runs/gradcheck.json`"),
        ("Core", "Layers, losses, optimizers (Linear/Conv/BN/Pool/Dropout; CE; SGD-momentum, Adam/AdamW)",
         "all gradchecked or unit-tested", "`acenet/{ops,nn,optim}.py`, `tests/`"),
        ("Core", "MLP on Fashion-MNIST with train/val curves",
         f"test {f4(J('mlp_test', 'adam', 'test_acc'))} (Adam), {f4(J('mlp_test', 'sgd', 'test_acc'))} (SGD)",
         "`runs/mlp_test.json`, `experiments/figures/mlp_sgd_vs_adam.png`"),
        ("Core", "Failure case analysed and fixed", "overfit + divergence diagnosed and fixed",
         "`docs/failure.md`, `runs/mlp_overfit_diagnosis.json`"),
        ("Advanced", "CNN (conv/pool/BN) with augmentation",
         f"E2 test {f4(e2f)} (flip TTA), A {f4(J('cnn_test', 'test_acc'))} (no TTA)", "`runs/test_once.json`, `runs/cnn_test.json`"),
        ("Advanced", "int8 post-training quantization, integer-only inference",
         f"E2 int8 test {f4(e2i)} ({pt(e2i - e2f)} pt), {b8['size']['ratio_vs_float32']:.1f}x smaller",
         "`acenet/quant.py`, `runs/cnn_{b,e2,d}_int8.json`"),
        ("Advanced", "Robustness to distribution shift",
         f"D shifted val {abl('D')['shifted_all']:.3f} vs E2 {abl('E2')['shifted_all']:.3f}", "`docs/robustness.md`, `runs/robustness.json`"),
        ("Bonus", "Knowledge distillation into a <= 50 KB int8 student",
         f"S1 (KD) int8 test {f4(s1i)}; S0 (CE, selected) {f4(s0i)}; both {J('distill', 'quant', 'S0', 'file_bytes'):,} B; "
         f"criterion >= {f4(e2f - 0.02)}: {'pass' if J('distill', 'deliverable', 'pass') and J('distill', 'final', 'test', 'pass') else 'FAIL'}",
         "`runs/distill.json`, `docs/ablation.md`"),
        ("Bonus", "LR finder + cosine warmup schedule",
         f"cosine warmup {J('ablation_6b', 'deltas_pt_val_tta', 'cosine warmup (E2 - E1)'):+.2f} pt val+TTA",
         "`runs/lr_finder.json`, `runs/lr_finder.png`"),
        ("Bonus", "Hidden-set evaluator (npz / idx / PNG, any size)",
         f"self-test {'pass' if J('hidden_eval_selftest', 'pass') else 'FAIL'} on {J('hidden_eval_selftest', 'n'):,} images, "
         f"{len(J('hidden_eval_selftest', 'models'))} models", "`experiments/hidden_eval.py`, `runs/hidden_eval_selftest.json`"),
        ("Repro", "Fresh-clone smoke test of every experiment",
         f"{'pass' if J('smoke_test', 'pass') else 'FAIL'}: {len(J('smoke_test', 'steps'))} steps in {hms(J('smoke_test', 'total_s'))}",
         "`scripts/smoke_test.py`, `runs/smoke_test.json`"),
    ]
    return table(["tier", "requirement", "result", "evidence"], rows)


def t_env():
    sm = J("smoke_test")
    gc = J("gradcheck")
    return table(["item", "value"], [("hardware", sm["cpu"]), ("platform", sm["platform"]),
                                     ("Python", gc["python"]), ("numpy", gc["numpy"]),
                                     ("test suite", f"{gc['pytest']['passed']} passed, 0 failed"),
                                     ("smoke test (fresh clone, all steps)", hms(sm["total_s"]) + f" + download {hms(sm['download_s'])}")])


def t_runtimes():
    smoke = {s["cmd"].replace("python -m ", "").replace(" --smoke", ""): s["s"] for s in J("smoke_test", "steps")}
    mlp = J("mlp_index")
    d = J("distill")

    def tag_sum(idx, pred):
        return sum(e["wall_time_s"] for e in idx if pred(e))

    full = {
        "experiments.mlp sweep": tag_sum(mlp, lambda e: e["tag"] == "sweep"),
        "experiments.mlp failures": tag_sum(mlp, lambda e: e["tag"] in ("overfit", "diverge")),
        "experiments.mlp run --tag fix --opt adam --lr 3e-4 --dropout 0.3 --wd 0.05": tag_sum(mlp, lambda e: e["tag"] == "fix"),
        "experiments.diagnose_overfit": tag_sum(mlp, lambda e: e["tag"].startswith("diag")),
        "experiments.cnn train": cnn_run("const")["wall_time_s"],
        "experiments.cnn ablation": cnn_run("erase")["wall_time_s"] + cnn_run("erase_ls")["wall_time_s"],
        "experiments.quant B": J("cnn_b_int8", "timing", "int8_s"),
        "experiments.robust finder": J("lr_finder", "wall_time_s"),
        "experiments.robust train": sum(cnn_run(t)["wall_time_s"] for t in ("e1_finder_const", "e2_finder_cosine",
                                                                             "d_shift_finder_cosine")),
        "experiments.distill cache": d["cache"]["build_s"],
        "experiments.distill cache_cropflip": d["cache_cropflip"]["build_s"],
        "experiments.distill screen": sum(v["wall_time_s"] + v["finder"]["wall_time_s"] for v in d["screen"].values()),
        "experiments.distill students": sum(v["wall_time_s"] for v in d["students"].values()),
    }
    notes = {"experiments.quant B": "int8 engine on 10k images only", "experiments.cnn train": "20 epochs",
             "experiments.robust train": "E1, E2, D: 20 epochs each", "experiments.distill students": "S0, S1: 40 epochs each",
             "experiments.distill screen": "3 finders + 3 x 10 epochs", "experiments.mlp sweep": "6 x 40 epochs"}
    rows = [(f"`python -m {cmd}`", hms(full.get(cmd)), f"{s:.1f} s", notes.get(cmd, "")) for cmd, s in smoke.items()]
    return table(["command", "full run (measured)", "smoke", "note"], rows)


def t_gradcheck():
    gc = J("gradcheck")
    items = list(gc["max_rel_err"].items())
    half = -(-len(items) // 2)
    left, right = items[:half], items[half:] + [("", None)] * (2 * half - len(items))
    return table(["op", "max rel. err", "op", "max rel. err"],
                 [(a, f"{x:.1e}", b, "" if y is None else f"{y:.1e}") for (a, x), (b, y) in zip(left, right)])


def t_mlp():
    rows = []
    for e in J("mlp_index"):
        if e["tag"] not in ("sweep", "overfit", "fix", "diverge"):
            continue
        test = next((J("mlp_test", o, "test_acc") for o in ("sgd", "adam") if J("mlp_test", o, "run_name") == e["run_name"]), None)
        reg = f"dropout {e['dropout']:g}, wd {e['weight_decay']:g}" if e["dropout"] or e["weight_decay"] else "none"
        rows.append((e["tag"], e["optimizer"], f"{e['lr']:g}", reg, e["epochs"], e["status"],
                     f4(e["best_val_acc"]), e["best_epoch"] or "–", f4(test), hms(e["wall_time_s"])))
    return table(["tag", "opt", "lr", "regularization", "epochs", "status", "best val", "best ep", "test (once)", "time"], rows)


def t_mlp_failure():
    o = J("mlp_overfit_diagnosis")
    keys = [("val_acc", "val acc", f4), ("val_loss", "val NLL", f4), ("n_wrong", "# wrong", str),
            ("loss_share_from_wrong", "loss share from wrong", lambda v: pct(v) + "%"),
            ("mean_conf_wrong", "mean confidence when wrong", f4),
            ("frac_wrong_conf_gt_0.99", "wrong with conf > 0.99", lambda v: pct(v) + "%"),
            ("weight_l2_norm", "weight L2 norm", lambda v: f"{v:.1f}")]
    rows = [(lab, fn(o["overfit"]["best"][k]), fn(o["overfit"]["last"][k]), fn(o["fix"]["best"][k]), fn(o["fix"]["last"][k]))
            for k, lab, fn in keys]
    return table(["", f"overfit @ best ep {o['overfit']['best']['epoch']}", f"overfit @ ep {o['overfit']['last']['epoch']}",
                  f"fix @ best ep {o['fix']['best']['epoch']}", f"fix @ ep {o['fix']['last']['epoch']}"], rows)


def t_cnn():
    rows = []
    for rid in ("A", "B", "C", "E1", "E2", "D"):
        r, t = teacher_row(rid), abl(rid)
        test = "–" if t["test"] is None else f4(t["test"]) + ("" if t["test_tta"] else " (no TTA)")
        rows.append((rid, r["description"], r["best_epoch"], f4(t["val"]), f4(t["val_tta"]), test, f"{t['ece']:.4f}",
                     hms(r["wall_time_s"])))
    return table(["id", "recipe", "best ep", "val", "val+TTA", "test (once)", "ECE", "train time"], rows)


def t_per_class():
    ids = ["A", "B", "C", "E2", "D"]
    rows = [(c, *[f"{teacher_row(r)['per_class_val_acc_tta'][c]:.3f}" for r in ids],
             f"{student_row('S0')['per_class_val_acc_tta'][c]:.3f}") for c in CLASSES]
    return table(["class (val+TTA)", *ids, "S0"], rows)


def t_int8():
    tests = J("test_once")
    rows = []
    for rid, fn, ftest in (("B", "cnn_b_int8", J("cnn_ablation", "test", "test_acc")),
                           ("E2", "cnn_e2_int8", tests["E2_float"]["test_acc"]),
                           ("D", "cnn_d_int8", tests["D_float"]["test_acc"])):
        q = J(fn)
        c = q["calibrators"][q.get("selected_calibrator", "p99.99")] if "selected_calibrator" in q else q["calibrators"]["p99.99"]
        rows.append((rid, f4(q["float_val"]["val_acc_tta"] if "val_acc_tta" in q["float_val"] else abl(rid)["val_tta"]),
                     f4(c["val_acc_tta"]), f4(ftest), f4(q["test"]["test_acc"]), pt(q["test"]["test_acc"] - ftest),
                     f"{c['agree_int_vs_float']:.4f}", f"{q['size']['file_bytes'] / 1024:.1f} KB"))
    for sid in ("S0", "S1"):
        q = J("distill", "quant", sid)
        rows.append((sid, f4(abl(sid)["val_tta"]), f4(q["val_acc_tta"]), f4(q["float_test"]), f4(q["int8_test"]),
                     pt(q["int8_test"] - q["float_test"]), f"{q['agree_int_vs_float']:.4f}", f"{q['file_bytes']:,} B"))
    return table(["model", "float val+TTA", "int8 val+TTA", "float test", "int8 test", "Δ test (pt)",
                  "int8 = float pred. (val)", "int8 file"], rows)


def t_int8_size():
    s = J("cnn_b_int8", "size")
    rows = [(k, f"{v['w']:,}", f"{v['b']:,}", f"{v['M0']:,}", f"{v['shift']:,}", f"{v['total']:,}",
             f"{v['float32_params_bytes']:,}") for k, v in s["layers"].items()]
    rows.append(("**total**", "", "", "", "", f"**{s['payload_bytes']:,}** (+{s['npz_overhead_bytes']:,} npz)",
                 f"**{1024 * s['float32_kib']:,.0f}**"))
    c = J("distill", "candidates", "P")
    rows.append(("student P (.ace8)", f"{c['int8_weights_bytes']:,}", "", f"{c['per_channel_bytes']:,} (b+M0+shift)", "",
                 f"{c['int8_file_bytes']:,} (header {c['header_bytes']})", f"{4 * c['params_folded']:,}"))
    return table(["layer", "int8 W", "int32 b", "int32 M0", "int8 shift", "bytes", "float32 bytes"], rows)


def t_int8_speed():
    t = J("cnn_b_int8", "timing")
    return table(["engine (model B, numpy, 10k test images, batch 200)", "seconds", "ms / image"],
                 [("float32, BN folded", f"{t['float32_folded_numpy_s']:.1f}", f"{1000 * t['float32_folded_numpy_s'] / t['n']:.2f}"),
                  ("int8 integer-only", f"{t['int8_s']:.1f}", f"{t['int8_ms_per_image']:.2f}")])


def t_int8_sens():
    rows = [(k, f"{v['drop'] * 100:+.2f}", f"{J('cnn_e2_int8', 'sensitivity', k, 'drop') * 100:+.2f}")
            for k, v in J("cnn_b_int8", "sensitivity").items()]
    return table(["layer quantized alone (fake-quant)", "B Δ val (pt)", "E2 Δ val (pt)"], rows)


def t_robust():
    groups = ("rotate", "noise", "brightness-mul", "brightness-add")
    rows = []
    for rid in ("A", "B", "C", "E2", "D"):
        r = teacher_row(rid)
        rows.append((rid, f4(r["val_acc"]), *[f"{r['shifted_val_acc'][g]:.3f}" for g in groups], f"{r['shifted_val_acc_all']:.3f}"))
    for sid in ("S0", "S1"):
        q = J("distill", "quant", sid)
        rows.append((f"{sid} int8", f4(q["val_acc"]), *[f"{q['shifted_val_acc'][g]:.3f}" for g in groups],
                     f"{q['shifted_val_acc_all']:.3f}"))
    n = teacher_row("A")["shifted_val_n"]
    return table(["model (no TTA)", "clean val", *[f"{g} (n={n[g]:,})" for g in groups], "shifted all"], rows)


def t_distill():
    d = J("distill")
    rows = []
    for sid in ("S0", "S1"):
        st, r, q, t = d["students"][sid], student_row(sid), d["quant"][sid], abl(sid)
        loss = "CE" if st["kd"] is None else f"KD T={st['kd'][0]:g}, α={st['kd'][1]:g}"
        rows.append((sid + (" (selected)" if d["final"]["selected"] == sid else " (KD deliverable)"), loss,
                     f4(r["val_acc"]), f4(r["val_acc_tta"]), f"{r['val_nll']:.4f}", f"{r['ece']:.4f}",
                     f4(q["val_acc_tta"]), f4(q["float_test"]), f4(q["int8_test"]), hms(st["wall_time_s"])))
    return table(["student", "loss", "val", "val+TTA", "val NLL", "ECE", "int8 val+TTA", "float test", "int8 test",
                  "train time"], rows)


def t_distill_candidates():
    d = J("distill")
    rows = [(cid, c["description"], f"{c['params_trainable']:,}", f"{c['int8_file_bytes']:,}", f"{c['macs_per_image'] / 1e6:.1f}M",
             f"{d['screen'][cid]['s_per_epoch']:.0f}", f4(d["screen"][cid]["best_val_acc"]))
            for cid, c in d["candidates"].items()]
    return table(["id", "architecture", "params", "int8 bytes", "MACs", "s/epoch", "10-ep CE val"], rows)


def t_distill_cache():
    rows = []
    for k, label in (("cache", "v1 crop+flip+erase (screen only)"), ("cache_cropflip", "v2 crop+flip (S0, S1)")):
        c = J("distill", k)
        accs = c["teacher_train_acc_on_copies"]
        rows.append((label, c["K"], f"{c['n']:,}", f"{c['total_mb']:.0f} MB", hms(c["build_s"]),
                     f"{min(accs):.3f}–{max(accs):.3f}"))
    return table(["cache", "copies", "images / copy", "size", "build", "teacher (E2) acc on copies"], rows)


def t_lr():
    f = J("lr_finder")
    rows = [("CNN A recipe", f"{f['min_loss_lr']:.3g}", f"{f['suggested_min_div10']:.3g}", f"{f['suggested_steepest']:.3g}",
             f"min/10 ({f['ratio_vs_1e-3']:.1f}x the hand-picked 1e-3)")]
    for cid, s in J("distill", "screen").items():
        fi = s["finder"]
        rows.append((f"student {cid}", f"{fi['min_loss_lr']:.3g}", f"{fi['suggested_min_div10']:.3g}",
                     f"{fi['suggested_steepest']:.3g}", "min/10"))
    deltas = J("ablation_6b", "deltas_pt_val_tta")
    t1 = table(["finder run", "lr at min loss", "min / 10", "steepest", "used"], rows)
    t2 = table(["change", "Δ val+TTA (pt)"], [(k, f"{deltas[k]:+.2f}") for k in
                                              ("finder lr, constant (E1 - A)", "cosine warmup (E2 - E1)")])
    return t1 + "\n\n" + t2


def t_ablation():
    rows = []
    for t in J("ablation_6b", "table"):
        test = "–" if t["test"] is None else f4(t["test"]) + ("" if t["test_tta"] else " (no TTA)")
        rows.append((t["id"], f4(t["val"]), f4(t["val_tta"]), test, "–" if t["ece"] is None else f"{t['ece']:.4f}",
                     f"{t['shifted_all']:.3f}", f"{t['noise']:.3f}", f"{t['params']:,}", f"{t['size_kb']:.1f}"))
    return table(["model", "val", "val+TTA", "test (once)", "ECE", "shifted all", "noise", "params", "size KB"], rows)


def t_deltas():
    return table(["change", "Δ (pt)"], [(k, f"{v:+.2f}") for k, v in J("ablation_6b", "deltas_pt_val_tta").items()])


def t_test_log():
    store = J("test_once")
    rows = [("A (Phase 3)", f4(J("cnn_test", "test_acc")), "no", "`runs/cnn_test.json`"),
            ("B (Phase 3b)", f4(J("cnn_ablation", "test", "test_acc")), "flip", "`runs/cnn_ablation.json`"),
            ("B int8 (Phase 4)", f4(J("cnn_b_int8", "test", "test_acc")), "flip", "`runs/cnn_b_int8.json`"),
            *[(k.replace("student_", "").replace("_", " "), f4(v["test_acc"]), "flip", "`runs/test_once.json`")
              for k, v in store.items()]]
    mlp = [(f"MLP {o}", f4(J("mlp_test", o, "test_acc")), "no", "`runs/mlp_test.json`") for o in ("sgd", "adam")]
    return table(["model", "test acc", "TTA", "record"], mlp + rows)


def t_hidden():
    sel = J("hidden_eval_selftest")
    note = (f"Format self-test ({sel['n']:,} shifted-val images saved as npz / idx.gz / PNG): images, labels and every "
            f"model's predictions identical across formats: **{'pass' if sel['pass'] else 'FAIL'}**.")
    if not exists("hidden_eval_hidden"):
        return note + "\n\n_Hidden set not released yet. After `python -m experiments.hidden_eval PATH`, re-run " \
                      "`python -m scripts.make_tables` to fill this table from `runs/hidden_eval_hidden.json`._"
    h = J("hidden_eval_hidden")
    rows = [(k, f4(v["clean_test"]), f4(v["acc"]), "–" if v["clean_test"] is None else pt(v["acc"] - v["clean_test"]))
            for k, v in h["models"].items()]
    return (note + "\n\n" + table(["model (flip TTA)", "clean test", f"hidden (n={h['n']:,})", "Δ (pt)"], rows) +
            "\n\nPer-class: `runs/hidden_eval_hidden_per_class.md`. Figure: `runs/hidden_eval_hidden_vs_test.png`.")


TABLES = {n[2:]: fn for n, fn in globals().items() if n.startswith("t_") and callable(fn)}
MARK = re.compile(r"(<!-- table:(\w+) -->\n)(.*?)(<!-- /table:\2 -->)", re.S)


def render(text, allow_missing=False):
    used = re.findall(r"<!-- table:(\w+) -->", text)
    errors = [f"README references unknown table '{n}'" for n in used if n not in TABLES]
    out = {}
    for n in used:
        if n in TABLES:
            try:
                out[n] = TABLES[n]()
            except (Missing, StopIteration) as e:
                msg = f"table '{n}': {e or 'referenced run not found in index'}"
                if allow_missing:
                    out[n] = f"_MISSING: {msg}_"
                    print("MISSING", msg)
                else:
                    errors.append(msg)
    if errors:
        raise SystemExit("make_tables FAILED:\n  " + "\n  ".join(errors))
    unused = sorted(set(TABLES) - set(used))
    return MARK.sub(lambda m: m.group(1) + out[m.group(2)] + "\n" + m.group(4), text), used, unused


def main():
    args = sys.argv[1:]
    with open(README) as f:
        text = f.read()
    new, used, unused = render(text, allow_missing="--smoke" in args or os.environ.get("ACE_SMOKE") == "1")
    if "--check" in args:
        if new != text:
            raise SystemExit("README.md tables are out of date: run python -m scripts.make_tables")
        print(f"README.md up to date ({len(used)} tables)")
        return
    with open(README, "w") as f:
        f.write(new)
    print(f"wrote {len(used)} tables into README.md" + (f"; not used: {unused}" if unused else ""))


if __name__ == "__main__":
    main()
