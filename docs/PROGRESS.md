# ACE-Net progress (Phases 0–7 done; project closed out)

| model (test once, flip TTA unless noted) | val+TTA | test float | test int8 | shifted all | noise | size |
|---|---|---|---|---|---|---|
| MLP 784-256-128-10 (Adam) | – | 0.8916 (no TTA) | – | – | – | – |
| A: CNN 32-32-64-64-fc128, AdamW 1e-3, crop+flip | 0.9393 | 0.9325 (no TTA) | – | 0.683 | 0.293 | – |
| B = A + random erasing | 0.9398 | 0.9349 | 0.9344 | 0.760 | 0.506 | 467 KB npz |
| **E2 = A + finder lr + CosineWarmup — FINAL clean model / teacher** | **0.9450** | **0.9403** | **0.9399** | 0.671 | 0.252 | 467 KB npz |
| D = B + shift aug + finder lr + CosineWarmup | 0.9368 | 0.9278 | 0.9282 | **0.910** | **0.875** | 467 KB npz |
| **S0 student (P, CE) — FINAL student (selected on int8 val)** | 0.9367 (int8 0.9337) | 0.9304 | **0.9281** | 0.628 | 0.242 | 43,770 B |
| S1 student (P, KD T=4 α=0.9) — KD deliverable, tested Phase 7 | 0.9337 (int8 0.9323) | 0.9299 | 0.9287 | 0.638 | 0.210 | 43,770 B |

Criterion (int8 test ≥ E2 float − 2 pt = 0.9203): S0 pass (0.9281), S1 pass (0.9287). S1 − S0 = +0.06 pt test (6 images, < 1 s.e.); selection was fixed on val before either test.
Sources: runs/test_once.json (all 6b+ test numbers), runs/distill.json (`final`, `deliverable`), runs/ablation_6b.json, docs/ablation.md.

## Phase 7 (pre-registered in commit ba2ccb9 before any Phase 7 evaluation)
- (a) S1 tested once as the required distilled deliverable (`distill deliverable`, guarded by `once()`); S0 stays selected. `final()`'s
  "no other student touched test" assert now exempts only `DELIVERABLE = "S1"`.
- (b) Hidden-set scope: E2, B, D (float + int8) + S0, S1 (int8), flip TTA. `experiments/hidden_eval.py` evaluates all 8 in one command ->
  runs/hidden_eval_<name>.{json,_per_class.md,_predictions.csv,_confusion.png,_vs_test.png}. Self-test re-run: PASS (npz/idx/PNG give
  identical images, labels and predictions for all 8; 56×56 RGB variant agrees 100%); 9.5 min.
- `scripts/make_tables.py`: all 21 README tables generated from runs/*.json (`--check` verifies up to date; fails loudly on a missing key).
- `tests/conftest.py` writes runs/gradcheck.json (27 ops, worst 1.5e-9; 147 tests) on a full green run.
- `--smoke` on every experiment script (experiments/__init__.py sets ACE_SMOKE=1 → 2048/512/512 subsets, 1 epoch per fit).
  `scripts/smoke_test.py`: fresh clone, empty runs/, download, 33 pipeline steps + pytest + make_tables → runs/smoke_test.json. PASS (~8 min).
- Smoke-test fixes: `distill ablation` crashed when Phase 3b selects A (B untested); hardcoded B test numbers in `robust.select` and
  `quant.FLOAT_B_TEST` replaced by reads from runs/cnn_ablation.json / cnn_b_int8.json (values unchanged: 0.9349 / 0.9344).
- tests/test_repo.py checks every README/docs link, image and backticked path exists (73 targets).

## Decisions / findings (carried)
- KD did not help on the selection metric (−0.30 val+TTA S1−S0) but improved NLL/ECE and rotate/brightness-add; candidate reasons in README.
- Student int8 cost (−0.30 val+TTA) is 10× E2's (−0.03). int8 engine is ~6× slower than float in numpy (no int8 GEMM).
- Selection-rule revision (6b, commit 693f9c2): B-vs-D rule → best val+TTA over all runs (E2), before E2 touched test.

## Open issues
- Hidden set not yet released: run `python -m experiments.hidden_eval PATH` then `python -m scripts.make_tables`; README hidden table fills itself.
- Runtimes of evaluation-only stages were never logged for the full runs (README shows "not logged"; smoke times are measured).
- Student is not robust (noise 0.24). Untried: shift-aug cache, D as teacher, T/α sweep. ~660 MB of KD caches in data/ can be deleted.
