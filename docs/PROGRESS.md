# ACE-Net progress (Phases 0–6b done; next: Phase 7, not started)

| model (test once, flip TTA unless noted) | val+TTA | test float | test int8 | shifted all | noise | size |
|---|---|---|---|---|---|---|
| MLP 784-256-128-10 (Adam) | – | 0.8916 (no TTA) | – | – | – | – |
| A: CNN 32-32-64-64-fc128, AdamW 1e-3, crop+flip | 0.9393 | 0.9325 (no TTA) | – | 0.683 | 0.293 | – |
| B = A + random erasing (Phase 3b/4 final) | 0.9398 | 0.9349 | 0.9344 | 0.760 | 0.506 | 467 KB npz |
| **E2 = A + finder lr + CosineWarmup — FINAL clean model / teacher** | **0.9450** | **0.9403** | **0.9399** | 0.671 | 0.252 | 467 KB npz |
| D = B + shift aug + finder lr + CosineWarmup | 0.9368 | 0.9278 | 0.9282 | **0.910** | **0.875** | 467 KB npz |
| **S0 student (P, 42.6k params, CE) — FINAL student** | 0.9367 (int8 0.9337) | 0.9304 | **0.9281** | 0.628 | 0.242 | **43,770 B** |
| S1 student (P, KD T=4 α=0.9) | 0.9337 (int8 0.9323) | – | – | 0.638 | 0.210 | 43,770 B |

Full table and per-trick deltas: docs/ablation.md and runs/ablation_6b.json. Phase 6b state: runs/distill.json. All test numbers: runs/test_once.json.
Checkpoints: runs/<run>/best.npz. Teachers via `entry(rid)`/`load_model` (experiments/cnn_ablation.py); students via `load_student(sid)` (experiments/distill.py).
Int8 files: runs/cnn_{b,e2,d}_int8.npz (Phase 4 format) and runs/student_S{0,1}_int8.ace8 (compact; `Int8Model.load_compact`).

## Phase 6b (pre-registered in commit 693f9c2 before any evaluation)
- Selection: the final clean model is the best val+TTA among all val-evaluated runs, which is E2. The hidden-set report will cover exactly E2, B and D. Teacher = E2.
- New pieces: `ops.depthwise_conv2d` (offset form, 2–3× faster than grouped im2col; gradchecked), `nn.GlobalAvgPool2D`, and `ops.kd_loss` (T²-scaled KL + CE; gradchecked).
  Also: `data.CacheLoader`, `fit(loss_fn=)`, dwconv + integer GAP in the int8 engine, and the compact `.ace8` format (header 56 B; round-trip test). Test-once guard: `distill.once()`.
- Screen (10 ep CE, AdamW + finder lr + CosineWarmup, cache v1): val P 0.9262 (68 s/ep), DS 0.9268 (215 s/ep), H 0.9255 (168 s/ep). P picked (within 0.3 pt). 40 epochs.
- Student P: conv 16-24 @28 | 48 @14 | 64 @7, GAP, fc10. 6.2M MACs, lr 5.7e-2 (finder min/10). 72–76 s/epoch.

## Deviations (user-directed, all recorded before the affected runs)
- Cache v1 (crop+flip+erase) gave E2 only 0.873 train acc, so it was used for the screen only. Cache v2 (data/kd_cache_cropflip) is crop+flip, same K=4 and seed 21: E2 0.953, 332 MB, built in 186 s. All S runs use v2.
- Screen pick: P if within 0.3 pt of the best (DS was best by 0.07 pt). S-run epochs: 30 if > 120 s/ep (not triggered).
- S2 gate (val only): S1 int8 val+TTA 0.9323 ≥ 0.925, so **S2 was not trained**.
- Final student = best **int8** val+TTA among the trained S runs (S0 0.9337 > S1 0.9323). Only S0 touched test, once. The code asserts this.
- Teachers E2 and D (step 1) were tested before the cache/student work, per plan. D's int8 used the fixed p99.99 calibrator, as in 6a.

## Decisions / findings
- Criterion: S0 int8 test 0.9281 ≥ E2 float 0.9403 − 2.0 = 0.9203 → **PASS** (−1.22 pt vs E2 float, −1.18 vs E2 int8). Size is 43,770 B ≤ 50,000.
- KD did not help on the selection metric (−0.30 val+TTA), though it improved ECE, NLL and rotate/brightness. The teacher is only 95% right on the cache.
- The student's int8 cost (−0.30 val+TTA) is 10× E2's (−0.03). Student shifted val by kind (int8): rotate 0.872, noise 0.242, brightness-mul 0.918, brightness-add 0.666.

## Phase 7 pre-registration (recorded before any Phase 7 evaluation)
- (a) S1 (KD) is the task's required distilled-student deliverable. It is tested ONCE (float + int8, flip TTA) for that purpose only. S0 remains the selected best small model whatever S1 scores.
- (b) The hidden-set report covers E2, B, D (declared in 6b) plus S0 int8 and S1 int8, added now, before release. Float + int8 where applicable, flip TTA.

## Open issues / Phase 7
- experiments/hidden_eval.py still reads runs/final_model.json (B). Extend it to E2, B and D (float + int8), as pre-registered, before the hidden set is released.
- The student is not robust (noise 0.24). Untried: student trained on a shift-aug cache, or D as teacher. The ~300 MB caches in data/ can be deleted once no longer needed.
