# ACE-Net progress (Phases 0–6a done; next: distillation, not started)

| model | val | val+TTA | ECE | shifted all | noise | test (once) |
|---|---|---|---|---|---|---|
| MLP 784-256-128-10 (SGD / Adam) | 0.8985 / 0.8983 | – | – | – | – | 0.8880 / 0.8916 |
| A: CNN AdamW 1e-3 wd 5e-4, 20 ep, crop+flip | 0.9362 | 0.9393 | 0.0066 | 0.683 | 0.293 | 0.9325 |
| **B = A + random erasing (FINAL)** | 0.9377 | **0.9398** | 0.0066 | 0.760 | 0.506 | **0.9349** (TTA) |
| C = B + label smoothing 0.1 | 0.9360 | 0.9383 | 0.0848 | 0.732 | 0.450 | – |
| E1 = A + finder lr 9.1e-3, constant | 0.9387 | 0.9398 | 0.0087 | 0.691 | 0.352 | – |
| E2 = A + finder lr + CosineWarmup(1 ep) | **0.9448** | **0.9450** | 0.0043 | 0.671 | 0.252 | – |
| D = B + shift aug + finder lr + CosineWarmup | 0.9360 | 0.9368 | 0.0129 | **0.910** | **0.875** | – |
| **B int8 PTQ (p99.99) + TTA (FINAL int8)** | 0.9368 | 0.9387 | – | – | – | **0.9344** |

All rows: runs/ablation.json (merged 3b + 6a format). Robustness numbers: runs/robustness.json and docs/robustness.md (draft).
Checkpoints (best.npz; `entry(rid)`/`load_model` in experiments/cnn_ablation.py): A cnn_const_20261019-114445 · B cnn_erase_20261019-123428 ·
C cnn_erase_ls_… · E1 cnn_e1_finder_const_20261019-151959 · E2 cnn_e2_finder_cosine_20261019-160440 · D cnn_d_shift_finder_cosine_20261019-170443.
Final model config (checkpoint, int8 npz, train mean/std): runs/final_model.json.

## Phase 5+6a
- LR finder (acenet/optim.py `lr_finder`, runs/lr_finder.{json,png}): 1e-6→1 exponential over 422 steps, no early stop. The smoothed-loss min is at 9.1e-2, so min/10 = 9.1e-3 (used). The steepest-descent point is 1.5e-4.
  9.1× A's lr, so E1 was run. Caveat: the EMA (0.98) lags about 0.7 decades, which biases both suggestions.
- CosineWarmup is per batch: linear warmup over 1 epoch, then cosine to 0 (tests in tests/test_optim.py).
- Shift aug (`data.shift_augment`, own rng seed 7): it runs after crop/flip and before erasing. It adds 1.85 ms per batch of 128 (0.83 → 2.68 ms).
- Selection rule, fixed before results: B vs D on val+TTA, with ties within 0.15 pt going to shifted-val. D − B = −0.30 pt, so **B is kept**. Test was not touched, and D was not quantized.
- Noise mechanism: clipping lifts the background mean to σ/√(2π), measured 0.05–0.11. A loses 24–30 pt to each component (background, object) alone and 64 pt to both. Erasing (B) fixes the background part; D fixes most of both.
- Wall times (s): E1 2679, E2 3569 (about 1,100 s of that is CPU contention from other apps in epochs 1–4), D 2528. Normal speed is 117–140 s per epoch.
- experiments/hidden_eval.py auto-detects .npz, idx(.gz) and PNG folders (stdlib PNG codec). It converts to 28×28 grayscale, logging each conversion, and runs the final float and int8 models with TTA.
  Self-test on shifted val (uint8): npz, idx and png give identical predictions (float 0.7672, int8 0.7672). A 56×56 RGB PNG variant agrees 100% with the reference.

## Key decisions
- Test-once guards: runs/mlp_test.json, cnn_test.json, cnn_ablation.json "test", cnn_b_int8.json "test", final_model.json "test_float".
- Selection is always on val (split seed 0; run seed 1). Calibration (ECE and int8 scales) uses train data only.
- Shifted val is not an independent check for D, because its augmentation covers the same kinds and ranges.

## Open issues
- The tradeoff: B is best by the fixed clean-val rule (noise 0.51), and D is far more robust (noise 0.875). For a shifted hidden set, D would be the better deployment. The rule chose B.
- E2's schedule gives +0.5 pt clean. D's train acc of 0.887 (augmented) and its best epoch of 20 suggest it underfits. An untested candidate is E2's recipe plus shift aug, trained longer.
- Shirt is still the weakest class (0.80–0.86).

## Phase 6b pre-registration (written 2026-10-19, before any 6b evaluation)
- (a) Selection rule revised: the final clean model is chosen by val+TTA among ALL val-evaluated runs (A, B, C, E1, E2, D) → **E2** (0.9450).
  Val-only selection; E2's test set is still untouched. B stays the int8 reference already reported.
- (b) Hidden-set report will cover exactly **E2, B and D** (float + int8, flip TTA). Declared before the hidden set is released.
- (c) Distillation teacher = **E2** (single model, no ensemble).

## Next: distillation (not started). Hidden set: run `python -m experiments.hidden_eval <path> --name hidden` and fill docs/robustness.md §5.
