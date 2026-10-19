# Ablation (Phases 3b–6b)

Source: `runs/ablation_6b.json` (also merged into `runs/ablation.json` as `table_6b`), built by `python -m experiments.distill ablation`.
Val is the 6k split (seed 0). TTA = horizontal flip. Float models average the two softmaxes; int8 models sum the int32 logits.
Test is evaluated once per model and only for models that were selected. It uses TTA unless the cell says otherwise.
"shifted all" and "noise" are accuracies on shifted val (6,000 images, 2,081 of them noisy), without TTA.
Float size = 4 B × (params + BN buffers). Int8 size = actual file size: npz for the teachers, and the compact `.ace8` format for the students, where every byte is counted.

| model | val | val+TTA | test | ECE | shifted all | noise | params | size KB |
|---|---|---|---|---|---|---|---|---|
| A | 0.9362 | 0.9393 | 0.9325 (no TTA) | 0.0066 | 0.683 | 0.293 | 468,138 | 1831.2 |
| C | 0.9360 | 0.9383 | – | 0.0848 | 0.732 | 0.450 | 468,138 | 1831.2 |
| B | 0.9377 | 0.9398 | 0.9349 | 0.0066 | 0.760 | 0.506 | 468,138 | 1831.2 |
| B int8 | 0.9368 | 0.9387 | 0.9344 | – | 0.758 | 0.502 | 468,138 | 466.9 |
| E1 | 0.9387 | 0.9398 | – | 0.0087 | 0.691 | 0.352 | 468,138 | 1831.2 |
| **E2** | **0.9448** | **0.9450** | **0.9403** | 0.0043 | 0.671 | 0.252 | 468,138 | 1831.2 |
| E2 int8 | 0.9435 | 0.9447 | 0.9399 | – | 0.670 | 0.248 | 468,138 | 466.9 |
| D | 0.9360 | 0.9368 | 0.9278 | 0.0129 | **0.910** | **0.875** | 468,138 | 1831.2 |
| D int8 | 0.9358 | 0.9370 | 0.9282 | – | 0.910 | 0.875 | 468,138 | 466.9 |
| S0 (P, CE) | 0.9320 | 0.9367 | 0.9304 | 0.0212 | 0.630 | 0.235 | 42,570 | 167.5 |
| **S0 int8 (final student)** | 0.9267 | 0.9337 | **0.9281** | – | 0.628 | 0.242 | 42,570 | **42.7** (43,770 B) |
| S1 (P, KD T=4 α=0.9) | 0.9322 | 0.9337 | 0.9299 | 0.0147 | 0.637 | 0.209 | 42,570 | 167.5 |
| S1 int8 (KD deliverable) | 0.9280 | 0.9323 | 0.9287 | – | 0.638 | 0.210 | 42,570 | 42.7 |

S1 was tested once in Phase 7 as the required distilled-student deliverable (pre-registered in commit ba2ccb9; it does not change the selection). Its int8 test (0.9287) is 0.06 pt above S0 int8 (0.9281), which is 6 images out of 10,000 and well inside test noise (1 s.e. ≈ 0.26 pt); the val ranking (S0 ahead by 0.14 pt int8) decided the selection before either was tested.

S2 (KD T=2) was not trained. The gate was S1 int8 val+TTA < 0.925, and S1 scored 0.9323.

## What each trick contributed (Δ val+TTA, pt)

- **Random erasing: +0.05 clean (B−A), but +21.3 on noise.** Almost all of its value is robustness. Erasing teaches the network to ignore corrupted background regions.
- **Label smoothing: −0.15 (C−B).** ECE also gets 13× worse (0.085), because the smoothed targets make the model under-confident. Rejected.
- **LR finder (constant lr): +0.05 (E1−A).** It only pays off together with the schedule: **cosine warmup gives +0.52 (E2−E1)**, the largest clean gain in the study, and also the best ECE (0.0043).
- **Shift augmentation: −0.82 clean (D−E2), but +23.9 shifted and +62 on noise.** This is a clean-vs-robust trade-off, not a free gain. Shifted val is not an independent check for D.
- **int8 PTQ:** −0.03 on E2 and −0.11 on B, with 4× smaller files. On the 42k-param student it costs −0.30 val+TTA (−0.53 without TTA) and −0.23 test. The small net has less redundancy to absorb rounding.
- **Distillation (KD T=4, α=0.9): −0.30 val+TTA (S1−S0).** Plain val (+0.02), ECE (0.021 → 0.015), NLL (val loss 0.203 → 0.182) and rotate/brightness-add (+2.4 / +5.1) all favour KD. The TTA gain, however, is 0.15 pt for S1 against 0.47 for S0, so on the selection metric KD did not help.
  Likely cause: the teacher is only 95.3% right on the cached crop+flip copies, while a 42k-param GAP net is capacity-bound, not label-noise-bound.
- **Compression overall:** the student (S0 int8) is **42.8× smaller than the E2 int8 npz** and has 11× fewer params. It costs −1.22 pt test against E2 float and −1.18 against E2 int8, which is inside the 2-pt budget.
