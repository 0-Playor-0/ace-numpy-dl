# ACE-Net progress (Phases 0–3b done; next: Phase 4)

| model | val | test (evaluated once) | source |
|---|---|---|---|
| MLP 784-256-128-10, SGD mom 0.9 lr 0.01 | 0.8985 | 0.8880 | runs/mlp_test.json |
| MLP, Adam lr 3e-4 | 0.8983 | 0.8916 | runs/mlp_test.json |
| CNN A (baseline, AdamW 1e-3 wd 5e-4, 20 ep, crop+flip) | 0.9362 | 0.9325 | runs/cnn_test.json |
| **CNN B + flip TTA (final CNN)** = A + random erasing | 0.9398 | **0.9349** | runs/cnn_ablation.json |
| CNN C = B + label smoothing 0.1 | 0.9360 (TTA 0.9383) | not evaluated | runs/cnn_ablation.json |

CNN: 468,138 params (1,828.7 KiB fp32) + 640 BN buffers. 4×conv3x3(32,32,64,64)+BN+ReLU, 2 maxpool, FC128+BN1d, dropout 0.3.
Checkpoints (best-val, `build_model` in experiments/cnn.py, `load_checkpoint` in acenet/train.py):
A runs/cnn_const_20261019-114445/best.npz (ep 20) · B runs/cnn_erase_20261019-123428/best.npz (ep 19) · C runs/cnn_erase_ls_20261019-132511/best.npz (ep 20)
ECE (15 bins, val): A 0.0066 · B 0.0066 · C 0.0848.

| shifted val (no TTA) | n | A | B | C |
|---|---|---|---|---|
| rotate ±15° | 2020 | 0.899 | 0.902 | 0.906 |
| noise σ 0.1–0.3 | 2081 | 0.293 | 0.506 | 0.450 |
| brightness ×(1±0.3) | 941 | 0.925 | 0.933 | 0.935 |
| brightness ±0.2 | 958 | 0.839 | 0.842 | 0.775 |
| clean val | 6000 | 0.936 | 0.938 | 0.936 |

## Key decisions
- Test-once rule: guard files (runs/mlp_test.json, cnn_test.json, cnn_ablation.json "test") refuse re-evaluation. Selection always on val (split seed 0; run seed 1).
- No conv/linear bias directly before BN (BN cancels it; gradient is exactly 0).
- No label smoothing for the teacher (C is miscalibrated, ECE 0.085, and worse on brightness-add).
- Calibration is fit on train data only (val stays for selection; test untouched).
- Crop padding uses standardized black (−0.81), not 0. Shifts applied on [0,1] before standardization.
- MLP: no dropout/wd in selected configs (all ≥ 88.5% val); overfitting write-up in docs/failure.md.

## Open issues
- Noise robustness: 0.29 (A) → 0.51 (B); still far below clean 0.94. Clipped noise lifts background 0 → 0.08 mean.
- Upper-body confusion: Shirt (0.80–0.86), T-shirt, Pullover, Coat trade errors; no variant fixes all four.
- B vs A+TTA margin is 3/6000 val images — within noise.

## Environment
- NumPy 2.5.4 on Accelerate (Apple M1, 8 cores = 4P+4E). sgemm ≈ 830 GFLOP/s; identical with VECLIB_MAXIMUM_THREADS = 1/4/8/default (Accelerate runs GEMM on the AMX unit, so thread count is not the bottleneck). No change made.
- CNN step: 281 ms/batch of 128 (fwd+bwd), ~115–155 s/epoch measured.
