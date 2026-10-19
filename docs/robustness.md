# Robustness to distribution shift (draft, Phase 5+6a)

Data: `data/shifted_val.npz`, the 6,000 val images with one shift each (rotation ±15°, Gaussian noise σ 0.1–0.3,
brightness ×(1±0.3) or ±0.2), applied on [0, 1] pixels and clipped. All numbers are val only, no TTA unless marked.
Sources: `runs/ablation.json`, `runs/robustness.json`, figures `runs/robustness_shift.png`, `runs/robustness_noise.png`.

**Caveat:** D was trained with augmentation from the same shift families and ranges (noise σ 0.05–0.3). For D,
shifted val measures how well it learned those shifts, not generalization to unseen shifts. The hidden set is the
only independent check.

## 1. A vs B vs D by shift kind

| group | n | A (crop+flip) | B (A + erasing) | E2 (A + finder lr + cosine) | D (B + shift aug + finder lr + cosine) |
|---|---|---|---|---|---|
| clean val | 6000 | 0.936 | 0.938 | **0.945** | 0.936 |
| rotate ±15° | 2020 | 0.899 | 0.902 | 0.907 | **0.925** |
| noise σ 0.1–0.3 | 2081 | 0.293 | 0.506 | 0.252 | **0.875** |
| brightness ×(1±0.3) | 941 | 0.925 | 0.933 | **0.936** | 0.932 |
| brightness ±0.2 | 958 | 0.839 | 0.842 | 0.822 | **0.934** |
| shifted, all | 6000 | 0.683 | 0.760 | 0.671 | **0.910** |

Rotation by |angle|, buckets 0–5° / 5–10° / 10–15°: A 0.929/0.899/0.871, B 0.937/0.906/0.866, D 0.937/0.923/0.917.

A better clean model doesn't mean a more robust one. E2 has the best clean accuracy but the worst noise accuracy (0.25).

## 2. Noise mechanism

Clipping Gaussian noise to [0, 1] does two different things to an FMNIST image (50.1% of pixels are exactly 0):
- **Background lift:** at a 0 pixel, only positive noise survives the clip, so the background mean becomes
  E[max(0, σz)] = σ/√(2π) ≈ 0.40σ. Measured: 0.050 / 0.071 / 0.090 / 0.110 for the σ bins below, which matches theory
  (0.050 / 0.070 / 0.090 / 0.109). The background turns into a grey, salt-only speckle field that never appears in training data.
- **Object speckle:** zero-mean noise on garment pixels. Mean |Δ| is 0.09 → 0.18 across the bins.

Counterfactuals use the exact noisy images, with the noise kept only on background pixels or only on object pixels
(the other region keeps its clean values):

| model | variant | σ 0.10–0.15 | 0.15–0.20 | 0.20–0.25 | 0.25–0.30 | all |
|---|---|---|---|---|---|---|
| A | full noise | 0.435 | 0.293 | 0.232 | 0.209 | 0.293 |
| A | background only | 0.738 | 0.641 | 0.582 | 0.572 | 0.634 |
| A | object only | 0.780 | 0.707 | 0.681 | 0.624 | 0.698 |
| B | full noise | 0.667 | 0.536 | 0.430 | 0.387 | 0.506 |
| B | background only | 0.900 | 0.875 | 0.858 | 0.850 | 0.871 |
| B | object only | 0.817 | 0.760 | 0.729 | 0.719 | 0.757 |
| D | full noise | 0.910 | 0.890 | 0.866 | 0.835 | 0.875 |
| D | background only | 0.925 | 0.920 | 0.925 | 0.908 | 0.919 |
| D | object only | 0.904 | 0.886 | 0.871 | 0.872 | 0.883 |

("Full noise with the background re-zeroed" is identical to "object only" by construction and is therefore omitted.)

- In A, each component on its own costs 24–30 pt. Together they cost 64 pt, which is far more than either alone,
  so the two effects compound.
- Random erasing (B) mostly fixes background lift (0.63 → 0.87): its N(0, 1) patches teach the model to ignore
  textured non-object regions. It helps object speckle much less (0.70 → 0.76).

## 3. What fixed it

D = B + shift augmentation (each image independently gets rotation, noise σ ~ U(0.05, 0.3) and brightness, each
with p = 0.3, applied in [0, 1] space with a clip) + finder lr 9.1e-3 + cosine warmup. Results:
- Noise accuracy goes from 0.506 to 0.875, and background lift is close to solved (0.919 when only the background is noisy).
- Brightness-add goes from 0.842 to 0.934, and rotation from 10–15° goes from 0.866 to 0.917.
- Cost: clean val drops to 0.936 (val+TTA 0.9368 vs B's 0.9398), ECE rises from 0.0066 to 0.0129, and the augmentation adds 1.85 ms per
  batch, which is under 1% of a step.
- On augmented training batches at epoch 20, D's train accuracy is 0.887, against 0.915 for B and 0.942 for E2.
  D's best val came at the last epoch, even though the learning rate had decayed to 0. Erasing plus shift
  augmentation is a harder task, and 20 epochs may be too few for it. This is not tested.

## 4. Remaining gap

- Noise is still 6 pt below clean (0.875 vs 0.936) and degrades with σ (0.91 → 0.84). Object speckle is the larger
  remaining part (0.883 with only the object noisy).
- Rotation from 10–15° is 2 pt below rotation under 5°.
- The fixed selection rule (val+TTA first) kept B as the final model: D is 0.30 pt lower on clean val. If the hidden
  set is shifted, B will reproduce the 0.29–0.51 noise accuracy. Deploying D, or a longer-trained D, is a decision for
  the hidden-set setting, not something to tune on val.
- Shirt is still the weakest class in every model (0.80–0.86).

## 5. Hidden-set results

_Placeholder: run `python -m experiments.hidden_eval <path> --name hidden` and record float and int8 (flip TTA)
accuracy, per-class accuracy, the confusion matrix and the format and conversion log from `runs/hidden_eval_hidden.json`._
