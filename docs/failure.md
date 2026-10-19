# Failure: the unregularized MLP overfits (val loss doubles while val accuracy stays flat)

**Run:** `runs/mlp_overfit_adam_lr0.0003_20261019-112130/`: 784-256-128-10 MLP, Adam lr 3e-4
(the lr the sweep selected), batch 128, no dropout, no weight decay, 60 epochs, seed 1.
**Figures:** `experiments/figures/mlp_overfit.png` (the failure),
`experiments/figures/mlp_overfit_fix.png` (before/after).
**Diagnosis numbers:** `runs/mlp_overfit_diagnosis.json`, produced by `experiments/diagnose_overfit.py`.
That script re-runs both configs with the same seeds. Both re-runs reproduce the logged
`metrics.csv` exactly, then the script compares the epoch-20 and epoch-60 weights on the validation set.

All numbers below are on the validation split. The test set was not used here.

## Symptom

![overfitting run](../experiments/figures/mlp_overfit.png)

| epoch | train loss | train acc | val loss | val acc |
|---|---|---|---|---|
| 1  | 0.5389 | 0.8076 | 0.3991 | 0.8518 |
| 13 | 0.1765 | 0.9366 | **0.2846** (min) | 0.8980 |
| 20 | 0.1241 | 0.9564 | 0.3090 | **0.8983** (best, checkpointed) |
| 40 | 0.0512 | 0.9816 | 0.4229 | 0.8928 |
| 60 | 0.0272 | 0.9904 | 0.5826 | 0.8898 |

Train loss keeps falling, 20x from epoch 1 to 60. Val loss bottoms out at epoch 13 and then
rises steadily, ending 2.05x its minimum. Val accuracy is the odd one out: it plateaus near 89–90%
from about epoch 10 and only drifts down 0.85 points by epoch 60. The same shape appears in every
sweep run. For example, SGD lr 0.03 goes from val loss 0.305 at epoch 10 to 0.533 at epoch 40,
with val acc 0.8895 → 0.8920. So this is not specific to Adam.

## Diagnosis

The gap between rising val loss and flat val accuracy means the model is not mostly getting *more*
val samples wrong. It is getting the same kind of samples wrong *with more confidence*.
Comparing the best-val checkpoint (epoch 20) with the final weights (epoch 60):

| val-set statistic | epoch 20 | epoch 60 |
|---|---|---|
| misclassified samples | 610 | 661 (+51) |
| share of total val loss contributed by misclassified samples | 83.3% | 96.9% |
| mean NLL on correct samples | 0.057 | 0.020 |
| mean NLL on misclassified samples | 2.53 | 5.13 |
| mean max-softmax confidence on misclassified samples | 0.730 | 0.858 |
| misclassified samples predicted with confidence > 0.99 | 4.8% | 30.0% |
| mean logit range (max − min) per sample | 23.2 | 43.3 |
| global L2 norm of the weight matrices | 33.3 | 41.5 |

The model can always lower training cross-entropy by scaling up its logits on points it already
classifies correctly, and with no penalty on weight size it does: the weight norm grows 25% and the
logit range nearly doubles. On the training set, which reaches 99% accuracy, that only pays off. On
val, roughly 10% of samples are wrong whatever the epoch (at both epochs, 81% of errors are confusions
within the five upper-body classes: T-shirt, pullover, dress, coat, shirt). Those samples now get near-certain wrong predictions, and since NLL is unbounded,
their loss doubles (2.53 → 5.13 nats each). That is what drives val loss up. Accuracy only counts
the argmax, so it barely moves. This is memorization plus overconfidence, not an optimizer bug.
Training is stable throughout: the mean gradient norm per epoch stays between 0.93 and 2.07, and dead ReLUs
stay at ≤0.52%.

## Fix

There are two parts:

1. **Already in place: best-val checkpointing.** `fit()` keeps the best-val-accuracy weights
   (`best.npz`), so the model handed to evaluation is epoch 20 (val 0.8983), not epoch 60 (val
   0.8898). This contains the damage but does not remove the cause, and val loss at the checkpoint
   (0.309) is already past its minimum (0.285 at epoch 13).
2. **Regularization:** dropout p=0.3 after each hidden ReLU plus decoupled weight decay (AdamW,
   wd=0.05). Everything else is the same: Adam lr 3e-4, seed 1, 60 epochs. This is a single
   untuned run: `runs/mlp_fix_adam_lr0.0003_do0.3_wd0.05_20261019-112223/`.

## Before / after

![before/after](../experiments/figures/mlp_overfit_fix.png)

| | no regularization | dropout 0.3 + AdamW 0.05 |
|---|---|---|
| min val loss (epoch) | 0.2846 (13) | **0.2697** (36) |
| val loss at epoch 60 | 0.5826 | **0.2913** |
| val loss rise, min → epoch 60 | +105% | +8% |
| best val acc (epoch) | 0.8983 (20) | **0.9040** (57) |
| val acc at epoch 60 | 0.8898 | **0.9012** |
| misclassified val samples > 0.99 confident, at epoch 60 | 30.0% | 5.9% |
| weight L2 norm, at epoch 60 | 41.5 | 31.4 |
| wall time | 32.5 s | 40.8 s |

With regularization, val loss is essentially flat from epoch ~20 to 60, and val accuracy keeps
improving slowly instead of decaying. Overconfident errors stay at the level the unregularized
model had at its best epoch. Train loss in the fix run is higher (0.147 vs 0.027 at epoch 60), but
it is a running average measured *with dropout active*, so it is not directly comparable.

## Note on test selection

The Phase 2 rule was to add dropout/weight decay only if val acc fell below 88.5%. Every sweep
config cleared that: best SGD 0.8985, best Adam 0.8983. So the configs evaluated on test are the
unregularized sweep runs, and this fix run (tag `fix`) was deliberately excluded from test
selection, even though its val accuracy is higher. It has not been evaluated on test.
