import numpy as np

from experiments.cnn_ablation import confusion, ece, per_class_acc


def test_ece_known_values():
    # all predictions at confidence 0.9 and 90% correct -> perfectly calibrated
    y = np.zeros(10, dtype=np.int64)
    p = np.tile([0.9, 0.1], (10, 1))
    p[9] = [0.1, 0.9]  # one wrong at confidence 0.9
    assert abs(ece(p, y) - 0.0) < 1e-12
    # confidence 0.6, all correct -> ECE 0.4
    assert abs(ece(np.tile([0.6, 0.4], (10, 1)), y) - 0.4) < 1e-12
    # two bins: half at 1.0 correct, half at 0.5 wrong -> 0.5 * |0 - 0.5| = 0.25
    p = np.array([[1.0, 0.0]] * 5 + [[0.5, 0.5]] * 5)
    y = np.array([0] * 5 + [1] * 5)  # argmax of [0.5, 0.5] is 0 -> wrong
    assert abs(ece(p, y) - 0.25) < 1e-12


def test_confusion_and_per_class():
    y = np.array([0, 0, 1, 2, 2, 2])
    pred = np.array([0, 1, 1, 2, 0, 2])
    cm = confusion(pred, y)
    assert cm.shape == (10, 10) and cm.sum() == 6
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[2, 0] == 1 and cm[2, 2] == 2
    full_y = np.repeat(np.arange(10), 2)
    full_p = full_y.copy()
    full_p[::2] = 0
    acc = per_class_acc(full_p, full_y)
    assert acc["T-shirt/top"] == 1.0 and acc["Trouser"] == 0.5
