import numpy as np

from acenet import ops
from acenet.data import load_splits
from acenet.tensor import Tensor, no_grad


def test_mlp_overfits_64_samples():
    rng = np.random.default_rng(0)
    d = load_splits(rng, flatten=True)
    X, y = d["X_train"][:64], d["y_train"][:64]

    hidden = 128
    W1 = Tensor((rng.standard_normal((784, hidden)) * np.sqrt(2 / 784)).astype(np.float32), requires_grad=True)
    b1 = Tensor(np.zeros(hidden, dtype=np.float32), requires_grad=True)
    W2 = Tensor((rng.standard_normal((hidden, 10)) * np.sqrt(1 / hidden)).astype(np.float32), requires_grad=True)
    b2 = Tensor(np.zeros(10, dtype=np.float32), requires_grad=True)
    params = [W1, b1, W2, b2]
    x = Tensor(X)

    lr = np.float32(0.05)
    for step in range(1, 500):
        logits = ops.relu(x @ W1 + b1) @ W2 + b2
        loss = ops.softmax_cross_entropy(logits, y)
        for p in params:
            p.zero_grad()
        loss.backward()
        with no_grad():
            for p in params:
                p.data -= lr * p.grad
        acc = (logits.data.argmax(axis=1) == y).mean()
        if acc == 1.0:
            break
    print(f"100% train accuracy at step {step}, loss {loss.item():.4f}")
    assert acc == 1.0, f"train acc {acc:.3f} after {step} steps"
    assert W1.data.dtype == np.float32
