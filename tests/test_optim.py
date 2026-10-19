import numpy as np
import pytest

from acenet.optim import SGD, Adam
from acenet.tensor import Tensor

A = np.array([1.0, 4.0, 0.5, 2.0])
C = np.array([0.3, -1.0, 2.0, 0.0])
W0 = np.array([1.0, 1.0, -1.0, 0.5])
STEPS = 3


def toy_grad(w):
    """Gradient of f(w) = 0.5 * sum(A * (w - C)**2)."""
    return A * (w - C)


def run_optimizer(make):
    w = Tensor(W0.copy(), requires_grad=True)
    opt = make([w])
    traj = []
    for _ in range(STEPS):
        opt.zero_grad()
        f = (Tensor(A) * (w - Tensor(C)) ** 2).sum() * 0.5
        f.backward()
        opt.step()
        traj.append(w.data.copy())
    return np.array(traj)


def ref_sgd(lr, momentum=0.0, nesterov=False, wd=0.0):
    w, buf, traj = W0.copy(), np.zeros_like(W0), []
    for _ in range(STEPS):
        g = toy_grad(w) + wd * w
        if momentum:
            buf = momentum * buf + g
            g = g + momentum * buf if nesterov else buf
        w = w - lr * g
        traj.append(w)
    return np.array(traj)


def ref_adam(lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=False):
    w, m, v, traj = W0.copy(), np.zeros_like(W0), np.zeros_like(W0), []
    for t in range(1, STEPS + 1):
        g = toy_grad(w)
        if wd and decoupled:
            w = w - lr * wd * w
        elif wd:
            g = g + wd * w
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(w)
    return np.array(traj)


CASES = [
    ("sgd", lambda p: SGD(p, lr=0.1), ref_sgd(0.1)),
    ("sgd_momentum", lambda p: SGD(p, lr=0.1, momentum=0.9), ref_sgd(0.1, 0.9)),
    ("sgd_nesterov", lambda p: SGD(p, lr=0.1, momentum=0.9, nesterov=True), ref_sgd(0.1, 0.9, True)),
    ("sgd_momentum_wd", lambda p: SGD(p, lr=0.1, momentum=0.9, weight_decay=0.01),
     ref_sgd(0.1, 0.9, wd=0.01)),
    ("adam", lambda p: Adam(p, lr=0.1), ref_adam(0.1)),
    ("adam_betas_eps", lambda p: Adam(p, lr=0.05, betas=(0.8, 0.99), eps=1e-6),
     ref_adam(0.05, 0.8, 0.99, 1e-6)),
    ("adam_l2", lambda p: Adam(p, lr=0.1, weight_decay=0.1), ref_adam(0.1, wd=0.1)),
    ("adamw", lambda p: Adam(p, lr=0.1, weight_decay=0.1, decoupled=True),
     ref_adam(0.1, wd=0.1, decoupled=True)),
]


@pytest.mark.parametrize("make,ref", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_matches_reference(make, ref):
    traj = run_optimizer(make)
    assert np.allclose(traj, ref, rtol=0, atol=1e-7), np.abs(traj - ref).max()
    assert not np.allclose(traj[-1], W0)


def test_adam_l2_differs_from_adamw():
    assert not np.allclose(run_optimizer(CASES[6][1]), run_optimizer(CASES[7][1]))


def test_zero_grad_and_skip_none():
    w = Tensor(W0.copy(), requires_grad=True)
    w.grad = np.ones_like(W0)
    opt = SGD([w], lr=0.1)
    opt.zero_grad()
    assert w.grad is None
    opt.step()
    assert np.array_equal(w.data, W0)


def test_cosine_warmup_values():
    from acenet.optim import CosineWarmup
    w = Tensor(np.zeros(2), requires_grad=True)
    opt = SGD([w], lr=0.1)
    spe, warm, total, lo = 10, 1, 5, 1e-3
    s = CosineWarmup(opt, warm, total, spe, min_lr=lo)
    W, T = warm * spe, total * spe
    assert s.lr_at(0) == pytest.approx(0.1 / W)                  # first warmup step
    assert s.lr_at(W - 1) == pytest.approx(0.1)                  # warmup end = base lr
    assert s.lr_at(W) == pytest.approx(0.1)                      # cosine start
    assert s.lr_at(W + (T - W) // 2) == pytest.approx((0.1 + lo) / 2)  # cosine midpoint
    assert s.lr_at(T) == pytest.approx(lo)                       # end
    lrs = [s.step() for _ in range(T)]
    assert opt.lr == lrs[-1] and lrs[W - 1] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(lrs[W - 1:], lrs[W:]))     # monotone decay after warmup
    assert lo < lrs[-1] < lo + 0.1 * (1 - np.cos(np.pi / (T - W))) / 2 + 1e-12  # last update just above min


def test_lr_finder_toy():
    from acenet import nn
    from acenet.data import DataLoader
    from acenet.optim import lr_finder
    rng = np.random.default_rng(0)
    X = rng.standard_normal((512, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    loader = DataLoader(X, y, 32, shuffle=True, rng=rng)
    res = lr_finder(lambda: nn.Sequential(nn.Linear(8, 2, np.random.default_rng(1))),
                    lambda p, lr: SGD(p, lr=lr), loader, lr_min=1e-4, lr_max=1e3, num_steps=200)
    assert 1e-4 <= res["suggested_min_div10"] < res["min_loss_lr"] <= 1e3
    assert res["min_smoothed_loss"] < res["smoothed"][0]
    assert np.all(np.diff(res["lrs"]) > 0)
