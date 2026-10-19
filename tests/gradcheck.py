"""Finite-difference gradient checking (float64, central differences)."""

import numpy as np

from acenet.tensor import Tensor

EPS = 1e-6
TOL = 1e-5

# op name -> max rel_err seen across its cases; filled by tests, printed by conftest.
RESULTS = {}


def rel_err(a, n):
    return float(np.linalg.norm(a - n) / np.maximum(np.linalg.norm(a) + np.linalg.norm(n), 1e-12))


def gradcheck(fn, inputs, eps=EPS, seed=0):
    """Compare autograd gradients of fn against central differences.

    fn maps Tensors to a Tensor of any shape. A non-scalar output y is reduced to the
    scalar sum(y * R) with a fixed random R, applied outside autograd via y.backward(R).
    inputs: list of float arrays (cast to float64). Returns the max rel_err over inputs.
    """
    xs = [np.array(x, dtype=np.float64) for x in inputs]
    tensors = [Tensor(x.copy(), requires_grad=True) for x in xs]
    out = fn(*tensors)
    R = np.random.default_rng(seed).standard_normal(out.shape)
    out.backward(R)

    def f():
        return float((fn(*[Tensor(x) for x in xs]).data * R).sum())

    worst = 0.0
    for x, t in zip(xs, tensors):
        num = np.zeros_like(x)
        flat_x, flat_n = x.reshape(-1), num.reshape(-1)
        for i in range(flat_x.size):
            orig = flat_x[i]
            flat_x[i] = orig + eps
            fp = f()
            flat_x[i] = orig - eps
            fm = f()
            flat_x[i] = orig
            flat_n[i] = (fp - fm) / (2 * eps)
        ana = np.zeros_like(x) if t.grad is None else t.grad
        assert ana.dtype == np.float64
        worst = max(worst, rel_err(ana, num))
    return worst


def format_table(results=None):
    results = RESULTS if results is None else results
    width = max([len("op")] + [len(k) for k in results])
    lines = [f"{'op':<{width}} | max rel_err", f"{'-' * width}-+-------------"]
    lines += [f"{k:<{width}} | {v:.3e}" for k, v in results.items()]
    return "\n".join(lines)
