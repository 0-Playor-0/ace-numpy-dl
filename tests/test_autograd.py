import numpy as np
import pytest

from acenet import ops
from acenet.tensor import Tensor, no_grad
from tests.gradcheck import RESULTS, TOL, gradcheck


def normal(*shape):
    return lambda rng: rng.standard_normal(shape)


def positive(*shape):
    return lambda rng: rng.uniform(0.5, 2.0, shape)


def away_from_zero(*shape):
    """|x| in [0.1, 1]: keeps ReLU kinks and division poles out of the finite-difference stencil."""
    return lambda rng: rng.uniform(0.1, 1.0, shape) * rng.choice([-1.0, 1.0], shape)


def inputs(*makers):
    return lambda rng: [m(rng) for m in makers]


LABELS = np.array([0, 2, 1, 2, 0, 1, 1, 2])


def _mlp_inputs(rng):
    """X, W1, b1, W2, b2 with every hidden pre-activation at |z| > 1e-2."""
    while True:
        X = rng.standard_normal((8, 5))
        W1 = rng.standard_normal((5, 6)) * 0.5
        b1 = rng.standard_normal(6) * 0.1
        if np.abs(X @ W1 + b1).min() > 1e-2:
            return [X, W1, b1, rng.standard_normal((6, 3)) * 0.5, rng.standard_normal(3) * 0.1]


def _mlp(X, W1, b1, W2, b2):
    h = ops.relu(X @ W1 + b1)
    return ops.softmax_cross_entropy(h @ W2 + b2, LABELS)


# (case id, op name for the table, fn, input maker)
CASES = [
    ("add", "add", lambda a, b: a + b, inputs(normal(3, 4), normal(3, 4))),
    ("sub", "sub", lambda a, b: a - b, inputs(normal(3, 4), normal(3, 4))),
    ("neg", "neg", lambda a: -a, inputs(normal(3, 4))),
    ("mul", "mul", lambda a, b: a * b, inputs(normal(3, 4), normal(3, 4))),
    ("div", "div", lambda a, b: a / b, inputs(normal(3, 4), away_from_zero(3, 4))),
    ("scalar_rsub_rdiv", "div", lambda a: (2.0 - a) / 3.0 + 1.5 / a, inputs(positive(3, 4))),
    ("pow_3", "pow", lambda a: a ** 3, inputs(normal(3, 4))),
    ("pow_0.5", "pow", lambda a: a ** 0.5, inputs(positive(3, 4))),
    ("pow_-1.5", "pow", lambda a: a ** -1.5, inputs(positive(3, 4))),
    ("matmul_2d", "matmul", lambda a, b: a @ b, inputs(normal(3, 4), normal(4, 5))),
    ("matmul_batched", "matmul", lambda a, b: a @ b, inputs(normal(2, 3, 4), normal(2, 4, 5))),
    ("matmul_batch_bcast", "matmul", lambda a, b: a @ b, inputs(normal(2, 3, 4), normal(4, 5))),
    ("matmul_batch_bcast_1", "matmul", lambda a, b: a @ b, inputs(normal(1, 3, 4), normal(2, 4, 5))),
    ("sum_all", "sum", lambda a: a.sum(), inputs(normal(3, 4))),
    ("sum_axis0", "sum", lambda a: a.sum(axis=0), inputs(normal(3, 4))),
    ("sum_axis1_keepdims", "sum", lambda a: a.sum(axis=1, keepdims=True), inputs(normal(3, 4))),
    ("sum_axes02_3d", "sum", lambda a: a.sum(axis=(0, 2)), inputs(normal(2, 3, 4))),
    ("sum_neg_axis_keepdims", "sum", lambda a: a.sum(axis=-1, keepdims=True), inputs(normal(2, 3, 4))),
    ("mean_all", "mean", lambda a: a.mean(), inputs(normal(3, 4))),
    ("mean_axis0", "mean", lambda a: a.mean(axis=0), inputs(normal(3, 4))),
    ("mean_axis1_keepdims", "mean", lambda a: a.mean(axis=1, keepdims=True), inputs(normal(3, 4))),
    ("mean_axes12_keepdims", "mean", lambda a: a.mean(axis=(1, 2), keepdims=True), inputs(normal(2, 3, 4))),
    ("max_all", "max", lambda a: a.max(), inputs(normal(3, 4))),
    ("max_axis1", "max", lambda a: a.max(axis=1), inputs(normal(3, 4))),
    ("max_axis0_keepdims", "max", lambda a: a.max(axis=0, keepdims=True), inputs(normal(3, 4))),
    ("max_axis2_3d", "max", lambda a: a.max(axis=2), inputs(normal(2, 3, 4))),
    ("reshape", "reshape", lambda a: a.reshape(4, 6), inputs(normal(2, 3, 4))),
    ("reshape_flatten", "reshape", lambda a: a.reshape(2, -1), inputs(normal(2, 3, 4))),
    ("transpose_axes", "transpose", lambda a: a.transpose(1, 2, 0), inputs(normal(2, 3, 4))),
    ("transpose_T", "transpose", lambda a: a.T, inputs(normal(3, 4))),
    ("exp", "exp", lambda a: a.exp(), inputs(normal(3, 4))),
    ("log", "log", lambda a: a.log(), inputs(positive(3, 4))),
    ("relu", "relu", lambda a: a.relu(), inputs(away_from_zero(3, 4))),
    ("log_softmax_last", "log_softmax", lambda a: ops.log_softmax(a), inputs(normal(4, 5))),
    ("log_softmax_axis0", "log_softmax", lambda a: ops.log_softmax(a, axis=0), inputs(normal(4, 5))),
    ("log_softmax_3d", "log_softmax", lambda a: ops.log_softmax(a, axis=1), inputs(normal(2, 3, 4))),
    ("softmax_cross_entropy", "softmax_cross_entropy",
     lambda a: ops.softmax_cross_entropy(a, LABELS), inputs(normal(8, 3))),
    ("bcast_(4,3)+(3,)", "broadcasting", lambda a, b: a + b, inputs(normal(4, 3), normal(3))),
    ("bcast_(4,1)*(1,5)", "broadcasting", lambda a, b: a * b, inputs(normal(4, 1), normal(1, 5))),
    ("bcast_(2,3,4)+(4,)", "broadcasting", lambda a, b: a + b, inputs(normal(2, 3, 4), normal(4))),
    ("bcast_(2,3,4)-(3,1)", "broadcasting", lambda a, b: a - b, inputs(normal(2, 3, 4), normal(3, 1))),
    ("bcast_(4,3)/(3,)", "broadcasting", lambda a, b: a / b, inputs(normal(4, 3), away_from_zero(3))),
    ("reused_x*x+x", "reused input", lambda x: x * x + x, inputs(normal(3, 4))),
    ("reused_deep", "reused input", lambda x: ((x * 2).exp() * x).sum() + (x * x).mean(), inputs(normal(3, 4))),
    ("mlp_2layer_sce", "mlp end-to-end", _mlp, _mlp_inputs),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_gradcheck(case):
    _, op, fn, make = case
    err = gradcheck(fn, make(np.random.default_rng(0)))
    RESULTS[op] = max(RESULTS.get(op, 0.0), err)
    assert err < TOL, f"rel_err {err:.3e}"


# Non-gradcheck behaviour ---------------------------------------------------------

def test_max_ties_split_evenly():
    x = Tensor(np.array([[1.0, 3.0, 3.0], [2.0, 2.0, 2.0]]), requires_grad=True)
    x.max(axis=1).sum().backward()
    assert np.allclose(x.grad, [[0, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]])


def test_backward_needs_grad_for_nonscalar():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()


def test_gradients_accumulate_across_backward_calls():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    (x * x).sum().backward()
    (x * x).sum().backward()
    assert np.allclose(x.grad, 4 * x.data)
    x.zero_grad()
    assert x.grad is None


def test_no_grad_skips_graph():
    x = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        y = (x * 2).sum()
    assert not y.requires_grad and y._parents == () and y._backward is None
    assert ((x * 2).sum()).requires_grad


def test_deep_chain_no_recursion_limit():
    x = Tensor(np.array(1.0), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 0.0
    y.backward()
    assert x.grad == 1.0


def test_float32_preserved():
    x = Tensor(np.ones((4, 3), dtype=np.float32), requires_grad=True)
    w = Tensor(np.ones((3, 2), dtype=np.float32), requires_grad=True)
    loss = ops.softmax_cross_entropy(ops.relu(x @ w + 1.0) * 0.5 - 1.0, np.array([0, 1, 0, 1]))
    assert loss.dtype == np.float32
    loss.backward()
    assert x.grad.dtype == np.float32 and w.grad.dtype == np.float32


def test_ndarray_on_left_dispatches_to_tensor():
    x = Tensor(np.ones(3), requires_grad=True)
    y = np.full(3, 2.0) * x
    assert isinstance(y, Tensor)
    y.sum().backward()
    assert np.allclose(x.grad, 2.0)
