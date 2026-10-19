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


def _cnn_inputs(rng):
    """x, conv W, gamma, beta, linear W, linear b; BN outputs kept at |z| > 1e-2 for the ReLU.

    No conv bias: BN removes it, so its true gradient is exactly 0 and a relative error on it only
    measures roundoff (see test_bias_before_batchnorm_has_zero_grad)."""
    while True:
        x = rng.standard_normal((2, 1, 6, 5))
        Wc = rng.standard_normal((3, 1, 3, 3)) * 0.5
        gamma, beta = rng.uniform(0.5, 1.5, 3), rng.standard_normal(3) * 0.1
        z = ops.batchnorm(ops.conv2d(Tensor(x), Tensor(Wc), None, 1, 1),
                          Tensor(gamma), Tensor(beta), (0, 2, 3)).data
        if np.abs(z).min() > 1e-2:
            return [x, Wc, gamma, beta, rng.standard_normal((18, 4)) * 0.5, rng.standard_normal(4) * 0.1]


def _cnn(x, Wc, gamma, beta, Wl, bl):
    h = ops.relu(ops.batchnorm(ops.conv2d(x, Wc, None, 1, 1), gamma, beta, (0, 2, 3)))
    h = ops.maxpool2d(h, 2, 2).reshape(2, -1)  # (2, 3, 3, 2) -> (2, 18)
    return ops.softmax_cross_entropy(h @ Wl + bl, LABELS[:2])


CONV_CASES = [
    (f"conv2d_s{s}_p{p}", "conv2d",
     (lambda s, p: lambda x, W, b: ops.conv2d(x, W, b, stride=s, padding=p))(s, p),
     inputs(normal(2, 3, 7, 6), normal(4, 3, 3, 3), normal(4)))
    for s in (1, 2) for p in (0, 1)
] + [
    ("conv2d_kernel2x3_nobias", "conv2d", lambda x, W: ops.conv2d(x, W, None, stride=(2, 1), padding=(1, 0)),
     inputs(normal(2, 2, 5, 6), normal(3, 2, 2, 3))),
    ("maxpool_k2_s2", "maxpool2d", lambda x: ops.maxpool2d(x, 2, 2), inputs(normal(2, 3, 6, 8))),
    ("maxpool_k2_s2_odd", "maxpool2d", lambda x: ops.maxpool2d(x, 2, 2), inputs(normal(2, 2, 7, 5))),
    ("maxpool_k3_s2", "maxpool2d", lambda x: ops.maxpool2d(x, 3, 2), inputs(normal(2, 2, 7, 7))),
    ("batchnorm_(N)", "batchnorm", lambda x, g, b: ops.batchnorm(x, g, b, (0,)),
     inputs(normal(6, 4), positive(4), normal(4))),
    ("batchnorm_(N,H,W)", "batchnorm", lambda x, g, b: ops.batchnorm(x, g, b, (0, 2, 3)),
     inputs(normal(3, 2, 4, 5), positive(2), normal(2))),
    ("cnn_conv_bn_relu_pool_linear_ce", "cnn end-to-end", _cnn, _cnn_inputs),
]


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
    ("softmax_cross_entropy_ls0.1", "softmax_cross_entropy",
     lambda a: ops.softmax_cross_entropy(a, LABELS, label_smoothing=0.1), inputs(normal(8, 3))),
    ("bcast_(4,3)+(3,)", "broadcasting", lambda a, b: a + b, inputs(normal(4, 3), normal(3))),
    ("bcast_(4,1)*(1,5)", "broadcasting", lambda a, b: a * b, inputs(normal(4, 1), normal(1, 5))),
    ("bcast_(2,3,4)+(4,)", "broadcasting", lambda a, b: a + b, inputs(normal(2, 3, 4), normal(4))),
    ("bcast_(2,3,4)-(3,1)", "broadcasting", lambda a, b: a - b, inputs(normal(2, 3, 4), normal(3, 1))),
    ("bcast_(4,3)/(3,)", "broadcasting", lambda a, b: a / b, inputs(normal(4, 3), away_from_zero(3))),
    ("reused_x*x+x", "reused input", lambda x: x * x + x, inputs(normal(3, 4))),
    ("reused_deep", "reused input", lambda x: ((x * 2).exp() * x).sum() + (x * x).mean(), inputs(normal(3, 4))),
    ("mlp_2layer_sce", "mlp end-to-end", _mlp, _mlp_inputs),
] + CONV_CASES


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


def test_conv2d_matches_direct_loop():
    rng = np.random.default_rng(0)
    x, W, b = rng.standard_normal((2, 3, 6, 5)), rng.standard_normal((4, 3, 3, 3)), rng.standard_normal(4)
    out = ops.conv2d(Tensor(x), Tensor(W), Tensor(b), stride=2, padding=1).data
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ref = np.zeros((2, 4, 3, 3))
    for i in range(3):
        for j in range(3):
            patch = xp[:, :, 2 * i:2 * i + 3, 2 * j:2 * j + 3]
            ref[:, :, i, j] = np.einsum("nchw,fchw->nf", patch, W) + b
    assert np.allclose(out, ref)


def test_maxpool_ties_go_to_first_index():
    x = Tensor(np.ones((1, 1, 2, 2)), requires_grad=True)
    ops.maxpool2d(x, 2, 2).sum().backward()
    assert x.grad[0, 0].tolist() == [[1.0, 0.0], [0.0, 0.0]]


def test_batchnorm_normalizes():
    x = Tensor(np.random.default_rng(0).standard_normal((8, 3, 4, 4)) * 3 + 2)
    out, mu, var = ops.batchnorm(x, Tensor(np.ones(3)), Tensor(np.zeros(3)), (0, 2, 3), return_stats=True)
    assert np.allclose(out.data.mean(axis=(0, 2, 3)), 0, atol=1e-7)
    assert np.allclose(out.data.var(axis=(0, 2, 3)), 1, atol=1e-3)
    assert np.allclose(mu, x.data.mean(axis=(0, 2, 3))) and mu.shape == (3,)


def test_bias_before_batchnorm_has_zero_grad():
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((4, 2, 5, 5)))
    W, b = Tensor(rng.standard_normal((3, 2, 3, 3))), Tensor(rng.standard_normal(3), requires_grad=True)
    out = ops.batchnorm(ops.conv2d(x, W, b, 1, 1), Tensor(np.ones(3)), Tensor(np.zeros(3)), (0, 2, 3))
    out.backward(rng.standard_normal(out.shape))
    assert np.abs(b.grad).max() < 1e-10


def test_label_smoothing_matches_explicit_target():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((6, 4))
    y = np.array([0, 3, 1, 1, 2, 0])
    eps = 0.1
    target = (1 - eps) * np.eye(4)[y] + eps / 4
    logp = logits - logits.max(1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(1, keepdims=True))
    t = Tensor(logits, requires_grad=True)
    loss = ops.softmax_cross_entropy(t, y, label_smoothing=eps)
    assert np.isclose(loss.item(), -(target * logp).sum(1).mean())
    loss.backward()
    assert np.allclose(t.grad, (np.exp(logp) - target) / 6)
    assert np.isclose(ops.softmax_cross_entropy(Tensor(logits), y, 0.0).item(), -logp[np.arange(6), y].mean())
