"""Phase 6b: depthwise conv, KD loss, GAP/dwconv in the int8 engine, compact format, test-once guard."""

import json

import numpy as np
import pytest

from acenet import nn, ops
from acenet.quant import (Int8Model, build_qspec, calibrate, compact_bytes, float_forward, fold_bn, gap_int,
                          quantize_input)
from acenet.tensor import Tensor, no_grad
from tests.gradcheck import RESULTS, TOL, gradcheck

Y = np.array([0, 2, 1, 2, 0, 1])
TEACHER = 3 * np.random.default_rng(9).standard_normal((6, 3))


def normal(*shape):
    return lambda rng: rng.standard_normal(shape)


CASES = [
    ("dwconv_s1_p1_bias", "depthwise_conv2d", lambda x, W, b: ops.depthwise_conv2d(x, W, b, 1, 1),
     [normal(2, 3, 6, 5), normal(3, 1, 3, 3), normal(3)]),
    ("dwconv_s2_p0", "depthwise_conv2d", lambda x, W: ops.depthwise_conv2d(x, W, None, 2, 0),
     [normal(2, 4, 7, 6), normal(4, 1, 3, 3)]),
    ("dwconv_k2x3_s(1,2)_p(1,0)", "depthwise_conv2d",
     lambda x, W: ops.depthwise_conv2d(x, W, None, (1, 2), (1, 0)), [normal(2, 2, 5, 7), normal(2, 1, 2, 3)]),
    # the teacher is a constant: gradients are checked w.r.t. the student logits only
    ("kd_T4_a0.9", "kd_loss", lambda s: ops.kd_loss(s, TEACHER, Y, T=4.0, alpha=0.9), [normal(6, 3)]),
    ("kd_T2_a0.5", "kd_loss", lambda s: ops.kd_loss(s, TEACHER, Y, T=2.0, alpha=0.5), [normal(6, 3)]),
    ("kd_T1_a1", "kd_loss", lambda s: ops.kd_loss(s, TEACHER / 3, Y, T=1.0, alpha=1.0), [normal(6, 3)]),
    ("gap", "global_avg_pool", lambda x: nn.GlobalAvgPool2D()(x), [normal(2, 3, 4, 5)]),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_gradcheck(case):
    _, op, fn, makers = case
    rng = np.random.default_rng(0)
    err = gradcheck(fn, [m(rng) for m in makers])
    RESULTS[op] = max(RESULTS.get(op, 0.0), err)
    assert err < TOL, f"rel_err {err:.3e}"


def test_depthwise_matches_direct_loop():
    rng = np.random.default_rng(1)
    x, W, b = rng.standard_normal((2, 3, 6, 5)), rng.standard_normal((3, 1, 3, 3)), rng.standard_normal(3)
    out = ops.depthwise_conv2d(Tensor(x), Tensor(W), Tensor(b), 1, 1).data
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ref = np.zeros_like(out)
    for n in range(2):
        for c in range(3):
            for i in range(6):
                for j in range(5):
                    ref[n, c, i, j] = (xp[n, c, i:i + 3, j:j + 3] * W[c, 0]).sum() + b[c]
    np.testing.assert_allclose(out, ref, atol=1e-12)


def test_depthwise_equals_grouped_dense_conv():
    rng = np.random.default_rng(2)
    x, W = rng.standard_normal((2, 3, 5, 5)), rng.standard_normal((3, 1, 3, 3))
    dense = np.zeros((3, 3, 3, 3))
    for c in range(3):
        dense[c, c] = W[c, 0]
    np.testing.assert_allclose(ops.depthwise_conv2d(Tensor(x), Tensor(W), None, 1, 1).data,
                               ops.conv2d(Tensor(x), Tensor(dense), None, 1, 1).data, atol=1e-12)


def test_kd_loss_values():
    rng = np.random.default_rng(3)
    s, t = rng.standard_normal((6, 3)), 3 * rng.standard_normal((6, 3))
    T, a = 4.0, 0.9
    ps = np.exp(s / T) / np.exp(s / T).sum(1, keepdims=True)
    pt = np.exp(t / T) / np.exp(t / T).sum(1, keepdims=True)
    kl = (pt * (np.log(pt) - np.log(ps))).sum(1).mean()
    p = np.exp(s) / np.exp(s).sum(1, keepdims=True)
    ce = -np.log(p[np.arange(6), Y]).mean()
    got = ops.kd_loss(Tensor(s), t, Y, T=T, alpha=a).item()
    assert got == pytest.approx(a * T * T * kl + (1 - a) * ce, rel=1e-10)
    assert ops.kd_loss(Tensor(s), s, Y, T=T, alpha=1.0).item() == pytest.approx(0.0, abs=1e-12)
    assert ops.kd_loss(Tensor(s), t, Y, T=T, alpha=0.0).item() == pytest.approx(ce, rel=1e-12)


def test_kd_loss_teacher_gets_no_gradient():
    s, t = Tensor(np.zeros((6, 3)), requires_grad=True), Tensor(np.ones((6, 3)), requires_grad=True)
    ops.kd_loss(s, t, Y).backward()
    assert s.grad is not None and t.grad is None


def test_gap_int_rounds_half_up():
    x = np.zeros((1, 2, 2, 3), np.int8)
    x[0, :, :, 0] = [[1, 1], [0, 0]]       # 0.5  -> 1
    x[0, :, :, 1] = [[-1, -1], [0, 0]]     # -0.5 -> 0 (half up)
    x[0, :, :, 2] = [[127, 127], [127, 126]]  # 126.75 -> 127
    np.testing.assert_array_equal(gap_int(x), [[1, 0, 127]])


def ds_cnn(rng):
    """Small depthwise-separable CNN with GAP, non-trivial BN statistics, eval mode."""
    model = nn.Sequential(
        nn.Conv2D(1, 4, 3, rng, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU(), nn.MaxPool2D(2),
        nn.DepthwiseConv2D(4, 3, rng, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU(),
        nn.Conv2D(4, 8, 1, rng, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
        nn.GlobalAvgPool2D(), nn.Linear(8, 3, rng),
    )
    for m in model.layers:
        if isinstance(m, nn._BatchNorm):
            f = m.gamma.data.shape[0]
            m.gamma.data[:] = rng.uniform(0.5, 1.5, f)
            m.beta.data[:] = rng.normal(0, 0.3, f)
            m.running_mean[:] = rng.normal(0, 0.5, f)
            m.running_var[:] = rng.uniform(0.5, 2.0, f)
    return model.eval()


@pytest.fixture
def ds_setup():
    rng = np.random.default_rng(0)
    model = ds_cnn(rng)
    X = rng.normal(0, 1, (64, 1, 8, 8)).astype(np.float32)
    fops = fold_bn(model)
    qspec = build_qspec(fops, calibrate(fops, X, "p99.99"))
    return model, X, fops, qspec, Int8Model.from_qspec(fops, qspec)


def test_ds_fold_matches_eval_model(ds_setup):
    model, X, fops, _, _ = ds_setup
    assert [o["op"] for o in fops].count("dwconv") == 1 and [o["op"] for o in fops].count("gap") == 1
    with no_grad():
        ref = model(Tensor(X)).data
    assert np.abs(float_forward(fops, X) - ref).max() < 1e-4


def test_ds_engine_integer_and_matches_fake_quant(ds_setup):
    _, X, fops, qspec, qm = ds_setup
    trace = []
    zi = qm.forward(quantize_input(X, qm.input_scale), trace=trace)
    assert all(np.issubdtype(v.dtype, np.integer) for v in trace)
    zf = float_forward(fops, X, qspec=qspec)
    scale = qspec[-1]["s_in"] * qspec[-1]["s_w"].max()
    assert (zi.argmax(1) == zf.argmax(1)).mean() >= 0.95
    assert np.abs(zi * scale - zf).max() < 0.05 * np.abs(zf).max()


def test_compact_roundtrip(ds_setup, tmp_path):
    _, X, _, _, qm = ds_setup
    path = tmp_path / "m.ace8"
    n = qm.save_compact(path)
    assert path.stat().st_size == n == len(compact_bytes(qm))
    qm2 = Int8Model.load_compact(path)
    assert qm2.arch == json.loads(json.dumps(qm.arch)) and qm2.input_scale == np.float32(qm.input_scale)
    for a, b in zip(qm.layers, qm2.layers):
        for k in ("w", "b", "M0", "shift"):
            assert a[k].dtype == b[k].dtype
            np.testing.assert_array_equal(a[k].reshape(b[k].shape), b[k])
    np.testing.assert_array_equal(qm.logits(X, flip=True), qm2.logits(X, flip=True))
    assert compact_bytes(qm2) == compact_bytes(qm)
    payload = sum(L["w"].size + 9 * L["b"].size for L in qm.layers)
    assert n - payload < 64  # header is small


def test_compact_rejects_bad_files(ds_setup):
    from acenet.quant import from_compact_bytes
    buf = compact_bytes(ds_setup[-1])
    with pytest.raises(ValueError):
        from_compact_bytes(Int8Model, b"XXXX" + buf[4:])
    with pytest.raises(ValueError):
        from_compact_bytes(Int8Model, buf + b"\0")


def test_test_once_guard(tmp_path):
    from experiments.distill import once
    calls = []
    path = str(tmp_path / "t.json")
    assert once("m", lambda: calls.append(1) or {"acc": 0.5}, path) == {"acc": 0.5}
    assert once("m", lambda: calls.append(1) or {"acc": 0.9}, path) == {"acc": 0.5}
    assert calls == [1]
