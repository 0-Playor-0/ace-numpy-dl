import numpy as np
import pytest

from acenet import nn
from acenet.quant import (INT32_MAX, Int8Model, build_qspec, calibrate, check_accumulator_bound,
                          float_forward, fold_bn, int_matmul, quantize_input, quantize_multiplier,
                          requantize)
from acenet.tensor import Tensor, no_grad


def small_cnn(rng):
    model = nn.Sequential(
        nn.Conv2D(1, 4, 3, rng, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU(), nn.MaxPool2D(2),
        nn.Conv2D(4, 6, 3, rng, padding=1, bias=False), nn.BatchNorm2d(6), nn.ReLU(), nn.MaxPool2D(2),
        nn.Flatten(), nn.Dropout(0.3, rng),
        nn.Linear(6 * 2 * 2, 8, rng, bias=False), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 3, rng),
    )
    # non-trivial BN statistics and affine params
    for m in model.layers:
        if isinstance(m, nn._BatchNorm):
            f = m.gamma.data.shape[0]
            m.gamma.data[:] = rng.uniform(0.5, 1.5, f)
            m.beta.data[:] = rng.normal(0, 0.3, f)
            m.running_mean[:] = rng.normal(0, 0.5, f)
            m.running_var[:] = rng.uniform(0.5, 2.0, f)
    model.eval()
    return model


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    model = small_cnn(rng)
    X = rng.normal(0, 1, (16, 1, 8, 8)).astype(np.float32)
    ops = fold_bn(model)
    qspec = build_qspec(ops, calibrate(ops, X, "minmax"))
    return model, X, ops, qspec


def test_fold_matches_eval_model(setup):
    model, X, ops, _ = setup
    with no_grad():
        ref = model(Tensor(X)).data
    assert np.abs(float_forward(ops, X) - ref).max() < 1e-4


def test_engine_intermediates_are_integer(setup):
    _, X, ops, qspec = setup
    qm = Int8Model.from_qspec(ops, qspec)
    trace = []
    out = qm.forward(quantize_input(X, qm.input_scale), trace=trace)
    assert len(trace) > 10
    for v in trace:
        assert np.issubdtype(v.dtype, np.integer), v.dtype
    assert out.dtype == np.int32
    for L in qm.layers:
        assert L["w"].dtype == np.int8 and L["b"].dtype == np.int32
        assert L["M0"].dtype == np.int32 and L["shift"].dtype == np.int8


def test_engine_agrees_with_fake_quant(setup):
    _, X, ops, qspec = setup
    qm = Int8Model.from_qspec(ops, qspec)
    a = qm.forward(quantize_input(X, qm.input_scale)).argmax(1)
    b = float_forward(ops, X, qspec=qspec).argmax(1)
    assert (a == b).mean() >= 0.9


def test_save_load_roundtrip(setup, tmp_path):
    _, X, ops, qspec = setup
    qm = Int8Model.from_qspec(ops, qspec)
    qm.save(tmp_path / "m.npz")
    qm2 = Int8Model.load(tmp_path / "m.npz")
    xq = quantize_input(X, qm.input_scale)
    np.testing.assert_array_equal(qm.forward(xq), qm2.forward(xq))


def test_multiplier_precision():
    rng = np.random.default_rng(1)
    M = 10.0 ** rng.uniform(-8, 1, 1000)
    M0, rs = quantize_multiplier(M)
    assert M0.min() >= 2 ** 30 and M0.max() < 2 ** 31
    np.testing.assert_allclose(M0 * 2.0 ** -rs.astype(np.float64), M, rtol=2 ** -30)


def test_requantize_within_one_lsb():
    rng = np.random.default_rng(2)
    F = 16
    acc = rng.integers(-2 ** 24, 2 ** 24, (5000, F)).astype(np.int32)
    acc[:10] = rng.integers(-INT32_MAX, INT32_MAX, (10, F))  # extreme accumulators
    M = 10.0 ** rng.uniform(-7, -1, F)
    M0, rs = quantize_multiplier(M)
    q = requantize(acc, M0, rs)
    assert q.dtype == np.int8
    ref = np.clip(np.floor(acc.astype(np.float64) * M + 0.5), -127, 127)
    assert np.abs(q.astype(np.int64) - ref).max() <= 1
    # half-way points round up
    M0h, rsh = quantize_multiplier(np.array([0.5]))
    np.testing.assert_array_equal(requantize(np.array([[1], [3], [-1], [-3]], np.int32), M0h, rsh).ravel(),
                                  [1, 2, 0, -1])


def test_int_matmul_exact():
    rng = np.random.default_rng(3)
    a = rng.integers(-127, 128, (50, 300)).astype(np.int8)
    w = rng.integers(-127, 128, (7, 300)).astype(np.int8)
    b = rng.integers(-1000, 1000, 7).astype(np.int32)
    out = int_matmul(a, w.astype(np.int32), b)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, a.astype(np.int64) @ w.astype(np.int64).T + b)


def test_overflow_assert_fires():
    K = 140_000  # 127 * 127 * 140000 > 2^31 - 1
    a = np.zeros((1, K), np.int8)
    w = np.zeros((1, K), np.int32)
    with pytest.raises(AssertionError, match="overflow"):
        int_matmul(a, w, np.zeros(1, np.int32))
    check_accumulator_bound(3136, np.zeros(1, np.int32))  # the real FC1 fan-in fits
    with pytest.raises(AssertionError):
        check_accumulator_bound(100, np.array([INT32_MAX], np.int64))
