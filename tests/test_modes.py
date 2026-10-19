import numpy as np
import pytest

from acenet import nn
from acenet.tensor import Tensor


def model(rng):
    return nn.Sequential(
        nn.Sequential(nn.Conv2D(2, 3, 3, rng, padding=1, bias=False), nn.BatchNorm2d(3), nn.ReLU()),
        nn.MaxPool2D(2), nn.Flatten(), nn.Dropout(0.5, rng),
        nn.Linear(12, 4, rng, bias=False), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2, rng))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def x_batch(rng, n=8):
    return Tensor((rng.standard_normal((n, 2, 4, 4)) * 2 + 1).astype(np.float32))


def test_eval_train_propagate_through_nested_sequential(rng):
    m = model(rng)
    m.eval()
    assert len(m.modules()) == 12 and all(not mod.training for mod in m.modules())
    m.train()
    assert all(mod.training for mod in m.modules())


def test_dropout_identity_in_eval(rng):
    d = nn.Dropout(0.5, rng)
    x = Tensor(rng.standard_normal((16, 10)).astype(np.float32))
    d.eval()
    assert d(x) is x
    d.train()
    assert not np.array_equal(d(x).data, x.data)


@pytest.mark.parametrize("bn_cls,shape", [(nn.BatchNorm1d, (32, 5)), (nn.BatchNorm2d, (8, 5, 3, 4))])
def test_bn_train_updates_running_stats(rng, bn_cls, shape):
    bn = bn_cls(5)
    axes = bn._axes
    x = (rng.standard_normal(shape) * 3 + 2).astype(np.float32)
    bn(Tensor(x))
    m = int(np.prod([shape[i] for i in axes]))
    assert np.allclose(bn.running_mean, 0.9 * 0 + 0.1 * x.mean(axis=axes), atol=1e-6)
    assert np.allclose(bn.running_var, 0.9 * 1 + 0.1 * x.var(axis=axes) * m / (m - 1), atol=1e-5)
    assert np.allclose(x.var(axis=axes, ddof=1), x.var(axis=axes) * m / (m - 1))


@pytest.mark.parametrize("bn_cls,shape", [(nn.BatchNorm1d, (32, 5)), (nn.BatchNorm2d, (8, 5, 3, 4))])
def test_bn_eval_uses_running_stats_and_is_frozen(rng, bn_cls, shape):
    bn = bn_cls(5)
    for _ in range(3):
        bn(Tensor((rng.standard_normal(shape) * 3 + 2).astype(np.float32)))
    bn.gamma.data[:] = rng.uniform(0.5, 1.5, 5)
    bn.beta.data[:] = rng.standard_normal(5)
    rm, rv = bn.running_mean.copy(), bn.running_var.copy()
    bn.eval()
    x = Tensor((rng.standard_normal(shape)).astype(np.float32))
    a, b = bn(x).data, bn(x).data
    assert np.array_equal(a, b)
    assert np.array_equal(bn.running_mean, rm) and np.array_equal(bn.running_var, rv)
    view = (1, -1) + (1,) * (len(shape) - 2)
    ref = (x.data - rm.reshape(view)) / np.sqrt(rv.reshape(view) + bn.eps) * bn.gamma.data.reshape(view) \
        + bn.beta.data.reshape(view)
    assert np.allclose(a, ref, atol=1e-5)
    # one sample in eval does not depend on the rest of the batch
    assert np.allclose(bn(Tensor(x.data[:1])).data, a[:1], atol=1e-6)


def test_model_eval_is_deterministic_and_freezes_bn(rng):
    m = model(rng)
    for _ in range(2):
        m(x_batch(rng))
    m.eval()
    before = m.state_dict()
    x = x_batch(rng)
    assert np.array_equal(m(x).data, m(x).data)
    after = m.state_dict()
    assert all(np.array_equal(before[k], after[k]) for k in before)


def test_running_stats_in_state_dict_not_parameters(rng):
    m = model(rng)
    names = [n for n, _ in m.named_parameters()]
    sd = m.state_dict()
    assert "layers.0.layers.1.running_mean" in sd and "layers.5.running_var" in sd
    assert not any("running" in n for n in names)
    assert "layers.0.layers.1.gamma" in names and "layers.5.beta" in names
    m(x_batch(rng))
    m2 = model(np.random.default_rng(1))
    m2.load_state_dict(m.state_dict())
    m.eval(), m2.eval()
    x = x_batch(rng)
    assert np.array_equal(m(x).data, m2(x).data)
