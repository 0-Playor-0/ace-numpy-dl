import numpy as np

from acenet import nn
from acenet.tensor import Tensor


def make_mlp(rng, p=0.5):
    return nn.Sequential(nn.Linear(6, 5, rng), nn.ReLU(), nn.Dropout(p, rng), nn.Linear(5, 3, rng))


def test_parameters_and_names():
    model = make_mlp(np.random.default_rng(0))
    names = [n for n, _ in model.named_parameters()]
    assert names == ["layers.0.W", "layers.0.b", "layers.3.W", "layers.3.b"]
    assert len(model.parameters()) == 4


def test_linear_init():
    lin = nn.Linear(2000, 300, np.random.default_rng(0))
    assert lin.W.dtype == np.float32 and lin.W.shape == (2000, 300)
    assert abs(lin.W.data.std() - np.sqrt(2 / 2000)) < 1e-3
    assert np.all(lin.b.data == 0)


def test_train_eval_propagates():
    model = make_mlp(np.random.default_rng(0))
    model.eval()
    assert all(not m.training for m in model.modules())
    model.train()
    assert all(m.training for m in model.modules())


def test_dropout():
    rng = np.random.default_rng(0)
    d = nn.Dropout(0.25, rng)
    x = Tensor(np.ones((2000, 50), dtype=np.float32))
    out = d(x).data
    assert set(np.unique(out)) <= {0.0, np.float32(1 / 0.75)}
    assert abs((out == 0).mean() - 0.25) < 0.01
    assert abs(out.mean() - 1) < 0.02
    d.eval()
    assert d(x) is x


def test_state_dict_roundtrip():
    a, b = make_mlp(np.random.default_rng(0)), make_mlp(np.random.default_rng(1))
    x = Tensor(np.random.default_rng(2).standard_normal((4, 6)).astype(np.float32))
    a.eval(), b.eval()
    assert not np.allclose(a(x).data, b(x).data)
    sd = a.state_dict()
    b.load_state_dict(sd)
    assert np.array_equal(a(x).data, b(x).data)
    sd["layers.0.W"][:] = 0  # state_dict holds copies
    assert not np.all(a.layers[0].W.data == 0)


def test_relu_alive_tracking():
    r = nn.ReLU()
    r.tracking = True
    r(Tensor(np.array([[1.0, -1.0, -2.0], [-1.0, -1.0, -3.0]])))
    r(Tensor(np.array([[-1.0, -1.0, 3.0]])))
    assert r.alive.tolist() == [True, False, True]
