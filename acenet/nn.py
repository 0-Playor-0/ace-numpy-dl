"""Neural-network modules built on Tensor."""

import numpy as np

from . import ops
from .tensor import Tensor


class Module:
    """Base class. Parameters are Tensor attributes with requires_grad; children are Module
    attributes or lists/tuples of Modules. Both are discovered in attribute order."""

    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _members(self):
        for name, value in vars(self).items():
            if isinstance(value, (Tensor, Module)):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, (Tensor, Module)):
                        yield f"{name}.{i}", v

    def children(self):
        return [m for _, m in self._members() if isinstance(m, Module)]

    def modules(self):
        """This module and all descendants, depth first, without recursion."""
        out, stack = [], [self]
        while stack:
            m = stack.pop()
            out.append(m)
            stack.extend(reversed(m.children()))
        return out

    def named_parameters(self, prefix=""):
        out, stack = [], [(prefix, self)]
        while stack:
            pre, m = stack.pop()
            kids = []
            for name, v in m._members():
                full = f"{pre}{name}"
                if isinstance(v, Tensor):
                    if v.requires_grad:
                        out.append((full, v))
                else:
                    kids.append((full + ".", v))
            stack.extend(reversed(kids))
        return out

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self):
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        params = dict(self.named_parameters())
        if set(params) != set(state):
            raise KeyError(f"state_dict keys differ: missing {sorted(set(params) - set(state))}, "
                           f"unexpected {sorted(set(state) - set(params))}")
        for name, p in params.items():
            v = np.asarray(state[name])
            if v.shape != p.shape:
                raise ValueError(f"{name}: shape {v.shape} != {p.shape}")
            p.data = v.astype(p.dtype, copy=True)


class Linear(Module):
    """y = x @ W + b. W ~ N(0, 2 / in_features) (He-normal, for ReLU nets), b = 0."""

    def __init__(self, in_features, out_features, rng, dtype=np.float32):
        super().__init__()
        std = np.sqrt(2.0 / in_features)
        self.W = Tensor((rng.standard_normal((in_features, out_features)) * std).astype(dtype),
                        requires_grad=True)
        self.b = Tensor(np.zeros(out_features, dtype=dtype), requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.b


class ReLU(Module):
    """ReLU. While self.tracking is set, self.alive accumulates which units were ever positive."""

    def __init__(self):
        super().__init__()
        self.tracking = False
        self.alive = None

    def forward(self, x):
        out = ops.relu(x)
        if self.tracking:
            seen = (out.data > 0).reshape(-1, out.shape[-1]).any(axis=0)
            self.alive = seen if self.alive is None else self.alive | seen
        return out


class Dropout(Module):
    """Inverted dropout: zero each unit with probability p and scale survivors by 1/(1-p) in training;
    identity in eval."""

    def __init__(self, p, rng):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("dropout p must be in [0, 1)")
        self.p = p
        self.rng = rng

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (self.rng.random(x.shape) >= self.p).astype(x.dtype) / x.dtype.type(1 - self.p)
        return x * Tensor(keep)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)
