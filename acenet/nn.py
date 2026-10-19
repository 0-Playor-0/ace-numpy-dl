"""Neural-network modules built on Tensor."""

import numpy as np

from . import ops
from .tensor import Tensor


class Module:
    """Base class. Parameters are Tensor attributes with requires_grad; children are Module
    attributes or lists/tuples of Modules. Both are discovered in attribute order.
    _buffers names ndarray attributes that belong in state_dict() but are not parameters."""

    _buffers = ()

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

    def named_modules(self, prefix=""):
        """(dotted prefix, module) for this module and all descendants, depth first, no recursion."""
        out, stack = [], [(prefix, self)]
        while stack:
            pre, m = stack.pop()
            out.append((pre, m))
            kids = [(f"{pre}{name}.", v) for name, v in m._members() if isinstance(v, Module)]
            stack.extend(reversed(kids))
        return out

    def modules(self):
        return [m for _, m in self.named_modules()]

    def named_parameters(self, prefix=""):
        return [(f"{pre}{name}", v) for pre, m in self.named_modules(prefix)
                for name, v in m._members() if isinstance(v, Tensor) and v.requires_grad]

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def named_buffers(self, prefix=""):
        """Non-trainable state (e.g. BN running stats) listed in each module's _buffers."""
        return [(f"{pre}{name}", m, name) for pre, m in self.named_modules(prefix) for name in m._buffers]

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
        """Parameters and buffers as {dotted name: ndarray copy}."""
        state = {name: p.data.copy() for name, p in self.named_parameters()}
        state.update({full: getattr(m, name).copy() for full, m, name in self.named_buffers()})
        return state

    def load_state_dict(self, state):
        params = dict(self.named_parameters())
        buffers = {full: (m, name) for full, m, name in self.named_buffers()}
        expected = set(params) | set(buffers)
        if expected != set(state):
            raise KeyError(f"state_dict keys differ: missing {sorted(expected - set(state))}, "
                           f"unexpected {sorted(set(state) - expected)}")
        for name, p in params.items():
            v = np.asarray(state[name])
            if v.shape != p.shape:
                raise ValueError(f"{name}: shape {v.shape} != {p.shape}")
            p.data = v.astype(p.dtype, copy=True)
        for full, (m, name) in buffers.items():
            cur = getattr(m, name)
            v = np.asarray(state[full])
            if v.shape != cur.shape:
                raise ValueError(f"{full}: shape {v.shape} != {cur.shape}")
            setattr(m, name, v.astype(cur.dtype, copy=True))


class Linear(Module):
    """y = x @ W + b. W ~ N(0, 2 / in_features) (He-normal, for ReLU nets), b = 0.
    bias=False drops b (redundant right before batch norm)."""

    def __init__(self, in_features, out_features, rng, bias=True, dtype=np.float32):
        super().__init__()
        std = np.sqrt(2.0 / in_features)
        self.W = Tensor((rng.standard_normal((in_features, out_features)) * std).astype(dtype),
                        requires_grad=True)
        self.b = Tensor(np.zeros(out_features, dtype=dtype), requires_grad=True) if bias else None

    def forward(self, x):
        out = x @ self.W
        return out + self.b if self.b is not None else out


class Conv2D(Module):
    """NCHW convolution. W (out, in, k, k) ~ N(0, 2 / (in * k * k)) (He-normal), b = 0.
    bias=False drops b (redundant right before batch norm)."""

    def __init__(self, in_channels, out_channels, kernel_size, rng, stride=1, padding=0, bias=True,
                 dtype=np.float32):
        super().__init__()
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        std = np.sqrt(2.0 / (in_channels * kh * kw))
        self.W = Tensor((rng.standard_normal((out_channels, in_channels, kh, kw)) * std).astype(dtype),
                        requires_grad=True)
        self.b = Tensor(np.zeros(out_channels, dtype=dtype), requires_grad=True) if bias else None
        self.stride, self.padding = stride, padding

    def forward(self, x):
        return ops.conv2d(x, self.W, self.b, self.stride, self.padding)


class MaxPool2D(Module):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return ops.maxpool2d(x, self.kernel_size, self.stride)


class _BatchNorm(Module):
    """Batch norm. Train: batch statistics, and running stats updated as
    running = (1 - momentum) * running + momentum * batch (unbiased var). Eval: running stats only."""

    _buffers = ("running_mean", "running_var")
    _axes = None

    def __init__(self, num_features, momentum=0.1, eps=1e-5, dtype=np.float32):
        super().__init__()
        self.gamma = Tensor(np.ones(num_features, dtype=dtype), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=dtype), requires_grad=True)
        self.running_mean = np.zeros(num_features, dtype=dtype)
        self.running_var = np.ones(num_features, dtype=dtype)
        self.momentum, self.eps = momentum, eps

    def forward(self, x):
        if x.ndim != len(self._axes) + 1:
            raise ValueError(f"{type(self).__name__} expects {len(self._axes) + 1}D input, got {x.shape}")
        if self.training:
            out, mean, var = ops.batchnorm(x, self.gamma, self.beta, self._axes, self.eps, return_stats=True)
            m = int(np.prod([x.shape[i] for i in self._axes]))
            mom = self.momentum
            self.running_mean = ((1 - mom) * self.running_mean + mom * mean).astype(self.running_mean.dtype)
            self.running_var = ((1 - mom) * self.running_var + mom * var * (m / max(m - 1, 1))
                                ).astype(self.running_var.dtype)
            return out
        shape = [1] * x.ndim
        shape[1] = -1
        # The running stats are constants; gamma/beta stay differentiable in eval.
        inv = Tensor((1 / np.sqrt(self.running_var + self.eps)).reshape(shape).astype(x.dtype))
        mean = Tensor(self.running_mean.reshape(shape).astype(x.dtype))
        return (x - mean) * inv * self.gamma.reshape(shape) + self.beta.reshape(shape)


class BatchNorm1d(_BatchNorm):
    """Over (N,) for (N, F) input."""
    _axes = (0,)


class BatchNorm2d(_BatchNorm):
    """Per channel over (N, H, W) for NCHW input."""
    _axes = (0, 2, 3)


class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class ReLU(Module):
    """ReLU. While self.tracking is set, self.alive accumulates which units (features, or channels
    for NCHW input) were ever positive."""

    def __init__(self):
        super().__init__()
        self.tracking = False
        self.alive = None

    def forward(self, x):
        out = ops.relu(x)
        if self.tracking:
            # a unit is a feature for (N, F) input, a channel for NCHW input
            axes = (0, 2, 3) if out.ndim == 4 else tuple(range(out.ndim - 1))
            seen = (out.data > 0).any(axis=axes)
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
