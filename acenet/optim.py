"""Optimizers. Updates are applied in place to each parameter's .data."""

import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with optional (Nesterov) momentum and L2 weight decay.

    g = grad + wd * p;  buf = momentum * buf + g;  p -= lr * (g + momentum * buf if nesterov else buf)
    """

    def __init__(self, params, lr, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(params, lr)
        if nesterov and momentum == 0:
            raise ValueError("nesterov needs momentum > 0")
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.bufs = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, buf in zip(self.params, self.bufs):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            if self.momentum:
                buf *= self.momentum
                buf += g
                g = g + self.momentum * buf if self.nesterov else buf
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam with bias correction.

    decoupled=False: weight decay is L2 added to the gradient (classic Adam).
    decoupled=True:  AdamW, p *= 1 - lr * wd before the Adam step.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 decoupled=False):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                if self.decoupled:
                    p.data *= 1 - self.lr * self.weight_decay
                else:
                    g = g + self.weight_decay * p.data
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p.data -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
