"""Differentiable operations on Tensors. Each op computes its forward value with NumPy and
attaches a closure that maps the output gradient to one gradient per input."""

import builtins

import numpy as np

from .tensor import Tensor, is_grad_enabled


# Helpers -------------------------------------------------------------------------

def _as_tensor(x, like=None):
    if isinstance(x, Tensor):
        return x
    dtype = like.dtype if like is not None and np.issubdtype(like.dtype, np.floating) else None
    return Tensor(np.asarray(x, dtype=dtype))


def _pair(a, b):
    like = a if isinstance(a, Tensor) else b
    return _as_tensor(a, like), _as_tensor(b, like)


def _make(data, parents, backward):
    out = Tensor(data)
    if is_grad_enabled() and builtins.any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._parents = parents
        out._backward = backward
    return out


def _axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(a % ndim for a in axis))


def _expand(g, shape, axes, keepdims):
    """Broadcast a reduced gradient back to the input shape."""
    if not keepdims:
        g = np.expand_dims(g, axes)
    return np.broadcast_to(g, shape)


def unbroadcast(grad, shape):
    """Sum grad down to shape: over leading dims NumPy added and over dims that were 1."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


# Elementwise ---------------------------------------------------------------------

def add(a, b):
    a, b = _pair(a, b)
    return _make(a.data + b.data, (a, b),
                 lambda g: (unbroadcast(g, a.shape), unbroadcast(g, b.shape)))


def sub(a, b):
    a, b = _pair(a, b)
    return _make(a.data - b.data, (a, b),
                 lambda g: (unbroadcast(g, a.shape), unbroadcast(-g, b.shape)))


def neg(a):
    return _make(-a.data, (a,), lambda g: (-g,))


def mul(a, b):
    a, b = _pair(a, b)
    return _make(a.data * b.data, (a, b),
                 lambda g: (unbroadcast(g * b.data, a.shape), unbroadcast(g * a.data, b.shape)))


def div(a, b):
    a, b = _pair(a, b)
    return _make(a.data / b.data, (a, b),
                 lambda g: (unbroadcast(g / b.data, a.shape),
                            unbroadcast(-g * a.data / (b.data * b.data), b.shape)))


def pow(a, p):
    """a ** p for a scalar exponent p."""
    if isinstance(p, Tensor) or np.ndim(p) != 0:
        raise TypeError("pow supports scalar exponents only")
    p = float(p)
    return _make(a.data ** p, (a,), lambda g: (g * p * a.data ** (p - 1),))


def exp(a):
    out = np.exp(a.data)
    return _make(out, (a,), lambda g: (g * out,))


def log(a):
    return _make(np.log(a.data), (a,), lambda g: (g / a.data,))


def relu(a):
    mask = a.data > 0
    return _make(a.data * mask, (a,), lambda g: (g * mask,))


# Linear algebra ------------------------------------------------------------------

def matmul(a, b):
    """a @ b for 2D or batched (..., n, k) @ (..., k, m) inputs with batch broadcasting."""
    a, b = _pair(a, b)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("matmul needs inputs with at least 2 dims")

    def backward(g):
        ga = g @ np.swapaxes(b.data, -1, -2)
        gb = np.swapaxes(a.data, -1, -2) @ g
        return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)

    return _make(a.data @ b.data, (a, b), backward)


# Reductions ----------------------------------------------------------------------

def sum(a, axis=None, keepdims=False):
    axes = _axes(axis, a.ndim)
    return _make(a.data.sum(axis=axes, keepdims=keepdims), (a,),
                 lambda g: (_expand(g, a.shape, axes, keepdims).copy(),))


def mean(a, axis=None, keepdims=False):
    axes = _axes(axis, a.ndim)
    n = int(np.prod([a.shape[i] for i in axes]))
    return _make(a.data.mean(axis=axes, keepdims=keepdims), (a,),
                 lambda g: (_expand(g, a.shape, axes, keepdims) / np.asarray(n, a.dtype),))


def max(a, axis=None, keepdims=False):
    """Max over axis. The gradient goes to the argmax; tied maxima share it evenly."""
    axes = _axes(axis, a.ndim)
    m = a.data.max(axis=axes, keepdims=True)
    mask = a.data == m
    count = mask.sum(axis=axes, keepdims=True).astype(a.dtype)
    out = m if keepdims else np.squeeze(m, axis=axes)
    return _make(out, (a,), lambda g: (mask * _expand(g, a.shape, axes, keepdims) / count,))


# Shape ---------------------------------------------------------------------------

def reshape(a, shape):
    return _make(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),))


def transpose(a, axes=None):
    axes = tuple(range(a.ndim))[::-1] if axes is None else tuple(axes)
    inv = tuple(np.argsort(axes))
    return _make(a.data.transpose(axes), (a,), lambda g: (g.transpose(inv),))


# Softmax family ------------------------------------------------------------------

def log_softmax(a, axis=-1):
    shifted = a.data - a.data.max(axis=axis, keepdims=True)
    out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return _make(out, (a,),
                 lambda g: (g - np.exp(out) * g.sum(axis=axis, keepdims=True),))


def softmax_cross_entropy(logits, labels):
    """Mean cross-entropy of (N, C) logits against integer labels (N,), fused.

    Forward uses log-sum-exp; backward is (softmax - onehot) / N.
    """
    y = labels.data if isinstance(labels, Tensor) else np.asarray(labels)
    if logits.ndim != 2 or y.shape != (logits.shape[0],):
        raise ValueError(f"expected (N, C) logits and (N,) labels, got {logits.shape}, {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError("labels must be integers")
    N = logits.shape[0]
    rows = np.arange(N)
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    logp = shifted - lse
    loss = -logp[rows, y].mean()

    def backward(g):
        d = np.exp(logp)
        d[rows, y] -= 1
        return (g * d / np.asarray(N, logits.dtype),)

    return _make(np.asarray(loss, dtype=logits.dtype), (logits,), backward)
