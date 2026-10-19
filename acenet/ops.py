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


def softmax_cross_entropy(logits, labels, label_smoothing=0.0):
    """Mean cross-entropy of (N, K) logits against integer labels (N,), fused.

    Target = (1 - eps) * onehot + eps / K (eps = label_smoothing). Forward uses log-sum-exp;
    backward is (softmax - target) / N.
    """
    y = labels.data if isinstance(labels, Tensor) else np.asarray(labels)
    if logits.ndim != 2 or y.shape != (logits.shape[0],):
        raise ValueError(f"expected (N, C) logits and (N,) labels, got {logits.shape}, {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError("labels must be integers")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    N, K = logits.shape
    eps = float(label_smoothing)
    rows = np.arange(N)
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    logp = shifted - lse
    loss = -logp[rows, y].mean()
    if eps:
        loss = (1 - eps) * loss - eps / K * logp.sum(axis=1).mean()

    def backward(g):
        d = np.exp(logp)
        if eps:
            d -= eps / K
            d[rows, y] -= 1 - eps
        else:
            d[rows, y] -= 1
        return (g * d / np.asarray(N, logits.dtype),)

    return _make(np.asarray(loss, dtype=logits.dtype), (logits,), backward)


# Convolution / pooling / normalization -------------------------------------------

def _pair_int(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def conv2d(x, W, b=None, stride=1, padding=0):
    """2D cross-correlation. x (N, C, H, W), W (F, C, kh, kw), b (F,) -> (N, F, Ho, Wo).

    im2col through sliding_window_view, then one batched matmul. Backward: dW/db by matmul,
    dx by col2im with one strided slice-add per kernel offset (kh * kw adds, no per-pixel loops).
    Columns are ordered (kh, kw, C) so each offset's gradient block is contiguous in col2im.
    """
    sh, sw = _pair_int(stride)
    ph, pw = _pair_int(padding)
    N, C, H, Wd = x.shape
    F, Cw, kh, kw = W.shape
    if C != Cw:
        raise ValueError(f"conv2d: input has {C} channels, weight expects {Cw}")
    xp = np.pad(x.data, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if ph or pw else x.data
    Hp, Wp = xp.shape[2:]
    Ho, Wo = (Hp - kh) // sh + 1, (Wp - kw) // sw + 1
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
    # (N, C, Ho, Wo, kh, kw) -> cols (N, kh*kw*C, Ho*Wo)
    cols = np.ascontiguousarray(win.transpose(0, 4, 5, 1, 2, 3)).reshape(N, kh * kw * C, Ho * Wo)
    Wm = W.data.transpose(0, 2, 3, 1).reshape(F, -1)
    out = np.matmul(Wm, cols).reshape(N, F, Ho, Wo)
    if b is not None:
        out += b.data.reshape(1, F, 1, 1)
    parents = (x, W) if b is None else (x, W, b)

    def backward(g):
        g3 = g.reshape(N, F, Ho * Wo)
        dW = np.matmul(g3, cols.transpose(0, 2, 1)).sum(axis=0)
        dW = dW.reshape(F, kh, kw, C).transpose(0, 3, 1, 2)
        dx = None
        if x.requires_grad:
            dcols = np.matmul(np.ascontiguousarray(Wm.T), g3).reshape(N, kh, kw, C, Ho, Wo)
            dxp = np.zeros_like(xp)
            for i in range(kh):
                for j in range(kw):
                    dxp[:, :, i:i + sh * Ho:sh, j:j + sw * Wo:sw] += dcols[:, i, j]
            dx = dxp[:, :, ph:ph + H, pw:pw + Wd]
        if b is None:
            return dx, dW
        return dx, dW, g.sum(axis=(0, 2, 3))

    return _make(out, parents, backward)


def maxpool2d(x, k=2, stride=None):
    """Max pooling over k x k windows (NCHW). The gradient goes to the first argmax in each window
    (row-major order within the window). Loops only over the k*k window offsets."""
    kh, kw = _pair_int(k)
    sh, sw = _pair_int(stride if stride is not None else k)
    N, C, H, Wd = x.shape
    Ho, Wo = (H - kh) // sh + 1, (Wd - kw) // sw + 1

    def view(i, j):
        return x.data[:, :, i:i + sh * (Ho - 1) + 1:sh, j:j + sw * (Wo - 1) + 1:sw]

    out = view(0, 0).copy()
    idx = np.zeros(out.shape, dtype=np.int8 if kh * kw < 128 else np.int32)
    for i in range(kh):
        for j in range(kw):
            if i == j == 0:
                continue
            v = view(i, j)
            better = v > out  # strict: ties keep the earlier offset
            np.copyto(out, v, where=better)
            idx[better] = i * kw + j

    def backward(g):
        dx = np.zeros_like(x.data)
        for i in range(kh):
            for j in range(kw):
                dx[:, :, i:i + sh * (Ho - 1) + 1:sh, j:j + sw * (Wo - 1) + 1:sw] += g * (idx == i * kw + j)
        return (dx,)

    return _make(out, (x,), backward)


def batchnorm(x, gamma, beta, axes, eps=1e-5, return_stats=False):
    """Batch normalization with batch statistics over axes, e.g. (0,) for (N, F) or (0, 2, 3) for NCHW.

    gamma/beta have x's shape with the reduced axes removed. Closed-form backward:
    dx = gamma / sqrt(var + eps) * (g - mean(g) - xhat * mean(g * xhat)).
    With return_stats, also returns the batch (mean, biased var), each of gamma's shape.
    """
    axes = _axes(axes, x.ndim)
    keep = tuple(1 if i in axes else s for i, s in enumerate(x.shape))
    m = int(np.prod([x.shape[i] for i in axes]))
    mu = x.data.mean(axis=axes, keepdims=True)
    xhat = x.data - mu
    var = np.mean(xhat * xhat, axis=axes, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat *= inv
    gam = gamma.data.reshape(keep)
    out = xhat * gam
    out += beta.data.reshape(keep)

    def backward(g):
        dbeta = g.sum(axis=axes, keepdims=True)
        dgamma = (g * xhat).sum(axis=axes, keepdims=True)
        dx = xhat * (dgamma / m)
        np.subtract(g, dx, out=dx)
        dx -= dbeta / m
        dx *= gam * inv
        return dx, dgamma.reshape(gamma.shape), dbeta.reshape(beta.shape)

    res = _make(out, (x, gamma, beta), backward)
    if return_stats:
        return res, mu.reshape(gamma.shape), var.reshape(gamma.shape)
    return res


def depthwise_conv2d(x, W, b=None, stride=1, padding=0):
    """Depthwise 2D cross-correlation (groups = channels). x (N, C, H, W), W (C, 1, kh, kw), b (C,)
    -> (N, C, Ho, Wo).

    Equivalent to grouped im2col (each channel's (Ho*Wo, kh*kw) window matrix times its kh*kw filter),
    but computed as kh*kw in-place multiply-adds of strided shifted views, without materializing the
    windows: 2-3x faster than the im2col form on CPU (measured 62 vs 164 ms fwd+bwd, 128x32x28x28).
    Backward: dW[:, i, j] = sum over (N, Ho, Wo) of g * shifted view; dx by col2im with one
    strided slice-add per kernel offset (as conv2d). No loops over samples or pixels.
    """
    sh, sw = _pair_int(stride)
    ph, pw = _pair_int(padding)
    N, C, H, Wd = x.shape
    Cw, one, kh, kw = W.shape
    if Cw != C or one != 1:
        raise ValueError(f"depthwise_conv2d: weight {W.shape} does not match {C} input channels")
    xp = np.pad(x.data, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if ph or pw else x.data
    Hp, Wp = xp.shape[2:]
    Ho, Wo = (Hp - kh) // sh + 1, (Wp - kw) // sw + 1
    w = W.data.reshape(C, kh, kw)

    def view(a, i, j):
        return a[:, :, i:i + sh * (Ho - 1) + 1:sh, j:j + sw * (Wo - 1) + 1:sw]

    out = np.zeros((N, C, Ho, Wo), dtype=np.result_type(x.data, W.data))
    tmp = np.empty_like(out)
    for i in range(kh):
        for j in range(kw):
            np.multiply(view(xp, i, j), w[:, i, j].reshape(1, C, 1, 1), out=tmp)
            out += tmp
    if b is not None:
        out += b.data.reshape(1, C, 1, 1)
    parents = (x, W) if b is None else (x, W, b)

    def backward(g):
        dW = np.empty_like(w)
        for i in range(kh):
            for j in range(kw):
                dW[:, i, j] = np.einsum("nchw,nchw->c", g, view(xp, i, j))
        dx = None
        if x.requires_grad:
            dxp = np.zeros_like(xp)
            t = np.empty_like(g)
            for i in range(kh):
                for j in range(kw):
                    np.multiply(g, w[:, i, j].reshape(1, C, 1, 1), out=t)
                    view(dxp, i, j)[...] += t
            dx = dxp[:, :, ph:ph + H, pw:pw + Wd]
        dW = dW.reshape(C, 1, kh, kw)
        if b is None:
            return dx, dW
        return dx, dW, g.sum(axis=(0, 2, 3))

    return _make(out, parents, backward)


# Distillation ----------------------------------------------------------------------

def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.9):
    """Hinton et al. (2015) distillation loss, built from existing ops:

        alpha * T^2 * KL(softmax(t/T) || softmax(s/T)) + (1 - alpha) * CE(s, y)

    KL is summed over classes and averaged over the batch; it includes the teacher-entropy term,
    so it is exactly 0 when s == t. teacher_logits is a constant (ndarray or Tensor, no gradient).
    T^2 keeps the soft-target gradient scale independent of T.
    """
    t = teacher_logits.data if isinstance(teacher_logits, Tensor) else np.asarray(teacher_logits)
    t = t.astype(student_logits.dtype) / student_logits.dtype.type(T)
    log_pt = t - t.max(axis=1, keepdims=True)
    log_pt = log_pt - np.log(np.exp(log_pt).sum(axis=1, keepdims=True))
    log_ps = log_softmax(student_logits / float(T), axis=1)
    kl = (Tensor(np.exp(log_pt)) * (Tensor(log_pt) - log_ps)).sum(axis=1).mean()
    loss = kl * float(alpha * T * T)
    if alpha < 1.0:
        loss = loss + softmax_cross_entropy(student_logits, labels) * float(1.0 - alpha)
    return loss
