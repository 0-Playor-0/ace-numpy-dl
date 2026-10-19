"""Tensor with reverse-mode automatic differentiation."""

import contextlib

import numpy as np

_grad_enabled = True


def is_grad_enabled():
    return _grad_enabled


@contextlib.contextmanager
def no_grad():
    """Ops run inside this block do not record a graph."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


class Tensor:
    """An ndarray plus the graph needed to backpropagate into it.

    _parents: the input Tensors of the op that produced this one.
    _backward: maps the upstream gradient to a tuple of gradients, one per parent.
    """

    __array_priority__ = 1000  # make ndarray <op> Tensor dispatch to Tensor's reflected ops

    def __init__(self, data, requires_grad=False, dtype=None):
        self.data = np.asarray(data, dtype=dtype)
        if requires_grad and not np.issubdtype(self.data.dtype, np.floating):
            raise TypeError(f"only floating tensors can require grad, got {self.data.dtype}")
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = ()
        self._backward = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return _ops.transpose(self)

    def __repr__(self):
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    def __len__(self):
        return len(self.data)

    def zero_grad(self):
        self.grad = None

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def backward(self, grad=None):
        """Backpropagate from this tensor, accumulating into .grad of every leaf that requires grad."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=self.data.dtype)
        if grad.shape != self.shape:
            raise ValueError(f"grad shape {grad.shape} does not match tensor shape {self.shape}")

        # Iterative post-order DFS -> topological order (parents before children).
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in visited:
                    stack.append((p, False))

        # Gradients for this pass live in a local table and are summed per node,
        # so a node used several times receives the sum of all its contributions.
        grads = {id(self): grad}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:  # leaf
                node.grad = g.copy() if node.grad is None else node.grad + g
                continue
            for p, pg in zip(node._parents, node._backward(g)):
                if p.requires_grad and pg is not None:
                    pid = id(p)
                    grads[pid] = pg if pid not in grads else grads[pid] + pg

    # Operator overloads -----------------------------------------------------------
    def __add__(self, other):
        return _ops.add(self, other)

    def __radd__(self, other):
        return _ops.add(other, self)

    def __sub__(self, other):
        return _ops.sub(self, other)

    def __rsub__(self, other):
        return _ops.sub(other, self)

    def __mul__(self, other):
        return _ops.mul(self, other)

    def __rmul__(self, other):
        return _ops.mul(other, self)

    def __truediv__(self, other):
        return _ops.div(self, other)

    def __rtruediv__(self, other):
        return _ops.div(other, self)

    def __neg__(self):
        return _ops.neg(self)

    def __pow__(self, p):
        return _ops.pow(self, p)

    def __matmul__(self, other):
        return _ops.matmul(self, other)

    def __rmatmul__(self, other):
        return _ops.matmul(other, self)

    # Method forms -----------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        return _ops.sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return _ops.mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return _ops.max(self, axis, keepdims)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _ops.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return _ops.transpose(self, axes or None)

    def exp(self):
        return _ops.exp(self)

    def log(self):
        return _ops.log(self)

    def relu(self):
        return _ops.relu(self)


from . import ops as _ops  # noqa: E402  (ops imports Tensor; bound late to break the cycle)
