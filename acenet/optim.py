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


class CosineWarmup:
    """Per-batch schedule: linear warmup to the optimizer's base lr, then cosine decay to min_lr.

    Call step() once before every optimizer update; it sets opt.lr = lr_at(t) and advances t.
    lr_at(t), t = 0-based update index, T = total_epochs * steps_per_epoch, W = warmup steps:
      t <  W: base * (t + 1) / W                       (lr_at(W - 1) == base)
      t >= W: min + (base - min) * (1 + cos(pi * (t - W) / (T - W))) / 2   (lr_at(T) == min)
    """

    def __init__(self, opt, warmup_epochs, total_epochs, steps_per_epoch, min_lr=0.0):
        self.opt = opt
        self.base_lr = opt.lr
        self.min_lr = min_lr
        self.warmup = int(round(warmup_epochs * steps_per_epoch))
        self.total = int(total_epochs * steps_per_epoch)
        if not 0 <= self.warmup < self.total:
            raise ValueError("need 0 <= warmup steps < total steps")
        self.t = 0

    def lr_at(self, t):
        if t < self.warmup:
            return self.base_lr * (t + 1) / self.warmup
        p = min(1.0, (t - self.warmup) / (self.total - self.warmup))
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + np.cos(np.pi * p))

    def step(self):
        self.opt.lr = float(self.lr_at(self.t))
        self.t += 1
        return self.opt.lr


def lr_finder(model_factory, opt_factory, loader, lr_min=1e-6, lr_max=1.0, num_steps=None,
              diverge_factor=4.0, beta=0.98):
    """LR range test (Smith 2017): train a fresh model while raising lr exponentially per batch.

    model_factory() -> model; opt_factory(params, lr) -> optimizer; loader yields (x, y) batches.
    Runs num_steps updates (default: one epoch, len(loader)) from lr_min to lr_max, stopping early
    once the smoothed loss exceeds diverge_factor x its minimum (or goes non-finite).
    Smoothing: bias-corrected EMA with factor beta.
    Suggestions:
      steepest: lr where d(smoothed loss)/d(log lr) is most negative (first 10% of steps skipped)
      min_div10: lr at the smoothed-loss minimum divided by 10
    Returns dict with lrs, losses, smoothed losses and both suggestions.
    """
    from . import ops
    from .tensor import Tensor

    model = model_factory()
    model.train()
    num_steps = num_steps or len(loader)
    opt = opt_factory(model.parameters(), lr_min)
    mult = (lr_max / lr_min) ** (1.0 / max(1, num_steps - 1))
    lrs, losses, smooth = [], [], []
    avg, best, stop = 0.0, np.inf, "max_lr"
    it = iter(loader)
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(num_steps):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            opt.lr = lr_min * mult ** t
            loss = ops.softmax_cross_entropy(model(Tensor(xb)), yb)
            lv = float(loss.data)
            if not np.isfinite(lv):
                stop = "non-finite loss"
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            avg = beta * avg + (1 - beta) * lv
            s = avg / (1 - beta ** (t + 1))
            lrs.append(opt.lr)
            losses.append(lv)
            smooth.append(s)
            best = min(best, s)
            if s > diverge_factor * best:
                stop = f"smoothed loss > {diverge_factor}x min"
                break
    lrs, smooth_a = np.array(lrs), np.array(smooth)
    i_min = int(smooth_a.argmin())
    skip = max(1, len(lrs) // 10)
    slope = np.gradient(smooth_a, np.log(lrs))
    i_steep = skip + int(slope[skip:i_min + 1].argmin()) if i_min > skip else int(slope.argmin())
    return {"lrs": lrs.tolist(), "losses": losses, "smoothed": smooth, "steps": len(lrs),
            "stop_reason": stop, "min_loss_lr": float(lrs[i_min]), "min_smoothed_loss": float(smooth_a[i_min]),
            "suggested_min_div10": float(lrs[i_min] / 10), "suggested_steepest": float(lrs[i_steep])}


def plot_lr_finder(res, path, title="LR finder"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["lrs"], res["losses"], color="0.75", lw=0.8, label="batch loss")
    ax.plot(res["lrs"], res["smoothed"], color="C0", lw=1.8, label="smoothed (EMA 0.98)")
    for key, c, lab in (("suggested_min_div10", "C3", "min-loss lr / 10 (used)"),
                        ("suggested_steepest", "C2", "steepest descent")):
        ax.axvline(res[key], color=c, ls="--", lw=1.2, label=f"{lab}: {res[key]:.2e}")
    ax.axvline(res["min_loss_lr"], color="0.4", ls=":", lw=1, label=f"min smoothed loss: {res['min_loss_lr']:.2e}")
    ax.set_xscale("log")
    top = min(max(res["smoothed"]), 4 * res["min_smoothed_loss"]) * 1.1
    ax.set_ylim(0, max(top, res["smoothed"][0] * 1.1))
    ax.set_xlabel("learning rate (log)")
    ax.set_ylabel("training loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
