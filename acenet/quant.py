"""Post-training int8 quantization of a Sequential conv/linear/BN/ReLU/maxpool CNN (plain numpy).

Pipeline:
  fold_bn(model)                -> float layer list with BN folded into the preceding conv/linear
  calibrate(layers, X, method)  -> per-tensor activation scales (inputs of every conv/linear)
  build_int8(layers, scales)    -> Int8Model: int8 weights (per output channel), int32 biases,
                                   fixed-point requantization multipliers (M0 in Q31 + right shift)
  quantize_input(x, s)          -> the ONE float -> int step (model boundary)
  Int8Model.forward(xq)         -> int32 logits, integer arithmetic only
  fake_quant_forward(...)       -> float32 simulation of the same scheme (optionally per layer)

Scheme: symmetric, zero_point = 0 everywhere, int8 range [-127, 127] (−128 unused).
Activations are kept NHWC inside the engines; conv columns are ordered (kh, kw, C).
Compute layers: conv, dwconv (depthwise, Wm (C, kh*kw)), linear. Global average pooling ("gap")
is integer: the int8 input is summed and divided by H*W with round-half-up, keeping the scale, so
the scale of the layer after a gap is calibrated on the gap's INPUT.

Serialization: Int8Model.save/load (.npz, Phase 4) and save_compact/load_compact, a flat
little-endian binary (small header + raw arrays) whose size is the deployed model size.
"""

import json
import struct

import numpy as np

from acenet import nn

QMAX = 127
COMPUTE = ("conv", "dwconv", "linear")
INT32_MAX = 2 ** 31 - 1
CALIBRATORS = ("minmax", "p99.99")


# ---------------------------------------------------------------- BN folding

def fold_bn(model):
    """Eval-mode Sequential -> list of ops (dicts). conv/linear followed by BN get the BN folded in:
    W' = W * g / sqrt(var + eps) per output channel, b' = (b - mean) * g / sqrt(var + eps) + beta.
    Dropout (identity in eval) is dropped. Weights are stored as matrices Wm (F, K): conv K order
    is (kh, kw, C); linear Wm = W.T. Folding is done in float64, stored as float32."""
    ops, layers = [], list(model.layers)
    i = 0
    while i < len(layers):
        m = layers[i]
        if isinstance(m, (nn.Conv2D, nn.DepthwiseConv2D, nn.Linear)):
            W = m.W.data.astype(np.float64)
            F = W.shape[1] if isinstance(m, nn.Linear) else W.shape[0]
            b = m.b.data.astype(np.float64) if m.b is not None else np.zeros(F)
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            if isinstance(nxt, nn._BatchNorm):
                g = nxt.gamma.data.astype(np.float64) / np.sqrt(nxt.running_var.astype(np.float64) + nxt.eps)
                b = (b - nxt.running_mean.astype(np.float64)) * g + nxt.beta.data.astype(np.float64)
                W = W * (g.reshape(1, -1) if isinstance(m, nn.Linear) else g.reshape(-1, 1, 1, 1))
                i += 1
            if isinstance(m, nn.DepthwiseConv2D):
                if m.stride != 1:
                    raise NotImplementedError("only stride-1 convs")
                C, _, kh, kw = W.shape
                ops.append({"op": "dwconv", "Wm": W.reshape(C, kh * kw).astype(np.float32),
                            "b": b.astype(np.float32), "k": (kh, kw), "pad": int(m.padding)})
            elif isinstance(m, nn.Conv2D):
                if m.stride != 1:
                    raise NotImplementedError("only stride-1 convs")
                Fo, C, kh, kw = W.shape
                ops.append({"op": "conv", "Wm": W.transpose(0, 2, 3, 1).reshape(Fo, -1).astype(np.float32),
                            "b": b.astype(np.float32), "k": (kh, kw), "pad": int(m.padding)})
            else:
                ops.append({"op": "linear", "Wm": np.ascontiguousarray(W.T).astype(np.float32),
                            "b": b.astype(np.float32)})
        elif isinstance(m, nn.ReLU):
            ops.append({"op": "relu"})
        elif isinstance(m, nn.MaxPool2D):
            if m.stride not in (None, m.kernel_size) or not isinstance(m.kernel_size, int):
                raise NotImplementedError("only non-overlapping square pooling")
            ops.append({"op": "pool", "k": m.kernel_size})
        elif isinstance(m, nn.Flatten):
            ops.append({"op": "flatten"})
        elif isinstance(m, nn.GlobalAvgPool2D):
            ops.append({"op": "gap"})
        elif isinstance(m, nn.Dropout):
            pass
        else:
            raise NotImplementedError(type(m).__name__)
        i += 1
    return ops


def compute_layers(ops):
    return [o for o in ops if o["op"] in COMPUTE]


# ---------------------------------------------------------------- shared NHWC helpers

def im2col(x, k, pad):
    """x (N, H, W, C) -> cols (N*Ho*Wo, kh*kw*C) in x's dtype, (N, Ho, Wo). Zero padding (exact for
    int8 with zero_point 0)."""
    kh, kw = k
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    N, H, W, C = x.shape
    Ho, Wo = H - kh + 1, W - kw + 1
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))  # N, Ho, Wo, C, kh, kw
    return win.transpose(0, 1, 2, 4, 5, 3).reshape(N * Ho * Wo, kh * kw * C), (N, Ho, Wo)


def dw_windows(x, k, pad):
    """Grouped im2col for depthwise conv: x (N, H, W, C) -> (N, Ho, Wo, C, kh*kw) in x's dtype."""
    kh, kw = k
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    N, H, W, C = x.shape
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))  # N, Ho, Wo, C, kh, kw
    return np.ascontiguousarray(win).reshape(N, H - kh + 1, W - kw + 1, C, kh * kw)


def gap_int(x):
    """Integer global average pool: int8 (N, H, W, C) -> int8 (N, C), round(sum / HW) half up,
    computed as floor((2 * sum + HW) / (2 * HW)) in int32. Same scale as the input."""
    hw = x.shape[1] * x.shape[2]
    s = x.sum(axis=(1, 2), dtype=np.int32)
    return ((2 * s + hw) // (2 * hw)).astype(np.int8)


def maxpool_nhwc(x, k):
    N, H, W, C = x.shape
    return x[:, :H // k * k, :W // k * k].reshape(N, H // k, k, W // k, k, C).max(axis=(2, 4))


def flatten_nhwc(x):
    """Flatten in the float model's NCHW (C, H, W) order so linear weights need no permutation."""
    return x.transpose(0, 3, 1, 2).reshape(x.shape[0], -1) if x.ndim == 4 else x


def to_nhwc(x):
    return np.ascontiguousarray(x.transpose(0, 2, 3, 1))


# ---------------------------------------------------------------- float / fake-quant forward

def round_half_up(x):
    return np.floor(x + x.dtype.type(0.5))


def fake_q(x, s):
    """Quantize-dequantize, symmetric int8, round half up (as the integer requantizer)."""
    s = np.float32(s)
    return np.clip(round_half_up(x / s), -QMAX, QMAX) * s


def float_forward(ops, x, qspec=None, quantized=None, hook=None):
    """Folded float32 forward on NCHW x -> logits (N, F_last).

    qspec (from build_qspec) turns on fake quantization for compute layers whose index is in
    `quantized` (default: all): the layer's input activation, weights and bias are put on their
    int8 / int32 grids. hook(i, h) sees the input of compute layer i (float, before quantization)."""
    h = to_nhwc(x.astype(np.float32))
    li, hooked = 0, False
    for o in ops:
        if o["op"] in COMPUTE:
            if hook is not None and not hooked:
                hook(li, h)
            hooked = False
            Wm, b = o["Wm"], o["b"]
            if qspec is not None and (quantized is None or li in quantized):
                q = qspec[li]
                h = fake_q(h, q["s_in"])
                Wm = q["w"].astype(np.float32) * q["s_w"][:, None]
                b = q["b"].astype(np.float32) * (np.float32(q["s_in"]) * q["s_w"])
            if o["op"] == "conv":
                cols, (N, Ho, Wo) = im2col(h, o["k"], o["pad"])
                h = (cols @ Wm.T + b).reshape(N, Ho, Wo, -1)
            elif o["op"] == "dwconv":
                h = np.einsum("nhwck,ck->nhwc", dw_windows(h, o["k"], o["pad"]), Wm) + b
            else:
                h = h @ Wm.T + b
            li += 1
        elif o["op"] == "gap":
            # the next compute layer's input scale covers the gap INPUT (as in the int engine)
            if hook is not None:
                hook(li, h)
                hooked = True
            if qspec is not None and (quantized is None or li in quantized):
                s = np.float32(qspec[li]["s_in"])
                h = round_half_up(fake_q(h, s).mean(axis=(1, 2)) / s) * s
            else:
                h = h.mean(axis=(1, 2))
        elif o["op"] == "relu":
            h = np.maximum(h, 0)
        elif o["op"] == "pool":
            h = maxpool_nhwc(h, o["k"])
        elif o["op"] == "flatten":
            h = flatten_nhwc(h)
    return h


def batched(fn, X, batch=200):
    return np.concatenate([fn(X[i:i + batch]) for i in range(0, len(X), batch)])


# ---------------------------------------------------------------- calibration and scales

def calibrate(ops, X, method, batch=200):
    """Per-tensor scales for the input of every compute layer (index 0 = the network input),
    from float activations of the folded model on X. minmax: max|x| / 127; p99.99: 99.99th
    percentile of |x| / 127 (over all elements, zeros included)."""
    if method not in CALIBRATORS:
        raise ValueError(method)
    acts = {}

    def hook(i, h):
        acts.setdefault(i, []).append(np.abs(h).ravel())

    for j in range(0, len(X), batch):
        float_forward(ops, X[j:j + batch], hook=hook)
    scales = []
    for i in range(len(acts)):
        a = np.concatenate(acts[i])
        r = float(a.max()) if method == "minmax" else float(np.percentile(a, 99.99))
        scales.append(r / QMAX if r > 0 else 1.0)
    return scales


def quantize_multiplier(M):
    """Real multiplier(s) M > 0 -> (M0 int32 in [2^30, 2^31), right shift rs) with M ≈ M0 * 2^-rs."""
    M = np.asarray(M, dtype=np.float64)
    if np.any(M <= 0):
        raise ValueError("multiplier must be positive")
    m, e = np.frexp(M)                          # M = m * 2^e, m in [0.5, 1)
    M0 = np.round(m * 2.0 ** 31).astype(np.int64)
    carry = M0 == 2 ** 31
    M0[carry] //= 2
    e = e + carry
    rs = 31 - e
    if np.any(rs < 1) or np.any(rs > 62):
        raise ValueError(f"multiplier out of range: {M}")
    return M0.astype(np.int32), rs.astype(np.int8)


def build_qspec(ops, act_scales):
    """Per compute layer: s_in, per-channel s_w, int8 w (F, K), int32 b (scale s_in*s_w), and the
    requantization multiplier. Hidden layers: M = s_in*s_w[c]/s_out with s_out = next layer's
    s_in. Last layer (logits): M = s_w[c]/max(s_w), i.e. rescale all classes to one common
    int32 scale s_in*max(s_w) so argmax/TTA-sum over classes is valid (no clipping)."""
    layers = compute_layers(ops)
    spec = []
    for i, o in enumerate(layers):
        s_in = float(act_scales[i])
        amax = np.abs(o["Wm"]).max(axis=1).astype(np.float64)
        s_w = np.where(amax > 0, amax / QMAX, 1.0)
        w = np.clip(np.round(o["Wm"] / s_w[:, None]), -QMAX, QMAX).astype(np.int8)
        bq = np.round(o["b"].astype(np.float64) / (s_in * s_w))
        if np.abs(bq).max() > INT32_MAX:
            raise OverflowError(f"layer {i}: bias does not fit int32")
        last = i == len(layers) - 1
        M = s_w / s_w.max() if last else s_in * s_w / float(act_scales[i + 1])
        M0, rs = quantize_multiplier(M)
        spec.append({"s_in": s_in, "s_w": s_w.astype(np.float32), "w": w, "b": bq.astype(np.int32),
                     "M0": M0, "shift": rs, "M": M, "last": last})
    return spec


# ---------------------------------------------------------------- integer-only engine

def quantize_input(x, s_in):
    """Model boundary, the only float -> int step: float32 standardized NCHW image -> int8."""
    return np.clip(round_half_up(x.astype(np.float32) / np.float32(s_in)), -QMAX, QMAX).astype(np.int8)


def check_accumulator_bound(K, b):
    """|sum_k q_x q_w + b| <= 127*127*K + max|b| must fit int32 (inputs are in [-127, 127])."""
    bound = QMAX * QMAX * int(K) + (int(np.abs(b.astype(np.int64)).max()) if b.size else 0)
    assert bound <= INT32_MAX, f"int32 accumulator may overflow: bound {bound} (K={K})"


def int_matmul(a, w32, b):
    """a int8 (M, K) @ w (F, K)^T + b int32 -> int32 (M, F). w32 is the int8 weight pre-widened."""
    check_accumulator_bound(a.shape[1], b)
    acc = np.einsum("mk,fk->mf", a.astype(np.int32), w32)
    acc += b
    return acc


def requantize(acc, M0, shift):
    """int32 acc * M (M = M0 * 2^-shift, per last-axis channel) -> int8, via int64:
    (acc * M0 + 2^(shift-1)) >> shift (arithmetic shift = round half up), clipped to [-127, 127]."""
    return np.clip(rescale(acc, M0, shift), -QMAX, QMAX).astype(np.int8)


def rescale(acc, M0, shift):
    sh = shift.astype(np.int64)
    p = acc.astype(np.int64) * M0.astype(np.int64)
    p += np.left_shift(np.int64(1), sh - 1)
    return np.right_shift(p, sh)


class Int8Model:
    """Integer-only inference. Holds int8 weights, int32 biases, int32 M0 and int8 shifts per
    channel; the input scale is the only float and is used solely by quantize_input."""

    def __init__(self, arch, input_scale, layers):
        self.arch, self.input_scale, self.layers = arch, float(input_scale), layers
        for L in self.layers:
            L["w32"] = L["w"].reshape(L["w"].shape[0], -1).astype(np.int32)

    @classmethod
    def from_qspec(cls, ops, qspec):
        arch, layers, li = [], [], 0
        for o in ops:
            a = {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items() if k not in ("Wm", "b")}
            arch.append(a)
            if o["op"] in COMPUTE:
                q = qspec[li]
                layers.append({"w": q["w"], "b": q["b"], "M0": q["M0"], "shift": q["shift"]})
                li += 1
        return cls(arch, qspec[0]["s_in"], layers)

    def forward(self, xq, trace=None):
        """xq int8 NCHW -> int32 logits at a common scale. trace: optional list collecting every
        intermediate array."""
        assert xq.dtype == np.int8
        h = np.ascontiguousarray(xq.transpose(0, 2, 3, 1))
        li, n = 0, len(self.layers)

        def t(v):
            if trace is not None:
                trace.append(v)
            return v

        t(h)
        for a in self.arch:
            if a["op"] in COMPUTE:
                L = self.layers[li]
                if a["op"] == "conv":
                    cols, (N, Ho, Wo) = im2col(h, a["k"], a["pad"])
                    acc = t(int_matmul(t(cols), L["w32"], L["b"]))
                elif a["op"] == "dwconv":
                    win = t(dw_windows(h, a["k"], a["pad"]))
                    check_accumulator_bound(win.shape[-1], L["b"])
                    N, Ho, Wo = win.shape[:3]
                    acc = np.einsum("nhwck,ck->nhwc", win.astype(np.int32), L["w32"])
                    acc += L["b"]
                    acc = t(acc.reshape(N * Ho * Wo, -1))
                else:
                    N, acc = h.shape[0], t(int_matmul(h, L["w32"], L["b"]))
                if li == n - 1:
                    out = rescale(acc, L["M0"], L["shift"])
                    assert np.abs(out).max(initial=0) <= INT32_MAX
                    h = t(out.astype(np.int32))
                else:
                    h = t(requantize(acc, L["M0"], L["shift"]))
                    if a["op"] != "linear":
                        h = h.reshape(N, Ho, Wo, -1)
                li += 1
            elif a["op"] == "gap":
                h = t(gap_int(h))
            elif a["op"] == "relu":
                h = t(np.maximum(h, np.int8(0)))
            elif a["op"] == "pool":
                h = t(maxpool_nhwc(h, a["k"]))
            elif a["op"] == "flatten":
                h = t(flatten_nhwc(h))
        return h

    def logits(self, X, flip=False, batch=200):
        """Float NCHW images -> int32 logits; flip=True adds the logits of hflip(x) (flip TTA)."""
        def f(xb):
            z = self.forward(quantize_input(xb, self.input_scale))
            if flip:
                z = z + self.forward(quantize_input(np.ascontiguousarray(xb[..., ::-1]), self.input_scale))
            return z
        return batched(f, X, batch)

    # -------- serialization (uncompressed .npz, no pickle)

    def arrays(self):
        out = {"arch": np.array(json.dumps(self.arch)), "input_scale": np.float32(self.input_scale)}
        for i, L in enumerate(self.layers):
            for k in ("w", "b", "M0", "shift"):
                out[f"l{i}_{k}"] = L[k]
        return out

    def save(self, path):
        np.savez(path, **self.arrays())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            arch = json.loads(str(d["arch"]))
            n = sum(a["op"] in COMPUTE for a in arch)
            layers = [{k: d[f"l{i}_{k}"] for k in ("w", "b", "M0", "shift")} for i in range(n)]
            return cls(arch, float(d["input_scale"]), layers)

    # -------- compact flat binary (the deployed artifact; every byte counts toward the size)
    #
    # little-endian. header: b"ACE8", version u8, n_ops u8, input_scale f32
    # per op: code u8 (+ conv/dwconv: kh u8, kw u8, pad u8, F u16, K u16 | linear: F u16, K u16 | pool: k u8)
    # then per compute layer in order: w int8 [F*K] (row-major (F, K)), b int32 [F], M0 int32 [F], shift int8 [F]

    def save_compact(self, path):
        buf = compact_bytes(self)
        with open(path, "wb") as f:
            f.write(buf)
        return len(buf)

    @classmethod
    def load_compact(cls, path):
        with open(path, "rb") as f:
            return from_compact_bytes(cls, f.read())


MAGIC, VERSION = b"ACE8", 1
OPCODES = {"conv": 1, "dwconv": 2, "linear": 3, "relu": 4, "pool": 5, "flatten": 6, "gap": 7}


def compact_bytes(qm):
    head = [MAGIC, struct.pack("<BBf", VERSION, len(qm.arch), qm.input_scale)]
    body, li = [], 0
    for a in qm.arch:
        head.append(struct.pack("<B", OPCODES[a["op"]]))
        if a["op"] in COMPUTE:
            L = qm.layers[li]
            F, K = L["w"].shape[0], int(np.prod(L["w"].shape[1:]))
            if a["op"] == "linear":
                head.append(struct.pack("<HH", F, K))
            else:
                head.append(struct.pack("<BBBHH", a["k"][0], a["k"][1], a["pad"], F, K))
            body += [np.ascontiguousarray(L["w"], "<i1").tobytes(), np.ascontiguousarray(L["b"], "<i4").tobytes(),
                     np.ascontiguousarray(L["M0"], "<i4").tobytes(), np.ascontiguousarray(L["shift"], "<i1").tobytes()]
            li += 1
        elif a["op"] == "pool":
            head.append(struct.pack("<B", a["k"]))
    return b"".join(head + body)


def from_compact_bytes(cls, buf):
    if buf[:4] != MAGIC:
        raise ValueError("not an ACE8 file")
    version, n_ops, input_scale = struct.unpack_from("<BBf", buf, 4)
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    names = {v: k for k, v in OPCODES.items()}
    pos, arch, shapes = 10, [], []
    for _ in range(n_ops):
        op = names[buf[pos]]
        pos += 1
        a = {"op": op}
        if op in ("conv", "dwconv"):
            kh, kw, pad, F, K = struct.unpack_from("<BBBHH", buf, pos)
            pos += 7
            a.update(k=[kh, kw], pad=pad)
            shapes.append((F, K))
        elif op == "linear":
            shapes.append(struct.unpack_from("<HH", buf, pos))
            pos += 4
        elif op == "pool":
            a["k"] = buf[pos]
            pos += 1
        arch.append(a)
    layers = []
    for F, K in shapes:
        w = np.frombuffer(buf, "<i1", F * K, pos).reshape(F, K).astype(np.int8)
        pos += F * K
        b = np.frombuffer(buf, "<i4", F, pos).astype(np.int32)
        M0 = np.frombuffer(buf, "<i4", F, pos + 4 * F).astype(np.int32)
        shift = np.frombuffer(buf, "<i1", F, pos + 8 * F).astype(np.int8)
        pos += 9 * F
        layers.append({"w": w, "b": b, "M0": M0, "shift": shift})
    if pos != len(buf):
        raise ValueError(f"trailing bytes: parsed {pos} of {len(buf)}")
    return cls(arch, input_scale, layers)
