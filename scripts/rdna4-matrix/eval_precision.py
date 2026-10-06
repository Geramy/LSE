"""Accuracy study for RDNA4 precision/sparsity levers on the real Qwen3.8-27B Q4 checkpoint.

Runs the MLX checkpoint on the Apple GPU (FP32 activations, weights exactly the Q4
group-64 affine values) and emulates, per linear layer inside the decoder blocks:
  - activation quantisation formats (INT8 / FP8-E4M3, per token-group-64 or per token)
  - 2:4 structured weight sparsity on the dequantised Q4 weights (magnitude or Wanda)
Reports perplexity over wikitext-2 test windows, mean KL vs the reference arm and
top-1 agreement with the reference arm.
"""
import argparse, json, math, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--windows", type=int, default=4)
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--arms", default="ref,int8_g64,fp8_g64,fp8_tok,int8_tok,bf8_g64")
ap.add_argument("--out", required=True)
args = ap.parse_args()

model, tok = load(args.model)
model.set_dtype(mx.float32)
mx.eval(model.parameters())

# ---------------------------------------------------------------- activation formats
def q_int8(x, group):
    shp = x.shape
    xr = x.reshape(*shp[:-1], -1, group) if group else x[..., None, :]
    amax = mx.max(mx.abs(xr), axis=-1, keepdims=True)
    step = amax / 127.0
    q = mx.clip(mx.round(xr / mx.maximum(step, 1e-30)), -127, 127)
    return (q * step).reshape(shp)

def q_fp8(x, group, fmax, fn):
    shp = x.shape
    xr = x.reshape(*shp[:-1], -1, group) if group else x[..., None, :]
    amax = mx.max(mx.abs(xr), axis=-1, keepdims=True)
    scale = mx.maximum(amax / fmax, 1e-30)
    y = mx.clip(xr / scale, -fmax, fmax)
    yq = fn(y)
    return (yq * scale).reshape(shp)

def e4m3(y):
    return mx.from_fp8(mx.to_fp8(y)).astype(mx.float32)

def e5m2(y):
    # E5M2 round-to-nearest-even emulation: 2 mantissa bits, min normal 2^-14.
    a = mx.abs(y)
    e = mx.floor(mx.log2(mx.maximum(a, 2.0 ** -14)))
    step = mx.power(2.0, e - 2)
    return mx.round(y / step) * step

ACT = {
    "ref": None,
    "int8_g64": lambda x: q_int8(x, 64),
    "int8_tok": lambda x: q_int8(x, 0),
    "fp8_g64": lambda x: q_fp8(x, 64, 448.0, e4m3),
    "fp8_tok": lambda x: q_fp8(x, 0, 448.0, e4m3),
    "bf8_g64": lambda x: q_fp8(x, 64, 57344.0, e5m2),
}

# ---------------------------------------------------------------- patched linear
STATE = {"act": None, "calib": None}
orig_call = nn.QuantizedLinear.__call__

def patched(self, x):
    if getattr(self, "_study", False):
        if STATE["calib"] is not None:
            xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            acc = STATE["calib"].get(id(self))
            s = mx.sum(xf * xf, axis=0)
            STATE["calib"][id(self)] = s if acc is None else acc + s
        f = STATE["act"]
        if STATE.get("rawsum") and getattr(self, "_dense", None) is None:
            # LSE's current INT8 kernel: dot term on quantised activations, bias term on the
            # raw activation group sums.
            xf = x.astype(mx.float32)
            xq = q_int8(xf, 64)
            y = mx.quantized_matmul(xq, self["weight"], scales=self["scales"], biases=mx.zeros_like(self["biases"]),
                                    transpose=True, group_size=self.group_size, bits=self.bits)
            gs = xf.reshape(*xf.shape[:-1], -1, 64).sum(-1)
            return y + gs @ self["biases"].T
        if f is not None and x.shape[-1] % 64 == 0:
            x = f(x.astype(mx.float32))
        dense = getattr(self, "_dense", None)
        if dense is not None:
            y = x.astype(mx.float32) @ dense.astype(mx.float32).T
            if "bias" in self:
                y = y + self["bias"]
            return y
    return orig_call(self, x)

nn.QuantizedLinear.__call__ = patched

layers = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
targets = []
for li, layer in enumerate(layers):
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            mod._study = True
            targets.append((li, name, mod))
print(f"{len(targets)} quantised linear layers in decoder blocks", flush=True)
print("examples:", [(li, n, tuple(m.weight.shape)) for li, n, m in targets[:12]], flush=True)

# ---------------------------------------------------------------- data
def tokens_of(path, n):
    text = open(path, encoding="utf-8").read()
    ids = tok.encode(text)
    return ids[: n]

corpus = tokens_of(args.corpus, args.windows * args.ctx)
assert len(corpus) == args.windows * args.ctx
windows = [mx.array(corpus[i * args.ctx:(i + 1) * args.ctx])[None] for i in range(args.windows)]

def forward_logprobs(w):
    logits = model(w)
    lp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
    return lp[0, :-1]  # predicts tokens 1..ctx-1

def run_arm(label, ref_store=None):
    t0 = time.time()
    nll, n, kl_sum, agree = 0.0, 0, 0.0, 0
    tops = []
    for wi, w in enumerate(windows):
        lp = forward_logprobs(w)
        tgt = w[0, 1:]
        tok_lp = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        top = mx.argmax(lp, axis=-1)
        if ref_store is not None and label != "ref":
            rlp, rtop = ref_store[wi]
            p = mx.exp(rlp)
            kl = mx.sum(p * (rlp - lp), axis=-1)
            kl_sum += float(mx.sum(kl))
            agree += int(mx.sum(top == rtop))
        mx.eval(tok_lp, top)
        nll -= float(mx.sum(tok_lp)); n += tok_lp.shape[0]
        tops.append((lp, top) if label == "ref" else None)
        if label == "ref":
            mx.eval(lp)
    r = {"arm": label, "targets": n, "ce": nll / n, "ppl": math.exp(nll / n), "secs": round(time.time() - t0, 1)}
    if ref_store is not None and label != "ref":
        r["kl_vs_ref"] = kl_sum / n
        r["top1_agree_vs_ref"] = agree / n
    print(json.dumps(r), flush=True)
    return r, tops

results = []
ref_store = None
arm_list = args.arms.split(",")

def set_sparse(kind, which):
    """Prune selected layers 2:4 along K on the dequantised Q4 weights."""
    calib = STATE.get("calib_norms")
    count = 0
    for li, name, mod in targets:
        mod._dense = None
        if which == "ffn" and "mlp" not in name:
            continue
        w = mx.dequantize(mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits).astype(mx.float32)
        N, K = w.shape
        if K % 4:
            continue
        metric = mx.abs(w)
        if kind == "wanda":
            metric = metric * mx.sqrt(calib[id(mod)])[None, :]
        m4 = metric.reshape(N, K // 4, 4)
        # keep the two largest of each group of four
        order = mx.argsort(m4, axis=-1)
        rank = mx.argsort(order, axis=-1)
        keep = (rank >= 2).reshape(N, K)
        mod._dense = mx.where(keep, w, 0.0).astype(mx.float16)
        mx.eval(mod._dense)
        count += 1
    print(f"pruned {count} layers 2:4 ({kind}, {which})", flush=True)

def set_w8(kind):
    """Requantise the dequantised Q4 weights to 8 bits with one scale per output row."""
    for li, name, mod in targets:
        w = mx.dequantize(mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits).astype(mx.float32)
        amax = mx.max(mx.abs(w), axis=1, keepdims=True)
        if kind == "w8i_row":
            step = mx.maximum(amax / 127.0, 1e-30)
            wq = mx.clip(mx.round(w / step), -127, 127) * step
        elif kind == "w8f_row":
            sc = mx.maximum(amax / 448.0, 1e-30)
            wq = e4m3(mx.clip(w / sc, -448, 448)) * sc
        else:
            raise SystemExit(kind)
        mod._dense = wq.astype(mx.float16)
        mx.eval(mod._dense)
    print(f"requantised {len(targets)} layers ({kind})", flush=True)

def clear_sparse():
    for _, _, mod in targets:
        mod._dense = None

for arm in arm_list:
    parts = arm.split("+")
    act = None; sparse = None
    for p in parts:
        if p == "int8_g64raw":
            act = None
        elif p in ACT:
            act = ACT[p]
        elif p.startswith("sp_") or p.startswith("w8"):
            sparse = p
        else:
            raise SystemExit(f"unknown arm part {p}")
    if sparse and sparse.startswith("w8"):
        set_w8(sparse)
    elif sparse:
        kind, which = sparse[3:].split("_")  # sp_mag_all, sp_wanda_ffn ...
        if kind == "wanda" and "calib_norms" not in STATE:
            STATE["calib"] = {}
            STATE["act"] = None
            clear_sparse()
            calib_ids = tokens_of(args.calib, 4 * args.ctx)
            for i in range(4):
                out = model(mx.array(calib_ids[i * args.ctx:(i + 1) * args.ctx])[None])
                mx.eval(out, list(STATE["calib"].values()))
            STATE["calib_norms"] = STATE["calib"]
            STATE["calib"] = None
            print("calibration done", flush=True)
        set_sparse(kind, which)
    else:
        clear_sparse()
    STATE["act"] = act
    STATE["rawsum"] = "int8_g64raw" in parts
    r, tops = run_arm(arm if arm != "ref" else "ref", ref_store)
    if arm == "ref":
        ref_store = tops
    results.append(r)
    json.dump(results, open(args.out, "w"), indent=1)
clear_sparse()
print("done", flush=True)
