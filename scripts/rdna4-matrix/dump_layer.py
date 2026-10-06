"""Dump one decoder layer's Q4 MLP weights (raw MLX packing, bf16 scales/biases), the real
inputs those projections see on wikitext, and the FP32 reference outputs, for the GPU
prototype check."""
import sys, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

model_dir, corpus, out_dir, layer, tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.makedirs(out_dir, exist_ok=True)
model, tok = load(model_dir)
model.set_dtype(mx.float32)
captured = {}
orig = nn.QuantizedLinear.__call__
def patched(self, x):
    tag = getattr(self, "_tag", None)
    if tag and tag not in captured:
        captured[tag] = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    return orig(self, x)
nn.QuantizedLinear.__call__ = patched
mlp = model.language_model.model.layers[layer].mlp
mlp.up_proj._tag = "up"; mlp.down_proj._tag = "down"
ids = tok.encode(open(corpus, encoding="utf-8").read())
start = 20000  # away from the windows used for perplexity
x = mx.array(ids[start:start + tokens])[None]
mx.eval(model(x))
raw = {}
idx = __import__("json").load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
for proj in ("up_proj", "down_proj"):
    for part in ("weight", "scales", "biases"):
        key = f"language_model.model.layers.{layer}.mlp.{proj}.{part}"
        raw[f"{proj}.{part}"] = mx.load(os.path.join(model_dir, idx[key]))[key]
for proj, tag in (("up_proj", "up"), ("down_proj", "down")):
    w, s, b = raw[f"{proj}.weight"], raw[f"{proj}.scales"], raw[f"{proj}.biases"]
    assert w.dtype == mx.uint32 and s.dtype == mx.bfloat16, (w.dtype, s.dtype)
    xin = captured[tag]
    deq = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=4)
    y = xin @ deq.T
    mx.eval(y)
    N, K = deq.shape
    np.array(w).tofile(f"{out_dir}/{tag}.wq.bin")
    np.array(s.view(mx.uint16)).tofile(f"{out_dir}/{tag}.ws.bin")
    np.array(b.view(mx.uint16)).tofile(f"{out_dir}/{tag}.wb.bin")
    np.array(xin).tofile(f"{out_dir}/{tag}.x.bin")
    np.array(y).tofile(f"{out_dir}/{tag}.y.bin")
    print(tag, "M", xin.shape[0], "N", N, "K", K, "x absmax", float(mx.max(mx.abs(xin))), "y rms", float(mx.sqrt(mx.mean(y * y))))
