import sys, numpy as np, mlx.core as mx
d, tag, M, N, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
G = K // 64
wq = mx.array(np.fromfile(f"{d}/{tag}.wq.bin", np.uint32).reshape(N, K // 8))
ws = mx.array(np.fromfile(f"{d}/{tag}.ws.bin", np.uint16).reshape(N, G)).view(mx.bfloat16).astype(mx.float32)
wb = mx.array(np.fromfile(f"{d}/{tag}.wb.bin", np.uint16).reshape(N, G)).view(mx.bfloat16).astype(mx.float32)
x = mx.array(np.fromfile(f"{d}/{tag}.x.bin", np.float32, M * K).reshape(M, K))
y = mx.array(np.fromfile(f"{d}/{tag}.y.bin", np.float32, M * N).reshape(M, N))
W = mx.dequantize(wq, ws, wb, group_size=64, bits=4)
def rel(a): return float(mx.sqrt(mx.sum((a - y) ** 2) / mx.sum(y * y)))
print("exact recompute", rel(x @ W.T))
xr = x.reshape(M, G, 64); amax = mx.max(mx.abs(xr), axis=-1, keepdims=True); st = amax / 127
xi = (mx.clip(mx.round(xr / mx.maximum(st, 1e-30)), -127, 127) * st).reshape(M, K)
print("int8_g64", rel(xi @ W.T))
am = mx.max(mx.abs(x), axis=-1, keepdims=True); sc = am / 448
xf = mx.from_fp8(mx.to_fp8(mx.clip(x / sc, -448, 448))).astype(mx.float32) * sc
print("fp8_tok", rel(xf @ W.T))
sc2 = amax / 448
xg = (mx.from_fp8(mx.to_fp8(mx.clip(xr / sc2, -448, 448))).astype(mx.float32) * sc2).reshape(M, K)
print("fp8_g64", rel(xg @ W.T))
