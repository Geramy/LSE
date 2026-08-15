#!/usr/bin/env python3
"""Dump reference activations from lemonseed for the C++ differential tests.

Run against the real checkpoint with MLX on CPU. Output is a safetensors file
the C++ side reads with its own loader, so no extra parser is needed.

    PYTHONPATH=reference/lemonseed .venv/bin/python scripts/dump_reference.py

lemonseed is used via PYTHONPATH, never installed, so the reference checkout is
left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Vendor fast-kernel paths off: they are selected by hasattr, so on a CPU-only
# box the Metal variants are picked and then fail at call time. The pure-MLX
# path is also the right oracle — no hand-written kernel between the reference
# algorithm and the numbers the C++ side is compared against.
for _killswitch in (
    "LEMONSEED_MOE_ROUTE_METAL", "LEMONSEED_MOE_ELEM_METAL",
    "LEMONSEED_MOD_SCATTER_METAL", "LEMONSEED_RESIDUAL_METAL",
    "LEMONSEED_GDN_METAL", "LEMONSEED_GDN_GLUE_HIP",
):
    os.environ.setdefault(_killswitch, "0")

import mlx.core as mx

# The oracle must be the clean CPU path: the ROCm backend adds GPU-specific
# numerics (and its runtime JIT needs ROCM_PATH pointed at the 7.13 component).
mx.set_default_device(mx.cpu)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/Documents/Dev/LDE/model"))
    ap.add_argument("--out", default="tests/data/reference.safetensors")
    ap.add_argument("--tokens", default="1,42,1337,248043,7,7,99,12345")
    args = ap.parse_args()

    from lemonseed.config import Config
    from lemonseed.model import HybridLM

    model_dir = Path(args.model)
    cfg = Config.from_dict(json.loads((model_dir / "model.json").read_text()))
    # fp32 on both sides: the CPU GatherMM used by the stacked experts is
    # fp32-only, and running the reference in fp32 from the same widened bf16
    # weights makes any C++ mismatch algorithmic rather than precision noise.
    cfg.dtype = "float32"
    print(f"config: {cfg.num_layers} layers, hidden {cfg.hidden_size}, "
          f"vocab {cfg.vocab_size}")

    model = HybridLM(cfg)
    weights = {k: v.astype(mx.float32)
               for k, v in mx.load(str(model_dir / "model.safetensors")).items()}
    model.load_weights(list(weights.items()), strict=False)
    # nn.Module.training defaults to True, which routes MoD through the
    # training-only top-k capacity gather. An inference engine must be compared
    # against the inference path: per-token sigmoid over every position.
    model.eval()
    mx.eval(model.parameters())
    print(f"loaded {len(weights)} tensors")

    ids = [int(t) for t in args.tokens.split(",")]
    tokens = mx.array([ids], dtype=mx.int32)
    print(f"tokens: {ids}")

    out: dict[str, mx.array] = {}
    out["tokens"] = tokens.astype(mx.int32)

    # Embedding, then every block boundary, so a mismatch localizes to a layer
    # instead of just "the logits differ".
    from lemonseed.model import embed_tokens

    x = embed_tokens(model.embed, tokens)
    mx.eval(x)
    out["embed"] = x.astype(mx.float32)

    for i, block in enumerate(model.blocks):
        x, aux, _ = block(x)
        mx.eval(x, aux)
        out[f"block.{i}"] = x.astype(mx.float32)
        out[f"block.{i}.aux"] = aux.astype(mx.float32).reshape(1)

    h = model.final_norm(x)
    mx.eval(h)
    out["final_norm"] = h.astype(mx.float32)

    # Logits for the last position only: [T, 248320] would be enormous.
    logits = model._lm_head(h[:, -1:, :])
    mx.eval(logits)
    out["logits_last"] = logits.astype(mx.float32)

    # Sub-layer probes for every block, replaying the stack so each probe's
    # input is the one the real forward pass uses. The block-N probes below use
    # synthetic inputs (embed, or block.N-1 straight into norm2) and so cannot
    # localize a mismatch inside an assembled block.
    xs = out["embed"]
    for i, block in enumerate(model.blocks):
        n1 = block.norm1(xs)
        mx.eval(n1)
        out[f"probe.b{i}.norm1"] = n1.astype(mx.float32)
        if block.is_attn:
            mixed, _ = block.mixer(n1)
        else:
            mixed, _ = block.mixer(n1)
        mx.eval(mixed)
        out[f"probe.b{i}.mixer"] = mixed.astype(mx.float32)

        mid = xs + mixed
        n2 = block.norm2(mid)
        mx.eval(n2)
        out[f"probe.b{i}.norm2"] = n2.astype(mx.float32)

        routed, shared, _a = block.moe(n2, split_shared=True)
        gated = block.mod.gate_all(n2, routed)
        mx.eval(routed, shared, gated)
        out[f"probe.b{i}.routed"] = routed.astype(mx.float32)
        out[f"probe.b{i}.shared"] = shared.astype(mx.float32)
        out[f"probe.b{i}.gated"] = gated.astype(mx.float32)

        xs = mid + shared + gated
        mx.eval(xs)

    # Isolated sub-layer probes on the real weights, so each C++ layer can be
    # tested on its own rather than only end-to-end.
    x0 = out["embed"]
    b0 = model.blocks[0]
    n1 = b0.norm1(x0)
    mx.eval(n1)
    out["probe.block0.norm1"] = n1.astype(mx.float32)
    m0, _ = b0.mixer(n1)
    mx.eval(m0)
    out["probe.block0.mixer_gdn"] = m0.astype(mx.float32)

    # MoE/MoD probes on block 0, matching the C++ decomposition exactly:
    # routed experts (ungated), shared expert, and the MoD-gated routed output.
    x1 = x0 + m0
    n2 = b0.norm2(x1)
    mx.eval(n2)
    out["probe.block0.norm2"] = n2.astype(mx.float32)
    routed, shared, _aux = b0.moe(n2, split_shared=True)
    mx.eval(routed, shared)
    out["probe.block0.moe_routed"] = routed.astype(mx.float32)
    out["probe.block0.moe_shared"] = shared.astype(mx.float32)
    gated = b0.mod.gate_all(n2, routed)
    mx.eval(gated)
    out["probe.block0.mod_gated"] = gated.astype(mx.float32)

    # Layer 19 is the only fully-populated MoE layer (all 8 routed experts
    # trained); layers 0/1/9 have effectively zero routed weights, so routing
    # cannot be validated there.
    xL = out["block.18"]
    b19 = model.blocks[19]
    n2L = b19.norm2(xL)
    mx.eval(n2L)
    out["probe.block19.norm2"] = n2L.astype(mx.float32)
    rL, sL, _a = b19.moe(n2L, split_shared=True)
    mx.eval(rL, sL)
    out["probe.block19.moe_routed"] = rL.astype(mx.float32)
    out["probe.block19.moe_shared"] = sL.astype(mx.float32)

    b3 = model.blocks[3]
    n1b = b3.norm1(x0)
    mx.eval(n1b)
    out["probe.block3.norm1"] = n1b.astype(mx.float32)
    m3, _ = b3.mixer(n1b)
    mx.eval(m3)
    out["probe.block3.mixer_attn"] = m3.astype(mx.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_path), out)

    meta = out_path.with_suffix(".json")
    meta.write_text(json.dumps({
        "tokens": ids,
        "model": str(model_dir),
        "mlx_backend": str(mx.default_device()),
        "tensors": {k: list(v.shape) for k, v in out.items()},
    }, indent=2))

    print(f"wrote {out_path} ({len(out)} tensors)")
    for k, v in out.items():
        print(f"  {k:28s} {list(v.shape)}")


if __name__ == "__main__":
    main()
