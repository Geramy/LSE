#!/usr/bin/env python3
"""Convert an unsharded BF16 DFlash2 checkpoint to affine Q8/group64.

Requires NumPy. The source checkpoint is read only; the destination must be new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np

GROUP_SIZE = 64
REFERENCE_URL = (
    "https://github.com/z-lab/dflash/blob/"
    "07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py"
)
CODEBOOKS = {
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bf16(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def widen(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


def nearest(values: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(values) + np.float32(0.5)), values)


def read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated safetensors payload")
    return data


def quantize_matrix(stream, start: int, shape: list[int]):
    rows, columns = shape
    if rows <= 0 or columns <= 0 or columns % GROUP_SIZE:
        raise ValueError(f"matrix shape is not group64 compatible: {shape}")
    packed = np.empty((rows, columns // 4), dtype="<u4")
    scales = np.empty((rows, columns // GROUP_SIZE), dtype="<u2")
    biases = np.empty_like(scales)
    for first in range(0, rows, 128):
        count = min(128, rows - first)
        stream.seek(start + first * columns * 2)
        raw = np.frombuffer(read_exact(stream, count * columns * 2), dtype="<u2")
        values = widen(raw).reshape(count, columns // GROUP_SIZE, GROUP_SIZE)
        lo, hi = values.min(axis=-1), values.max(axis=-1)
        low_edge = np.abs(lo) > np.abs(hi)
        scale = np.maximum((hi - lo) / np.float32(255), np.float32(1e-7))
        scale = np.where(low_edge, scale, -scale)
        edge = np.where(low_edge, lo, hi)
        code_at_edge = nearest(edge / scale)
        scale = np.divide(edge, code_at_edge, out=scale.copy(), where=code_at_edge != 0)
        bias = np.where(code_at_edge != 0, edge, np.float32(0))
        stored_scale, stored_bias = bf16(scale), bf16(bias)
        scale, bias = widen(stored_scale), widen(stored_bias)
        inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
        codes = np.clip(nearest((values - bias[..., None]) * inverse[..., None]), 0, 255)
        codes = codes.astype(np.uint32).reshape(count, columns // 4, 4)
        packed[first:first + count] = (
            codes[:, :, 0] | (codes[:, :, 1] << 8)
            | (codes[:, :, 2] << 16) | (codes[:, :, 3] << 24)
        )
        scales[first:first + count] = stored_scale
        biases[first:first + count] = stored_bias
    return packed, scales, biases


def convert(source: Path, destination: Path, repository: str, revision: str) -> None:
    source, destination = source.resolve(), destination.resolve()
    if destination == source or destination.exists():
        raise ValueError("destination must be a new directory separate from the source")
    weight = source / "model.safetensors"
    config = json.loads((source / "config.json").read_text())
    if "DFlash2DraftModel" not in config.get("architectures", []):
        raise ValueError("source config does not describe a DFlash2DraftModel")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("source must be an unquantized BF16 checkpoint")
    with weight.open("rb") as stream:
        header_size = struct.unpack("<Q", read_exact(stream, 8))[0]
        if header_size > weight.stat().st_size - 8:
            raise ValueError("invalid safetensors header size")
        header = json.loads(read_exact(stream, header_size))
    data_start = 8 + header_size
    source_hash = sha256(weight)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + ".", dir=destination.parent) as tmp:
        work = Path(tmp)
        payload = work / "model.payload"
        tensors, offset = {}, 0

        def write_tensor(stream, name, dtype, shape, data):
            nonlocal offset
            tensors[name] = {"dtype": dtype, "shape": list(shape),
                             "data_offsets": [offset, offset + len(data)]}
            stream.write(data)
            offset += len(data)

        with weight.open("rb") as incoming, payload.open("wb") as outgoing:
            for original_name, entry in header.items():
                if original_name == "__metadata__":
                    continue
                shape, (begin, end) = entry["shape"], entry["data_offsets"]
                if begin < 0 or end < begin or data_start + end > weight.stat().st_size:
                    raise ValueError(f"invalid tensor offsets: {original_name}")
                name = original_name + ".weight" if original_name in CODEBOOKS else original_name
                if len(shape) != 2 or not name.endswith(".weight"):
                    incoming.seek(data_start + begin)
                    write_tensor(outgoing, name, entry["dtype"], shape,
                                 read_exact(incoming, end - begin))
                    continue
                if entry["dtype"] != "BF16" or end - begin != shape[0] * shape[1] * 2:
                    raise ValueError(f"expected a BF16 matrix: {original_name}")
                packed, scales, biases = quantize_matrix(incoming, data_start + begin, shape)
                stem = name[:-7]
                write_tensor(outgoing, name, "U32", packed.shape, packed.tobytes())
                write_tensor(outgoing, stem + ".scales", "BF16", scales.shape, scales.tobytes())
                write_tensor(outgoing, stem + ".biases", "BF16", biases.shape, biases.tobytes())
                print(f"{name}: {shape[0]} x {shape[1]}", flush=True)
        raw_header = json.dumps(tensors, separators=(",", ":")).encode()
        raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
        output_weight = work / "model.safetensors"
        with output_weight.open("wb") as stream, payload.open("rb") as data:
            stream.write(struct.pack("<Q", len(raw_header)))
            stream.write(raw_header)
            shutil.copyfileobj(data, stream, 8 * 1024 * 1024)
        payload.unlink()
        config["quantization"] = {"bits": 8, "group_size": GROUP_SIZE, "mode": "affine"}
        (work / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        manifest = {
            "source_repository": repository,
            "source_revision": revision,
            "source_sha256": source_hash,
            "reference_implementation": REFERENCE_URL,
            "format": "MLX affine Q8 group64",
            "conversion": "FP32 min/max; BF16 scale and bias; codes selected against stored scale/bias",
            "bytes": output_weight.stat().st_size,
            "sha256": sha256(output_weight),
            "tensors": len(tensors),
        }
        (work / "source-repository.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(work, destination)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="BF16 checkpoint directory")
    parser.add_argument("destination", type=Path, help="new Q8 checkpoint directory")
    parser.add_argument("--source-repository", required=True, help="source Hugging Face repository")
    parser.add_argument("--source-revision", required=True, help="pinned source commit")
    args = parser.parse_args()
    convert(args.source, args.destination, args.source_repository, args.source_revision)


if __name__ == "__main__":
    main()
