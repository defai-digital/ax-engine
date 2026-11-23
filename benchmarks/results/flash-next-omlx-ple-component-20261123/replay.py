"""Check oMLX's Flash Next PLE row addressing on the pinned AXQ MXFP4 pack."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import struct
import subprocess

import mlx.core as mx
import numpy as np


REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
PACK_HASHES = {
    "model-manifest.json": "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef",
    "model.safetensors.index.json": "37344288777e7e78b35901cb1d572dad0dc640cb9cedb90024860b932a657696",
}
OMLX = "5eeec7757f83bfc99eeb87abe49686d00a0bba1f"
MLX_VLM = "138d7617d39653b39a5cbc379b1b5907a4481d67"
VENDOR_DIR = "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm"
VENDOR_HASH = "65704a21e611ad0c69c38524ec5f0b4e4292a3f9a87b10aee3cdc7cdd5623e9e"
SHARDS = 128
ROWS_PER_SHARD = 2_500_012
WIDTH = 160
PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
FIELDS = {
    "weight": ("U32", "<u4", 40),
    "scales": ("BF16", "<u2", 5),
    "biases": ("BF16", "<u2", 5),
}


def head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def positional_rows(pack, index, coordinates):
    buffers = {
        name: np.empty((len(coordinates), width), dtype=dtype)
        for name, (_, dtype, width) in FIELDS.items()
    }
    payload_bytes = 0
    with ExitStack() as stack:
        files = {}
        for number, (shard, row) in enumerate(coordinates):
            for name, (dtype, numpy_dtype, width) in FIELDS.items():
                key = f"{PREFIX}.shards.{shard}.{name}"
                filename = index[key]
                if filename not in files:
                    handle = stack.enter_context((pack / filename).open("rb"))
                    size = struct.unpack("<Q", handle.read(8))[0]
                    files[filename] = (handle.fileno(), 8 + size, json.loads(handle.read(size)))
                fd, data_start, header = files[filename]
                tensor = header[key]
                assert tensor["dtype"] == dtype and tensor["shape"] == [ROWS_PER_SHARD, width]
                offset, end = tensor["data_offsets"]
                row_bytes = width * np.dtype(numpy_dtype).itemsize
                assert end - offset == ROWS_PER_SHARD * row_bytes
                raw = os.pread(fd, row_bytes, data_start + offset + row * row_bytes)
                assert len(raw) == row_bytes
                buffers[name][number] = np.frombuffer(raw, dtype=numpy_dtype)
                payload_bytes += len(raw)
    values = mx.dequantize(
        mx.array(buffers["weight"]),
        mx.array(buffers["scales"]).view(mx.bfloat16),
        mx.array(buffers["biases"]).view(mx.bfloat16),
        bits=8,
        group_size=32,
        mode="affine",
    )
    return values.astype(mx.bfloat16), payload_bytes


def expect_exception(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def replay(pack, omlx, mlx_vlm):
    from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
    import mlx_vlm as base_vlm

    assert head(omlx) == OMLX and head(mlx_vlm) == MLX_VLM
    assert Path(base_vlm.__file__).resolve().parent == mlx_vlm / "mlx_vlm"
    assert version("mlx") == "0.32.4"
    for name, digest in PACK_HASHES.items():
        assert sha256((pack / name).read_bytes()).hexdigest() == digest
    index = json.loads((pack / "model.safetensors.index.json").read_bytes())["weight_map"]
    keys = {key for key in index if key.startswith(PREFIX + ".")}
    assert keys == {
        f"{PREFIX}.shards.{shard}.{name}" for shard in range(SHARDS) for name in FIELDS
    }
    assert apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    vendor = Path(language.__file__).resolve()
    assert vendor == omlx / VENDOR_DIR / "models/qwen4_exp/language.py"
    assert sha256(vendor.read_bytes()).hexdigest() == VENDOR_HASH

    vocab = SHARDS * ROWS_PER_SHARD
    coordinates = [
        (shard, row)
        for row in (0, ROWS_PER_SHARD // 2, ROWS_PER_SHARD - 1)
        for shard in range(SHARDS - 1, -1, -1)
    ]
    coordinates += [coordinates[i] for i in (0, 127, 128, 255, 256, 383, 0, 383)]
    oracle, payload_bytes = positional_rows(pack, index, coordinates)
    ids = mx.array(
        [shard * ROWS_PER_SHARD + row for shard, row in coordinates], dtype=mx.int64
    ).reshape(4, -1)
    expected = np.asarray(oracle.astype(mx.float32)).astype("<f4", copy=False).tobytes()
    reader = language.DiskBackedShardedEmbedding(pack, PREFIX, vocab, WIDTH, SHARDS)
    modes = {}
    try:
        for prefetch in (False, True):
            if prefetch:
                reader.prefetch(ids)
            actual = reader(ids)
            assert actual.shape == (*ids.shape, WIDTH) and actual.dtype == mx.bfloat16
            observed = np.asarray(actual.astype(mx.float32)).astype("<f4", copy=False).tobytes()
            assert observed == expected
            assert reader.last_prefetch_hit == prefetch
            assert reader.last_touched_shards == tuple(range(SHARDS))
            assert reader.rows_read == ids.size and reader.last_uploads == len(FIELDS)
            modes["prefetched" if prefetch else "synchronous"] = {
                "values_equal": ids.size * WIDTH,
                "bf16_as_f32le_sha256": sha256(observed).hexdigest(),
            }
        for invalid in (-1, vocab):
            expect_exception(IndexError, lambda: reader(mx.array([invalid], dtype=mx.int64)))
    finally:
        reader.close()
    expect_exception(RuntimeError, lambda: reader(ids))
    return {
        "scope": "PLE component row addressing and prefetch",
        "pack_revision": REVISION,
        "pack_hashes": PACK_HASHES,
        "omlx_commit": OMLX,
        "mlx_vlm_commit": MLX_VLM,
        "vendor_sha256": VENDOR_HASH,
        "mlx_version": version("mlx"),
        "layout": {"shards": SHARDS, "bits": 8, "group_size": 32, "width": WIDTH},
        "file_count": len({index[key] for key in keys}),
        "positions_per_mode": ids.size,
        "positional_oracle_payload_bytes": payload_bytes,
        "modes": modes,
        "negative_controls": ["negative index", "past vocabulary", "closed reader"],
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pack", type=Path)
    cli.add_argument("omlx", type=Path)
    cli.add_argument("mlx_vlm", type=Path)
    args = cli.parse_args()
    receipt = replay(args.pack.resolve(), args.omlx.resolve(), args.mlx_vlm.resolve())
    print(json.dumps(receipt, indent=2, sort_keys=True))
