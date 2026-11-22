"""Measure byte fidelity and allocation for typed MLX Metal views."""

from gc import collect
from hashlib import sha256
from importlib.metadata import version
import json

import mlx.core as mx

VIEW_BYTES = 65536


def probe(base, width, dtype, start):
    collect()
    mx.synchronize()
    before = mx.get_active_memory()
    typed = mx.view(base[start:start + VIEW_BYTES], dtype)
    mx.eval(typed)
    mx.synchronize()
    extra = mx.get_active_memory() - before
    wanted_extra = VIEW_BYTES if start % width else 0
    if extra != wanted_extra:
        raise ValueError(f"Expected {wanted_extra} bytes of typed-view allocation, observed {extra}")
    # Materialize a GPU operation over the resulting view before checking bytes.
    output = mx.bitwise_xor(mx.view(typed, mx.uint8), mx.array(17, dtype=mx.uint8))
    mx.eval(output)
    expected = bytes((index % 256) ^ 173 ^ 17 for index in range(start, start + VIEW_BYTES))
    actual = bytes(output.tolist())
    if actual != expected:
        raise ValueError("Metal typed view changed the byte pattern")
    return {"byte_offset": start, "scalar_aligned": start % width == 0,
            "extra_active_mlx_bytes": extra, "output_sha256": sha256(actual).hexdigest()}


def main():
    versions = {name: version(name) for name in ("mlx", "mlx-metal")}
    if set(versions.values()) != {"0.32.4"}:
        raise ValueError("The recorded diagnostic uses MLX and MLX Metal 0.32.4")
    mx.set_default_device(mx.gpu)
    initial = mx.arange(131072, dtype=mx.uint32).astype(mx.uint8)
    base = mx.bitwise_xor(initial, mx.array(173, dtype=mx.uint8))
    mx.eval(base)
    del initial
    cases = {}
    for label, width, dtype in (("BF16", 2, mx.bfloat16), ("U32", 4, mx.uint32), ("I64", 8, mx.int64)):
        cases[label] = [probe(base, width, dtype, start) for start in (16384, 16385)]
    print(json.dumps({"scope": "M2 Ultra synthetic MLX Metal typed views", "versions": versions,
                      "view_bytes": VIEW_BYTES, "cases": cases}, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
