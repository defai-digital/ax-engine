#!/usr/bin/env python3
"""Run an isolated MLX-VLM affine8 PLE diagnostic on the immutable Flash Next pack."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import time
from unittest.mock import patch

import mlx.core as mx
import numpy as np

if __package__:
    from . import probe_flash_next_affine8_ple as ple
    from . import run_flash_next_reference as reference
else:
    import probe_flash_next_affine8_ple as ple
    import run_flash_next_reference as reference

TOKEN_BUDGET = 32


@dataclass(frozen=True)
class Plane:
    path: Path
    offset: int
    row_bytes: int


class Affine8Table:
    row_width = 160

    def __init__(self, root: Path):
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        config = json.loads((root / "config.json").read_text())
        quantization = config.get("quantization_config") or config["quantization"]
        prefixes = {key.rsplit(".", 1)[0] for key in index if ple.MARKER in key}
        shards = sorted(prefixes, key=lambda prefix: int(prefix.partition(ple.MARKER)[2]))
        numbers = [int(prefix.partition(ple.MARKER)[2]) for prefix in shards]
        if not numbers or numbers != list(range(config["text_config"]["split_ngram_parts"])):
            raise ValueError("PLE requires the complete ordered shard set")

        headers = {}
        self.shards: list[tuple[int, dict[str, Plane]]] = []
        for prefix in shards:
            settings = quantization.get(prefix, quantization)
            if any(settings.get(key) != value for key, value in ple.AFFINE8.items()):
                raise ValueError(f"unsupported PLE quantization for {prefix}")
            planes = {}
            row_count = None
            for plane, (dtype, columns, row_bytes) in ple.LAYOUT.items():
                key = f"{prefix}.{plane}"
                path = root / index[key]
                if path.parent != root or not path.resolve().is_relative_to(root):
                    raise ValueError("PLE member escapes the model directory")
                if path not in headers:
                    with path.open("rb") as handle:
                        length, = struct.unpack("<Q", handle.read(8))
                        if not 0 < length <= 16 * 1024 * 1024:
                            raise ValueError("invalid safetensors header length")
                        headers[path] = (8 + length, json.loads(handle.read(length)))
                payload, header = headers[path]
                spec = header[key]
                rows = spec["shape"][0]
                begin, end = spec["data_offsets"]
                if (spec["dtype"] != dtype or spec["shape"] != [rows, columns] or rows <= 0
                        or (row_count is not None and row_count != rows)
                        or begin < 0 or end - begin != rows * row_bytes
                        or payload + end > path.stat().st_size):
                    raise ValueError(f"invalid affine8 row layout for {key}")
                row_count = rows
                planes[plane] = Plane(path, payload + begin, row_bytes)
            self.shards.append((row_count, planes))
        self.ends = np.cumsum([rows for rows, _ in self.shards], dtype=np.int64)
        self.row_count = int(self.ends[-1])
        self.row_reads = 0
        self.bytes_read = 0

    def __call__(self, indices: mx.array) -> mx.array:
        ids = np.asarray(indices)
        if (not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= self.row_count)):
            raise IndexError("invalid PLE row index")
        shape = (*ids.shape, self.row_width)
        if not ids.size:
            return mx.zeros(shape, dtype=mx.bfloat16)
        unique, restore = np.unique(ids.reshape(-1), return_inverse=True)
        shard_indices = np.searchsorted(self.ends, unique, side="right")
        payload: dict[str, bytearray] = {plane: bytearray() for plane in ple.LAYOUT}
        with ExitStack() as stack:
            opened = {}
            for shard in np.unique(shard_indices):
                start = int(self.ends[shard - 1]) if shard else 0
                local = [int(index) - start for index in unique[shard_indices == shard]]
                rows, planes = self.shards[shard]
                for plane, source in planes.items():
                    if source.path not in opened:
                        opened[source.path] = stack.enter_context(source.path.open("rb"))
                    payload[plane].extend(ple.read_rows(
                        opened[source.path], source.offset, rows, source.row_bytes, local,
                    ))
        self.row_reads += len(unique)
        self.bytes_read += sum(map(len, payload.values()))
        values = ple.dequantize_rows({plane: bytes(data) for plane, data in payload.items()})
        return values[mx.array(restore, dtype=mx.int32)].reshape(shape)


def load_language_model(root: Path, table: Affine8Table | None):
    from mlx_vlm import utils
    from mlx_vlm.models.qwen4_exp import ple_storage

    # The pinned loader can rewrite a header when a dtype fallback is needed.
    with ExitStack() as stack:
        stack.enter_context(patch.object(utils, "_load_safetensors", mx.load))
        if table is not None:
            original_config = utils.load_config

            def configured(path, **kwargs):
                config = original_config(path, **kwargs)
                if Path(path).resolve() == root:
                    config["text_config"]["ple_storage"] = {"manifest": str(root), "cache_rows": 0}
                return config

            def attach(path, *, cache_rows=None):
                if Path(path) != root or cache_rows != 0:
                    raise ValueError("unexpected diagnostic PLE constructor settings")
                return table

            stack.enter_context(patch.object(utils, "load_config", configured))
            stack.enter_context(patch.object(ple_storage, "QuantizedMMapNGramEmbedding", attach))
        return utils.load_model(root, lazy=True, strict=True).language_model


def run(root: Path, mode: str, output: Path) -> None:
    from mlx_vlm.models.qwen4_exp import language

    versions = {name: importlib.metadata.version(name) for name in reference.PINNED}
    language_sha = reference.digest(Path(language.__file__))
    if versions != reference.PINNED or language_sha != reference.LANGUAGE_SHA256:
        raise ValueError("diagnostic requires the pinned MLX-VLM environment")
    if mx.default_device() != mx.gpu:
        raise ValueError("diagnostic requires Metal dequantization")
    root = root.resolve()
    inventory = json.loads(ple.INVENTORY.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "ax-engine.flash-next.affine8-ple-diagnostic.v1",
        "reference": {"runtime": "mlx-vlm-affine8-ple-diagnostic", "commit": reference.MLX_VLM_COMMIT},
        "mode": mode, "completed": False, "qualification": False, "release_ready": False,
        "pack_revision": inventory["revision"], "inventory_sha256": reference.digest(ple.INVENTORY),
        "versions": versions, "language_sha256": language_sha, "backend": "metal",
        "hardware": {
            "soc": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
            "os": platform.mac_ver()[0],
        },
        "prefill_chunk": reference.PREFILL_CHUNK, "output_tokens": TOKEN_BUDGET, "cases": [],
    }

    def save(phase):
        record["phase"] = phase
        temporary = output / "result.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "result.json")
        print(phase, flush=True)

    save("prehash")
    pack_before = reference.native.validate_inventory(root, inventory)
    runtime_before = reference.runtime_hashes()
    mx.set_memory_limit(reference.MEMORY_LIMIT)
    mx.set_cache_limit(reference.CACHE_LIMIT)
    table = Affine8Table(root) if mode == "bounded" else None
    save("loading")
    started = time.monotonic()
    model = load_language_model(root, table)
    record["weights_materialized"] = reference.materialize(model, mx)
    record["active_after_load_bytes"] = mx.get_active_memory()
    record["load_seconds"] = time.monotonic() - started
    save("loaded")
    vocab = reference.matrix.pack_vocab_size(root)
    for length in reference.LENGTHS:
        ids = reference.matrix.workload_ids(vocab, length, 0)
        result = reference.generate(model, mx, ids, TOKEN_BUDGET, stop_on_eos=False)
        if len(result["generated_ids"]) != TOKEN_BUDGET:
            raise ValueError(f"short reference output for {length} tokens")
        record["cases"].append({"prompt_tokens": length, "prompt_ids_sha256": reference.matrix.ids_sha(ids), **result})
        save("case")
    if table is not None:
        if not table.row_reads:
            raise ValueError("external PLE was never used")
        record["row_reads"] = table.row_reads
        record["bounded_payload_bytes"] = table.bytes_read
    record["peak_mlx_memory_bytes"] = mx.get_peak_memory()
    record["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    del model
    mx.clear_cache()
    save("posthash")
    if reference.native.validate_inventory(root, inventory) != pack_before:
        raise ValueError("pack changed during the diagnostic")
    if reference.runtime_hashes() != runtime_before:
        raise ValueError("reference runtime changed during the diagnostic")
    record["runtime_files_bound"] = len(runtime_before)
    record["completed"] = True
    save("complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("bounded", "resident"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model_dir, args.mode, args.output)


if __name__ == "__main__":
    main()
