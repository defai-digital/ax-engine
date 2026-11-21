#!/usr/bin/env python3
"""Run an isolated MLX-VLM affine8 PLE diagnostic on the immutable Flash Next pack."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
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


@dataclass
class RowIoTiming:
    lookups: int = 0
    # Converting the lazy MLX indices can wait for upstream GPU work.
    index_host_seconds: float = 0.0
    plan_seconds: float = 0.0
    select_and_read_seconds: float = 0.0
    # Includes the host payload copies used to construct the lazy graph.
    graph_build_seconds: float = 0.0


class Affine8Table:
    row_width = 160

    def __init__(self, root: Path, profile_row_io: bool = False):
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
        self.row_io = RowIoTiming() if profile_row_io else None

    def __call__(self, indices: mx.array) -> mx.array:
        timing = self.row_io
        started = time.perf_counter() if timing is not None else 0.0
        ids = np.asarray(indices)
        indices_ready = time.perf_counter() if timing is not None else 0.0
        if (not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= self.row_count)):
            raise IndexError("invalid PLE row index")
        shape = (*ids.shape, self.row_width)
        if not ids.size:
            return mx.zeros(shape, dtype=mx.bfloat16)
        unique, restore = np.unique(ids.reshape(-1), return_inverse=True)
        shard_indices = np.searchsorted(self.ends, unique, side="right")
        payload: dict[str, bytearray] = {plane: bytearray() for plane in ple.LAYOUT}
        planned = time.perf_counter() if timing is not None else 0.0
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
        read = time.perf_counter() if timing is not None else 0.0
        self.row_reads += len(unique)
        self.bytes_read += sum(map(len, payload.values()))
        values = ple.dequantize_rows({plane: bytes(data) for plane, data in payload.items()})
        result = values[mx.array(restore, dtype=mx.int32)].reshape(shape)
        if timing is not None:
            timing.lookups += 1
            timing.index_host_seconds += indices_ready - started
            timing.plan_seconds += planned - indices_ready
            timing.select_and_read_seconds += read - planned
            timing.graph_build_seconds += time.perf_counter() - read
        return result


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


@dataclass(frozen=True)
class FullWorkload:
    items: list[dict]
    prompts: list[dict]
    vocab_size: int
    prompt_hashes: dict[str, str]
    evidence_hashes: dict[str, str]


def prepare_full_workload(root: Path, ax_matrix: Path) -> FullWorkload:
    qa = reference.qa
    items_path = qa.FROZEN / "full-qa-items.json"
    prompts_path = qa.FROZEN / "full-qa-inputs.json"
    items = qa.load_items(items_path)
    frozen = json.loads(prompts_path.read_text())
    evidence = {"items_sha256": reference.digest(items_path),
                "inputs_sha256": reference.digest(prompts_path),
                "ax_matrix_sha256": reference.digest(ax_matrix)}
    for name in ("closed_checks", "qa_checks", "checkers", "prompt_def"):
        evidence[f"{name}_sha256"] = reference.digest(qa.FROZEN / f"{name}.py")
    if (evidence["items_sha256"] != frozen["items_sha256"]
            or evidence["checkers_sha256"] != frozen["checker_sha256"]
            or evidence["prompt_def_sha256"] != frozen["prompt_definition_sha256"]):
        raise ValueError("QA definitions differ from the frozen inputs")
    prompts = frozen["prepared"]
    if [item["id"] for item in items] != [prompt["id"] for prompt in prompts]:
        raise ValueError("frozen QA prompts do not follow the item order")

    expected = {"revision": reference.native.PACK_REVISION,
                "inventory_sha256": reference.digest(ple.INVENTORY),
                "lengths": list(reference.LENGTHS), "output_tokens": reference.OUTPUT_TOKENS,
                "warmups": reference.WARMUPS, "measured": reference.MEASURED,
                "workload_seed": reference.matrix.WORKLOAD_SEED, "temperature": 0}
    recorded = json.loads(ax_matrix.read_text())
    if any(recorded.get(key) != value for key, value in expected.items()):
        raise ValueError("AX matrix uses a different pack or workload")
    vocab = reference.matrix.pack_vocab_size(root)
    hashes = reference.window_hashes(vocab, reference.LENGTHS)
    for mode in ("disabled", "default"):
        reference.verify_prompt_hashes(recorded["runs"][mode]["prompt_id_hashes"], hashes)
    return FullWorkload(items, prompts, vocab, hashes, evidence)


def generate_case(model, ids: list[int], budget: int, *, stop_on_eos: bool = True,
                  table: Affine8Table | None = None) -> dict:
    timing = table.row_io if table is not None else None
    before = asdict(timing) if timing is not None else {}
    result = reference.generate(model, mx, ids, budget, stop_on_eos=stop_on_eos)
    if timing is not None:
        result["ple_row_io"] = {key: value - before[key] for key, value in asdict(timing).items()}
    return result


def run_full_workload(model, tokenizer, workload: FullWorkload, record: dict, save,
                      table: Affine8Table | None = None) -> None:
    qa_rows: list[dict] = []
    record["qa"] = {"rows": qa_rows}
    for item, prompt in zip(workload.items, workload.prompts, strict=True):
        ids = prompt["input_ids"]
        if tokenizer.encode(prompt["rendered"], add_special_tokens=False) != ids:
            raise ValueError(f"tokenizer disagrees with the frozen input IDs: {item['id']}")
        result = generate_case(model, ids, reference.qa.budget_for(item), table=table)
        text = tokenizer.decode(result["generated_ids"], skip_special_tokens=True)
        qa_rows.append({"id": item["id"], "prompt_tokens": len(ids), "text": text, **result})
        save("qa")
    grades = reference.qa.grade(workload.items, {row["id"]: row["text"] for row in qa_rows})
    record["qa"].update(grades=grades,
                        failures=sorted(item for item, grade in grades.items() if not grade["passed"]),
                        normal_stops=all(row["normal_stop"] for row in qa_rows))
    save("qa")

    cells: list[dict] = []
    record["matrix"] = {"vocab_size": workload.vocab_size, "prompt_id_hashes": workload.prompt_hashes,
                        "cells": cells}
    for length in reference.LENGTHS:
        for repeat in range(reference.WARMUPS + reference.MEASURED):
            ids = reference.matrix.workload_ids(workload.vocab_size, length, repeat)
            result = generate_case(model, ids, reference.OUTPUT_TOKENS, stop_on_eos=False, table=table)
            if len(result["generated_ids"]) != reference.OUTPUT_TOKENS:
                raise ValueError(f"short reference output for {length} tokens")
            cells.append({"length": length, "run": repeat, "warmup": repeat < reference.WARMUPS, **result})
            save("matrix")
    record["matrix"]["summary"] = reference.summarize_matrix(cells, reference.LENGTHS)


def run(root: Path, mode: str, output: Path, ax_matrix: Path | None = None,
        profile_row_io: bool = False) -> None:
    if profile_row_io and mode != "bounded":
        raise ValueError("row profiling requires bounded mode")
    from mlx_vlm.models.qwen4_exp import language

    versions = {name: importlib.metadata.version(name) for name in reference.PINNED}
    language_sha = reference.digest(Path(language.__file__))
    if versions != reference.PINNED or language_sha != reference.LANGUAGE_SHA256:
        raise ValueError("diagnostic requires the pinned MLX-VLM environment")
    if mx.default_device() != mx.gpu:
        raise ValueError("diagnostic requires Metal dequantization")
    root = root.resolve()
    inventory = json.loads(ple.INVENTORY.read_bytes())
    workload = prepare_full_workload(root, ax_matrix) if ax_matrix is not None else None
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
        "prefill_chunk": reference.PREFILL_CHUNK,
        "output_tokens": reference.OUTPUT_TOKENS if workload is not None else TOKEN_BUDGET,
        "workload": "frozen-qa-and-matrix" if workload is not None else "smoke",
    }
    if profile_row_io:
        record["profile_row_io"] = True
    if workload is not None:
        record["workload_hashes"] = workload.evidence_hashes
    else:
        record["cases"] = []

    def save(phase):
        record["phase"] = phase
        temporary = output / "result.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "result.json")
        print(phase, flush=True)

    save("prehash")
    pack_before = reference.native.validate_inventory(root, inventory)
    runtime_before = reference.runtime_hashes()
    if workload is not None:
        record["host_quiet"] = reference.native.wait_for_quiet()
    mx.set_memory_limit(reference.MEMORY_LIMIT)
    mx.set_cache_limit(reference.CACHE_LIMIT)
    table = Affine8Table(root, profile_row_io) if mode == "bounded" else None
    save("loading")
    started = time.monotonic()
    model = load_language_model(root, table)
    record["weights_materialized"] = reference.materialize(model, mx)
    record["active_after_load_bytes"] = mx.get_active_memory()
    record["load_seconds"] = time.monotonic() - started
    save("loaded")
    if workload is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        run_full_workload(model, tokenizer, workload, record, save, table)
    else:
        vocab = reference.matrix.pack_vocab_size(root)
        for length in reference.LENGTHS:
            ids = reference.matrix.workload_ids(vocab, length, 0)
            result = generate_case(model, ids, TOKEN_BUDGET, stop_on_eos=False, table=table)
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
    if workload is not None:
        if prepare_full_workload(root, ax_matrix).evidence_hashes != workload.evidence_hashes:
            raise ValueError("frozen workload evidence changed during the diagnostic")
        record["host_idle_after"] = reference.native.cpu_idle_percent()
    record["runtime_files_bound"] = len(runtime_before)
    record["completed"] = True
    save("complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("bounded", "resident"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ax-matrix", type=Path,
                        help="run all frozen QA and matrix cases, checking prompt hashes against this native result")
    parser.add_argument("--profile-row-io", action="store_true",
                        help="measure host index synchronization, selection, reads and graph construction per case")
    args = parser.parse_args()
    if args.profile_row_io and args.mode != "bounded":
        parser.error("--profile-row-io requires bounded mode")
    run(args.model_dir, args.mode, args.output, args.ax_matrix, args.profile_row_io)


if __name__ == "__main__":
    main()
