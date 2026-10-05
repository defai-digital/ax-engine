#!/usr/bin/env python3
"""Run the pinned MLX-VLM reference for Flash Next on the same pack bytes.

The reviewed upstream `mlx_lm` cannot load `qwen4_exp`, so the independently
implemented MLX-VLM language graph (0.7.0rc0, MLX 0.32.2, Transformers
5.18.0.dev0) is the labelled, non-primary reference for this family only. Two
phases share one model load:

* `qa`: the 105 frozen items (256-token cap, 64 for the long lookup) with greedy
  decoding on the frozen rendered prompts, graded with the frozen checkers.
* `matrix`: 512, 2048 and 8192 input tokens, 128 output tokens, two warmups and
  three measured runs per cell, on exactly the token windows the AX generate
  matrix uses (verified by hash against an AX matrix result).

Timing is in process: TTFT is prefill plus the first token; decode is the rate
of the tokens after the first. Prefill uses 1,024-token chunks and decoding is
pipelined one step ahead like `mlx_lm`'s generate loop. AX is timed through an
HTTP stream, so this is not an apples-to-apples transport; it is a labelled
reference, never an AX-versus-mlx-lm result.

Run it with the pinned reference environment's Python. Dry-run is weight-free.
This records reference observations for one pack/SKU; it is not qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Iterator

try:
    import check_flash_next_native_support as native
    import run_flash_next_generate_matrix as matrix
    import run_flash_next_qa_cohort as qa
except ImportError:
    from scripts import check_flash_next_native_support as native
    from scripts import run_flash_next_generate_matrix as matrix
    from scripts import run_flash_next_qa_cohort as qa

PINNED = {"mlx": "0.32.2", "mlx-metal": "0.32.2", "mlx-vlm": "0.7.0rc0",
          "transformers": "5.18.0.dev0", "tokenizers": "0.23.2"}
LANGUAGE_SHA256 = "30f059d6289451bde0f5ef816090ff946ebc857d1b557f83415e2ba867d76199"
MLX_VLM_COMMIT = "6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb"
EOS_IDS = {248044, 248046}
PREFILL_CHUNK = 1024
MEMORY_LIMIT = 144 * 2**30
CACHE_LIMIT = 2**30
OUTPUT_TOKENS = matrix.OUTPUT_TOKENS
LENGTHS = matrix.LENGTHS
WARMUPS = matrix.WARMUPS
MEASURED = matrix.MEASURED
PHASES = ("qa", "matrix")
RUNTIME_DISTRIBUTIONS = ("mlx", "mlx-metal", "mlx-vlm")


def collect(stream: Iterable[int], budget: int, eos: set[int],
            clock: Callable[[], float] = time.monotonic) -> tuple[list[int], int | None, list[float]]:
    """Drain a token stream: generated IDs, the terminal EOS (or None), a time per generated ID."""
    generated: list[int] = []
    times: list[float] = []
    for token in stream:
        if token in eos:
            return generated, token, times
        generated.append(token)
        times.append(clock())
        if len(generated) >= budget:
            break
    return generated, None, times


def timing_record(start: float, times: list[float], end: float) -> dict[str, Any]:
    """TTFT, the post-first-token decode window and its rate from per-token times."""
    record: dict[str, Any] = {"generated_tokens": len(times), "total_seconds": round(end - start, 6)}
    if times:
        record["ttft_seconds"] = round(times[0] - start, 6)
        if len(times) > 1 and times[-1] > times[0]:
            record["decode_tokens"] = len(times) - 1
            record["decode_seconds"] = round(times[-1] - times[0], 6)
            record["decode_tok_s"] = (len(times) - 1) / (times[-1] - times[0])
    return record


def token_stream(model: Any, mx: Any, ids: list[int], chunk: int = PREFILL_CHUNK) -> Iterator[int]:
    """Greedy decode with chunked prefill; after the first token one step stays in flight."""
    cache = model.make_cache()
    for offset in range(0, len(ids) - 1, chunk):
        part = ids[offset: min(offset + chunk, len(ids) - 1)]
        logits = model(mx.array([part], dtype=mx.uint32), cache=cache).logits
        mx.eval(logits)
        del logits

    def step(token: Any) -> Any:
        logits = model(token, cache=cache).logits[:, -1, :]
        return mx.argmax(logits.astype(mx.float32), axis=-1)

    current = step(mx.array([[ids[-1]]], dtype=mx.uint32))
    mx.async_eval(current)
    yield int(current.item())
    upcoming = step(current[None])
    mx.async_eval(upcoming)
    try:
        while True:
            after = step(upcoming[None])
            mx.async_eval(after)
            yield int(upcoming.item())
            upcoming = after
    finally:
        mx.synchronize()


def generate(model: Any, mx: Any, ids: list[int], budget: int) -> dict[str, Any]:
    start = time.monotonic()
    generated, terminal, times = collect(token_stream(model, mx, ids), budget, EOS_IDS)
    record = {"generated_ids": generated, "terminal_eos_id": terminal,
              "normal_stop": terminal is not None,
              **timing_record(start, times, time.monotonic())}
    mx.clear_cache()
    return record


def window_hashes(vocab_size: int, lengths: tuple[int, ...]) -> dict[str, str]:
    """Hashes of the exact token IDs this runner feeds, keyed like the AX matrix result."""
    return {f"cell-{length}-{run}": matrix.ids_sha(matrix.workload_ids(vocab_size, length, run))
            for length in lengths for run in range(WARMUPS + MEASURED)}


def verify_prompt_hashes(ax_hashes: dict[str, str], ours: dict[str, str]) -> dict[str, str]:
    """When an AX matrix result is supplied, every window must be the exact IDs AX used."""
    for key, value in ours.items():
        if ax_hashes.get(key) != value:
            raise ValueError(f"reference prompt window differs from the AX matrix: {key}")
    return ours


def summarize_matrix(cells: list[dict[str, Any]], lengths: tuple[int, ...]) -> dict[str, Any]:
    """Medians over the measured runs; a measured run that stopped early is reported, not hidden."""
    out: dict[str, Any] = {}
    for length in lengths:
        measured = [c for c in cells if c["length"] == length and not c["warmup"]]
        rates = [c["decode_tok_s"] for c in measured if c.get("decode_tok_s")]
        out[str(length)] = {
            "measured_runs": len(measured),
            "full_length_runs": sum(c["generated_tokens"] == OUTPUT_TOKENS for c in measured),
            "decode_tok_s_median": statistics.median(rates) if rates else None,
            "ttft_s_median": statistics.median(c["ttft_seconds"] for c in measured) if measured else None,
        }
    return out


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def runtime_hashes() -> dict[str, str]:
    paths: set[str] = set()
    for name in RUNTIME_DISTRIBUTIONS:
        dist = importlib.metadata.distribution(name)
        for file in dist.files or []:
            located = dist.locate_file(file)
            if not str(file).endswith(".pyc") and located.is_file():
                paths.add(str(located.resolve()))
    return {path: digest(Path(path)) for path in sorted(paths)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--ax-matrix", type=Path,
                        help="optional AX generate-matrix result.json whose prompt hashes must match")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--limit-items", type=int, help="smoke runs: first N QA items")
    parser.add_argument("--lengths", type=int, nargs="+", help="smoke runs: input lengths to measure")
    args = parser.parse_args()
    lengths = tuple(args.lengths) if args.lengths else LENGTHS
    contract = {"repo_id": native.PRIMARY_REPO, "revision": native.PACK_REVISION,
                "reference": {"runtime": "mlx-vlm", "commit": MLX_VLM_COMMIT, "versions": PINNED,
                              "language_sha256": LANGUAGE_SHA256,
                              "role": "labelled non-primary reference; mlx_lm cannot load qwen4_exp"},
                "phases": list(args.phases), "prefill_chunk": PREFILL_CHUNK,
                "memory_limit_bytes": MEMORY_LIMIT, "cache_limit_bytes": CACHE_LIMIT,
                "qa_items": 105, "qa_budgets": {"short": qa.SHORT_BUDGET, "long": qa.LONG_BUDGET},
                "output_tokens": OUTPUT_TOKENS, "lengths": list(lengths), "warmups": WARMUPS,
                "measured": MEASURED, "workload_seed": matrix.WORKLOAD_SEED, "temperature": 0,
                "timing": "in process; TTFT = prefill + first token; decode = tokens after the first",
                "qualification": False, "release_ready": False,
                "scope": "reference observations; no AX-versus-mlx-lm claim; MTP-S/P/D not assessed"}
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return 0
    if not all((args.model_dir, args.inventory, args.output)):
        parser.error("live run requires model-dir, inventory and output")
    if platform.system() != "Darwin":
        parser.error("live run requires macOS on the Ultra-class 192 GiB+ target")
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if "Ultra" not in chip or memory < 192 * 1024**3:
        parser.error("live run requires the Ultra-class 192 GiB+ Flash Next target")
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise ValueError(f"reference environment differs from the pin: {versions}")

    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp import language
    from mlx_vlm.utils import load_model
    from transformers import AutoTokenizer

    language_sha = digest(Path(language.__file__))
    if language_sha != LANGUAGE_SHA256:
        raise ValueError("MLX-VLM qwen4_exp language module differs from the audited file")
    inventory = json.loads(args.inventory.read_text())
    storage = native.storage_info(args.model_dir)
    manifest_before = native.validate_inventory(args.model_dir, inventory)
    runtime_before = runtime_hashes()
    args.output.mkdir(parents=True, exist_ok=False)
    host_quiet = native.wait_for_quiet()
    record: dict[str, Any] = {**contract, "completed": False, "phase": "loading", "versions": versions,
                              "host_quiet": host_quiet,
                              "language_sha256": language_sha, "inventory_sha256": native.sha256(args.inventory),
                              "hardware": {"form_factor": native.form_factor(), "soc": chip,
                                           "memory_bytes": memory, "os": platform.mac_ver()[0],
                                           "storage": {"declared": inventory["storage"], **storage}}}

    def save(phase: str | None = None) -> None:
        if phase:
            record["phase"] = phase
        temp = args.output / "result.tmp"
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(args.output / "result.json")
        print(json.dumps({"phase": record["phase"]}), flush=True)

    save("loading")
    mx.set_memory_limit(MEMORY_LIMIT)
    mx.set_cache_limit(CACHE_LIMIT)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                              trust_remote_code=False)
    items = qa.load_items(qa.FROZEN / "full-qa-items.json")
    if "matrix" in args.phases:
        vocab_size = matrix.pack_vocab_size(args.model_dir)
        window_ids = window_hashes(vocab_size, lengths)
        if args.ax_matrix:
            ax_hashes = json.loads(args.ax_matrix.read_text())["runs"]["disabled"]["prompt_id_hashes"]
            verify_prompt_hashes(ax_hashes, window_ids)
    model = load_model(args.model_dir, lazy=True, strict=True).language_model
    if "qa" in args.phases:
        prepared = json.loads((qa.FROZEN / "full-qa-inputs.json").read_text())["prepared"]
        if [p["id"] for p in prepared] != [i["id"] for i in items]:
            raise ValueError("frozen prepared inputs do not follow the frozen item order")
        rows: list[dict[str, Any]] = []
        record["qa"] = {"rows": rows}
        save("qa")
        for item, prompt in list(zip(items, prepared, strict=True))[: args.limit_items]:
            if tokenizer.encode(prompt["rendered"], add_special_tokens=False) != prompt["input_ids"]:
                raise ValueError(f"tokenizer disagrees with the frozen input IDs: {item['id']}")
            result = generate(model, mx, prompt["input_ids"], qa.budget_for(item))
            text = tokenizer.decode(result["generated_ids"], skip_special_tokens=True)
            rows.append({"id": item["id"], "prompt_tokens": len(prompt["input_ids"]), "text": text,
                         "peak_mlx_memory_bytes": mx.get_peak_memory(), **result})
            print(f"qa {len(rows)}: {item['id']} {result['generated_tokens']} tok", flush=True)
            save()
        graded = qa.grade(items[: len(rows)], {r["id"]: r["text"] for r in rows})
        record["qa"].update(grades=graded,
                            failures=sorted(i for i, g in graded.items() if not g["passed"]),
                            normal_stops=all(r["normal_stop"] for r in rows))
        save()
    if "matrix" in args.phases:
        record["matrix"] = {"vocab_size": vocab_size, "prompt_id_hashes": window_ids, "cells": []}
        save("matrix")
        for length in lengths:
            for run in range(WARMUPS + MEASURED):
                result = generate(model, mx, matrix.workload_ids(vocab_size, length, run), OUTPUT_TOKENS)
                cell = {"length": length, "run": run, "warmup": run < WARMUPS, **result}
                record["matrix"]["cells"].append(cell)
                print(f"matrix {length} run {run}{' (warmup)' if cell['warmup'] else ''}: "
                      f"ttft {cell.get('ttft_seconds', 0):.2f}s "
                      f"decode {round(cell.get('decode_tok_s', 0), 2)} tok/s "
                      f"{cell['generated_tokens']} tok", flush=True)
                save()
        record["matrix"]["summary"] = summarize_matrix(record["matrix"]["cells"], lengths)
    record["peak_mlx_memory_bytes"] = mx.get_peak_memory()
    record["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    del model
    mx.clear_cache()
    save("posthash")
    manifest_after = native.validate_inventory(args.model_dir, inventory)
    if manifest_before != manifest_after:
        raise ValueError("pack manifest changed during the reference run")
    if runtime_hashes() != runtime_before:
        raise ValueError("reference runtime files changed during the run")
    record["runtime_files_bound"] = len(runtime_before)
    record["host_idle_after"] = native.cpu_idle_percent()
    record["completed"] = True
    save("complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
