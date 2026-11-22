"""Replay the bounded reference and passive CPU phase attribution."""
from collections import Counter, defaultdict
from gzip import decompress
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

from observe_bounded_cpu import CPU_STATES, tick_delta

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
NATIVE = ROOT.parent / "flash-next-current-matrix-m5-20261122" / "result.json.gz"
SIGNED = (ROOT.parent / "qualification" / "2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0"
          / "reference" / "result.json")
FROZEN_RUNNER = (ROOT.parent / "flash-next-affine8-reference-refresh-m5-20261121"
                 / "run_flash_next_affine8_reference.py.gz")
FROZEN_QA = "benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-qa"
sys.path.insert(0, str(REPO / "scripts"))
import check_flash_next_release_gates as gate  # noqa: E402

QA_FIELDS = ("id", "prompt_tokens", "generated_ids", "text", "normal_stop",
             "terminal_eos_id", "generated_tokens")
CELL_FIELDS = ("length", "run", "warmup", "generated_ids", "normal_stop",
               "terminal_eos_id", "generated_tokens")
PHASES = ("prelaunch", "prehash", "loading", "qa", "matrix", "posthash", "complete")


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def read(name):
    blob = (ROOT / name).read_bytes()
    return decompress(blob) if name.endswith(".gz") else blob


def identical(actual, frozen, fields, count):
    require(len(actual) == len(frozen) == count, "Incomplete reference observation")
    for index, (row, expected) in enumerate(zip(actual, frozen, strict=True)):
        require(all(row[field] == expected[field] for field in fields),
                f"Pinned reference changed at row {index}")


def cpu_intervals(rows):
    groups = defaultdict(list)
    prior = None
    for row in rows:
        current = row["phase"]
        previous = row["previous_phase"]
        require(current in PHASES and previous in PHASES
                and PHASES.index(current) >= PHASES.index(previous),
                "Observed reference phase moved backwards")
        if prior:
            require(previous == prior["phase"]
                    and math.isclose(row["monotonic_seconds"] - prior["monotonic_seconds"],
                                     row["interval_seconds"], abs_tol=1e-9)
                    and tick_delta(prior["ticks"], row["ticks"]) == row["deltas"],
                    "CPU tick or wall clock chain is inconsistent")
        deltas = row["deltas"]
        require(deltas.keys() == set(CPU_STATES)
                and all(type(delta) is int and 0 <= delta < 2**32 for delta in deltas.values())
                and math.isfinite(row["interval_seconds"]) and row["interval_seconds"] > 0,
                "Malformed CPU interval")
        require(sum(deltas.values()) > 0 and math.isclose(
            row["host_idle_percent"], 100 * deltas["idle"] / sum(deltas.values()), abs_tol=1e-9
        ), "Idle percentage disagrees with the tick delta")
        if current == previous:
            groups[current].append(row)
        prior = row
    require(rows and rows[0]["previous_phase"] == "prelaunch" and rows[-1]["phase"] == "complete",
            "CPU observer does not cover the full bounded run")
    summary = {}
    for phase, group in groups.items():
        idle_ticks = sum(row["deltas"]["idle"] for row in group)
        total_ticks = sum(sum(row["deltas"].values()) for row in group)
        summary[phase] = {"intervals": len(group), "seconds": sum(row["interval_seconds"] for row in group),
                          "weighted_idle_percent": 100 * idle_ticks / total_ticks,
                          "min_idle_percent": min(row["host_idle_percent"] for row in group),
                          "max_idle_percent": max(row["host_idle_percent"] for row in group)}
    return summary


def replay():
    transport = json.loads(read("transport.json"))
    for path, expected in transport.items():
        require(sha256(read(path)).hexdigest() == expected, f"Captured bytes changed for {path}")
    require(read("phase-status.txt") == b"bounded-reference-with-cpu-attribution 0\n",
            "Bounded reference did not exit successfully")
    source = json.loads(read("source.json"))
    raw_native = decompress(NATIVE.read_bytes())
    raw_signed = SIGNED.read_bytes()
    require(sha256(raw_native).hexdigest() == source["ax-matrix.json"]
            and sha256(raw_signed).hexdigest() == source["signed-m5-reference.json"],
            "Reference or native input digest changed")
    archived = {"scripts/run_flash_next_affine8_reference.py": decompress(FROZEN_RUNNER.read_bytes()),
                **{f"scripts/{name}": read(f"{name}.gz") for name in
                   ("qualify_qwen38_flash_next.py", "watch_flash_next_reference.py")}}
    for path, expected in source.items():
        if path.startswith(("scripts/", "benchmarks/")):
            original = archived[path] if path in archived else (REPO / path).read_bytes()
            require(sha256(original).hexdigest() == expected, f"Frozen source changed for {path}")
    bounded, native, signed = map(json.loads, (read("bounded.json.gz"), raw_native, raw_signed))
    build = json.loads(read("matrix-build.json"))
    require(native["server_sha256"] == build["binaries_sha256"]["ax-engine-server"]
            and native["repeats_per_arm"] == 2,
            "Bounded comparison is not tied to the current production matrix")
    require(bounded["completed"] is True and bounded["phase"] == "complete"
            and bounded["mode"] == "bounded" and bounded["backend"] == "metal"
            and bounded["reference"]["runtime"] == "mlx-vlm-affine8-ple-diagnostic"
            and bounded["qualification"] is False and bounded["release_ready"] is False,
            "Unexpected reference type or status")
    require(bounded["workload_hashes"]["ax_matrix_sha256"] == source["ax-matrix.json"]
            and bounded["pack_revision"] == native["revision"] == signed["revision"]
            and bounded["inventory_sha256"] == native["inventory_sha256"] == signed["inventory_sha256"]
            and all(bounded["hardware"][key] == native["hardware"][key]
                    for key in ("soc", "memory_bytes", "os")),
            "Reference and native matrix identity differ")
    for path, field in (("full-qa-items.json", "items_sha256"),
                        ("full-qa-inputs.json", "inputs_sha256"),
                        *((f"{name}.py", f"{name}_sha256") for name in
                          ("closed_checks", "qa_checks", "checkers", "prompt_def"))):
        require(bounded["workload_hashes"][field] == source[f"{FROZEN_QA}/{path}"],
                f"Frozen QA input changed for {path}")
    require(bounded["reference"]["commit"] == signed["reference"]["commit"]
            and all(bounded[key] == signed[key] for key in (
                "versions", "language_sha256", "runtime_files_bound", "prefill_chunk")),
            "Reference runtime no longer matches the pinned identity")
    qa, matrix = bounded["qa"], bounded["matrix"]
    identical(qa["rows"], signed["qa"]["rows"], QA_FIELDS, 105)
    identical(matrix["cells"], signed["matrix"]["cells"], CELL_FIELDS, 15)
    require(qa["grades"] == signed["qa"]["grades"] and qa["failures"] == signed["qa"]["failures"]
            and matrix["prompt_id_hashes"] == signed["matrix"]["prompt_id_hashes"],
            "Frozen quality grades or fixed prompts changed")
    thresholds = json.loads(gate.THRESHOLDS.read_bytes())
    numeric = gate.gate_reference_relative(native, bounded, thresholds)[0]
    idle = [bounded["host_quiet"]["cpu_idle_percent"], bounded["host_idle_after"]]
    floor = thresholds["timing_host"]["min_cpu_idle_percent"]
    pressure = Counter(json.loads(line)["pressure"] for line in read("memory-samples.jsonl.gz").splitlines())
    require(pressure and len(numeric["cells"]) == 6, "Missing pressure or numeric observations")
    intervals = cpu_intervals([json.loads(line) for line in read("cpu-phases.jsonl.gz").splitlines()])
    return {"scope": "Independent bounded PLE diagnostic with passive CPU observation",
            "qualification": False, "release_ready": False,
            "native_server_sha256": native["server_sha256"],
            "native_matrix_sha256": source["ax-matrix.json"],
            "matched_qa_rows": len(qa["rows"]), "matched_matrix_rows": len(matrix["cells"]),
            "normal_pressure_samples": pressure.get(1, 0), "all_pressure_samples_normal": set(pressure) == {1},
            "peak_mlx_memory_bytes": bounded["peak_mlx_memory_bytes"],
            "reference_timing_idle_percent": idle,
            "timing_floor_met": all(floor <= value <= 100 for value in idle),
            "numeric_bounds_met": numeric["passed"], "cells": numeric["cells"],
            "single_phase_cpu_intervals": intervals}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
