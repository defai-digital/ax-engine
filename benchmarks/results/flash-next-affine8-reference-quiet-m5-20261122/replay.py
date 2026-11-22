"""Check the quiet repeat against the signed M5 bounded PLE controls."""
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-affine8-reference-refresh-m5-20261121"
INPUTS_SHA = "ef2cdc95bbeaf6f7533b0603a18a6551e118d3cfbb4687089bf714be28b2a005"
sys.path.insert(0, str(PRIOR))
import compare_outputs as previous  # noqa: E402


def replay(root=ROOT):
    assert previous.replay() == json.loads((PRIOR / "comparison.json").read_bytes())
    import check_flash_next_release_gates as evaluator

    raw_bounded = gzip.decompress((root / "bounded-result.json.gz").read_bytes())
    samples = gzip.decompress((root / "memory-samples.jsonl.gz").read_bytes())
    raw_native = gzip.decompress((PRIOR / "native-matrix.json.gz").read_bytes())
    original = json.loads(gzip.decompress((PRIOR / "bounded-result.json.gz").read_bytes()))
    bounded, native = map(json.loads, (raw_bounded, raw_native))
    sources = (root / "source.json").read_bytes()
    launch = (root / "inputs.sha256").read_bytes()
    assert sha256(launch).hexdigest() == INPUTS_SHA
    input_hashes = {name: digest for digest, name in (line.split() for line in launch.decode().splitlines())}
    assert sha256(sources).hexdigest() == input_hashes["source.json"]
    assert sources == (PRIOR / "source.json").read_bytes()
    assert (root / "phase-status.txt").read_bytes() == b"bounded-quiet-repeat 0\n"
    for field in ("schema", "mode", "pack_revision", "backend", "reference", "versions",
                  "language_sha256", "runtime_files_bound", "prefill_chunk", "inventory_sha256",
                  "hardware", "workload_hashes"):
        assert bounded[field] == original[field], field
    assert bounded["completed"] is True and bounded["phase"] == "complete"
    assert bounded["qualification"] is False and bounded["release_ready"] is False
    qa, matrix = bounded["qa"], bounded["matrix"]
    previous.identical_rows(qa["rows"], original["qa"]["rows"], previous.QA_FIELDS, 105)
    assert qa["grades"] == original["qa"]["grades"] and qa["failures"] == original["qa"]["failures"]
    previous.identical_rows(matrix["cells"], original["matrix"]["cells"], previous.MATRIX_FIELDS, 15)
    assert matrix["prompt_id_hashes"] == original["matrix"]["prompt_id_hashes"]
    assert all(len(cell["generated_ids"]) == 128 for cell in matrix["cells"])
    limits = json.loads(evaluator.THRESHOLDS.read_bytes())
    floor = limits["timing_host"]["min_cpu_idle_percent"]
    idle = (bounded["host_quiet"]["cpu_idle_percent"], bounded["host_idle_after"])
    idle_before_load = json.loads((root / "preflight.json").read_bytes())["cpu_idle_percent"]
    assert floor <= idle_before_load <= 100 and all(0 <= value <= 100 for value in idle)
    pressure = Counter(json.loads(line)["pressure"] for line in samples.splitlines())
    assert pressure
    numeric = evaluator.gate_reference_relative(native, bounded, limits)[0]
    assert len(numeric["cells"]) == 6
    return {
        "scope": "Independent bounded PLE diagnostic",
        "qualification": False, "release_ready": False,
        "bounded_result_sha256": sha256(raw_bounded).hexdigest(),
        "resource_samples_sha256": sha256(samples).hexdigest(),
        "native_matrix_sha256": sha256(raw_native).hexdigest(),
        "matched_qa_rows_and_grades": len(qa["rows"]),
        "matched_fixed_matrix_cells": len(matrix["cells"]),
        "peak_mlx_memory_bytes": bounded["peak_mlx_memory_bytes"],
        "peak_process_rss_bytes": bounded["peak_process_rss_bytes"],
        "pressure_samples": dict(sorted(pressure.items())),
        "all_sampled_pressure_normal": set(pressure) == {1},
        "reference_cpu_idle_before_percent": idle[0],
        "reference_cpu_idle_after_percent": idle[1],
        "timing_host_floor_met": all(floor <= value for value in idle),
        "numeric_bounds_met": numeric["passed"],
        "cells": numeric["cells"],
    }


if __name__ == "__main__":
    result = json.dumps(replay(), indent=2, sort_keys=True) + "\n"
    if result != (ROOT / "comparison.json").read_text():
        raise ValueError("Replayed result differs from the published comparison")
    print(result, end="")
