"""Compare the guarded native build with the previously measured bounded reference."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "benchmarks/results"
NATIVE_QA = RESULTS / "flash-next-guarded-reader-qa-m5-20261123"
NATIVE_MATRIX = RESULTS / "flash-next-guarded-reader-matrix-m5-20261123"
BOUNDED = RESULTS / "flash-next-affine8-reference-m5-20261121"
FILES = {
    NATIVE_QA / "evidence.tar.xz": "5c0e94079426f40ab4791550403e1eabf37293295804b63285b7ce57f60d8658",
    NATIVE_MATRIX / "evidence.tar.xz": "2e102e2027fb0e6faf4017da435b73964671af878accd16086316612d9ac2cac",
    BOUNDED / "result.json.gz": "4e2e6d87d099341b7f20515546cc747051656f8cd142504aeeae76477d58bd6e",
    ROOT / "scripts/flash_next_release_thresholds.json": (
        "17a39e0229cf734dca04ffa5abb4718a6c7b1c9c773891385a37396a1625345b"
    ),
}
NATIVE_SHA = "c1a7c3f9dfb81613fc370758a0eece5f6105683d53baab7ea4bc3a371cb81de2"


def verified_receipt(directory):
    outcome = runpy.run_path(str(directory / "replay.py"))["replay"](
        ROOT, directory / "evidence.tar.xz"
    )
    assert outcome == json.loads((directory / "comparison.json").read_bytes())
    assert outcome["server_sha256"] == NATIVE_SHA and outcome["timing_eligible"] is True
    return outcome


def replay():
    for path, expected in FILES.items():
        assert sha256(path.read_bytes()).hexdigest() == expected, path.name
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_flash_next_release_gates as gates

    qa_replay = verified_receipt(NATIVE_QA)
    matrix_replay = verified_receipt(NATIVE_MATRIX)
    parity = runpy.run_path(str(BOUNDED / "compare_outputs.py"))["replay"]()
    assert parity == json.loads((BOUNDED / "comparison.json").read_bytes())
    assert parity["matched_qa_rows_and_grades"] == 105
    assert parity["matched_fixed_matrix_cells"] == 15
    reference_bytes = gzip.decompress((BOUNDED / "result.json.gz").read_bytes())
    reference = json.loads(reference_bytes)
    assert sha256(reference_bytes).hexdigest() == parity["full"]["result_sha256"]
    thresholds = json.loads(gates.THRESHOLDS.read_bytes())

    with tarfile.open(NATIVE_QA / "evidence.tar.xz") as archive:
        qa = json.load(archive.extractfile("result.json"))
        raw = {
            mode: [json.load(archive.extractfile(name))
                   for name in sorted(archive.getnames()) if name.startswith(mode + "/")]
            for mode in gates.AX_ARMS
        }
    with tarfile.open(NATIVE_MATRIX / "evidence.tar.xz") as archive:
        matrix = json.load(archive.extractfile("result.json"))

    assert qa["hardware"] == matrix["hardware"]
    for name, value in reference["hardware"].items():
        assert qa["hardware"][name] == value
    assert qa["hardware"]["storage"]["internal"]
    assert qa["inventory_sha256"] == matrix["inventory_sha256"] == reference["inventory_sha256"]
    assert qa["revision"] == matrix["revision"] == reference["pack_revision"]
    assert qa["server_sha256"] == matrix["server_sha256"] == NATIVE_SHA
    assert qa["ordinary_launch"] is matrix["ordinary_launch"] is True
    assert reference["mode"] == "bounded" and reference["qualification"] is False
    idle = [reference["host_quiet"]["cpu_idle_percent"], reference["host_idle_after"]]
    assert all(thresholds["timing_host"]["min_cpu_idle_percent"] <= value <= 100 for value in idle)

    quality, = gates.gate_quality(qa, reference, thresholds, raw)
    relative, = gates.gate_reference_relative(matrix, reference, thresholds)
    return {
        "scope": "Guarded native M5 versus bounded PLE diagnostic on the same OS",
        "qualification": False,
        "release_ready": False,
        "native_server_sha256": NATIVE_SHA,
        "pack_revision": qa["revision"],
        "inventory_sha256": qa["inventory_sha256"],
        "native_raw_qa_sha256": qa_replay["qa_receipt_sha256"],
        "native_raw_matrix_sha256": matrix_replay["matrix_sha256"],
        "bounded_result_sha256": parity["full"]["result_sha256"],
        "signed_ordinary_reference_sha256": parity["signed_reference_sha256"],
        "bounded_ordinary_parity": {
            "qa": parity["matched_qa_rows_and_grades"],
            "matrix": parity["matched_fixed_matrix_cells"],
        },
        "timing_cpu_idle_percent": {
            "qa": qa_replay["timing_cpu_idle_percent"],
            "matrix": matrix_replay["timing_cpu_idle_percent"],
            "bounded": idle,
        },
        "quality": {key: quality[key] for key in (
            "passed", "ax_failures", "reference_failures", "ax_only_failures"
        )},
        "performance": {"numeric_bounds_met": relative["passed"], "cells": relative["cells"]},
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
