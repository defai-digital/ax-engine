"""Evaluate the complete current Flash Next native campaign against the frozen board."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
SCRIPTS = ROOT.parents[2] / "scripts"
DELIVERY = RESULTS / "flash-next-fresh-delivery-38c90b27a-m5-20261122" / "evidence.tar.xz"
QA = RESULTS / "flash-next-native-qa-38c90b27a-m5-20261122" / "evidence.tar.xz"
LIFECYCLE = RESULTS / "flash-next-native-lifecycle-38c90b27a-m5-20261122" / "lifecycle.json"
SIGNED = RESULTS / "flash-next-native-resident-refresh-m5-20261122" / "evidence.tar.xz"
REFERENCE = RESULTS / "flash-next-ordinary-reference-m5-20261121" / "partial-result.json"
sys.path.insert(0, str(SCRIPTS))
import check_flash_next_release_gates as gates
import qualify_qwen38_flash_next as qualify


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def member(archive, name):
    row = archive.getmember(name)
    require(row.isfile(), f"Expected a raw file: {name}")
    with archive.extractfile(row) as handle:
        return handle.read()


def signature(run):
    payload = {
        "cells": [[case["length"], case["run"], case["warmup"], case["output_tokens"]] for case in run["cells"]],
        "trajectories": [[case["index"], case["output_tokens"]] for case in run["trajectories"]],
        "prompt_id_hashes": run["prompt_id_hashes"],
    }
    require(len(payload["cells"]) == 15 and len(payload["trajectories"]) == 16, "Incomplete matrix run")
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def replay(evidence=ROOT / "evidence.tar.xz", reference=REFERENCE):
    raw = {"lifecycle": LIFECYCLE.read_bytes(), "reference": reference.read_bytes()}
    with tarfile.open(DELIVERY) as original:
        raw["delivery"] = member(original, "delivery.json")
        delivered = member(original, "native-support.json")
        build = json.loads(member(original, "build.json"))
        inputs = {name: digest for digest, name in (line.split() for line in
                  member(original, "inputs.sha256").decode().splitlines())}
    with tarfile.open(QA) as archive:
        raw["qa"] = member(archive, "result.json")
        items = gates.qa_cohort.load_items(gates.QA_ITEMS)
        qa_rows = {mode: [json.loads(member(archive, f'{mode}/{i:03d}-{item["id"]}.json'))
                          for i, item in enumerate(items)] for mode in gates.AX_ARMS}
    with tarfile.open(evidence) as archive:
        raw["matrix"] = member(archive, "observed/generate-matrix/result.json")
        repeats_raw = member(archive, "observed/generate-matrix/raw-runs.json")
        repeat = json.loads(repeats_raw)
        support = member(archive, "observed/native-support/result.json")
        terminal = json.loads(member(archive, "observed/comparison.json"))
        rejection = json.loads(member(archive, "observed/explicit-skip.json"))
        require(member(archive, "status.txt") == b"0\n", "The primary native owner did not finish")
    results = {name: json.loads(data) for name, data in raw.items()}
    require(terminal["build"] == build and terminal["qualification"] is False
            and terminal["observed_sha256"] == {name: sha256(content).hexdigest() for name, content in
                    (("native-support", support), ("qa", raw["qa"]), ("lifecycle", raw["lifecycle"]),
                     ("generate-matrix", raw["matrix"]))}, "Current native receipts did not bind the delivery build")
    require(rejection["completed"] is True and rejection["code"] == 1 and rejection["payload_markers"]
            and set(rejection["payload_markers"].values()) == {0}, "Explicit Required sidecar conflict entered payload startup")
    require(all(row["server_sha256"] == build["server_sha256"] and row["inventory_sha256"] == build["inventory_sha256"]
                for row in (results[key] for key in ("delivery", "qa", "lifecycle", "matrix"))),
            "A current native result used a different binary or pack")
    with tarfile.open(SIGNED) as archive:
        oracle_raw = member(archive, "driver/signed-native.json")
    require(sha256(oracle_raw).hexdigest() == terminal["reference_control_sha256"] == inputs["signed-native.json"],
            "Signed output oracle changed")
    oracle = json.loads(oracle_raw)["matrix"]
    require(repeat.keys() == oracle.keys() and all(len(repeat[mode]) == 2 and
                all(signature(run) == oracle[mode] for run in repeat[mode]) for mode in gates.AX_ARMS),
            "At least one native process differs from the signed token trajectories")
    matrix = results["matrix"]
    require(not gates.matrix_repeat_problems(matrix, repeat)
            and terminal["matched_independent_matrix_arrays"] == 124,
            "Native matrix repeat identity, selection or count changed")
    thresholds = json.loads(gates.THRESHOLDS.read_bytes())
    board = gates.evaluate(results, thresholds, qualify.contract(), repeat, delivered, qa_rows)
    return {**board, "native_build_commit": build["commit"], "native_server_sha256": build["server_sha256"],
            "matching_signed_matrix_arrays": 124, "repeat_token_identity": matrix["repeat_token_identity"],
            "native_matrix_cpu_idle_percent": [matrix["host_quiet"]["cpu_idle_percent"], matrix["host_idle_after"]],
            "matrix_readiness_seconds": {mode: matrix["runs"][mode]["ready_seconds"] for mode in gates.AX_ARMS},
            "timing_eligible_native_legs": terminal["timing_eligible_legs"],
            "raw_sha256": {**{key: sha256(data).hexdigest() for key, data in raw.items()},
                           "matrix_repeats": sha256(repeats_raw).hexdigest(), "delivered_support": sha256(delivered).hexdigest()}}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
