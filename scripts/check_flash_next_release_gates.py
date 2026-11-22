#!/usr/bin/env python3
"""Evaluate the Flash Next direct-default release gates from recorded evidence.

Reads the delivery, QA, lifecycle, generate-matrix and reference results,
the raw delivery controls, QA rows and matrix repeats, and the frozen thresholds in
`flash_next_release_thresholds.json`. Reports one pass/fail line per gate.
Nothing here runs a model. The verdict covers only the stated release scope:
direct decoding is the default, model MTP stays an explicit experimental
opt-in, and MTP-S, MTP-P and MTP-D remain `not_assessed`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any

try:
    import run_flash_next_qa_cohort as qa_cohort
    import run_flash_next_generate_matrix as matrix_runner
except ImportError:
    from scripts import run_flash_next_qa_cohort as qa_cohort
    from scripts import run_flash_next_generate_matrix as matrix_runner

SCRIPTS = Path(__file__).resolve().parent
THRESHOLDS = SCRIPTS / "flash_next_release_thresholds.json"
GIB = 1024**3
# Flash Next qualification and certification standard (ADR-049); mirrors
# check_flash_next_native_support.TARGET_*, which the live harnesses enforce.
TARGET_FORM_FACTOR = "MacBook Pro"
TARGET_SOC = "Apple M5 Max"
TARGET_MEMORY_GIB = 128
TARGET_DESCRIPTION = "MacBook Pro, Apple M5 Max, 128 GB"
AX_ARMS = ("disabled", "default")
QA_ITEMS = qa_cohort.FROZEN / "full-qa-items.json"
# The release gate checks the frozen workload independently of the runners.
OUTPUT_TOKENS = 128
LENGTHS = (512, 2048, 8192)
WARMUPS = 2
MEASURED = 3
TRAJECTORIES = 16
AX_REPEATS = 2
LONG_ID = "long_context_record_mass"
REFERENCE_VERSIONS = {"mlx": "0.32.2", "mlx-metal": "0.32.2", "mlx-vlm": "0.7.0rc0",
                      "transformers": "5.18.0.dev0", "tokenizers": "0.23.2"}
REFERENCE_LANGUAGE_SHA256 = "30f059d6289451bde0f5ef816090ff946ebc857d1b557f83415e2ba867d76199"
REFERENCE_COMMIT = "6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb"
REFERENCE_EXECUTION_SETTINGS = {"prefill_chunk": 1024, "memory_limit_bytes": 144 * GIB,
                                "cache_limit_bytes": GIB}
LIFECYCLE_ACTIONS = ("baseline", "sse", "budget_1", "budget_2", "stop", "disconnect", "recovery")
LIFECYCLE_COUNTERS = frozenset({
    "ax_engine_jobs_in_flight",
    "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued",
    "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
})
RESULT_FILES = {"delivery": "delivery/result.json", "qa": "qa/result.json",
                "lifecycle": "lifecycle/result.json", "matrix": "generate-matrix/result.json",
                "reference": "reference/result.json"}
MATRIX_REPEATS_FILE = "generate-matrix/raw-runs.json"
DELIVERY_CONTROLS_FILE = "delivery/native-support/result.json"


def gate(name: str, passed: bool, detail: str, **data: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "detail": detail, **data}


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def positive_measurement(value: Any) -> bool:
    return type(value) in (int, float) and 0 < value <= sys.float_info.max


def delivered_support(delivery: dict[str, Any]) -> dict[str, Any]:
    support = delivery.get("native_support")
    return support if isinstance(support, dict) else {}


def delivered_modes(delivery: dict[str, Any]) -> list[str]:
    modes = delivered_support(delivery).get("modes")
    return modes if isinstance(modes, list) and all(isinstance(mode, str) for mode in modes) else []


def delivery_control_problems(delivery: dict[str, Any], raw: bytes | None) -> list[str]:
    if raw is None:
        return ["delivery: missing raw native controls"]
    try:
        controls = json.loads(raw, object_pairs_hook=unique_json_members)
    except (ValueError, RecursionError):
        return ["delivery: invalid raw native controls"]
    if not isinstance(controls, dict):
        return ["delivery: invalid raw native controls"]

    support = delivered_support(delivery)
    fields = {"passed": "functional_controls_passed", "modes": "modes",
              "readiness_seconds": "readiness_seconds", "page_cache": "page_cache"}
    projected = {key: controls.get(source) for key, source in fields.items()}
    if (type(projected["passed"]) is not bool or not isinstance(projected["modes"], list)
            or not isinstance(projected["readiness_seconds"], dict)
            or not isinstance(projected["page_cache"], str)):
        return ["delivery: invalid raw native control summary"]
    problems = []
    if support.get("result_sha256") != hashlib.sha256(raw).hexdigest():
        problems.append("delivery: raw native control SHA-256 differs from the delivery receipt")
    recorded = {key: support.get(key) for key in fields}
    if json.dumps(projected, sort_keys=True) != json.dumps(recorded, sort_keys=True):
        problems.append("delivery: raw native controls differ from the delivery summary")
    if any(controls.get(key) != delivery.get(key) for key in ("server_sha256", "inventory_sha256", "revision")):
        problems.append("delivery: raw native controls differ in server, inventory or revision")
    return problems


def gate_delivery(delivery: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    spec, target = th["delivery"], th["target"]
    verification = delivery.get("verification") or {}
    support = delivered_support(delivery)
    problems = []
    if not delivery.get("completed"):
        problems.append("delivery did not complete")
    if not delivery.get("fresh_cache"):
        problems.append("the first attempt did not start from an empty cache")
    if (delivery.get("download") or {}).get("exit_code") != 0:
        problems.append("downloader did not exit 0")
    if delivery.get("revision") != target["pack_revision"]:
        problems.append("revision is not the pinned one")
    if verification.get("members_verified") != target["published_members"] \
            or verification.get("bytes_verified") != target["published_bytes"] \
            or not verification.get("all_sizes_and_sha256_match"):
        problems.append("members, bytes or hashes do not match the published pack")
    if not verification.get("native_manifest_runtime_ready"):
        problems.append("generated native manifest is not runtime ready")
    if delivered_modes(delivery) != spec["functional_controls_on_delivered_copy"] \
            or support.get("passed") is not True:
        problems.append("functional controls did not pass in the required delivery order")
    attempts = delivery.get("previous_attempts") or []
    resumed = f" after {len(attempts)} failed attempt(s) that were resumed" if attempts else ""
    if delivery.get("transport_override"):
        resumed += f" (transport override {delivery['transport_override']})"
    return [gate("delivery", not problems, "; ".join(problems) or
                 f"{verification.get('members_verified')} members, "
                 f"{verification.get('bytes_verified')} bytes verified from an empty cache{resumed}",
                 download_seconds=(delivery.get("download") or {}).get("elapsed_seconds"),
                 failed_attempts=len(attempts))]


def gate_readiness(delivery: dict[str, Any], runs: list[tuple[str, Any]],
                   th: dict[str, Any]) -> list[dict[str, Any]]:
    spec = th["readiness"]
    support = delivered_support(delivery)
    recorded = support.get("readiness_seconds")
    delivered = {mode: seconds if positive_measurement(seconds) else None
                 for mode, seconds in recorded.items()} if isinstance(recorded, dict) else {}
    modes = delivered_modes(delivery)
    required = th["delivery"]["functional_controls_on_delivered_copy"]
    first_mode = required[0]
    problems = []
    if modes != required:
        problems.append("recorded modes do not match the required delivery order")
    if delivered.keys() != set(required):
        problems.append("readiness must be recorded for all delivered modes, including the first load")
    for mode, seconds in delivered.items():
        limit = spec["first_load_after_delivery_seconds_max"] if mode == first_mode \
            else spec["warm_seconds_max"]
        if seconds is None:
            problems.append(f"delivered {mode} readiness is missing or invalid")
        elif seconds > limit:
            problems.append(f"delivered {mode} readiness {seconds}s exceeds {limit}s")
    valid = []
    for label, seconds in runs:
        if not positive_measurement(seconds):
            problems.append(f"{label} readiness is missing or invalid")
        else:
            valid.append(seconds)
            if seconds > spec["warm_seconds_max"]:
                problems.append(f"{label} readiness {seconds}s exceeds {spec['warm_seconds_max']}s")
    worst = max(valid, default=0)
    return [gate("readiness", not problems, "; ".join(problems) or
                 f"first load after delivery {delivered.get(first_mode)}s; "
                 f"staged-pack worst {worst}s", delivered=delivered)]


def gate_memory(qa: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    limit = th["memory"]["peak_server_rss_gib_max"]
    rss = {mode: qa["summary"]["modes"][mode].get("peak_rss_kib") for mode in AX_ARMS}
    peaks = {mode: kib / (GIB // 1024) if positive_measurement(kib) else None
             for mode, kib in rss.items()}
    invalid = [mode for mode, peak in peaks.items() if peak is None]
    over = {mode: round(peak, 2) for mode, peak in peaks.items() if peak is not None and peak > limit}
    problems = []
    if invalid:
        problems.append(f"missing or invalid peak server RSS for {invalid}")
    if over:
        problems.append(f"peak server RSS {over} exceeds {limit} GiB")
    return [gate("memory", not problems, "; ".join(problems) or
                 f"peak server RSS {{{', '.join(f'{m}: {p:.1f}' for m, p in peaks.items())}}} GiB "
                 f"within {limit} GiB",
                 peaks_gib={mode: round(peak, 3) if peak is not None else None for mode, peak in peaks.items()})]


def gate_long_lookup(qa: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    limit = th["long_lookup"]["wall_seconds_max"]
    recorded = {mode: qa["summary"]["modes"][mode].get("long_context_elapsed_seconds")
                for mode in AX_ARMS}
    seconds = {mode: value if positive_measurement(value) else None for mode, value in recorded.items()}
    bad = {mode: s for mode, s in seconds.items() if s is None or s > limit}
    return [gate("long_lookup", not bad, f"{bad} outside {limit}s" if bad else
                 f"{th['long_lookup']['prompt_tokens']}-token lookup took {seconds} s (limit {limit} s)",
                 seconds=seconds)]


def cohort_failures(grades: Any, expected_ids: set[str], reported: Any) -> set[str] | None:
    if (not isinstance(grades, dict) or grades.keys() != expected_ids
            or any(not isinstance(grade, dict) or type(grade.get("passed")) is not bool
                   for grade in grades.values())):
        return None
    failures = {item_id for item_id, grade in grades.items() if not grade["passed"]}
    if (not isinstance(reported, list) or len(reported) != len(failures)
            or not all(isinstance(item_id, str) for item_id in reported)
            or set(reported) != failures):
        return None
    return failures


def native_qa_problems(qa: dict[str, Any], items: list[dict[str, Any]],
                       raw_rows: dict[str, list[Any]]) -> list[str]:
    problems = []
    texts = {}
    recorded_grades = qa.get("grades") or {}
    for mode in AX_ARMS:
        rows = raw_rows.get(mode, [])
        if len(rows) != len(items) or any(
            not isinstance(row, dict)
            or type(row.get("index")) is not int or row["index"] != index
            or row.get("id") != item["id"]
            or type(row.get("budget")) is not int or row["budget"] != qa_cohort.budget_for(item)
            or not isinstance(row.get("text"), str)
            or row.get("finish_reason") != "stop"
            for index, (row, item) in enumerate(zip(rows, items))
        ):
            problems.append(f"{mode}: raw QA rows must match the frozen items and stop normally")
            continue
        texts[mode] = {row["id"]: row["text"] for row in rows}
        grades = qa_cohort.grade(items, texts[mode])
        if not isinstance(recorded_grades, dict) or grades != recorded_grades.get(mode):
            problems.append(f"{mode}: recorded QA grades disagree with the raw row replay")
    if texts.keys() == set(AX_ARMS) and texts["default"] != texts["disabled"]:
        problems.append("raw QA text differs between default and explicit direct")
    return problems


def gate_quality(qa: dict[str, Any], reference: dict[str, Any], th: dict[str, Any],
                 raw_rows: dict[str, list[Any]]) -> list[dict[str, Any]]:
    items = qa_cohort.load_items(QA_ITEMS)
    expected_ids = {item["id"] for item in items}
    modes = qa["summary"]["modes"]
    ref_qa = reference.get("qa")
    if not isinstance(ref_qa, dict):
        ref_qa = {}
    problems = native_qa_problems(qa, items, raw_rows)
    rows = ref_qa.get("rows")
    ref_failures = None
    if reference.get("completed") is not True or not isinstance(rows, list) or len(rows) != len(items):
        problems.append("reference QA is incomplete")
    else:
        inputs = json.loads((qa_cohort.FROZEN / "full-qa-inputs.json").read_text())["prepared"]
        if (len(inputs) != len(items)
                or any(not isinstance(row, dict) or row.get("id") != item["id"]
                       or prompt["id"] != item["id"]
                       or type(row.get("prompt_tokens")) is not int
                       or row["prompt_tokens"] != len(prompt["input_ids"])
                       or not isinstance(row.get("text"), str)
                       for row, item, prompt in zip(rows, items, inputs))):
            problems.append("reference QA rows do not match the frozen prompts in order")
        else:
            grades = qa_cohort.grade(items, {row["id"]: row["text"] for row in rows})
            recorded_grades = ref_qa.get("grades")
            ref_failures = cohort_failures(recorded_grades, expected_ids, ref_qa.get("failures"))
            if grades != recorded_grades or ref_failures is None:
                problems.append("reference QA grades or failures disagree with the frozen checker replay")
                ref_failures = None
    if qa.get("items") != len(items) or qa.get("items_sha256") != sha256_file(QA_ITEMS):
        problems.append("AX QA does not bind the frozen cohort")
    ax_grades = qa.get("grades")
    if not isinstance(ax_grades, dict):
        ax_grades = {}
    ax_failures: dict[str, list[str]] = {}
    ax_only: dict[str, list[str] | None] = {}
    for mode in AX_ARMS:
        summary = modes.get(mode, {})
        if summary.get("completed") != len(items) or not summary.get("normal_stops") or summary.get("route_errors"):
            problems.append(f"{mode}: incomplete cohort, abnormal stop or route error")
        reported = summary.get("quality_failures")
        failures = cohort_failures(ax_grades.get(mode), expected_ids, reported)
        ax_failures[mode] = reported if isinstance(reported, list) and all(
            isinstance(item_id, str) for item_id in reported
        ) else []
        if failures is None or summary.get("passed") != len(items) - len(failures):
            problems.append(f"{mode}: missing or inconsistent per-item QA grades")
        extra = sorted(failures - ref_failures) if failures is not None and ref_failures is not None else None
        ax_only[mode] = extra
        if extra is not None and len(extra) > th["quality"]["ax_only_failures_max"]:
            problems.append(f"{mode}: failures the reference passes: {extra}")
    identity = (qa["summary"].get("text_identity_vs_disabled") or {}).get("default") or {}
    if identity.get("matching_text_pairs") != len(items) or identity.get("differing_text_ids") != []:
        problems.append("default and direct do not match on all frozen QA responses")
    return [gate("quality", not problems, "; ".join(problems) or
                 f"{len(items)}/{len(items)} complete in both arms, default text equals direct text, "
                 f"every AX failure also fails on the reference ({sorted(ref_failures or [])})",
                 ax_failures=ax_failures,
                 reference_failures=sorted(ref_failures) if ref_failures is not None else None,
                 ax_only_failures=ax_only)]


def gate_lifecycle(lifecycle: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    spec = th["lifecycle"]
    problems = []
    for mode in spec["modes_required"]:
        runs = [run for run in lifecycle.get("runs", []) if run["mode"] == mode]
        if len(runs) != 1:
            problems.append(f"{mode}: expected one lifecycle run")
            continue
        (run,) = runs
        actions = run.get("actions")
        if (not isinstance(actions, list) or len(actions) != spec["actions_per_mode"]
                or not all(isinstance(action, dict) for action in actions)
                or tuple(action.get("action") for action in actions) != LIFECYCLE_ACTIONS
                or any(action.get("mode") != mode for action in actions)):
            problems.append(f"{mode}: missing, repeated, misordered or mislabeled actions")
            continue
        for action in actions:
            counters = action.get("drained")
            if (not isinstance(counters, dict) or not LIFECYCLE_COUNTERS.issubset(counters)
                    or any(type(value) not in (int, float) or value != 0 for value in counters.values())):
                problems.append(f"{mode}/{action['action']}: missing or nonzero drain counters")
        exit_code = run.get("exit_code_after_stop")
        if type(exit_code) is not int or exit_code != 0:
            problems.append(f"{mode}: server did not exit cleanly")
    if lifecycle.get("lifecycle_passed") is not True:
        problems.append("lifecycle result is not marked passed")
    return [gate("lifecycle", not problems, "; ".join(problems) or
                 f"{spec['actions_per_mode']} actions in each of {spec['modes_required']} with drained counters")]


def cell_stats(rows: list[dict[str, Any]], *, streamed: bool) -> dict[str, Any]:
    """Compute medians from complete outputs and their post-first-event windows."""
    measured = [r for r in rows if r.get("warmup") is False]
    rates, ttfts = [], []
    full_length = 0
    for row in measured:
        ids = row.get("output_tokens" if streamed else "generated_ids")
        if (not isinstance(ids, list) or len(ids) != OUTPUT_TOKENS
                or any(type(token) is not int or token < 0 for token in ids)
                or (not streamed and (type(row.get("generated_tokens")) is not int
                                      or row["generated_tokens"] != len(ids)))):
            continue
        full_length += 1
        timing = row.get("timing", row) if streamed else row
        if not isinstance(timing, dict):
            continue
        seconds = timing.get("decode_seconds")
        count = timing.get("decode_tokens")
        first = timing.get("ttft_seconds")
        first_count = timing.get("first_event_tokens") if streamed else 1
        if (type(first_count) is int and 0 < first_count < len(ids)
                and type(count) is int and count == len(ids) - first_count
                and positive_measurement(seconds) and positive_measurement(first)):
            rate = count / seconds
            if positive_measurement(rate):
                rates.append(rate)
                ttfts.append(first)
    complete_timing = len(rates) == len(measured) == MEASURED
    return {"measured": len(measured), "full_length": full_length,
            "decode_tok_s": statistics.median(rates) if complete_timing else None,
            "ttft_s": statistics.median(ttfts) if complete_timing else None}


def gate_reference_relative(matrix: dict[str, Any], reference: dict[str, Any],
                            th: dict[str, Any]) -> list[dict[str, Any]]:
    spec = th["reference_relative"]
    ref_matrix = reference.get("matrix") or {}
    ref_cells = ref_matrix.get("cells") or []
    ax_cells = {arm: matrix["runs"][arm]["cells"] for arm in AX_ARMS}
    expected_runs = {(length, run, run < WARMUPS) for length in LENGTHS
                     for run in range(WARMUPS + MEASURED)}
    expected_hash_keys = {f"cell-{length}-{run}" for length, run, _ in expected_runs}
    problems, table = [], {}
    if reference.get("completed") is not True or not ref_cells:
        problems.append("reference matrix is incomplete")
    for label, rows in {"reference": ref_cells, **ax_cells}.items():
        valid = [(row.get("length"), row.get("run"), row.get("warmup")) for row in rows
                 if type(row.get("length")) is int and type(row.get("run")) is int
                 and isinstance(row.get("warmup"), bool)]
        if len(rows) != len(expected_runs) or set(valid) != expected_runs:
            problems.append(f"{label}: expected two warmups and three measurements at each of {LENGTHS}")
    ref_hashes = ref_matrix.get("prompt_id_hashes") or {}
    reference_hashes_valid = (isinstance(ref_hashes, dict) and ref_hashes.keys() == expected_hash_keys
                              and all(map(is_sha256, ref_hashes.values())))
    if not reference_hashes_valid:
        problems.append("reference: expected a SHA-256 for every fixed-workload prompt")
    for arm in AX_ARMS:
        ax_hashes = {key: value for key, value in matrix["runs"][arm]["prompt_id_hashes"].items()
                     if key.startswith("cell-")}
        if ax_hashes.keys() != expected_hash_keys or not all(map(is_sha256, ax_hashes.values())):
            problems.append(f"{arm}: expected a SHA-256 for every fixed-workload prompt")
        elif reference_hashes_valid and ax_hashes != ref_hashes:
            problems.append(f"{arm}: prompt token IDs differ from the reference's")
    for length in LENGTHS:
        ref = cell_stats([c for c in ref_cells if c.get("length") == length], streamed=False)
        if ref["full_length"] != ref["measured"]:
            problems.append(f"{length}: reference has short or invalid measured outputs")
        for arm in AX_ARMS:
            rows = [c for c in ax_cells[arm] if c.get("length") == length]
            ax = cell_stats(rows, streamed=True)
            if ax["full_length"] != ax["measured"]:
                problems.append(f"{length}/{arm}: AX has short or invalid measured outputs")
            if not (ax["decode_tok_s"] and ref["decode_tok_s"] and ax["ttft_s"] and ref["ttft_s"]):
                problems.append(f"{length}/{arm}: missing timing")
                continue
            decode_ratio = ax["decode_tok_s"] / ref["decode_tok_s"]
            ttft_ratio = ax["ttft_s"] / ref["ttft_s"]
            if not all(map(positive_measurement, (decode_ratio, ttft_ratio))):
                problems.append(f"{length}/{arm}: invalid timing ratios")
                continue
            table[f"{length}/{arm}"] = {"decode_ratio": round(decode_ratio, 4),
                                        "ttft_ratio": round(ttft_ratio, 4),
                                        "ax_decode_tok_s": round(ax["decode_tok_s"], 3),
                                        "reference_decode_tok_s": round(ref["decode_tok_s"], 3),
                                        "ax_ttft_s": round(ax["ttft_s"], 3),
                                        "reference_ttft_s": round(ref["ttft_s"], 3)}
            if decode_ratio < spec["decode_ratio_min"]:
                problems.append(f"{length}/{arm}: decode ratio {decode_ratio:.3f} < {spec['decode_ratio_min']}")
            if ttft_ratio > spec["ttft_ratio_max"]:
                problems.append(f"{length}/{arm}: TTFT ratio {ttft_ratio:.3f} > {spec['ttft_ratio_max']}")
    return [gate("reference_relative", not problems and bool(table), "; ".join(problems) or
                 f"all {len(table)} AX-arm cells within decode >= {spec['decode_ratio_min']} and "
                 f"TTFT <= {spec['ttft_ratio_max']} of the reference", cells=table)]


def gate_mtp_policy(delivery: dict[str, Any], qa: dict[str, Any], lifecycle: dict[str, Any],
                    matrix: dict[str, Any], th: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    problems = []
    if "default" not in delivered_modes(delivery):
        problems.append("default mode missing from the delivered-copy controls")
    verified = {"qa": qa["summary"]["modes"]["default"]["verified_steps"],
                "matrix": sum(t["verified_steps"] for t in matrix["runs"]["default"]["trajectories"])
                + sum(c["verified_steps"] for c in matrix["runs"]["default"]["cells"])}
    for action_run in lifecycle.get("runs", []):
        if action_run["mode"] == "default":
            verified["lifecycle"] = action_run.get("final_verified_steps")
    for where, steps in verified.items():
        if steps != th["mtp"]["default_verified_steps"]:
            problems.append(f"default mode verified {steps} MTP steps in {where}")
    if contract.get("mtp_certification") != th["mtp"]["gate_state"]:
        problems.append("MTP-S/P/D are not all not_assessed in the qualification contract")
    return [gate("mtp_policy", not problems, "; ".join(problems) or
                 "default mode never activated the drafter in any harness; required MTP stays an "
                 "explicit opt-in and MTP-S/P/D remain not_assessed", default_verified_steps=verified)]


def matrix_repeat_problems(matrix: dict[str, Any], repeats: Any) -> list[str]:
    if not isinstance(repeats, dict) or repeats.keys() != set(AX_ARMS):
        return ["matrix: raw repeats must record both direct arms"]
    problems = []
    for mode in AX_ARMS:
        runs = repeats[mode]
        if not isinstance(runs, list) or len(runs) != AX_REPEATS:
            problems.append(f"matrix/{mode}: missing raw process repeats")
            continue
        selected = matrix["runs"][mode]
        scores = [matrix_repeat_score(run, selected, mode) for run in runs]
        if any(score is None for score in scores):
            problems.append(f"matrix/{mode}: raw repeats lack matching complete workloads")
            continue
        identity = matrix_runner.repeat_token_identity({mode: runs})[mode]
        if (identity["compared"] != len(LENGTHS) * (WARMUPS + MEASURED) + TRAJECTORIES
                or identity["identical"] != identity["compared"]):
            problems.append(f"matrix/{mode}: raw repeats do not have identical output tokens")
        if any(type(row.get("verified_steps")) not in (int, float) or row["verified_steps"] != 0
               for run in runs for row in (*run["cells"], *run["trajectories"])):
            problems.append(f"matrix/{mode}: raw repeats require zero verified MTP steps")
        fastest = runs[scores.index(min(scores))]
        try:
            matches = (json.dumps(fastest, sort_keys=True, allow_nan=False)
                       == json.dumps(selected, sort_keys=True, allow_nan=False))
        except (ValueError, RecursionError):
            matches = False
        if not matches:
            problems.append(f"matrix/{mode}: selected run differs from the fastest raw repeat")
    return problems


def matrix_repeat_score(run: Any, selected: dict[str, Any], mode: str) -> float | None:
    """The summed median TTFT, after the repeat's output and workload checks."""
    if (not isinstance(run, dict) or run.get("mode") != mode
            or run.get("prompt_id_hashes") != selected.get("prompt_id_hashes")):
        return None
    for label, keys in (("cells", ("length", "run", "warmup")), ("trajectories", ("index",))):
        rows = run.get(label)
        expected = selected[label]
        if not isinstance(rows, list) or len(rows) != len(expected):
            return None
        for row, original in zip(rows, expected, strict=True):
            if (not isinstance(row, dict) or row.get("status") != "finished"
                    or any(type(row.get(key)) is not type(original.get(key))
                           or row.get(key) != original.get(key) for key in keys)):
                return None
            tokens = row.get("output_tokens")
            if (not isinstance(tokens, list) or not 0 < len(tokens) <= OUTPUT_TOKENS
                    or (label == "cells" and len(tokens) != OUTPUT_TOKENS)
                    or not all(type(token) is int and token >= 0 for token in tokens)):
                return None
    medians = []
    for length in LENGTHS:
        timings = [row.get("timing", row) for row in run["cells"]
                   if row.get("length") == length and row.get("warmup") is False]
        if not timings or any(not isinstance(timing, dict)
                              or not positive_measurement(timing.get("ttft_seconds"))
                              for timing in timings):
            return None
        medians.append(statistics.median(timing["ttft_seconds"] for timing in timings))
    score = sum(medians)
    return score if positive_measurement(score) else None


def gate_integrity(results: dict[str, dict[str, Any]], th: dict[str, Any],
                   matrix_repeats: Any, delivery_controls: bytes | None) -> list[dict[str, Any]]:
    problems = delivery_control_problems(results["delivery"], delivery_controls)
    for digest_key, names in (("server_sha256", ("delivery", "qa", "lifecycle", "matrix")),
                              ("inventory_sha256", RESULT_FILES)):
        digests = {name: results[name].get(digest_key) for name in names}
        missing = [name for name, digest in digests.items() if not is_sha256(digest)]
        if missing:
            problems.append(f"{digest_key} missing or invalid in {missing}")
        elif len(set(digests.values())) != 1:
            problems.append(f"{digest_key} differs across results: {digests}")
    for name, result in results.items():
        hardware = result.get("hardware") or {}
        if hardware.get("form_factor") != TARGET_FORM_FACTOR or hardware.get("soc", "").strip() != TARGET_SOC \
                or hardware.get("memory_bytes", 0) < TARGET_MEMORY_GIB * GIB:
            problems.append(f"{name}: not the {TARGET_DESCRIPTION} standard host")
        if result.get("qualification") or result.get("release_ready"):
            problems.append(f"{name}: a harness result claims qualification")
    comparison_os = [(results[name].get("hardware") or {}).get("os")
                     for name in ("qa", "matrix", "reference")]
    if not all(isinstance(version, str) and version.strip() for version in comparison_os):
        problems.append("QA, native matrix and reference must record a macOS version")
    elif len(set(comparison_os)) != 1:
        problems.append("QA, native matrix and reference must use the same macOS version")
    floor = th["timing_host"]["min_cpu_idle_percent"]
    for name in ("qa", "lifecycle", "matrix", "reference"):
        quiet = results[name].get("host_quiet")
        before = quiet.get("cpu_idle_percent") if isinstance(quiet, dict) else None
        samples = (before, results[name].get("host_idle_after"))
        if not all(type(value) in (int, float) and floor <= value <= 100 for value in samples):
            problems.append(f"{name}: before and after must record CPU idle between {floor}% and 100%")
    matrix = results["matrix"]
    repeats = matrix.get("repeat_token_identity") or {}
    if matrix.get("repeats_per_arm") != AX_REPEATS or repeats.keys() != set(AX_ARMS):
        problems.append(f"matrix: expected {AX_REPEATS} independent processes for each direct arm")
    expected_pairs = len(LENGTHS) * (WARMUPS + MEASURED) + TRAJECTORIES
    trajectory_hashes = {f"traj-{index:02d}" for index in range(TRAJECTORIES)}
    for mode in AX_ARMS:
        identity = repeats.get(mode) or {}
        run = matrix["runs"][mode]
        indices = [row.get("index") for row in run["trajectories"]]
        hashes = {key: value for key, value in run["prompt_id_hashes"].items() if key.startswith("traj-")}
        if (indices != list(range(TRAJECTORIES)) or any(type(index) is not int for index in indices)
                or hashes.keys() != trajectory_hashes or not all(map(is_sha256, hashes.values()))
                or len(set(hashes.values())) != TRAJECTORIES):
            problems.append(f"matrix/{mode}: expected the {TRAJECTORIES} distinct trajectory prompts")
        if identity.get("compared") != expected_pairs:
            problems.append(f"matrix/{mode}: expected {expected_pairs} repeat token comparisons")
        elif identity.get("identical") != expected_pairs:
            problems.append(f"matrix/{mode}: repeats of the same arm emitted different tokens")
    problems.extend(matrix_repeat_problems(matrix, matrix_repeats))
    reference = results["reference"]
    pinned = reference.get("reference")
    if not isinstance(pinned, dict):
        pinned = {}
    if pinned.get("runtime") != "mlx-vlm" or pinned.get("commit") != REFERENCE_COMMIT:
        problems.append("reference: expected the pinned MLX-VLM checkout")
    if reference.get("versions") != REFERENCE_VERSIONS or pinned.get("versions") != REFERENCE_VERSIONS:
        problems.append("reference: package versions differ from the frozen runtime")
    if (reference.get("language_sha256") != REFERENCE_LANGUAGE_SHA256
            or pinned.get("language_sha256") != REFERENCE_LANGUAGE_SHA256):
        problems.append("reference: language module hash differs from the audited source")
    changed_settings = [name for name, expected in REFERENCE_EXECUTION_SETTINGS.items()
                        if type(reference.get(name)) is not int or reference[name] != expected]
    if changed_settings:
        problems.append(f"reference: frozen execution settings differ for {changed_settings}")
    if reference.get("completed") is True:
        runtime_files = reference.get("runtime_files_bound")
        if reference.get("phase") != "complete" or type(runtime_files) is not int or runtime_files <= 0:
            problems.append("reference: no completed runtime-file recheck")
    return [gate("integrity", not problems, "; ".join(problems) or
                 "one server binary and pack inventory, the pinned reference runtime, and the target host")]


def evaluate(results: dict[str, dict[str, Any]], th: dict[str, Any],
             contract: dict[str, Any], matrix_repeats: Any, delivery_controls: bytes | None,
             raw_qa_rows: dict[str, list[Any]]) -> dict[str, Any]:
    qa, lifecycle, matrix = results["qa"], results["lifecycle"], results["matrix"]
    reference, delivery = results["reference"], results["delivery"]
    runs = [(f"qa/{mode}", qa["summary"]["modes"][mode].get("ready_seconds")) for mode in AX_ARMS]
    runs += [(f"lifecycle/{run['mode']}", run.get("ready_seconds")) for run in lifecycle.get("runs", [])
             if run["mode"] in AX_ARMS]
    runs += [(f"matrix/{mode}", matrix["runs"][mode].get("ready_seconds")) for mode in AX_ARMS]
    gates = [*gate_delivery(delivery, th), *gate_integrity(results, th, matrix_repeats, delivery_controls),
             *gate_quality(qa, reference, th, raw_qa_rows),
             *gate_lifecycle(lifecycle, th), *gate_memory(qa, th), *gate_readiness(delivery, runs, th),
             *gate_long_lookup(qa, th), *gate_reference_relative(matrix, reference, th),
             *gate_mtp_policy(delivery, qa, lifecycle, matrix, th, contract)]
    return {"gates": gates, "release_ready": all(g["passed"] for g in gates)}


def source_state() -> dict[str, Any]:
    root = SCRIPTS.parent
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                          "--untracked-files=no"], text=True).strip())
    return {"commit": commit, "tracked_tree_dirty": dirty}


def unique_json_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded = dict(pairs)
    if len(decoded) != len(pairs):
        raise ValueError("duplicate member in raw evidence")
    return decoded


def read_qa_rows(qa_dir: Path) -> tuple[dict[str, list[Any]], dict[str, str]]:
    items = qa_cohort.load_items(QA_ITEMS)
    rows = {}
    digests = {}
    for mode in AX_ARMS:
        parsed = []
        digest = hashlib.sha256()
        for index, item in enumerate(items):
            try:
                raw = (qa_dir / mode / f"{index:03d}-{item['id']}.json").read_bytes()
                # Hash fixed-width row digests in the frozen index order.
                digest.update(hashlib.sha256(raw).digest())
                parsed.append(json.loads(raw, object_pairs_hook=unique_json_members))
            except (OSError, ValueError, RecursionError):
                break
        rows[mode] = parsed
        if len(parsed) == len(items):
            digests[f"qa_{mode}_rows"] = digest.hexdigest()
    return rows, digests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write the verdict JSON here")
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    args = parser.parse_args()
    import qualify_qwen38_flash_next as qualify
    th = json.loads(args.thresholds.read_text())
    if th.get("release_scope") != qualify.RELEASE_SCOPE:
        raise SystemExit("thresholds release_scope differs from the pinned Flash Next direct-default scope")
    paths = {name: args.evidence_dir / relative for name, relative in RESULT_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing evidence: {missing}")
    inputs = {name: path.read_bytes() for name, path in paths.items()}
    results = {name: json.loads(data) for name, data in inputs.items()}
    raw_path = args.evidence_dir / MATRIX_REPEATS_FILE
    try:
        inputs["matrix_raw"] = raw_path.read_bytes()
        matrix_repeats = json.loads(inputs["matrix_raw"], object_pairs_hook=unique_json_members)
    except (OSError, ValueError, RecursionError):
        matrix_repeats = None
    try:
        inputs["delivery_controls"] = (args.evidence_dir / DELIVERY_CONTROLS_FILE).read_bytes()
    except OSError:
        pass
    raw_qa_rows, qa_digests = read_qa_rows(paths["qa"].parent)
    verdict = evaluate(results, th, qualify.contract(), matrix_repeats, inputs.get("delivery_controls"), raw_qa_rows)
    verdict.update(
        schema=qualify.RELEASE_VERDICT_SCHEMA, release_scope=th["release_scope"],
        pack_revision=th["target"]["pack_revision"], thresholds_sha256=sha256_file(args.thresholds),
        evidence_sha256={**{name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()}, **qa_digests},
        mtp_certification=qualify.contract()["mtp_certification"], source=source_state(),
        disclosure=th["disclosure"])
    for item in verdict["gates"]:
        print(f"{'PASS' if item['passed'] else 'FAIL'}  {item['gate']}: {item['detail']}")
    print("release_ready:", verdict["release_ready"])
    if args.output:
        args.output.write_text(json.dumps(verdict, indent=2) + "\n")
    return 0 if verdict["release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
