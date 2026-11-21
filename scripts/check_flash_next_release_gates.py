#!/usr/bin/env python3
"""Evaluate the Flash Next direct-default release gates from recorded evidence.

Reads the result files produced by the delivery, QA, lifecycle, generate-matrix
and reference harnesses plus the frozen thresholds in
`flash_next_release_thresholds.json`, and reports one pass/fail line per gate.
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
# The release gate checks the frozen workload independently of the runners.
OUTPUT_TOKENS = 128
LENGTHS = (512, 2048, 8192)
WARMUPS = 2
MEASURED = 3
TRAJECTORIES = 16
AX_REPEATS = 2
LONG_ID = "long_context_record_mass"
RESULT_FILES = {"delivery": "delivery/result.json", "qa": "qa/result.json",
                "lifecycle": "lifecycle/result.json", "matrix": "generate-matrix/result.json",
                "reference": "reference/result.json"}


def gate(name: str, passed: bool, detail: str, **data: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "detail": detail, **data}


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def positive_measurement(value: Any) -> bool:
    return type(value) in (int, float) and 0 < value <= sys.float_info.max


def delivered_modes(delivery: dict[str, Any]) -> list[str]:
    modes = (delivery.get("native_support") or {}).get("modes")
    return modes if isinstance(modes, list) and all(isinstance(mode, str) for mode in modes) else []


def gate_delivery(delivery: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    spec, target = th["delivery"], th["target"]
    verification = delivery.get("verification") or {}
    support = delivery.get("native_support") or {}
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
    if sorted(delivered_modes(delivery)) != sorted(spec["functional_controls_on_delivered_copy"]) \
            or not support.get("passed"):
        problems.append("functional controls did not pass in every required mode")
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
    support = delivery.get("native_support") or {}
    recorded = support.get("readiness_seconds")
    delivered = {mode: seconds if positive_measurement(seconds) else None
                 for mode, seconds in recorded.items()} if isinstance(recorded, dict) else {}
    modes = delivered_modes(delivery)
    first_mode = modes[0] if modes else None
    required = th["delivery"]["functional_controls_on_delivered_copy"]
    problems = []
    if sorted(modes) != sorted(required) or delivered.keys() != set(required):
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


def gate_quality(qa: dict[str, Any], reference: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    modes = qa["summary"]["modes"]
    ref_qa = reference.get("qa") or {}
    problems = []
    rows = ref_qa.get("rows")
    reference_complete = reference.get("completed") is True and isinstance(rows, list) and len(rows) == 105
    recorded_failures = ref_qa.get("failures")
    if not reference_complete:
        problems.append("reference QA is incomplete")
    elif not isinstance(recorded_failures, list) or not all(isinstance(item, str) for item in recorded_failures):
        problems.append("reference QA has no valid failure list")
    ref_failures = set(recorded_failures) if not problems else None
    ax_only: dict[str, list[str] | None] = {}
    for mode in AX_ARMS:
        summary = modes.get(mode, {})
        if summary.get("completed") != 105 or not summary.get("normal_stops") or summary.get("route_errors"):
            problems.append(f"{mode}: incomplete cohort, abnormal stop or route error")
        extra = sorted(set(summary.get("quality_failures", [])) - ref_failures) if ref_failures is not None else None
        ax_only[mode] = extra
        if extra is not None and len(extra) > th["quality"]["ax_only_failures_max"]:
            problems.append(f"{mode}: failures the reference passes: {extra}")
    differing = qa["summary"]["text_identity_vs_disabled"]["default"]["differing_text_ids"]
    if differing:
        problems.append(f"default text differs from direct on {differing}")
    return [gate("quality", not problems, "; ".join(problems) or
                 "105/105 complete in both arms, default text equals direct text, every AX failure "
                 f"also fails on the reference ({sorted(ref_failures or [])})",
                 ax_failures={m: modes[m]["quality_failures"] for m in AX_ARMS},
                 reference_failures=sorted(ref_failures) if ref_failures is not None else None,
                 ax_only_failures=ax_only)]


def gate_lifecycle(lifecycle: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    spec = th["lifecycle"]
    by_mode = {run["mode"]: run for run in lifecycle.get("runs", [])}
    problems = []
    for mode in spec["modes_required"]:
        run = by_mode.get(mode)
        if run is None or len(run["actions"]) != spec["actions_per_mode"]:
            problems.append(f"{mode}: missing or incomplete actions")
            continue
        if any(any(value != 0 for value in action["drained"].values()) for action in run["actions"]):
            problems.append(f"{mode}: counters did not drain")
        if run.get("exit_code_after_stop") != 0:
            problems.append(f"{mode}: server did not exit cleanly")
    if not lifecycle.get("lifecycle_passed"):
        problems.append("lifecycle result is not marked passed")
    return [gate("lifecycle", not problems, "; ".join(problems) or
                 f"{spec['actions_per_mode']} actions in each of {spec['modes_required']} with drained counters")]


def cell_stats(rows: list[dict[str, Any]], tokens: Any) -> dict[str, Any]:
    """Medians over measured runs of one cell; every measured run must be full length."""
    measured = [r for r in rows if r.get("warmup") is False]
    full = [r for r in measured if tokens(r) == OUTPUT_TOKENS]
    rates, ttfts = [], []
    for row in full:
        timing = row.get("timing", row)
        seconds = timing.get("decode_seconds")
        count = timing.get("decode_tokens")
        first = timing.get("ttft_seconds")
        if all(map(positive_measurement, (seconds, count, first))):
            rate = count / seconds
            if positive_measurement(rate):
                rates.append(rate)
                ttfts.append(first)
    complete_timing = len(rates) == len(measured) == MEASURED
    return {"measured": len(measured), "full_length": len(full),
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
    if not reference.get("completed") or not ref_cells:
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
        ref = cell_stats([c for c in ref_cells if c.get("length") == length], lambda r: r["generated_tokens"])
        if ref["full_length"] != ref["measured"]:
            problems.append(f"{length}: reference has short measured outputs")
        for arm in AX_ARMS:
            rows = [c for c in ax_cells[arm] if c.get("length") == length]
            ax = cell_stats(rows, lambda r: len(r["output_tokens"]))
            if ax["full_length"] != ax["measured"]:
                problems.append(f"{length}/{arm}: AX has short measured outputs")
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


def gate_integrity(results: dict[str, dict[str, Any]], th: dict[str, Any]) -> list[dict[str, Any]]:
    problems = []
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
    if results["reference"].get("versions", {}).get("mlx-vlm") != "0.7.0rc0":
        problems.append("reference runtime is not the pinned MLX-VLM")
    return [gate("integrity", not problems, "; ".join(problems) or
                 "one server binary and one pack inventory across all results; hardware is the target SKU")]


def evaluate(results: dict[str, dict[str, Any]], th: dict[str, Any],
             contract: dict[str, Any]) -> dict[str, Any]:
    qa, lifecycle, matrix = results["qa"], results["lifecycle"], results["matrix"]
    reference, delivery = results["reference"], results["delivery"]
    runs = [(f"qa/{mode}", qa["summary"]["modes"][mode].get("ready_seconds")) for mode in AX_ARMS]
    runs += [(f"lifecycle/{run['mode']}", run.get("ready_seconds")) for run in lifecycle.get("runs", [])
             if run["mode"] in AX_ARMS]
    runs += [(f"matrix/{mode}", matrix["runs"][mode].get("ready_seconds")) for mode in AX_ARMS]
    gates = [*gate_delivery(delivery, th), *gate_integrity(results, th), *gate_quality(qa, reference, th),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write the verdict JSON here")
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    args = parser.parse_args()
    import qualify_qwen38_flash_next as qualify
    th = json.loads(args.thresholds.read_text())
    paths = {name: args.evidence_dir / relative for name, relative in RESULT_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing evidence: {missing}")
    results = {name: json.loads(path.read_text()) for name, path in paths.items()}
    verdict = evaluate(results, th, qualify.contract())
    verdict.update(
        schema="ax-engine.flash-next.release-verdict.v1", release_scope=th["release_scope"],
        pack_revision=th["target"]["pack_revision"], thresholds_sha256=sha256_file(args.thresholds),
        evidence_sha256={name: sha256_file(path) for name, path in paths.items()},
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
